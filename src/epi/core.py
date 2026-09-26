"""Core pipeline for the GPS-driven Energy Proxy Index (EPI) study."""
import glob
import numpy as np
import pandas as pd
from numba import njit

import os
RAW = os.environ.get("EPI_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
KUWAIT_BBOX = (28.5, 30.2, 46.5, 48.6)  # lat_min, lat_max, lon_min, lon_max
MAX_GAP_S = 10.0       # segment break when consecutive fixes are more than 10 s apart
MIN_DUR_S = 60.0       # minimum segment duration
MIN_PTS = 10           # minimum fixes per segment
MIN_DIST_M = 500.0     # minimum travelled distance
MIN_SATS = 3
HEADING_MIN_SPEED = 10 / 3.6  # heading treated as valid above 10 km/h


def haversine(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_raw():
    frames = []
    for f in sorted(glob.glob(f"{RAW}/GPS*_0.csv")):
        d = pd.read_csv(f)
        d["file"] = f.split("/")[-1].replace(".csv", "")
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["t"] = pd.to_datetime(d["GPSTime"], errors="coerce")
    ll = d["loc"].str.split(",", expand=True).astype(float)
    d["lat"], d["lon"] = ll[0], ll[1]
    d["os"] = np.where(d["CallerType"].str.startswith("Android"), "Android",
                       np.where(d["CallerType"].str.startswith("iP"), "iOS", "Other"))
    return d


def clean(d, log=None):
    """Apply record-level quality filters and return cleaned frame plus a filter log."""
    log = log if log is not None else []
    n0 = len(d); log.append(("Raw records", n0, d.CallerID.nunique()))
    la0, la1, lo0, lo1 = KUWAIT_BBOX
    d = d[d.lat.between(la0, la1) & d.lon.between(lo0, lo1)]
    log.append(("Inside Kuwait bounding box", len(d), d.CallerID.nunique()))
    d = d[d.t.notna() & (d.Speed >= 0)]
    log.append(("Valid timestamp and non-negative speed", len(d), d.CallerID.nunique()))
    d = d[d.NumOfSatellite >= MIN_SATS]
    log.append((f"At least {MIN_SATS} satellites", len(d), d.CallerID.nunique()))
    d = d.drop_duplicates(["CallerID", "t"]).sort_values(["CallerID", "t"]).reset_index(drop=True)
    log.append(("Duplicate fixes removed", len(d), d.CallerID.nunique()))
    # positional consistency: drop fixes whose implied speed is physically implausible
    same = d.CallerID.eq(d.CallerID.shift())
    dt = d.t.diff().dt.total_seconds()
    dist = haversine(d.lat.shift(), d.lon.shift(), d.lat, d.lon)
    vpos = dist / dt
    bad = same & (dt > 0) & (vpos > 70.0)  # > 252 km/h between fixes
    d = d[~bad].reset_index(drop=True)
    log.append(("Positional jumps (>252 km/h implied) removed", len(d), d.CallerID.nunique()))
    d["v"] = d.Speed / 3.6  # km/h -> m/s
    d["hdg"] = np.where(d.Heading >= 0, d.Heading % 360.0, np.nan)
    return d, log


def segment(d, log=None, max_gap=MAX_GAP_S):
    log = log if log is not None else []
    same = d.CallerID.eq(d.CallerID.shift())
    dt = d.t.diff().dt.total_seconds()
    brk = (~same) | (dt > max_gap) | (dt <= 0)
    d = d.assign(seg=brk.cumsum())
    g = d.groupby("seg")
    t0 = g.t.transform("min"); t1 = g.t.transform("max")
    d["dur"] = (t1 - t0).dt.total_seconds()
    d["npts"] = g.t.transform("size")
    d = d[(d.dur >= MIN_DUR_S) & (d.npts >= MIN_PTS)]
    d["tsec"] = (d.t - d.groupby("seg").t.transform("min")).dt.total_seconds()
    # distance from integrated Doppler speed (trapezoid)
    dd = d.groupby("seg").apply(
        lambda s: np.trapezoid(s.v.values, s.tsec.values), include_groups=False)
    keep = dd[(dd >= MIN_DIST_M)].index
    d = d[d.seg.isin(keep)].reset_index(drop=True)
    log.append((f"Segments >= {MIN_DUR_S:.0f} s, >= {MIN_PTS} fixes, >= {MIN_DIST_M:.0f} m",
                len(d), d.CallerID.nunique(), d.seg.nunique()))
    return d, log


# ---------------------------------------------------------------- Kalman / RTS
@njit(cache=True)
def kalman_rts(t, z, q, r):
    """Constant-acceleration Kalman filter with Rauch-Tung-Striebel smoothing for
    irregularly sampled speed. State [v, a]; white-jerk process noise with spectral
    density q (m^2/s^5); measurement variance r (m^2/s^2).
    Returns smoothed v, a and the innovation log-likelihood."""
    n = t.shape[0]
    xf = np.zeros((n, 2)); Pf = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))
    x = np.array([z[0], 0.0]); P = np.array([[r, 0.0], [0.0, 1.0]])
    ll = 0.0
    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
            x = F @ x
            P = F @ P @ F.T + Q
        xp[k] = x; Pp[k] = P
        S = P[0, 0] + r
        y = z[k] - x[0]
        K0 = P[0, 0] / S; K1 = P[1, 0] / S
        if k > 0:
            ll += -0.5 * (np.log(2 * np.pi * S) + y * y / S)
        x = np.array([x[0] + K0 * y, x[1] + K1 * y])
        P = np.array([[(1 - K0) * P[0, 0], (1 - K0) * P[0, 1]],
                      [P[1, 0] - K1 * P[0, 0], P[1, 1] - K1 * P[0, 1]]])
        xf[k] = x; Pf[k] = P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        dt = t[k + 1] - t[k]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, 0], xs[:, 1], ll


def smooth_all(d, q, r, vcol="v"):
    """Apply the smoother segment by segment. Returns arrays aligned to d."""
    vs = np.empty(len(d)); acc = np.empty(len(d))
    seg = d.seg.values; t = d.tsec.values; z = d[vcol].values
    idx = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    for i in range(len(idx) - 1):
        a, b = idx[i], idx[i + 1]
        v_, a_, _ = kalman_rts(t[a:b].astype(np.float64), z[a:b].astype(np.float64), q, r)
        vs[a:b] = np.clip(v_, 0, None); acc[a:b] = a_
    return vs, acc


def loglik_total(d, q, r):
    seg = d.seg.values; t = d.tsec.values; z = d.v.values
    idx = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    tot = 0.0
    for i in range(len(idx) - 1):
        a, b = idx[i], idx[i + 1]
        tot += kalman_rts(t[a:b].astype(np.float64), z[a:b].astype(np.float64), q, r)[2]
    return tot


def raw_accel(d):
    """Central finite-difference acceleration from unfiltered Doppler speed."""
    g = d.groupby("seg")
    dv = g.v.shift(-1) - g.v.shift(1)
    dt = g.tsec.shift(-1) - g.tsec.shift(1)
    a = dv / dt
    fwd = (g.v.shift(-1) - d.v) / (g.tsec.shift(-1) - d.tsec)
    bwd = (d.v - g.v.shift(1)) / (d.tsec - g.tsec.shift(1))
    return a.fillna(fwd).fillna(bwd).values


# ---------------------------------------------------------------- energy models
# VT-CPFM-1 parameters for a 2014 Cadillac SRX (mid-size SUV), as reported by
# Shafik, Eteifa and Rakha (2022, Table II); model form from Rakha et al. (2011).
SRX = dict(a0=7.89e-4, a1=-5.77e-19, a2=2.27e-6, cr0=1.75, cr1=0.0328, cr2=4.55,
           cd=0.39, ch=0.95, af=3.33, eta=0.92, m=2388.0, rho=1.2256)


def vtcpfm(v_ms, a, p=SRX, grade=0.0):
    """Instantaneous fuel rate (L/s) and wheel power (kW)."""
    vk = v_ms * 3.6
    R = (p["rho"] / 25.92) * p["cd"] * p["ch"] * p["af"] * vk ** 2 \
        + 9.8066 * p["m"] * p["cr0"] / 1000.0 * (p["cr1"] * vk + p["cr2"]) \
        + 9.8066 * p["m"] * grade
    P = (R + 1.04 * p["m"] * a) / (3600.0 * p["eta"]) * vk
    Pp = np.clip(P, 0, None)
    fc = np.where(P >= 0, p["a0"] + p["a1"] * Pp + p["a2"] * Pp ** 2, p["a0"])
    return fc, P


def vsp(v, a, grade=0.0):
    """Vehicle specific power (kW/t), Jimenez-Palacios light-duty form."""
    return v * (1.1 * a + 9.81 * grade + 0.132) + 0.000302 * v ** 3


# ---------------------------------------------------------------- segment features
def seg_features(d, vcol="vs", acol="acc", p=SRX):
    """Aggregate point-level kinematics to segment-level indicators."""
    d = d.copy()
    g = d.groupby("seg")
    # time weights: half-interval rule for irregular sampling
    dtp = (g.tsec.shift(-1) - d.tsec).fillna(0)
    dtm = (d.tsec - g.tsec.shift(1)).fillna(0)
    d["w"] = 0.5 * (dtp + dtm)
    v = d[vcol].values; a = d[acol].values; w = d.w.values
    d["dist"] = v * w
    d["pos_pow"] = v * np.clip(a, 0, None) * w          # for RPA
    fc, P = vtcpfm(v, a, p)
    d["fuel"] = fc * w
    vs_ = vsp(v, a)
    d["vsp_pos"] = np.clip(vs_, 0, None) * w           # kJ/t
    d["idle"] = (v < 1.0).astype(float) * w
    d["a2w"] = a ** 2 * w
    d["aw"] = a * w
    # heading rate (deg/s) from raw heading, valid only when moving
    h = d.hdg.values
    hn = g.hdg.shift(-1).values
    tn = g.tsec.shift(-1).values
    vn = g[vcol].shift(-1).values
    dth = ((hn - h + 180.0) % 360.0) - 180.0
    dts = tn - d.tsec.values
    ok = (~np.isnan(dth)) & (dts > 0) & (v > HEADING_MIN_SPEED) & (vn > HEADING_MIN_SPEED)
    om = np.where(ok, dth / np.where(dts > 0, dts, np.nan), np.nan)
    d["om2w"] = np.where(ok, om ** 2 * dts, 0.0)
    d["omw"] = np.where(ok, dts, 0.0)
    agg = d.groupby("seg").agg(
        device=("CallerID", "first"), os=("os", "first"), file=("file", "first"),
        t_start=("t", "min"), n=("t", "size"), dur=("w", "sum"), dist=("dist", "sum"),
        pos_pow=("pos_pow", "sum"), fuel=("fuel", "sum"), vsp_pos=("vsp_pos", "sum"),
        idle=("idle", "sum"), a2w=("a2w", "sum"), aw=("aw", "sum"),
        om2w=("om2w", "sum"), omw=("omw", "sum"),
        lat=("lat", "median"), lon=("lon", "median"),
        vmax=(vcol, "max"), med_dt=("w", "median"))
    agg["v_mean"] = agg.dist / agg.dur                         # m/s
    agg["RPA"] = agg.pos_pow / agg.dist                        # m/s^2
    am = agg.aw / agg.dur
    agg["sigma_a"] = np.sqrt(np.clip(agg.a2w / agg.dur - am ** 2, 0, None))  # m/s^2
    agg["omega_rms"] = np.sqrt(agg.om2w / agg.omw.replace(0, np.nan))       # deg/s
    agg["head_cov"] = agg.omw / agg.dur                        # share of time with valid heading
    agg["idle_frac"] = agg.idle / agg.dur
    agg["FI"] = agg.fuel / (agg.dist / 1000.0) * 100.0         # L/100 km
    agg["TEI"] = agg.vsp_pos / (agg.dist / 1000.0)             # kJ/(t km)
    # cruise reference: fuel at constant mean speed for the same distance
    fcc, _ = vtcpfm(agg.v_mean.values, np.zeros(len(agg)), p)
    agg["FI_cruise"] = fcc * agg.dur / (agg.dist / 1000.0) * 100.0
    agg["DFS"] = 1.0 - agg.FI_cruise / agg.FI                  # dynamic fuel share
    return agg.drop(columns=["pos_pow", "fuel", "vsp_pos", "idle", "a2w", "aw", "om2w", "omw"])


COMPONENTS = ["RPA", "sigma_a", "omega_rms"]


def ecdf_score(x, ref):
    ref = np.sort(ref[~np.isnan(ref)])
    return np.searchsorted(ref, x, side="right") / len(ref)


def epi(agg, weights=(1 / 3, 1 / 3, 1 / 3), ref=None, comps=COMPONENTS, method="ecdf"):
    """EPI in [0, 100]; higher = higher inferred energy demand (less eco-efficient)."""
    ref = agg if ref is None else ref
    s = np.zeros(len(agg))
    for w, c in zip(weights, comps):
        x = agg[c].values
        if method == "ecdf":
            u = ecdf_score(x, ref[c].values)
        elif method == "minmax":
            lo, hi = np.nanpercentile(ref[c], [1, 99])
            u = np.clip((x - lo) / (hi - lo), 0, 1)
        elif method == "z":
            from scipy.stats import norm
            mu, sd = np.nanmean(ref[c]), np.nanstd(ref[c])
            u = norm.cdf((x - mu) / sd)
        s += w * np.nan_to_num(u, nan=0.5)
    return 100.0 * s / sum(weights)
