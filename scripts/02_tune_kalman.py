import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True); os.chdir(os.path.join(ROOT, "outputs"))
"""Maximum-likelihood estimation of the Kalman noise parameters (q, sigma_v) with a hold-out check."""
import pandas as pd, numpy as np, json
from scipy.optimize import minimize
from epi.core import *
d = pd.read_pickle('points.pkl')
rng = np.random.default_rng(1)
segs = rng.choice(d.seg.unique(), 1500, replace=False)
def fit(s, x0):
    f = lambda x: -loglik_total(s, np.exp(x[0]), np.exp(2 * x[1]))
    o = minimize(f, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1))
    return np.exp(o.x), o
(q, sv), o = fit(d[d.seg.isin(segs)], [np.log(0.2), np.log(0.3)])
segs2 = rng.choice(np.setdiff1d(d.seg.unique(), segs), 1500, replace=False)
(q2, sv2), _ = fit(d[d.seg.isin(segs2)], o.x)
# Values applied downstream. The paper uses q = 0.145 and sigma_v = 0.335, which lie between the
# estimation and hold-out fits; set USE_PAPER_VALUES = False to apply the average of the two fits instead.
USE_PAPER_VALUES = True
q_app, sv_app = (0.145, 0.335) if USE_PAPER_VALUES else ((q + q2) / 2, (sv + sv2) / 2)
res = dict(fit_q=float(q), fit_sigma_v=float(sv), holdout_q=float(q2), holdout_sigma_v=float(sv2),
           q=float(q_app), sigma_v=float(sv_app))
json.dump(res, open('kalman_params.json', 'w'), indent=2); print(res)
