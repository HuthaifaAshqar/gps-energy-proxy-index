import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True); os.chdir(os.path.join(ROOT, "outputs"))
"""Monte Carlo robustness: speed/heading noise injection and sampling-rate thinning.
Usage:
  python scripts/05_montecarlo.py all                          # full design used in the paper
  python scripts/05_montecarlo.py noise <sigma_v_mps> <sigma_heading_deg> <n_reps>
  python scripts/05_montecarlo.py thin  <keep_probability> 0 <n_reps>
"""
import glob, json, warnings
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from epi.core import *
warnings.filterwarnings('ignore')
P = json.load(open('kalman_params.json')); Q, SV = P['q'], P['sigma_v']
d = pd.read_pickle('points_s.pkl'); base = pd.read_pickle('agg.pkl'); d = d[d.seg.isin(base.index)]
E2 = ['RPA', 'sigma_a']; base['EPI2'] = epi(base, (.5, .5), comps=E2)
terc = lambda x: pd.qcut(x, 3, labels=False)

def run(dd):
    dd = dd.copy(); dd['vs'], dd['acc'] = smooth_all(dd, Q, SV ** 2)
    a = seg_features(dd); a = a.loc[a.index.intersection(base.index)]
    a['EPI'] = epi(a); a['EPI2'] = epi(a, (.5, .5), comps=E2); b = base.loc[a.index]
    return dict(r3=spearmanr(a.EPI, b.EPI)[0], k3=cohen_kappa_score(terc(a.EPI), terc(b.EPI)),
                r2=spearmanr(a.EPI2, b.EPI2)[0], k2=cohen_kappa_score(terc(a.EPI2), terc(b.EPI2)),
                d3=spearmanr(a.EPI, b.DFS)[0], d2=spearmanr(a.EPI2, b.DFS)[0], n=len(a))

def condition(mode, a1, a2, n):
    rng = np.random.default_rng(abs(hash((mode, a1, a2))) % 2 ** 32); R = []
    for _ in range(n):
        if mode == 'noise':
            dd = d.copy(); dd['v'] = np.clip(dd.v + rng.normal(0, a1, len(dd)), 0, None)
            dd['hdg'] = (dd.hdg + rng.normal(0, a2, len(dd))) % 360
            R.append(dict(exp='noise', sv=a1, sh=a2, **run(dd)))
        else:
            first = d.seg.ne(d.seg.shift()); last = d.seg.ne(d.seg.shift(-1))
            dd = d[(rng.random(len(d)) < a1) | first | last].copy()
            R.append(dict(exp='thin', keep=a1, med_dt=dd.groupby('seg').tsec.diff().median(), **run(dd)))
    os.makedirs('mc', exist_ok=True); pd.DataFrame(R).to_pickle(f'mc/{mode}_{a1}_{a2}.pkl'); print(mode, a1, a2, 'done')

if sys.argv[1] == 'all':
    for sv in [0.25, 0.5, 1.0]:
        for sh in [2, 5, 10]:
            condition('noise', sv, sh, 20 if (sv == 0.5 or sh == 5) else 8)
    for k in [0.8, 0.67, 0.5, 0.33]:
        condition('thin', k, 0, 20)
else:
    condition(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
pd.concat([pd.read_pickle(f) for f in glob.glob('mc/*.pkl')]).to_pickle('mc_all.pkl')
