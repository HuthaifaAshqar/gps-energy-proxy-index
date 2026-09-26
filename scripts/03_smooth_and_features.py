import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True); os.chdir(os.path.join(ROOT, "outputs"))
import pandas as pd, numpy as np, time
from epi.core import *
import json; _p=json.load(open('kalman_params.json')); Q,SV=_p['q'],_p['sigma_v']
d=pd.read_pickle('points.pkl'); t0=time.time()
d['vs'],d['acc']=smooth_all(d,Q,SV**2); print('smooth',time.time()-t0)
d['acc_raw']=raw_accel(d)
d.to_pickle('points_s.pkl')
agg=seg_features(d)
aggr=seg_features(d.assign(vs=d.v),vcol='vs',acol='acc_raw')
print(len(agg)); print(agg[['v_mean','RPA','sigma_a','omega_rms','head_cov','idle_frac','FI','TEI','DFS','med_dt']].describe().T.round(3))
print('raw sigma_a', aggr.sigma_a.describe().round(3).to_dict())
print((agg.head_cov<0.5).mean())
agg.to_pickle('agg_all.pkl'); aggr.to_pickle('agg_raw.pkl')
print('acc pct', np.percentile(d.acc,[0.1,1,50,99,99.9]), np.percentile(d.acc_raw.dropna(),[0.1,1,50,99,99.9]))
