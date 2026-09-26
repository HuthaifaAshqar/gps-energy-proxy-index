import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True); os.chdir(os.path.join(ROOT, "outputs"))
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings("ignore")
from epi.core import *
d=load_raw()
raw_stats=dict(n=len(d),dev=d.CallerID.nunique(),os=d.os.value_counts().to_dict(),
  sats=d.NumOfSatellite.value_counts().sort_index().to_dict(),neg=(d.Speed<0).mean(),hneg=(d.Heading<0).mean(),
  files=d.groupby('file').agg(t0=('t','min'),t1=('t','max'),n=('t','size'),dev=('CallerID','nunique')))
d,log=clean(d)
same=d.CallerID.eq(d.CallerID.shift()); raw_stats['dt']=d.t.diff().dt.total_seconds()[same]
d,log=segment(d,log)
for l in log: print(l)
d.to_pickle('points.pkl'); pickle.dump((raw_stats,log),open('meta.pkl','wb'))
print(d.seg.nunique(), d.groupby('seg').dur.first().describe())
