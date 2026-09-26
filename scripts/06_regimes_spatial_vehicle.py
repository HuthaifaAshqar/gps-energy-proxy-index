import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True); os.chdir(os.path.join(ROOT, "outputs"))
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, rankdata, mannwhitneyu, kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from epi.core import *
agg=pd.read_pickle('agg.pkl'); E2=['RPA','sigma_a']; agg['EPI2']=epi(agg,(.5,.5),comps=E2)
import os
out={}; rng=np.random.default_rng(7)
if os.path.exists('res_part1.pkl'):
    out=pickle.load(open('res_part1.pkl','rb')); agg=pd.read_pickle('agg_p1.pkl')
else:
  if True:
    # 1 weights Dirichlet
    W=rng.dirichlet([1,1,1],1000); r=[]
    for w in W:
        e=epi(agg,tuple(w)); r.append((spearmanr(e,agg.EPI)[0],spearmanr(e,agg.DFS)[0]))
    r=np.array(r); out['w_mc']=r; print('weights',np.percentile(r,[2.5,50,97.5],axis=0))
    # 2 vehicle parameters
    d=pd.read_pickle('points_s.pkl'); d=d[d.seg.isin(agg.index)]
    VR=[]
    for i in range(100):
        p=dict(SRX); p.update(m=rng.uniform(1200,3000),cd=rng.uniform(0.28,0.45),af=rng.uniform(2.0,3.4),
            a0=rng.uniform(5e-4,1e-3),a1=0.0,a2=rng.uniform(1e-6,4e-6))
        a=seg_features(d,p=p).loc[agg.index]
        VR.append(dict(**{k:p[k] for k in ['m','cd','af','a0','a2']},FI_med=a.FI.median(),
            rDFS_base=spearmanr(a.DFS,agg.DFS)[0],rFI_base=spearmanr(a.FI,agg.FI)[0],
            r3=spearmanr(agg.EPI,a.DFS)[0],r2=spearmanr(agg.EPI2,a.DFS)[0],r3FI=spearmanr(agg.EPI,a.FI)[0]))
    VR=pd.DataFrame(VR); out['veh_mc']=VR; print(VR.describe().round(3))
    # 3 device-level ICC (speed-adjusted)
    def resid(y,x):
        ry,rx=rankdata(y),rankdata(x); c=np.polyfit(rx,ry,3); return ry-np.polyval(c,rx)
    agg['EPI_adj']=resid(agg.EPI,agg.v_mean); agg['EPI2_adj']=resid(agg.EPI2,agg.v_mean)
    cnt=agg.device.value_counts(); dv=agg[agg.device.isin(cnt[cnt>=3].index)]
    def icc(df,col):
        g=df.groupby('device')[col]; k=g.size(); n=len(df); a=len(k)
        grand=df[col].mean(); ssb=(k*(g.mean()-grand)**2).sum(); ssw=((df[col]-g.transform('mean'))**2).sum()
        msb=ssb/(a-1); msw=ssw/(n-a); k0=(n-(k**2).sum()/n)/(a-1)
        return (msb-msw)/(msb+(k0-1)*msw)
    ic={c:icc(dv,c) for c in ['EPI','EPI_adj','EPI2','EPI2_adj','v_mean']}
    null=[icc(dv.assign(EPI_adj=rng.permutation(dv.EPI_adj.values)),'EPI_adj') for _ in range(200)]
    ic['null95']=np.percentile(null,97.5); ic['n_dev']=dv.device.nunique(); ic['n_seg']=len(dv)
    out['icc']=ic; print(ic)

    pickle.dump(out,open('res_part1.pkl','wb')); agg.to_pickle('agg_p1.pkl')
# 4 clustering
F=['v_mean','RPA','sigma_a','omega_rms']; X=agg[F].copy(); X['omega_rms']=np.log1p(X.omega_rms); X['RPA']=np.log(X.RPA.clip(lower=1e-3)); X['sigma_a']=np.log(X.sigma_a.clip(lower=1e-3))
Xs=((X-X.mean())/X.std()).values; sil={}
for k in range(2,8):
    km=KMeans(k,n_init=10,random_state=0).fit(Xs); sil[k]=silhouette_score(Xs,km.labels_,sample_size=4000,random_state=0)
print('sil',sil); out['sil']=sil
K=4; km=KMeans(K,n_init=20,random_state=0).fit(Xs); agg['cl']=km.labels_
prof=agg.groupby('cl').agg(n=('EPI','size'),v=('v_mean','median'),RPA=('RPA','median'),sig=('sigma_a','median'),om=('omega_rms','median'),
   EPI=('EPI','median'),EPI2=('EPI2','median'),FI=('FI','median'),DFS=('DFS','median'),TEI=('TEI','median'))
prof['v']=prof.v*3.6; print(prof.round(3)); out['prof']=prof
# 5 temporal
agg['date']=agg.t_start.dt.date; agg['dow']=agg.t_start.dt.day_name(); agg['hour']=agg.t_start.dt.hour
agg['weekend']=agg.dow.isin(['Friday','Saturday'])
tm=agg.groupby('file').agg(date=('t_start','min'),n=('EPI','size'),dev=('device','nunique'),v=('v_mean','median'),EPI=('EPI','median'),EPI_adj=('EPI_adj','median'),DFS=('DFS','median'))
tm['v']*=3.6; print(tm.round(2)); out['tm']=tm
mw=mannwhitneyu(agg.EPI_adj[agg.weekend],agg.EPI_adj[~agg.weekend]); 
out['weekend']=dict(n_we=agg.weekend.sum(),n_wd=(~agg.weekend).sum(),med_we=agg.EPI[agg.weekend].median(),med_wd=agg.EPI[~agg.weekend].median(),
   adj_we=agg.EPI_adj[agg.weekend].median(),adj_wd=agg.EPI_adj[~agg.weekend].median(),p=mw.pvalue,
   r_rb=1-2*mw.statistic/(agg.weekend.sum()*(~agg.weekend).sum()),v_we=agg.v_mean[agg.weekend].median()*3.6,v_wd=agg.v_mean[~agg.weekend].median()*3.6)
print(out['weekend']); print(agg.hour.value_counts().sort_index())
# 6 spatial grid + Moran's I
agg['gx']=(agg.lon/0.01).round().astype(int); agg['gy']=(agg.lat/0.01).round().astype(int)
cell=agg.groupby(['gx','gy']).agg(n=('EPI','size'),EPI=('EPI','median'),EPI_adj=('EPI_adj','median'),v=('v_mean','median'),lat=('lat','median'),lon=('lon','median')).reset_index()
cell=cell[cell.n>=5]; out['cell']=cell
def moran(z,xy,k=6,perm=999):
    nn=NearestNeighbors(n_neighbors=k+1).fit(xy); idx=nn.kneighbors(xy,return_distance=False)[:,1:]
    z=z-z.mean(); I=lambda zz:(len(zz)/(k*len(zz)))*(zz*zz[idx].sum(1)).sum()/(zz**2).sum()
    I0=I(z); nulls=np.array([I(rng.permutation(z)) for _ in range(perm)]); return I0,(np.sum(nulls>=I0)+1)/(perm+1)
xy=np.c_[cell.lon*np.cos(np.radians(29.3)),cell.lat]
out['moran_EPI']=moran(cell.EPI.values,xy); out['moran_adj']=moran(cell.EPI_adj.values,xy)
print('cells',len(cell),out['moran_EPI'],out['moran_adj'])
# OS
out['os']=agg.groupby('os').agg(n=('EPI','size'),EPI=('EPI','median'),EPI_adj=('EPI_adj','median'),dt=('med_dt','median'))
print(out['os'])
agg.to_pickle('agg2.pkl'); pickle.dump(out,open('res_more.pkl','wb'))
