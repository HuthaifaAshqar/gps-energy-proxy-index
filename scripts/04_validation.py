import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True); os.chdir(os.path.join(ROOT, "outputs"))
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from epi.core import *
agg=pd.read_pickle('agg_all.pkl'); agg=agg[agg.head_cov>=0.5].copy()
agg['EPI']=epi(agg)
out={}
def pr(x,y,z):  # partial spearman controlling z
    rx,ry,rz=[rankdata(a) for a in (x,y,z)]
    ex=rx-np.polyval(np.polyfit(rz,rx,1),rz); ey=ry-np.polyval(np.polyfit(rz,ry,1),rz)
    return np.corrcoef(ex,ey)[0,1]
tg=['FI','DFS','TEI']
rows=[]
for c in COMPONENTS+['EPI','v_mean']:
    rows.append([c]+[spearmanr(agg[c],agg[t])[0] for t in tg]+[pr(agg[c],agg[t],agg.v_mean) for t in tg])
cor=pd.DataFrame(rows,columns=['var']+[f'rho_{t}' for t in tg]+[f'prho_{t}' for t in tg]); print(cor.round(3)); out['cor']=cor
# component inter-correlation
print(agg[COMPONENTS+['v_mean']].corr('spearman').round(3)); out['compcor']=agg[COMPONENTS+['v_mean']].corr('spearman')
# speed bins
agg['vbin']=pd.cut(agg.v_mean*3.6,[0,40,70,100,200],labels=['<40','40-70','70-100','>100'])
sb=agg.groupby('vbin').apply(lambda g: pd.Series(dict(n=len(g),EPI=g.EPI.median(),FI=g.FI.median(),DFS=g.DFS.median(),
    rho_FI=spearmanr(g.EPI,g.FI)[0],rho_DFS=spearmanr(g.EPI,g.DFS)[0],rho_TEI=spearmanr(g.EPI,g.TEI)[0])))
print(sb.round(3)); out['speedbins']=sb
# ablation / variants
def ev(name,e):
    return [name,spearmanr(e,agg.FI)[0],spearmanr(e,agg.DFS)[0],spearmanr(e,agg.TEI)[0],pr(e,agg.FI,agg.v_mean),pr(e,agg.DFS,agg.v_mean)]
V=[]
V.append(ev('RPA only',epi(agg,(1,),comps=['RPA'])))
V.append(ev('sigma_a only',epi(agg,(1,),comps=['sigma_a'])))
V.append(ev('omega only',epi(agg,(1,),comps=['omega_rms'])))
V.append(ev('RPA + sigma_a',epi(agg,(.5,.5),comps=['RPA','sigma_a'])))
V.append(ev('EPI (equal, ECDF)',agg.EPI))
V.append(ev('EPI (equal, min-max)',epi(agg,method='minmax')))
V.append(ev('EPI (equal, z-probit)',epi(agg,method='z')))
Z=(agg[COMPONENTS]-agg[COMPONENTS].mean())/agg[COMPONENTS].std()
pca=PCA(3).fit(Z); wp=np.abs(pca.components_[0]); wp=wp/wp.sum(); out['pca']=(pca.explained_variance_ratio_,wp)
V.append(ev('EPI (PCA weights)',epi(agg,tuple(wp))))
P=agg[COMPONENTS].rank(pct=True); P=P/P.sum(); Ent=-(P*np.log(P)).sum()/np.log(len(P)); we=(1-Ent)/(1-Ent).sum(); out['entropy_w']=we.values
V.append(ev('EPI (entropy weights)',epi(agg,tuple(we.values))))
V.append(ev('Speed CV',agg.sigma_a*0+ (agg.sigma_a/agg.v_mean)))
V.append(ev('Inverse mean speed',1/agg.v_mean))
var=pd.DataFrame(V,columns=['index','rho_FI','rho_DFS','rho_TEI','prho_FI','prho_DFS']);print(var.round(3)); out['variants']=var
print('pca',out['pca'],'entropy',we.round(3).to_dict())
# Kalman vs raw differencing
aggr=pd.read_pickle('agg_raw.pkl').loc[agg.index]
kr=dict(sigma_a_raw=aggr.sigma_a.median(),sigma_a_kf=agg.sigma_a.median(),RPA_raw=aggr.RPA.median(),RPA_kf=agg.RPA.median(),
  rho_sig=spearmanr(aggr.sigma_a,agg.sigma_a)[0],rho_rpa=spearmanr(aggr.RPA,agg.RPA)[0],
  epi_raw_vs_kf=spearmanr(epi(aggr.assign(omega_rms=agg.omega_rms)),agg.EPI)[0],
  rhoDFS_raw=spearmanr(epi(aggr.assign(omega_rms=agg.omega_rms)),agg.DFS)[0])
print(kr); out['kf_vs_raw']=kr
agg.to_pickle('agg.pkl'); pickle.dump(out,open('res_main.pkl','wb'))
print(len(agg), agg.device.nunique())
