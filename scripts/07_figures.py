import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True); os.chdir(os.path.join(ROOT, "outputs"))
FIGDIR=os.path.join(ROOT,'figures','')
os.makedirs(FIGDIR,exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
 'axes.titlesize':9.5,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})
C=dict(blue='#1f4e79',teal='#2a9d8f',orange='#e76f51',gold='#e9c46a',grey='#6c757d',red='#b5332e',light='#a8dadc')

from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams['font.family']='DejaVu Sans'
fig,ax=plt.subplots(figsize=(12,5.6)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,62)
INK='#222222'; EDGE='#3a3a3a'
col={1:('#eef3f8','#1f4e79'),2:('#eaf5f3','#1e7a6e'),3:('#f5f5f5','#555555'),'s':('#fcefe9','#c2562f')}
def box(x,y,w,h,title,body,st):
    fc,ec=col[st]
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.8',fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h-1.7,title,ha='center',va='top',fontsize=8.4,weight='bold',color=INK)
    ax.text(x+w/2,y+h-4.6,body,ha='center',va='top',fontsize=7.1,color=INK,linespacing=1.4)
def arrow(pts,c=EDGE,ls='-'):
    for (x1,y1),(x2,y2) in zip(pts[:-2],pts[1:-1]): ax.plot([x1,x2],[y1,y2],c=c,lw=1.1,ls=ls,solid_capstyle='butt')
    ax.annotate('',xy=pts[-1],xytext=pts[-2],arrowprops=dict(arrowstyle='-|>,head_length=0.5,head_width=0.25',color=c,lw=1.1,ls=ls,shrinkA=0,shrinkB=0))
# stage bands
bands=[(42,19.5,'Stage 1\nData preparation',1),(22,19.5,'Stage 2\nIndex construction',2),(1,19.5,'Stage 3\nEvaluation',3)]
for y,h,t,s in bands:
    ax.add_patch(Rectangle((0,y),100,h,fc=col[s][0],ec='none',alpha=.35,zorder=-2))
    ax.text(3.6,y+h/2,t,ha='center',va='center',rotation=90,fontsize=8.2,weight='bold',color=col[s][1])
# Stage 1
X0=9; W=15.6; G=2.75; y1=45.5; H1=13.5
titles=[('Crowdsourced GPS','1.0 M fixes\n10,592 devices\nspeed, heading, time'),
        ('Quality control','spatial extent, ≥ 3 satellites\nvalid speed, duplicates\npositional jumps'),
        ('Segmentation','split at gaps > 10 s\n≥ 60 s, ≥ 10 fixes, ≥ 500 m\n9,903 segments'),
        ('Kalman–RTS','constant acceleration\nvariable time step\nq, $\\sigma_v$ by max. likelihood'),
        ('Kinematic signals','speed $v(t)$\nacceleration $a(t)$\nheading rate $\\omega(t)$')]
xs=[X0+i*(W+G) for i in range(5)]
for x,(t,b) in zip(xs,titles): box(x,y1,W,H1,t,b,1)
for x in xs[:-1]: arrow([(x+W,y1+H1/2),(x+W+G,y1+H1/2)])
# Stage 2
y2=25.5; H2=13.5
sx=X0; sw=3*W+2*G
box(sx,y2,sw,H2,'Physics-based surrogates (validation only)','VT-CPFM-1 fuel intensity FI (L/100 km)\ndynamic fuel share DFS = 1 − FI$_{cruise}$ / FI\npositive tractive energy intensity (VSP)','s')
cx=xs[2]+0.0; cw=2*W+G-0.0
box(xs[3]-(W+G)*0+0,y2,W,H2,'Kinematic components','RPA: energy demand\n$\\sigma_a$: accel. volatility\n$\\omega_{rms}$: heading volatility',2)
box(xs[4],y2,W,H2,'Energy Proxy Index','ECDF normalisation\n$EPI = 100\\,\\Sigma_j\\, w_j F_j(x_j)$\nEPI-3, EPI-L',2)
# stage1->stage2 arrows
kx=xs[4]+W/2
arrow([(kx,y1),(kx,y1-2.5),(xs[3]+W/2,y1-2.5),(xs[3]+W/2,y2+H2)])
arrow([(xs[3]+W,y2+H2/2),(xs[4],y2+H2/2)])
arrow([(kx-3,y1),(kx-3,y1-1.2)],c=col['s'][1],ls='--') if False else None
ax.plot([kx,kx],[y1,y1-2.5],c=EDGE,lw=1.1)
arrow([(xs[3]+W/2,y1-2.5),(sx+sw/2,y1-2.5),(sx+sw/2,y2+H2)],c=col['s'][1],ls='--')
# Stage 3
y3=3.5; H3=13; W3=(5*W+4*G-3*G)/4
t3=[('Convergent validity','Spearman and partial $\\rho$\nspeed bands\ncomponent ablation'),
    ('Monte Carlo robustness','measurement noise\nsampling-rate thinning\nweights, vehicle parameters'),
    ('Driving regimes','k-means clustering\nspeed and components\nfour regimes'),
    ('Spatial–temporal','0.01° grid, Moran’s I\nweekday vs weekend\nbetween-device ICC')]
x3=[X0+i*(W3+G) for i in range(4)]
for x,(t,b) in zip(x3,t3): box(x,y3,W3,H3,t,b,3)
bus=21
ax.plot([x3[0]+W3/2,max(x3[3]+W3/2,xs[4]+W/2)],[bus,bus],c=EDGE,lw=1.1)
ax.plot([xs[4]+W/2,xs[4]+W/2],[y2,bus],c=EDGE,lw=1.1)
ax.plot([sx+sw/2,sx+sw/2],[y2,bus],c=col['s'][1],lw=1.1,ls='--')
for x in x3: arrow([(x+W3/2,bus),(x+W3/2,y3+H3)])
# legend
ax.plot([66,69],[61,61],c=EDGE,lw=1.1); ax.text(69.6,61,'data / index flow',va='center',fontsize=7.2)
ax.plot([81,84],[61,61],c=col['s'][1],lw=1.1,ls='--'); ax.text(84.6,61,'validation-only flow',va='center',fontsize=7.2)
plt.savefig(FIGDIR+'fig1_framework.png',dpi=300,bbox_inches='tight',pad_inches=0.05); print('ok')


import pandas as pd, numpy as np, pickle
raw,log=pickle.load(open('meta.pkl','rb')); a=pd.read_pickle('agg2.pkl'); d=pd.read_pickle('points_s.pkl')
# Fig2 data characteristics
fig,ax=plt.subplots(1,3,figsize=(7.4,2.3))
dt=raw['dt']; dt=dt[(dt>0)&(dt<=30)]
ax[0].hist(dt,bins=np.arange(0.5,30.5,1),color=C['blue']); ax[0].set_xlabel('Interval between fixes (s)'); ax[0].set_ylabel('Count'); ax[0].axvline(10,ls='--',c=C['orange'],lw=1); ax[0].set_title('(a) Sampling interval')
ax[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[1].hist(a.dur,bins=np.arange(60,700,20),color=C['teal']); ax[1].set_xlabel('Segment duration (s)'); ax[1].set_title('(b) Segment duration')
ax[2].hist(a.v_mean*3.6,bins=np.arange(0,160,5),color=C['gold'],ec='white',lw=.3); ax[2].set_xlabel('Segment mean speed (km/h)'); ax[2].set_title('(c) Mean speed')
plt.tight_layout(); plt.savefig(FIGDIR+'fig2_data.png')
# Fig3 Kalman example: choose a segment with urban stop-go, long enough
cand=a[(a.dur>240)&(a.dur<420)&(a.v_mean*3.6<45)&(a.v_mean*3.6>20)].sort_values('sigma_a',ascending=False)
sid=cand.index[len(cand)//4]; s=d[d.seg==sid]
fig,ax=plt.subplots(2,1,figsize=(7.2,3.9),sharex=True)
ax[0].plot(s.tsec,s.v*3.6,'o',ms=2.8,c=C['grey'],label='Raw Doppler speed'); ax[0].plot(s.tsec,s.vs*3.6,c=C['blue'],lw=1.4,label='Kalman–RTS estimate')
ax[0].set_ylabel('Speed (km/h)'); ax[0].legend(frameon=False,loc='upper right')
ax[1].plot(s.tsec,s.acc_raw,'-',c=C['light'],lw=1,label='Finite difference (raw)'); ax[1].plot(s.tsec,s.acc,c=C['orange'],lw=1.4,label='Kalman–RTS estimate')
ax[1].axhline(0,c='k',lw=.5); ax[1].set_ylabel('Acceleration (m/s²)'); ax[1].set_xlabel('Time within segment (s)'); ax[1].legend(frameon=False,loc='upper right',ncol=2)
plt.tight_layout(); plt.savefig(FIGDIR+'fig3_kalman.png'); print('seg',sid,len(s),s.tsec.diff().median())
# Fig5 validation: EPI vs DFS hexbin and within speed bins
from scipy.stats import spearmanr
fig,ax=plt.subplots(1,2,figsize=(7.4,3.0))
hb=ax[0].hexbin(a.EPI,a.DFS*100,gridsize=40,cmap='Blues',mincnt=1,bins='log'); ax[0].set_xlabel('EPI'); ax[0].set_ylabel('Dynamic fuel share, DFS (%)')
ax[0].set_title(f'(a) EPI vs DFS, ρ = {spearmanr(a.EPI,a.DFS)[0]:.2f}'); plt.colorbar(hb,ax=ax[0],label='Segments (log scale)')
bins=['<40','40-70','70-100','>100']; cols=[C['red'],C['orange'],C['teal'],C['blue']]
for b,cc in zip(bins,cols):
    g=a[a.vbin==b]; q=pd.qcut(g.EPI,10,labels=False); m=g.groupby(q).agg(e=('EPI','median'),f=('FI','median'))
    ax[1].plot(m.e,m.f,'o-',c=cc,ms=3,label=f'{b} km/h (ρ={spearmanr(g.EPI,g.FI)[0]:.2f})')
ax[1].set_xlabel('EPI (decile medians)'); ax[1].set_ylabel('VT-CPFM fuel intensity (L/100 km)'); ax[1].set_title('(b) EPI vs FI within speed bands'); ax[1].legend(frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig(FIGDIR+'fig4_validation.png')
# Fig7 clusters
prof=pickle.load(open('res_more.pkl','rb'))['prof']
names={1:'C1 Free-flow cruising',2:'C2 Steady arterial/expressway',0:'C3 Moderate urban',3:'C4 Volatile stop-and-go'}
fig,ax=plt.subplots(1,2,figsize=(7.4,3.0))
cc={1:C['blue'],2:C['teal'],0:C['gold'],3:C['red']}
samp=a.sample(4000,random_state=1)
for k in [1,2,0,3]:
    g=samp[samp.cl==k]; ax[0].scatter(g.v_mean*3.6,g.EPI,s=3,alpha=.35,c=cc[k],label=names[k])
ax[0].set_xlabel('Segment mean speed (km/h)'); ax[0].set_ylabel('EPI'); ax[0].legend(frameon=False,fontsize=6.8,markerscale=3,loc='upper right'); ax[0].set_title('(a) Regimes in the speed–EPI plane')
order=[1,2,0,3]; x=np.arange(4); w=.38
ax[1].bar(x-w/2,[prof.loc[k,'FI'] for k in order],w,color=C['grey'],label='FI (L/100 km)')
ax2=ax[1].twinx(); ax2.bar(x+w/2,[prof.loc[k,'DFS']*100 for k in order],w,color=[cc[k] for k in order],label='DFS (%)')
ax[1].set_xticks(x); ax[1].set_xticklabels(['C1','C2','C3','C4']); ax[1].set_ylabel('Median FI (L/100 km)'); ax2.set_ylabel('Median DFS (%)')
ax2.spines['right'].set_visible(True); ax[1].set_title('(b) Surrogate energy by regime (grey: FI; colour: DFS)')
plt.tight_layout(); plt.savefig(FIGDIR+'fig6_regimes.png')
# Fig8 spatial
cell=pickle.load(open('res_more.pkl','rb'))['cell']
fig,ax=plt.subplots(figsize=(5.2,4.2))
sc=ax.scatter(cell.lon,cell.lat,c=cell.EPI,s=np.clip(cell.n,5,80)*0.7,cmap='RdYlBu_r',vmin=20,vmax=80,marker='s',edgecolors='none')
plt.colorbar(sc,ax=ax,label='Median EPI of segments in cell')
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)'); ax.set_aspect(1/np.cos(np.radians(29.3)))
ax.set_xlim(47.65,48.25); ax.set_ylim(28.95,29.45)
plt.tight_layout(); plt.savefig(FIGDIR+'fig7_spatial.png'); print((cell.lon.between(47.65,48.25)&cell.lat.between(28.95,29.45)).mean())


import pandas as pd, numpy as np, pickle
m=pd.read_pickle('mc_all.pkl'); o=pickle.load(open('res_more.pkl','rb'))
fig,ax=plt.subplots(1,3,figsize=(7.4,2.5))
n=m[(m.exp=='noise')&(m.sh==5)].groupby('sv')[['r3','r2','k3','k2']].mean()
ax[0].plot(n.index,n.r3,'o-',c=C['orange'],label='EPI-3, ρ'); ax[0].plot(n.index,n.r2,'s-',c=C['blue'],label='EPI-L, ρ')
ax[0].plot(n.index,n.k3,'o--',c=C['orange'],label='EPI-3, κ'); ax[0].plot(n.index,n.k2,'s--',c=C['blue'],label='EPI-L, κ')
ax[0].set_xlabel('Added speed noise σ (m/s)'); ax[0].set_ylabel('Agreement with baseline'); ax[0].set_ylim(0.5,1.01); ax[0].set_title('(a) Measurement noise'); ax[0].legend(frameon=False,fontsize=6.5)
t=m[m.exp=='thin'].groupby('med_dt')[['r3','r2','k3','k2']].mean(); t=m[m.exp=='thin'].groupby('keep')[['med_dt','r3','r2','k3','k2']].mean().sort_values('keep',ascending=False)
xs=list(range(5)); lab=['100%\n(3 s)']+[f'{k:.0%}\n({d:.0f} s)' for k,d in zip(t.index,t.med_dt)]
ax[1].plot(xs,[1]+list(t.r3),'o-',c=C['orange']); ax[1].plot(xs,[1]+list(t.r2),'s-',c=C['blue'])
ax[1].plot(xs,[1]+list(t.k3),'o--',c=C['orange']); ax[1].plot(xs,[1]+list(t.k2),'s--',c=C['blue'])
ax[1].set_xticks(xs); ax[1].set_xticklabels(lab,fontsize=6.5); ax[1].set_xlabel('Fixes retained (median interval)'); ax[1].set_ylim(0.5,1.01); ax[1].set_title('(b) Sampling-rate thinning')
V=o['veh_mc']; ax[2].hist(V.r3,bins=15,alpha=.75,color=C['orange'],label='EPI-3'); ax[2].hist(V.r2,bins=15,alpha=.75,color=C['blue'],label='EPI-L')
ax[2].set_xlabel('ρ with vehicle-specific DFS'); ax[2].set_ylabel('Vehicle draws'); ax[2].set_title('(c) Vehicle-parameter uncertainty'); ax[2].legend(frameon=False)
plt.tight_layout(); plt.savefig(FIGDIR+'fig5_robustness.png')
