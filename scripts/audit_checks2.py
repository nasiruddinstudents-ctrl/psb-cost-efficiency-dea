"""Second audit round: (1) Type I/II identities; (2) price-of-capital winsorisation; (3) NPA DDF with
output-oriented direction (0,+y,-b); (4) conditional-efficiency bandwidth sensitivity and local sample sizes;
(5) pooled-frontier-defining bank-years and leave-one-bank-out metafrontier TGR; (6) annual sample accounting."""
import numpy as np, pandas as pd, sys
from scipy.optimize import linprog
from scipy import stats
D = sys.argv[1]
p = pd.read_csv(f'{D}/panel_B.csv'); r = pd.read_csv(f'{D}/dea_scores.csv')
Xc, Wc, Yc = ['funds','employees','fixed_assets_r'], ['w_funds','w_labour','w_capital'], ['advances_r','investments_r','other_income_r']
sx = p[Xc].mean().values; X = p[Xc].values/sx; Y = (p[Yc]/p[Yc].mean()).values; Bv = (p.gnpa_r/p.gnpa_r.mean()).values
n = len(p); m = s = 3; H = dict(method='highs'); per = pd.cut(p.year,[2008,2015,2020,2023],labels=['2009–15','2016–20','2021–23'])
out = {}
# (1) identities
for t in ['vrs','crs']:
    out[f'id_I_{t}'] = np.abs(r[f'CE_{t}'] - r[f'TE_I_{t}']*r[f'AE_I_{t}']).max()
    out[f'id_II_{t}'] = np.abs(r[f'CE_{t}'] - r[f'TE_II_{t}']*r[f'AE_II_{t}']).max()
    out[f'ineq_{t}'] = int(((r[f'CE_{t}'] <= r[f'TE_II_{t}']+1e-7) & (r[f'TE_II_{t}'] <= r[f'TE_I_{t}']+1e-7)).sum())
print('(1) max|CE-TE*AE| Type I / II (VRS):', out['id_I_vrs'], out['id_II_vrs'], '| inequality holds for', out['ineq_vrs'], 'of', n)
def ce(o, R, W):
    k=len(R); c=np.r_[W[o],np.zeros(k)]
    A=np.r_[np.c_[-np.eye(m),X[R].T],np.c_[np.zeros((s,m)),-Y[R].T]]
    return linprog(c,A,np.r_[np.zeros(m),-Y[o]],np.r_[np.zeros(m),np.ones(k)][None],[1],bounds=[(0,None)]*(m+k),**H).fun/(W[o]@X[o])
ALL = np.arange(n)
# (2) winsorise w_capital at 1st/99th percentiles
wc = p.w_capital.clip(*p.w_capital.quantile([.01,.99]))
print('(2) w_capital distribution: p1 %.3f p50 %.3f p99 %.3f max %.3f'%tuple(p.w_capital.quantile([.01,.5,.99]).tolist()+[p.w_capital.max()]))
W2 = np.c_[p.w_funds, p.w_labour, wc].astype(float)*sx
c2 = np.array([ce(o, ALL, W2) for o in ALL])
g = pd.DataFrame({'per':per,'own':p.ownership,'c':c2}).groupby(['per','own'],observed=True).c.mean().unstack()
print('    Spearman with main %.3f'%stats.spearmanr(r.CE_vrs, c2)[0]); print(g.round(3).to_string())
# (3) output-oriented DDF direction (0,+y,-b)
def ddf_out(o):
    c=np.r_[-1,np.zeros(n)]
    A=np.r_[np.c_[np.zeros((m,1)),X.T],np.c_[Y[o][:,None],-Y.T]]
    return linprog(c,A,np.r_[X[o],-Y[o]],np.r_[Bv[o],Bv][None],[Bv[o]],bounds=[(0,None)]*(n+1),**H).x[0]
d3 = 1/(1+np.array([ddf_out(o) for o in ALL]))
print('(3) DDF direction (0,+y,-b): Spearman with main NPA-adjusted %.3f; PSB %.3f PVT %.3f MW p=%.2g'%(stats.spearmanr(r.NPA_eff,d3)[0],
      d3[p.psb==1].mean(), d3[p.psb==0].mean(), stats.mannwhitneyu(d3[p.psb==1],d3[p.psb==0]).pvalue))
# (4) conditional bandwidth sensitivity
Wm = p[Wc].values*sx
rows=[]
for z in ['gnpa_ratio','car','ln_assets']:
    v = p[z].values; h0 = 1.06*v.std()*n**(-1/5)
    for f in [0.5, 1.0, 2.0]:
        h = h0*f; loc = np.array([(np.abs(v-v[o])<=h).sum() for o in ALL])
        q = np.array([ce(o, np.where(np.abs(v-v[o])<=h)[0], Wm) for o in ALL]) / r.CE_vrs.values
        rho, pv = stats.spearmanr(v, q)
        rows.append(dict(Z=z, bandwidth_factor=f, h=h, local_n_median=int(np.median(loc)), local_n_min=int(loc.min()), spearman=rho, p=pv))
C = pd.DataFrame(rows); C.to_csv(f'{D}/table_conditional_bandwidth.csv', index=False); print('(4)'); print(C.round(3).to_string())
# (5) frontier-defining bank-years and leave-one-bank-out TGR
fd = r[r.CE_vrs > 0.9999]
print('(5) pooled-frontier bank-years (CE=1):', len(fd), dict(fd.ownership.value_counts()), sorted(fd.bank.unique()))
psb_idx = np.where(p.psb==1)[0]
lobo=[]
for b in p.bank.unique():
    keep = np.where(p.bank != b)[0]; kp = np.intersect1d(keep, psb_idx)
    pooled = np.array([ce(o, keep, Wm) for o in kp]); grp = np.array([ce(o, kp, Wm) for o in kp])
    t = pd.Series(pooled/grp).groupby(per.values[kp]).mean()
    lobo.append(dict(dropped=b, **{str(k):v for k,v in t.items()}))
L = pd.DataFrame(lobo); L.to_csv(f'{D}/table_metafrontier_lobo.csv', index=False)
print('    LOBO PSB TGR range by period:'); print(L[['2009–15','2016–20','2021–23']].agg(['min','max']).round(3).to_string())
for k in ['2009–15','2016–20','2021–23']: print('   ', k, 'most influential drop:', L.loc[L[k].idxmax(),'dropped'], round(L[k].max(),3))
# (6) annual sample accounting
yrs = sorted(p.year.unique()); acc=[]
for y in yrs:
    cur = p[p.year==y]; prev = set(p[p.year==y-1].bank); nxt = set(p[p.year==y+1].bank)
    acc.append(dict(year=y, PSB=int((cur.psb==1).sum()), Private=int((cur.psb==0).sum()), total=len(cur),
                    entries=len(set(cur.bank)-prev) if y>yrs[0] else 0, exits_after=len(set(cur.bank)-nxt) if y<yrs[-1] else 0))
A = pd.DataFrame(acc); A.to_csv(f'{D}/table_sample_accounting.csv', index=False); print('(6)'); print(A.to_string(index=False))
