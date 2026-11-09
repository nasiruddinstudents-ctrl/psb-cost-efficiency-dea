"""(1) Daraio & Simar (2005, 2007) conditional cost/technical efficiency for continuous Z
(GNPA ratio, CRAR, log real assets): reference set = bank-years with |z_j - z_o| <= h (uniform kernel),
h = 1.06*sd*n^(-1/5). Ratio Q = eff(x,y|z)/eff(x,y) >= 1; Q rising with z => z unfavourable
(acts like an extra input that shifts the frontier). Summarised by z-quintile + Spearman(Q, z).
(2) Cost metafrontier (O'Donnell, Rao & Battese 2008): group frontiers for PSB and private,
technology gap ratio TGR = CE_meta / CE_group."""
import numpy as np, pandas as pd, sys
from scipy.optimize import linprog
from scipy import stats
D = sys.argv[1]
p = pd.read_csv(f'{D}/panel_B.csv'); r = pd.read_csv(f'{D}/dea_scores.csv')
Xc, Wc, Yc = ['funds','employees','fixed_assets_r'], ['w_funds','w_labour','w_capital'], ['advances_r','investments_r','other_income_r']
sx = p[Xc].mean().values; X = p[Xc].values/sx; W = p[Wc].values*sx; Y = (p[Yc]/p[Yc].mean()).values
n = len(p); m = s = 3; H = dict(method='highs')
def te(o, R):
    k=len(R); c=np.r_[1,np.zeros(k)]
    A=np.r_[np.c_[-X[o][:,None],X[R].T],np.c_[np.zeros((s,1)),-Y[R].T]]
    return linprog(c,A,np.r_[np.zeros(m),-Y[o]],np.r_[0,np.ones(k)][None],[1],bounds=[(0,None)]*(k+1),**H).x[0]
def ce(o, R):
    k=len(R); c=np.r_[W[o],np.zeros(k)]
    A=np.r_[np.c_[-np.eye(m),X[R].T],np.c_[np.zeros((s,m)),-Y[R].T]]
    return linprog(c,A,np.r_[np.zeros(m),-Y[o]],np.r_[np.zeros(m),np.ones(k)][None],[1],bounds=[(0,None)]*(m+k),**H).fun/(W[o]@X[o])
res = r[['dmu','year','bank','ownership','TE_I_vrs','CE_vrs']].copy(); summ = []
for z in ['gnpa_ratio','car','ln_assets']:
    v = p[z].values; h = 1.06*v.std()*n**(-1/5)
    tc = np.empty(n); cc = np.empty(n)
    for o in range(n):
        R = np.where(np.abs(v - v[o]) <= h)[0]; tc[o] = te(o, R); cc[o] = ce(o, R)
    res[f'QTE_{z}'] = tc/res.TE_I_vrs; res[f'QCE_{z}'] = cc/res.CE_vrs
    q = pd.qcut(v, 5, labels=[f'Q{i}' for i in range(1,6)])
    for kind in ('QTE','QCE'):
        prof = res.groupby(q, observed=True)[f'{kind}_{z}'].mean()
        rho, pv = stats.spearmanr(v, res[f'{kind}_{z}'])
        summ.append(dict(Z=z, ratio=kind, bandwidth=h, **{k: prof[k] for k in prof.index}, spearman=rho, p=pv,
                         reading='unfavourable (raises cost frontier)' if (rho>0 and pv<.05) else
                                 ('favourable' if (rho<0 and pv<.05) else 'no clear frontier effect')))
    print(z, 'done', flush=True)
S = pd.DataFrame(summ); S.to_csv(f'{D}/table_conditional.csv', index=False); print(S.round(4).to_string())
G = {g: np.where(p.ownership==g)[0] for g in ['PSB','PVT']}
res['CE_group'] = np.nan
for g, idx in G.items(): res.loc[idx, 'CE_group'] = [ce(o, idx) for o in idx]
res['TGR'] = res.CE_vrs/res.CE_group
res.to_csv(f'{D}/conditional_metafrontier_scores.csv', index=False)
M = res.assign(period=pd.cut(res.year,[2008,2015,2020,2023],labels=['2009-15','2016-20','2021-23'])) \
       .pivot_table(index='period', columns='ownership', values=['CE_group','TGR','CE_vrs'], observed=True)
M.to_csv(f'{D}/table_metafrontier.csv'); print(M.round(3).to_string())
a, b = res[res.ownership=='PSB'].TGR, res[res.ownership=='PVT'].TGR
print('TGR mean PSB', a.mean().round(3), 'PVT', b.mean().round(3), 'MW p', stats.mannwhitneyu(a,b).pvalue)
