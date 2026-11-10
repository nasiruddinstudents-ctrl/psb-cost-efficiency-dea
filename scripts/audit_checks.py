"""Audit checks: (a) sequential (no look-ahead) frontier: bank-year t benchmarked only against bank-years <= t;
(b) NPA-adjusted DDF with net instead of gross NPAs; (c) bank-level paired Type I vs II test (52 bank means);
(d) ownership comparison by period, full sample vs excluding Yes Bank and SBI C&I; (e) cost reconciliation check."""
import numpy as np, pandas as pd, sys
from scipy.optimize import linprog
from scipy import stats
D = sys.argv[1]
p = pd.read_csv(f'{D}/panel_B.csv'); r = pd.read_csv(f'{D}/dea_scores.csv'); rob = pd.read_csv(f'{D}/robustness_scores.csv')
Xc, Wc, Yc = ['funds','employees','fixed_assets_r'], ['w_funds','w_labour','w_capital'], ['advances_r','investments_r','other_income_r']
sx = p[Xc].mean().values; X = p[Xc].values/sx; W = p[Wc].values*sx; Y = (p[Yc]/p[Yc].mean()).values; m = s = 3
H = dict(method='highs')
# (e) reconciliation: sum of w*x equals interest expended + operating expenses (real)
recon = (p[Xc].values*p[Wc].values).sum(1) - p.total_cost_r.values
print('(e) max |sum(w*x) - total cost| (Rs crore, real):', np.abs(recon).max().round(6))
def ce(o, R):
    k=len(R); c=np.r_[W[o],np.zeros(k)]
    A=np.r_[np.c_[-np.eye(m),X[R].T],np.c_[np.zeros((s,m)),-Y[R].T]]
    return linprog(c,A,np.r_[np.zeros(m),-Y[o]],np.r_[np.zeros(m),np.ones(k)][None],[1],bounds=[(0,None)]*(m+k),**H).fun/(W[o]@X[o])
seq = np.array([ce(o, np.where(p.year <= p.year[o])[0]) for o in range(len(p))])
r['CE_sequential'] = seq
r['period'] = pd.cut(r.year,[2008,2015,2020,2023],labels=['2009–15','2016–20','2021–23'])
print('(a) sequential frontier: Spearman with pooled', round(stats.spearmanr(r.CE_vrs, seq)[0],3))
rows=[]
for per,g in r.groupby('period',observed=True):
    a,b=g[g.ownership=='PSB'].CE_sequential,g[g.ownership=='PVT'].CE_sequential
    rows.append(dict(period=per,CE_seq_PSB=a.mean(),CE_seq_PVT=b.mean(),diff=a.mean()-b.mean(),MW_p=stats.mannwhitneyu(a,b).pvalue))
A=pd.DataFrame(rows); print(A.round(4).to_string())
surv = p[(p.year==2023)&(p.psb==1)].bank.unique()
s12 = r[r.bank.isin(surv)]
w1=s12[s12.year<=2012].groupby('bank').CE_sequential.mean(); w2=s12[s12.year>=2020].groupby('bank').CE_sequential.mean()
print('(a) 12 survivors, sequential: %.3f -> %.3f, Wilcoxon p=%.3f'%(w1.mean(),w2.mean(),stats.wilcoxon(w1,w2).pvalue))
# (b) DDF with net NPAs
Bn = (p.nnpa*100/p.deflator).values; Bn = Bn/np.nanmean(Bn)
ok = np.where(np.isfinite(Bn) & (Bn>0))[0]
def ddf(o, R, Bv):
    k=len(R); c=np.r_[-1,np.zeros(k)]
    A=np.r_[np.c_[X[o][:,None],X[R].T],np.c_[Y[o][:,None],-Y[R].T]]
    return linprog(c,A,np.r_[X[o],-Y[o]],np.r_[Bv[o],Bv[R]][None],[Bv[o]],bounds=[(0,None)]*(k+1),**H).x[0]
nn = np.full(len(p),np.nan); nn[ok]=[1/(1+ddf(o,ok,Bn)) for o in ok]
r['NPA_eff_net']=nn; g=r.dropna(subset=['NPA_eff_net'])
print('(b) net-NPA DDF: n=%d, Spearman with gross-NPA version %.3f; mean PSB %.3f PVT %.3f; MW p=%.3g'%(len(g),stats.spearmanr(g.NPA_eff,g.NPA_eff_net)[0],
      g[g.ownership=='PSB'].NPA_eff_net.mean(),g[g.ownership=='PVT'].NPA_eff_net.mean(),stats.mannwhitneyu(g[g.ownership=='PSB'].NPA_eff_net,g[g.ownership=='PVT'].NPA_eff_net).pvalue))
print('    gross-NPA version same sample: PSB %.3f PVT %.3f, MW p=%.3g'%(g[g.ownership=='PSB'].NPA_eff.mean(),g[g.ownership=='PVT'].NPA_eff.mean(),stats.mannwhitneyu(g[g.ownership=='PSB'].NPA_eff,g[g.ownership=='PVT'].NPA_eff).pvalue))
# (c) bank-level paired tests
bm = r.groupby('bank')[['TE_I_vrs','TE_II_vrs','AE_I_vrs','AE_II_vrs']].mean()
for a_,b_ in [('TE_I_vrs','TE_II_vrs'),('AE_I_vrs','AE_II_vrs')]:
    d=bm[a_]-bm[b_]; print('(c) bank-level %s-%s: n=%d mean %.4f t p=%.2g Wilcoxon p=%.2g banks with nonzero diff %d'%(a_,b_,len(d),d.mean(),stats.ttest_rel(bm[a_],bm[b_]).pvalue,stats.wilcoxon(bm[a_],bm[b_]).pvalue,(d.abs()>1e-4).sum()))
# (d) full vs excluding outliers
x = r.merge(rob[['dmu','CE_excl_outliers']],on='dmu')
rows=[]
for per,g in x.groupby('period',observed=True):
    for lab,col,gg in [('Full sample','CE_vrs',g),('Excl. Yes Bank & SBI C&I','CE_excl_outliers',g.dropna(subset=['CE_excl_outliers']))]:
        a,b=gg[gg.ownership=='PSB'][col],gg[gg.ownership=='PVT'][col]
        rows.append(dict(period=str(per),sample=lab,n_psb=len(a),n_pvt=len(b),CE_psb=a.mean(),CE_pvt=b.mean(),diff=a.mean()-b.mean(),MW_p=stats.mannwhitneyu(a,b).pvalue))
T7=pd.DataFrame(rows); T7.to_csv(f'{D}/table7_full_vs_excl.csv',index=False); print(T7.round(4).to_string())
A.to_csv(f'{D}/table_sequential.csv',index=False)
r[['dmu','CE_sequential','NPA_eff_net']].to_csv(f'{D}/audit_scores.csv',index=False)
pd.DataFrame({'bank':w1.index,'seq_0912':w1.values,'seq_2023':w2.values}).to_csv(f'{D}/table_12_surviving_sequential.csv',index=False)
