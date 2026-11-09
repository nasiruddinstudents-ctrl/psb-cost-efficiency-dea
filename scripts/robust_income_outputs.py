"""Robustness: income-based outputs as in the original submission (net interest income, non-interest income),
same three inputs and prices, pooled VRS frontier."""
import numpy as np, pandas as pd, sys
from scipy.optimize import linprog
from scipy import stats
D=sys.argv[1]
p=pd.read_csv(f'{D}/panel_B.csv'); r=pd.read_csv(f'{D}/dea_scores.csv')
p['nii_r']=(p.interest_earned-p.interest_expended)*100/p.deflator
ok=np.where(p.nii_r>0)[0]
Xc,Wc=['funds','employees','fixed_assets_r'],['w_funds','w_labour','w_capital']
sx=p[Xc].mean().values; X=p[Xc].values/sx; W=p[Wc].values*sx; Y=(p[['nii_r','other_income_r']]/p[['nii_r','other_income_r']].mean()).values
def ce(o,R):
    k=len(R); c=np.r_[W[o],np.zeros(k)]
    A=np.r_[np.c_[-np.eye(3),X[R].T],np.c_[np.zeros((2,3)),-Y[R].T]]
    return linprog(c,A,np.r_[np.zeros(3),-Y[o]],np.r_[np.zeros(3),np.ones(k)][None],[1],bounds=[(0,None)]*(3+k),method='highs').fun/(W[o]@X[o])
v=np.full(len(p),np.nan); v[ok]=[ce(o,ok) for o in ok]
r['CE_income_outputs']=v; g=r.dropna(subset=['CE_income_outputs'])
rho=stats.spearmanr(g.CE_vrs,g.CE_income_outputs)[0]
g=g.assign(period=pd.cut(g.year,[2008,2015,2020,2023],labels=['2009-15','2016-20','2021-23']))
t=g.pivot_table(index='period',columns='ownership',values='CE_income_outputs',observed=True)
print('n',len(g),'dropped (NII<=0):',len(p)-len(ok),'spearman',round(rho,3)); print(t.round(3))
r[['dmu','CE_income_outputs']].to_csv(f'{D}/robust_income_outputs.csv',index=False)
