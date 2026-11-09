"""Robustness: (a) PSB-only pooled frontier, (b) excluding SBI C&I and Yes Bank, (c) 12 surviving PSBs,
2009-12 vs 2020-23 on the common pooled frontier (the original paper's two windows, now comparable)."""
import numpy as np, pandas as pd, sys
from scipy.optimize import linprog
from scipy import stats
D = sys.argv[1]
p = pd.read_csv(f'{D}/panel_B.csv'); r = pd.read_csv(f'{D}/dea_scores.csv')
Xc, Wc, Yc = ['funds','employees','fixed_assets_r'], ['w_funds','w_labour','w_capital'], ['advances_r','investments_r','other_income_r']
sx = p[Xc].mean().values; X = p[Xc].values/sx; W = p[Wc].values*sx; Y = (p[Yc]/p[Yc].mean()).values; m = s = 3
def ce(o, R):
    k = len(R); c = np.r_[W[o], np.zeros(k)]
    A = np.r_[np.c_[-np.eye(m), X[R].T], np.c_[np.zeros((s,m)), -Y[R].T]]
    res = linprog(c, A, np.r_[np.zeros(m), -Y[o]], np.r_[np.zeros(m), np.ones(k)][None], [1], bounds=[(0,None)]*(m+k), method='highs')
    return res.fun/(W[o]@X[o])
out = r[['dmu','year','bank','ownership','CE_vrs']].copy()
psb = np.where(p.psb == 1)[0]
out['CE_psb_frontier'] = np.nan
out.loc[psb, 'CE_psb_frontier'] = [ce(o, psb) for o in psb]
keep = np.where(~p.bank.str.contains('SBI COMMERCIAL|YES BANK'))[0]
out['CE_excl_outliers'] = np.nan
out.loc[keep, 'CE_excl_outliers'] = [ce(o, keep) for o in keep]
out.to_csv(f'{D}/robustness_scores.csv', index=False)
rows = []
for col in ['CE_psb_frontier','CE_excl_outliers']:
    g = out.dropna(subset=[col])
    rows.append(dict(check=col, n=len(g), spearman_with_main=stats.spearmanr(g.CE_vrs, g[col])[0],
                     mean_main=g.CE_vrs.mean(), mean_alt=g[col].mean()))
print(pd.DataFrame(rows).round(3).to_string())
g = out[out.CE_excl_outliers.notna()]
print('\nExcl outliers, PSB-PVT by period:'); g = g.assign(period=pd.cut(g.year,[2008,2015,2020,2023]))
print(g.pivot_table(index='period', columns='ownership', values='CE_excl_outliers', observed=True).round(3))
surv = p[p.year == 2023].bank[p.psb == 1].tolist()
s12 = out[out.bank.isin(surv)]
w1 = s12[s12.year.between(2009, 2012)].groupby('bank').CE_vrs.mean()
w2 = s12[s12.year.between(2020, 2023)].groupby('bank').CE_vrs.mean()
t = stats.wilcoxon(w1, w2); tt = stats.ttest_rel(w1, w2)
S = pd.DataFrame({'CE_2009_12': w1, 'CE_2020_23': w2}); S['change'] = S.CE_2020_23 - S.CE_2009_12
S.to_csv(f'{D}/table_12_surviving.csv'); print('\n12 surviving PSBs:\n', S.round(3).to_string())
print(f'mean {w1.mean():.3f} -> {w2.mean():.3f}; paired t p={tt.pvalue:.3f}; Wilcoxon p={t.pvalue:.3f}')
print('\nCE by year, 12 surviving PSBs:'); print(s12.groupby('year').CE_vrs.mean().round(3).to_string())
