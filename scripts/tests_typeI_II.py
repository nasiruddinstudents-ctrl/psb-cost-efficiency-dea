import pandas as pd, numpy as np, sys
from scipy import stats
D = sys.argv[1]
r = pd.read_csv(f'{D}/dea_scores.csv')
rows = []
for grp, g in [('All', r)] + list(r.groupby('ownership')):
    for a, b, lab in [('TE_I_vrs','TE_II_vrs','TE: Type I vs Type II'), ('AE_I_vrs','AE_II_vrs','AE: Type I vs Type II')]:
        d = g[a] - g[b]
        t = stats.ttest_rel(g[a], g[b]); w = stats.wilcoxon(g[a], g[b])
        rows.append(dict(sample=grp, comparison=lab, n=len(g), mean_I=g[a].mean(), mean_II=g[b].mean(),
                         mean_diff=d.mean(), share_differ=(d.abs() > 1e-4).mean(), t=t.statistic, p_t=t.pvalue,
                         wilcoxon_W=w.statistic, p_w=w.pvalue))
T2 = pd.DataFrame(rows); T2.to_csv(f'{D}/table2_typeI_vs_II.csv', index=False)
print(T2.round(4).to_string())
cols = ['TE_I_vrs','TE_II_vrs','AE_I_vrs','AE_II_vrs','CE_vrs','CE_crs','CE_contemp_vrs','NPA_eff']
S = r[cols].corr('spearman'); S.to_csv(f'{D}/table3_spearman.csv'); print(S.round(3).to_string())
# PSB vs PVT difference tests on CE by period
r['period'] = pd.cut(r.year, [2008, 2015, 2020, 2023], labels=['2009-15 pre-AQR', '2016-20 AQR/IBC/merger', '2021-23 post-merger'])
rows = []
for per, g in r.groupby('period', observed=True):
    a, b = g[g.ownership=='PSB'].CE_vrs, g[g.ownership=='PVT'].CE_vrs
    rows.append(dict(period=per, n_psb=len(a), n_pvt=len(b), CE_psb=a.mean(), CE_pvt=b.mean(),
                     diff=a.mean()-b.mean(), MW_p=stats.mannwhitneyu(a, b).pvalue))
P = pd.DataFrame(rows); P.to_csv(f'{D}/table_psb_vs_pvt_period.csv', index=False); print(P.round(4).to_string())
