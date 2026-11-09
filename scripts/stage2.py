"""Second stage on bank-year observations.
M1 Simar-Wilson (2007) Algorithm 1: truncated (at 1) normal regression of Farrell-inverse cost inefficiency
   delta = 1/CE on Z, parametric bootstrap (B=2000) percentile CIs.
M2 Same on bootstrap bias-corrected CE (Simar-Wilson 1998) as dependent variable.
M3 Same with TE (bias-corrected) as dependent variable.
M4 OLS on CE (efficiency scale) with bank-clustered SEs (McDonald 2009 robustness).
Positive coefficient in M1-M3 = HIGHER inefficiency (LOWER efficiency)."""
import numpy as np, pandas as pd, sys, statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm
D, B = sys.argv[1], int(sys.argv[2])
p = pd.read_csv(f'{D}/panel_B.csv'); r = pd.read_csv(f'{D}/dea_scores.csv'); bs = pd.read_csv(f'{D}/boot_scores.csv')
d = p.merge(r[['dmu','CE_vrs']], on='dmu').merge(bs[['dmu','CE_bc','TE_bc']], on='dmu')
d['psb_x_2021_23'] = d.psb * (d.year >= 2021)
Zc = ['psb','gnpa_ratio','car','ln_assets','psl_share','merged_post','psb_x_2021_23']
Yd = pd.get_dummies(d.year, prefix='y', drop_first=True).astype(float)
Z = pd.concat([pd.Series(1.0, index=d.index, name='const'), d[Zc].astype(float), Yd], axis=1)
Zm = Z.values; k = Zm.shape[1]
def nll(th, y):
    b, ls = th[:k], th[k]; sg = np.exp(ls); mu = Zm @ b
    return -(norm.logpdf((y-mu)/sg) - ls - norm.logsf((1-mu)/sg)).sum()
def fit(y, x0=None):
    if x0 is None:
        ols = np.linalg.lstsq(Zm, y, rcond=None)[0]; x0 = np.r_[ols, np.log(np.std(y - Zm@ols))]
    res = minimize(nll, x0, args=(y,), method='BFGS', options={'maxiter': 5000, 'gtol': 1e-6})
    return res.x
def algo1(y, label):
    th = fit(y); b, sg = th[:k], np.exp(th[k]); mu = Zm @ b
    rng = np.random.default_rng(42); draws = np.empty((B, k+1))
    a = (1 - mu)/sg
    for i in range(B):
        e = truncnorm.rvs(a, np.inf, loc=0, scale=sg, random_state=rng)
        draws[i] = fit(mu + e, th)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0); se = draws.std(0)
    names = list(Z.columns) + ['sigma(log)']
    t = pd.DataFrame({'coef': th, 'boot_se': se, 'ci_lo': lo, 'ci_hi': hi}, index=names)
    t['sig_95'] = (t.ci_lo > 0) | (t.ci_hi < 0); t['model'] = label
    return t
tabs = [algo1(1/d.CE_vrs.values, 'M1 SW-Alg1: 1/CE'),
        algo1(1/d.CE_bc.values, 'M2 SW: 1/CE bias-corrected'),
        algo1(1/d.TE_bc.values, 'M3 SW: 1/TE bias-corrected')]
ols = sm.OLS(d.CE_vrs.values, Zm).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.bank)[0]})
ci = ols.conf_int()
t4 = pd.DataFrame({'coef': ols.params, 'boot_se': ols.bse, 'ci_lo': ci[:,0], 'ci_hi': ci[:,1]}, index=Z.columns)
t4['sig_95'] = (t4.ci_lo > 0) | (t4.ci_hi < 0); t4['model'] = 'M4 OLS on CE, bank-clustered SE'
T = pd.concat(tabs + [t4]); T.to_csv(f'{D}/table5_second_stage.csv')
show = T.loc[T.index.isin(Zc)].reset_index().rename(columns={'index':'var'})
print(show.round(4).to_string())
print('\nN =', len(d), '| banks =', d.bank.nunique(), '| year FE included')
d[['dmu','bank','year','ownership'] + Zc + ['CE_vrs','CE_bc','TE_bc']].to_csv(f'{D}/stage2_data.csv', index=False)
