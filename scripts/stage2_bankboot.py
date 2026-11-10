"""Bank-level (cluster) bootstrap for second-stage model M1 and M2: resample the 52 banks with replacement
(all years of a drawn bank together), refit the truncated regression, percentile 95% intervals. B replications."""
import numpy as np, pandas as pd, sys
from scipy.optimize import minimize
from scipy.stats import norm
D, B = sys.argv[1], int(sys.argv[2])
d = pd.read_csv(f'{D}/stage2_data.csv')
Zc = ['psb','gnpa_ratio','car','ln_assets','psl_share','merged_post','psb_x_2021_23']
Yd = pd.get_dummies(d.year, prefix='y', drop_first=True).astype(float)
Z = pd.concat([pd.Series(1.0, index=d.index, name='const'), d[Zc].astype(float), Yd], axis=1)
names = list(Z.columns)
def fit(Zm, y, x0=None):
    k = Zm.shape[1]
    def nll(th):
        b, ls = th[:k], th[k]; sg = np.exp(ls); mu = Zm @ b
        return -(norm.logpdf((y-mu)/sg) - ls - norm.logsf((1-mu)/sg)).sum()
    if x0 is None:
        o = np.linalg.lstsq(Zm, y, rcond=None)[0]; x0 = np.r_[o, np.log(np.std(y - Zm@o))]
    return minimize(nll, x0, method='BFGS', options={'maxiter':5000,'gtol':1e-6}).x
banks = d.bank.unique(); idx = {b: np.where(d.bank.values == b)[0] for b in banks}
out = []
for label, y in [('M1 1/CE', 1/d.CE_vrs.values), ('M2 1/CE bias-corrected', 1/d.CE_bc.values)]:
    Zm = Z.values; th = fit(Zm, y); rng = np.random.default_rng(2026); draws = []
    for i in range(B):
        rows = np.concatenate([idx[b] for b in rng.choice(banks, len(banks))])
        Zb = Zm[rows]; keep = Zb.std(0) > 0; keep[0] = True          # drop columns constant in the draw
        tb = np.full(len(names)+1, np.nan); est = fit(Zb[:, keep], y[rows])
        tb[np.r_[np.where(keep)[0], len(names)]] = est; draws.append(tb)
    dr = np.array(draws); lo, hi = np.nanpercentile(dr, [2.5, 97.5], axis=0)
    for j, v in enumerate(names):
        if v in Zc: out.append(dict(model=label, var=v, coef=th[j], ci_lo=lo[j], ci_hi=hi[j], sig_95=(lo[j]>0) or (hi[j]<0), valid_draws=int(np.isfinite(dr[:,j]).sum())))
    print(label, 'done', flush=True)
o = pd.DataFrame(out); o.to_csv(f'{D}/table13_bank_bootstrap.csv', index=False); print(o.round(4).to_string())
