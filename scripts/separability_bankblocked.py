"""Bank-blocked version: sample halves and jackknife halves are formed by splitting BANKS, so no bank appears in both halves."""
"""Daraio, Simar & Wilson (2018) test of the separability condition, with
Kneip, Simar & Wilson (2015) generalized-jackknife bias correction and subsample means
(p+q=6 > 3 under VRS, kappa=2/(p+q+1)<1/2). Discrete conditioning variables.
H0: Z does not shift the attainable set (separability holds) -> second stage meaningful as is."""
import numpy as np, pandas as pd, sys
from scipy.optimize import linprog
from scipy.stats import norm
D = sys.argv[1]; SPLITS = int(sys.argv[2]); K = int(sys.argv[3])
p = pd.read_csv(f'{D}/panel_B.csv')
Xc, Yc = ['funds','employees','fixed_assets_r'], ['advances_r','investments_r','other_income_r']
X = (p[Xc] / p[Xc].mean()).values; Y = (p[Yc] / p[Yc].mean()).values
n = len(p); m = s = 3; kappa = 2/(m+s+1)
def te(o, R):
    k = len(R); c = np.r_[1, np.zeros(k)]
    A = np.r_[np.c_[-X[o][:,None], X[R].T], np.c_[np.zeros((s,1)), -Y[R].T]]
    r = linprog(c, A, np.r_[np.zeros(m), -Y[o]], np.r_[0, np.ones(k)][None], [1], bounds=[(0,None)]*(k+1), method='highs')
    return r.x[0]
def scores(idx, z=None):
    idx = np.asarray(idx)
    if z is None: return np.array([te(o, idx) for o in idx])
    return np.array([te(o, idx[z[idx] == z[o]]) for o in idx])
def bias(idx, z, rng):
    full = scores(idx, z).mean(); half = []
    for _ in range(K):
        ub = p.bank.values[idx]; bb = rng.permutation(np.unique(ub)); h1 = set(bb[:len(bb)//2]); m1 = np.isin(ub, list(h1)); a, b = idx[m1], idx[~m1]
        half.append((scores(a, z).mean() + scores(b, z).mean())/2)
    return (np.mean(half) - full) / (2**kappa - 1)
tests = {'Ownership (PSB vs private)': p.psb.values,
         'GNPA ratio terciles': pd.qcut(p.gnpa_ratio, 3, labels=False).values,
         'Period (2009-15 / 2016-20 / 2021-23)': pd.cut(p.year, [2008,2015,2020,2023], labels=False).values}
out = []
for name, z in tests.items():
    for sp in range(SPLITS):
        rng = np.random.default_rng(7 + sp)
        banks = p.bank.unique(); bp = rng.permutation(banks); g1 = set(bp[:len(bp)//2]); S1 = np.where(p.bank.isin(g1))[0]; S2 = np.where(~p.bank.isin(g1))[0]
        u, c = scores(S1), scores(S2, z)
        n1k, n2k = int(len(S1)**(2*kappa)), int(len(S2)**(2*kappa))
        mu1 = rng.choice(u, n1k, replace=False).mean(); mu2 = rng.choice(c, n2k, replace=False).mean()
        B1, B2 = bias(S1, None, rng), bias(S2, z, rng)
        tau = ((mu1 - mu2) - (B1 - B2)) / np.sqrt(u.var(ddof=1)/n1k + c.var(ddof=1)/n2k)
        out.append(dict(Z=name, split=sp, mu_uncond=mu1, mu_cond=mu2, B_uncond=B1, B_cond=B2, tau=tau, p=2*norm.sf(abs(tau))))
    print(name, 'done', flush=True)
o = pd.DataFrame(out); o.to_csv(f'{D}/separability_splits_bankblocked.csv', index=False)
summ = o.groupby('Z').agg(splits=('p','size'), median_tau=('tau','median'), median_p=('p','median'),
                          share_reject_5pct=('p', lambda x: (x < .05).mean()))
summ.to_csv(f'{D}/table_separability_bankblocked.csv'); print(summ.round(4).to_string())
