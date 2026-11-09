"""Simar & Wilson (1998) smoothed homogeneous bootstrap, input-oriented VRS TE on the pooled frontier,
and the same procedure for cost efficiency (CE) re-solved against each pseudo-frontier.
Outputs bias-corrected scores and 95% percentile intervals (Farrell scale, <=1)."""
import numpy as np, pandas as pd, sys
from scipy.optimize import linprog
from multiprocessing import Pool
D, B = sys.argv[1], int(sys.argv[2])
p = pd.read_csv(f'{D}/panel_B.csv'); r = pd.read_csv(f'{D}/dea_scores.csv')
Xc, Wc, Yc = ['funds','employees','fixed_assets_r'], ['w_funds','w_labour','w_capital'], ['advances_r','investments_r','other_income_r']
sx = p[Xc].mean().values; sy = p[Yc].mean().values
X = p[Xc].values/sx; W = p[Wc].values*sx; Y = p[Yc].values/sy
n, m, s = X.shape[0], 3, 3
H = dict(method='highs')
def te(xo, yo, XR):
    c = np.r_[1, np.zeros(n)]
    A = np.r_[np.c_[-xo[:,None], XR.T], np.c_[np.zeros((s,1)), -Y.T]]
    res = linprog(c, A, np.r_[np.zeros(m), -yo], np.r_[0, np.ones(n)][None], [1], bounds=[(0,None)]*(n+1), **H)
    return res.x[0]
def cemin(o, XR):
    c = np.r_[W[o], np.zeros(n)]
    A = np.r_[np.c_[-np.eye(m), XR.T], np.c_[np.zeros((s,m)), -Y.T]]
    res = linprog(c, A, np.r_[np.zeros(m), -Y[o]], np.r_[np.zeros(m), np.ones(n)][None], [1], bounds=[(0,None)]*(m+n), **H)
    return res.fun / (W[o] @ X[o])
theta = r.TE_I_vrs.values; delta = 1/theta                       # Shephard input distance >= 1
refl = np.r_[delta, 2-delta]; sd = refl.std(ddof=1)
iqr = np.subtract(*np.percentile(refl, [75, 25]))
h = 0.9 * min(sd, iqr/1.34) * (2*n)**(-1/5)
sdd = delta.std(ddof=1)
front = X / delta[:,None]                                         # projected frontier inputs
def rep(b):
    rng = np.random.default_rng(1000 + b)
    bstar = rng.choice(refl, n)
    dd = bstar + h*rng.standard_normal(n)
    dd = bstar.mean() + (dd - bstar.mean())/np.sqrt(1 + h**2/sdd**2)
    dd = np.where(dd < 1, 2 - dd, dd)
    XR = front * dd[:,None]                                       # pseudo-sample inputs
    return np.array([te(X[o], Y[o], XR) for o in range(n)]), np.array([cemin(o, XR) for o in range(n)])
if __name__ == '__main__':
    with Pool(int(__import__("os").cpu_count())) as pool:
        out = []
        for i, o in enumerate(pool.imap(rep, range(B), chunksize=5), 1):
            out.append(o)
            if i % 50 == 0: print(f'{i}/{B} replications', flush=True)
    T = np.array([o[0] for o in out]); Cs = np.array([o[1] for o in out])  # B x n (Farrell scale)
    np.save(f'{D}/boot_TE.npy', T); np.save(f'{D}/boot_CE.npy', Cs)
    res = pd.DataFrame({'dmu': r.dmu})
    for name, est, bs in (('TE', theta, T), ('CE', r.CE_vrs.values, Cs)):
        d_hat, d_bs = 1/est, 1/bs
        bias = d_bs.mean(0) - d_hat
        d_bc = d_hat - bias
        lo, hi = np.percentile(d_bs - d_hat, [2.5, 97.5], axis=0)
        res[f'{name}_hat'] = est; res[f'{name}_bc'] = 1/d_bc
        res[f'{name}_lo'] = 1/(d_hat - lo); res[f'{name}_hi'] = 1/(d_hat - hi)
        res[f'{name}_bias_ok'] = bias**2 / (d_bs.var(0)) > 1/3   # SW rule: correct only if bias^2/var > 1/3
    res.to_csv(f'{D}/boot_scores.csv', index=False)
    print('h=', round(h,4), 'saved', T.shape)
