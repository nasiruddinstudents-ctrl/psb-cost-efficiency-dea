"""Bank-level (cluster) bootstrap of the ownership comparison WITH frontier re-estimation:
resample the 52 banks with replacement (all years together), re-estimate the pooled VRS cost frontier
on the resampled panel, and recompute PSB minus private mean CE by sub-period, for both the
balance-sheet-output and income-output models. Percentile 95% intervals. Means weight bank-years equally."""
import numpy as np, pandas as pd, sys
from scipy.optimize import linprog
from multiprocessing import Pool
D, B = sys.argv[1], int(sys.argv[2])
p = pd.read_csv(f'{D}/panel_B.csv')
p['nii_r'] = (p.interest_earned - p.interest_expended) * 100 / p.deflator
Xc, Wc = ['funds','employees','fixed_assets_r'], ['w_funds','w_labour','w_capital']
sx = p[Xc].mean().values; X = p[Xc].values/sx; W = p[Wc].values*sx
Y1 = (p[['advances_r','investments_r','other_income_r']] / p[['advances_r','investments_r','other_income_r']].mean()).values
Y2 = (p[['nii_r','other_income_r']] / p[['nii_r','other_income_r']].mean()).values
per = pd.cut(p.year, [2008,2015,2020,2023], labels=False).values; psb = p.psb.values
banks = p.bank.unique(); rows_of = {b: np.where(p.bank.values == b)[0] for b in banks}
def ce_all(idx, Y):
    Xr, Yr = X[idx], Y[idx]; k = len(idx); out = np.empty(k); m = 3; s = Y.shape[1]
    for j in range(k):
        o = idx[j]; c = np.r_[W[o], np.zeros(k)]
        A = np.r_[np.c_[-np.eye(m), Xr.T], np.c_[np.zeros((s,m)), -Yr.T]]
        r = linprog(c, A, np.r_[np.zeros(m), -Y[o]], np.r_[np.zeros(m), np.ones(k)][None], [1], bounds=[(0,None)]*(m+k), method='highs')
        out[j] = r.fun / (W[o] @ X[o])
    return out
def gaps(idx, ce):
    g = []
    for q in range(3):
        msk = per[idx] == q; a = ce[msk & (psb[idx]==1)]; b = ce[msk & (psb[idx]==0)]
        g.append(a.mean() - b.mean())
    a = ce[psb[idx]==1]; b = ce[psb[idx]==0]; g.append(a.mean() - b.mean())
    return g
def one(bi):
    rng = np.random.default_rng(5000 + bi)
    idx = np.concatenate([rows_of[b] for b in rng.choice(banks, len(banks))])
    ok2 = idx[p.nii_r.values[idx] > 0]
    return gaps(idx, ce_all(idx, Y1)) + gaps(ok2, ce_all(ok2, Y2))
if __name__ == '__main__':
    full = np.arange(len(p)); ok = full[p.nii_r.values > 0]
    est = gaps(full, ce_all(full, Y1)) + gaps(ok, ce_all(ok, Y2))
    with Pool(2) as pool:
        res = []
        for i, r in enumerate(pool.imap(one, range(B)), 1):
            res.append(r)
            if i % 20 == 0: print(f'{i}/{B}', flush=True)
    R = np.array(res); lo, hi = np.percentile(R, [2.5, 97.5], axis=0)
    lab = [(m, q) for m in ['Balance-sheet outputs','Income outputs'] for q in ['2009–15','2016–20','2021–23','All years']]
    T = pd.DataFrame([dict(model=m, period=q, gap_PSB_minus_PVT=e, ci_lo=l, ci_hi=h, excludes_zero=(l>0) or (h<0)) for (m,q),e,l,h in zip(lab, est, lo, hi)])
    T.to_csv(f'{D}/table_bank_bootstrap_gaps.csv', index=False); print(T.round(4).to_string())
