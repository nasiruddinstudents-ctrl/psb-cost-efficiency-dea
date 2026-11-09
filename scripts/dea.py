"""
Pooled-frontier cost DEA for Indian PSBs + private banks, 2009-2023 (real 2011-12 Rs).
Inputs  x: loanable funds (deposits+borrowings), employees, fixed assets
Prices  w: interest expended/funds, staff expense/employee, other opex/fixed assets
Outputs y: advances, investments, other (non-interest) income
Bad     b: gross NPAs
Models (reference set = ALL bank-years, i.e. one pooled/global frontier):
  TE_I   radial input-oriented technical efficiency (Farrell)
  CE     cost efficiency  = min_x w'x / w'x_o            (Fare-Grosskopf-Lovell)
  AE_I   = CE / TE_I                                     (Type I, radial decomposition)
  TE_II  = w'(theta*x_o - s*) / w'x_o, s* = max-slack Phase II at theta*  (slack-adjusted)
  AE_II  = CE / TE_II                                    (Type II, slack-adjusted)
  -> CE = TE_I*AE_I = TE_II*AE_II ;  CE <= TE_II <= TE_I
  DDF_NPA: directional distance beta, g=(-x,+y,-b), CRS, weak disposability of b
  Global Malmquist (Pastor & Lovell 2005, CRS input-oriented): M = TE_G(t+1)/TE_G(t) = EC x BPC
"""
import numpy as np, pandas as pd, sys
from scipy.optimize import linprog
D = sys.argv[1]
p = pd.read_csv(f'{D}/panel_B.csv').reset_index(drop=True)
Xc, Wc, Yc = ['funds','employees','fixed_assets_r'], ['w_funds','w_labour','w_capital'], ['advances_r','investments_r','other_income_r']
# deflate rupee price of labour consistently: w_labour already real per employee; w_funds, w_capital are ratios
sx = p[Xc].mean().values; sy = p[Yc].mean().values; sb = p.gnpa_r.mean()
X = p[Xc].values / sx; W = p[Wc].values * sx; Y = p[Yc].values / sy; B = p.gnpa_r.values / sb
n, m, s = len(p), X.shape[1], Y.shape[1]
assert np.allclose((X*W).sum(1)*1, (p[Xc].values*p[Wc].values).sum(1))
H = dict(method='highs')

def te_radial(o, R, vrs):
    k = len(R)  # vars: theta, lambda_R
    c = np.r_[1, np.zeros(k)]
    A = np.r_[np.c_[-X[o][:,None], X[R].T], np.c_[np.zeros((s,1)), -Y[R].T]]
    b = np.r_[np.zeros(m), -Y[o]]
    Aeq = np.r_[0, np.ones(k)][None] if vrs else None
    r = linprog(c, A, b, Aeq, [1] if vrs else None, bounds=[(0,None)]*(k+1), **H)
    return r.x[0]

def slacks(o, R, vrs, th):
    k = len(R)  # vars: lambda, s- (m), s+ (s); max sum s-/x_o + s+/y_o
    c = -np.r_[np.zeros(k), 1/X[o], 1/np.maximum(Y[o],1e-9)]
    Aeq = np.r_[np.c_[X[R].T, np.eye(m), np.zeros((m,s))], np.c_[Y[R].T, np.zeros((s,m)), -np.eye(s)]]
    beq = np.r_[th*X[o], Y[o]]
    if vrs: Aeq = np.r_[Aeq, np.r_[np.ones(k), np.zeros(m+s)][None]]; beq = np.r_[beq, 1]
    r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0,None)]*(k+m+s), **H)
    return r.x[k:k+m]

def cost_min(o, R, vrs):
    k = len(R)  # vars: xtilde (m), lambda
    c = np.r_[W[o], np.zeros(k)]
    A = np.r_[np.c_[-np.eye(m), X[R].T], np.c_[np.zeros((s,m)), -Y[R].T]]
    b = np.r_[np.zeros(m), -Y[o]]
    Aeq = np.r_[np.zeros(m), np.ones(k)][None] if vrs else None
    r = linprog(c, A, b, Aeq, [1] if vrs else None, bounds=[(0,None)]*(m+k), **H)
    return r.fun / (W[o] @ X[o])

def ddf_npa(o, R):
    k = len(R)  # vars: beta, lambda ; max beta
    c = np.r_[-1, np.zeros(k)]
    A = np.r_[np.c_[X[o][:,None], X[R].T], np.c_[Y[o][:,None], -Y[R].T]]
    b = np.r_[X[o], -Y[o]]
    Aeq = np.r_[B[o], B[R]][None]; beq = [B[o]]
    r = linprog(c, A, b, Aeq, beq, bounds=[(0,None)]*(k+1), **H)
    return r.x[0]

ALL = np.arange(n); res = []
for o in range(n):
    yr = p.year[o]; Rt = np.where(p.year == yr)[0]
    d = {}
    for tag, vrs in (('vrs', True), ('crs', False)):
        th = te_radial(o, ALL, vrs); sl = slacks(o, ALL, vrs, th); ce = cost_min(o, ALL, vrs)
        ce = min(ce, th)  # numerical guard
        te2 = W[o] @ (th*X[o] - sl) / (W[o] @ X[o])
        d.update({f'TE_I_{tag}': th, f'CE_{tag}': ce, f'AE_I_{tag}': ce/th,
                  f'TE_II_{tag}': te2, f'AE_II_{tag}': ce/te2,
                  f'slack_share_{tag}': 1 - te2/th})
    d['TE_contemp_crs'] = te_radial(o, Rt, False)
    d['CE_contemp_vrs'] = cost_min(o, Rt, True)
    d['DDF_beta_npa'] = ddf_npa(o, ALL)
    res.append(d)
    if o % 100 == 0: print('done', o, flush=True)
r = pd.concat([p[['dmu','year','bank','ownership']], pd.DataFrame(res)], axis=1)
r['NPA_eff'] = 1/(1+r.DDF_beta_npa)
r.to_csv(f'{D}/dea_scores.csv', index=False)

# Global Malmquist on consecutive bank-years
r = r.sort_values(['bank','year']); rows = []
for bk, g in r.groupby('bank'):
    g = g.set_index('year')
    for t in g.index:
        if t+1 in g.index:
            M = g.loc[t+1,'TE_I_crs'] / g.loc[t,'TE_I_crs']          # TE vs global (CRS) frontier
            EC = g.loc[t+1,'TE_contemp_crs'] / g.loc[t,'TE_contemp_crs']
            rows.append(dict(bank=bk, t=t, t1=t+1, ownership=g.loc[t+1,'ownership'], M_G=M, EC=EC, BPC=M/EC))
pd.DataFrame(rows).to_csv(f'{D}/malmquist_global.csv', index=False)
print('saved')
