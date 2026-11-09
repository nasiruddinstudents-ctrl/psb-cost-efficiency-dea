"""Option B panel: public + private sector banks, 2009-2023, real 2011-12 rupees."""
import pandas as pd, numpy as np, sys
D = sys.argv[1]
r = lambda n: pd.read_csv(f'{D}/{n}.csv').drop(columns='group', errors='ignore').drop_duplicates(['year','bank'])
C = pd.read_csv(f'{D}/C.csv')
g = C[['year','bank','group']].drop_duplicates(['year','bank'])
g['ownership'] = np.select([g.group.str.contains('PUBLIC|NATIONALISED|STATE BANK OF INDIA AND', na=False),
                            g.group.str.contains('PRIVATE', na=False)], ['PSB','PVT'], 'OTHER')
g = g[g.ownership != 'OTHER']
p = g
for d in [r('A'), pd.concat([r('L1'), r('L2')]).drop_duplicates(['year','bank']), r('E'), r('X'), r('N'), r('R'), r('P')]:
    p = p.merge(d, on=['year','bank'], how='left')
p = p[(p.year >= 2009) & (p.year <= 2023)].copy()
p['employees'] = p[['officers','clerks','substaff']].sum(axis=1, min_count=1)
p['borrowings'] = p.borrowings.fillna(0)          # RBI '-' = nil
# GDP deflator, base 2011-12=100, keyed by bank year (March Y = fiscal year Y-1/Y)
# FY2008-09..FY2020-21: World Bank NY.GDP.DEFL.ZS (India, fiscal-year basis);
# FY2021-22, FY2022-23: MoSPI PE 31 May 2023 nominal/real GDP (234.71/149.26, 272.41/160.06)
defl = {2009:77.74,2010:83.21,2011:91.97,2012:100.00,2013:107.93,2014:114.61,2015:118.43,
        2016:121.13,2017:125.05,2018:130.02,2019:134.87,2020:139.68,2021:146.13,
        2022:round(234.71/149.26*100,2),2023:round(272.41/160.06*100,2)}
p['deflator'] = p.year.map(defl)
f = 100 / p.deflator
p['funds'] = (p.deposits + p.borrowings) * f
p['fixed_assets_r'] = p.fixed_assets * f
p['advances_r'] = p.advances * f
p['investments_r'] = p.investments * f
p['other_income_r'] = p.other_income * f
p['gnpa_r'] = p.gnpa * f
p['other_opex'] = p.opex_total - p.staff_exp
p['w_funds'] = p.interest_expended / (p.deposits + p.borrowings)          # rate, unit-free
p['w_labour'] = p.staff_exp * f / p.employees                              # real Rs crore per employee
p['w_capital'] = p.other_opex / p.fixed_assets                             # ratio, unit-free
p['total_cost_r'] = (p.interest_expended + p.opex_total) * f
p['gnpa_ratio'] = p.gnpa / p.advances * 100
p['total_assets_r'] = (p.capital.fillna(0) + p.reserves.fillna(0) + p.deposits + p.borrowings + p.other_liab.fillna(0)) * f
p['ln_assets'] = np.log(p.total_assets_r)
# PSB acquirers after amalgamation (first full fiscal year post-merger)
acq = {'STATE BANK OF INDIA':2018, 'BANK OF BARODA':2020, 'PUNJAB NATIONAL BANK':2021,
       'CANARA BANK':2021, 'UNION BANK OF INDIA':2021, 'INDIAN BANK':2021}
p['merged_post'] = [int(b in acq and y >= acq[b]) for b, y in zip(p.bank, p.year)]
p['psb'] = (p.ownership == 'PSB').astype(int)
need = ['funds','employees','fixed_assets_r','advances_r','investments_r','other_income_r',
        'w_funds','w_labour','w_capital','gnpa_r']
ok = p[need].notna().all(axis=1) & (p[need[:-1]] > 0).all(axis=1)
print('dropped (incomplete/non-positive):')
print(p.loc[~ok, ['year','bank','ownership']].to_string(index=False))
p = p[ok & ~p.bank.str.contains('BHARATIYA MAHILA')].copy()   # start-up bank, 2014-17, incomplete data
p['dmu'] = p.bank + '_' + p.year.astype(str)
p.to_csv(f'{D}/panel_B.csv', index=False)
print('\nFinal:', len(p), 'bank-years |', p.bank.nunique(), 'banks')
print(p.groupby(['year','ownership']).size().unstack().to_string())
print('\nIDBI labels:', p[p.bank.str.contains('IDBI')][['year','ownership']].values.tolist())
