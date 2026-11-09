import openpyxl, glob, re, sys, warnings, pandas as pd
warnings.filterwarnings('ignore')
RAW = sys.argv[1]; OUT = sys.argv[2]

def yr(v):
    if v is None: return None
    if hasattr(v, 'year') and hasattr(v, 'month'): return v.year
    s = str(v).strip()
    m = re.match(r'^(\d{4})\s*-\s*(\d{2,4})$', s)
    if m:
        e = m.group(2); return int(m.group(1)[:2] + e) if len(e) == 2 else int(e)
    m = re.match(r'^(\d{4})(\.0)?$', s)
    return int(m.group(1)) if m else None

def num(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().replace(',', '')
    if s in ('-', '', '--', 'NA', '..', '…'): return None
    try: return float(s)
    except: return None

def parse(path, sheet, cols, data_start, headers=False):
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)[sheet]
    out, y, group = [], None, None
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < data_start: continue
        r = list(r) + [None] * 50
        yy = yr(r[1])
        if yy: y = yy
        name = r[2]
        if name is None or y is None: continue
        name = re.sub(r'\s+', ' ', str(name)).strip().rstrip('*#@').strip()
        vals = {k: num(r[c]) for k, c in cols.items()}
        if headers and all(r[c] is None for c in cols.values()):
            group = name; continue           # bank-group header row (CRAR table only)
        out.append(dict(year=y, bank=name, group=group, **vals))
    return pd.DataFrame(out)

f = lambda pat: glob.glob(f'{RAW}/*{pat}*')[0]
A = parse(f('Liabilities_and_Assets'), 'ASSETS',
          dict(investments=8, advances=20, fixed_assets=33), 7)
L1 = parse(f('Liabilities_and_Assets'), 'LIABILITIES_Upto2019',
           dict(capital=3, reserves=4, deposits=11, borrowings=21, other_liab=28), 7)
L2 = parse(f('Liabilities_and_Assets'), 'LIABILITIES_2020_Onwards',
           dict(capital=3, reserves=4, deposits=10, borrowings=20, other_liab=27), 7)
E = parse(f('Earnings_and_Expenses'), 'EARNINGS',
          dict(interest_earned=7, other_income=14), 8)
X = parse(f('Earnings_and_Expenses'), 'EXPENSES',
          dict(interest_expended=6, staff_exp=7, depreciation=11, opex_total=19,
               provisions=21, operating_profit=22), 8)
N = parse(f('Non_Performing'), 'Report 1',
          dict(gnpa_open=3, gnpa_add=4, gnpa_red=5, gnpa_wo=6, gnpa=7, nnpa=9), 9)
R = parse(f('Select_Ratios'), 'Report 1',
          dict(cd_ratio=4, psl_share=9, cost_income_proxy=16, roa=23, roe=24, cost_of_funds=27, bus_per_emp=32, car=34, car_t1=35), 7)
P = parse(f('Employees'), 'Report 1', dict(officers=3, clerks=4, substaff=5), 4)
C = parse(f('Capital_Adequacy'), 'STRBI B3',
          dict(b1_tot=5, b2_tot=8, b3_t1=9, b3_t2=10), 8, headers=True)
for n, d in dict(A=A, L1=L1, L2=L2, E=E, X=X, N=N, R=R, P=P, C=C).items():
    d.to_csv(f'{OUT}/{n}.csv', index=False)
    print(n, d.shape, 'years', d.year.min(), d.year.max())
