# Volume versus Value: Cost Efficiency of Indian Public Sector Banks on a Common Frontier with Private Banks, 2009–2023

Replication code and data for the paper of the same title (Mohammad Nasir Uddin, Westcliff University).

The code estimates the cost efficiency of all Indian public sector banks (PSBs) and private sector banks,
2009–2023 (646 bank-years, 52 banks), against one pooled DEA cost frontier in 2011–12 prices, and reproduces
every table and figure in the paper.

## Data
`data/raw/` holds the six bank-wise tables from the Reserve Bank of India, *Statistical Tables Relating to Banks
in India*, downloaded unedited from the RBI Database on Indian Economy (https://data.rbi.org.in, Publications →
Time-Series Publications) on 8 November 2026:

| File | RBI table | Used for |
|---|---|---|
| T01 | Liabilities and Assets of Scheduled Commercial Banks | deposits, borrowings, advances, investments, fixed assets |
| T02 | Earnings and Expenses of Scheduled Commercial Banks | interest expended/earned, staff and operating expenses, other income |
| T03 | Bank-wise Capital Adequacy Ratios (CRAR) | RBI bank-group labels (ownership) |
| T06 | Movement of Non-Performing Assets | gross and net NPAs |
| T11 | Select Ratios of Scheduled Commercial Banks | CRAR, priority-sector share |
| T12 | Bank-wise and Category-wise Employees | employees |

Rupee values are deflated with India's implicit GDP deflator (base 2011–12 = 100): World Bank WDI
(NY.GDP.DEFL.ZS) to 2020–21, extended with MoSPI nominal/real GDP for 2021–22 and 2022–23. The values are
listed in `scripts/build_panel_b.py`.

## How to run
```bash
pip install -r requirements.txt
bash run_all.sh            # paper settings: 2,000 bootstrap replications for scores and for the second stage (about 2 hours on 2 cores; faster with more)
bash run_all.sh --quick    # fast check (~3 minutes); DEA scores identical, bootstrap results approximate
```
Output is written to `results/`. The results reported in the paper are in `results_paper/`
(including `PSB_PVT_DEA_results_v2.xlsx`, which collects all tables in one workbook).

## Pipeline
| Step | Script | Paper |
|---|---|---|
| 1 | `parse.py` – read the RBI tables | Section 4 |
| 2 | `build_panel_b.py` – public + private panel, deflation, input prices | Tables 2–3 |
| 3 | `dea.py` – pooled cost DEA, Type I/II decomposition, NPA directional distance function, global Malmquist | Tables 4, 9 |
| 4 | `tests_typeI_II.py` – Type I vs II tests, Spearman matrix, ownership by period | Tables 5–7 |
| 5 | `robustness.py` – PSB-only frontier, outliers, 12 surviving PSBs | Section 6.9, Table 8 |
| 6 | `robust_income_outputs.py` – income-based outputs | Section 6.4, Figure 1b |
| 7 | `bootstrap_dea.py` – Simar & Wilson (1998) bootstrap | Table 4 |
| 8 | `separability.py` – Daraio, Simar & Wilson (2018) test, bank-year splits | Table 10 (right) |
| 8b | `separability_bankblocked.py` – same test, halves formed by banks (main) | Table 10 (left) |
| 9 | `conditional.py` – conditional efficiency, cost metafrontier | Tables 11–12 |
| 10 | `stage2.py` – second stage, models M1–M4 | Table 13 |
| 10b | `audit_checks.py` – sequential (no look-ahead) frontier, net-NPA DDF, bank-level Type I/II tests, full vs exclusion sample, cost reconciliation | Tables 7, Sections 6.2, 6.9 |
| 10c | `stage2_bankboot.py` – bank-level (cluster) bootstrap of M1–M2 | Table 14 |
| 10d | `bank_bootstrap_frontier.py` – bank-level bootstrap of PSB − private gaps with the frontier re-estimated in each resample | Table 7, Section 6.4 |
| 10e | `audit_checks2.py` – Type I/II identities, capital-price winsorising, DDF direction, conditional bandwidths, leave-one-bank-out metafrontier, sample accounting | Sections 6.7, 6.9; Table A2 |
| 11 | `tables.py` – summary tables and workbook | Table 4, A1 |
| 12 | `make_figure.py` | Figure 1 |

Linear programmes are solved with SciPy's HiGHS solver. Random seeds are fixed in each script.

Tested environment: Python 3.13.18, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, openpyxl 3.1.5, matplotlib 3.11.2. Means in the paper weight bank-years equally unless stated; Mann–Whitney tests treat bank-years as independent and are descriptive, while bank-level bootstrap intervals (10c, 10d) treat banks as the sampling unit.

## Data checks
- Type I/II identities: CE = TE × AE holds under both decompositions (max error 3e-16) and CE ≤ TE(II) ≤ TE(I) for all 646 bank-years (`audit_checks2.py`).
- Cost reconciliation: for every bank-year, Σ (input quantity × input price) equals interest expended + operating expenses (checked in `dea.py` and `audit_checks.py`).
- Bank names are taken from the RBI tables; RBI footnote markers (*, #, @) on merged or reclassified banks are stripped in `parse.py`.
- Ownership is RBI's bank-group label in each year (IDBI Bank: PSB to 2018, private from 2019). Bharatiya Mahila Bank is excluded (incomplete data); bank-years with missing or non-positive inputs/outputs are dropped and listed by `build_panel_b.py`.

## Citation
See `CITATION.cff`. Licence: MIT (code). The RBI data remain subject to the RBI's terms of use.
