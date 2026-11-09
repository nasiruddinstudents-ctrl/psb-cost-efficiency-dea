#!/usr/bin/env bash
# Reproduces every table and figure in the paper. Usage: bash run_all.sh [--quick]
# --quick uses 20 bootstrap replications (minutes) instead of the paper's settings (several hours).
set -euo pipefail
cd "$(dirname "$0")"
OUT=results; mkdir -p $OUT
B_SCORES=200; B_STAGE2=2000; SPLITS=20; K=10
if [[ "${1:-}" == "--quick" ]]; then B_SCORES=20; B_STAGE2=50; SPLITS=3; K=3; fi
python3 scripts/parse.py data/raw $OUT                 # 1. read the six RBI tables
python3 scripts/build_panel_b.py $OUT                  # 2. public + private panel, real 2011-12 prices
python3 scripts/dea.py $OUT                            # 3. pooled cost DEA, Type I/II, NPA DDF, global Malmquist
python3 scripts/tests_typeI_II.py $OUT                 # 4. Type I vs II tests, Spearman matrix, ownership by period
python3 scripts/robustness.py $OUT                     # 5. PSB-only frontier, outliers, 12 surviving PSBs
python3 scripts/robust_income_outputs.py $OUT          # 6. income-based outputs
python3 scripts/bootstrap_dea.py $OUT $B_SCORES        # 7. Simar-Wilson (1998) bootstrap
python3 scripts/separability.py $OUT $SPLITS $K        # 8. Daraio-Simar-Wilson (2018) separability test
python3 scripts/conditional.py $OUT                    # 9. conditional efficiency + metafrontier
python3 scripts/stage2.py $OUT $B_STAGE2               # 10. second stage (M1-M4)
python3 scripts/tables.py $OUT                         # 11. summary tables + results workbook
python3 scripts/make_figure.py $OUT                    # 12. Figure 1
echo "Done. Results in $OUT/"
