import pandas as pd, numpy as np, json, sys
D = sys.argv[1]
r = pd.read_csv(f'{D}/dea_scores.csv').merge(pd.read_csv(f'{D}/boot_scores.csv')[['dmu','CE_bc','CE_lo','CE_hi','TE_bc']], on='dmu')
r['period'] = pd.cut(r.year,[2008,2015,2020,2023],labels=['2009–15','2016–20','2021–23'])
T1 = r.groupby(['period','ownership'],observed=True)[['TE_I_vrs','TE_II_vrs','AE_I_vrs','AE_II_vrs','CE_vrs','CE_bc','NPA_eff']].mean().round(3)
T1.loc[('All years','PSB'),:] = r[r.ownership=='PSB'][T1.columns].mean().round(3).values
T1.loc[('All years','PVT'),:] = r[r.ownership=='PVT'][T1.columns].mean().round(3).values
T1.to_csv(f'{D}/table1_summary.csv')
bank = r[r.ownership=='PSB'].groupby('bank').agg(years=('year', lambda x: f'{x.min()}–{x.max()}'), TE_I=('TE_I_vrs','mean'),
    TE_II=('TE_II_vrs','mean'), AE_I=('AE_I_vrs','mean'), AE_II=('AE_II_vrs','mean'), CE=('CE_vrs','mean'),
    CE_bc=('CE_bc','mean'), CE_lo=('CE_lo','mean'), CE_hi=('CE_hi','mean')).round(3).sort_values('CE', ascending=False)
bank.to_csv(f'{D}/tableA1_psb_banks.csv')
with pd.ExcelWriter(f'{D}/PSB_PVT_DEA_results_v2.xlsx') as w:
    for name, f in [('Readme',None),('T1_summary','table1_summary.csv'),('T2_TypeI_vs_II','table2_typeI_vs_II.csv'),
                    ('T3_Spearman','table3_spearman.csv'),('T4_PSB_vs_PVT','table_psb_vs_pvt_period.csv'),
                    ('T5_second_stage','table5_second_stage.csv'),('T6_separability','table_separability.csv'),
                    ('T6b_sep_splits','separability_splits.csv'),('T7_conditional','table_conditional.csv'),
                    ('T8_metafrontier','table_metafrontier.csv'),('T9_12_surviving','table_12_surviving.csv'),
                    ('A1_PSB_bank_means','tableA1_psb_banks.csv'),('Scores_all','dea_scores.csv'),
                    ('Bootstrap_scores','boot_scores.csv'),('Robustness_scores','robustness_scores.csv'),
                    ('Conditional_scores','conditional_metafrontier_scores.csv'),('Malmquist','malmquist_global.csv'),
                    ('Panel_data','panel_B.csv')]:
        if f is None:
            pd.DataFrame({'note':['Source: RBI DBIE, Statistical Tables Relating to Banks in India (Tables 1,2,3,6,11,12), downloaded Nov 8 2026.',
              'Sample: public + private sector banks 2009–2023 (March year-end), 646 bank-years, 52 banks; Bharatiya Mahila Bank excluded.',
              'Rupee values deflated by India GDP deflator (2011-12=100).','All numbers produced by scripts in /scripts (parse, build_panel_b, dea, bootstrap_dea, tests_typeI_II, separability, robustness, conditional, stage2).']}).to_excel(w,name,index=False)
        else:
            pd.read_csv(f'{D}/{f}').round(4).to_excel(w, name, index=False)
print(T1.to_string()); print(bank.to_string())
