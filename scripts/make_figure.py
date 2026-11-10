"""Figure 1: mean cost efficiency by ownership, intermediation vs income-based outputs."""
import sys, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1]
r = pd.read_csv(f'{D}/dea_scores.csv').merge(pd.read_csv(f'{D}/robust_income_outputs.csv'), on='dmu')
C = {'PSB':'#2a78d6','PVT':'#eb6834'}; M = {'PSB':'o','PVT':'s'}; N = {'PSB':'Public sector banks','PVT':'Private sector banks'}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)
for a, col, tt in [(ax[0],'CE_vrs','(a) Balance-sheet-output model'), (ax[1],'CE_income_outputs','(b) Income-output model')]:
    m = r.groupby(['year','ownership'])[col].mean().unstack()
    for o in ['PSB','PVT']: a.plot(m.index, m[o], color=C[o], lw=2, marker=M[o], ms=5, label=N[o])
    a.axvspan(2015.5, 2020.5, color='#e9e8e4', zorder=0)
    a.text(2018, 0.97, 'AQR / IBC / mergers', transform=a.get_xaxis_transform(), ha='center', va='top', fontsize=8, color='#5f5e5a')
    a.set_title(tt, fontsize=10, loc='left'); a.set_xlabel('Fiscal year ending March', fontsize=9); a.tick_params(labelsize=8)
    a.grid(axis='y', color='#dddcd6', lw=0.6); [a.spines[s].set_visible(False) for s in ['top','right']]; a.set_xticks(range(2009, 2024, 2))
ax[0].set_ylabel('Mean cost efficiency', fontsize=9); ax[0].legend(frameon=False, fontsize=8, loc='lower left')
plt.tight_layout(); plt.savefig(f'{D}/figure1.png')
