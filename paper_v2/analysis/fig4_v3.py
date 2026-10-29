import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'font.family': 'DejaVu Sans'})
C = pd.read_csv('contrasts_v3.csv'); L = pd.read_csv('levels_v3.csv'); RS = pd.read_csv('readers_stats.csv')
T = pd.read_csv('data/p1all_test.csv.gz')
P = T.pivot_table(index=['exp', 'seed'], columns='model', values='dice', aggfunc='mean')
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw=dict(wspace=0.38, hspace=0.75))
EXP = {'E1': ('With skip connections (E1)', '#1f4e9c'), 'E2': ('Without skip connections (E2)', '#c0392b')}
MODS = ('hq', 'hc', 'hf', 'ht'); names = ['HQB-\nUNet', 'HQB-\ncls', 'HQB-\nfree', 'HQB-\ntrig']
a = ax[0, 0]; a.axhline(0, color='0.5', lw=0.6, ls='--')
for i, m in enumerate(('hc', 'hf', 'ht')):
    for j, exp in enumerate(('E1', 'E2')):
        x0 = i + (-0.18 if exp == 'E1' else 0.18); col = EXP[exp][1]
        ps = (P.loc[exp]['hq'] - P.loc[exp][m]).values
        a.plot([x0 - 0.07] * len(ps), ps, 'o', ms=3, color=col, alpha=0.55, mew=0, label=('single seeds' if (i == 0 and j == 0) else None))
        r = C[(C.set == 'test') & (C.metric == 'dice') & (C.group == exp) & (C.contrast == f'hq-{m}')].iloc[0]
        a.errorbar(x0 + 0.07, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt='D', ms=4, color=col, capsize=2, lw=0.8, label=(EXP[exp][0] if i == 0 else None))
a.set_xticks(range(3)); a.set_xticklabels(['HQB-UNet\n− HQB-cls', 'HQB-UNet\n− HQB-free', 'HQB-UNet\n− HQB-trig'], fontsize=7.5)
a.set_ylabel('Difference in per-image Dice\n(test set, proxy masks)'); a.set_xlim(-0.5, 2.5); a.set_title('Circuit contrasts', fontsize=8.5)
a.legend(fontsize=6.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, handletextpad=0.2, columnspacing=0.8)
b = ax[0, 1]
for i, m in enumerate(MODS):
    for exp in ('E1', 'E2'):
        col = EXP[exp][1]; x0 = i + (-0.15 if exp == 'E1' else 0.15)
        for met, mk, dx in (('drop_const', 'x', -0.06), ('drop_swapavg', 'D', 0.06)):
            r = L[(L.set == 'test') & (L.metric == met) & (L.group == exp) & (L.model == m)].iloc[0]
            b.errorbar(x0 + dx, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt=mk, ms=4, color=col, mfc=('white' if mk == 'D' else col), capsize=1.5, lw=0.7)
b.plot([], [], 'x', color='0.3', label='constant code'); b.plot([], [], 'D', mfc='white', color='0.3', ms=4, label='random training codes (mean of 5)')
b.set_xticks(range(4)); b.set_xticklabels(names, fontsize=7.5); b.set_ylabel('Mean drop in per-image Dice\n(test set)')
b.set_xlim(-0.5, 3.5); b.set_ylim(-0.03, 0.95); b.set_title('Code replaced at test time', fontsize=8.5)
b.legend(fontsize=6.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, handletextpad=0.2, columnspacing=0.8)
def refpanel(c, getter, rule, ylab, title):
    c.axhline(rule, color='0.3', lw=0.8, ls=':'); c.text(3.45, rule + 0.015, 'labelling rule', ha='right', fontsize=6.5, color='0.3')
    for i, m in enumerate(MODS):
        for exp in ('E1', 'E2'):
            r = getter(exp, m); col = EXP[exp][1]; x0 = i + (-0.12 if exp == 'E1' else 0.12)
            c.errorbar(x0, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt='o', ms=4, color=col, capsize=2, lw=0.8)
    c.set_xticks(range(4)); c.set_xticklabels(names, fontsize=7.5); c.set_ylabel(ylab); c.set_xlim(-0.5, 3.5); c.set_title(title, fontsize=8.5)
refpanel(ax[1, 0], lambda e, m: RS[(RS.analysis == 'model vs reader1') & (RS.metric == 'full') & (RS.exp == e) & (RS.model == m)].iloc[0],
         RS[(RS.analysis == 'rule vs reader1') & (RS.metric == 'full')].est.iloc[0], 'Per-image Dice against\nmanual outlines (test set)', 'Manual reference (84 test images)')
ax[1, 0].set_ylim(0.6, 1.0)
refpanel(ax[1, 1], lambda e, m: L[(L.set == 'mias') & (L.metric == 'dice_expert') & (L.group == e) & (L.model == m)].iloc[0],
         0.8653, 'Per-image Dice against\nexpert masks (mini-MIAS)', 'External expert reference')
ax[1, 1].set_ylim(0, 1)
for k, x in zip('abcd', ax.flat):
    x.tick_params(labelsize=7.5); x.text(-0.3, 1.06, k, transform=x.transAxes, fontsize=11, fontweight='bold')
import os; os.makedirs('figs', exist_ok=True); out = 'figs/fig4'
fig.savefig(out + '.png', dpi=600, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); fig.savefig(out + '.eps', bbox_inches='tight')
