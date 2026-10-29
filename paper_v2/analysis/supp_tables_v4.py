import pandas as pd, numpy as np, json
C = pd.read_csv('contrasts_v4.csv'); L = pd.read_csv('levels_v4.csv'); SH = pd.read_csv('shots_v4.csv'); RS = pd.read_csv('readers_stats.csv')
R = pd.read_csv('data/p1all_runs.csv.gz'); T = pd.read_csv('data/p1all_test.csv.gz')
E = pd.read_csv('data/p1all_dmid.csv.gz'); M = pd.read_csv('data/p1all_mias.csv.gz').drop_duplicates(['run', 'file_id'])
NAME = {'hq': 'HQB-UNet', 'hc': 'HQB-cls', 'hf': 'HQB-free', 'ht': 'HQB-trig'}
def f4(v):
    if abs(v) < 0.00005: return '0.0000' if abs(v) < 0.000005 else ('−' if v < 0 else '') + f'{abs(v):.5f}'
    return ('−' if v < 0 else '') + f'{abs(v):.4f}'
ci = lambda r: f'{f4(r.est)} ({f4(r.lo)}, {f4(r.hi)})'
fp = lambda p: '0.0001' if p <= 0.0001 else (f'{p:.4f}' if p < 0.01 else f'{p:.2f}')
out = {}
# S11 runs
for X, k in ((T, 'test'), (E, 'dmid')):
    sw = [c for c in X.columns if c.startswith('drop_swap') and c[-1].isdigit()]; X['drop_swapavg'] = X[sw].mean(axis=1)
g = T.groupby('run')[['dice', 'drop_const', 'drop_swapavg']].mean(); gd = E.groupby('run').dice.mean(); gm = M.groupby('run').dice_expert.mean()
rows = ['| Experiment | Model | Seed | Epochs run | Tested epoch | Final learning rate | Test Dice | DMID Dice | Mini-MIAS Dice (expert) | Constant-code drop (test) | Swap drop (test) |', '|---|---|---|---|---|---|---|---|---|---|---|']
R['mo'] = R.model.map({'hq': 0, 'hc': 1, 'hf': 2, 'ht': 3})
for _, r in R.sort_values(['exp', 'mo', 'seed']).iterrows():
    lr = float(r.final_lr); lrs = {1.0: '10^−4^', 0.5: '5 × 10^−5^', 0.25: '2.5 × 10^−5^'}[round(lr / 1e-4, 2)]
    tag = '' if (r.model != 'ht' and not (r.exp == 'E2' and r.seed >= 3)) else '†'
    rows.append(f'| {r.exp} | {NAME[r.model]}{tag} | {r.seed} | {r.epochs_run} | {r.best_epoch} | {lrs} | {g.loc[r.run, "dice"]:.4f} | {gd.loc[r.run]:.4f} | {gm.loc[r.run]:.4f} | {f4(g.loc[r.run, "drop_const"])} | {f4(g.loc[r.run, "drop_swapavg"])} |')
out['S11'] = '\n'.join(rows)
# S12 severing
rows = ['| Set | Experiment | Model | Constant code | Random training codes (mean of 5) | Images with absolute constant-code change > 0.05 |', '|---|---|---|---|---|---|']
for s, sn in (('test', 'Test'), ('dmid', 'DMID')):
    for g_ in ('E1', 'E2'):
        for m in ('hq', 'hc', 'hf', 'ht'):
            a = L[(L.set == s) & (L.metric == 'drop_const') & (L.group == g_) & (L.model == m)].iloc[0]
            b = L[(L.set == s) & (L.metric == 'drop_swapavg') & (L.group == g_) & (L.model == m)].iloc[0]
            rows.append(f'| {sn} | {g_} | {NAME[m]} | {ci(a)} | {ci(b)} | {100 * a.share_abs_gt05:.1f}% |')
out['S12'] = '\n'.join(rows)
# S13 external
rows = ['| Set and reference | Experiment | HQB-UNet | HQB-cls | HQB-free | HQB-trig | HQB-UNet − HQB-cls | HQB-UNet − HQB-free | HQB-UNet − HQB-trig |', '|---|---|---|---|---|---|---|---|---|']
for s, met, lab in (('dmid', 'dice', 'DMID (510), rule masks'), ('mias', 'dice_expert', 'Mini-MIAS (322), expert breast + pectoral'),
                    ('mias', 'dice_expert_breast', 'Mini-MIAS (322), expert breast only'), ('mias', 'dice_proxy', 'Mini-MIAS (322), rule masks')):
    for g_ in ('E1', 'E2'):
        lv = [ci(L[(L.set == s) & (L.metric == met) & (L.group == g_) & (L.model == m)].iloc[0]) for m in ('hq', 'hc', 'hf', 'ht')]
        cc = []
        for b in ('hc', 'hf', 'ht'):
            r = C[(C.set == s) & (C.metric == met) & (C.group == g_) & (C.contrast == f'hq-{b}')].iloc[0]; cc.append(f'{ci(r)}; *P* = {fp(r.p)}')
        rows.append(f'| {lab} | {g_} | ' + ' | '.join(lv + cc) + ' |')
out['S13'] = '\n'.join(rows)
# S14 shots
rows = ['| Set | Experiment | Seeds | 100 shots | 1,000 shots | 10,000 shots |', '|---|---|---|---|---|---|']
for s, sn in (('test', 'Test'), ('dmid', 'DMID')):
    for g_, sd in (('E1', '0–4'), ('E2s3', '0–2'), ('E2', '0–4')):
        cells = [ci(SH[(SH.set == s) & (SH.group == g_) & (SH.shots == n)].iloc[0]) for n in (100, 1000, 10000)]
        rows.append(f'| {sn} | {g_[:2]} | {sd} | ' + ' | '.join(cells) + ' |')
out['S14'] = '\n'.join(rows)
# S15 manual reference
q = lambda **k: RS[np.logical_and.reduce([RS[a] == b for a, b in k.items()])].iloc[0]
rows = ['| Reference | Experiment | HQB-UNet | HQB-cls | HQB-free | HQB-trig | HQB-UNet − HQB-cls | HQB-UNet − HQB-free | HQB-UNet − HQB-trig |', '|---|---|---|---|---|---|---|---|---|']
for met, lab in (('full', 'Reader 1, breast + pectoral (84)'), ('breast', 'Reader 1, breast only (84)')):
    for e in ('E1', 'E2'):
        lv = [ci(q(analysis='model vs reader1', metric=met, exp=e, model=m)) for m in ('hq', 'hc', 'hf', 'ht')]
        cc = [f"{ci(q(analysis='contrast vs reader1', metric=met, exp=e, model='hq-' + b))}; *P* = {fp(q(analysis='contrast vs reader1', metric=met, exp=e, model='hq-' + b).p)}" for b in ('hc', 'hf', 'ht')]
        rows.append(f'| {lab} | {e} | ' + ' | '.join(lv + cc) + ' |')
for rd, lab in (('reader1', 'Reader 1, 20-image subset'), ('reader2', 'Reader 2, 20-image subset')):
    for e in ('E1', 'E2'):
        lv = [ci(q(analysis='model vs reader, subset of 20', metric=rd, exp=e, model=m)) for m in ('hq', 'hc', 'hf', 'ht')]
        rows.append(f'| {lab} | {e} | ' + ' | '.join(lv) + ' | – | – | – |')
mr = []
for met in ('full', 'breast'):
    for e in ('E1', 'E2'):
        mr.append(f"{e} {'breast + pectoral' if met == 'full' else 'breast only'}: " + ', '.join(f"{NAME[m]} {ci(q(analysis='model - rule, vs reader1', metric=met, exp=e, model=m))}" for m in ('hq', 'hc', 'hf', 'ht')))
out['S15'] = '\n'.join(rows); out['S15_rule'] = '; '.join(mr)
json.dump(out, open('supp_tables_v4.json', 'w'), ensure_ascii=False, indent=1)
for k, v in out.items(): print('==', k); print(v[:1500])
