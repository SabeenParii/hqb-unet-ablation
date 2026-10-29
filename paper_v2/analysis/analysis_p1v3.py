# Paper 1 v3: same analysis as analysis_p1v2.py (crossed hierarchical bootstrap over clusters and seeds, 10,000 replicates, rng(0)),
# extended to HQB-trig (ht) in E1 and E2 and to E2 seeds 0-4. Prespecified v3 family (Holm): E1 hq-ht, E2(0-4) hq-ht, hq-hc, hq-hf.
import numpy as np, pandas as pd, sys
src = open(__import__('os').path.join(__import__('os').path.dirname(__file__) or '.', 'analysis_p1v2.py')).read()
ns = {'B': 10000}; exec(src.split("D = '")[0], ns)
exec('def _f():\n    pass', ns)
exec(src[src.index('def boot_stat'):src.index('def contrast')], ns)
boot_stat, holm = ns['boot_stat'], ns['holm']
A = 'data'; OUT = '.'
T = pd.read_csv(f'{A}/p1all_test.csv.gz'); E = pd.read_csv(f'{A}/p1all_dmid.csv.gz')
M = pd.read_csv(f'{A}/p1all_mias.csv.gz').drop_duplicates(['run', 'file_id'], keep='first')   # per-run MIAS files hold each image twice (resumed evaluation)
T['cluster'] = T.patient_id.astype(str); E['cluster'] = E.file_id.astype(str); M['cluster'] = M.file_id.astype(str)
assert M.groupby('run').file_id.nunique().eq(322).all() and M.groupby('run').size().eq(322).all()
for X in (T, E):
    sw = [c for c in X.columns if c.startswith('drop_swap') and c[-1].isdigit()]
    X['drop_swapavg'] = X[sw].mean(axis=1)
    if 'dice_check_switch' in X: assert (X.dice_check_switch - X.dice).abs().max() < 1e-3
def sub(X, exp, seeds):
    return X[(X.exp == exp) & X.seed.isin(seeds)]
def contrast(X, a, b, col):
    x = X[X.model == a][['file_id', 'cluster', 'seed', col]].rename(columns={col: 'a'})
    y = X[X.model == b][['file_id', 'cluster', 'seed', col]].rename(columns={col: 'b'})
    m = x.merge(y, on=['file_id', 'cluster', 'seed']); assert len(m) == len(x) == len(y)
    m['v'] = m.a - m.b
    return boot_stat(m, np.random.default_rng(0))
def level(X, mod, col):
    t = X[X.model == mod][['cluster', 'seed', col]].rename(columns={col: 'v'})
    return boot_stat(t.dropna(), np.random.default_rng(0))
GROUPS = {'E1': ('E1', range(5)), 'E2s3': ('E2', range(3)), 'E2': ('E2', range(5))}
MODS = {'E1': ('hq', 'hc', 'hf', 'ht'), 'E2s3': ('hq', 'hc', 'hf'), 'E2': ('hq', 'hc', 'hf', 'ht')}
SETS = {'test': (T, ['dice']), 'dmid': (E, ['dice']), 'mias': (M, ['dice_expert', 'dice_expert_breast', 'dice_proxy'])}
PAIRS = (('hq', 'hc'), ('hq', 'hf'), ('hq', 'ht'), ('ht', 'hc'), ('ht', 'hf'), ('hc', 'hf'))
if __name__ == '__main__':
    rows = []
    for sname, (X, cols) in SETS.items():
        for col in cols:
            for g, (exp, seeds) in GROUPS.items():
                Y = sub(X, exp, seeds)
                for a, b in PAIRS:
                    if a not in MODS[g] or b not in MODS[g]: continue
                    e, lo, hi, p = contrast(Y, a, b, col)
                    rows.append(dict(set=sname, metric=col, group=g, contrast=f'{a}-{b}', est=e, lo=lo, hi=hi, p=p))
    C = pd.DataFrame(rows)
    t = C[(C.set == 'test')]
    v2 = [(('E1', 'hq-hc')), ('E1', 'hq-hf'), ('E2s3', 'hq-hc'), ('E2s3', 'hq-hf')]
    v3 = [('E1', 'hq-ht'), ('E2', 'hq-ht'), ('E2', 'hq-hc'), ('E2', 'hq-hf')]
    for name, fam in (('holm_v2', v2), ('holm_v3', v3)):
        idx = [t[(t.group == g) & (t.contrast == c)].index[0] for g, c in fam]
        C[name] = np.nan; C.loc[idx, name] = holm(C.loc[idx, 'p'].values)
    C['resolved'] = (C.lo > 0) | (C.hi < 0)
    C.to_csv(f'{OUT}/contrasts_v3.csv', index=False)
    lv = []
    for sname, (X, cols) in SETS.items():
        extra = [c for c in ('drop_const', 'drop_swapavg', 'dice_shots100', 'dice_shots1000', 'dice_shots10000') if c in X]
        for col in cols + extra:
            for g, (exp, seeds) in GROUPS.items():
                Y = sub(X, exp, seeds)
                for mod in MODS[g]:
                    if Y[Y.model == mod][col].notna().sum() == 0: continue
                    e, lo, hi, _ = level(Y, mod, col)
                    lv.append(dict(set=sname, metric=col, group=g, model=mod, est=e, lo=lo, hi=hi,
                                   share_abs_gt05=float((Y[Y.model == mod][col].abs() > 0.05).mean()) if col.startswith('drop') else np.nan))
    L = pd.DataFrame(lv); L.to_csv(f'{OUT}/levels_v3.csv', index=False)
    sh = []
    for sname, X in (('test', T), ('dmid', E)):
        for n in (100, 1000, 10000):
            c = f'dice_shots{n}'
            if c not in X: continue
            for g, (exp, seeds) in GROUPS.items():
                Y = sub(X, exp, seeds); Y = Y[Y.model == 'hq'].copy(); Y['v'] = Y[c] - Y.dice
                if Y.v.notna().sum() == 0: continue
                e, lo, hi, p = boot_stat(Y.dropna(subset=['v']), np.random.default_rng(0))
                sh.append(dict(set=sname, shots=n, group=g, est=e, lo=lo, hi=hi, p=p))
    pd.DataFrame(sh).to_csv(f'{OUT}/shots_v3.csv', index=False)
    pd.set_option('display.width', 220)
    print(C[C.set == 'test'].round(4).to_string()); print(pd.DataFrame(sh).round(4).to_string())
