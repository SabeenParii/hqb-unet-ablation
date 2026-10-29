# Paper 1 v2 (E1 with skips, seeds 0-4; E2 without skips, seeds 0-2): prespecified analysis.
# Crossed hierarchical bootstrap over clusters (test: patients; DMID, mini-MIAS: images) and seeds, 10,000 replicates, rng(0).
import numpy as np, pandas as pd, os, json
D = 'data/v2'; OUT = 'stats'; os.makedirs(OUT, exist_ok=True); B = 10000
T = pd.read_csv(f'{D}/p1v2_test.csv.gz'); E = pd.read_csv(f'{D}/p1v2_dmid.csv.gz'); M = pd.read_csv(f'{D}/p1v2_mias.csv.gz')
T['cluster'] = T.patient_id.astype(str); E['cluster'] = E.file_id.astype(str); M['cluster'] = M.file_id.astype(str)
assert M.groupby('run').file_id.nunique().eq(322).all() and not M.duplicated(['run', 'file_id']).any()
for X in (T, E):
    sw = [c for c in X.columns if c.startswith('drop_swap') and c[-1].isdigit()]
    X['drop_swapavg'] = X[sw].mean(axis=1)
    if 'dice_check_switch' in X: assert (X.dice_check_switch - X.dice).abs().max() < 1e-3

def boot_stat(u, rng):
    seeds = np.sort(u.seed.unique()); cl = np.sort(u.cluster.unique())
    ci = {c: i for i, c in enumerate(cl)}; si = {s: i for i, s in enumerate(seeds)}
    S = np.zeros((len(cl), len(seeds))); N = np.zeros((len(cl), len(seeds)))
    for c, s, v in zip(u.cluster, u.seed, u.v):
        S[ci[c], si[s]] += v; N[ci[c], si[s]] += 1
    est = S.sum() / N.sum()
    wc = rng.multinomial(len(cl), np.ones(len(cl)) / len(cl), size=B)
    ws = rng.multinomial(len(seeds), np.ones(len(seeds)) / len(seeds), size=B)
    bs = np.einsum('bc,cs,bs->b', wc, S, ws) / np.einsum('bc,cs,bs->b', wc, N, ws)
    p = max(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())), 1.0 / B)
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj

def contrast(X, exp, a, b, col):
    t = X[X.exp == exp]
    x = t[t.model == a][['file_id', 'cluster', 'seed', col]].rename(columns={col: 'a'})
    y = t[t.model == b][['file_id', 'cluster', 'seed', col]].rename(columns={col: 'b'})
    m = x.merge(y, on=['file_id', 'cluster', 'seed']); assert len(m) == len(x) == len(y)
    m['v'] = m.a - m.b
    return boot_stat(m, np.random.default_rng(0))

def level(X, exp, mod, col):
    t = X[(X.exp == exp) & (X.model == mod)][['cluster', 'seed', col]].rename(columns={col: 'v'})
    return boot_stat(t.dropna(), np.random.default_rng(0))

SETS = {'test': (T, ['dice']), 'dmid': (E, ['dice']), 'mias': (M, ['dice_expert', 'dice_expert_breast', 'dice_proxy'])}
rows = []
for sname, (X, cols) in SETS.items():
    for col in cols:
        for exp in ('E1', 'E2'):
            for a, b in (('hq', 'hc'), ('hq', 'hf'), ('hc', 'hf')):
                e, lo, hi, p = contrast(X, exp, a, b, col)
                rows.append(dict(set=sname, metric=col, exp=exp, contrast=f'{a}-{b}', est=e, lo=lo, hi=hi, p=p))
C = pd.DataFrame(rows)
pri = (C.set == 'test') & (C.contrast != 'hc-hf')
C['holm_primary'] = np.nan; C.loc[pri, 'holm_primary'] = holm(C.loc[pri, 'p'])
C['resolved'] = (C.lo > 0) | (C.hi < 0)
C.to_csv(f'{OUT}/contrasts.csv', index=False)

lv = []
for sname, (X, cols) in SETS.items():
    extra = [c for c in ('drop_const', 'drop_swapavg', 'dice_shots100', 'dice_shots1000', 'dice_shots10000') if c in X]
    for col in cols + extra:
        for exp in ('E1', 'E2'):
            for mod in ('hq', 'hc', 'hf'):
                if X[(X.exp == exp) & (X.model == mod)][col].notna().sum() == 0: continue
                e, lo, hi, _ = level(X, exp, mod, col)
                lv.append(dict(set=sname, metric=col, exp=exp, model=mod, est=e, lo=lo, hi=hi))
L = pd.DataFrame(lv); L.to_csv(f'{OUT}/levels.csv', index=False)

# shot noise: Dice(shots) - exact Dice, HQB-UNet only
sh = []
for sname, (X, _) in (('test', SETS['test']), ('dmid', SETS['dmid'])):
    for n in (100, 1000, 10000):
        c = f'dice_shots{n}'
        if c not in X: continue
        for exp in ('E1', 'E2'):
            t = X[(X.exp == exp) & (X.model == 'hq')].copy(); t['v'] = t[c] - t.dice
            e, lo, hi, p = boot_stat(t, np.random.default_rng(0))
            sh.append(dict(set=sname, shots=n, exp=exp, est=e, lo=lo, hi=hi, p=p))
pd.DataFrame(sh).to_csv(f'{OUT}/shots.csv', index=False)

# per-run summary
R = []
for sname, (X, cols) in SETS.items():
    g = X.groupby(['exp', 'model', 'seed'])[cols + [c for c in ('drop_const', 'drop_swapavg') if c in X]].mean()
    g.columns = [f'{sname}_{c}' for c in g.columns]; R.append(g)
pd.concat(R, axis=1).reset_index().to_csv(f'{OUT}/per_run.csv', index=False)
pd.set_option('display.width', 200)
print(C.round(4).to_string()); print(L.round(4).to_string()); print(pd.DataFrame(sh).round(4).to_string())
