# Reproduces the external-set contrasts of the manuscript (DMID and mini-MIAS; Supplementary Table S13) from the public
# per-image, per-seed scores in this folder, with the crossed hierarchical bootstrap of the paper (10,000 replicates, rng(0)).
# Usage (from this folder):  python reproduce_external.py
import numpy as np, pandas as pd

B = 10000


def boot(u, rng):
    seeds = np.sort(u.seed.unique()); cl = np.sort(u.cluster.unique())
    ci = {c: i for i, c in enumerate(cl)}; si = {s: i for i, s in enumerate(seeds)}
    S = np.zeros((len(cl), len(seeds))); N = np.zeros((len(cl), len(seeds)))
    for c, s, v in zip(u.cluster, u.seed, u.v):
        S[ci[c], si[s]] += v; N[ci[c], si[s]] += 1
    wc = rng.multinomial(len(cl), np.ones(len(cl)) / len(cl), size=B)
    ws = rng.multinomial(len(seeds), np.ones(len(seeds)) / len(seeds), size=B)
    bs = np.einsum('bc,cs,bs->b', wc, S, ws) / np.einsum('bc,cs,bs->b', wc, N, ws)
    p = max(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())), 1.0 / B)
    return S.sum() / N.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p


E = pd.read_csv('external_dmid_per_image.csv.gz'); E['cluster'] = E.file_id
M = pd.read_csv('external_mias_per_image.csv.gz'); M['cluster'] = ((M.file_id.str[3:].astype(int) + 1) // 2).astype(str)   # one woman = images 2k-1, 2k
ref = pd.read_csv('contrasts_v4.csv')
rows = []
for name, X, metrics in (('dmid', E, ['dice']), ('mias', M, ['dice_expert', 'dice_expert_breast', 'dice_proxy'])):
    for col in metrics:
        for exp in ('E1', 'E2'):
            for a, b in (('hq', 'hc'), ('hq', 'hf'), ('hq', 'ht')):
                t = X[X.exp == exp]
                x = t[t.model == a][['file_id', 'cluster', 'seed', col]].rename(columns={col: 'a'})
                y = t[t.model == b][['file_id', 'cluster', 'seed', col]].rename(columns={col: 'b'})
                m = x.merge(y, on=['file_id', 'cluster', 'seed']); m['v'] = m.a - m.b
                est, lo, hi, p = boot(m, np.random.default_rng(0))
                r = ref[(ref.set == name) & (ref.metric == col) & (ref.group == exp) & (ref.contrast == f'{a}-{b}')].iloc[0]
                rows.append(dict(set=name, metric=col, exp=exp, contrast=f'{a}-{b}', est=est, lo=lo, hi=hi, p=p,
                                 max_abs_diff_to_paper=max(abs(est - r.est), abs(lo - r.lo), abs(hi - r.hi))))
out = pd.DataFrame(rows); pd.set_option('display.width', 200)
print(out.round(4).to_string(index=False))
print('largest difference from the reported values:', out.max_abs_diff_to_paper.max())
