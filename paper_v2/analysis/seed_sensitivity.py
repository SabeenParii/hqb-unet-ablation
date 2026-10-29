# Seed sensitivity of the circuit contrasts (revision of the Scientific Reports manuscript).
# For the fixed-budget analysis (five seeds, epoch-50 weights), E1 and E2 (five seeds each), per contrast:
#   crossed hierarchical bootstrap (patients x seeds, as in the paper), bootstrap with the seeds of each model resampled
#   independently, seed-level differences with a t interval (n = 5, df = 4), and leave-one-seed-out crossed bootstraps.
# 10,000 replicates, rng(0). Input: data/p1all_test.csv.gz (E1/E2) and data/m3_all_per_image.csv (fixed budget).
import numpy as np, pandas as pd
from scipy import stats

B = 10000
T = pd.read_csv('data/p1all_test.csv.gz')
F = pd.read_csv('data/m3_all_per_image.csv')
F = F[F.policy == 'final'].assign(exp='FB', model=lambda d: d.model.map({'hybrid_quantum': 'hq', 'hybrid_classical': 'hc', 'hybrid_none': 'hf'}))
D = pd.concat([T[['exp', 'model', 'seed', 'file_id', 'patient_id', 'dice']], F[['exp', 'model', 'seed', 'file_id', 'patient_id', 'dice']]])
D['cluster'] = D.patient_id.astype(str)


def tables(u, col, seeds, cl):
    ci = {c: i for i, c in enumerate(cl)}; si = {s: i for i, s in enumerate(seeds)}
    S = np.zeros((len(cl), len(seeds))); N = np.zeros((len(cl), len(seeds)))
    for c, s, v in zip(u.cluster, u.seed, u[col]):
        S[ci[c], si[s]] += v; N[ci[c], si[s]] += 1
    return S, N


def crossed(m, rng):
    seeds = np.sort(m.seed.unique()); cl = np.sort(m.cluster.unique())
    S, N = tables(m, 'v', seeds, cl)
    wc = rng.multinomial(len(cl), np.ones(len(cl)) / len(cl), size=B)
    ws = rng.multinomial(len(seeds), np.ones(len(seeds)) / len(seeds), size=B)
    bs = np.einsum('bc,cs,bs->b', wc, S, ws) / np.einsum('bc,cs,bs->b', wc, N, ws)
    p = max(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())), 1.0 / B)
    return S.sum() / N.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p


def independent(m, rng):
    seeds = np.sort(m.seed.unique()); cl = np.sort(m.cluster.unique())
    Sa, N = tables(m, 'a', seeds, cl); Sb, _ = tables(m, 'b', seeds, cl)
    wc = rng.multinomial(len(cl), np.ones(len(cl)) / len(cl), size=B)
    wa = rng.multinomial(len(seeds), np.ones(len(seeds)) / len(seeds), size=B)
    wb = rng.multinomial(len(seeds), np.ones(len(seeds)) / len(seeds), size=B)
    den_a = np.einsum('bc,cs,bs->b', wc, N, wa); den_b = np.einsum('bc,cs,bs->b', wc, N, wb)
    bs = np.einsum('bc,cs,bs->b', wc, Sa, wa) / den_a - np.einsum('bc,cs,bs->b', wc, Sb, wb) / den_b
    p = max(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())), 1.0 / B)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), p


def paired(exp, a, b):
    t = D[D.exp == exp]
    x = t[t.model == a][['file_id', 'cluster', 'seed', 'dice']].rename(columns={'dice': 'a'})
    y = t[t.model == b][['file_id', 'cluster', 'seed', 'dice']].rename(columns={'dice': 'b'})
    m = x.merge(y, on=['file_id', 'cluster', 'seed']); assert len(m) == len(x) == len(y)
    m['v'] = m.a - m.b
    return m


rows, loso = [], []
for exp, pairs in (('FB', [('hq', 'hc'), ('hq', 'hf')]), ('E1', [('hq', 'hc'), ('hq', 'hf'), ('hq', 'ht')]),
                   ('E2', [('hq', 'hc'), ('hq', 'hf'), ('hq', 'ht')])):
    for a, b in pairs:
        m = paired(exp, a, b)
        est, lo, hi, p = crossed(m, np.random.default_rng(0))
        ilo, ihi, ip = independent(m, np.random.default_rng(0))
        sd = m.groupby('seed').v.mean(); n = len(sd)
        tl, th = stats.t.interval(0.95, n - 1, loc=sd.mean(), scale=sd.std(ddof=1) / np.sqrt(n))
        rows.append(dict(analysis=exp, contrast=f'{a}-{b}', n_seeds=n, est=est, lo=lo, hi=hi, p=p, indep_lo=ilo, indep_hi=ihi, indep_p=ip,
                         seed_diffs=' '.join(f'{v:+.4f}' for v in sd.values), seed_mean=sd.mean(), seed_sd=sd.std(ddof=1), t_lo=tl, t_hi=th,
                         seeds_positive=int((sd > 0).sum())))
        for s in sorted(m.seed.unique()):
            e2, l2, h2, p2 = crossed(m[m.seed != s], np.random.default_rng(0))
            loso.append(dict(analysis=exp, contrast=f'{a}-{b}', left_out=int(s), est=e2, lo=l2, hi=h2, p=p2))
R = pd.DataFrame(rows); L = pd.DataFrame(loso)
agg = L.groupby(['analysis', 'contrast']).agg(loso_est_min=('est', 'min'), loso_est_max=('est', 'max'), loso_lo_min=('lo', 'min'),
                                              loso_hi_max=('hi', 'max'), loso_n_excluding_zero=('p', lambda p: int((p < 0.05).sum())))
R = R.merge(agg.reset_index(), on=['analysis', 'contrast'])
R.to_csv('seed_sensitivity.csv', index=False); L.to_csv('seed_sensitivity_loso.csv', index=False)
pd.set_option('display.width', 250)
print(R.drop(columns=['seed_diffs']).round(4).to_string(index=False))
print(R[['analysis', 'contrast', 'seed_diffs']].to_string(index=False))
