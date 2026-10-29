# Manual reference annotations: reader 1 (first author, all 84 test images) and reader 2 (physician, 20 images), scored against the
# rule-based proxy and every E1/E2 run. Same crossed patient x seed bootstrap as the main analysis (rng(0), 10,000 replicates).
import numpy as np, pandas as pd
import analysis_p1v3 as A
S = pd.read_csv('data/annot/reader1/annot_scores.csv.gz'); Q = pd.read_csv('data/annot/reader1/annot_qc.csv')
S2 = pd.read_csv('data/annot/reader2/scores_20_both_readers.csv.gz'); IR = pd.read_csv('data/annot/reader2/inter_reader.csv')
pidmap = A.T.drop_duplicates('file_id').set_index('file_id').patient_id
for X in (S, S2): X['cluster'] = X.file_id.map(pidmap).astype(str)
assert S.cluster.notna().all() and S2.cluster.ne('nan').all()
Q['cluster'] = Q.file_id.map(pidmap).astype(str); IR['cluster'] = IR.file_id.map(pidmap).astype(str)
G = {'E1': range(5), 'E2': range(5)}
rows = []
def lvl(t, label, **kw):
    e, lo, hi, p = A.boot_stat(t, np.random.default_rng(0)); rows.append(dict(analysis=label, est=e, lo=lo, hi=hi, p=p, **kw))
# rule vs reader 1 (84 images; seeds absent -> single pseudo-seed)
for col in ('rule_vs_annot_full', 'rule_vs_annot_breast'):
    lvl(Q.assign(seed=0, v=Q[col]), 'rule vs reader1', metric=col.split('_')[-1])
for exp in ('E1', 'E2'):
    for col in ('dice_full', 'dice_breast'):
        t = S[(S.exp == exp) & S.seed.isin(G[exp])]
        for m in ('hq', 'hc', 'hf', 'ht'):
            lvl(t[t.model == m].assign(v=lambda d: d[col]), 'model vs reader1', exp=exp, model=m, metric=col.split('_')[-1])
            r = Q.set_index('file_id')['rule_vs_annot_' + col.split('_')[-1]]
            u = t[t.model == m].copy(); u['v'] = u[col] - u.file_id.map(r)
            lvl(u, 'model - rule, vs reader1', exp=exp, model=m, metric=col.split('_')[-1])
        for a, b in (('hq', 'hc'), ('hq', 'hf'), ('hq', 'ht')):
            x = t[t.model == a][['file_id', 'cluster', 'seed', col]].merge(t[t.model == b][['file_id', 'cluster', 'seed', col]], on=['file_id', 'cluster', 'seed'])
            assert len(x) == 84 * 5
            lvl(x.assign(v=x[col + '_x'] - x[col + '_y']), 'contrast vs reader1', exp=exp, model=f'{a}-{b}', metric=col.split('_')[-1])
# reader 2 subset
for col in ('inter_full', 'inter_breast', 'rule_vs_r1_full', 'rule_vs_r2_full'):
    lvl(IR.assign(seed=0, v=IR[col]), 'subset of 20', metric=col)
for exp in ('E1', 'E2'):
    for m in ('hq', 'hc', 'hf', 'ht'):
        for rd in ('reader1', 'reader2'):
            t = S2[(S2.exp == exp) & (S2.model == m) & (S2.reader == rd)]
            lvl(t.assign(v=t.dice_full), 'model vs reader, subset of 20', exp=exp, model=m, metric=rd)
        a = S2[(S2.exp == exp) & (S2.model == m)].pivot_table(index=['file_id', 'cluster', 'seed'], columns='reader', values='dice_full').reset_index()
        lvl(a.assign(v=a.reader2 - a.reader1), 'reader2 - reader1 score, subset of 20', exp=exp, model=m, metric='full')
R = pd.DataFrame(rows); R.to_csv('readers_stats.csv', index=False)
# descriptive extras
v = IR.groupby('view')[['inter_full', 'inter_breast']].agg(['mean', 'min'])
rank1 = S2[S2.reader == 'reader1'].groupby(['exp', 'model']).dice_full.mean(); rank2 = S2[S2.reader == 'reader2'].groupby(['exp', 'model']).dice_full.mean()
print(v.round(4)); print('pectoral agreement', (IR.pect_r1 == IR.pect_r2).sum(), 'of', len(IR), '| MLO', (IR.view == 'MLO').sum(), '| pect r1', IR.pect_r1.sum())
print('rank identical within exp:', all((rank1[e].rank() == rank2[e].rank()).all() for e in ('E1', 'E2')), '| overall', (rank1.rank() == rank2.rank()).all())
print('patients in subset', IR.cluster.nunique(), '| reader1 MLO with pectoral', Q[Q.view == 'MLO'].has_pectoral.mean(), Q.view.value_counts().to_dict())
pd.set_option('display.width', 220); print(R.round(4).to_string())
