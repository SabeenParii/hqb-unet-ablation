import numpy as np, pandas as pd
d=pd.read_csv('m3_all_per_image.csv')
fid=d[(d.model=='hybrid_quantum')&(d.seed==0)&(d.policy=='final')].file_id.to_numpy()
pid=d.set_index('file_id').loc[fid].groupby(level=0).patient_id.first().loc[fid].to_numpy()
upat=np.unique(pid); pidx=[np.where(pid==u)[0] for u in upat]
S=sorted(d.seed.unique())
def A(m,pol): return np.stack([d[(d.model==m)&(d.seed==s)&(d.policy==pol)].set_index('file_id').loc[fid,'dice'].to_numpy() for s in S])
def boot(X,Y=None,paired=True,n=10000,seed=0):
    r=np.random.default_rng(seed); out=np.empty(n); k=X.shape[0]
    for t in range(n):
        pp=np.concatenate([pidx[i] for i in r.integers(0,len(pidx),len(pidx))])
        sa=r.integers(0,k,k)
        if Y is None: out[t]=X[np.ix_(sa,pp)].mean(); continue
        sb=sa if paired else r.integers(0,k,k)
        out[t]=X[np.ix_(sa,pp)].mean()-Y[np.ix_(sb,pp)].mean()
    return out
rows=[]
M={'Q':'hybrid_quantum','C':'hybrid_classical','F':'hybrid_none'}
for pol in ['final','min20','es']:
    D={k:A(v,pol) for k,v in M.items()}
    for k in D:
        b=boot(D[k]); rows.append(dict(policy=pol,comp=k,mean=D[k].mean(),lo=np.percentile(b,2.5),hi=np.percentile(b,97.5),seeds=np.round(D[k].mean(1),4).tolist()))
    for a,c in [('Q','C'),('Q','F'),('F','C')]:
        diff=D[a].mean()-D[c].mean(); b=boot(D[a],D[c]); bi=boot(D[a],D[c],paired=False)
        p=max(min(1,2*min((b<=0).mean(),(b>=0).mean())),1e-4)
        rows.append(dict(policy=pol,comp=f'{a}-{c}',mean=diff,lo=np.percentile(b,2.5),hi=np.percentile(b,97.5),lo90=np.percentile(b,5),hi90=np.percentile(b,95),p=p,
            ind_lo=np.percentile(bi,2.5),ind_hi=np.percentile(bi,97.5),seeds=np.round(D[a].mean(1)-D[c].mean(1),4).tolist()))
R=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print(R.round(4).to_string(index=False)); R.to_csv('m3_stats.csv',index=False)
