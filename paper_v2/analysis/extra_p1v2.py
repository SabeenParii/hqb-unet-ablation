import numpy as np, pandas as pd, sys
sys.argv=['x']; exec(open('analysis_p1v2.py').read().split("SETS = {")[0])   # functions + data
man = pd.read_csv('data/v2/mias_manifest.csv').drop_duplicates('file_id').set_index('file_id')
rows=[]
# models minus rule vs expert masks (mini-MIAS), per image, paired
for col, rc in (('dice_expert','dice_rule_vs_expert_full'),('dice_expert_breast','dice_rule_vs_expert_breast')):
    for exp in ('E1','E2'):
        for mod in ('hq','hc','hf'):
            t=M[(M.exp==exp)&(M.model==mod)].copy(); t['v']=t[col]-t.file_id.map(man[rc])
            e,lo,hi,p=boot_stat(t,np.random.default_rng(0)); rows.append(dict(analysis='model - rule (vs expert)',metric=col,exp=exp,model=mod,est=e,lo=lo,hi=hi,p=p))
# sensitivity: exclude 5 images with flipped orientation
keep = man.index[man.orientation=='T']
M2 = M[M.file_id.isin(keep)]
for col in ('dice_expert','dice_expert_breast'):
    for exp in ('E1','E2'):
        for a,b in (('hq','hc'),('hq','hf')):
            e,lo,hi,p=contrast(M2,exp,a,b,col); rows.append(dict(analysis='contrast, 317 images (orientation T only)',metric=col,exp=exp,model=f'{a}-{b}',est=e,lo=lo,hi=hi,p=p))
# severing: E2 minus E1 drop per model is not paired (different runs); report levels only. Fraction of images with |change|>0.05
for sname,X in (('test',T),('dmid',E)):
    for exp in ('E1','E2'):
        for mod in ('hq','hc','hf'):
            t=X[(X.exp==exp)&(X.model==mod)]
            rows.append(dict(analysis='share of images with |const drop|>0.05',metric=sname,exp=exp,model=mod,est=(t.drop_const.abs()>0.05).mean()))
R=pd.DataFrame(rows); R.to_csv('stats/extra.csv',index=False); pd.set_option('display.width',200); print(R.round(4).to_string())
print('rule vs expert on MIAS: full %.4f breast %.4f' % (man.dice_rule_vs_expert_full.mean(), man.dice_rule_vs_expert_breast.mean()))
