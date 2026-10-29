import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'font.family':'DejaVu Sans'})
C=pd.read_csv('stats/contrasts.csv'); L=pd.read_csv('stats/levels.csv'); T=pd.read_csv('data/v2/p1v2_test.csv.gz')
P=T.pivot_table(index=['exp','seed'],columns='model',values='dice',aggfunc='mean')
fig,ax=plt.subplots(1,3,figsize=(7.2,2.9),gridspec_kw=dict(wspace=0.62,width_ratios=[1.15,1,1]))
EXP={'E1':('With skip connections (E1)','#1f4e9c'),'E2':('Without skip connections (E2)','#c0392b')}
a=ax[0]; a.axhline(0,color='0.5',lw=0.6,ls='--')
for i,(m,lab) in enumerate((('hc','HQB-UNet\n− HQB-cls'),('hf','HQB-UNet\n− HQB-free'))):
    for j,exp in enumerate(('E1','E2')):
        x0=i+(-0.18 if exp=='E1' else 0.18); col=EXP[exp][1]
        ps=(P.loc[exp]['hq']-P.loc[exp][m]).values
        a.plot([x0-0.07]*len(ps),ps,'o',ms=3,color=col,alpha=0.55,mew=0,label=('single seeds' if (i==0 and j==0) else None))
        r=C[(C.set=='test')&(C.metric=='dice')&(C.exp==exp)&(C.contrast==f'hq-{m}')].iloc[0]
        a.errorbar(x0+0.07,r.est,yerr=[[r.est-r.lo],[r.hi-r.est]],fmt='D',ms=4,color=col,capsize=2,lw=0.8,label=(EXP[exp][0] if i==0 else None))
a.set_xticks([0,1]); a.set_xticklabels(['HQB-UNet\n− HQB-cls','HQB-UNet\n− HQB-free'],fontsize=7.5)
a.set_ylabel('Difference in per-image Dice\n(test set)'); a.set_xlim(-0.5,1.5); a.set_title('Circuit contrasts',fontsize=8.5)
a.text(-0.32,1.05,'a',transform=a.transAxes,fontsize=11,fontweight='bold')
a.legend(fontsize=6.5,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.25),ncol=1,handletextpad=0.2)
b=ax[1]
names=['HQB-\nUNet','HQB-\ncls','HQB-\nfree']
for i,m in enumerate(('hq','hc','hf')):
    for exp in ('E1','E2'):
        col=EXP[exp][1]; x0=i+(-0.15 if exp=='E1' else 0.15)
        for met,mk,dx in (('drop_const','x',-0.06),('drop_swapavg','D',0.06)):
            r=L[(L.set=='test')&(L.metric==met)&(L.exp==exp)&(L.model==m)].iloc[0]
            b.errorbar(x0+dx,r.est,yerr=[[r.est-r.lo],[r.hi-r.est]],fmt=mk,ms=4,color=col,mfc=('white' if mk=='D' else col),capsize=1.5,lw=0.7)
b.plot([],[],'x',color='0.3',label='constant code'); b.plot([],[],'D',mfc='white',color='0.3',ms=4,label='random training codes (mean of 5)')
b.set_xticks(range(3)); b.set_xticklabels(names,fontsize=7.5); b.set_ylabel('Mean drop in per-image Dice\n(test set)')
b.set_xlim(-0.5,2.5); b.set_ylim(-0.03,0.95); b.set_title('Code replaced at test time',fontsize=8.5)
b.text(-0.32,1.05,'b',transform=b.transAxes,fontsize=11,fontweight='bold')
b.legend(fontsize=6.5,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.25),ncol=1,handletextpad=0.2)
c=ax[2]; c.axhline(0.8653,color='0.3',lw=0.8,ls=':')
c.text(2.45,0.8653+0.012,'labelling rule',ha='right',fontsize=6.5,color='0.3')
for i,m in enumerate(('hq','hc','hf')):
    for exp in ('E1','E2'):
        col=EXP[exp][1]; x0=i+(-0.12 if exp=='E1' else 0.12)
        r=L[(L.set=='mias')&(L.metric=='dice_expert')&(L.exp==exp)&(L.model==m)].iloc[0]
        c.errorbar(x0,r.est,yerr=[[r.est-r.lo],[r.hi-r.est]],fmt='o',ms=4,color=col,capsize=2,lw=0.8)
c.set_xticks(range(3)); c.set_xticklabels(names,fontsize=7.5); c.set_ylabel('Per-image Dice against\nexpert masks (mini-MIAS)')
c.set_xlim(-0.5,2.5); c.set_ylim(0,1); c.set_title('External expert reference',fontsize=8.5)
c.text(-0.32,1.05,'c',transform=c.transAxes,fontsize=11,fontweight='bold')
for x in ax: x.tick_params(labelsize=7.5)
import os; os.makedirs('figs', exist_ok=True); out='figs/fig4_v2'
fig.savefig(out+'.png',dpi=600,bbox_inches='tight'); fig.savefig(out+'.pdf',bbox_inches='tight'); fig.savefig(out+'.eps',bbox_inches='tight')
