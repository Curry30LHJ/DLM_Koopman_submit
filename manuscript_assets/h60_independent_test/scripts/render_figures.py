"""Render archived Tables 2/3 and Figures 2/4 without model inference.

--check-only compares rendered PNG pixels and table text with accepted assets.
--output-dir must name a new directory. Figure 5 is deliberately not touched.
"""
import argparse, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from PIL import Image

A=Path(__file__).resolve().parents[1]
D=A/'data'
MODELS=['mlp','lstm','hakan','m5','m6','m8']
ORDER=['mlp','lstm','hakan','m6','m8','m5']
COL=dict(zip(MODELS,['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9']))
LS=dict(zip(MODELS,['-','--','-.','-',':',(0,(5,1.6))]))
LAB=dict(zip(MODELS,['MLP–K','LSTM–K','HaKAN–K','DLM–K','Markov RBF','Delay–RBF']))
TEX=dict(zip(MODELS,[r'MLP--K~\cite{Lusch2018}',r'LSTM--K~\cite{Han2024DRKO}',r'HaKAN--K~\cite{Hasan2026}','DLM--K',r'Markov RBF~\cite{Williams2015,Korda2018}',r'Delay--RBF~\cite{Pan2020TimeDelay}']))
FONT=8.5*(6.7*72)/(0.8*468.3324)
plt.rcParams.update({'font.family':'STIXGeneral','font.size':FONT,'mathtext.fontset':'stix',
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65})
parser=argparse.ArgumentParser()
group=parser.add_mutually_exclusive_group(required=True)
group.add_argument('--check-only',action='store_true')
group.add_argument('--output-dir',type=Path)
args=parser.parse_args()
if args.output_dir:
    args.output_dir.mkdir(parents=True,exist_ok=False)
curve=pd.read_csv(D/'independent_test_H1_H60.csv')
checks={'order':ORDER,'font':'STIXGeneral','effective_font_pt_at_0.8_linewidth':8.5,'new_uncertainty_bands':False}

def handles():
    return [Line2D([],[],color='#222',lw=1.15,ls='--',label='Truth')]+[Line2D([],[],color=COL[m],lw=1.65 if m=='m5' else 1,ls=LS[m],label=LAB[m]) for m in ORDER]

def labels(fig):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for ax,letter in zip(fig.axes,'abcd'):
        p=ax.get_position(); y=ax.xaxis.label.get_window_extent(renderer).y0/fig.bbox.height
        fig.text((p.x0+p.x1)/2,y-4/(72*fig.get_figheight()),f'({letter})',ha='center',va='top')

def line(ax,t,y,m):
    ax.plot(t,y,color=COL[m],ls=LS[m],lw=1.65 if m=='m5' else 1,zorder=4 if m=='m5' else 2)

def finish(fig,name):
    checks[name]={'axes':len(fig.axes)}
    if args.check_only:
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=600);buf.seek(0)
        with Image.open(buf) as rendered,Image.open(A/f'figures/{name}.png') as accepted:
            identical=np.array_equal(np.asarray(rendered),np.asarray(accepted))
        checks[name]['accepted_png_pixels_equal']=bool(identical)
        assert identical,f'{name}: rendering differs; check matplotlib/font versions'
    else:
        for ext in ['pdf','svg','png']: fig.savefig(args.output_dir/f'{name}.{ext}',dpi=600)
    plt.close(fig)

for system in ['lorenz','three_tank']:
    r=pd.read_csv(D/('lorenz_fixed_origin_H60.csv' if system=='lorenz' else 'figure4_median_window_H60.csv'),**({'float_precision':'round_trip'} if system=='three_tank' else {}))
    n=3 if system=='lorenz' else 9
    assert len(r)==6*60*n and np.isfinite(r[['truth','prediction']]).all().all()
    assert not r.duplicated(['model','component','horizon']).any()
    assert (r.groupby(['component','horizon']).truth.nunique()==1).all()
    assert set(r.trajectory)==({'fresh_000'} if n==3 else {'fresh_003'}) and set(r.origin)==({20} if n==3 else {43})
    fig,axs=plt.subplots(2,2,figsize=(6.7,7.0)) if n==3 else plt.subplots(3,3,figsize=(6.7,5.5),sharex=True)
    for component in range(n):
        ax=axs.flat[component]; truth=r[(r.component==component)&(r.model=='m5')].sort_values('horizon')
        scale=1/30 if n==3 else .005
        ax.plot(truth.horizon*scale,truth.truth,color='#222',ls='--',lw=1.15,zorder=5)
        for m in ORDER:
            q=r[(r.component==component)&(r.model==m)].sort_values('horizon');line(ax,q.horizon*scale,q.prediction,m)
        ax.grid(axis='y',lw=.3,color='#ddd')
        if n==3:
            ax.set(xlabel='Time',ylabel=f'$x_{component+1}$',xlim=(1/30,2));ax.set_xticks([.5,1,1.5,2]);ax.yaxis.set_major_locator(MaxNLocator(5))
        else:
            row,col=divmod(component,3)
            ax.set_ylabel([rf'$x_{{A,{row+1}}}$',rf'$x_{{B,{row+1}}}$',rf'$T_{row+1}$ (K)'][col],labelpad=4)
            ax.set_xlim(0,.3);ax.set_xticks([0,.1,.2,.3]);ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_formatter(FormatStrFormatter(['%.4f','%.3f','%.0f'][col]))
            if row==2:ax.set_xlabel('Time (h)',labelpad=3)
    hs=handles();hs=[hs[i] for i in [0,4,1,5,2,6,3]]
    fig.legend(hs,[h.get_label() for h in hs],loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.995),handlelength=1.4,columnspacing=.75,handletextpad=.35)
    if n==3:
        ax=axs.flat[3]
        for m in ORDER:
            q=curve[(curve.system==system)&(curve.model==m)].sort_values('horizon');line(ax,q.horizon/30,q.endpoint,m)
        ax.axvline(1,color='#555',ls=':',lw=.7)
        ax.set(xlabel='Time',ylabel='Error',xlim=(1/30,2),ylim=(0,None));ax.set_xticks([.5,1,1.5,2]);ax.grid(axis='y',lw=.3,color='#ddd')
        fig.subplots_adjust(top=.88,wspace=.45,hspace=.62,left=.16,right=.97,bottom=.14);labels(fig)
        finish(fig,'fig_2_lorenz_evaluation_h60')
    else:
        fig.subplots_adjust(top=.85,wspace=.72,hspace=.30,left=.14,right=.98,bottom=.10)
        finish(fig,'fig_4_cstr_representative_h60')

for system,name,label in [('lorenz','Lorenz','tab:lorenz-independent'),('three_tank','CSTR','tab:primary')]:
    block=[r'\begin{table}[pos=!htbp]',rf'\caption{{Mean {name} prediction error on the independent simulated test set. Bold values are the lowest among the six compared predictors.}}',rf'\label{{{label}}}',r'\centering\small',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',r'Model & step 1 & step 5 & step 10 & step 20 & step 30 & step 60 \\\midrule']
    table=curve[curve.system==system].pivot(index='model',columns='horizon',values='endpoint')
    for m in ORDER:
        nums=[]
        for h in [1,5,10,20,30,60]:
            v=f'{table.loc[m,h]:.4g}';nums.append(r'\textbf{'+v+'}' if table.loc[m,h]==table[h].min() else v)
        block.append(TEX[m]+' & '+' & '.join(nums)+r' \\')
    block.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    content='\n'.join(block)+'\n'
    if args.check_only:
        assert content==(A/f'tables/{system}_table.tex').read_text(encoding='utf-8')
    else:
        (args.output_dir/f'{system}_table.tex').write_text(content,encoding='utf-8')
    checks[f'{system}_table_matches']=True
if args.output_dir:
    (args.output_dir/'REPRODUCTION_CHECKS.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
print(curve[curve.horizon.isin([1,5,10,20,30,60])].pivot(index=['system','model'],columns='horizon',values='endpoint').to_string())
