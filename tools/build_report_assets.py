"""Build truthful documentation figures from retained evidence. No training/network."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets'

def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    receipts=read_rows(ROOT/'docs/evidence/kaggle-submissions.csv')
    receipts.sort(key=lambda x:x['date'])
    scores=[float(x['publicScore']) for x in receipts]
    best=[]
    for score in scores: best.append(max([score]+best[-1:]))
    names={56824542:'v1',56825418:'v2',56826952:'v4',56827009:'v4b',56827698:'v5',56829831:'v9',56830042:'v10',56831596:'v15',56831621:'v18',56832161:'v25',56841525:'v38',56841543:'v43',56841584:'v45',56841621:'v21 exact',56841650:'v46',56841675:'v47',56841712:'v21 port'}
    labels=[names.get(int(x['ref']),x['ref']) for x in receipts]
    x=list(range(1,len(scores)+1))
    fig,ax=plt.subplots(figsize=(13.6,5.5))
    ax.plot(x,scores,marker='o',linewidth=1.5,label='Every submitted candidate')
    ax.step(x,best,where='post',linewidth=2.2,label='Best score so far')
    ax.set_xticks(x,labels,rotation=35,ha='right')
    ax.set_ylim(.775,.839);ax.yaxis.set_major_formatter(PercentFormatter(1,1))
    ax.set_xlabel('Submission order in the 2026 campaign (not equal time intervals)')
    ax.set_ylabel('Recorded Kaggle Public accuracy')
    ax.set_title('17 submissions: improvement included regressions',loc='left',fontweight='bold',pad=14)
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',frameon=False)
    i=labels.index('v47');ax.annotate('v47: 0.83014',xy=(x[i],scores[i]),xytext=(x[i]-3.5,.834),arrowprops={'arrowstyle':'->'})
    fig.text(.07,.015,'Source: docs/evidence/kaggle-submissions.csv | Historical Public feedback was used in selection.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'submission-history.png',dpi=160);plt.close(fig)

    rows=read_rows(ROOT/'exports/v42/summary.csv')
    selected={r['family']:r for r in rows if r['variant']=='p3_full'}
    keys=['repeated','group','pseudo'];delta=[float(selected[k]['mean_delta'])*100 for k in keys]
    fig,ax=plt.subplots(figsize=(10,4.6))
    bars=ax.barh(['Repeated splits (6)','Group stress splits (2)','Matched holdouts (5)'],delta)
    ax.bar_label(bars,labels=[f'+{v:.3f} pp' for v in delta],padding=7)
    ax.set_xlim(0,max(delta)*1.3);ax.set_xlabel('Mean accuracy difference vs the typed + EB parent, percentage points')
    ax.set_title('P3-full: reported local differences, not independent proof',loc='left',fontweight='bold',pad=14)
    ax.grid(axis='x',alpha=.2)
    fig.text(.07,.015,'Source: exports/v42/summary.csv | Overlapping data; model seeds also differ between parent and variants.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(OUT/'p3-validation-deltas.png',dpi=160);plt.close(fig)

    bag=next(r for r in read_rows(ROOT/'exports/v44/summary.csv') if r['family']=='bagged')
    v25=next(r for r in read_rows(ROOT/'exports/v25/validation_surfaces.csv') if r['surface']=='fixed')
    oof=[.85410,float(v25['majority_accuracy']),float(bag['mean_p3_accuracy'])]
    public=[.79665,.79186,.79665]
    fig,ax=plt.subplots(figsize=(10,5))
    pos=[0,1,2];width=.34
    a=ax.bar([p-width/2 for p in pos],oof,width,label='Reported local OOF')
    b=ax.bar([p+width/2 for p in pos],public,width,label='Submitted Public')
    ax.bar_label(a,labels=[f'{v:.5f}' for v in oof],padding=4,fontsize=10)
    ax.bar_label(b,labels=[f'{v:.5f}' for v in public],padding=4,fontsize=10)
    ax.set_xticks(pos,['v5 benchmark','v25 majority','v43/v44 P3 bag'])
    ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1,0))
    ax.set_ylabel('Accuracy');ax.set_title('Local evaluation and Public feedback are different evidence',loc='left',fontweight='bold',pad=14)
    ax.legend(loc='upper center',ncol=2,frameon=False);ax.grid(axis='y',alpha=.2)
    fig.text(.07,.015,'Historical protocols differ. v47 has no independently certified OOF score and is deliberately not assigned one.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(OUT/'local-vs-public.png',dpi=160);plt.close(fig)

    hero='''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="420" viewBox="0 0 1440 420" role="img" aria-labelledby="title desc">
<title id="title">Titanic x GPT Web Experiment</title><desc id="desc">Human-guided semi-automated experimentation. Antigravity initial setup, GPT web session execution, recorded Public best 0.83014. Evidence and limitations documented.</desc>
<defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#071c2e"/><stop offset="1" stop-color="#143c4a"/></linearGradient></defs>
<rect width="1440" height="420" rx="20" fill="url(#bg)"/>
<g fill="none" stroke="#ffffff" opacity=".06"><path d="M0 100H1440M0 200H1440M0 300H1440M150 0V420M350 0V420M550 0V420M750 0V420M950 0V420M1150 0V420M1350 0V420"/></g>
<text x="70" y="85" font-family="Arial,sans-serif" font-size="19" letter-spacing="4" fill="#8ee1d5">HUMAN-GUIDED / SEMI-AUTOMATED / DOCUMENTED</text>
<text x="65" y="169" font-family="Arial,sans-serif" font-weight="bold" font-size="62" fill="#ffffff">Titanic x GPT Web</text>
<text x="70" y="225" font-family="Arial,sans-serif" font-size="29" fill="#b9d2dd">Custom skills. Prompts. Experiments. Honest evidence.</text>
<text x="70" y="292" font-family="Arial,sans-serif" font-size="21" fill="#ffffff">Antigravity setup  /  GPT web-session execution</text>
<text x="70" y="348" font-family="Arial,sans-serif" font-size="17" fill="#b9d2dd">An observational case study, not a leakage-free performance certificate.</text>
<rect x="1045" y="105" width="325" height="205" rx="18" fill="#ffffff" opacity=".08"/>
<text x="1080" y="152" font-family="Arial,sans-serif" font-size="18" fill="#c7e3e9">RECORDED PUBLIC BEST</text>
<text x="1074" y="224" font-family="Arial,sans-serif" font-weight="bold" font-size="60" fill="#8ee1d5">0.83014</text>
<text x="1080" y="274" font-family="Arial,sans-serif" font-size="18" fill="#ffffff">v47 / October 2026 campaign</text>
</svg>'''
    (OUT/'hero.svg').write_text(hero,encoding='utf-8')
    print('Built hero.svg, submission-history.png, p3-validation-deltas.png, local-vs-public.png')

if __name__=='__main__': main()
