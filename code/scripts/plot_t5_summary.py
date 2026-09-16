#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--summary',nargs='+',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
rows=[]
for name in args.summary:
 payload=json.loads(Path(name).read_text(encoding='utf-8'))
 rows.extend(payload.get('candidates',[]))
seen={}
for row in rows:
 if row.get('recall@20') is not None:
  seen[row['name']]=row
rows=list(seen.values())
rows.sort(key=lambda r: (-(float(r.get('recall@20') or 0.0)), -(float(r.get('ndcg@10') or 0.0))))
labels=[r['name'].replace('t5_v10_books_tune_','').replace('books_tune_','') for r in rows]
x=np.arange(len(rows))
fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True)
axes[0].bar(x-0.2,[float(r.get('recall@20') or 0) for r in rows],width=0.4,label='Recall@20')
axes[0].bar(x+0.2,[float(r.get('ndcg@10') or 0) for r in rows],width=0.4,label='NDCG@10')
axes[0].set_ylabel('score'); axes[0].set_title('T5 candidate retrieval quality'); axes[0].legend(); axes[0].grid(axis='y',alpha=.25)
axes[1].bar(x,[float(r.get('coverage') or 0) for r in rows],color='#2e86c1')
axes[1].set_ylabel('coverage'); axes[1].set_title('Catalog coverage'); axes[1].grid(axis='y',alpha=.25)
axes[1].set_xticks(x,labels,rotation=45,ha='right',fontsize=8)
fig.tight_layout(); Path(args.output).parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=220); print(args.output)
