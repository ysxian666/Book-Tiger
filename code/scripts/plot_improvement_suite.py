#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser(); parser.add_argument('--summary',nargs='+',required=True); parser.add_argument('--output',required=True); args=parser.parse_args()
rows=[]; seen=set()
for name in args.summary:
 payload=json.loads(Path(name).read_text(encoding='utf-8'))
 for row in payload.get('candidates',[]):
  if row.get('success', True) and row['name'] not in seen:
   rows.append(row); seen.add(row['name'])
order={
 'books_improve_00_content_baseline':0,
 'books_improve_01_gate':1,
 'books_improve_02_frequency':2,
 'books_improve_03_usage':3,
 'books_improve_04_hard_reset':4,
 'books_improve_05_geometry':5,
 'books_improve_06_collision_all':6,
 'books_improve_07_min_vocab':7,
 'books_improve_a1_cross_attention':8,
 'books_improve_a2_frequency_strong':9,
 'books_improve_a3_geometry5':10,
 'books_improve_a4_kmeans_off':11,
}
rows.sort(key=lambda r:order.get(r['name'],999))
labels=[r['name'].replace('books_improve_','') for r in rows]
x=np.arange(len(rows))
fig,axes=plt.subplots(4,1,figsize=(14,14),sharex=True)
axes[0].plot(x,[r['mean_utilization'] for r in rows],marker='o',color='#2e86c1'); axes[0].set_ylabel('utilization'); axes[0].set_ylim(0,1.03)
axes[1].plot(x,[r['base_collision_rate'] for r in rows],marker='o',color='#c0392b'); axes[1].set_ylabel('collision'); axes[1].set_ylim(0,1.03)
axes[2].plot(x,[float(r.get('recall@20') or 0) for r in rows],marker='o',color='#1e8449'); axes[2].set_ylabel('Recall@20')
axes[3].plot(x,[float(r.get('p99_ms') or 0) for r in rows],marker='o',color='#d68910'); axes[3].set_ylabel('P99 ms')
for ax in axes: ax.grid(alpha=.25); ax.set_axisbelow(True)
axes[0].set_title('TIGER improvement chain on quick data')
axes[3].set_xticks(x,labels,rotation=45,ha='right',fontsize=8)
fig.tight_layout(); Path(args.output).parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=220); print(args.output)
