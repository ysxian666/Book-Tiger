#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

root=Path('results/improvement_suite')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
base=read(root/'books_improve_00_content_baseline/semantic_id_report.json'); base_m=read(root/'books_improve_00_content_baseline/metrics/tiger_trie_test.json')
p1=read(root/'books_improve_04_hard_reset/semantic_id_report.json'); p1_m=read(root/'books_improve_04_hard_reset/metrics/tiger_trie_test.json')
p2=read(root/'books_improve_07_min_vocab/semantic_id_report.json'); seed_files=sorted(root.glob('user_seed_*_beam20_*.json'))
seed_metrics=[read(path) for path in seed_files]
p2_m={key:sum(float(row[key]) for row in seed_metrics)/len(seed_metrics) for key in ['recall@20','ndcg@10','coverage','p99_ms']}
labels=['Baseline','Point 1\nCodebook fix','Point 2\nCompact SID + Hybrid']
series=[
 ('Utilization (%)',[base['mean_utilization']*100,p1['mean_utilization']*100,p2['mean_utilization']*100],'#2471a3'),
 ('Base collision (%)',[base['base_collision_rate']*100,p1['base_collision_rate']*100,p2['base_collision_rate']*100],'#c0392b'),
 ('Recall@20',[base_m['recall@20'],p1_m['recall@20'],p2_m['recall@20']],'#1e8449'),
 ('NDCG@10',[base_m['ndcg@10'],p1_m['ndcg@10'],p2_m['ndcg@10']],'#8e44ad'),
 ('Coverage',[base_m['coverage'],p1_m['coverage'],p2_m['coverage']],'#16a085'),
 ('P99 latency (ms)',[base_m['p99_ms'],p1_m['p99_ms'],p2_m['p99_ms']],'#d68910'),
]
fig,axes=plt.subplots(2,3,figsize=(15,8))
for ax,(title,values,color) in zip(axes.flat,series):
 bars=ax.bar(labels,values,color=color)
 ax.set_title(title); ax.grid(axis='y',alpha=.25); ax.set_axisbelow(True)
 for bar,value in zip(bars,values):
  ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.4g}',ha='center',va='bottom',fontsize=8)
fig.suptitle('Two optimization points on quick Amazon Books')
fig.tight_layout(rect=[0,0,1,0.96]); Path('results/two_optimization_points.png').parent.mkdir(parents=True,exist_ok=True); fig.savefig('results/two_optimization_points.png',dpi=220); print('saved results/two_optimization_points.png')
