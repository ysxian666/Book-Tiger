#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

parser=argparse.ArgumentParser(); parser.add_argument('--pattern',required=True); parser.add_argument('--output',required=True); args=parser.parse_args()
rows=[]
for metrics_name in sorted(glob.glob(args.pattern)):
 metrics=json.loads(Path(metrics_name).read_text(encoding='utf-8'))
 stem=Path(metrics_name).name
 training_name=metrics_name.replace('.metrics.json','.training.json')
 training=json.loads(Path(training_name).read_text(encoding='utf-8')) if Path(training_name).exists() else {}
 rows.append({'name':stem.replace('.metrics.json',''),'best_valid_loss':training.get('best_valid_loss'),'epochs':len(training.get('history',[])),'recall@20':metrics.get('recall@20'),'ndcg@10':metrics.get('ndcg@10'),'coverage':metrics.get('coverage'),'invalid_sid_rate':metrics.get('invalid_sid_rate'),'p99_ms':metrics.get('p99_ms')})
rows.sort(key=lambda r: (-(r['recall@20'] or 0), -(r['ndcg@10'] or 0)))
Path(args.output).write_text(json.dumps({'candidates':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'candidates':rows},ensure_ascii=False,indent=2))
