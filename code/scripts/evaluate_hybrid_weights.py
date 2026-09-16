#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiger_rec.config import load_config
from tiger_rec.evaluation.protocol import evaluate_retriever
from tiger_rec.retrieval.hybrid import HybridRetriever
from tiger_rec.retrieval.tiger import TigerRetriever
from tiger_rec.retrieval.traditional import build_traditional_retriever

parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--traditional',default='itemcf'); parser.add_argument('--weights',nargs='+',type=float,default=[0.0,0.05,0.1,0.2,0.3,0.5,1.0,2.0]); parser.add_argument('--output',required=True); args=parser.parse_args()
cfg=load_config(args.config)
traditional=build_traditional_retriever(cfg,args.traditional)
tiger=TigerRetriever(cfg,constrained=True)
rows=[]
for tiger_weight in args.weights:
 retriever=HybridRetriever([traditional,tiger],weights=[1.0,tiger_weight])
 retriever.name=f'hybrid_weight_{args.traditional}_{tiger_weight:g}'
 metrics=evaluate_retriever(cfg,retriever,split='test',max_users=0)
 row={k:metrics.get(k) for k in ['recall@10','recall@20','ndcg@10','ndcg@20','long_tail_recall@20','cold_start_recall@20','coverage','invalid_sid_rate','p50_ms','p95_ms','p99_ms','wall_time_sec']}
 row['tiger_weight']=tiger_weight
 rows.append(row)
 print(row)
Path(args.output).write_text(json.dumps({'traditional':args.traditional,'rows':rows},indent=2),encoding='utf-8')
