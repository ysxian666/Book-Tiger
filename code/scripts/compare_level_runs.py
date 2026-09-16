#!/usr/bin/env python3
"""Compare a 3-level and 4-level RQ-VAE/TIGER run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiger_rec.config import load_config


def read_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def gather(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    sid_report = read_json(cfg.paths.artifact('semantic_id_report.json'))
    timing = read_json(cfg.paths.artifact('pipeline_timing.json'))
    metrics = read_json(cfg.paths.artifact('metrics', 'tiger_trie_test.json'))
    stage_seconds = {row['stage']: float(row['seconds']) for row in timing.get('timings', [])}
    utilization = {
        key: float(value)
        for key, value in sid_report.items()
        if key == 'mean_utilization' or key.startswith('utilization_level_')
    }
    return {
        'config': str(Path(config_path).resolve()),
        'num_levels': int(sid_report.get('token_space', {}).get('num_levels', -1)),
        'length_mode': sid_report.get('length_mode'),
        'mean_sid_length': float(sid_report.get('mean_sid_length', 0.0)),
        'min_sid_length': int(sid_report.get('min_sid_length', 0)),
        'max_sid_length': int(sid_report.get('max_sid_length', 0)),
        'mean_utilization': float(sid_report.get('mean_utilization', 0.0)),
        'utilization': utilization,
        'base_collision_rate': float(sid_report.get('base_collision_rate', 0.0)),
        'semantic_reallocated': int(sid_report.get('semantic_reallocated', 0)),
        'suffix_resolved': int(sid_report.get('suffix_resolved', 0)),
        'resolution_seconds': float(sid_report.get('resolution_seconds', 0.0)),
        'stage_seconds': stage_seconds,
        'total_seconds': float(timing.get('total_seconds', 0.0)),
        'metrics': {
            'recall@20': float(metrics.get('recall@20', 0.0)),
            'ndcg@10': float(metrics.get('ndcg@10', 0.0)),
            'long_tail_recall@20': float(metrics.get('long_tail_recall@20', 0.0)),
            'cold_start_recall@20': float(metrics.get('cold_start_recall@20', 0.0)),
            'invalid_sid_rate': float(metrics.get('invalid_sid_rate', 0.0)),
            'p99_ms': float(metrics.get('p99_ms', 0.0)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config3', required=True)
    parser.add_argument('--config4', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument('--min-utilization', type=float, default=0.8)
    parser.add_argument('--min-recall-gain', type=float, default=0.002)
    parser.add_argument('--max-time-ratio', type=float, default=1.5)
    args = parser.parse_args()

    three = gather(args.config3)
    four = gather(args.config4)
    recall_gain = four['metrics']['recall@20'] - three['metrics']['recall@20']
    ndcg_gain = four['metrics']['ndcg@10'] - three['metrics']['ndcg@10']
    time_ratio = four['total_seconds'] / max(three['total_seconds'], 1e-12)
    utilization_ok_3 = three['mean_utilization'] >= float(args.min_utilization)
    utilization_ok_4 = four['mean_utilization'] >= float(args.min_utilization)
    utilization_ok = utilization_ok_3 and utilization_ok_4
    meaningful_gain = recall_gain >= float(args.min_recall_gain)
    time_ok = time_ratio <= float(args.max_time_ratio)
    recommended_levels = 4 if utilization_ok and meaningful_gain and time_ok else 3
    reasons = []
    if not utilization_ok_3:
        reasons.append(
            f"3-level mean utilization {three['mean_utilization']:.4f} < {args.min_utilization:.4f}"
        )
    if not utilization_ok_4:
        reasons.append(
            f"4-level mean utilization {four['mean_utilization']:.4f} < {args.min_utilization:.4f}"
        )
    if not meaningful_gain:
        reasons.append(
            f"Recall@20 gain {recall_gain:+.6f} < required {args.min_recall_gain:.6f}"
        )
    if not time_ok:
        reasons.append(f"4-level pipeline time ratio {time_ratio:.4f} > {args.max_time_ratio:.4f}")
    if not reasons:
        reasons.append('4 levels passed utilization, recall gain, and runtime constraints')

    comparison = {
        'three_levels': three,
        'four_levels': four,
        'delta': {
            'recall@20': recall_gain,
            'ndcg@10': ndcg_gain,
            'mean_utilization': four['mean_utilization'] - three['mean_utilization'],
            'base_collision_rate': four['base_collision_rate'] - three['base_collision_rate'],
            'total_seconds': four['total_seconds'] - three['total_seconds'],
            'time_ratio_4_over_3': time_ratio,
        },
        'decision': {
            'recommended_num_levels': recommended_levels,
            'utilization_ok': utilization_ok,
            'utilization_ok_3': utilization_ok_3,
            'utilization_ok_4': utilization_ok_4,
            'meaningful_recall_gain': meaningful_gain,
            'time_budget_ok': time_ok,
            'thresholds': {
                'min_utilization': float(args.min_utilization),
                'min_recall_gain': float(args.min_recall_gain),
                'max_time_ratio': float(args.max_time_ratio),
            },
            'reasons': reasons,
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

    headers = ['metric', '3_levels', '4_levels', 'delta/ratio']
    print('	'.join(headers))
    print('	'.join(['num_levels', str(three['num_levels']), str(four['num_levels']), '-']))
    print('	'.join(['mean_utilization', f"{three['mean_utilization']:.6f}", f"{four['mean_utilization']:.6f}", f"{four['mean_utilization'] - three['mean_utilization']:+.6f}"]))
    print('	'.join(['base_collision_rate', f"{three['base_collision_rate']:.6f}", f"{four['base_collision_rate']:.6f}", f"{four['base_collision_rate'] - three['base_collision_rate']:+.6f}"]))
    print('	'.join(['recall@20', f"{three['metrics']['recall@20']:.6f}", f"{four['metrics']['recall@20']:.6f}", f"{recall_gain:+.6f}"]))
    print('	'.join(['ndcg@10', f"{three['metrics']['ndcg@10']:.6f}", f"{four['metrics']['ndcg@10']:.6f}", f"{ndcg_gain:+.6f}"]))
    print('	'.join(['long_tail_recall@20', f"{three['metrics']['long_tail_recall@20']:.6f}", f"{four['metrics']['long_tail_recall@20']:.6f}", f"{four['metrics']['long_tail_recall@20'] - three['metrics']['long_tail_recall@20']:+.6f}"]))
    print('	'.join(['cold_start_recall@20', f"{three['metrics']['cold_start_recall@20']:.6f}", f"{four['metrics']['cold_start_recall@20']:.6f}", f"{four['metrics']['cold_start_recall@20'] - three['metrics']['cold_start_recall@20']:+.6f}"]))
    print('	'.join(['invalid_sid_rate', f"{three['metrics']['invalid_sid_rate']:.6f}", f"{four['metrics']['invalid_sid_rate']:.6f}", f"{four['metrics']['invalid_sid_rate'] - three['metrics']['invalid_sid_rate']:+.6f}"]))
    print('	'.join(['p99_ms', f"{three['metrics']['p99_ms']:.6f}", f"{four['metrics']['p99_ms']:.6f}", f"{four['metrics']['p99_ms'] / max(three['metrics']['p99_ms'], 1e-12):.6f}x"]))
    print('	'.join(['pipeline_seconds', f"{three['total_seconds']:.6f}", f"{four['total_seconds']:.6f}", f"{time_ratio:.6f}x"]))
    print(f"recommended_num_levels={recommended_levels}")
    print('reasons=' + '; '.join(reasons))


if __name__ == '__main__':
    main()
