#!/usr/bin/env python3
"""Summarize RQ-VAE codebook sweeps with collapse and collision diagnostics."""
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
    report = read_json(cfg.paths.artifact('semantic_id_report.json'), required=False)
    timing = read_json(cfg.paths.artifact('pipeline_timing.json'), required=False)
    if not report:
        return {
            'name': Path(config_path).stem,
            'config': str(Path(config_path).resolve()),
            'success': False,
            'error': 'semantic_id_report.json missing',
        }

    stage_seconds = {row['stage']: float(row['seconds']) for row in timing.get('timings', [])}
    items = int(report.get('items', 0))
    unique_base_codes = int(report.get('unique_base_codes', 0))
    suffix_resolved = int(report.get('suffix_resolved', 0))
    semantic_reallocated = int(report.get('semantic_reallocated', 0))
    collision = float(report.get('base_collision_rate', 0.0))
    utilization = float(report.get('mean_utilization', 0.0))
    max_suffix_used = int(report.get('max_suffix_used', -1))

    denominator = max(items, 1)
    row = {
        'name': Path(config_path).stem,
        'config': str(Path(config_path).resolve()),
        'success': True,
        'num_levels': int(cfg.sid.num_levels),
        'codebook_size': int(cfg.sid.codebook_size),
        'hidden_dim': int(cfg.sid.hidden_dim),
        'latent_dim': int(cfg.sid.latent_dim),
        'epochs': int(cfg.sid.epochs),
        'length_mode': str(cfg.sid.length_mode),
        'conflict_reallocation_mode': str(cfg.sid.conflict_reallocation_mode),
        'usage_weight': float(cfg.sid.usage_weight),
        'entropy_weight': float(cfg.sid.entropy_weight),
        'temperature': float(cfg.sid.temperature),
        'hard_usage_weight': float(cfg.sid.hard_usage_weight),
        'joint_collision_weight': float(cfg.sid.joint_collision_weight),
        'dead_code_reset': bool(cfg.sid.dead_code_reset),
        'dead_code_reset_interval': int(cfg.sid.dead_code_reset_interval),
        'dead_code_threshold': float(cfg.sid.dead_code_threshold),
        'usage_ema_decay': float(cfg.sid.usage_ema_decay),
        'items': items,
        'unique_base_codes': unique_base_codes,
        'duplicate_items': int(report.get('duplicate_items', 0)),
        'unique_base_code_ratio': unique_base_codes / denominator,
        'mean_utilization': utilization,
        'mean_perplexity': float(report.get('mean_perplexity', 0.0)),
        'base_collision_rate': collision,
        'resolved_collision_rate': 0.0,
        'semantic_reallocated': semantic_reallocated,
        'semantic_reallocated_ratio': semantic_reallocated / denominator,
        'suffix_resolved': suffix_resolved,
        'suffix_resolved_ratio': suffix_resolved / denominator,
        'max_suffix_used': max_suffix_used,
        'required_suffix_slots': max_suffix_used + 1 if max_suffix_used >= 0 else 0,
        'mean_sid_length': float(report.get('mean_sid_length', 0.0)),
        'sid_seconds': stage_seconds.get('sid'),
        'utilization': {
            key: float(value)
            for key, value in report.items()
            if key.startswith('utilization_level_')
        },
        'perplexity': {
            key: float(value)
            for key, value in report.items()
            if key.startswith('perplexity_level_')
        },
    }
    row['balanced_loss'] = 0.5 * collision + 0.5 * (1.0 - utilization)
    row['code_health'] = utilization * (1.0 - collision)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    rows = [gather(path) for path in args.configs]
    successful = [row for row in rows if row.get('success')]
    by_utilization = sorted(
        successful,
        key=lambda row: (
            -row['mean_utilization'],
            -row['unique_base_code_ratio'],
            row['base_collision_rate'],
        ),
    )
    by_collision = sorted(
        successful,
        key=lambda row: (
            row['base_collision_rate'],
            -row['mean_utilization'],
            row['suffix_resolved_ratio'],
        ),
    )
    by_balance = sorted(
        successful,
        key=lambda row: (
            row['balanced_loss'],
            row['base_collision_rate'],
            -row['mean_utilization'],
        ),
    )
    summary = {
        'candidates': rows,
        'successful': successful,
        'best_by_utilization': by_utilization[0]['name'] if by_utilization else None,
        'best_by_collision': by_collision[0]['name'] if by_collision else None,
        'best_by_balance': by_balance[0]['name'] if by_balance else None,
        'rankings': {
            'collision_first': [row['name'] for row in by_collision],
            'utilization_first': [row['name'] for row in by_utilization],
            'balanced_loss': [row['name'] for row in by_balance],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    headers = [
        'config', 'levels', 'codebook', 'latent', 'epochs', 'hard_usage',
        'collision_w', 'realloc', 'reset', 'ema', 'dead_thr', 'util', 'ppl', 'unique', 'collision',
        'reallocated', 'suffix', 'max_suffix', 'mean_len', 'sid_sec', 'health',
    ]
    print('\t'.join(headers))
    for row in by_collision:
        print('\t'.join([
            row['name'],
            str(row['num_levels']),
            str(row['codebook_size']),
            str(row['latent_dim']),
            str(row['epochs']),
            f"{row['hard_usage_weight']:.2f}",
            f"{row['joint_collision_weight']:.4f}",
            str(row['conflict_reallocation_mode']),
            f"{row['dead_code_reset_interval']}",
            f"{row['usage_ema_decay']:.2f}",
            f"{row['dead_code_threshold']:.3f}",
            f"{row['mean_utilization']:.4f}",
            f"{row['mean_perplexity']:.4f}",
            f"{row['unique_base_code_ratio']:.4f}",
            f"{row['base_collision_rate']:.4f}",
            str(row['semantic_reallocated']),
            str(row['suffix_resolved']),
            str(row['max_suffix_used']),
            f"{row['mean_sid_length']:.3f}",
            'n/a' if row['sid_seconds'] is None else f"{row['sid_seconds']:.2f}",
            f"{row['code_health']:.4f}",
        ]))
    print(f"best_by_collision={by_collision[0]['name'] if by_collision else None}")
    print(f"best_by_utilization={by_utilization[0]['name'] if by_utilization else None}")
    print(f"best_by_balance={by_balance[0]['name'] if by_balance else None}")
    print(f'saved={output}')


if __name__ == '__main__':
    main()
