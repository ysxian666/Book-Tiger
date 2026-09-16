#!/usr/bin/env python3
"""Choose max_conflict_tokens from actual collision-group requirements."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiger_rec.config import load_config
from tiger_rec.semantic_id.collisions import resolve_semantic_id_collisions
from tiger_rec.semantic_id.token_space import TokenSpace
from tiger_rec.utils import write_json


def load_inputs(cfg):
    item_ids_path = cfg.paths.artifact('semantic_item_ids.json')
    base_path = cfg.paths.artifact('semantic_base_codes.npy')
    topk_path = cfg.paths.artifact('semantic_topk_codes.npy')
    for path in (item_ids_path, base_path, topk_path):
        if not path.exists():
            raise FileNotFoundError(
                f'missing {path}; run `python scripts/build_semantic_ids.py '
                f'--config {cfg.paths.artifact("run_config.json")}` first'
            )
    with item_ids_path.open('r', encoding='utf-8') as f:
        item_ids = [str(value) for value in json.load(f)]
    base_codes = np.load(base_path, allow_pickle=False)
    topk_codes = np.load(topk_path, allow_pickle=False)
    table = pq.read_table(cfg.paths.artifact('catalog.parquet'), columns=['item_id', 'train_count'])
    counts_by_item = {
        str(item_id): int(count)
        for item_id, count in zip(
            table.column('item_id').to_pylist(),
            table.column('train_count').to_pylist(),
        )
    }
    counts = np.asarray([counts_by_item[item_id] for item_id in item_ids], dtype=np.int64)
    return item_ids, base_codes, topk_codes, counts


def next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (int(value) - 1).bit_length()


def evaluate(cap: int, cfg, item_ids, base_codes, topk_codes, counts) -> dict[str, Any]:
    token_space = TokenSpace(
        num_levels=int(cfg.sid.num_levels),
        codebook_size=int(cfg.sid.codebook_size),
        max_conflict_tokens=int(cap),
        use_behavior_tokens=bool(cfg.generator.use_behavior_tokens),
        use_time_buckets=bool(cfg.generator.use_time_buckets),
        num_time_buckets=int(cfg.generator.num_time_buckets),
    )
    started = time.perf_counter()
    try:
        assignments, report = resolve_semantic_id_collisions(
            item_ids=item_ids,
            base_codes=base_codes,
            token_space=token_space,
            train_counts=counts,
            topk_codes=topk_codes,
            max_suffix_tokens=cap,
            length_mode=cfg.sid.length_mode,
        )
    except RuntimeError as exc:
        return {
            'max_conflict_tokens': int(cap),
            'success': False,
            'elapsed_seconds': float(time.perf_counter() - started),
            'token_vocab_size': int(token_space.vocab_size),
            'error': str(exc),
        }
    return {
        'max_conflict_tokens': int(cap),
        'success': True,
        'elapsed_seconds': float(time.perf_counter() - started),
        'token_vocab_size': int(token_space.vocab_size),
        'max_suffix_used': int(report.max_suffix_used),
        'required_suffix_slots': int(report.max_suffix_used + 1),
        'semantic_reallocated': int(report.semantic_reallocated),
        'suffix_resolved': int(report.suffix_resolved),
        'base_collision_rate': float(report.base_collision_rate),
        'mean_sid_length': float(report.mean_sid_length),
        'unique_sids': int(len({tuple(row.token_ids) for row in assignments.values()})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidates', nargs='+', type=int, default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument('--safety-factor', type=float, default=2.0)
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    cfg = load_config(args.config)
    item_ids, base_codes, topk_codes, counts = load_inputs(cfg)
    candidates = sorted(set(int(value) for value in args.candidates))
    # The maximum possible collision group after anchoring is bounded by item count.
    candidates = sorted(set(candidates + [len(item_ids) + 1]))
    results = [evaluate(cap, cfg, item_ids, base_codes, topk_codes, counts) for cap in candidates]
    successful = [row for row in results if row['success']]
    if not successful:
        raise SystemExit('no tested max_conflict_tokens value can resolve all collisions')

    required = max(int(row['required_suffix_slots']) for row in successful)
    recommended = next_power_of_two(max(1, int(np.ceil(required * float(args.safety_factor)))))
    selected = min((row for row in successful if row['max_conflict_tokens'] >= required), key=lambda row: row['max_conflict_tokens'])
    report = {
        'config': str(Path(args.config).resolve()),
        'num_levels': int(cfg.sid.num_levels),
        'length_mode': cfg.sid.length_mode,
        'items': len(item_ids),
        'required_suffix_slots': required,
        'safety_factor': float(args.safety_factor),
        'recommended_max_conflict_tokens': recommended,
        'smallest_tested_success': selected,
        'candidates': results,
    }
    output = Path(args.output) if args.output else cfg.paths.artifact('max_conflict_tuning.json')
    write_json(output, report)

    print(f'items={len(item_ids)} num_levels={cfg.sid.num_levels} length_mode={cfg.sid.length_mode}')
    print(f'required_suffix_slots={required} safety_factor={args.safety_factor}')
    print(f'recommended_max_conflict_tokens={recommended}')
    for row in results:
        if row['success']:
            print(
                f"cap={row['max_conflict_tokens']:>6} ok "
                f"max_suffix_used={row['max_suffix_used']:>5} "
                f"reallocated={row['semantic_reallocated']:>6} "
                f"suffix={row['suffix_resolved']:>6} "
                f"vocab={row['token_vocab_size']:>6} "
                f"time={row['elapsed_seconds']:.4f}s"
            )
        else:
            print(f"cap={row['max_conflict_tokens']:>6} failed")
    print(f'saved={output}')


if __name__ == '__main__':
    main()
