#!/usr/bin/env python3
"""Summarize TIGER semantic-ID and generator candidate metrics."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiger_rec.config import load_config


def read_json(path: Path, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def gather(name: str, metrics_suffix: str = '') -> dict[str, Any]:
    cfg = load_config(name)
    sid = read_json(cfg.paths.artifact('semantic_id_report.json'))
    train = read_json(cfg.paths.artifact('generator_training.json'))
    trie = read_json(cfg.paths.artifact(f'metrics/tiger_trie_test{metrics_suffix}.json'))
    standard = read_json(cfg.paths.artifact(f'metrics/tiger_standard_test{metrics_suffix}.json'))
    return {
        'name': Path(name).stem,
        'success': bool(sid and trie),
        'levels': int(cfg.sid.num_levels),
        'codebook_size': int(cfg.sid.codebook_size),
        'latent_dim': int(cfg.sid.latent_dim),
        'reallocation_mode': str(cfg.sid.conflict_reallocation_mode),
        'items': int(sid.get('items', 0)),
        'mean_utilization': float(sid.get('mean_utilization', 0.0)),
        'base_collision_rate': float(sid.get('base_collision_rate', 0.0)),
        'unique_base_code_ratio': float(sid.get('unique_base_codes', 0)) / max(int(sid.get('items', 0)), 1),
        'semantic_reallocated_ratio': float(sid.get('semantic_reallocated', 0)) / max(int(sid.get('items', 0)), 1),
        'suffix_resolved': int(sid.get('suffix_resolved', 0)),
        'mean_sid_length': float(sid.get('mean_sid_length', 0.0)),
        'token_vocab_size': int(sid.get('token_vocab_size', 0)),
        'minimized_conflict_tokens': int(sid.get('token_space', {}).get('max_conflict_tokens', cfg.sid.max_conflict_tokens)),
        'best_valid_loss': train.get('best_valid_loss'),
        'recall@20': trie.get('recall@20'),
        'ndcg@10': trie.get('ndcg@10'),
        'ndcg@20': trie.get('ndcg@20'),
        'coverage': trie.get('coverage'),
        'invalid_sid_rate': trie.get('invalid_sid_rate'),
        'p99_ms': trie.get('p99_ms'),
        'standard_recall@20': standard.get('recall@20'),
        'standard_ndcg@10': standard.get('ndcg@10'),
        'standard_invalid_sid_rate': standard.get('invalid_sid_rate'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--metrics-suffix', default='')
    args = parser.parse_args()
    rows = [gather(name, args.metrics_suffix) for name in args.configs]
    rows.sort(key=lambda row: (-(float(row.get('recall@20') or 0.0)), -float(row.get('ndcg@10') or 0.0)))
    payload = {'candidates': rows, 'best_recall20': rows[0]['name'] if rows and rows[0].get('recall@20') is not None else None}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('config\tlevels\tK\tlatent\tutil\tcollision\tSIDlen\tvars\trealloc\tvalid_loss\tRecall@20\tNDCG@10\tinvalid\tp99_ms')
    for row in rows:
        print('\t'.join([
            row['name'], str(row['levels']), str(row['codebook_size']), str(row['latent_dim']),
            f"{row['mean_utilization']:.4f}", f"{row['base_collision_rate']:.4f}",
            f"{row['mean_sid_length']:.2f}", str(row['token_vocab_size']), f"{row['semantic_reallocated_ratio']:.4f}",
            'n/a' if row['best_valid_loss'] is None else f"{float(row['best_valid_loss']):.4f}",
            'n/a' if row['recall@20'] is None else f"{float(row['recall@20']):.6f}",
            'n/a' if row['ndcg@10'] is None else f"{float(row['ndcg@10']):.6f}",
            'n/a' if row['invalid_sid_rate'] is None else f"{float(row['invalid_sid_rate']):.6f}",
            'n/a' if row['p99_ms'] is None else f"{float(row['p99_ms']):.2f}",
        ]))
    print(f'saved={output}')


if __name__ == '__main__':
    main()
