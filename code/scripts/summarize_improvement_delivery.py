#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

ORDER = {
    'books_improve_00_content_baseline': 0,
    'books_improve_01_gate': 1,
    'books_improve_02_frequency': 2,
    'books_improve_03_usage': 3,
    'books_improve_04_hard_reset': 4,
    'books_improve_05_geometry': 5,
    'books_improve_06_collision_all': 6,
    'books_improve_07_min_vocab': 7,
    'books_improve_a1_cross_attention': 8,
    'books_improve_a2_frequency_strong': 9,
    'books_improve_a3_geometry5': 10,
    'books_improve_a4_kmeans_off': 11,
    'books_improve_x_text_hash': 12,
    'books_improve_p1_user_tokens': 13,
}


def read(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def timing_for(stem: str, logs: Path) -> dict[str, Any]:
    matches = sorted(logs.glob(f'*{stem}*.timing.json')) + sorted(logs.glob(f'*{stem}*.log.timing.json'))
    return read(matches[-1]) if matches else {}


def gather(root: Path, logs: Path) -> list[dict[str, Any]]:
    rows = []
    for directory in sorted(root.glob('books_improve_*')):
        sid_path = directory / 'semantic_id_report.json'
        if not sid_path.exists():
            continue
        sid = read(sid_path)
        gen = read(directory / 'generator_training.json')
        rq = read(directory / 'rqvae_training.json')
        trie = read(directory / 'metrics/tiger_trie_test.json')
        evaluation_beam = 5
        if directory.name == 'books_improve_07_min_vocab':
            beam5 = sorted(logs.glob('beam_sweep_5_*.json'))
            if beam5:
                trie = read(beam5[-1])
        if directory.name == 'books_improve_p1_user_tokens':
            seed_rows = [read(path) for path in sorted(logs.glob('user_seed_*_beam20_*.json'))]
            seed_rows = [row for row in seed_rows if row]
            if seed_rows:
                metric_keys = [
                    'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20', 'mrr@10', 'mrr@20',
                    'long_tail_recall@10', 'long_tail_recall@20',
                    'cold_start_recall@10', 'cold_start_recall@20',
                    'coverage', 'invalid_sid_rate', 'mean_ms', 'p50_ms', 'p95_ms', 'p99_ms',
                    'wall_time_sec', 'users',
                ]
                trie = {
                    key: sum(float(row[key]) for row in seed_rows) / len(seed_rows)
                    for key in metric_keys if all(key in row for row in seed_rows)
                }
                trie['seed_runs'] = len(seed_rows)
                evaluation_beam = 20
        standard = read(directory / 'metrics/tiger_standard_test.json')
        config = read(Path('configs') / f'{directory.name}.json')
        timing = timing_for(directory.name, logs)
        last_rq = (rq.get('history') or [{}])[-1]
        rows.append({
            'name': directory.name,
            'mean_utilization': sid.get('mean_utilization'),
            'base_collision_rate': sid.get('base_collision_rate'),
            'unique_base_codes': sid.get('unique_base_codes'),
            'semantic_reallocated': sid.get('semantic_reallocated'),
            'suffix_resolved': sid.get('suffix_resolved'),
            'max_suffix_used': sid.get('max_suffix_used'),
            'token_vocab_size': sid.get('token_vocab_size'),
            'mean_sid_length': sid.get('mean_sid_length'),
            'evaluation_beam': evaluation_beam,
            'levels': (config.get('sid') or {}).get('num_levels'),
            'codebook_size': (config.get('sid') or {}).get('codebook_size'),
            'latent_dim': (config.get('sid') or {}).get('latent_dim'),
            'fusion_type': (config.get('sid') or {}).get('fusion_type'),
            'reallocation_mode': (config.get('sid') or {}).get('conflict_reallocation_mode'),
            'inverse_frequency_power': (config.get('sid') or {}).get('inverse_frequency_power'),
            'configured_max_conflict_tokens': (config.get('sid') or {}).get('max_conflict_tokens'),
            'reconstruction': last_rq.get('reconstruction'),
            'best_valid_loss': gen.get('best_valid_loss'),
            'generator_epochs': len(gen.get('history', [])),
            'recall@20': trie.get('recall@20'),
            'ndcg@10': trie.get('ndcg@10'),
            'long_tail_recall@20': trie.get('long_tail_recall@20'),
            'cold_start_recall@20': trie.get('cold_start_recall@20'),
            'coverage': trie.get('coverage'),
            'invalid_sid_rate': trie.get('invalid_sid_rate'),
            'p99_ms': trie.get('p99_ms'),
            'standard_recall@20': standard.get('recall@20'),
            'standard_invalid_sid_rate': standard.get('invalid_sid_rate'),
            'standard_p99_ms': standard.get('p99_ms'),
            'total_seconds': timing.get('total_seconds'),
            'sid_seconds': timing.get('sid_seconds'),
            'generator_seconds': timing.get('generator_seconds'),
            'evaluation_seconds': timing.get('evaluation_seconds'),
        })
    return sorted(rows, key=lambda row: ORDER.get(row['name'], 999))


def fmt(value: Any, digits: int = 4, percent: bool = False) -> str:
    if value is None:
        return 'n/a'
    number = float(value) * (100.0 if percent else 1.0)
    return f'{number:.{digits}f}' + ('%' if percent else '')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/improvement_suite')
    parser.add_argument('--logs', default='results/improvement_suite')
    parser.add_argument('--output-json', default='results/improvement_suite/delivery_summary.json')
    parser.add_argument('--output-md', default='docs/tiger_improvement_results.md')
    args = parser.parse_args()
    rows = gather(Path(args.root), Path(args.logs))
    Path(args.output_json).write_text(json.dumps({'candidates': rows, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [
        '# TIGER quick-data 改进结果表', '',
        '| 配置 | levels/K/latent | util | collision | suffix | vocab | SID len | valid loss | Recall@20 | NDCG@10 | long-tail | cold-start | coverage | invalid | P99 ms | total s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        name = row['name'].replace('books_improve_', '')
        lines.append('| ' + ' | '.join([
            name,
            f"{row['levels']}/{row['codebook_size']}/{row['latent_dim']}",
            fmt(row['mean_utilization'], percent=True),
            fmt(row['base_collision_rate'], percent=True),
            str(row['suffix_resolved']),
            str(row['token_vocab_size']),
            fmt(row['mean_sid_length'], 2),
            fmt(row['best_valid_loss']),
            fmt(row['recall@20'], 6),
            fmt(row['ndcg@10'], 6),
            fmt(row['long_tail_recall@20'], 6),
            fmt(row['cold_start_recall@20'], 6),
            fmt(row['coverage'], 6),
            fmt(row['invalid_sid_rate'], 6),
            fmt(row['p99_ms'], 2),
            fmt(row['total_seconds'], 1),
        ]) + ' |')
    Path(args.output_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
