"""Benchmark fixed-length and variable-length Semantic ID policies.

This script does not retrain the RQ-VAE. It reuses the base codes and top-k
codewords produced by ``build_semantic_ids.py`` and measures collision
resolution, trie construction, token growth, and estimated decoder/encoder
complexity for both SID length policies.
"""
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
from tiger_rec.semantic_id.build import build_trie_from_semantic_ids
from tiger_rec.semantic_id.collisions import resolve_semantic_id_collisions
from tiger_rec.semantic_id.token_space import TokenSpace
from tiger_rec.utils import write_json


def trie_node_count(node: Any) -> int:
    count = 0
    stack = [node]
    while stack:
        current = stack.pop()
        count += 1
        stack.extend(current.children.values())
    return count


def load_codes(cfg) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with cfg.paths.artifact("semantic_item_ids.json").open("r", encoding="utf-8") as f:
        item_ids = [str(value) for value in json.load(f)]
    base_codes = np.load(cfg.paths.artifact("semantic_base_codes.npy"), allow_pickle=False)
    topk_path = cfg.paths.artifact("semantic_topk_codes.npy")
    if not topk_path.exists():
        raise FileNotFoundError(
            f"missing {topk_path}; rerun `python scripts/build_semantic_ids.py --config ...` "
            "with the updated code"
        )
    topk_codes = np.load(topk_path, allow_pickle=False)
    table = pq.read_table(cfg.paths.artifact("catalog.parquet"), columns=["item_id", "train_count"])
    counts_by_item = {
        str(item_id): int(count)
        for item_id, count in zip(
            table.column("item_id").to_pylist(),
            table.column("train_count").to_pylist(),
        )
    }
    counts = np.asarray([counts_by_item[item_id] for item_id in item_ids], dtype=np.int64)
    return item_ids, base_codes, topk_codes, counts


def run_mode(
    mode: str,
    item_ids: list[str],
    base_codes: np.ndarray,
    topk_codes: np.ndarray,
    counts: np.ndarray,
    token_space,
    repeats: int,
) -> dict[str, Any]:
    resolution_times: list[float] = []
    trie_times: list[float] = []
    report = None
    assignments = None
    trie = None

    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        assignments, report = resolve_semantic_id_collisions(
            item_ids=item_ids,
            base_codes=base_codes,
            token_space=token_space,
            train_counts=counts,
            topk_codes=topk_codes,
            max_suffix_tokens=token_space.max_conflict_tokens,
            length_mode=mode,
        )
        resolution_times.append(time.perf_counter() - started)

        sid_map = {item_id: assignment.to_dict() for item_id, assignment in assignments.items()}
        started = time.perf_counter()
        trie = build_trie_from_semantic_ids(sid_map)
        trie_times.append(time.perf_counter() - started)

    if assignments is None or report is None or trie is None:
        raise RuntimeError("benchmark did not run")

    lengths = np.asarray([len(assignments[item_id].token_ids) for item_id in item_ids], dtype=np.int64)
    total_tokens = int(lengths.sum())
    suffix_fraction = float(report.suffix_resolved / max(report.items, 1))
    base_levels = int(token_space.num_levels)
    # The final +1 in the decoder estimate is EOS. Encoder history also contains
    # optional behavior/time tokens, so report both raw SID and including extras.
    behavior_tokens = 1 if token_space.use_behavior_tokens else 0
    time_tokens = 1 if token_space.use_time_buckets else 0
    mean_sid = float(lengths.mean()) if lengths.size else 0.0
    fixed_sid = float(base_levels + 1)
    variable_sid = float(base_levels + suffix_fraction)
    extra = behavior_tokens + time_tokens
    return {
        "length_mode": mode,
        "items": int(len(item_ids)),
        "resolution_seconds_mean": float(np.mean(resolution_times)),
        "trie_build_seconds_mean": float(np.mean(trie_times)),
        "collision_report": report.__dict__,
        "total_sid_tokens": total_tokens,
        "mean_sid_tokens_measured": mean_sid,
        "min_sid_tokens": int(lengths.min()) if lengths.size else 0,
        "max_sid_tokens": int(lengths.max()) if lengths.size else 0,
        "trie_nodes": trie_node_count(trie.root),
        "estimated_decode_steps": variable_sid if mode == "variable" else fixed_sid,
        "estimated_history_sid_tokens_per_item": mean_sid,
        "estimated_history_total_tokens_per_item": mean_sid + extra,
        "suffix_item_fraction": suffix_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--modes", nargs="+", choices=["variable", "fixed"], default=["variable", "fixed"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    item_ids, base_codes, topk_codes, counts = load_codes(cfg)
    token_space = TokenSpace(
        num_levels=int(cfg.sid.num_levels),
        codebook_size=int(cfg.sid.codebook_size),
        max_conflict_tokens=int(cfg.sid.max_conflict_tokens),
        use_behavior_tokens=bool(cfg.generator.use_behavior_tokens),
        use_time_buckets=bool(cfg.generator.use_time_buckets),
        num_time_buckets=int(cfg.generator.num_time_buckets),
    )

    results = {
        mode: run_mode(mode, item_ids, base_codes, topk_codes, counts, token_space, args.repeats)
        for mode in args.modes
    }
    comparison: dict[str, Any] = {}
    if "variable" in results and "fixed" in results:
        variable = results["variable"]
        fixed = results["fixed"]
        comparison = {
            "fixed_vs_variable_resolution_time": (
                fixed["resolution_seconds_mean"] / max(variable["resolution_seconds_mean"], 1e-12)
            ),
            "fixed_vs_variable_trie_build_time": (
                fixed["trie_build_seconds_mean"] / max(variable["trie_build_seconds_mean"], 1e-12)
            ),
            "fixed_vs_variable_sid_token_count": (
                fixed["total_sid_tokens"] / max(variable["total_sid_tokens"], 1)
            ),
            "fixed_vs_variable_decode_steps": (
                fixed["estimated_decode_steps"] / max(variable["estimated_decode_steps"], 1e-12)
            ),
            "fixed_vs_variable_history_encoder_tokens": (
                fixed["estimated_history_total_tokens_per_item"]
                / max(variable["estimated_history_total_tokens_per_item"], 1e-12)
            ),
        }
        ratio = comparison["fixed_vs_variable_history_encoder_tokens"]
        comparison["estimated_encoder_attention_work_ratio"] = ratio * ratio

    report = {
        "config": str(Path(args.config).resolve()),
        "repeats": int(args.repeats),
        "results": results,
        "comparison": comparison,
    }
    output = Path(args.output) if args.output else cfg.paths.artifact("sid_length_benchmark.json")
    write_json(output, report)

    print(f"{'mode':<10} {'resolve_s':>12} {'trie_s':>12} {'mean_len':>10} {'total_tokens':>14} {'trie_nodes':>12}")
    for mode, row in results.items():
        print(
            f"{mode:<10} {row['resolution_seconds_mean']:>12.6f} "
            f"{row['trie_build_seconds_mean']:>12.6f} "
            f"{row['mean_sid_tokens_measured']:>10.4f} "
            f"{row['total_sid_tokens']:>14,} "
            f"{row['trie_nodes']:>12,}"
        )
    if comparison:
        print("\nfixed / variable ratios:")
        for key, value in comparison.items():
            print(f"{key}: {value:.4f}x")
    print(f"\nbenchmark saved to {output}")


if __name__ == "__main__":
    main()
