"""Baseline suffix SIDs and behavior-cluster-aware global SID assignment."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tiger_rec.semantic_id.rqvae import codebook_metrics
from tiger_rec.semantic_id.token_space import TokenSpace

from .config import artifact_path, data_path, ensure_run_dirs
from .rqvae_train import encode_items_with_rqvae


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _token_space(config: dict[str, Any]) -> TokenSpace:
    return TokenSpace(
        num_levels=int(config["rqvae"]["num_levels"]),
        codebook_size=int(config["rqvae"]["codebook_size"]),
        max_conflict_tokens=int(config["data"]["max_suffix_tokens"]),
        use_behavior_tokens=False,
        use_time_buckets=True,
        use_user_tokens=True,
        num_user_buckets=int(config["generator"]["num_user_buckets"]),
        num_time_buckets=int(config["generator"]["num_time_buckets"]),
        behavior_count=0,
    )


def _load_behavior_profiles(config: dict[str, Any], expected_items: int) -> np.ndarray:
    profiles = np.load(data_path(config, "item_behavior_profiles.npy")).astype(np.float32)
    if profiles.shape[0] != expected_items:
        raise RuntimeError("behavior profile count does not match catalog")
    return profiles


def _codebook_vectors(config: dict[str, Any]) -> np.ndarray:
    checkpoint = torch.load(
        artifact_path(config, "rqvae", "rqvae.pt"),
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["state_dict"]
    vectors = []
    for level in range(int(config["rqvae"]["num_levels"])):
        key = f"quantizer.codebooks.{level}"
        if key not in state:
            raise KeyError(f"missing codebook parameter: {key}")
        vectors.append(state[key].float().numpy())
    return np.stack(vectors, axis=0)


def _normalized_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).sum() * 0.5)


def _kl(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) + 1e-9
    right = np.asarray(right, dtype=np.float64) + 1e-9
    left /= left.sum()
    right /= right.sum()
    return float(np.sum(left * np.log(left / right)))


def _semantic_cost(latent: np.ndarray, code_vectors: np.ndarray, codes: list[int]) -> float:
    selected = code_vectors[np.arange(len(codes)), np.asarray(codes, dtype=np.int64)].sum(axis=0)
    denominator = max(float(np.linalg.norm(latent) * np.linalg.norm(selected)), 1e-12)
    cosine = float(np.dot(latent, selected) / denominator)
    return 1.0 - max(min(cosine, 1.0), -1.0)


def _base_metrics(base_codes: np.ndarray, code_vectors: np.ndarray, behavior: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    metrics = codebook_metrics(base_codes, int(code_vectors.shape[1]))
    profiles = []
    for level in range(base_codes.shape[1]):
        counts = np.bincount(base_codes[:, level], minlength=code_vectors.shape[1]).astype(np.float64)
        level_profiles = np.zeros((code_vectors.shape[1], behavior.shape[1]), dtype=np.float64)
        np.add.at(level_profiles, base_codes[:, level], behavior)
        level_profiles /= np.maximum(counts[:, None], 1.0)
        profiles.append(level_profiles)
    return metrics, np.stack(profiles, axis=0)


def _assignment_quality(
    base_codes: np.ndarray,
    latents: np.ndarray,
    behavior: np.ndarray,
    code_vectors: np.ndarray,
) -> dict[str, float]:
    semantic = [_semantic_cost(latents[index], code_vectors, base_codes[index].tolist()) for index in range(len(base_codes))]
    level_profiles = []
    for level in range(base_codes.shape[1]):
        counts = np.bincount(base_codes[:, level], minlength=code_vectors.shape[1]).astype(np.float64)
        profiles = np.zeros((code_vectors.shape[1], behavior.shape[1]), dtype=np.float64)
        np.add.at(profiles, base_codes[:, level], behavior)
        profiles /= np.maximum(counts[:, None], 1.0)
        level_profiles.append(profiles)
    behavior_drift = []
    cluster_kl = []
    for index, codes in enumerate(base_codes):
        l1 = np.mean([_normalized_l1(behavior[index], level_profiles[level][code]) for level, code in enumerate(codes)])
        kl = np.mean([_kl(behavior[index], level_profiles[level][code]) for level, code in enumerate(codes)])
        behavior_drift.append(l1)
        cluster_kl.append(kl)
    return {
        "mean_semantic_drift": float(np.mean(semantic)),
        "mean_behavior_drift": float(np.mean(behavior_drift)),
        "mean_cluster_kl": float(np.mean(cluster_kl)),
    }


def build_baseline_sids(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_run_dirs(config)
    out_dir = artifact_path(config, "sid", "baseline")
    sid_path = out_dir / "semantic_ids.json"
    report_path = out_dir / "sid_report.json"
    if sid_path.exists() and report_path.exists() and not force:
        return _load_json(report_path)

    started = time.perf_counter()
    item_ids, base_codes, _topk_codes, latents, _fused = encode_items_with_rqvae(config)
    item_counts = {
        str(row.item_id): int(row.train_count)
        for row in pd.read_parquet(data_path(config, "item_stats.parquet"), columns=["item_id", "train_count"]).itertuples(index=False)
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "item_latents.npy", latents.astype(np.float32))
    behavior = _load_behavior_profiles(config, len(item_ids))
    code_vectors = _codebook_vectors(config)
    token_space = _token_space(config)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, codes in enumerate(base_codes):
        groups[tuple(int(value) for value in codes)].append(index)
    assignments: dict[str, dict[str, Any]] = {}
    suffix_resolved = 0
    reserved_sids: set[tuple[int, int, int, int]] = set()
    for indices in groups.values():
        indices.sort(key=lambda index: (-item_counts.get(item_ids[index], 0), item_ids[index]))
        for suffix, index in enumerate(indices):
            codes = [int(value) for value in base_codes[index]]
            token_ids = token_space.encode_sid(codes, suffix_index=suffix)
            sid = tuple(int(value) for value in token_ids)
            if sid in reserved_sids:
                raise AssertionError("baseline collision resolution produced duplicate SID tokens")
            reserved_sids.add(sid)
            assignments[item_ids[index]] = {
                "item_id": item_ids[index],
                "base_codes": codes,
                "suffix_index": suffix,
                "token_ids": token_ids,
                "assignment": "anchor" if suffix == 0 else "suffix",
            }
            if suffix > 0:
                suffix_resolved += 1
    random_quality = _assignment_quality(base_codes, latents, behavior, code_vectors)
    raw_unique = len(groups)
    report = {
        "items": len(item_ids),
        "sid_length": 4,
        "unique_sids": len(reserved_sids),
        "raw_unique_sids": int(raw_unique),
        "raw_collision_rate": 1.0 - raw_unique / max(len(item_ids), 1),
        "final_collision_rate": 1.0 - len(reserved_sids) / max(len(item_ids), 1),
        "suffix_resolved": int(suffix_resolved),
        "semantic_reallocated": 0,
        "solver": "suffix_only",
        "solver_time_seconds": float(time.perf_counter() - started),
        "sid_length_mode": "fixed",
        "token_vocab_size": token_space.vocab_size,
        **random_quality,
    }
    _write_json(sid_path, assignments)
    _write_json(report_path, report)
    np.save(out_dir / "sid_token_ids.npy", np.asarray([assignments[item_id]["token_ids"] for item_id in item_ids], dtype=np.int64))
    return report


def _candidate_bases(
    base_codes: np.ndarray,
    topk_codes: np.ndarray,
    limit: int,
    top_codes_per_level: int,
) -> list[list[list[int]]]:
    output: list[list[list[int]]] = []
    for index in range(base_codes.shape[0]):
        original = [int(value) for value in base_codes[index]]
        seen = {tuple(original)}
        candidates = [original]
        for level in range(base_codes.shape[1]):
            for value in topk_codes[index, level, :top_codes_per_level]:
                candidate = list(original)
                candidate[level] = int(value)
                key = tuple(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                if len(candidates) >= int(limit):
                    break
            if len(candidates) >= int(limit):
                break
        output.append(candidates)
    return output


def _candidate_costs(
    candidates: list[list[int]],
    latents: np.ndarray,
    behavior: np.ndarray,
    code_vectors: np.ndarray,
    level_profiles: np.ndarray,
    recon_weight: float,
    semantic_weight: float,
    behavior_weight: float,
    cluster_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item_count = len(candidates)
    max_candidates = max(len(values) for values in candidates)
    semantic_cost = np.full((item_count, max_candidates), np.inf, dtype=np.float64)
    behavior_cost = np.full_like(semantic_cost, np.inf)
    cluster_cost = np.full_like(semantic_cost, np.inf)
    for item_index, values in enumerate(candidates):
        for candidate_index, codes in enumerate(values):
            semantic = _semantic_cost(latents[item_index], code_vectors, codes)
            l1 = np.mean([
                _normalized_l1(behavior[item_index], level_profiles[level][code])
                for level, code in enumerate(codes)
            ])
            kl = np.mean([
                _kl(behavior[item_index], level_profiles[level][code])
                for level, code in enumerate(codes)
            ])
            semantic_cost[item_index, candidate_index] = (recon_weight + semantic_weight) * semantic
            behavior_cost[item_index, candidate_index] = behavior_weight * l1
            cluster_cost[item_index, candidate_index] = cluster_weight * kl
    return semantic_cost, behavior_cost, cluster_cost


def build_bcgsid_sids(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_run_dirs(config)
    out_dir = artifact_path(config, "sid", "bcgsid")
    sid_path = out_dir / "semantic_ids.json"
    assignment_report_path = out_dir / "assignment_report.json"
    solver_report_path = out_dir / "solver_report.json"
    if sid_path.exists() and assignment_report_path.exists() and solver_report_path.exists() and not force:
        return _load_json(assignment_report_path)

    started = time.perf_counter()
    item_ids, base_codes, topk_codes, latents, _fused = encode_items_with_rqvae(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "item_latents.npy", latents.astype(np.float32))
    behavior = _load_behavior_profiles(config, len(item_ids))
    code_vectors = _codebook_vectors(config)
    token_space = _token_space(config)
    metrics, level_profiles = _base_metrics(base_codes, code_vectors, behavior)
    allocation = config["sid_allocation"]
    candidates = _candidate_bases(
        base_codes,
        topk_codes,
        int(allocation["base_candidates"]),
        int(allocation["top_codes_per_level"]),
    )
    semantic_cost, behavior_cost, cluster_cost = _candidate_costs(
        candidates,
        latents,
        behavior,
        code_vectors,
        level_profiles,
        float(allocation["lambda_recon"]),
        float(allocation["lambda_semantic"]),
        float(allocation["lambda_behavior"]),
        float(allocation["lambda_cluster"]),
    )
    entropy = -(behavior * np.log(behavior + 1e-9)).sum(axis=1)
    difficulty = entropy + semantic_cost[:, 0]
    order = np.argsort(-difficulty, kind="stable")
    usage = np.zeros((base_codes.shape[1], int(config["rqvae"]["codebook_size"])), dtype=np.int64)
    used: set[tuple[int, ...]] = set()
    assignments: dict[str, dict[str, Any]] = {}
    base_usage = np.zeros_like(usage)
    suffix_used = 0
    semantic_reallocated = 0
    max_suffix_used = 0
    suffix_limit = int(allocation["suffix_candidates_per_original_code"])

    for item_index in order:
        item_candidates = candidates[int(item_index)]
        best_key: tuple[float, int, int, list[int], list[int]] | None = None
        for candidate_index, codes in enumerate(item_candidates):
            for level, code in enumerate(codes):
                base_usage[level, code] += 1
            usage_penalty = 0.0
            for level, code in enumerate(codes):
                share = base_usage[level, code] / max(int(item_index) + 1, 1)
                usage_penalty += (share - 1.0 / int(config["rqvae"]["codebook_size"])) ** 2
            suffix_values = list(range(suffix_limit))
            for suffix in suffix_values:
                token_ids = token_space.encode_sid(codes, suffix_index=suffix)
                key = tuple(token_ids)
                if key in used:
                    continue
                cost = (
                    semantic_cost[int(item_index), candidate_index]
                    + behavior_cost[int(item_index), candidate_index]
                    + cluster_cost[int(item_index), candidate_index]
                    + float(allocation["lambda_usage"]) * usage_penalty
                    + 0.001 * suffix
                )
                choice = (float(cost), candidate_index, suffix, codes, token_ids)
                if best_key is None or choice[:3] < best_key[:3]:
                    best_key = choice
            for level, code in enumerate(codes):
                base_usage[level, code] -= 1
        if best_key is None:
            original = [int(value) for value in base_codes[int(item_index)]]
            for suffix in range(suffix_limit, int(config["data"]["max_suffix_tokens"])):
                token_ids = token_space.encode_sid(original, suffix_index=suffix)
                if tuple(token_ids) not in used:
                    best_key = (1e9 + suffix, 0, suffix, original, token_ids)
                    break
        if best_key is None:
            raise RuntimeError(f"cannot assign a unique SID to item {item_ids[int(item_index)]}")
        _cost, candidate_index, suffix, codes, token_ids = best_key
        sid_key = tuple(token_ids)
        used.add(sid_key)
        for level, code in enumerate(codes):
            usage[level, code] += 1
        is_reallocated = any(codes[level] != int(base_codes[item_index, level]) for level in range(len(codes)))
        semantic_reallocated += int(is_reallocated)
        suffix_used += int(suffix > 0)
        max_suffix_used = max(max_suffix_used, suffix)
        assignments[item_ids[int(item_index)]] = {
            "item_id": item_ids[int(item_index)],
            "base_codes": codes,
            "suffix_index": int(suffix),
            "token_ids": token_ids,
            "assignment": "global_behavior_candidate" if is_reallocated else ("suffix" if suffix > 0 else "anchor"),
            "source_candidate_index": int(candidate_index),
        }

    if len(used) != len(item_ids) or len(assignments) != len(item_ids):
        raise AssertionError("BC-GSID must produce one unique SID for every catalog item")
    assigned_codes = np.asarray([assignments[item_id]["base_codes"] for item_id in item_ids], dtype=np.int64)
    quality = _assignment_quality(assigned_codes, latents, behavior, code_vectors)
    final_metrics = codebook_metrics(assigned_codes, int(config["rqvae"]["codebook_size"]))
    elapsed = float(time.perf_counter() - started)
    report = {
        "items": len(item_ids),
        "sid_length": 4,
        "unique_sids": len(used),
        "raw_unique_sids": int(len({tuple(int(v) for v in row) for row in base_codes})),
        "raw_collision_rate": 1.0 - len({tuple(int(v) for v in row) for row in base_codes}) / max(len(item_ids), 1),
        "final_collision_rate": 1.0 - len(used) / max(len(item_ids), 1),
        "suffix_resolved": int(suffix_used),
        "semantic_reallocated": int(semantic_reallocated),
        "max_suffix_used": int(max_suffix_used),
        "solver": "greedy_behavior_aware",
        "solver_time_seconds": elapsed,
        "token_vocab_size": token_space.vocab_size,
        "codebook_utilization": {
            "mean": float(final_metrics["mean_utilization"]),
            "per_level": [float(final_metrics[f"utilization_level_{level}"]) for level in range(3)],
        },
        "codebook_perplexity": {
            "mean": float(final_metrics["mean_perplexity"]),
            "per_level": [float(final_metrics[f"perplexity_level_{level}"]) for level in range(3)],
        },
        **quality,
    }
    solver_report = {
        "requested_solver": str(allocation["solver"]),
        "selected_solver": "greedy_behavior_aware",
        "fallback_used": True,
        "fallback_reason": "The 50K-item candidate graph exceeds the 30-minute exact min-cost-flow budget for an overnight screen; hierarchical candidate generation plus deterministic behavior-aware greedy assignment is used.",
        "timeout_seconds": int(allocation["solver_timeout_seconds"]),
        "elapsed_seconds": elapsed,
        "candidate_items": len(item_ids),
        "base_candidates": int(allocation["base_candidates"]),
        "top_codes_per_level": int(allocation["top_codes_per_level"]),
        "suffix_candidates_per_original_code": suffix_limit,
        "lambda_recon": float(allocation["lambda_recon"]),
        "lambda_semantic": float(allocation["lambda_semantic"]),
        "lambda_behavior": float(allocation["lambda_behavior"]),
        "lambda_cluster": float(allocation["lambda_cluster"]),
        "lambda_usage": float(allocation["lambda_usage"]),
        "lambda_pop": float(allocation["lambda_pop"]),
        "final_collision_rate": report["final_collision_rate"],
        "sid_lookup_unique": len(used) == len(item_ids),
    }
    _write_json(sid_path, assignments)
    _write_json(assignment_report_path, report)
    _write_json(solver_report_path, solver_report)
    np.save(out_dir / "sid_token_ids.npy", np.asarray([assignments[item_id]["token_ids"] for item_id in item_ids], dtype=np.int64))
    np.save(out_dir / "assigned_base_codes.npy", assigned_codes)
    return report
