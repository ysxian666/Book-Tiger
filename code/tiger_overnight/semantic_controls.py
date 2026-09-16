"""Shuffled-behavior and hard-semantic controls for BC-GSID."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from tiger_rec.semantic_id.rqvae import codebook_metrics
from tiger_rec.semantic_id.token_space import TokenSpace

from .config import artifact_path, ensure_run_dirs
from .rqvae_train import encode_items_with_rqvae
from .semantic_ids import (
    _assignment_quality,
    _base_metrics,
    _candidate_bases,
    _codebook_vectors,
    _kl,
    _load_behavior_profiles,
    _normalized_l1,
    _semantic_cost,
    _write_json,
)


def _raw_semantic_costs(
    candidates: list[list[list[int]]],
    latents: np.ndarray,
    code_vectors: np.ndarray,
) -> np.ndarray:
    max_candidates = max(len(values) for values in candidates)
    costs = np.full((len(candidates), max_candidates), np.inf, dtype=np.float64)
    for item_index, values in enumerate(candidates):
        for candidate_index, codes in enumerate(values):
            costs[item_index, candidate_index] = _semantic_cost(latents[item_index], code_vectors, codes)
    return costs


def _hard_semantic_allowed(
    raw_semantic: np.ndarray,
    budget_ratio: float,
) -> tuple[np.ndarray, float, float, float]:
    baseline_mean = float(raw_semantic[:, 0].mean())
    mean_budget = baseline_mean * float(budget_ratio)
    per_item_delta = max(0.0, mean_budget - baseline_mean)
    allowed = raw_semantic <= (raw_semantic[:, 0:1] + per_item_delta)
    return allowed, baseline_mean, mean_budget, per_item_delta


def _behavior_costs(
    candidates: list[list[list[int]]],
    behavior: np.ndarray,
    level_profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_candidates = max(len(values) for values in candidates)
    l1_costs = np.full((len(candidates), max_candidates), np.inf, dtype=np.float64)
    kl_costs = np.full_like(l1_costs, np.inf)
    for item_index, values in enumerate(candidates):
        for candidate_index, codes in enumerate(values):
            l1_costs[item_index, candidate_index] = np.mean([
                _normalized_l1(behavior[item_index], level_profiles[level][code])
                for level, code in enumerate(codes)
            ])
            kl_costs[item_index, candidate_index] = np.mean([
                _kl(behavior[item_index], level_profiles[level][code])
                for level, code in enumerate(codes)
            ])
    return l1_costs, kl_costs


def build_semantic_control_sids(config: dict[str, Any], control: str, force: bool = False) -> dict[str, Any]:
    if control not in {"bcgsid_shuffled", "bcgsid_hard"}:
        raise ValueError(f"unknown semantic control: {control}")
    ensure_run_dirs(config)
    out_dir = artifact_path(config, "sid", control)
    sid_path = out_dir / "semantic_ids.json"
    report_path = out_dir / "assignment_report.json"
    solver_path = out_dir / "solver_report.json"
    if sid_path.exists() and report_path.exists() and solver_path.exists() and not force:
        return __import__("json").loads(report_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    item_ids, base_codes, topk_codes, latents, _fused = encode_items_with_rqvae(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "item_latents.npy", latents.astype(np.float32))
    true_behavior = _load_behavior_profiles(config, len(item_ids))
    behavior = true_behavior
    behavior_control = "true"
    if control == "bcgsid_shuffled":
        rng = np.random.default_rng(int(config["seed"]) + 101)
        behavior = true_behavior[rng.permutation(len(item_ids))]
        behavior_control = "shuffled_item_behavior_profiles"

    code_vectors = _codebook_vectors(config)
    token_space = TokenSpace(
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
    _metrics, level_profiles = _base_metrics(base_codes, code_vectors, behavior)
    allocation = config["sid_allocation"]
    candidates = _candidate_bases(
        base_codes,
        topk_codes,
        int(allocation["base_candidates"]),
        int(allocation["top_codes_per_level"]),
    )
    raw_semantic = _raw_semantic_costs(candidates, latents, code_vectors)
    behavior_l1, behavior_kl = _behavior_costs(candidates, behavior, level_profiles)

    budget_ratio = float(config["sid_allocation"].get("hard_semantic_budget_ratio", 1.05))
    allowed, baseline_mean_semantic, mean_budget, per_item_delta = _hard_semantic_allowed(
        raw_semantic,
        budget_ratio,
    )
    if control != "bcgsid_hard":
        allowed = np.ones_like(raw_semantic, dtype=bool)

    semantic_cost = (float(allocation["lambda_recon"]) + float(allocation["lambda_semantic"])) * raw_semantic
    behavior_cost = float(allocation["lambda_behavior"]) * behavior_l1
    cluster_cost = float(allocation["lambda_cluster"]) * behavior_kl
    entropy = -(behavior * np.log(behavior + 1e-9)).sum(axis=1)
    order = np.argsort(-(entropy + raw_semantic[:, 0]), kind="stable")
    usage = np.zeros((base_codes.shape[1], int(config["rqvae"]["codebook_size"])), dtype=np.int64)
    base_usage = np.zeros_like(usage)
    used: set[tuple[int, ...]] = set()
    assignments: dict[str, dict[str, Any]] = {}
    suffix_limit = int(allocation["suffix_candidates_per_original_code"])
    suffix_used = 0
    semantic_reallocated = 0
    max_suffix_used = 0

    for item_index in order:
        item_candidates = candidates[int(item_index)]
        best_key: tuple[float, int, int, list[int], list[int]] | None = None
        for candidate_index, codes in enumerate(item_candidates):
            if not bool(allowed[int(item_index), candidate_index]):
                continue
            for level, code in enumerate(codes):
                base_usage[level, code] += 1
            usage_penalty = 0.0
            for level, code in enumerate(codes):
                share = base_usage[level, code] / max(int(item_index) + 1, 1)
                usage_penalty += (share - 1.0 / int(config["rqvae"]["codebook_size"])) ** 2
            for suffix in range(suffix_limit):
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
        used.add(tuple(token_ids))
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
            "assignment": "hard_semantic_candidate" if control == "bcgsid_hard" and is_reallocated else (
                "shuffled_behavior_candidate" if is_reallocated else ("suffix" if suffix > 0 else "anchor")
            ),
            "source_candidate_index": int(candidate_index),
        }

    if len(used) != len(item_ids) or len(assignments) != len(item_ids):
        raise AssertionError("semantic control must produce one unique SID per catalog item")
    assigned_codes = np.asarray([assignments[item_id]["base_codes"] for item_id in item_ids], dtype=np.int64)
    quality = _assignment_quality(assigned_codes, latents, true_behavior, code_vectors)
    final_metrics = codebook_metrics(assigned_codes, int(config["rqvae"]["codebook_size"]))
    elapsed = float(time.perf_counter() - started)
    semantic_constraint_satisfied = quality["mean_semantic_drift"] <= mean_budget
    report = {
        "control": control,
        "behavior_control": behavior_control,
        "hard_semantic_constraint": control == "bcgsid_hard",
        "semantic_budget_ratio": budget_ratio if control == "bcgsid_hard" else None,
        "baseline_mean_semantic_drift": baseline_mean_semantic,
        "semantic_drift_budget": mean_budget if control == "bcgsid_hard" else None,
        "semantic_constraint_satisfied": bool(semantic_constraint_satisfied),
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
        "control": control,
        "behavior_control": behavior_control,
        "hard_semantic_constraint": control == "bcgsid_hard",
        "semantic_budget_ratio": budget_ratio if control == "bcgsid_hard" else None,
        "per_item_semantic_delta": per_item_delta if control == "bcgsid_hard" else None,
        "semantic_constraint_satisfied": bool(semantic_constraint_satisfied),
        "elapsed_seconds": elapsed,
        "final_collision_rate": report["final_collision_rate"],
    }
    _write_json(sid_path, assignments)
    _write_json(report_path, report)
    _write_json(solver_path, solver_report)
    np.save(out_dir / "sid_token_ids.npy", np.asarray([assignments[item_id]["token_ids"] for item_id in item_ids], dtype=np.int64))
    np.save(out_dir / "assigned_base_codes.npy", assigned_codes)
    return report
