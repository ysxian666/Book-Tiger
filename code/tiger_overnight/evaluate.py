"""Unified test evaluation for the five overnight variants."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import artifact_path, data_path, ensure_run_dirs
from .decoding import beam_search, build_trie, exhaustive_rank
from .generator_data import TigerSequenceDataset, collate_sequences, load_sid_map_and_order, load_token_space, select_eval_users, sid_dir
from .generator_model import TigerGenerator
from .generator_train import VARIANT_SPECS


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _dcg(rank: int) -> float:
    return 1.0 / math.log2(int(rank) + 1.0)


def _empty_metrics() -> dict[str, float]:
    return {
        "recall@10": 0.0,
        "recall@20": 0.0,
        "ndcg@10": 0.0,
        "ndcg@20": 0.0,
        "mrr@20": 0.0,
    }


def _accumulate_user(metrics: dict[str, float], recommendations: list[str], target: str) -> None:
    rank = recommendations.index(target) + 1 if target in recommendations else 0
    if rank and rank <= 10:
        metrics["recall@10"] += 1.0
        metrics["ndcg@10"] += _dcg(rank)
    if rank and rank <= 20:
        metrics["recall@20"] += 1.0
        metrics["ndcg@20"] += _dcg(rank)
        metrics["mrr@20"] += 1.0 / rank


def _finalize_retrieval(metrics: dict[str, float], users: int, coverage: set[str], catalog_size: int, invalid: int, beams: int, returned: int, filtered: int) -> dict[str, Any]:
    output = {key: value / max(users, 1) for key, value in metrics.items()}
    output.update({
        "Coverage": len(coverage) / max(catalog_size, 1),
        "Hits": int(round(metrics["recall@20"] * users)),
        "Users": int(users),
        "invalid_sid_rate": invalid / max(beams, 1),
        "valid_sid_rate": 1.0 - invalid / max(beams, 1),
        "avg_returned_items": returned / max(users, 1),
        "seen_item_filtered": int(filtered),
    })
    return output


def _run_beam_protocol(
    model: TigerGenerator,
    dataset: TigerSequenceDataset,
    batch: Any,
    rows: list[dict[str, Any]],
    token_space: Any,
    token_to_item: dict[tuple[int, ...], str],
    constrained_trie: dict[tuple[int, ...], set[int]] | None,
    beam_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, Any], list[list[str]], list[float], int, int, int]:
    batch.input_ids = batch.input_ids.to(device)
    batch.attention_mask = batch.attention_mask.to(device)
    batch.long_context = batch.long_context.to(device)
    batch.short_context = batch.short_context.to(device)
    history_ids = [
        {dataset.item_ids[int(index)] for index in row["history_indices"].tolist()}
        for row in rows
    ]
    started = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        decoded = beam_search(
            model,
            batch.input_ids,
            batch.attention_mask,
            batch.long_context,
            batch.short_context,
            token_space,
            token_to_item,
            history_ids,
            beam_size=int(beam_size),
            constrained_trie=constrained_trie,
            max_steps=int(token_space.max_sid_length),
        )
    elapsed = time.perf_counter() - started
    metrics = _empty_metrics()
    coverage: set[str] = set()
    invalid = 0
    beams = 0
    returned = 0
    filtered = 0
    recommendations: list[list[str]] = []
    for index, output in enumerate(decoded):
        recs = output.item_ids[:20]
        recommendations.append(recs)
        coverage.update(recs)
        _accumulate_user(metrics, recs, rows[index]["target_item_id"])
        invalid += output.invalid_beams
        beams += output.total_beams
        returned += len(recs)
        filtered += max(0, output.total_beams - output.invalid_beams - len(recs))
    finalized = _finalize_retrieval(
        metrics,
        users=len(rows),
        coverage=coverage,
        catalog_size=len(dataset.item_ids),
        invalid=invalid,
        beams=beams,
        returned=returned,
        filtered=filtered,
    )
    return finalized, recommendations, [elapsed / max(len(rows), 1)] * len(rows), int(invalid), int(beams), int(filtered)


def _metrics_from_recommendations(rows: list[dict[str, Any]], recommendations: list[list[str]], catalog_size: int) -> dict[str, Any]:
    metrics = _empty_metrics()
    coverage: set[str] = set()
    for row, recs in zip(rows, recommendations):
        _accumulate_user(metrics, recs[:20], str(row["target_item_id"]))
        coverage.update(recs[:20])
    output = {key: value / max(len(rows), 1) for key, value in metrics.items()}
    output.update({
        "Coverage": len(coverage) / max(catalog_size, 1),
        "Hits": int(round(metrics["recall@20"])),
        "Users": len(rows),
        "avg_returned_items": float(np.mean([len(recs[:20]) for recs in recommendations])) if recommendations else 0.0,
    })
    return output


def _group_metrics(rows: list[dict[str, Any]], recommendations: list[list[str]], frequency_map: dict[str, str]) -> dict[str, Any]:
    buckets: dict[str, list[int]] = {
        "history_length<=5": [],
        "6<=history_length<=20": [],
        "history_length>20": [],
        "cold_item": [],
        "few_shot_item": [],
        "long_tail_item": [],
        "head_item": [],
    }
    for index, row in enumerate(rows):
        history_length = int(row["history_length"])
        if history_length <= 5:
            buckets["history_length<=5"].append(index)
        elif history_length <= 20:
            buckets["6<=history_length<=20"].append(index)
        else:
            buckets["history_length>20"].append(index)
        frequency_group = frequency_map.get(str(row["target_item_id"]), "unknown")
        if frequency_group in buckets:
            buckets[frequency_group].append(index)
    output: dict[str, Any] = {}
    for name, indices in buckets.items():
        if not indices:
            output[name] = {"Users": 0, **_empty_metrics()}
            continue
        subset_rows = [rows[index] for index in indices]
        subset_recs = [recommendations[index] for index in indices]
        output[name] = _metrics_from_recommendations(subset_rows, subset_recs, catalog_size=1)
    return output


def _load_frequency_map(config: dict[str, Any]) -> dict[str, str]:
    table = pd.read_parquet(data_path(config, "item_stats.parquet"), columns=["item_id", "frequency_group"])
    return {str(row.item_id): str(row.frequency_group) for row in table.itertuples(index=False)}


def evaluate_variant(
    config: dict[str, Any],
    variant: str,
    force: bool = False,
    exhaustive: bool = True,
) -> dict[str, Any]:
    ensure_run_dirs(config)
    output_path = artifact_path(config, "eval", f"{variant}.json")
    if output_path.exists() and not force:
        return _load_json(output_path)
    if variant not in VARIANT_SPECS:
        raise KeyError(f"unknown variant: {variant}")
    spec = VARIANT_SPECS[variant]
    checkpoint_path = artifact_path(config, "generators", f"{spec['artifact_name']}.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing generator checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["generator"]["bf16"]) and device.type == "cuda"
    token_space = load_token_space(config)
    sid_map, item_ids = load_sid_map_and_order(config, spec["sid_mode"])
    item_latents = np.load(sid_dir(config, spec["sid_mode"]) / "item_latents.npy")
    model, metadata = TigerGenerator.load(
        str(checkpoint_path),
        token_space,
        config["generator"],
        int(item_latents.shape[1]),
        config["user_context"] if spec["soft_prefix"] else None,
        map_location=device,
    )
    model = model.to(device).eval()
    test_users = select_eval_users(config, "test", int(config["evaluation"]["users"]), int(config["seed"]))
    dataset = TigerSequenceDataset(
        config,
        sid_mode=spec["sid_mode"],
        split="test",
        use_soft_prefix=bool(spec["soft_prefix"]),
        user_filter=set(test_users),
        shuffle_user_context=bool(spec.get("shuffle_user_context", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
        pin_memory=False,
    )
    token_to_item = {tuple(int(value) for value in row["token_ids"]): item_id for item_id, row in sid_map.items()}
    trie = build_trie({item_id: [int(value) for value in sid_map[item_id]["token_ids"]] for item_id in item_ids})
    frequency_map = _load_frequency_map(config)
    all_rows: list[dict[str, Any]] = []
    standard_recs: list[list[str]] = []
    trie_recs: list[list[str]] = []
    standard_latencies: list[float] = []
    trie_latencies: list[float] = []
    standard_invalid = 0
    standard_beams = 0
    standard_filtered = 0
    trie_filtered = 0
    trie_invalid = 0
    trie_beams = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        row_start = batch_index * int(config["evaluation"]["batch_size"])
        rows = [dataset[int(index)] for index in range(row_start, min(row_start + int(batch.labels.shape[0]), len(dataset)))]
        standard, recs, latencies, invalid, beams, filtered = _run_beam_protocol(
            model, dataset, batch, rows, token_space, token_to_item, None,
            int(config["evaluation"]["beam"]), device, amp,
        )
        standard_recs.extend(recs)
        standard_latencies.extend(latencies)
        standard_invalid += invalid
        standard_beams += beams
        standard_filtered += filtered
        trie_result, recs, latencies, trie_invalid_batch, trie_beams_batch, trie_filtered_batch = _run_beam_protocol(
            model, dataset, batch, rows, token_space, token_to_item, trie,
            int(config["evaluation"]["beam"]), device, amp,
        )
        trie_recs.extend(recs)
        trie_latencies.extend(latencies)
        trie_filtered += trie_filtered_batch
        trie_invalid += trie_invalid_batch
        trie_beams += trie_beams_batch
        all_rows.extend(rows)
    standard_metrics = _metrics_from_recommendations(all_rows, standard_recs, len(dataset.item_ids))
    trie_metrics = _metrics_from_recommendations(all_rows, trie_recs, len(dataset.item_ids))
    latency_values = np.asarray(standard_latencies, dtype=np.float64) * 1000.0
    standard_metrics.update({
        "invalid_sid_rate": standard_invalid / max(standard_beams, 1),
        "valid_sid_rate": 1.0 - standard_invalid / max(standard_beams, 1),
        "seen_item_filtered": int(standard_filtered),
        "mean_ms": float(np.mean(latency_values)) if latency_values.size else 0.0,
        "p50_ms": float(np.percentile(latency_values, 50)) if latency_values.size else 0.0,
        "p95_ms": float(np.percentile(latency_values, 95)) if latency_values.size else 0.0,
        "p99_ms": float(np.percentile(latency_values, 99)) if latency_values.size else 0.0,
    })
    trie_latency_values = np.asarray(trie_latencies, dtype=np.float64) * 1000.0
    trie_metrics.update({
        "invalid_sid_rate": trie_invalid / max(trie_beams, 1),
        "valid_sid_rate": 1.0 - trie_invalid / max(trie_beams, 1),
        "seen_item_filtered": int(trie_filtered),
        "mean_ms": float(np.mean(trie_latency_values)) if trie_latency_values.size else 0.0,
        "p50_ms": float(np.percentile(trie_latency_values, 50)) if trie_latency_values.size else 0.0,
        "p95_ms": float(np.percentile(trie_latency_values, 95)) if trie_latency_values.size else 0.0,
        "p99_ms": float(np.percentile(trie_latency_values, 99)) if trie_latency_values.size else 0.0,
    })

    exhaustive_metrics: dict[str, Any] = {"users": 0, "enabled": False}
    exhaustive_count = 0
    if exhaustive and int(config["evaluation"]["exhaustive_users"]) > 0:
        exhaustive_count = min(int(config["evaluation"]["exhaustive_users"]), len(dataset))
        ex_dataset = TigerSequenceDataset(
            config,
            sid_mode=spec["sid_mode"],
            split="test",
            use_soft_prefix=bool(spec["soft_prefix"]),
            sample_limit=exhaustive_count,
            user_filter=set(test_users),
            shuffle_user_context=bool(spec.get("shuffle_user_context", False)),
        )
        ex_loader = DataLoader(ex_dataset, batch_size=min(8, int(config["evaluation"]["batch_size"])), shuffle=False, collate_fn=collate_sequences, num_workers=0)
        candidate_tokens = np.asarray([sid_map[item_id]["token_ids"] for item_id in item_ids], dtype=np.int64)
        ex_rows: list[dict[str, Any]] = []
        ex_recs: list[list[str]] = []
        for batch_index, batch in enumerate(ex_loader):
            row_start = batch_index * int(batch.labels.shape[0])
            rows = [ex_dataset[index] for index in range(row_start, min(row_start + int(batch.labels.shape[0]), len(ex_dataset)))]
            batch.input_ids = batch.input_ids.to(device)
            batch.attention_mask = batch.attention_mask.to(device)
            batch.long_context = batch.long_context.to(device)
            batch.short_context = batch.short_context.to(device)
            history_ids = [{ex_dataset.item_ids[int(index)] for index in row["history_indices"].tolist()} for row in rows]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                decoded = exhaustive_rank(
                    model,
                    batch.input_ids,
                    batch.attention_mask,
                    batch.long_context,
                    batch.short_context,
                    candidate_tokens,
                    item_ids,
                    history_ids,
                    top_k=20,
                )
            ex_rows.extend(rows)
            ex_recs.extend([output.item_ids for output in decoded])
        exhaustive_metrics = {"users": len(ex_rows), "enabled": True, **_metrics_from_recommendations(ex_rows, ex_recs, len(item_ids))}

    sid_report_path = sid_dir(config, spec["sid_mode"]) / (
        "sid_report.json" if spec["sid_mode"] == "baseline" else "assignment_report.json"
    )
    sid_report = _load_json(sid_report_path)
    if spec["sid_mode"] == "baseline":
        if "mean_utilization" in sid_report:
            codebook_utilization = float(sid_report["mean_utilization"])
            codebook_perplexity = float(sid_report["mean_perplexity"])
        else:
            rqvae_training = _load_json(artifact_path(config, "rqvae", "training.json"))
            codebook_utilization = float(rqvae_training["final"]["mean_utilization"])
            codebook_perplexity = float(rqvae_training["final"]["mean_perplexity"])
    else:
        codebook_utilization = float(sid_report["codebook_utilization"]["mean"])
        codebook_perplexity = float(sid_report["codebook_perplexity"]["mean"])
    result = {
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": metadata,
        "test_users": len(all_rows),
        "test_users_seed": int(config["seed"]),
        "beam": int(config["evaluation"]["beam"]),
        "standard_beam": standard_metrics,
        "trie_beam": trie_metrics,
        "exhaustive": exhaustive_metrics,
        "groups_standard_beam": _group_metrics(all_rows, standard_recs, frequency_map),
        "groups_trie_beam": _group_metrics(all_rows, trie_recs, frequency_map),
        "Semantic drift": sid_report["mean_semantic_drift"],
        "Behavior drift": sid_report["mean_behavior_drift"],
        "Cluster KL": sid_report["mean_cluster_kl"],
        "SID length": sid_report["sid_length"],
        "Unique SIDs": sid_report["unique_sids"],
        "Final collision rate": sid_report["final_collision_rate"],
        "Codebook utilization": codebook_utilization,
        "Codebook perplexity": codebook_perplexity,
        "wall_time_sec": float(time.perf_counter() - started),
    }
    _write_json(output_path, result)
    return result
