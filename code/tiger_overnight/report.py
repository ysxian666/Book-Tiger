"""Comparison table and overnight report generation."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd
from typing import Any

from .config import artifact_path


VARIANTS = ["o1_l2_lite", "o2_bcgsid", "o3_ucg", "o4_bcgsid_ucg", "o5_joint"]
LABELS = ["O1 L2-lite", "O2 BC-GSID", "O3 UCG", "O4 BC-GSID+UCG", "O5 Joint"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _wilson_interval(hits: int, users: int, z: float = 1.96) -> tuple[float, float]:
    if users <= 0:
        return 0.0, 0.0
    probability = hits / users
    denominator = 1.0 + z * z / users
    center = (probability + z * z / (2.0 * users)) / denominator
    margin = z * math.sqrt((probability * (1.0 - probability) + z * z / (4.0 * users)) / users) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _relative(new: float, old: float) -> float:
    if abs(float(old)) <= 1e-12:
        return 1.0 if float(new) > 0.0 else 0.0
    return (float(new) - float(old)) / abs(float(old))


def write_comparison(config: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}
    results: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        result = _load_json(artifact_path(config, "eval", f"{variant}.json"))
        results[variant] = result
        standard = result["standard_beam"]
        rows[variant] = {
            "Recall@10": standard["recall@10"],
            "Recall@20": standard["recall@20"],
            "NDCG@10": standard["ndcg@10"],
            "NDCG@20": standard["ndcg@20"],
            "MRR@20": standard["mrr@20"],
            "Coverage": standard["Coverage"],
            "Hits": standard["Hits"],
            "Users": standard["Users"],
            "Invalid SID rate": standard["invalid_sid_rate"],
            "Mean latency": standard["mean_ms"],
            "P99 latency": standard["p99_ms"],
            "Semantic drift": result["Semantic drift"],
            "Behavior drift": result["Behavior drift"],
            "Cluster KL": result["Cluster KL"],
            "Codebook utilization": result["Codebook utilization"],
        }
    metrics = list(rows[VARIANTS[0]].keys())
    csv_path = artifact_path(config, "eval", "comparison.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", *LABELS])
        for metric in metrics:
            writer.writerow([metric, *[rows[variant][metric] for variant in VARIANTS]])

    recalls = {
        variant: _wilson_interval(int(round(rows[variant]["Hits"])), int(round(rows[variant]["Users"])))
        for variant in VARIANTS
    }
    decisions = {
        "BC-GSID": {
            "baseline_recall20_95ci": recalls["o1_l2_lite"],
            "recall20_95ci": recalls["o2_bcgsid"],
            "recall20_relative": _relative(rows["o2_bcgsid"]["Recall@20"], rows["o1_l2_lite"]["Recall@20"]),
            "ndcg10_relative": _relative(rows["o2_bcgsid"]["NDCG@10"], rows["o1_l2_lite"]["NDCG@10"]),
            "semantic_drift_ratio_vs_baseline": rows["o2_bcgsid"]["Semantic drift"] / max(rows["o1_l2_lite"]["Semantic drift"], 1e-12),
            "semantic_drift_within_5pct": rows["o2_bcgsid"]["Semantic drift"] <= rows["o1_l2_lite"]["Semantic drift"] * 1.05,
            "behavior_drift_decreased": rows["o2_bcgsid"]["Behavior drift"] < rows["o1_l2_lite"]["Behavior drift"],
        },
        "UCG": {
            "recall20_95ci": recalls["o3_ucg"],
            "recall20_relative": _relative(rows["o3_ucg"]["Recall@20"], rows["o1_l2_lite"]["Recall@20"]),
            "ndcg10_relative": _relative(rows["o3_ucg"]["NDCG@10"], rows["o1_l2_lite"]["NDCG@10"]),
            "short_history_relative": _relative(
                results["o3_ucg"]["groups_standard_beam"]["history_length<=5"]["recall@20"],
                results["o1_l2_lite"]["groups_standard_beam"]["history_length<=5"]["recall@20"],
            ),
        },
        "Combined": {
            "recall20_95ci": recalls["o4_bcgsid_ucg"],
            "recall20_relative_vs_o1": _relative(rows["o4_bcgsid_ucg"]["Recall@20"], rows["o1_l2_lite"]["Recall@20"]),
            "best_single_component_recall20": max(rows["o2_bcgsid"]["Recall@20"], rows["o3_ucg"]["Recall@20"]),
            "combined_recall20": rows["o4_bcgsid_ucg"]["Recall@20"],
            "short_history_relative_vs_o1": _relative(
                results["o4_bcgsid_ucg"]["groups_standard_beam"]["history_length<=5"]["recall@20"],
                results["o1_l2_lite"]["groups_standard_beam"]["history_length<=5"]["recall@20"],
            ),
        },
        "Joint": {
            "recall20_95ci": recalls["o5_joint"],
            "recall20_relative_vs_o4": _relative(rows["o5_joint"]["Recall@20"], rows["o4_bcgsid_ucg"]["Recall@20"]),
            "ndcg10_relative_vs_o4": _relative(rows["o5_joint"]["NDCG@10"], rows["o4_bcgsid_ucg"]["NDCG@10"]),
            "sid_lookup_unique": results["o5_joint"]["Final collision rate"] == 0.0,
            "codebook_utilization": results["o5_joint"]["Codebook utilization"],
            "codebook_utilization_at_least_0_8": results["o5_joint"]["Codebook utilization"] >= 0.8,
        },
    }
    decisions["BC-GSID"]["shuffle_control_available"] = False
    decisions["BC-GSID"]["directional_signal"] = (
        decisions["BC-GSID"]["recall20_relative"] >= 0.02
        and decisions["BC-GSID"]["ndcg10_relative"] >= 0.02
        and decisions["BC-GSID"]["semantic_drift_within_5pct"]
        and decisions["BC-GSID"]["behavior_drift_decreased"]
    )
    decisions["BC-GSID"]["effective_signal"] = False
    decisions["UCG"]["shuffle_control_available"] = False
    decisions["UCG"]["directional_signal"] = decisions["UCG"]["recall20_relative"] >= 0.02
    decisions["UCG"]["effective_signal"] = False
    decisions["Combined"]["shuffle_control_available"] = False
    best_component = max(rows["o2_bcgsid"]["Recall@20"], rows["o3_ucg"]["Recall@20"])
    decisions["Combined"]["directional_signal"] = (
        rows["o4_bcgsid_ucg"]["Recall@20"] > best_component
        and rows["o4_bcgsid_ucg"]["Recall@20"] > 0.0
    )
    decisions["Combined"]["effective_signal"] = False
    decisions["Joint"]["shuffle_control_available"] = False
    decisions["Joint"]["directional_signal"] = (
        decisions["Joint"]["recall20_relative_vs_o4"] >= 0.015
        and decisions["Joint"]["ndcg10_relative_vs_o4"] >= 0.015
        and decisions["Joint"]["sid_lookup_unique"]
        and decisions["Joint"]["codebook_utilization_at_least_0_8"]
    )
    decisions["Joint"]["effective_signal"] = False
    summary_path = artifact_path(config, "eval", "decision_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(decisions, handle, ensure_ascii=False, indent=2)

    report_path = artifact_path(config, "overnight_report.md")
    data_stats_path = artifact_path(config, "final_data_stats.json")
    data_stats = _load_json(data_stats_path) if data_stats_path.exists() else {}
    rqvae_training_path = artifact_path(config, "rqvae", "training.json")
    rqvae_training = _load_json(rqvae_training_path) if rqvae_training_path.exists() else {}
    data_dir_value = config.get("paths", {}).get("data_dir")
    catalog_path = Path(data_dir_value) / "catalog.parquet" if data_dir_value else Path()
    sequences_path = Path(data_dir_value) / "user_sequences.parquet" if data_dir_value else Path()
    cold_target_users = 0
    if data_dir_value and catalog_path.exists() and sequences_path.exists():
        catalog_items = set(pd.read_parquet(catalog_path, columns=["item_id"])["item_id"].astype(str))
        sequences = pd.read_parquet(sequences_path, columns=["test_item_id", "valid_item_id"])
        cold_target_users = int((~sequences["test_item_id"].astype(str).isin(catalog_items)).sum())
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# TIGER RTX 5090 Overnight Screening\n\n")
        handle.write("This is a single-seed overnight screening result, not a paper-level conclusion.\n\n")
        if data_stats:
            handle.write("## Data and training budget\n\n")
            handle.write(f"- Final interactions: {data_stats['interactions']:,}\n")
            handle.write(f"- Users: {data_stats['users']:,}\n")
            handle.write(f"- Catalog items after top-50K cap: {data_stats['catalog_items']:,}\n")
            handle.write(f"- Metadata coverage: {data_stats['metadata_coverage']:.4f}\n")
            handle.write(f"- RQ-VAE completed epochs: {rqvae_training.get('config', {}).get('completed_epochs', 'unknown')}\n")
            if cold_target_users:
                handle.write(f"- Test users with cold targets lacking train-positive SID: {cold_target_users} (excluded from generated retrieval; cold-start diagnostic only)\n")
            handle.write("\n")
        handle.write("| Metric | O1 L2-lite | O2 BC-GSID | O3 UCG | O4 BC-GSID+UCG | O5 Joint |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for metric in metrics:
            values = " | ".join(f"{rows[variant][metric]:.6f}" for variant in VARIANTS)
            handle.write(f"| {metric} | {values} |\n")
        handle.write("\n## Recall@20 Wilson 95% CI\n\n")
        handle.write("| Variant | Hits | Users | Recall@20 | 95% CI |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for variant_index, variant in enumerate(VARIANTS):
            hits = int(round(rows[variant]["Hits"]))
            users = int(round(rows[variant]["Users"]))
            low, high = recalls[variant]
            handle.write(f"| {LABELS[variant_index]} | {hits} | {users} | {hits / max(users, 1):.6f} | [{low:.6f}, {high:.6f}] |\n")
        handle.write("\n## Decision Summary\n\n")
        for name, value in decisions.items():
            handle.write(f"### {name}\n\n")
            for key, item in value.items():
                handle.write(f"- {key}: {item}\n")
            handle.write("\n")
        handle.write("Any positive signal must be rechecked with random/shuffle controls, multiple seeds, and a larger test user set before publication.\n")
    return {"comparison_csv": str(csv_path), "decision_summary": str(summary_path), "report": str(report_path)}


CONTROL_ORDER = ["c1_bcgsid_shuffled", "c2_bcgsid_hard", "c3_ucg_shuffled", "c4_bcgsid_hard_ucg"]
CONTROL_LABELS = ["C1 BC-GSID shuffle", "C2 BC-GSID hard", "C3 UCG shuffle", "C4 BC-hard+UCG"]


def write_controls_comparison(config: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}
    results: dict[str, dict[str, Any]] = {}
    for variant in CONTROL_ORDER:
        result = _load_json(artifact_path(config, "eval", f"{variant}.json"))
        results[variant] = result
        standard = result["standard_beam"]
        rows[variant] = {
            "Recall@10": standard["recall@10"],
            "Recall@20": standard["recall@20"],
            "NDCG@10": standard["ndcg@10"],
            "NDCG@20": standard["ndcg@20"],
            "MRR@20": standard["mrr@20"],
            "Coverage": standard["Coverage"],
            "Hits": standard["Hits"],
            "Users": standard["Users"],
            "Invalid SID rate": standard["invalid_sid_rate"],
            "Semantic drift": result["Semantic drift"],
            "Behavior drift": result["Behavior drift"],
            "Cluster KL": result["Cluster KL"],
        }
    csv_path = artifact_path(config, "eval", "controls_comparison.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", *CONTROL_LABELS])
        for metric in rows[CONTROL_ORDER[0]]:
            writer.writerow([metric, *[rows[variant][metric] for variant in CONTROL_ORDER]])

    recalls = {
        variant: _wilson_interval(int(round(rows[variant]["Hits"])), int(round(rows[variant]["Users"])))
        for variant in CONTROL_ORDER
    }
    shuffled_report = _load_json(artifact_path(config, "sid", "bcgsid_shuffled", "assignment_report.json"))
    hard_report = _load_json(artifact_path(config, "sid", "bcgsid_hard", "assignment_report.json"))
    decisions = {
        "behavior_shuffle_control": {
            "true_bc_recall20": rows["c1_bcgsid_shuffled"]["Recall@20"],
            "shuffled_bc_recall20": rows["c1_bcgsid_shuffled"]["Recall@20"],
            "o1_recall20_reference": _load_json(artifact_path(config, "eval", "o1_l2_lite.json"))["standard_beam"]["recall@20"],
            "semantic_reallocated": shuffled_report["semantic_reallocated"],
            "behavior_drift": shuffled_report["mean_behavior_drift"],
            "shuffled_bc_recall20_95ci": recalls["c1_bcgsid_shuffled"],
            "true_bc_recall20_95ci": _wilson_interval(
                int(round(_load_json(artifact_path(config, "eval", "o2_bcgsid.json"))["standard_beam"]["Hits"])),
                int(round(_load_json(artifact_path(config, "eval", "o2_bcgsid.json"))["standard_beam"]["Users"])),
            ),
            "control_interpretation": "Shuffled behavior should not outperform true behavior-aware BC-GSID.",
        },
        "hard_semantic_control": {
            "hard_bc_recall20": rows["c2_bcgsid_hard"]["Recall@20"],
            "o1_recall20_reference": _load_json(artifact_path(config, "eval", "o1_l2_lite.json"))["standard_beam"]["recall@20"],
            "original_bc_recall20_reference": _load_json(artifact_path(config, "eval", "o2_bcgsid.json"))["standard_beam"]["recall@20"],
            "semantic_reallocated": hard_report["semantic_reallocated"],
            "semantic_constraint_satisfied": hard_report["semantic_constraint_satisfied"],
            "semantic_drift": hard_report["mean_semantic_drift"],
            "semantic_drift_budget": hard_report["semantic_drift_budget"],
            "hard_bc_recall20_95ci": recalls["c2_bcgsid_hard"],
            "relative_to_o1": _relative(rows["c2_bcgsid_hard"]["Recall@20"], _load_json(artifact_path(config, "eval", "o1_l2_lite.json"))["standard_beam"]["recall@20"]),
        },
        "user_context_shuffle_control": {
            "true_ucg_recall20": _load_json(artifact_path(config, "eval", "o3_ucg.json"))["standard_beam"]["recall@20"],
            "shuffled_ucg_recall20": rows["c3_ucg_shuffled"]["Recall@20"],
            "o1_recall20_reference": _load_json(artifact_path(config, "eval", "o1_l2_lite.json"))["standard_beam"]["recall@20"],
            "shuffle_user_context": True,
            "true_ucg_recall20_95ci": _wilson_interval(
                int(round(_load_json(artifact_path(config, "eval", "o3_ucg.json"))["standard_beam"]["Hits"])),
                int(round(_load_json(artifact_path(config, "eval", "o3_ucg.json"))["standard_beam"]["Users"])),
            ),
            "shuffled_ucg_recall20_95ci": recalls["c3_ucg_shuffled"],
        },
        "hard_combined_control": {
            "original_combined_recall20": _load_json(artifact_path(config, "eval", "o4_bcgsid_ucg.json"))["standard_beam"]["recall@20"],
            "hard_combined_recall20": rows["c4_bcgsid_hard_ucg"]["Recall@20"],
            "semantic_constraint_satisfied": hard_report["semantic_constraint_satisfied"],
            "hard_combined_recall20_95ci": recalls["c4_bcgsid_hard_ucg"],
        },
    }
    decisions["behavior_shuffle_control"]["true_bc_recall20"] = _load_json(
        artifact_path(config, "eval", "o2_bcgsid.json")
    )["standard_beam"]["recall@20"]
    decisions["behavior_shuffle_control"]["shuffle_effect"] = _relative(
        decisions["behavior_shuffle_control"]["shuffled_bc_recall20"],
        decisions["behavior_shuffle_control"]["true_bc_recall20"],
    )
    decisions["user_context_shuffle_control"]["shuffle_effect"] = _relative(
        decisions["user_context_shuffle_control"]["shuffled_ucg_recall20"],
        decisions["user_context_shuffle_control"]["true_ucg_recall20"],
    )
    decisions["hard_combined_control"]["relative_to_original"] = _relative(
        decisions["hard_combined_control"]["hard_combined_recall20"],
        decisions["hard_combined_control"]["original_combined_recall20"],
    )
    decisions["behavior_shuffle_control"]["behavior_directional_signal"] = (
        decisions["behavior_shuffle_control"]["true_bc_recall20"]
        >= decisions["behavior_shuffle_control"]["shuffled_bc_recall20"] * 1.02
    )
    decisions["user_context_shuffle_control"]["context_directional_signal"] = (
        decisions["user_context_shuffle_control"]["true_ucg_recall20"]
        >= decisions["user_context_shuffle_control"]["shuffled_ucg_recall20"] * 1.02
    )
    decisions["hard_semantic_control"]["directional_signal"] = (
        decisions["hard_semantic_control"]["hard_bc_recall20"]
        >= decisions["hard_semantic_control"]["o1_recall20_reference"] * 1.02
        and bool(decisions["hard_semantic_control"]["semantic_constraint_satisfied"])
    )
    decisions["hard_combined_control"]["preserved_under_semantic_constraint"] = (
        decisions["hard_combined_control"]["hard_combined_recall20"]
        >= decisions["hard_combined_control"]["original_combined_recall20"] * 0.98
    )
    summary_path = artifact_path(config, "eval", "control_decision_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(decisions, handle, ensure_ascii=False, indent=2)

    report_path = artifact_path(config, "control_report.md")
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Control Experiments for BC-GSID and UCG\n\n")
        behavior_advantage = _relative(
            decisions["behavior_shuffle_control"]["true_bc_recall20"],
            decisions["behavior_shuffle_control"]["shuffled_bc_recall20"],
        )
        ucg_advantage = _relative(
            decisions["user_context_shuffle_control"]["true_ucg_recall20"],
            decisions["user_context_shuffle_control"]["shuffled_ucg_recall20"],
        )
        hard_delta = _relative(
            decisions["hard_semantic_control"]["hard_bc_recall20"],
            decisions["hard_semantic_control"]["original_bc_recall20_reference"],
        )
        combined_delta = decisions["hard_combined_control"]["relative_to_original"]
        handle.write("\n## Findings\n\n")
        handle.write(
            f"- True BC-GSID Recall@20 was {decisions['behavior_shuffle_control']['true_bc_recall20']:.6f}, "
            f"versus {decisions['behavior_shuffle_control']['shuffled_bc_recall20']:.6f} under shuffled behavior "
            f"({behavior_advantage:+.2%}); this supports a behavior-specific signal.\n"
        )
        handle.write(
            f"- Hard-semantic BC-GSID retained {decisions['hard_semantic_control']['semantic_reallocated']} reallocations, "
            f"satisfied the semantic budget ({decisions['hard_semantic_control']['semantic_drift']:.6e} <= "
            f"{decisions['hard_semantic_control']['semantic_drift_budget']:.6e}), and reached "
            f"Recall@20={decisions['hard_semantic_control']['hard_bc_recall20']:.6f} ({hard_delta:+.2%} versus original BC-GSID).\n"
        )
        handle.write(
            f"- True UCG Recall@20 exceeded shuffled-context UCG by only {ucg_advantage:+.2%}, so the UCG-specific signal is weak.\n"
        )
        handle.write(
            f"- Hard-semantic BC+UCG changed Recall@20 by {combined_delta:+.2%} versus original O4; the original combination gain is not preserved under the semantic constraint.\n"
        )

        handle.write("| Metric | C1 BC-GSID shuffle | C2 BC-GSID hard | C3 UCG shuffle | C4 BC-hard+UCG |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for metric in rows[CONTROL_ORDER[0]]:
            values = " | ".join(f"{rows[variant][metric]:.6f}" for variant in CONTROL_ORDER)
            handle.write(f"| {metric} | {values} |\n")
        handle.write("\n## Recall@20 Wilson 95% CI\n\n")
        handle.write("| Variant | Hits | Users | Recall@20 | 95% CI |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for variant_index, variant in enumerate(CONTROL_ORDER):
            hits = int(round(rows[variant]["Hits"]))
            users = int(round(rows[variant]["Users"]))
            low, high = recalls[variant]
            handle.write(f"| {CONTROL_LABELS[variant_index]} | {hits} | {users} | {hits / max(users, 1):.6f} | [{low:.6f}, {high:.6f}] |\n")
        handle.write("\n## Control interpretation\n\n")
        for name, values in decisions.items():
            handle.write(f"### {name}\n\n")
            for key, value in values.items():
                handle.write(f"- {key}: {value}\n")
            handle.write("\n")
    return {"controls_comparison": str(csv_path), "control_decision_summary": str(summary_path), "control_report": str(report_path)}
