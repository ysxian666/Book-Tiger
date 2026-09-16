"""Stage orchestration for the RTX 5090 overnight screening."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .config import artifact_path, data_path, ensure_run_dirs
from .data import preprocess_books
from .evaluate import evaluate_variant
from .features import build_features
from .generator_train import train_generator
from .report import write_comparison
from .rqvae_train import train_rqvae
from .semantic_ids import build_baseline_sids, build_bcgsid_sids


GENERATOR_ORDER = ["o1_l2_lite", "o2_bcgsid", "o3_ucg", "o4_bcgsid_ucg", "o5_joint"]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def sync_required_artifacts(config: dict[str, Any]) -> None:
    run_dir = Path(config["paths"]["run_dir"])
    for name in ("sample_config.json", "final_data_stats.json", "item_cap_stats.json", "split_stats.json", "user_sampling_stats.json"):
        source = data_path(config, name)
        if source.exists():
            shutil.copyfile(source, run_dir / name)


def stage_preprocess(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    stats = preprocess_books(config, force=force)
    sync_required_artifacts(config)
    return stats


def stage_features(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    result = build_features(config, force=force)
    _write_json(artifact_path(config, "metadata", "feature_report.json"), result)
    return result


def stage_sid(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    baseline = build_baseline_sids(config, force=force)
    bcgsid = build_bcgsid_sids(config, force=force)
    return {"baseline": baseline, "bcgsid": bcgsid}


def stage_generators(config: dict[str, Any], force: bool = False, max_steps: int | None = None) -> dict[str, Any]:
    output = {}
    for variant in GENERATOR_ORDER:
        started = time.perf_counter()
        output[variant] = train_generator(config, variant, force=force, max_steps_override=max_steps)
        output[variant]["wall_seconds"] = time.perf_counter() - started
        if (
            variant == "o1_l2_lite"
            and int(output[variant].get("total_steps", 0)) <= 20000
            and int(config["generator"]["train_steps"]) > 20000
        ):
            config["generator"]["train_steps"] = 20000
            print("[pipeline] O1 used the 20K fallback; O2-O4 will use the same uniform 20K budget")
    return output


def stage_evaluate(config: dict[str, Any], force: bool = False, exhaustive: bool = True) -> dict[str, Any]:
    output = {}
    for variant in GENERATOR_ORDER:
        output[variant] = evaluate_variant(config, variant, force=force, exhaustive=exhaustive)
    output.update(write_comparison(config))
    return output


def run_pipeline(config: dict[str, Any], stage: str = "all", force: bool = False, max_steps: int | None = None, exhaustive: bool = True) -> dict[str, Any]:
    ensure_run_dirs(config)
    stage = str(stage)
    started = time.perf_counter()
    result: dict[str, Any] = {"stage": stage, "config": str(config.get("_config_path", "configs/overnight_5090.json"))}
    if stage in ("all", "preprocess"):
        result["preprocess"] = stage_preprocess(config, force=force)
    if stage in ("all", "features"):
        result["features"] = stage_features(config, force=force)
    if stage in ("all", "rqvae"):
        result["rqvae"] = train_rqvae(config, force=force)
    if stage in ("all", "sid"):
        result["sid"] = stage_sid(config, force=force)
    if stage in ("all", "generators", "train"):
        result["generators"] = stage_generators(config, force=force, max_steps=max_steps)
    if stage in ("all", "evaluate", "eval"):
        result["evaluation"] = stage_evaluate(config, force=force, exhaustive=exhaustive)
    if stage == "report":
        result["evaluation"] = write_comparison(config)
    result["elapsed_sec"] = float(time.perf_counter() - started)
    _write_json(artifact_path(config, "metadata", f"pipeline_{stage}.json"), result)
    return result
