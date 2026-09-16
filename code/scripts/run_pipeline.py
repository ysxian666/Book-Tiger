"""Run the end-to-end pipeline stage by stage with wall-clock timing."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_rec.config import load_config
from tiger_rec.utils import write_json
STAGES = ["prepare", "encode", "sid", "generator", "baselines", "ranker", "evaluate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    parser.add_argument("--methods", nargs="+", default=["tiger_trie"])
    parser.add_argument("--timing-file", default="")
    args = parser.parse_args()
    python = sys.executable
    config_path = str(Path(args.config).expanduser().resolve())
    cfg = load_config(config_path)
    timing_path = Path(args.timing_file).expanduser() if args.timing_file else cfg.paths.artifact("pipeline_timing.json")
    timings: list[dict[str, object]] = []
    pipeline_started = time.perf_counter()

    def run(label: str, command: list[str]) -> None:
        print("+", " ".join(command), flush=True)
        started = time.perf_counter()
        status = "ok"
        try:
            subprocess.run(command, check=True, cwd=str(ROOT))
        except BaseException:
            status = "failed"
            raise
        finally:
            elapsed = time.perf_counter() - started
            timings.append({
                "stage": label,
                "seconds": float(elapsed),
                "status": status,
                "command": command,
            })
            print(f"[timing] {label} {elapsed:.3f}s ({status})", flush=True)

    try:
        for stage in args.stages:
            if stage == "prepare":
                run("prepare", [python, "scripts/prepare_data.py", "--config", config_path])
            elif stage == "encode":
                run("encode", [python, "scripts/encode_items.py", "--config", config_path])
            elif stage == "sid":
                run("sid", [python, "scripts/build_semantic_ids.py", "--config", config_path])
            elif stage == "generator":
                run("generator", [python, "scripts/train_generator.py", "--config", config_path])
            elif stage == "baselines":
                run("baselines", [python, "scripts/train_baselines.py", "--config", config_path])
            elif stage == "ranker":
                run("ranker", [python, "scripts/train_ranker.py", "--config", config_path])
            elif stage == "evaluate":
                for method in args.methods:
                    command = [python, "scripts/evaluate.py", "--config", config_path, "--method", method]
                    if method in {"hybrid", "ranked_union"}:
                        command.extend(["--traditional-for-hybrid", "sasrec"])
                    run(f"evaluate:{method}", command)
    finally:
        report = {
            "config": config_path,
            "length_mode": cfg.sid.length_mode,
            "requested_stages": list(args.stages),
            "timings": timings,
            "total_seconds": float(time.perf_counter() - pipeline_started),
        }
        write_json(timing_path, report)
        print(f"[timing] wrote {timing_path}", flush=True)


if __name__ == "__main__":
    main()
