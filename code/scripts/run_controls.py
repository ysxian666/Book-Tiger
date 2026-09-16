"""Run shuffled and hard-semantic control experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from tiger_overnight.config import ensure_run_dirs, load_config
from tiger_overnight.evaluate import evaluate_variant
from tiger_overnight.generator_train import CONTROL_ORDER, train_generator
from tiger_overnight.report import write_controls_comparison
from tiger_overnight.semantic_controls import build_semantic_control_sids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "overnight_5090.json"))
    parser.add_argument("--stage", choices=["all", "sid", "train", "eval"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-exhaustive", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_run_dirs(config)
    result: dict[str, object] = {}
    if args.stage in {"all", "sid"}:
        result["sid"] = {
            mode: build_semantic_control_sids(config, mode, force=args.force)
            for mode in ("bcgsid_shuffled", "bcgsid_hard")
        }
    if args.stage in {"all", "train"}:
        result["generators"] = {
            variant: train_generator(
                config,
                variant,
                force=args.force,
                max_steps_override=args.max_steps,
            )
            for variant in CONTROL_ORDER
        }
    if args.stage in {"all", "eval"}:
        result["evaluation"] = {
            variant: evaluate_variant(
                config,
                variant,
                force=args.force,
                exhaustive=not args.no_exhaustive,
            )
            for variant in CONTROL_ORDER
        }
        result["report"] = write_controls_comparison(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
