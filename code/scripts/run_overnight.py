"""CLI entry point for the RTX 5090 overnight pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from tiger_overnight.config import load_config
from tiger_overnight.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "overnight_5090.json"))
    parser.add_argument("--stage", default="all", choices=["all", "preprocess", "features", "rqvae", "sid", "generators", "train", "evaluate", "eval", "report"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional smoke-test override; never use for the final 5090 run.")
    parser.add_argument("--no-exhaustive", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config["_config_path"] = str(Path(args.config).resolve())
    result = run_pipeline(config, stage=args.stage, force=args.force, max_steps=args.max_steps, exhaustive=not args.no_exhaustive)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
