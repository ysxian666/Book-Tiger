"""Installed console entry point for the overnight pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="all", choices=["all", "preprocess", "features", "rqvae", "sid", "generators", "train", "evaluate", "eval", "report"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-exhaustive", action="store_true")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    config["_config_path"] = str(Path(args.config).resolve())
    print(json.dumps(
        run_pipeline(
            config,
            stage=args.stage,
            force=args.force,
            max_steps=args.max_steps,
            exhaustive=not args.no_exhaustive,
        ),
        ensure_ascii=False,
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
