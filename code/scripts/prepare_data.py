"""Prepare Amazon Books interactions, leave-one-out splits, and catalog."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiger_rec.config import load_config
from tiger_rec.data.preprocess import run_preprocessing
from tiger_rec.utils import set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    stats = run_preprocessing(cfg)
    write_json(cfg.paths.artifact("run_config.json"), cfg.to_dict())
    print(stats)


if __name__ == "__main__":
    main()
