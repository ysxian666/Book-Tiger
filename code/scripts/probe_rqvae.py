"""Short codebook-health probe before the full overnight RQ-VAE run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from tiger_overnight.config import ensure_run_dirs, load_config
from tiger_overnight.rqvae_train import train_rqvae


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "overnight_5090.json")
    config["paths"]["run_dir"] = str(PROJECT_ROOT / "results" / "probe_rqvae")
    config["rqvae"].update({
        "epochs": 200,
        "usage_weight": 0.5,
        "hard_usage_weight": 1.0,
        "entropy_weight": 0.05,
        "temperature": 1.0,
        "dead_code_reset_interval": 10,
        "fused_std_floor": 0.02,
        "latent_std_floor": 0.02,
        "diversity_weight": 5.0,
    })
    ensure_run_dirs(config)
    result = train_rqvae(config, force=True)
    print(json.dumps({
        "completed_epochs": result["config"]["completed_epochs"],
        "final": result["final"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
