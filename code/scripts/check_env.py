"""Preflight checks for the RTX 5090 cloud run."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from tiger_overnight.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "overnight_5090.json"))
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checks: dict[str, object] = {}
    try:
        import torch
        checks["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        checks["cuda_version"] = torch.version.cuda
        checks["bf16_supported"] = bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False
        if torch.cuda.is_available():
            checks["device"] = torch.cuda.get_device_name(0)
            checks["capability"] = list(torch.cuda.get_device_capability(0))
            checks["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            checks["cuda_arch_list"] = list(torch.cuda.get_arch_list())
            checks["supports_sm120"] = any("sm_120" in value for value in torch.cuda.get_arch_list())
    except Exception as exc:
        checks["torch_error"] = str(exc)
    try:
        import duckdb
        import transformers
        checks["duckdb"] = duckdb.__version__
        checks["transformers"] = transformers.__version__
    except Exception as exc:
        checks["dependency_error"] = str(exc)
    run_dir = Path(config["paths"]["run_dir"])
    free = shutil.disk_usage(run_dir if run_dir.exists() else PROJECT_ROOT).free
    checks["free_disk_gb"] = round(free / 1024**3, 2)
    checks["min_free_disk_gb"] = float(config.get("runtime", {}).get("min_free_disk_gb", 25))
    checks["review_exists"] = Path(config["paths"]["review_path"]).exists()
    checks["meta_exists"] = Path(config["paths"]["meta_path"]).exists()
    checks["contract"] = {
        "sid_levels": int(config["rqvae"]["num_levels"]),
        "codebook_size": int(config["rqvae"]["codebook_size"]),
        "latent_dim": int(config["rqvae"]["latent_dim"]),
        "sid_length": int(config["sid"]["sid_length"]),
        "user_token_buckets": int(config["generator"]["num_user_buckets"]),
        "beam": int(config["generator"]["num_beams"]),
        "seed": int(config["seed"]),
    }
    checks["passed"] = bool(
        checks.get("cuda_available", False)
        and checks.get("bf16_supported", False)
        and checks.get("supports_sm120", False)
        and checks.get("review_exists", False)
        and checks.get("meta_exists", False)
        and checks.get("free_disk_gb", 0) >= checks.get("min_free_disk_gb", 25)
    ) or bool(args.allow_cpu)
    output_path = run_dir / "metadata" / "environment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, ensure_ascii=False, indent=2)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
