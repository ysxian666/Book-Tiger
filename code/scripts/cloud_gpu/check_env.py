#!/usr/bin/env python3
"""Check the local/cloud environment before a long TIGER run."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def gib(value: int) -> float:
    return float(value) / (1024 ** 3)


def system_memory_bytes() -> int | None:
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    return None


def check_file(label: str, path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"{label} missing: {path}")
        return
    print(f"{label}: {path} ({gib(path.stat().st_size):.2f} GiB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/books_cloud_smoke.json")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=150.0)
    args = parser.parse_args()

    os.chdir(ROOT)
    errors: list[str] = []
    print(f"project_root: {ROOT}")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {platform.platform()}")
    print(f"cwd: {Path.cwd()}")

    try:
        import duckdb
        import pyarrow
        import torch
        import transformers
    except Exception as exc:
        errors.append(f"dependency import failed: {exc!r}")
        duckdb = pyarrow = torch = transformers = None

    if torch is not None:
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"torch_cuda_version: {torch.version.cuda}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                print(
                    f"gpu[{index}]: {props.name}, "
                    f"{gib(int(props.total_memory)):.2f} GiB, capability={props.major}.{props.minor}"
                )
        elif args.require_cuda:
            errors.append("CUDA is required but torch.cuda.is_available() is false")
    if transformers is not None:
        print(f"transformers: {transformers.__version__}")
    if duckdb is not None:
        print(f"duckdb: {duckdb.__version__}")
    if pyarrow is not None:
        print(f"pyarrow: {pyarrow.__version__}")

    memory = system_memory_bytes()
    if memory is not None:
        print(f"system_ram: {gib(memory):.2f} GiB")

    usage = shutil.disk_usage(ROOT)
    print(f"disk_free: {gib(usage.free):.2f} GiB")
    if gib(usage.free) < args.min_free_gib:
        errors.append(
            f"only {gib(usage.free):.2f} GiB free on {ROOT}; "
            f"recommended minimum is {args.min_free_gib:.0f} GiB"
        )

    try:
        sys.path.insert(0, str(ROOT))
        from tiger_rec.config import load_config

        cfg = load_config(args.config)
        print(f"config: {Path(args.config).resolve()}")
        print(f"data_project_root: {cfg.paths.root}")
        print(f"output_dir: {cfg.paths.out}")
        check_file("review_path", Path(cfg.paths.review_path), errors)
        check_file("meta_path", Path(cfg.paths.meta_path), errors)
    except Exception as exc:
        errors.append(f"config check failed: {exc!r}")

    if errors:
        print("\nFAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nOK: environment and configured inputs are usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
