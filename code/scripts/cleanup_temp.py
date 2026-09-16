"""Remove disposable caches while retaining all experiment artifacts."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--keep-duckdb", action="store_true")
    parser.add_argument("--remove-latest", action="store_true")
    parser.add_argument("--remove-probe", action="store_true")
    parser.add_argument("--remove-hf-cache", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    targets = [root / ".pytest_cache", root / "cache" / "overnight", root / "tmp_debug", root / "tmp_patch"]
    targets.extend(root.rglob("__pycache__"))
    targets.extend((root / "data" / "raw").glob("*.part"))
    if not args.keep_duckdb:
        targets.append(root / "data" / "overnight" / "duckdb_tmp")
    if args.remove_latest:
        targets.extend(root.rglob("*_latest.pt"))
    if args.remove_probe:
        targets.append(root / "results" / "probe_rqvae")
        targets.append(root / "dist")
    if args.remove_hf_cache:
        targets.append(root / ".cache")
    removed = []
    for target in sorted(set(targets), key=lambda value: len(value.parts), reverse=True):
        resolved = target.resolve()
        if root not in resolved.parents and resolved != root:
            continue
        if resolved.is_dir():
            shutil.rmtree(resolved, ignore_errors=True)
            removed.append(str(resolved))
        elif resolved.is_file():
            resolved.unlink(missing_ok=True)
            removed.append(str(resolved))
    print("\n".join(removed) if removed else "nothing to clean")


if __name__ == "__main__":
    main()
