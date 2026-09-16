"""Link existing official Books files or download them when absent."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path


FILES = {
    "Books.jsonl.gz": {
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz",
        "expected_bytes": 6216820644,
        "existing_candidates": [Path(r"D:\SoftWareSpace\Projects\book\Books.jsonl.gz")],
    },
    "meta_Books.jsonl.gz": {
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz",
        "expected_bytes": 4942125770,
        "existing_candidates": [Path(r"D:\SoftWareSpace\Projects\book\meta_Books.jsonl.gz")],
    },
}


def _verify_size(path: Path, expected: int, strict: bool) -> None:
    actual = path.stat().st_size
    if int(actual) != int(expected):
        message = f"size mismatch for {path}: expected={expected} actual={actual}"
        if strict:
            raise RuntimeError(message)
        print(f"[data][warning] {message}")


def _download(url: str, target: Path, expected_bytes: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "tiger-amazon2023-overnight/0.2"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024 * 16)
    _verify_size(temporary, expected_bytes, strict=True)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-size", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for name, spec in FILES.items():
        target = raw / name
        if target.exists() and not args.force:
            _verify_size(target, int(spec["expected_bytes"]), args.strict_size)
            records.append({"file": str(target), "action": "existing", "official_url": spec["url"], "expected_bytes": int(spec["expected_bytes"]), "bytes": target.stat().st_size})
            continue
        linked = False
        for candidate in spec["existing_candidates"]:
            if candidate.exists():
                try:
                    os.link(candidate, target)
                    action = "hardlink"
                except OSError:
                    shutil.copy2(candidate, target)
                    action = "copy"
                _verify_size(target, int(spec["expected_bytes"]), args.strict_size)
                records.append({"file": str(target), "action": action, "source": str(candidate), "official_url": spec["url"], "expected_bytes": int(spec["expected_bytes"]), "bytes": target.stat().st_size})
                linked = True
                break
        if not linked:
            print(f"[data] downloading {spec['url']}")
            _download(str(spec["url"]), target, int(spec["expected_bytes"]))
            records.append({"file": str(target), "action": "download", "url": spec["url"], "expected_bytes": int(spec["expected_bytes"]), "bytes": target.stat().st_size})
    with (raw / "SOURCE.json").open("w", encoding="utf-8") as handle:
        json.dump({"files": records}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
