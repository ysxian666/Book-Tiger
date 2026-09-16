import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from tiger_overnight.config import load_config
from tiger_overnight.data import preprocess_books


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_preprocess_tiny_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "overnight_5090.json")
    review_path = tmp_path / "Books.jsonl.gz"
    meta_path = tmp_path / "meta_Books.jsonl.gz"
    rows = []
    timestamp = 1
    for user_index in range(6):
        for item_index in range(6):
            rows.append({
                "user_id": f"u{user_index}",
                "parent_asin": f"i{item_index}",
                "asin": f"i{item_index}",
                "timestamp": timestamp,
                "rating": 5.0,
                "verified_purchase": True,
            })
            timestamp += 1
    _write_jsonl_gz(review_path, rows)
    _write_jsonl_gz(meta_path, [{
        "parent_asin": f"i{index}",
        "title": f"Book {index}",
        "subtitle": None,
        "author": None,
        "main_category": "Books",
        "categories": ["Books"],
        "features": [],
        "description": [],
        "details": {},
        "store": None,
        "price": None,
        "average_rating": 5.0,
        "rating_number": 1,
        "bought_together": [],
    } for index in range(6)])
    config["paths"]["review_path"] = str(review_path)
    config["paths"]["meta_path"] = str(meta_path)
    config["paths"]["data_dir"] = str(tmp_path / "data")
    config["paths"]["run_dir"] = str(tmp_path / "run")
    config["paths"]["cache_dir"] = str(tmp_path / "cache")
    config["data"].update({
        "min_user_interactions": 3,
        "min_item_interactions": 2,
        "min_sequence_length": 3,
        "user_sample_min": 1,
        "user_sample_max": 10,
        "target_interactions": 1,
        "top_items": 3,
        "duckdb_threads": 2,
        "duckdb_memory_limit": "512MB",
    })
    stats = preprocess_books(config, force=True)
    assert stats["interactions"] >= 1
    assert stats["train_interactions"] > 0
    assert stats["catalog_items"] <= 3
    assert (tmp_path / "data" / "final_data_stats.json").exists()
