import json

import pyarrow.parquet as pq

from tiger_rec.config import Config
from tiger_rec.data.preprocess import run_preprocessing


def test_preprocess_synthetic_books(tmp_path):
    review_path = tmp_path / "reviews.jsonl"
    meta_path = tmp_path / "meta.jsonl"
    with review_path.open("w", encoding="utf-8") as f:
        for user_index in range(8):
            for position in range(6):
                item_id = f"item_{(user_index + position) % 8}"
                row = {
                    "rating": 5.0,
                    "title": "good",
                    "text": "good book",
                    "asin": item_id,
                    "parent_asin": item_id,
                    "user_id": f"user_{user_index}",
                    "timestamp": 1_700_000_000_000 + user_index * 10_000 + position,
                    "helpful_vote": 0,
                    "verified_purchase": True,
                }
                f.write(json.dumps(row) + "\n")
    with meta_path.open("w", encoding="utf-8") as f:
        for index in range(8):
            item_id = f"item_{index}"
            row = {
                "main_category": "Books",
                "title": f"Book {index}",
                "subtitle": "",
                "author": "Author",
                "average_rating": 4.5,
                "rating_number": 10,
                "features": ["feature"],
                "description": ["description"],
                "price": 9.99,
                "images": [],
                "videos": [],
                "store": "Store",
                "categories": ["Books", "Test"],
                "details": {"Language": "English"},
                "parent_asin": item_id,
                "bought_together": [],
            }
            f.write(json.dumps(row) + "\n")

    cfg = Config()
    cfg.paths.review_path = str(review_path)
    cfg.paths.meta_path = str(meta_path)
    cfg.paths.output_dir = str(tmp_path / "artifacts")
    cfg.paths.project_root = str(tmp_path)
    cfg.data.min_user_interactions = 5
    cfg.data.min_item_interactions = 2
    cfg.data.min_sequence_length = 5
    cfg.data.kcore_iterations = 3
    cfg.data.duckdb_memory_limit = "1GB"
    stats = run_preprocessing(cfg)
    assert stats["users"] == 8
    assert stats["catalog_items"] == 8
    train = pq.read_table(cfg.paths.artifact("train_interactions.parquet"))
    valid = pq.read_table(cfg.paths.artifact("valid_interactions.parquet"))
    test = pq.read_table(cfg.paths.artifact("test_interactions.parquet"))
    assert train.num_rows == 32
    assert valid.num_rows == 8
    assert test.num_rows == 8
