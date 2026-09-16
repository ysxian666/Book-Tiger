import json

from tiger_rec.config import Config
from tiger_rec.data.dataset import AmazonSequenceDataset
from tiger_rec.data.preprocess import run_preprocessing
from tiger_rec.features.collaborative import build_collaborative_embeddings
from tiger_rec.features.text_encoder import encode_catalog
from tiger_rec.evaluation.protocol import evaluate_retriever
from tiger_rec.retrieval.traditional import ItemCFRetriever, PopularityRetriever
from tiger_rec.semantic_id.build import build_semantic_id_artifacts, train_rqvae


def write_synthetic(root):
    review_path = root / "reviews.jsonl"
    meta_path = root / "meta.jsonl"
    with review_path.open("w", encoding="utf-8") as f:
        for user_index in range(10):
            for position in range(6):
                item_id = f"item_{(user_index + position) % 10}"
                f.write(json.dumps({
                    "rating": 5.0,
                    "title": "title",
                    "text": f"text {item_id}",
                    "asin": item_id,
                    "parent_asin": item_id,
                    "user_id": f"user_{user_index}",
                    "timestamp": 1_700_000_000_000 + user_index * 1000 + position,
                    "verified_purchase": True,
                }) + "\n")
    with meta_path.open("w", encoding="utf-8") as f:
        for index in range(10):
            f.write(json.dumps({
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
                "parent_asin": f"item_{index}",
                "bought_together": [],
            }) + "\n")
    return review_path, meta_path


def test_semantic_id_pipeline(tmp_path):
    review_path, meta_path = write_synthetic(tmp_path)
    cfg = Config()
    cfg.paths.review_path = str(review_path)
    cfg.paths.meta_path = str(meta_path)
    cfg.paths.output_dir = str(tmp_path / "artifacts")
    cfg.paths.project_root = str(tmp_path)
    cfg.data.min_user_interactions = 5
    cfg.data.min_item_interactions = 2
    cfg.data.min_sequence_length = 5
    cfg.data.kcore_iterations = 2
    cfg.data.duckdb_memory_limit = "1GB"
    cfg.text.model_name = "hash"
    cfg.text.output_dim = 8
    cfg.collaborative.svd_components = 4
    cfg.sid.epochs = 1
    cfg.sid.batch_size = 4
    cfg.sid.hidden_dim = 8
    cfg.sid.latent_dim = 4
    cfg.sid.codebook_size = 4
    cfg.sid.num_levels = 2
    cfg.sid.kmeans_init = False
    cfg.sid.max_conflict_tokens = 4
    cfg.sid.topk_for_conflict = 2
    run_preprocessing(cfg)
    encode_catalog(cfg)
    build_collaborative_embeddings(cfg)
    train_rqvae(cfg)
    report = build_semantic_id_artifacts(cfg)
    assert report["items"] == 10
    assert report["unresolved"] == 0
    semantic_ids = json.loads(cfg.paths.artifact("semantic_ids.json").read_text(encoding="utf-8"))
    assert len(semantic_ids) == 10
    train_dataset = AmazonSequenceDataset(cfg, "train")
    test_dataset = AmazonSequenceDataset(cfg, "test")
    assert len(train_dataset) > 0
    assert len(test_dataset) == 10
    sample = test_dataset[0]
    assert sample["input_ids"].numel() > 0
    metrics = evaluate_retriever(cfg, PopularityRetriever(cfg), split="test")
    assert metrics["users"] == 10.0
    itemcf_metrics = evaluate_retriever(cfg, ItemCFRetriever(cfg), split="test")
    assert itemcf_metrics["users"] == 10.0
