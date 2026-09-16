"""Typed configuration loading for the project.

All experiment parameters live in JSON files so that a full run can be
reproduced without editing Python code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints


T = TypeVar("T")


@dataclass
class PathConfig:
    """Input and output paths. Relative paths are resolved from project_root."""

    project_root: str = "."
    review_path: str = "data/review_Books.jsonl"
    meta_path: str = "data/meta_Books.jsonl"
    output_dir: str = "artifacts/books_debug"

    def resolve(self, base_dir: Path) -> "PathConfig":
        root = Path(self.project_root).expanduser()
        if not root.is_absolute():
            root = (base_dir / root).resolve()
        self.project_root = str(root)
        self.review_path = str((root / self.review_path).resolve()) if not Path(self.review_path).is_absolute() else str(Path(self.review_path).resolve())
        self.meta_path = str((root / self.meta_path).resolve()) if not Path(self.meta_path).is_absolute() else str(Path(self.meta_path).resolve())
        self.output_dir = str((root / self.output_dir).resolve()) if not Path(self.output_dir).is_absolute() else str(Path(self.output_dir).resolve())
        return self

    @property
    def root(self) -> Path:
        return Path(self.project_root)

    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    def artifact(self, *parts: str) -> Path:
        return self.out.joinpath(*parts)


@dataclass
class DataConfig:
    min_rating: float = 1.0
    require_verified_purchase: bool = False
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    kcore_iterations: int = 4
    min_sequence_length: int = 7
    max_reviews: int = 0
    max_users: int = 0
    max_items: int = 0
    max_meta_records: int = 0
    duckdb_memory_limit: str = "8GB"
    duckdb_threads: int = 8
    parquet_compression: str = "zstd"


@dataclass
class TextConfig:
    # "hash" gives a deterministic lightweight baseline for smoke tests.
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 128
    max_length: int = 256
    output_dim: int = 384
    normalize: bool = True
    query_prefix: str = ""
    passage_prefix: str = ""
    trust_remote_code: bool = False
    allow_hash_fallback: bool = True


@dataclass
class CollaborativeConfig:
    dim: int = 128
    svd_components: int = 128
    min_user_degree: int = 2
    normalize: bool = True
    random_state: int = 42


@dataclass
class RQVAEConfig:
    num_levels: int = 3
    codebook_size: int = 256
    text_dim: int = 384
    collab_dim: int = 128
    hidden_dim: int = 512
    latent_dim: int = 256
    fusion_type: str = "gate"  # gate | cross_attention | content_only
    gate_hidden_dim: int = 256
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    commitment_weight: float = 0.25
    codebook_weight: float = 0.25
    usage_weight: float = 0.01
    entropy_weight: float = 0.001
    hard_usage_weight: float = 0.0
    joint_collision_weight: float = 0.0
    dead_code_reset: bool = False
    dead_code_reset_interval: int = 10
    dead_code_threshold: float = 0.0
    dead_code_noise: float = 0.01
    dead_code_reset_full_data: bool = False
    dead_code_reset_error_aware: bool = False
    dead_code_reset_usage: str = "ema"  # ema | epoch
    usage_ema_decay: float = 0.9
    inverse_frequency_power: float = 0.5
    inverse_frequency_clip: float = 10.0
    temperature: float = 0.1
    kmeans_init: bool = True
    kmeans_max_iter: int = 100
    max_conflict_tokens: int = 16
    topk_for_conflict: int = 8
    conflict_reallocation_mode: str = "last"  # last | all
    length_mode: str = "variable"  # variable | fixed
    seed: int = 42


@dataclass
class GeneratorConfig:
    architecture: str = "t5-small"  # t5-tiny | t5-small | custom
    d_model: int = 512
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_history_items: int = 20
    max_input_tokens: int = 256
    max_target_tokens: int = 8
    use_time_buckets: bool = True
    use_user_tokens: bool = False
    num_user_buckets: int = 512
    num_time_buckets: int = 16
    use_behavior_tokens: bool = True
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    num_workers: int = 0
    early_stopping_patience: int = 4
    checkpoint_name: str = "tiger_generator.pt"
    decoder: str = "standard"  # standard | constrained
    num_beams: int = 20
    length_penalty: float = 1.0
    max_decode_steps: int = 8


@dataclass
class BaselineConfig:
    models: list[str] = field(default_factory=lambda: ["popularity", "itemcf", "twotower", "gru4rec", "sasrec", "bert4rec"])
    dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    max_history_items: int = 50
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    num_negatives: int = 64
    temperature: float = 0.07
    bert_mask_prob: float = 0.2
    early_stopping_patience: int = 5
    checkpoint_name: str = "baselines.pt"


@dataclass
class RankerConfig:
    type: str = "logistic"  # logistic | rrf
    max_candidates: int = 500
    train_negative_ratio: int = 20
    c_value: float = 1.0
    random_state: int = 42
    checkpoint_name: str = "ranker.joblib"


@dataclass
class EvaluationConfig:
    top_k: list[int] = field(default_factory=lambda: [10, 20, 50])
    primary_k: int = 20
    ranker_k: int = 10
    long_tail_quantile: float = 0.8
    cold_start_max_train_count: int = 5
    batch_size: int = 64
    metrics: list[str] = field(default_factory=lambda: [
        "recall", "ndcg", "mrr", "coverage", "long_tail_recall",
        "cold_start_recall", "invalid_sid_rate", "latency"
    ])


@dataclass
class Config:
    seed: int = 42
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    collaborative: CollaborativeConfig = field(default_factory=CollaborativeConfig)
    sid: RQVAEConfig = field(default_factory=RQVAEConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        data = dict(raw)
        data["paths"] = _populate_dataclass(PathConfig, data.get("paths", {}))
        data["data"] = _populate_dataclass(DataConfig, data.get("data", {}))
        data["text"] = _populate_dataclass(TextConfig, data.get("text", {}))
        data["collaborative"] = _populate_dataclass(CollaborativeConfig, data.get("collaborative", {}))
        data["sid"] = _populate_dataclass(RQVAEConfig, data.get("sid", {}))
        data["generator"] = _populate_dataclass(GeneratorConfig, data.get("generator", {}))
        data["baselines"] = _populate_dataclass(BaselineConfig, data.get("baselines", {}))
        data["ranker"] = _populate_dataclass(RankerConfig, data.get("ranker", {}))
        data["evaluation"] = _populate_dataclass(EvaluationConfig, data.get("evaluation", {}))
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


def _populate_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    if is_dataclass(values):
        return values  # type: ignore[return-value]
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")
    return cls(**values)


def _dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _dataclass_to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: _dataclass_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dataclass_to_dict(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = Config.from_dict(raw)
    cfg.paths.resolve(config_path.parent)
    cfg.paths.out.mkdir(parents=True, exist_ok=True)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
