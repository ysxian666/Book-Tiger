"""Configuration loader for the RTX 5090 overnight TIGER screening."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an overnight experiment config is invalid."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_path(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    root_value = str(config.get("project_root", "."))
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    else:
        root = root.resolve()
    config["project_root"] = str(root)
    paths = config.setdefault("paths", {})
    for key in ("review_path", "meta_path", "data_dir", "run_dir", "cache_dir"):
        value = paths.get(key)
        if value:
            paths[key] = _resolve_path(root, str(value))
    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    sid = config["rqvae"]
    if int(sid["num_levels"]) != 3 or int(sid["codebook_size"]) != 256:
        raise ConfigError("overnight run requires 3 semantic levels and codebook size 256")
    if int(sid["latent_dim"]) != 32:
        raise ConfigError("overnight run requires latent_dim=32")
    if int(config["sid"]["sid_length"]) != 4:
        raise ConfigError("overnight run requires SID length 4")
    if int(config["generator"]["num_beams"]) != 20:
        raise ConfigError("overnight run requires beam=20")
    if int(config["seed"]) != 42:
        raise ConfigError("overnight run requires seed=42")


def ensure_run_dirs(config: dict[str, Any]) -> None:
    paths = config["paths"]
    for value in (paths["data_dir"], paths["run_dir"], paths["cache_dir"]):
        Path(value).mkdir(parents=True, exist_ok=True)
    for relative in (
        "rqvae", "sid/baseline", "sid/bcgsid", "generators", "eval", "logs", "metadata"
    ):
        (Path(paths["run_dir"]) / relative).mkdir(parents=True, exist_ok=True)


def artifact_path(config: dict[str, Any], *parts: str) -> Path:
    return Path(config["paths"]["run_dir"]).joinpath(*parts)


def data_path(config: dict[str, Any], *parts: str) -> Path:
    return Path(config["paths"]["data_dir"]).joinpath(*parts)
