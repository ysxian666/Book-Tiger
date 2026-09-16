from pathlib import Path

from tiger_rec.config import load_config


def test_load_release_config():
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "baseline.json")
    assert cfg.paths.root.exists()
    assert cfg.sid.num_levels == 3
    assert cfg.text.model_name == "hyp1231/blair-roberta-base"
