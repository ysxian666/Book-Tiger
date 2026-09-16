from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from tiger_overnight.config import load_config


def test_overnight_contract():
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "overnight_5090.json")
    assert config["rqvae"]["num_levels"] == 3
    assert config["rqvae"]["codebook_size"] == 256
    assert config["rqvae"]["latent_dim"] == 32
    assert config["sid"]["sid_length"] == 4
    assert config["generator"]["num_user_buckets"] == 2000
    assert config["generator"]["num_beams"] == 20
    assert config["seed"] == 42
