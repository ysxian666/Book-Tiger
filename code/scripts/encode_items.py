"""Encode item text with BLaIR/BGE-style encoders and learn collaborative vectors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiger_rec.config import load_config
from tiger_rec.features.collaborative import build_collaborative_embeddings
from tiger_rec.features.text_encoder import encode_catalog
from tiger_rec.utils import set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-collab", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    result = {}
    if not args.skip_text:
        result["text"] = encode_catalog(cfg)
    if not args.skip_collab:
        result["collaborative"] = build_collaborative_embeddings(cfg)
    write_json(cfg.paths.artifact("item_embedding_manifest.json"), result)
    print(result)


if __name__ == "__main__":
    main()
