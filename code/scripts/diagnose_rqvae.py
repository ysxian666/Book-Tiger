"""Diagnose latent and codebook health for a trained RQ-VAE checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from tiger_overnight.config import load_config
from tiger_overnight.rqvae_train import load_aligned_item_features
from tiger_rec.semantic_id.rqvae import load_rqvae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    item_ids, text, collab, _counts = load_aligned_item_features(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_rqvae(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    text_tensor = torch.from_numpy(text).to(device)
    collab_tensor = torch.from_numpy(collab).to(device)
    with torch.inference_mode():
        fused = model.fusion(text_tensor, collab_tensor)
        latent = model.encoder(fused)
        quantized = model.quantizer(latent, topk=1)
    latent_np = latent.float().cpu().numpy()
    fused_np = fused.float().cpu().numpy()
    codes = quantized.codes.cpu().numpy()
    report = {
        "items": len(item_ids),
        "metadata": metadata,
        "latent_mean_norm": float(np.linalg.norm(latent_np.mean(axis=0))),
        "latent_mean_pairwise_distance": float(np.linalg.norm(latent_np - latent_np.mean(axis=0, keepdims=True), axis=1).mean()),
        "latent_per_dimension_std_mean": float(latent_np.std(axis=0).mean()),
        "fused_per_dimension_std_mean": float(fused_np.std(axis=0).mean()),
        "text_per_dimension_std_mean": float(text.std(axis=0).mean()),
        "collab_per_dimension_std_mean": float(collab.std(axis=0).mean()),
        "columns": {},
    }
    for level in range(codes.shape[1]):
        values, frequencies = np.unique(codes[:, level], return_counts=True)
        order = np.argsort(-frequencies)
        report["columns"][f"level_{level}"] = {
            "unique_codes": int(len(values)),
            "utilization": float(len(values) / int(config["rqvae"]["codebook_size"])),
            "top_codes": [
                {"code": int(values[index]), "count": int(frequencies[index])}
                for index in order[:10]
            ],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
