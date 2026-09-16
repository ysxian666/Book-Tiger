#!/usr/bin/env python3
"""Plot RQ-VAE training histories, including reset and utilization diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_history(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    history = payload.get('history', payload if isinstance(payload, list) else [])
    if not history:
        raise ValueError(f'no training history in {path}')
    return payload, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--prefix', default='rqvae_training')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for training_path in args.training:
        path = Path(training_path)
        metadata, history = load_history(path)
        stem = path.parent.name if path.name == 'rqvae_training.json' else path.stem
        epochs = np.asarray([row['epoch'] for row in history], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        for key in ['reconstruction', 'commitment', 'codebook', 'hard_usage', 'joint_collision']:
            values = [row.get(key, np.nan) for row in history]
            if any(np.isfinite(values)):
                axes[0].plot(epochs, values, label=key, linewidth=1.7)
        axes[0].set_ylabel('loss')
        axes[0].set_title(f'{stem}: training losses')
        axes[0].legend(ncol=3, fontsize=8)
        axes[0].grid(alpha=0.25)

        util_keys = [key for key in history[0] if key.startswith('epoch_utilization_level_')]
        for key in util_keys:
            values = [row.get(key, np.nan) for row in history]
            axes[1].plot(epochs, values, label=key.replace('epoch_utilization_', ''), linewidth=1.7)
        axes[1].set_ylabel('hard utilization')
        axes[1].set_ylim(0.0, 1.03)
        axes[1].set_title('Per-epoch codebook utilization')
        if util_keys:
            axes[1].legend(ncol=min(6, len(util_keys)), fontsize=8)
        axes[1].grid(alpha=0.25)

        reset_keys = sorted(key for key in history[0] if key.startswith('dead_codes_reset_level_'))
        if not reset_keys:
            reset_keys = ['dead_codes_reset'] if 'dead_codes_reset' in history[0] else []
        if reset_keys:
            bottom = np.zeros_like(epochs)
            for key in reset_keys:
                values = np.asarray([row.get(key, 0.0) for row in history], dtype=float)
                axes[2].bar(epochs, values, bottom=bottom, width=2.4, label=key.replace('dead_codes_reset_', ''))
                bottom += values
            axes[2].legend(ncol=min(6, len(reset_keys)), fontsize=8)
        axes[2].set_ylabel('reset code count')
        axes[2].set_xlabel('epoch')
        axes[2].set_title('Dead-code revival')
        axes[2].grid(axis='y', alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f'{args.prefix}_{stem}.png', dpi=220)
        plt.close(fig)

    print(f'saved training plots to {out}')


if __name__ == '__main__':
    main()
