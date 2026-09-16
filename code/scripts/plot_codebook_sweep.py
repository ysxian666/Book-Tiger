#!/usr/bin/env python3
"""Plot a codebook tuning summary produced by summarize_codebook_tuning.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def short_name(name: str) -> str:
    return name.replace('books_tune_', '').replace('books_tune4_', '').replace('_', ' ')


def pareto_mask(rows: list[dict]) -> list[bool]:
    mask = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            no_worse = (
                other['base_collision_rate'] <= row['base_collision_rate']
                and other['mean_utilization'] >= row['mean_utilization']
            )
            strictly_better = (
                other['base_collision_rate'] < row['base_collision_rate']
                or other['mean_utilization'] > row['mean_utilization']
            )
            if no_worse and strictly_better:
                dominated = True
                break
        mask.append(not dominated)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--prefix', default='codebook_sweep')
    args = parser.parse_args()

    payload = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    rows = [row for row in payload['successful'] if row.get('success')]
    if not rows:
        raise SystemExit('summary has no successful rows')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    collision = np.array([row['base_collision_rate'] for row in rows])
    utilization = np.array([row['mean_utilization'] for row in rows])
    suffix_ratio = np.array([row['suffix_resolved_ratio'] for row in rows])
    unique_ratio = np.array([row['unique_base_code_ratio'] for row in rows])
    pareto = np.array(pareto_mask(rows))

    fig, ax = plt.subplots(figsize=(11, 7))
    codebooks = sorted({int(row['codebook_size']) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(codebooks)))
    color_map = {value: colors[i] for i, value in enumerate(codebooks)}
    for row, x, y, keep in zip(rows, collision, utilization, pareto):
        ax.scatter(
            x, y,
            s=110 if keep else 65,
            marker='*' if keep else 'o',
            color=color_map[int(row['codebook_size'])],
            edgecolor='black' if keep else 'none',
            linewidth=0.8,
            alpha=0.9,
            zorder=3 if keep else 2,
        )
        ax.annotate(short_name(row['name']), (x, y), fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Base SID collision rate (lower is better)')
    ax.set_ylabel('Mean per-level codebook utilization (higher is better)')
    ax.set_title('Codebook collapse vs. joint SID collision')
    ax.grid(alpha=0.25)
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=color_map[k], label=f'K={k}') for k in codebooks]
    handles.append(plt.Line2D([0], [0], marker='*', linestyle='', color='none', markerfacecolor='gray', markeredgecolor='black', label='Pareto candidate'))
    ax.legend(handles=handles, loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f'{args.prefix}_scatter.png', dpi=220)
    plt.close(fig)

    order = np.argsort(collision)
    ordered = [rows[i] for i in order]
    labels = [short_name(row['name']) for row in ordered]
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    axes[0].bar(x, collision[order], color='#c0392b')
    axes[0].set_ylabel('Collision')
    axes[0].set_title('Lower is better')
    axes[1].bar(x, unique_ratio[order], color='#2471a3')
    axes[1].set_ylabel('Unique base code ratio')
    axes[1].set_title('Higher is better')
    axes[2].bar(x, suffix_ratio[order], color='#d68910')
    axes[2].set_ylabel('Suffix-resolved ratio')
    axes[2].set_title('Lower is better')
    for ax in axes:
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(x, labels, rotation=45, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f'{args.prefix}_bars.png', dpi=220)
    plt.close(fig)

    max_levels = max(len(row['utilization']) for row in rows)
    matrix = np.full((len(rows), max_levels), np.nan)
    for i, row in enumerate(rows):
        values = [row['utilization'].get(f'utilization_level_{level}', np.nan) for level in range(max_levels)]
        matrix[i, :] = values
    order = np.argsort(-utilization)
    matrix = matrix[order]
    labels = [short_name(rows[i]['name']) for i in order]
    fig, ax = plt.subplots(figsize=(10, max(5, len(rows) * 0.48)))
    image = ax.imshow(matrix, aspect='auto', cmap='YlGnBu', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(max_levels), [f'L{level}' for level in range(max_levels)])
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title('Per-level codebook utilization')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(image, ax=ax, label='utilization')
    fig.tight_layout()
    fig.savefig(out / f'{args.prefix}_utilization_heatmap.png', dpi=220)
    plt.close(fig)

    print(f'saved plots to {out}')


if __name__ == '__main__':
    main()
