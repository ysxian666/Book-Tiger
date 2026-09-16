#!/usr/bin/env python3
"""Link prepare/encode artifacts from one experiment into another."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tiger_rec.config import load_config

SHARED_ARTIFACTS = [
    'data_stats.json',
    'run_config.json',
    'interactions.parquet',
    'train_interactions.parquet',
    'valid_interactions.parquet',
    'test_interactions.parquet',
    'user_sequences.parquet',
    'item_stats.parquet',
    'catalog.parquet',
    'text_embeddings.npy',
    'text_item_ids.json',
    'text_embeddings_meta.json',
    'collab_embeddings.npy',
    'collab_item_ids.json',
    'collab_embeddings_meta.json',
    'item_embedding_manifest.json',
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-config', required=True)
    parser.add_argument('--target-config', required=True)
    parser.add_argument('--copy', action='store_true', help='copy files instead of creating symlinks')
    args = parser.parse_args()

    source = load_config(args.source_config)
    target = load_config(args.target_config)
    if source.paths.out == target.paths.out:
        raise SystemExit('source and target output directories must differ')
    target.paths.out.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in SHARED_ARTIFACTS:
        src = source.paths.artifact(name)
        dst = target.paths.artifact(name)
        if not src.exists():
            missing.append(name)
            continue
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                raise SystemExit(f'refusing to replace directory: {dst}')
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)
        print(f'{name}: {dst} -> {src}' if not args.copy else f'{name}: copied {src} -> {dst}')

    if missing:
        print('missing shared artifacts:')
        for name in missing:
            print(f'- {name}')
        return 1
    print(f'linked {len(SHARED_ARTIFACTS)} shared artifacts into {target.paths.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
