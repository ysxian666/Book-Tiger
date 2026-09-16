#!/usr/bin/env python3
"""Generate the controlled TIGER improvement suite on quick data."""
from __future__ import annotations
import copy, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs'
BASE = json.loads((CONFIG_DIR / 'books_quick_base.json').read_text(encoding='utf-8'))


def stage(name: str, *, fusion: str, levels: int, codebook: int, latent: int,
          inverse_power: float, usage: float, entropy: float, hard: float,
          reset: bool, reallocation: str, max_conflict: int,
          kmeans: bool = True) -> None:
    cfg = copy.deepcopy(BASE)
    cfg['paths']['output_dir'] = f'tiger_amazon2023/artifacts/{name}'
    cfg['sid'].update({
        'num_levels': levels,
        'codebook_size': codebook,
        'latent_dim': latent,
        'hidden_dim': 256,
        'fusion_type': fusion,
        'gate_hidden_dim': 256,
        'epochs': 100,
        'batch_size': 1024,
        'learning_rate': 0.0005,
        'inverse_frequency_power': inverse_power,
        'inverse_frequency_clip': 10.0,
        'usage_weight': usage,
        'entropy_weight': entropy,
        'hard_usage_weight': hard,
        'dead_code_reset': reset,
        'dead_code_reset_interval': 10,
        'dead_code_threshold': 0.0,
        'dead_code_noise': 0.02,
        'dead_code_reset_full_data': reset,
        'dead_code_reset_error_aware': reset,
        'dead_code_reset_usage': 'epoch',
        'joint_collision_weight': 0.0,
        'kmeans_init': kmeans,
        'max_conflict_tokens': max_conflict,
        'topk_for_conflict': 64,
        'conflict_reallocation_mode': reallocation,
        'length_mode': 'fixed',
    })
    cfg['generator'].update({
        'architecture': 't5-tiny',
        'd_model': 256,
        'd_ff': 1024,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'max_history_items': 20,
        'max_input_tokens': 256,
        'max_target_tokens': 8,
        'epochs': 100,
        'early_stopping_patience': 10,
        'batch_size': 64,
        'num_beams': 5,
        'max_decode_steps': 8,
    })
    (CONFIG_DIR / f'{name}.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build_suite() -> None:
    common = dict(codebook=256, latent=16, inverse_power=0.0, usage=0.0,
                  entropy=0.0, hard=0.0, reset=False, reallocation='last',
                  max_conflict=16384)
    stage('books_improve_00_content_baseline', fusion='content_only', levels=3, **common)
    stage('books_improve_01_gate', fusion='gate', levels=3, **common)
    stage('books_improve_02_frequency', fusion='gate', levels=3,
          codebook=256, latent=16, inverse_power=0.5, usage=0.0, entropy=0.0,
          hard=0.0, reset=False, reallocation='last', max_conflict=16384)
    stage('books_improve_03_usage', fusion='gate', levels=3,
          codebook=256, latent=16, inverse_power=0.5, usage=0.05, entropy=0.001,
          hard=0.0, reset=False, reallocation='last', max_conflict=16384)
    stage('books_improve_04_hard_reset', fusion='gate', levels=3,
          codebook=256, latent=16, inverse_power=0.5, usage=0.05, entropy=0.001,
          hard=0.2, reset=True, reallocation='last', max_conflict=16384)
    stage('books_improve_05_geometry', fusion='gate', levels=4,
          codebook=256, latent=4, inverse_power=0.5, usage=0.05, entropy=0.001,
          hard=0.2, reset=True, reallocation='last', max_conflict=16384)
    stage('books_improve_06_collision_all', fusion='gate', levels=4,
          codebook=256, latent=4, inverse_power=0.5, usage=0.05, entropy=0.001,
          hard=0.2, reset=True, reallocation='all', max_conflict=16384)
    stage('books_improve_07_min_vocab', fusion='gate', levels=4,
          codebook=256, latent=4, inverse_power=0.5, usage=0.05, entropy=0.001,
          hard=0.2, reset=True, reallocation='all', max_conflict=1)
    stage('books_improve_a1_cross_attention', fusion='cross_attention', levels=3,
          codebook=256, latent=16, inverse_power=0.0, usage=0.0, entropy=0.0,
          hard=0.0, reset=False, reallocation='last', max_conflict=16384)
    stage('books_improve_a2_frequency_strong', fusion='gate', levels=3,
          codebook=256, latent=16, inverse_power=1.0, usage=0.05, entropy=0.001,
          hard=0.2, reset=True, reallocation='last', max_conflict=16384)
    stage('books_improve_a3_geometry5', fusion='gate', levels=5,
          codebook=256, latent=4, inverse_power=0.5, usage=0.05, entropy=0.001,
          hard=0.2, reset=True, reallocation='all', max_conflict=1)
    stage('books_improve_a4_kmeans_off', fusion='gate', levels=4,
          codebook=256, latent=4, inverse_power=0.5, usage=0.05, entropy=0.001,
          hard=0.2, reset=True, reallocation='all', max_conflict=1,
          kmeans=False)


if __name__ == '__main__':
    build_suite()
    print('generated improvement configs')
