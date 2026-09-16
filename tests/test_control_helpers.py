from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from tiger_overnight.generator_data import sid_dir
from tiger_overnight.semantic_controls import _hard_semantic_allowed


def test_hard_semantic_budget_limits_candidate_increase():
    raw = np.array([
        [1.0e-4, 1.05e-4],
        [2.0e-4, 2.02e-4],
    ])
    allowed, baseline_mean, budget, delta = _hard_semantic_allowed(raw, 1.05)
    assert np.isclose(baseline_mean, 1.5e-4)
    assert np.isclose(budget, 1.575e-4)
    assert np.isclose(delta, 7.5e-6)
    assert allowed.tolist() == [[True, True], [True, True]]
    strict, _, _, _ = _hard_semantic_allowed(np.array([[1.0e-4, 2.0e-4]]), 1.05)
    assert strict.tolist() == [[True, False]]


def test_sid_mode_directories_are_explicit(tmp_path):
    config = {"paths": {"run_dir": str(tmp_path)}}
    assert sid_dir(config, "baseline").name == "baseline"
    assert sid_dir(config, "bcgsid").name == "bcgsid"
    assert sid_dir(config, "bcgsid_shuffled").name == "bcgsid_shuffled"
    assert sid_dir(config, "bcgsid_hard").name == "bcgsid_hard"
