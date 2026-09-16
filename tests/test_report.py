import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from tiger_overnight.report import VARIANTS, write_comparison


def test_comparison_report_contract(tmp_path):
    config = {"paths": {"run_dir": str(tmp_path)}}
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True)
    for variant_index, variant in enumerate(VARIANTS):
        value = float(variant_index + 1) / 100.0
        payload = {
            "standard_beam": {
                "recall@10": value,
                "recall@20": value,
                "ndcg@10": value,
                "ndcg@20": value,
                "mrr@20": value,
                "Coverage": value,
                "Hits": variant_index + 1,
                "Users": 2500,
                "invalid_sid_rate": 1.0 - value,
                "mean_ms": 10.0,
                "p99_ms": 20.0,
            },
            "groups_standard_beam": {
                "history_length<=5": {"recall@20": value}
            },
            "Semantic drift": value,
            "Behavior drift": 1.0 - value,
            "Cluster KL": value,
            "Final collision rate": 0.0,
            "Codebook utilization": 0.95,
        }
        (eval_dir / f"{variant}.json").write_text(json.dumps(payload), encoding="utf-8")
    output = write_comparison(config)
    assert Path(output["comparison_csv"]).exists()
    assert Path(output["decision_summary"]).exists()
    assert Path(output["report"]).exists()
