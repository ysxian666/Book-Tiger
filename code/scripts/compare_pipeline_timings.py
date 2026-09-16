"""Compare two ``pipeline_timing.json`` files stage by stage."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_timing(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variable_timing")
    parser.add_argument("fixed_timing")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    variable = load_timing(args.variable_timing)
    fixed = load_timing(args.fixed_timing)
    variable_stages = {row["stage"]: float(row["seconds"]) for row in variable.get("timings", [])}
    fixed_stages = {row["stage"]: float(row["seconds"]) for row in fixed.get("timings", [])}
    stages = list(dict.fromkeys([*variable_stages.keys(), *fixed_stages.keys()]))

    rows = []
    for stage in stages:
        variable_seconds = variable_stages.get(stage)
        fixed_seconds = fixed_stages.get(stage)
        ratio = (
            fixed_seconds / variable_seconds
            if variable_seconds not in {None, 0} and fixed_seconds is not None
            else None
        )
        rows.append({
            "stage": stage,
            "variable_seconds": variable_seconds,
            "fixed_seconds": fixed_seconds,
            "fixed_vs_variable_ratio": ratio,
            "difference_seconds": (
                fixed_seconds - variable_seconds
                if fixed_seconds is not None and variable_seconds is not None
                else None
            ),
        })

    report = {
        "variable": {
            "config": variable.get("config"),
            "length_mode": variable.get("length_mode"),
            "total_seconds": variable.get("total_seconds"),
        },
        "fixed": {
            "config": fixed.get("config"),
            "length_mode": fixed.get("length_mode"),
            "total_seconds": fixed.get("total_seconds"),
        },
        "stages": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    writer = csv.writer(sys.stdout)
    writer.writerow(["stage", "variable_s", "fixed_s", "fixed/variable", "fixed-variable_s"])
    for row in rows:
        writer.writerow([
            row["stage"],
            "" if row["variable_seconds"] is None else f"{row['variable_seconds']:.3f}",
            "" if row["fixed_seconds"] is None else f"{row['fixed_seconds']:.3f}",
            "" if row["fixed_vs_variable_ratio"] is None else f"{row['fixed_vs_variable_ratio']:.4f}",
            "" if row["difference_seconds"] is None else f"{row['difference_seconds']:.3f}",
        ])
    writer.writerow([
        "TOTAL",
        f"{float(variable.get('total_seconds', 0.0)):.3f}",
        f"{float(fixed.get('total_seconds', 0.0)):.3f}",
        (
            f"{float(fixed.get('total_seconds', 0.0)) / max(float(variable.get('total_seconds', 0.0)), 1e-12):.4f}"
            if variable.get("total_seconds")
            else ""
        ),
        f"{float(fixed.get('total_seconds', 0.0)) - float(variable.get('total_seconds', 0.0)):.3f}",
    ])


if __name__ == "__main__":
    main()
