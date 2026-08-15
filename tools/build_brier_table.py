"""Compatibility entry point for the public Brier table.

The actual calculation now lives in ``reproduce_public_evidence.py`` because it
uses released row-level predictions and computes clustered uncertainty instead
of merely copying metrics from training reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduce_public_evidence import BRIER_OUT, build_brier_summary


def _render(summary: dict) -> str:
    lines = [
        "| Calibrator | Validation Brier | Test Brier | Test minus market | Market-cluster CI95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        val = row["val"]
        test = row["test"]
        lo, hi = test["market_cluster_bootstrap"]["ci95"]
        lines.append(
            f"| {row['label']} | {val['mean_brier_across_seeds']:.6f} | "
            f"{test['mean_brier_across_seeds']:.6f} | {test['delta_vs_market']:+.6f} | "
            f"[{lo:+.6f}, {hi:+.6f}] |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fresh = build_brier_summary()
    expected = json.dumps(fresh, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not BRIER_OUT.is_file() or BRIER_OUT.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"stale generated artifact: {BRIER_OUT}")
        print("brier_summary.json is current")
        return 0
    BRIER_OUT.write_text(expected, encoding="utf-8")
    print(BRIER_OUT.relative_to(Path(__file__).resolve().parents[1]))
    print(_render(fresh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
