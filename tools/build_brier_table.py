"""Compatibility entry point for the public Brier table.

The actual calculation now lives in ``reproduce_public_evidence.py`` because it
uses released row-level predictions and computes clustered uncertainty instead
of merely copying metrics from training reports.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from reproduce_public_evidence import (
    BRIER_OUT,
    _canonical_json,
    _check_json,
    _write_canonical,
    build_brier_summary,
)


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
    if args.check:
        # Delegated rather than reimplemented. This entry point previously kept
        # its own copy of the comparison, which silently kept the bit exact and
        # newline dependent behaviour after reproduce_public_evidence.py was
        # fixed. Sharing the implementation is what stops that recurring.
        _check_json(BRIER_OUT, fresh)
        print("brier_summary.json is current")
        return 0
    _write_canonical(BRIER_OUT, _canonical_json(fresh))
    print(BRIER_OUT.relative_to(Path(__file__).resolve().parents[1]))
    print(_render(fresh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
