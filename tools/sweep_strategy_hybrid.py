"""Hybrid strategy sweep — allows different side rules in different TTC zones.

Sometimes the best strategy isn't one (ttc_range, side) tuple but a combination,
e.g. "BOTH sides in early TTC, DOWN-only in mid TTC". This script tests
hybrid policies on the Phase 1 signals.

A policy is a list of zones: [(ttc_lo, ttc_hi, sides_allowed), ...]. Signals
inside any zone with the matching side are candidates. The 2-entry / 10s
cooldown rule then applies across the union.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
SIG_PATH = REPO_ROOT / "data" / "datasets" / "poly_l2_only_v2_signals.parquet"
NOTIONAL = 5.0
COOLDOWN_S = 10.0
MAX_ENTRIES = 2

Zone = Tuple[float, float, FrozenSet[str]]
Policy = List[Zone]


def signal_qualifies(ttc_s: float, side: str, policy: Policy) -> bool:
    for lo, hi, sides in policy:
        if lo <= ttc_s < hi and side in sides:
            return True
    return False


def simulate(market_signals: pl.DataFrame, policy: Policy) -> List[dict]:
    """Walk signals; allow entries that match any zone; respect cooldown +
    max-2-per-market."""
    if market_signals.height == 0:
        return []
    cooldown_ns = int(COOLDOWN_S * 1_000_000_000)
    last_ts = -10**18
    n_entries = 0
    entries: List[dict] = []
    outcome = market_signals["outcome"][0]
    market_slug = market_signals["market_slug"][0]
    date = market_signals["date"][0]
    for row in market_signals.iter_rows(named=True):
        if n_entries >= MAX_ENTRIES:
            break
        if not signal_qualifies(float(row["ttc_s"]), row["side"], policy):
            continue
        if int(row["ts_ns"]) - last_ts < cooldown_ns:
            continue
        side = row["side"]
        fill = float(row["fill"])
        shares = NOTIONAL / fill
        won = (side == "UP" and outcome == "Up") or (side == "DOWN" and outcome == "Down")
        pnl = shares - NOTIONAL if won else -NOTIONAL
        entries.append({
            "date": date, "market_slug": market_slug, "outcome": outcome,
            "ts_ns": int(row["ts_ns"]), "ttc_s": float(row["ttc_s"]),
            "side": side, "fill": fill, "pnl": pnl, "won": won,
        })
        last_ts = int(row["ts_ns"])
        n_entries += 1
    return entries


def summary(entries: List[dict], label: str) -> dict:
    if not entries:
        return {"label": label, "n": 0}
    n = len(entries)
    wins = sum(1 for e in entries if e["won"])
    pnl = sum(e["pnl"] for e in entries)
    notional = NOTIONAL * n
    per_day = defaultdict(float)
    for e in entries:
        per_day[e["date"]] += e["pnl"]
    days_pnl = list(per_day.values())
    return {
        "label": label, "n": n, "win_rate": wins/n, "pnl": pnl, "roi": pnl/notional,
        "worst_day": min(days_pnl), "best_day": max(days_pnl),
        "losing_days": sum(1 for v in days_pnl if v < 0),
        "total_days": len(days_pnl),
        "avg_per_trade": pnl/n,
        "per_day": dict(per_day),
    }


def fmt(s: dict) -> str:
    if s["n"] == 0:
        return f"  {s['label']:55s} | no trades"
    return (f"  {s['label']:55s} | n={s['n']:>4} | wr={s['win_rate']*100:>5.1f}% | "
            f"pnl=${s['pnl']:>+8.2f} | roi={s['roi']*100:>+6.2f}% | "
            f"worst=${s['worst_day']:>+8.2f} | losing={s['losing_days']}/{s['total_days']} | "
            f"avg=${s['avg_per_trade']:>+.3f}")


def main():
    print(f"loading {SIG_PATH}")
    df = pl.read_parquet(SIG_PATH).sort(["market_slug", "ts_ns"])
    print(f"  rows={df.height:,}, markets={df['market_slug'].n_unique():,}")
    markets = list(df.partition_by("market_slug", as_dict=True).items())

    UP, DOWN, BOTH = frozenset({"UP"}), frozenset({"DOWN"}), frozenset({"UP", "DOWN"})

    # Policies to test. Each is (label, [zones]).
    policies: List[Tuple[str, Policy]] = [
        # Single-zone references.
        ("ref: [50,60) BOTH",                [(50, 60, BOTH)]),
        ("ref: [25,60) DOWN",                [(25, 60, DOWN)]),
        ("ref: [20,40) DOWN",                [(20, 40, DOWN)]),
        ("ref: [20,30) DOWN",                [(20, 30, DOWN)]),
        ("ref: [30,40) DOWN",                [(30, 40, DOWN)]),
        ("ref: [45,60) BOTH",                [(45, 60, BOTH)]),
        # Hybrids — different side rule by TTC zone.
        ("H1: [50,60) BOTH + [20,40) DOWN",  [(50, 60, BOTH), (20, 40, DOWN)]),
        ("H2: [45,60) BOTH + [20,40) DOWN",  [(45, 60, BOTH), (20, 40, DOWN)]),
        ("H3: [50,60) BOTH + [25,45) DOWN",  [(50, 60, BOTH), (25, 45, DOWN)]),
        ("H4: [50,60) BOTH + [30,45) DOWN",  [(50, 60, BOTH), (30, 45, DOWN)]),
        ("H5: [45,60) BOTH + [20,40) DOWN",  [(45, 60, BOTH), (20, 40, DOWN)]),
        ("H6: [45,60) UP+DN + [25,45) DOWN", [(45, 60, BOTH), (25, 45, DOWN)]),
        ("H7: [50,60) UP-only + [25,45) DOWN", [(50, 60, UP), (25, 45, DOWN)]),
        ("H8: [45,60) UP-only + [20,40) DOWN", [(45, 60, UP), (20, 40, DOWN)]),
        # DOWN-everywhere variants.
        ("H9: [25,55) DOWN",                 [(25, 55, DOWN)]),
        ("H10: [30,55) DOWN",                [(30, 55, DOWN)]),
        # Wide BOTH only in the very-opening 5 seconds + DOWN later.
        ("H11: [55,60) BOTH + [20,45) DOWN", [(55, 60, BOTH), (20, 45, DOWN)]),
    ]

    results: List[dict] = []
    for label, policy in policies:
        all_entries: List[dict] = []
        for _slug, mdf in markets:
            all_entries.extend(simulate(mdf, policy))
        s = summary(all_entries, label)
        results.append(s)

    print("\n=== ALL POLICIES, sorted by ROI ===")
    for s in sorted([r for r in results if r["n"] > 0], key=lambda x: -x["roi"]):
        print(fmt(s))

    print("\n=== ALL POLICIES, sorted by PnL ===")
    for s in sorted([r for r in results if r["n"] > 0], key=lambda x: -x["pnl"]):
        print(fmt(s))

    print("\n=== ALL POLICIES, sorted by WORST-DAY (drawdown) ===")
    for s in sorted([r for r in results if r["n"] >= 100], key=lambda x: -x["worst_day"]):
        print(fmt(s))

    # Per-day breakdown for the top-PnL hybrid.
    hybrids_with_n = [r for r in results if r["n"] > 0 and r["label"].startswith("H")]
    if hybrids_with_n:
        top = sorted(hybrids_with_n, key=lambda x: -x["pnl"])[0]
        print(f"\n=== PER-DAY: {top['label']} (top hybrid by PnL) ===")
        for d, v in sorted(top["per_day"].items()):
            print(f"  {d}: ${v:+.2f}")


if __name__ == "__main__":
    main()
