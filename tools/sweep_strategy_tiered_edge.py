"""Tiered-edge strategy sweep.

Hypothesis (user): the further out (higher TTC) we are, the noisier the
signal, so we should require a HIGHER min_edge there. Closer to close, the
model is more reliable, so a LOWER min_edge is fine.

A policy is a list of (ttc_lo, ttc_hi, sides_allowed, min_edge_override) zones.
Signals that fall inside a zone AND meet the per-zone min_edge are candidates.
The standard 2-entry / 10s-cooldown rule applies on top.

Execution-style annotation (does NOT affect PnL in this backtest, but
flagged in the policy label so the live bot knows what to do):
  - ttc < 30s zones: FAK (immediate-or-cancel at quoted best). Fills may
    not happen if size isn't there. We compensate by REQUIRING a higher
    edge in these zones so the trades that do go through are the strongest.
  - ttc >= 30s zones: aggressive taker (walks the book). Should fill.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
SIG_PATH = REPO_ROOT / "data" / "datasets" / "poly_l2_only_v2_signals.parquet"
NOTIONAL = 5.0
COOLDOWN_S = 10.0
MAX_ENTRIES = 2

UP = frozenset({"UP"})
DOWN = frozenset({"DOWN"})
BOTH = frozenset({"UP", "DOWN"})

# zone = (ttc_lo, ttc_hi, sides, min_edge_override)
Zone = Tuple[float, float, FrozenSet[str], float]
Policy = List[Zone]


def qualifies(row: dict, policy: Policy) -> bool:
    t = float(row["ttc_s"])
    side = row["side"]
    ev = float(row["ev"])
    for lo, hi, sides, min_edge in policy:
        if lo <= t < hi and side in sides and ev >= min_edge:
            return True
    return False


def simulate(market_signals: pl.DataFrame, policy: Policy) -> List[dict]:
    if market_signals.height == 0:
        return []
    cooldown_ns = int(COOLDOWN_S * 1_000_000_000)
    last_ts = -10**18
    n = 0
    entries: List[dict] = []
    outcome = market_signals["outcome"][0]
    slug = market_signals["market_slug"][0]
    date = market_signals["date"][0]
    for row in market_signals.iter_rows(named=True):
        if n >= MAX_ENTRIES:
            break
        if not qualifies(row, policy):
            continue
        if int(row["ts_ns"]) - last_ts < cooldown_ns:
            continue
        side = row["side"]
        fill = float(row["fill"])
        shares = NOTIONAL / fill
        won = (side == "UP" and outcome == "Up") or (side == "DOWN" and outcome == "Down")
        pnl = shares - NOTIONAL if won else -NOTIONAL
        entries.append({"date": date, "market_slug": slug, "outcome": outcome,
                        "ts_ns": int(row["ts_ns"]), "ttc_s": float(row["ttc_s"]),
                        "side": side, "fill": fill, "ev": float(row["ev"]),
                        "pnl": pnl, "won": won})
        last_ts = int(row["ts_ns"])
        n += 1
    return entries


def summary(entries: List[dict], label: str) -> dict:
    if not entries:
        return {"label": label, "n": 0}
    n = len(entries)
    wins = sum(1 for e in entries if e["won"])
    pnl = sum(e["pnl"] for e in entries)
    per_day = defaultdict(float)
    for e in entries:
        per_day[e["date"]] += e["pnl"]
    dp = list(per_day.values())
    return {
        "label": label, "n": n, "win_rate": wins/n, "pnl": pnl,
        "roi": pnl / (NOTIONAL * n),
        "worst_day": min(dp), "best_day": max(dp),
        "losing_days": sum(1 for v in dp if v < 0),
        "total_days": len(dp),
        "avg_per_trade": pnl/n,
        "per_day": dict(per_day),
    }


def fmt(s: dict) -> str:
    if s["n"] == 0:
        return f"  {s['label']:65s} | no trades"
    return (f"  {s['label']:65s} | n={s['n']:>4} | wr={s['win_rate']*100:>5.1f}% | "
            f"pnl=${s['pnl']:>+8.2f} | roi={s['roi']*100:>+6.2f}% | "
            f"worst=${s['worst_day']:>+7.2f} | losing={s['losing_days']}/{s['total_days']} | "
            f"avg=${s['avg_per_trade']:>+.3f}")


def main():
    print(f"loading {SIG_PATH}")
    df = pl.read_parquet(SIG_PATH).sort(["market_slug", "ts_ns"])
    print(f"  rows={df.height:,}, markets={df['market_slug'].n_unique():,}")
    print(f"  ev range: min={float(df['ev'].min()):.3f}, max={float(df['ev'].max()):.3f}, "
          f"mean={float(df['ev'].mean()):.3f}")
    print(f"  ev quantiles: p50={float(df['ev'].quantile(0.50)):.3f}, "
          f"p75={float(df['ev'].quantile(0.75)):.3f}, "
          f"p90={float(df['ev'].quantile(0.90)):.3f}, "
          f"p95={float(df['ev'].quantile(0.95)):.3f}")
    print()
    markets = list(df.partition_by("market_slug", as_dict=True).items())

    # Test policies — vary min_edge per zone.
    # All policies use the H6 structure: [45,60) BOTH + [25,45) DOWN.
    # Baseline is uniform min_edge=0.04 (Phase 1's threshold).
    policies: List[Tuple[str, Policy]] = [
        # === Baseline (uniform 0.04) ===
        ("H6 baseline                    early=0.04 / mid=0.04",
            [(45, 60, BOTH, 0.04), (25, 45, DOWN, 0.04)]),

        # === Tier the EARLY (45-60s) zone progressively tighter ===
        ("H6 early-tighten-1             early=0.05 / mid=0.04",
            [(45, 60, BOTH, 0.05), (25, 45, DOWN, 0.04)]),
        ("H6 early-tighten-2             early=0.06 / mid=0.04",
            [(45, 60, BOTH, 0.06), (25, 45, DOWN, 0.04)]),
        ("H6 early-tighten-3             early=0.08 / mid=0.04",
            [(45, 60, BOTH, 0.08), (25, 45, DOWN, 0.04)]),
        ("H6 early-tighten-4             early=0.10 / mid=0.04",
            [(45, 60, BOTH, 0.10), (25, 45, DOWN, 0.04)]),
        ("H6 early-tighten-5             early=0.12 / mid=0.04",
            [(45, 60, BOTH, 0.12), (25, 45, DOWN, 0.04)]),

        # === Tier the MID (25-45s) zone progressively tighter ===
        ("H6 mid-tighten-1               early=0.04 / mid=0.05",
            [(45, 60, BOTH, 0.04), (25, 45, DOWN, 0.05)]),
        ("H6 mid-tighten-2               early=0.04 / mid=0.06",
            [(45, 60, BOTH, 0.04), (25, 45, DOWN, 0.06)]),
        ("H6 mid-tighten-3               early=0.04 / mid=0.08",
            [(45, 60, BOTH, 0.04), (25, 45, DOWN, 0.08)]),

        # === Both zones tightened uniformly ===
        ("H6 both-tighten-1              early=0.05 / mid=0.05",
            [(45, 60, BOTH, 0.05), (25, 45, DOWN, 0.05)]),
        ("H6 both-tighten-2              early=0.06 / mid=0.06",
            [(45, 60, BOTH, 0.06), (25, 45, DOWN, 0.06)]),
        ("H6 both-tighten-3              early=0.08 / mid=0.08",
            [(45, 60, BOTH, 0.08), (25, 45, DOWN, 0.08)]),

        # === Tiered: HIGH edge requirement early, LOWER later (my hypothesis) ===
        ("H6 tiered-A                    early=0.06 / mid=0.04",
            [(45, 60, BOTH, 0.06), (25, 45, DOWN, 0.04)]),
        ("H6 tiered-B                    early=0.08 / mid=0.05",
            [(45, 60, BOTH, 0.08), (25, 45, DOWN, 0.05)]),
        ("H6 tiered-C                    early=0.10 / mid=0.05",
            [(45, 60, BOTH, 0.10), (25, 45, DOWN, 0.05)]),
        ("H6 tiered-D                    early=0.10 / mid=0.06",
            [(45, 60, BOTH, 0.10), (25, 45, DOWN, 0.06)]),
        ("H6 tiered-E                    early=0.12 / mid=0.06",
            [(45, 60, BOTH, 0.12), (25, 45, DOWN, 0.06)]),

        # === 3-zone splits: early (taker) / mid-late (taker) / FAK zone (<30s) ===
        # FAK zone at [25,30)s gets the HIGHEST edge requirement because in live
        # only the strongest signals will actually fill.
        ("3z-baseline                    [45,60)BOTH 0.04 / [30,45)DOWN 0.04 / [25,30)DOWN 0.04 FAK",
            [(45, 60, BOTH, 0.04), (30, 45, DOWN, 0.04), (25, 30, DOWN, 0.04)]),
        ("3z-tiered-A                    early 0.06 / mid 0.04 / FAK 0.06",
            [(45, 60, BOTH, 0.06), (30, 45, DOWN, 0.04), (25, 30, DOWN, 0.06)]),
        ("3z-tiered-B                    early 0.08 / mid 0.05 / FAK 0.08",
            [(45, 60, BOTH, 0.08), (30, 45, DOWN, 0.05), (25, 30, DOWN, 0.08)]),
        ("3z-tiered-C                    early 0.10 / mid 0.05 / FAK 0.08",
            [(45, 60, BOTH, 0.10), (30, 45, DOWN, 0.05), (25, 30, DOWN, 0.08)]),
        ("3z-tiered-D                    early 0.10 / mid 0.06 / FAK 0.10",
            [(45, 60, BOTH, 0.10), (30, 45, DOWN, 0.06), (25, 30, DOWN, 0.10)]),
        ("3z-tiered-E                    early 0.12 / mid 0.06 / FAK 0.10",
            [(45, 60, BOTH, 0.12), (30, 45, DOWN, 0.06), (25, 30, DOWN, 0.10)]),

        # === Reverse — what if the LATE zone needs MORE edge instead? (sanity check) ===
        ("REVERSE: early 0.04 / mid 0.08",
            [(45, 60, BOTH, 0.04), (25, 45, DOWN, 0.08)]),
        ("REVERSE: early 0.04 / mid 0.10",
            [(45, 60, BOTH, 0.04), (25, 45, DOWN, 0.10)]),
    ]

    results: List[dict] = []
    for label, policy in policies:
        ents = []
        for _slug, mdf in markets:
            ents.extend(simulate(mdf, policy))
        results.append(summary(ents, label))

    print("=== TIERED EDGE SWEEP — ALL POLICIES, sorted by ROI ===")
    for s in sorted([r for r in results if r["n"] > 0], key=lambda x: -x["roi"]):
        print(fmt(s))

    print("\n=== ALL POLICIES, sorted by PnL ===")
    for s in sorted([r for r in results if r["n"] > 0], key=lambda x: -x["pnl"]):
        print(fmt(s))

    print("\n=== ALL POLICIES (>=100 trades), sorted by WORST-DAY ===")
    for s in sorted([r for r in results if r["n"] >= 100], key=lambda x: -x["worst_day"]):
        print(fmt(s))

    # Composite: high PnL, small drawdown, enough trades.
    def composite(s):
        # 1 unit of worst-day == 0.5 units of pnl (drawdown matters more).
        return s["worst_day"] + 0.5 * s["pnl"]
    print("\n=== ALL POLICIES (>=100 trades), sorted by COMPOSITE (worst_day + 0.5*pnl) ===")
    for s in sorted([r for r in results if r["n"] >= 100], key=composite, reverse=True):
        print(fmt(s))

    # Per-day for the top composite.
    top = sorted([r for r in results if r["n"] >= 100], key=composite, reverse=True)[0]
    print(f"\n=== PER-DAY: {top['label']} ===")
    for d, v in sorted(top["per_day"].items()):
        print(f"  {d}: ${v:+.2f}")


if __name__ == "__main__":
    main()
