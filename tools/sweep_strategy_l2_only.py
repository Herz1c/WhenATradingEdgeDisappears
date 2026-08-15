"""Phase 2: load Phase 1 signals, sweep (TTC window × side) strategies under
the 2-entry / 10s-cooldown rule.

Per market, given a config (ttc_min, ttc_max, sides_allowed):
  1. Filter signals to allowed side and ttc in window.
  2. Walk in chronological order.
  3. Take the first signal as entry 1.
  4. Skip everything within 10 s of entry 1.
  5. Take the next qualifying signal as entry 2.
  6. Stop at 2 entries.

Aggregate per config: total trades, win rate, PnL, ROI, worst-day PnL,
losing days. Rank by PnL and by worst-day (drawdown proxy).

The strategy sweep is pure-python and runs in seconds — Phase 1 is the
expensive bit and only needs to be re-run when the model or filters change.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent

NOTIONAL = 5.0
COOLDOWN_S = 10.0
MAX_ENTRIES_PER_MARKET = 2

# Configs to evaluate — focused on TTC >= 20s where liquidity is realistic.
# (User domain knowledge: fills at expected price are rare in the [10,20) window.)
TTC_WINDOWS: List[Tuple[float, float]] = [
    # wide windows starting at 20s and later
    (20, 60), (25, 60), (30, 60), (35, 60), (40, 60), (45, 60), (50, 60),
    # narrower windows that EXCLUDE both [10,20) and the very early zone
    (20, 50), (25, 50), (30, 50), (35, 50), (40, 50), (45, 55),
    (20, 45), (25, 45), (30, 45), (35, 45),
    (20, 40), (25, 40), (30, 40),
    (20, 35), (25, 35),
    (20, 30),
    # legacy reference points
    (10, 60), (10, 30),
]
SIDES: List[Tuple[str, frozenset]] = [
    ("UP", frozenset({"UP"})),
    ("DOWN", frozenset({"DOWN"})),
    ("BOTH", frozenset({"UP", "DOWN"})),
]


@dataclass
class Entry:
    date: str
    market_slug: str
    outcome: str
    ts_ns: int
    side: str
    fill: float
    pnl: float
    won: bool


def simulate(market_signals: pl.DataFrame, ttc_lo: float, ttc_hi: float,
             sides_allowed: frozenset) -> List[Entry]:
    """Apply the 2-entry / 10s-cooldown rule to one market's signals."""
    # market_signals is already sorted by ts_ns at load time, but the filter
    # above can take rows out of order — re-sort defensively.
    df = (
        market_signals.filter(
            (pl.col("ttc_s") >= ttc_lo) &
            (pl.col("ttc_s") < ttc_hi) &
            (pl.col("side").is_in(list(sides_allowed)))
        )
        .sort("ts_ns")
    )
    if df.height == 0:
        return []
    cooldown_ns = int(COOLDOWN_S * 1_000_000_000)
    last_entry_ts = -10**18
    n_entries = 0
    entries: List[Entry] = []
    outcome = df["outcome"][0]
    market_slug = df["market_slug"][0]
    date = df["date"][0]
    for row in df.iter_rows(named=True):
        if n_entries >= MAX_ENTRIES_PER_MARKET:
            break
        if row["ts_ns"] - last_entry_ts < cooldown_ns:
            continue
        side = row["side"]
        fill = float(row["fill"])
        shares = NOTIONAL / fill
        won = (side == "UP" and outcome == "Up") or (side == "DOWN" and outcome == "Down")
        pnl = shares * 1.0 - NOTIONAL if won else -NOTIONAL
        entries.append(Entry(
            date=date, market_slug=market_slug, outcome=outcome,
            ts_ns=int(row["ts_ns"]), side=side, fill=fill,
            pnl=pnl, won=won,
        ))
        last_entry_ts = row["ts_ns"]
        n_entries += 1
    return entries


def summarize_config(entries: List[Entry], label: str) -> dict:
    if not entries:
        return {"label": label, "n_trades": 0, "pnl": 0.0,
                "win_rate": 0.0, "roi": 0.0, "worst_day_pnl": 0.0,
                "losing_days": 0, "total_days": 0, "avg_pnl_per_trade": 0.0,
                "best_day_pnl": 0.0}
    n = len(entries)
    n_wins = sum(1 for e in entries if e.won)
    total_pnl = sum(e.pnl for e in entries)
    notional = NOTIONAL * n
    # Per-day breakdown.
    per_day: Dict[str, float] = defaultdict(float)
    for e in entries:
        per_day[e.date] += e.pnl
    day_pnls = list(per_day.values())
    worst = min(day_pnls)
    best = max(day_pnls)
    losing_days = sum(1 for v in day_pnls if v < 0)
    return {
        "label": label,
        "n_trades": n,
        "pnl": total_pnl,
        "win_rate": n_wins / n,
        "roi": total_pnl / notional,
        "worst_day_pnl": worst,
        "best_day_pnl": best,
        "losing_days": losing_days,
        "total_days": len(day_pnls),
        "avg_pnl_per_trade": total_pnl / n,
        "per_day_pnl": per_day,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default="data/datasets/poly_l2_only_v2_signals.parquet")
    ap.add_argument("--top-k", type=int, default=10,
                    help="How many top configs to print per ranking.")
    args = ap.parse_args()

    sig_path = REPO_ROOT / args.signals
    print(f"loading signals from {sig_path}")
    df_all = pl.read_parquet(sig_path).sort(["market_slug", "ts_ns"])
    print(f"  rows={df_all.height:,}, markets={df_all['market_slug'].n_unique():,}, "
          f"dates={sorted(df_all['date'].unique().to_list())}")
    print(f"  by side: " + ", ".join(
        f"{s}={c}" for s, c in df_all.group_by("side").len().iter_rows()))
    print()

    market_groups = list(df_all.partition_by("market_slug", as_dict=True).items())
    print(f"grouped into {len(market_groups)} markets")
    print()

    # Evaluate each config.
    results: List[dict] = []
    for ttc_lo, ttc_hi in TTC_WINDOWS:
        for side_name, side_set in SIDES:
            all_entries: List[Entry] = []
            for _slug, df_mkt in market_groups:
                all_entries.extend(simulate(df_mkt, ttc_lo, ttc_hi, side_set))
            label = f"ttc=[{ttc_lo:>2.0f},{ttc_hi:>2.0f}) side={side_name:<4}"
            s = summarize_config(all_entries, label)
            s["ttc_lo"] = ttc_lo
            s["ttc_hi"] = ttc_hi
            s["side"] = side_name
            results.append(s)

    # Build display table.
    def fmt_row(r: dict) -> str:
        return (f"  {r['label']:35s} | "
                f"n={r['n_trades']:>4} | "
                f"wr={r['win_rate']*100:>5.1f}% | "
                f"pnl=${r['pnl']:>+9.2f} | "
                f"roi={r['roi']*100:>+6.2f}% | "
                f"worst_day=${r['worst_day_pnl']:>+8.2f} | "
                f"losing_days={r['losing_days']}/{r['total_days']} | "
                f"avg=${r['avg_pnl_per_trade']:>+.3f}")

    # Ranking 1: by total PnL (raw money).
    print(f"=== TOP {args.top_k} BY TOTAL PnL ===")
    for r in sorted(results, key=lambda x: -x["pnl"])[:args.top_k]:
        print(fmt_row(r))
    print()

    # Ranking 2: by worst-day PnL (drawdown proxy — higher = better).
    # Only consider configs with >= 50 trades so we don't reward tiny samples.
    eligible = [r for r in results if r["n_trades"] >= 50]
    print(f"=== TOP {args.top_k} BY WORST-DAY PnL (configs with >=50 trades) ===")
    for r in sorted(eligible, key=lambda x: -x["worst_day_pnl"])[:args.top_k]:
        print(fmt_row(r))
    print()

    # Ranking 3: by ROI (only configs with enough sample).
    eligible_roi = [r for r in results if r["n_trades"] >= 100]
    print(f"=== TOP {args.top_k} BY ROI (configs with >=100 trades) ===")
    for r in sorted(eligible_roi, key=lambda x: -x["roi"])[:args.top_k]:
        print(fmt_row(r))
    print()

    # Ranking 4: composite — high PnL with bounded drawdown.
    # Sort by (worst_day_pnl + 0.5 * pnl) — basically "good worst day but
    # decent total too". Tune the weight if you want.
    def score(r):
        return r["worst_day_pnl"] + 0.5 * r["pnl"]
    print(f"=== TOP {args.top_k} BY COMPOSITE (worst_day + 0.5*pnl) ===")
    for r in sorted(eligible_roi, key=score, reverse=True)[:args.top_k]:
        print(fmt_row(r))
    print()

    # Full table for reference.
    print("=== FULL TABLE (sorted by ROI) ===")
    for r in sorted(results, key=lambda x: -x["roi"]):
        print(fmt_row(r))
    print()

    # Per-day spread for the winning config.
    winner = sorted(eligible_roi, key=lambda x: -x["roi"])[0]
    print(f"=== WINNER PER-DAY PnL: {winner['label']} ===")
    for d, p in sorted(winner["per_day_pnl"].items()):
        print(f"  {d}: ${p:+.2f}")


if __name__ == "__main__":
    main()
