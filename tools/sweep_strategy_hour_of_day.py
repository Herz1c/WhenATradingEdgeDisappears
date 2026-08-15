"""Hour-of-day strategy analysis.

Hypothesis: retail-heavy hours (US evenings, weekends) have more edge than
algo-heavy hours (US daytime). Test by:
  1. Apply the best previous policy (3z-tiered-C: early 0.10 / mid 0.05 /
     FAK 0.08) untouched.
  2. Compute UTC hour from each entry's ts_ns.
  3. Report per-hour PnL, win rate, trade count.
  4. Sweep "hours allowed" subsets to find the best time-window filter.

The signals dataset already encodes ts_ns; we derive hour-of-day from it.

Sample size warning: 7 days * 24 hours = 168 hour-slots total but distributed
unevenly across days. Most informative signal will come from broad groupings
(sessions) rather than single hours.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
SIG_PATH = REPO_ROOT / "data" / "datasets" / "poly_l2_only_v2_signals.parquet"

NOTIONAL = 5.0
COOLDOWN_S = 10.0
MAX_ENTRIES = 2

UP, DOWN, BOTH = frozenset({"UP"}), frozenset({"DOWN"}), frozenset({"UP", "DOWN"})

# Best policy from previous sweep (3z-tiered-C).
BEST_POLICY = [
    (45, 60, BOTH, 0.10),   # early — high edge, both sides
    (30, 45, DOWN, 0.05),   # mid — DOWN only, low edge
    (25, 30, DOWN, 0.08),   # FAK zone — DOWN only, high edge
]


def signal_qualifies(row: dict, policy) -> bool:
    t = float(row["ttc_s"])
    side = row["side"]
    ev = float(row["ev"])
    for lo, hi, sides, min_edge in policy:
        if lo <= t < hi and side in sides and ev >= min_edge:
            return True
    return False


def simulate(market_signals: pl.DataFrame, policy, allowed_hours: Set[int] | None) -> List[dict]:
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
        if not signal_qualifies(row, policy):
            continue
        if int(row["ts_ns"]) - last_ts < cooldown_ns:
            continue
        # Hour-of-day filter (UTC).
        ts_s = int(row["ts_ns"]) // 1_000_000_000
        hour = (ts_s // 3600) % 24
        if allowed_hours is not None and hour not in allowed_hours:
            continue
        side = row["side"]
        fill = float(row["fill"])
        shares = NOTIONAL / fill
        won = (side == "UP" and outcome == "Up") or (side == "DOWN" and outcome == "Down")
        pnl = shares - NOTIONAL if won else -NOTIONAL
        entries.append({
            "date": date, "market_slug": slug, "outcome": outcome,
            "ts_ns": int(row["ts_ns"]), "hour_utc": hour, "ttc_s": float(row["ttc_s"]),
            "side": side, "fill": fill, "ev": float(row["ev"]),
            "pnl": pnl, "won": won,
        })
        last_ts = int(row["ts_ns"])
        n += 1
    return entries


def summary(entries: List[dict], label: str) -> dict:
    if not entries:
        return {"label": label, "n": 0, "pnl": 0.0, "roi": 0.0,
                "worst_day": 0.0, "losing_days": 0, "win_rate": 0.0,
                "avg_per_trade": 0.0, "per_day": {}}
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
        return f"  {s['label']:45s} | no trades"
    return (f"  {s['label']:45s} | n={s['n']:>4} | wr={s['win_rate']*100:>5.1f}% | "
            f"pnl=${s['pnl']:>+8.2f} | roi={s['roi']*100:>+6.2f}% | "
            f"worst=${s['worst_day']:>+7.2f} | losing={s['losing_days']}/{s['total_days']} | "
            f"avg=${s['avg_per_trade']:>+.3f}")


def main():
    print(f"loading {SIG_PATH}")
    df = pl.read_parquet(SIG_PATH).sort(["market_slug", "ts_ns"])
    print(f"  rows={df.height:,}, markets={df['market_slug'].n_unique():,}")
    markets = list(df.partition_by("market_slug", as_dict=True).items())

    # --- step 1: collect ALL entries (no hour filter) using BEST_POLICY ---
    all_entries: List[dict] = []
    for _slug, mdf in markets:
        all_entries.extend(simulate(mdf, BEST_POLICY, allowed_hours=None))
    s_all = summary(all_entries, "BASELINE (3z-tiered-C, all hours)")
    print(f"\n=== BASELINE ===")
    print(fmt(s_all))

    # --- step 2: per-hour stats under BEST_POLICY ---
    by_hour: Dict[int, List[dict]] = defaultdict(list)
    for e in all_entries:
        by_hour[e["hour_utc"]].append(e)

    print(f"\n=== PER-UTC-HOUR BREAKDOWN (BEST_POLICY, all 7 days) ===")
    print(f"  {'hour':>4} {'n':>5} {'wr':>6} {'pnl':>9} {'roi':>7} {'avg/trade':>10} {'us_session':>11}")
    hour_stats: List[Tuple[int, dict]] = []
    for h in range(24):
        es = by_hour.get(h, [])
        if not es:
            print(f"  {h:>4} {'-':>5} {'-':>6} {'-':>9} {'-':>7} {'-':>10}  (no trades)")
            continue
        n = len(es)
        w = sum(1 for e in es if e["won"])
        pnl = sum(e["pnl"] for e in es)
        roi = pnl / (NOTIONAL * n) * 100
        avg = pnl / n
        # Tag US trading session (roughly 13:00-22:00 UTC = 9am-6pm Eastern)
        us = "US-day" if 13 <= h <= 21 else ("US-eve" if 22 <= h or h <= 3 else "non-US")
        print(f"  {h:>4} {n:>5} {w/n*100:>5.1f}% ${pnl:>+7.2f} {roi:>+6.2f}% ${avg:>+9.3f}  {us:>11}")
        hour_stats.append((h, {"n": n, "wr": w/n, "pnl": pnl, "roi": roi/100,
                                "avg": avg, "wins": w}))

    # --- step 3: try session-based filters ---
    print(f"\n=== SESSION-BASED FILTERS ===")
    sessions: List[Tuple[str, Set[int]]] = [
        ("ALL (24h)", set(range(24))),
        ("US daytime (13-21 UTC)", set(range(13, 22))),
        ("US evening (22-03 UTC)", set([22, 23, 0, 1, 2, 3])),
        ("US full (13-03 UTC)", set(range(13, 24)) | set(range(0, 4))),
        ("Europe (06-12 UTC)", set(range(6, 13))),
        ("Asia (00-08 UTC)", set(range(0, 9))),
        ("Retail evening (20-04 UTC)", set([20, 21, 22, 23, 0, 1, 2, 3, 4])),
        ("Wide retail (18-06 UTC)", set([18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])),
        ("Skip US daytime (22-12 UTC)", set([22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
        ("Skip Asia (08-23 UTC)", set(range(8, 24))),
    ]
    session_results: List[dict] = []
    for label, hours in sessions:
        ents: List[dict] = []
        for _slug, mdf in markets:
            ents.extend(simulate(mdf, BEST_POLICY, allowed_hours=hours))
        s = summary(ents, label)
        session_results.append(s)
        print(fmt(s))

    # --- step 4: greedy add — start from best hour, add hours by per-trade EV ---
    print(f"\n=== GREEDY HOUR PICKING (add hours by avg PnL/trade, best first) ===")
    ranked_hours = sorted(hour_stats, key=lambda x: -x[1]["avg"])
    cumulative: Set[int] = set()
    greedy_results: List[dict] = []
    for h, info in ranked_hours:
        cumulative.add(h)
        ents = []
        for _slug, mdf in markets:
            ents.extend(simulate(mdf, BEST_POLICY, allowed_hours=cumulative))
        s = summary(ents, f"hours {sorted(cumulative)} ({len(cumulative)})")
        greedy_results.append(s)
        print(f"  {h:>4}h added (its avg=${info['avg']:+.3f})  -->  "
              f"n={s['n']:>4} pnl=${s['pnl']:>+8.2f} roi={s['roi']*100:>+6.2f}% "
              f"worst=${s['worst_day']:>+7.2f} losing={s['losing_days']}/{s['total_days']}")

    # Identify the cumulative set that gave best PnL.
    best_greedy = max(greedy_results, key=lambda x: x["pnl"])
    print(f"\nbest greedy stop: {best_greedy['label']}  ->  "
          f"PnL=${best_greedy['pnl']:+.2f} roi={best_greedy['roi']*100:+.2f}% "
          f"worst=${best_greedy['worst_day']:+.2f} losing={best_greedy['losing_days']}/{best_greedy['total_days']}")
    best_greedy_drawdown = max(greedy_results, key=lambda x: x["worst_day"])
    print(f"best greedy drawdown: {best_greedy_drawdown['label']}  ->  "
          f"PnL=${best_greedy_drawdown['pnl']:+.2f} roi={best_greedy_drawdown['roi']*100:+.2f}% "
          f"worst=${best_greedy_drawdown['worst_day']:+.2f} losing={best_greedy_drawdown['losing_days']}/{best_greedy_drawdown['total_days']}")


if __name__ == "__main__":
    main()
