"""Hold vs stop-loss exits, risk-normalized (Phase 5.3).

Runs the locked strategies through the shared engine with hold / stop_0.10 /
stop_0.15 exits on the frozen test split, then compares PnL at equal drawdown:
if a stop variant has a shallower maxDD, it supports proportionally larger
size at the same risk budget -> report equal-DD-scaled PnL alongside raw PnL.

Uses the cached deltas from the snooping audit (no model forward needed).

Output: artifacts/audit_v1/exit_policy_comparison.json (+ console table)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from backtest.episode_strategy_backtester import (  # noqa: E402
    ExitSpec, HOLD, find_entries, load_episode_split, load_lock, simulate_trade,
    summarize,
)

DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
LOCK = ROOT / "artifacts" / "strategy_locks" / "tcn_double_strategy_v1_lock.json"
DELTA_CACHE = ROOT / "artifacts" / "audit_v1" / "delta_test_cache.npy"
OUT = ROOT / "artifacts" / "audit_v1" / "exit_policy_comparison.json"

EXITS = [
    HOLD,
    ExitSpec("stop_0.10", stop_loss=0.10, slippage=0.03),
    ExitSpec("stop_0.15", stop_loss=0.15, slippage=0.03),
    ExitSpec("stop_0.10_take_0.30", stop_loss=0.10, take_profit=0.30, slippage=0.03),
]


def main() -> int:
    specs, _lock = load_lock(LOCK)
    delta = np.load(DELTA_CACHE)
    data = load_episode_split(DATASET, "test", delta=delta)

    results = {}
    entries_by_spec = {s.strategy_id: find_entries(data, s) for s in specs}
    for exit_spec in EXITS:
        all_trades = []
        per_slot = {}
        for s in specs:
            trades = [simulate_trade(data, e, s, exit_spec)
                      for e in entries_by_spec[s.strategy_id]]
            per_slot[s.strategy_id] = summarize(trades, data.n_ep)
            all_trades.extend(trades)
        comb = summarize(all_trades, data.n_ep)
        results[exit_spec.label] = {"combined": comb, "per_slot": per_slot}

    hold_dd = abs(results["hold"]["combined"]["max_drawdown"]) or 1.0
    table = []
    for label, r in results.items():
        c = r["combined"]
        dd = abs(c["max_drawdown"]) or 1e-9
        scale = hold_dd / dd
        table.append({
            "exit": label,
            "trades": c["trades"],
            "pnl": round(c["total_pnl"], 2),
            "max_dd": round(c["max_drawdown"], 2),
            "worst_day": round(c["worst_day"], 2),
            "positive_days": f"{c['positive_days']}/{c['active_days']}",
            "equal_dd_scale": round(scale, 2),
            "equal_dd_pnl": round(c["total_pnl"] * scale, 2),
            "exit_counts": c["exit_counts"],
        })
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "equal_dd_pnl = pnl scaled so each variant matches hold's maxDD "
                   "(linear sizing assumption); selection-era data (D2 caveats apply) — "
                   "use for RELATIVE exit comparison only, not as evidence of edge.",
           "table": table, "full": results}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    hdr = f"{'exit':22s} {'trades':>6s} {'pnl':>8s} {'maxDD':>8s} {'worst':>8s} {'posD':>6s} {'eqDD pnl':>9s}"
    print(hdr)
    for t in table:
        print(f"{t['exit']:22s} {t['trades']:6d} {t['pnl']:8.2f} {t['max_dd']:8.2f} "
              f"{t['worst_day']:8.2f} {t['positive_days']:>6s} {t['equal_dd_pnl']:9.2f}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
