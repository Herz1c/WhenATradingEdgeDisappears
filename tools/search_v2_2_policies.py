"""Search v2.2: constant-base sizing (5 shares, boost to 7.5 on big edge) with
a hard DD budget instead of the ratio objective (Plan v2.1, user revision).

The R1 ratio (pnl/|maxDD|) systematically prefers micro-risk configs (v2.1
came out with maxDD -3.68). I therefore changed the objective, and this
search instead:

  - sizing: shares = min(s_max, base + k * (EV - floor)), base = 5.0,
    s_max in {5.0 (pure constant), 7.5}, k in {0, 25, 50, 100}
  - selection rule R-DD15: among 2-fold WF survivors with selection-era
    maxDD >= -15 USD, rank by TOTAL PNL (not ratio)
  - portfolio: greedy add low-corr slots while combined maxDD stays >= -15

Everything else identical to search_v2_1_policies (calibrated ensemble probs,
quantile floors, selection era 06-23..07-11 = spent, forward shadow decides).

Output: artifacts/tcn_v2_2_policy_search/{grid.csv, report.json}
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from backtest.episode_strategy_backtester import (  # noqa: E402
    HOLD, StrategySpec, find_entries, simulate_trade,
)
from search_v2_1_policies import (  # noqa: E402
    DIAG, PRICES, TTCS, calibrated_effective_delta, load_shards, to_episode_data,
)

OUT_DIR = ROOT / "artifacts" / "tcn_v2_2_policy_search"
BETAS = [0.75, 1.0, 1.25]
BASE = 5.0
SIZE_MAX = [5.0, 7.5]
SIZE_K = [0.0, 25.0, 50.0, 100.0]
DD_BUDGET = -15.0
DAILY_CAP = None          # no daily stop in search; RiskGate handles live
MIN_TRADES = 15
WF_FOLDS = [(8, 11), (11, 15)]
PORTFOLIO_MAX_SLOTS = 3
PORTFOLIO_CORR_MAX = 0.5


def eval_sized(trades: list[dict], k: float, s_max: float, floor: float,
               days: list[str]) -> dict:
    day_pnl = {d: 0.0 for d in days}
    n_taken = 0
    wins = 0
    for t in trades:
        shares = min(s_max, BASE + k * max(0.0, t["entry_ev"] - floor))
        pnl = t["unit_pnl"] * shares
        day_pnl[t["date"]] = day_pnl.get(t["date"], 0.0) + pnl
        n_taken += 1
        wins += int(t["resolved_win"])
    vals = np.asarray([day_pnl[d] for d in sorted(day_pnl)], dtype=np.float64)
    equity = np.cumsum(vals)
    dd = equity - np.maximum.accumulate(equity)
    total = float(vals.sum())
    return {
        "trades": n_taken, "pnl": round(total, 2),
        "win_rate": round(wins / n_taken, 3) if n_taken else None,
        "worst_day": round(float(vals.min()), 2) if vals.size else 0.0,
        "max_dd": round(float(dd.min()), 2) if vals.size else 0.0,
        "positive_days": int((vals > 0).sum()),
        "daily": {d: round(day_pnl[d], 4) for d in sorted(day_pnl)},
    }


def main() -> int:
    import torch
    torch.set_num_threads(8)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    floors = json.loads(DIAG.read_text(encoding="utf-8"))["proposed_floor_grid_v2_platt"]
    arrs = load_shards()
    days = arrs["days"]
    print(f"selection era: {arrs['y'].shape[0]} mkts, {len(days)} days; floors {floors}")
    data = to_episode_data(arrs, calibrated_effective_delta(arrs))
    print(f"deltas ready ({time.time() - t0:.0f}s)", flush=True)

    entry_rows = []
    for beta in BETAS:
        for lo, hi in TTCS:
            for plo, phi in PRICES:
                for floor in floors:
                    spec = StrategySpec(
                        strategy_id=f"b{beta:g}_ttc{lo:g}-{hi:g}_ev{floor:g}_px{plo:g}-{phi:g}",
                        beta=beta, ttc_min=lo, ttc_max=hi, ev_min=floor,
                        price_lo=plo, price_hi=phi, shares=1.0)
                    entries = find_entries(data, spec)
                    trades = []
                    for e in entries:
                        tr = simulate_trade(data, e, spec, HOLD, shares=1.0)
                        trades.append({
                            "date": tr["date"], "unit_pnl": tr["pnl"],
                            "entry_ev": float(e["ev"]),
                            "resolved_win": tr["resolved_win"],
                            "decision_ns": int(data.now_ns[int(e["ep"]), int(e["step"])]),
                        })
                    trades.sort(key=lambda t: t["decision_ns"])
                    entry_rows.append({"spec": spec, "floor": floor, "trades": trades})
        print(f"entry stage beta={beta:g} done ({time.time() - t0:.0f}s)", flush=True)

    grid_rows = []
    for er in entry_rows:
        for k in SIZE_K:
            for s_max in SIZE_MAX:
                if k == 0.0 and s_max != BASE:
                    continue      # k=0 boost is meaningless; keep one constant variant
                full = eval_sized(er["trades"], k, s_max, er["floor"], days)
                if full["trades"] < MIN_TRADES:
                    continue
                wf_ok = True
                for sel_n, val_n in WF_FOLDS:
                    sel_days, val_days = days[:sel_n], days[sel_n:val_n]
                    sel = eval_sized([t for t in er["trades"] if t["date"] in set(sel_days)],
                                     k, s_max, er["floor"], sel_days)
                    val = eval_sized([t for t in er["trades"] if t["date"] in set(val_days)],
                                     k, s_max, er["floor"], val_days)
                    if sel["pnl"] <= 0 or val["pnl"] <= 0:
                        wf_ok = False
                        break
                daily = full.pop("daily")
                grid_rows.append({
                    "config": f"{er['spec'].strategy_id}_k{k:g}_smax{s_max:g}",
                    "beta": er["spec"].beta, "ttc_min": er["spec"].ttc_min,
                    "ttc_max": er["spec"].ttc_max, "floor": er["floor"],
                    "price_lo": er["spec"].price_lo, "size_k": k, "size_max": s_max,
                    **full, "wf_survives": wf_ok, "_daily": daily,
                })
    print(f"grid: {len(grid_rows)} sized configs ({time.time() - t0:.0f}s)")

    eligible = [r for r in grid_rows
                if r["wf_survives"] and r["max_dd"] >= DD_BUDGET]
    eligible.sort(key=lambda r: -r["pnl"])
    print(f"WF survivors within DD budget {DD_BUDGET}: {len(eligible)}")

    portfolio, chosen_daily = [], []
    for r in eligible:
        d = np.asarray([r["_daily"][x] for x in sorted(r["_daily"])])
        if any(np.corrcoef(d, c)[0, 1] > PORTFOLIO_CORR_MAX
               for c in chosen_daily if d.std() > 0 and c.std() > 0):
            continue
        trial = (np.sum(chosen_daily, axis=0) + d) if chosen_daily else d
        eq = np.cumsum(trial)
        trial_dd = float((eq - np.maximum.accumulate(eq)).min())
        if trial_dd < DD_BUDGET:
            continue
        portfolio.append(r)
        chosen_daily.append(d)
        if len(portfolio) == PORTFOLIO_MAX_SLOTS:
            break
    combo = np.sum(chosen_daily, axis=0) if chosen_daily else np.zeros(len(days))
    eq = np.cumsum(combo)
    dd = eq - np.maximum.accumulate(eq)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "user revision: base 5 shares (boost <= 7.5), hard DD budget -15, "
                    "rank by PnL among WF survivors; selection era spent, forward decides",
        "days": days, "floors": floors, "dd_budget": DD_BUDGET,
        "grid_rows": len(grid_rows), "eligible": len(eligible),
        "top20_pnl": [{k2: v for k2, v in r.items() if k2 != "_daily"} for r in eligible[:20]],
        "portfolio": {
            "slots": [{k2: v for k2, v in r.items() if k2 != "_daily"} for r in portfolio],
            "combined_pnl": round(float(combo.sum()), 2),
            "combined_max_dd": round(float(dd.min()), 2) if combo.size else 0.0,
            "combined_worst_day": round(float(combo.min()), 2) if combo.size else 0.0,
            "combined_daily": {d: round(float(v), 2) for d, v in zip(sorted(days), combo)},
        },
    }
    with (OUT_DIR / "grid.csv").open("w", newline="", encoding="utf-8") as f:
        rows = [{k2: v for k2, v in r.items() if k2 != "_daily"} for r in grid_rows]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("== TOP 5 by PnL (WF + DD<=15):")
    for r in eligible[:5]:
        print(f"  {r['config']:52s} pnl {r['pnl']:+8.2f} DD {r['max_dd']:7.2f} "
              f"worst {r['worst_day']:7.2f} trades {r['trades']:4d} WR {r['win_rate']}")
    print("== PORTFOLIO:", [r["config"] for r in portfolio])
    print(f"   combined pnl {report['portfolio']['combined_pnl']:+.2f} "
          f"DD {report['portfolio']['combined_max_dd']:.2f} "
          f"worst {report['portfolio']['combined_worst_day']:.2f}")
    print(f"-> {OUT_DIR} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
