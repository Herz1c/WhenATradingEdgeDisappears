"""Search for a CONSISTENT strategy (smooth up-curve), guarding against
overfitting the 4 test days.

Protocol:
  calibrator : isotonic fit on 04-22, 04-30 only.
  SELECT days: 05-07, 05-15, 05-21  -> pick config here.
  TEST days  : 05-28, 06-01, 06-10, 06-11 -> untouched OOS report.

Selection filter targets consistency, not a single peak:
  - config must be POSITIVE on EVERY select day (hard filter), and
  - n_select >= MIN_TRADES, profit_factor_select >= MIN_PF.
  Rank eligible configs by SELECT Calmar (PnL/maxDD); tie-break worst-day PnL.
  If nothing passes the all-days-positive filter, relax to ">=1 down day".

Reports the winner's per-day PnL across all 7 OOS days and a full 7-day
equity curve (select+test) so consistency is judged over a longer window.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from calib2d_ev_backtest import load_model, load_days, score_model_p, fit_iso  # noqa: E402
from search_robust_strategy import base_frame, entries_for, risk, NOTIONAL  # noqa: E402

REPO = TOOLS.parent
ART = REPO / "artifacts_cleaned" / "poly_l2_only_v2"

CALIB = ["2026-04-22", "2026-04-30"]
SELECT = ["2026-05-07", "2026-05-15", "2026-05-21"]
TEST = ["2026-05-28", "2026-06-01", "2026-06-10", "2026-06-11"]
EVAL = SELECT + TEST

MIN_TRADES = 45
MIN_PF = 1.30

THRS = [0.12, 0.15, 0.18, 0.20, 0.25]
FILL_LO = [0.25, 0.30, 0.35, 0.40]
FILL_HI = [0.50, 0.55, 0.60]
EVF = 0.0


def subset(e: pl.DataFrame, days) -> pl.DataFrame:
    if e.height == 0:
        return e
    return e.filter(pl.col("date").is_in(days)).sort("recv_ts_ns")


def per_day_pnl(e: pl.DataFrame, days):
    out = {}
    for d in days:
        ed = e.filter(pl.col("date") == d)
        out[d] = float(ed["pnl"].sum()) if ed.height else 0.0
    return out


def main():
    model, feats, _ = load_model(ART)
    df_cal = load_days(CALIB)
    iso = fit_iso(score_model_p(model, feats, df_cal),
                  df_cal["label_up"].to_numpy().astype(np.float64))
    print("building base frame over 7 OOS days ...", flush=True)
    base = base_frame(model, feats, EVAL, iso)
    print(f"rows={base.height:,}", flush=True)

    configs = [(t, lo, hi) for t, lo, hi in itertools.product(THRS, FILL_LO, FILL_HI) if lo < hi]
    print(f"searching {len(configs)} configs (select on {SELECT}) ...", flush=True)

    results = []
    for thr, lo, hi in configs:
        e = entries_for(base, thr, lo, hi, EVF)
        e_sel = subset(e, SELECT)
        rs = risk(e_sel)
        pdp = per_day_pnl(e_sel, SELECT)
        results.append({"cfg": (thr, lo, hi), "sel": rs, "sel_days": pdp,
                        "min_day": min(pdp.values()) if pdp else 0.0,
                        "all_pos": all(v > 0 for v in pdp.values())})

    def eligible(r, require_all_pos):
        ok = r["sel"]["n"] >= MIN_TRADES and r["sel"]["pf"] >= MIN_PF and r["sel"]["pnl"] > 0
        if require_all_pos:
            ok = ok and r["all_pos"]
        return ok

    elig = [r for r in results if eligible(r, True)]
    relaxed = False
    if not elig:
        relaxed = True
        elig = [r for r in results if eligible(r, False)]
    elig.sort(key=lambda r: (r["sel"]["calmar"], r["min_day"]), reverse=True)
    print(f"\n{'(relaxed: all-pos filter dropped)' if relaxed else '(all-days-positive filter active)'}")

    print("\n=== TOP configs by SELECT Calmar (consistency-filtered) — with TEST transfer ===")
    print(f"{'thr':>4} {'band':>11} | {'S_n':>4} {'S_pnl':>7} {'S_dd':>6} {'S_clmr':>6} {'S_pf':>5} {'S_min':>6} | "
          f"{'T_n':>4} {'T_pnl':>7} {'T_dd':>6} {'T_clmr':>6} {'T_pf':>5} {'T_win':>5}")
    shown = []
    for r in elig[:15]:
        thr, lo, hi = r["cfg"]
        e = entries_for(base, thr, lo, hi, EVF)
        rt = risk(subset(e, TEST))
        shown.append((r, rt))
        print(f"{thr:>4.2f} {f'{lo:.2f}-{hi:.2f}':>11} | "
              f"{r['sel']['n']:>4} {r['sel']['pnl']:>+7.1f} {r['sel']['maxdd']:>6.1f} "
              f"{r['sel']['calmar']:>6.2f} {r['sel']['pf']:>5.2f} {r['min_day']:>+6.1f} | "
              f"{rt['n']:>4} {rt['pnl']:>+7.1f} {rt['maxdd']:>6.1f} {rt['calmar']:>6.2f} {rt['pf']:>5.2f} "
              f"{100*rt['win'] if rt['n'] else 0:>4.1f}")

    if not shown:
        print("no eligible config even after relaxing")
        return

    r, rt = shown[0]
    thr, lo, hi = r["cfg"]
    e = entries_for(base, thr, lo, hi, EVF)
    e_all = subset(e, EVAL)
    pnl = e_all["pnl"].to_numpy()
    eq = np.cumsum(pnl)
    dd = np.maximum.accumulate(eq) - eq
    pdp_all = per_day_pnl(e_all, EVAL)
    # day-of where test starts (index)
    dates = e_all["date"].to_list()
    n_select = sum(1 for d in dates if d in SELECT)

    rfull = risk(e_all)
    daily_vals = np.array([pdp_all[d] for d in EVAL])
    daily_sharpe = float(daily_vals.mean() / daily_vals.std()) if daily_vals.std() > 0 else float("inf")
    pos_days = int((daily_vals > 0).sum())

    print(f"\n>>> SELECTED: thr={thr} band=[{lo},{hi}]")
    print(f"    SELECT: n={r['sel']['n']} pnl=${r['sel']['pnl']:+.1f} dd=${r['sel']['maxdd']:.1f} "
          f"calmar={r['sel']['calmar']:.2f} pf={r['sel']['pf']:.2f}  per-day={ {d: round(v,1) for d,v in r['sel_days'].items()} }")
    print(f"    TEST  : n={rt['n']} pnl=${rt['pnl']:+.1f} dd=${rt['maxdd']:.1f} "
          f"calmar={rt['calmar']:.2f} pf={rt['pf']:.2f} win={100*rt['win']:.1f}%  "
          f"DD/PnL={rt['maxdd']/rt['pnl'] if rt['pnl']>0 else float('inf'):.2f}")
    print(f"    FULL 7-day: n={rfull['n']} pnl=${rfull['pnl']:+.1f} maxDD=${rfull['maxdd']:.1f} "
          f"calmar={rfull['calmar']:.2f} pf={rfull['pf']:.2f}")
    print(f"    consistency: positive days {pos_days}/7  daily Sharpe {daily_sharpe:.2f}")
    print(f"    per-day PnL across 7 OOS days:")
    for d in EVAL:
        tag = "SEL " if d in SELECT else "TEST"
        print(f"      [{tag}] {d}: ${pdp_all[d]:+8.2f}")

    # entry profile on TEST
    et = subset(e, TEST)
    fillt = et["fill"].to_numpy(); wont = et["won"].to_numpy().astype(int); pnlt = et["pnl"].to_numpy()
    print(f"\n    TEST entry: avg fill {fillt.mean():.3f} median {np.median(fillt):.3f} "
          f"%<0.5 {100*np.mean(fillt<0.5):.1f}%  avg win ${pnlt[pnlt>0].mean():+.2f} "
          f"avg loss ${pnlt[pnlt<0].mean():+.2f}")

    out = ART / "backtests" / "consistent_strategy_selected.json"
    out.write_text(json.dumps({
        "selected": {"thr": thr, "fill_min": lo, "fill_max": hi, "ev_frac": EVF,
                     "calib": CALIB, "select": SELECT, "test": TEST, "notional": NOTIONAL},
        "select_metrics": r["sel"], "select_per_day": r["sel_days"],
        "test_metrics": rt, "full7_metrics": rfull,
        "per_day_pnl_7d": pdp_all, "positive_days": pos_days, "daily_sharpe": daily_sharpe,
        "n_select_trades_prefix": n_select,
        "equity_curve_7d": [round(float(v), 3) for v in eq],
        "test_start_index": n_select,
    }, indent=2, default=float))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
