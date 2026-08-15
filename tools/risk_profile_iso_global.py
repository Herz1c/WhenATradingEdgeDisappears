"""Risk profile of the winning config from calib2d_ev_backtest:
   calibrator = global isotonic (fit on clean-OOS calib days)
   EV threshold = 0.10, fill in [0.30, 0.90], ttc >= 10.

Reconstructs the actual trade list on the 4 clean-OOS test days, ordered
chronologically by recv_ts_ns, and reports:
  - equity curve + max drawdown ($, % of peak, in trade-stakes)
  - longest winning / losing streak + streak distributions
  - PnL concentration: profit factor, avg win/loss, top-k winner share
    (is the curve consistent or carried by a few big wins?)
  - entry-price profile: avg/median fill, % bought below 0.50, histogram,
    split by side.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from calib2d_ev_backtest import (  # noqa: E402
    load_model, load_days, score_model_p, fit_iso, backtest,
)

REPO = TOOLS.parent
ART = REPO / "artifacts_cleaned" / "poly_l2_only_v2"

CALIB = ["2026-04-22", "2026-04-30", "2026-05-07"]
TEST = ["2026-05-28", "2026-06-01", "2026-06-10", "2026-06-11"]
THR = 0.10
NOTIONAL = 5.0
FILL_MIN, FILL_MAX = 0.30, 0.90


def streaks(won: np.ndarray):
    """Return (longest_win, longest_loss, list_of_runs[(is_win,len)])."""
    runs = []
    if len(won) == 0:
        return 0, 0, runs
    cur = won[0]; n = 1
    for w in won[1:]:
        if w == cur:
            n += 1
        else:
            runs.append((int(cur), n)); cur = w; n = 1
    runs.append((int(cur), n))
    lw = max((n for c, n in runs if c == 1), default=0)
    ll = max((n for c, n in runs if c == 0), default=0)
    return lw, ll, runs


def main():
    model, feats, _ = load_model(ART)
    df_cal = load_days(CALIB)
    p_cal = score_model_p(model, feats, df_cal)
    y_cal = df_cal["label_up"].to_numpy().astype(np.float64)
    iso = fit_iso(p_cal, y_cal)

    df_test = load_days(TEST)
    p_test = score_model_p(model, feats, df_test)
    cal_p = np.clip(iso.predict(p_test), 1e-6, 1 - 1e-6)

    bt_cols = ["market_slug", "recv_ts_ns", "ttc_s",
               "up_best_ask", "down_best_ask", "label_up", "date"]
    e = backtest(df_test.select(bt_cols), cal_p, THR, NOTIONAL,
                 FILL_MIN, FILL_MAX, 0.0)
    e = e.sort("recv_ts_ns")
    n = e.height
    pnl = e["pnl"].to_numpy()
    fill = e["fill"].to_numpy()
    won = e["won"].to_numpy().astype(int)
    side = e["side"].to_numpy()  # 1=UP, 0=DOWN
    dates = e["date"].to_list()

    total = float(pnl.sum())
    stake = NOTIONAL * n
    print(f"=== TRADES: {n}  |  net PnL ${total:+.2f}  |  ROI {100*total/stake:+.2f}%  "
          f"|  win {100*won.mean():.1f}% ===\n")

    # --- equity curve + drawdown ---
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    i_tr = int(np.argmax(dd))
    max_dd = float(dd[i_tr])
    peak_at = float(peak[i_tr])
    # peak index that set that running max
    i_pk = int(np.argmax(eq[:i_tr + 1])) if i_tr > 0 else 0
    final = float(eq[-1])
    print("--- DRAWDOWN ---")
    print(f"  final equity        : ${final:+.2f}")
    print(f"  peak equity         : ${float(peak.max()):+.2f}")
    print(f"  max drawdown        : ${max_dd:.2f}  "
          f"({100*max_dd/peak_at if peak_at > 0 else float('nan'):.1f}% of peak-at-trough, "
          f"{max_dd/NOTIONAL:.1f} stakes)")
    print(f"  drawdown span       : trade #{i_pk} (eq ${eq[i_pk]:+.2f}) -> "
          f"#{i_tr} (eq ${eq[i_tr]:+.2f})")
    # time underwater: longest run of dd>0
    uw = dd > 1e-9
    luw = cur = 0
    for u in uw:
        cur = cur + 1 if u else 0
        luw = max(luw, cur)
    print(f"  longest underwater  : {luw} trades")

    # --- streaks ---
    lw, ll, runs = streaks(won)
    win_runs = [r for c, r in runs if c == 1]
    loss_runs = [r for c, r in runs if c == 0]
    print("\n--- STREAKS ---")
    print(f"  longest winning streak : {lw}")
    print(f"  longest losing streak  : {ll}")
    print(f"  avg win run / loss run : {np.mean(win_runs):.2f} / {np.mean(loss_runs):.2f}")
    print(f"  # win runs / loss runs : {len(win_runs)} / {len(loss_runs)}")

    # --- concentration ---
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gw = float(wins.sum()); gl = float(-losses.sum())
    pf = gw / gl if gl > 0 else float("inf")
    order = np.argsort(pnl)[::-1]
    pnl_desc = pnl[order]
    def topk_share(k):
        k = min(k, n)
        return float(pnl_desc[:k].sum())
    print("\n--- PnL CONCENTRATION ---")
    print(f"  gross win ${gw:.2f} / gross loss ${gl:.2f}  | profit factor {pf:.2f}")
    print(f"  avg win ${wins.mean() if len(wins) else 0:+.3f} ({len(wins)})  |  "
          f"avg loss ${losses.mean() if len(losses) else 0:+.3f} ({len(losses)})  |  "
          f"payoff {abs(wins.mean()/losses.mean()) if len(losses) and len(wins) else float('nan'):.2f}")
    for k in (1, 3, 5, 10):
        s = topk_share(k)
        print(f"  top {k:>2} trades sum ${s:+.2f}  = {100*s/total if total else float('nan'):5.1f}% of net  "
              f"= {100*s/gw if gw else float('nan'):5.1f}% of gross win")
    n5 = max(1, round(0.05 * n))
    s5 = topk_share(n5)
    print(f"  top 5% ({n5} trades) = {100*s5/gw if gw else float('nan'):.1f}% of gross win, "
          f"{100*s5/total if total else float('nan'):.1f}% of net")
    # net excluding best trade
    print(f"  net w/o best trade  : ${total - pnl_desc[0]:+.2f}")
    print(f"  net w/o top 5       : ${total - topk_share(5):+.2f}")

    # --- entry price profile ---
    print("\n--- ENTRY PRICE (fill paid) ---")
    print(f"  avg fill   : {fill.mean():.3f}")
    print(f"  median fill: {np.median(fill):.3f}")
    print(f"  % below 0.50: {100*np.mean(fill < 0.5):.1f}%   "
          f"% at/above 0.50: {100*np.mean(fill >= 0.5):.1f}%")
    bins = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(f"  histogram (fill bucket -> n, win%, avg pnl):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (fill >= lo) & (fill < hi)
        if not m.any():
            print(f"    [{lo:.2f},{hi:.2f}): n=  0")
            continue
        print(f"    [{lo:.2f},{hi:.2f}): n={int(m.sum()):>3}  "
              f"win={100*won[m].mean():5.1f}%  avgpnl=${pnl[m].mean():+.3f}  "
              f"sumpnl=${pnl[m].sum():+.2f}")
    # by side
    print("  by side:")
    for sv, name in [(1, "UP"), (0, "DOWN")]:
        m = side == sv
        if not m.any():
            continue
        print(f"    {name:<4}: n={int(m.sum()):>3}  avg fill={fill[m].mean():.3f}  "
              f"%<0.5={100*np.mean(fill[m] < 0.5):4.1f}%  win={100*won[m].mean():5.1f}%  "
              f"pnl=${pnl[m].sum():+.2f}")

    # per-day equity recap
    print("\n--- PER-DAY ---")
    for d in TEST:
        md = np.array([dt == d for dt in dates])
        if not md.any():
            continue
        print(f"  {d}: n={int(md.sum()):>3}  pnl=${pnl[md].sum():+8.2f}  "
              f"win={100*won[md].mean():5.1f}%  avg fill={fill[md].mean():.3f}")

    out = ART / "backtests" / "risk_profile_iso_global_0p10.json"
    out.write_text(json.dumps({
        "config": {"calibrator": "iso_global", "thr": THR, "notional": NOTIONAL,
                   "fill_range": [FILL_MIN, FILL_MAX], "ttc_min": 10,
                   "calib_days": CALIB, "test_days": TEST},
        "n_trades": n, "net_pnl": total, "roi": total / stake,
        "win_rate": float(won.mean()),
        "max_drawdown_usd": max_dd, "max_dd_pct_of_peak": max_dd / peak_at if peak_at > 0 else None,
        "longest_underwater_trades": luw,
        "longest_win_streak": lw, "longest_loss_streak": ll,
        "profit_factor": pf, "gross_win": gw, "gross_loss": gl,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "top1_share_net": pnl_desc[0] / total if total else None,
        "top5_share_net": topk_share(5) / total if total else None,
        "avg_fill": float(fill.mean()), "median_fill": float(np.median(fill)),
        "pct_fill_below_0.5": float(np.mean(fill < 0.5)),
        "equity_curve": eq.round(3).tolist(),
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
