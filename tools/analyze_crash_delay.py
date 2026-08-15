"""For the final strategy entries, look at the 2-second delayed fill price and
ask: how often does a cheap entry's price CRASH below the 0.25 floor within 2s
(market turning against the position), and do those crashed entries lose?

Grouped by ORIGINAL fill band (0.25-0.30, 0.30-0.40, ...).
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import polars as pl

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from calib2d_ev_backtest import load_model, load_days, score_model_p, fit_iso  # noqa
from search_robust_strategy import base_frame, NOTIONAL  # noqa
from search_reentry_strategy import build_candidates, apply_policy  # noqa

ART = TOOLS.parent / "artifacts_cleaned" / "poly_l2_only_v2"
CALIB = ["2026-04-22", "2026-04-30"]
EVAL = ["2026-05-07", "2026-05-15", "2026-05-21",
        "2026-05-28", "2026-06-01", "2026-06-10", "2026-06-11"]
NS = 1_000_000_000
DELAY = 2.0
FLOOR = 0.25
BANDS = [(0.25, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60)]


def main():
    model, feats, _ = load_model(ART)
    df_cal = load_days(CALIB)
    iso = fit_iso(score_model_p(model, feats, df_cal),
                  df_cal["label_up"].to_numpy().astype(np.float64))
    base = base_frame(model, feats, EVAL, iso)
    entries = apply_policy(build_candidates(base), 1, 0.0, 0.0)

    slugs = {e["market_slug"] for e in entries}
    raw = load_days(EVAL).select(["market_slug", "recv_ts_ns", "up_best_ask", "down_best_ask"])
    sub = raw.filter(pl.col("market_slug").is_in(list(slugs))).sort("recv_ts_ns")
    streams = {}
    for part in sub.partition_by("market_slug"):
        s = part["market_slug"][0]
        ts = part["recv_ts_ns"].to_numpy()
        ua = part["up_best_ask"].to_numpy().astype(np.float64)
        da = part["down_best_ask"].to_numpy().astype(np.float64)
        mu = (ua > 0) & (ua < 1); md = (da > 0) & (da < 1)
        streams[s] = {1: (ts[mu], ua[mu]), 0: (ts[md], da[md])}

    def delayed(slug, side, t):
        ts, ask = streams[slug][side]
        if len(ts) == 0:
            return None
        i = np.searchsorted(ts, t + int(DELAY * NS), "left")
        if i >= len(ts):
            i = len(ts) - 1
        return float(ask[i])

    by_band = defaultdict(lambda: {"n": 0, "crash": 0, "crash_won": 0, "crash_pnl": 0.0,
                                   "ok_won": 0, "ok_n": 0, "ok_pnl": 0.0})
    for e in entries:
        of = e["fill"]; side = e["side"]
        df = delayed(slug=e["market_slug"], side=side, t=e["recv_ts_ns"])
        if df is None:
            continue
        band = next((b for b in BANDS if b[0] <= of < b[1] or (b == BANDS[-1] and of <= b[1])), None)
        if band is None:
            continue
        d = by_band[band]; d["n"] += 1
        won = (side == 1 and e["label_up"] == 1) or (side == 0 and e["label_up"] == 0)
        pnl = NOTIONAL / df * (1.0 if won else 0.0) - NOTIONAL
        if df < FLOOR:
            d["crash"] += 1; d["crash_won"] += int(won); d["crash_pnl"] += pnl
        else:
            d["ok_n"] += 1; d["ok_won"] += int(won); d["ok_pnl"] += pnl

    print(f"2s-delayed fill: how often does price crash below {FLOOR} (market turned against us)\n")
    print(f"{'orig band':>11} {'n':>4} {'crashed<0.25':>13} {'crash win%':>11} {'crash pnl':>10} | "
          f"{'kept win%':>10} {'kept pnl':>9}")
    tot_n = tot_c = 0
    for b in BANDS:
        d = by_band[b]
        if d["n"] == 0:
            print(f"{f'{b[0]:.2f}-{b[1]:.2f}':>11} {0:>4}"); continue
        cw = 100*d["crash_won"]/d["crash"] if d["crash"] else float('nan')
        kw = 100*d["ok_won"]/d["ok_n"] if d["ok_n"] else float('nan')
        print(f"{f'{b[0]:.2f}-{b[1]:.2f}':>11} {d['n']:>4} "
              f"{d['crash']:>4} ({100*d['crash']/d['n']:>4.1f}%) "
              f"{cw:>10.1f}% {d['crash_pnl']:>+10.2f} | "
              f"{kw:>9.1f}% {d['ok_pnl']:>+9.2f}")
        tot_n += d["n"]; tot_c += d["crash"]
    print(f"\nTOTAL entries={tot_n}  crashed below {FLOOR} within 2s = {tot_c} ({100*tot_c/tot_n:.1f}%)")
    # focus answer
    for b in [(0.25, 0.30), (0.30, 0.40)]:
        d = by_band[b]
        if d["n"]:
            print(f"  band {b[0]:.2f}-{b[1]:.2f}: {d['crash']}/{d['n']} = "
                  f"{100*d['crash']/d['n']:.1f}% crashed below 0.25")


if __name__ == "__main__":
    main()
