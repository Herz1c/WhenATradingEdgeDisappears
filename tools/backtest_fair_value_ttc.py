"""Backtest the fair_value_v1 model over DIFFERENT ttc entry windows.

Hypothesis under test (user): the bot enters too early; the edge lives in the
last ~60s. Compare the current live window ttc[20,90] against ttc(10,60) and a
few finer slices, on the strict forward-walk OOS days (model trained <=05-13).

Loads the SAVED model/calibrator (does NOT retrain), mirrors the exact live EV
gate (band [0.30,0.70], EV>=thr, one entry/market, taker fee), and plots a
cumulative-PnL equity curve per window.

Usage:
  py -3 tools/backtest_fair_value_ttc.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "datasets" / "fair_value_v1"
ART = ROOT / "artifacts" / "fair_value_v1"
OUT = ROOT / "artifacts" / "fair_value_v1" / "ttc_backtest.png"

TEST_START = "2026-05-14"   # strict OOS: model trained on <= 2026-05-13
TEST_END = "2026-12-31"

EV_THR = 0.10
PRICE_LO, PRICE_HI = 0.30, 0.70

# windows to compare: (label, ttc_min, ttc_max)
WINDOWS = [
    ("live  ttc[20,90]", 20, 90),
    ("prop  ttc(10,60]", 10, 60),
    ("comp  ttc(10,45]", 10, 45),
    ("late  ttc(10,30]", 10, 30),
]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load_days(d0: str, d1: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(DS / "*.parquet"))):
        d = os.path.basename(f)[:10]
        if d0 <= d <= d1:
            frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def taker_fee(price):
    return 0.072 * price * (1.0 - price)


def ev_backtest(df, p, *, ttc_min, ttc_max, thr=EV_THR):
    """One entry per market in time order, within the ttc window. Pick the side
    with max EV over thr, fill at that side's ask, pay $1 if it resolves your
    way, minus taker fee. Returns per-trade arrays (sorted by entry time)."""
    m = (df["ttc_s"].to_numpy() > ttc_min) & (df["ttc_s"].to_numpy() <= ttc_max)
    d = df[m]
    p = p[m.nonzero()[0]] if m.dtype == bool else p[m]
    # sort by time so "first eligible" = earliest snapshot, and equity curve is chronological
    order = np.argsort(d["now_ns"].to_numpy())
    d = d.iloc[order]
    pp = p[order]

    up_ask = d["up_best_ask"].to_numpy()
    dn_ask = d["down_best_ask"].to_numpy()
    y_up = d["resolved_up"].to_numpy()
    slug = d["market_slug"].to_numpy()
    ts = d["now_ns"].to_numpy()
    ev_up = pp - up_ask
    ev_dn = (1 - pp) - dn_ask
    take_up = (ev_up >= thr) & (ev_up >= ev_dn) & (up_ask > PRICE_LO) & (up_ask < PRICE_HI)
    take_dn = (ev_dn >= thr) & (ev_dn > ev_up) & (dn_ask > PRICE_LO) & (dn_ask < PRICE_HI)

    seen = set()
    pnl, wins_ts, won_flags = [], [], []
    for i in range(len(d)):
        s = slug[i]
        if s in seen:
            continue
        if take_up[i]:
            cost = up_ask[i] + taker_fee(up_ask[i])
            won = (y_up[i] == 1)
        elif take_dn[i]:
            cost = dn_ask[i] + taker_fee(dn_ask[i])
            won = (y_up[i] == 0)
        else:
            continue
        pnl.append((1.0 if won else 0.0) - cost)
        won_flags.append(int(won))
        wins_ts.append(ts[i])
        seen.add(s)
    return np.array(wins_ts), np.array(pnl), np.array(won_flags)


def main():
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(ART / "model.txt"))
    feats = json.loads((ART / "features.json").read_text())
    import joblib
    iso = joblib.load(ART / "calibrator.pkl")

    test = load_days(TEST_START, TEST_END)
    if test.empty:
        print("no OOS data"); return
    days = sorted(set(os.path.basename(p)[:10] for p in glob.glob(str(DS / "*.parquet"))
                      if TEST_START <= os.path.basename(p)[:10] <= TEST_END))
    print(f"OOS rows={len(test):,}  days={days[0]}..{days[-1]} ({len(days)})")

    X = test[feats].astype(float).values
    init = _logit(test["implied_p_up"].astype(float).values)
    raw = booster.predict(X, raw_score=True)
    p_raw = 1.0 / (1.0 + np.exp(-(init + raw)))
    p = iso.transform(p_raw)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.5),
                                   gridspec_kw={"width_ratios": [2.2, 1]})

    print(f"\n{'window':20s}{'trades':>8}{'win%':>8}{'PnL$1':>9}{'$/trade':>9}{'ROI%':>8}")
    results = {}
    for label, lo, hi in WINDOWS:
        ts, pnl, won = ev_backtest(test, p, ttc_min=lo, ttc_max=hi)
        n = len(pnl)
        if n == 0:
            print(f"{label:20s}{0:>8}"); continue
        wr = won.mean()
        tot = pnl.sum()
        roi = tot / n * 100
        results[label] = (ts, np.cumsum(pnl), n, wr, tot, roi)
        print(f"{label:20s}{n:>8}{wr*100:>7.1f}%{tot:>9.2f}{tot/n:>9.4f}{roi:>7.1f}%")
        # equity curve vs trade index
        axL.plot(np.arange(1, n + 1), np.cumsum(pnl), lw=1.8,
                 label=f"{label.strip()}  n={n} wr={wr*100:.0f}% ${tot:+.0f}")

    axL.axhline(0, color="0.6", lw=0.8)
    axL.set_xlabel("trade # (chronological)")
    axL.set_ylabel("cumulative PnL  ($1/share notional)")
    axL.set_title(f"fair_value_v1 equity curve by ttc window\nOOS {days[0]}..{days[-1]}  "
                  f"band[{PRICE_LO},{PRICE_HI}] EV>={EV_THR}")
    axL.legend(fontsize=8, loc="upper left")
    axL.grid(alpha=0.25)

    # right panel: PnL per trade by ttc bucket (direct test of "edge lives late")
    buckets = [(4, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90)]
    bx, by, bn = [], [], []
    for lo, hi in buckets:
        _, pnl, won = ev_backtest(test, p, ttc_min=lo, ttc_max=hi)
        if len(pnl) == 0:
            bx.append(f"{lo}-{hi}"); by.append(0); bn.append(0); continue
        bx.append(f"{lo}-{hi}"); by.append(pnl.mean()); bn.append(len(pnl))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in by]
    axR.bar(range(len(bx)), by, color=colors)
    axR.set_xticks(range(len(bx)))
    axR.set_xticklabels(bx, rotation=0, fontsize=8)
    for i, (v, nn) in enumerate(zip(by, bn)):
        axR.annotate(f"n={nn}", (i, v), ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=7)
    axR.axhline(0, color="0.4", lw=0.8)
    axR.set_xlabel("ttc bucket (s to close)")
    axR.set_ylabel("avg PnL per trade ($)")
    axR.set_title("edge by ttc bucket\n(each snapshot, no 1-entry cap)")
    axR.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
