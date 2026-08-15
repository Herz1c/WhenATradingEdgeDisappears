"""Backtest the fair_value_v1 model on the LIVE days (06-25/26/27), broken down
by side, to answer: did the bot make the RIGHT trades? In the original OOS
backtest both UP and DOWN were profitable; live the UP side bled. This replays
the SAME model + gate on the same days and reports UP vs DOWN per day, so we can
tell edge-decay apart from a live/backtest divergence.

Usage:
  py -3 tools/backtest_live_days_byside.py
  py -3 tools/backtest_live_days_byside.py --ttc-min 20 --ttc-max 90
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "datasets" / "fair_value_v1"
ART = ROOT / "artifacts" / "fair_value_v1"
DAYS = ["2026-06-25", "2026-06-26", "2026-06-27"]
EV_THR, PRICE_LO, PRICE_HI = 0.10, 0.30, 0.70


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def taker_fee(p):
    return 0.072 * p * (1 - p)


def main():
    import lightgbm as lgb, joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttc-min", type=float, default=10.0)
    ap.add_argument("--ttc-max", type=float, default=45.0)
    args = ap.parse_args()

    booster = lgb.Booster(model_file=str(ART / "model.txt"))
    feats = json.loads((ART / "features.json").read_text())
    iso = joblib.load(ART / "calibrator.pkl")

    print(f"=== fair_value_v1 backtest, gate ttc({args.ttc_min:.0f},{args.ttc_max:.0f}] "
          f"band[{PRICE_LO},{PRICE_HI}] EV>={EV_THR} ===\n")
    grand = defaultdict(lambda: [0, 0, 0.0])  # side -> n, wins, pnl
    print(f"{'day':12}{'UP n':>6}{'UP win':>8}{'UP pnl':>9}   {'DN n':>6}{'DN win':>8}{'DN pnl':>9}")
    for d in DAYS:
        f = DS / f"{d}.parquet"
        if not f.exists():
            print(f"{d:12}  (no parquet)"); continue
        df = pd.read_parquet(f)
        X = df[feats].astype(float).values
        init = _logit(df["implied_p_up"].astype(float).values)
        p = iso.transform(1.0 / (1.0 + np.exp(-(init + booster.predict(X, raw_score=True)))))

        m = (df["ttc_s"].to_numpy() > args.ttc_min) & (df["ttc_s"].to_numpy() <= args.ttc_max)
        d2 = df[m].copy(); p2 = p[m.nonzero()[0]]
        order = np.argsort(d2["now_ns"].to_numpy())
        d2 = d2.iloc[order]; p2 = p2[order]
        up_ask = d2["up_best_ask"].to_numpy(); dn_ask = d2["down_best_ask"].to_numpy()
        y_up = d2["resolved_up"].to_numpy(); slug = d2["market_slug"].to_numpy()
        ev_up = p2 - up_ask; ev_dn = (1 - p2) - dn_ask
        take_up = (ev_up >= EV_THR) & (ev_up >= ev_dn) & (up_ask > PRICE_LO) & (up_ask < PRICE_HI)
        take_dn = (ev_dn >= EV_THR) & (ev_dn > ev_up) & (dn_ask > PRICE_LO) & (dn_ask < PRICE_HI)

        seen = set(); day = defaultdict(lambda: [0, 0, 0.0])
        for i in range(len(d2)):
            s = slug[i]
            if s in seen: continue
            if take_up[i]:
                cost = up_ask[i] + taker_fee(up_ask[i]); won = y_up[i] == 1; side = "up"
            elif take_dn[i]:
                cost = dn_ask[i] + taker_fee(dn_ask[i]); won = y_up[i] == 0; side = "down"
            else:
                continue
            pnl = (1.0 if won else 0.0) - cost
            day[side][0] += 1; day[side][1] += int(won); day[side][2] += pnl
            grand[side][0] += 1; grand[side][1] += int(won); grand[side][2] += pnl
            seen.add(s)
        un, uw, up = day["up"]; dn, dw, dp = day["down"]
        print(f"{d:12}{un:>6}{(uw/un*100 if un else 0):>7.0f}%{up:>+9.2f}   "
              f"{dn:>6}{(dw/dn*100 if dn else 0):>7.0f}%{dp:>+9.2f}")
    un, uw, up = grand["up"]; dn, dw, dp = grand["down"]
    print(f"{'-'*70}")
    print(f"{'TOTAL':12}{un:>6}{(uw/un*100 if un else 0):>7.0f}%{up:>+9.2f}   "
          f"{dn:>6}{(dw/dn*100 if dn else 0):>7.0f}%{dp:>+9.2f}")
    print(f"\nbacktest total: {un+dn} trades, ${up+dp:+.2f}  (UP ${up:+.2f} / DOWN ${dp:+.2f})")


if __name__ == "__main__":
    main()
