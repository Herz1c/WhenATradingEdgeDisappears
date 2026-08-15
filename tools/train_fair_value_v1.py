"""Train the fair_value_v1 model and forward-walk evaluate it.

Model: LightGBM with init_score = logit(implied_p_up) — a RESIDUAL HEAD, so the
booster only learns the *correction* to the market price (the proven v2 trick).
Calibrate with global isotonic fit on a time-separated tail of the train set.

STRICT forward-walk: train on dates <= --train-end, test on dates >= --test-start
(never interleaved). Reports probability quality vs two baselines (the book
`implied_p_up`, and the analytic `p_bs`), and a fee-aware EV backtest on the test
set (one entry per market, taker fill at the quoted ask, Polymarket taker fee).

Usage:
  py -3 tools/train_fair_value_v1.py --train-end 2026-05-13 --test-start 2026-05-14
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "datasets" / "fair_value_v1"
ART = ROOT / "artifacts" / "fair_value_v1"

# Columns that are NOT model inputs.
META = {"market_slug", "now_ns", "resolved_up", "chainlink_close_price", "cex_source", "ttc_s"}
# chainlink_close_price = the future close = the LABEL in disguise -> must drop.
# Absolute price levels can proxy the date (temporal leak) -> drop; keep the deltas.
LEVEL_LEAK = {"cex_mid", "strike", "synthetic_chainlink_nowcast", "binance_mid"}
EXCLUDE = META | LEVEL_LEAK


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load_days(d0: str, d1: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(DS / "*.parquet")))
    frames = []
    for f in files:
        d = os.path.basename(f)[:10]
        if d0 <= d <= d1:
            frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE and df[c].dtype != object]


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    # rank-based AUC
    order = np.argsort(p)
    y = y[order]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def taker_fee(price):
    return 0.072 * price * (1.0 - price)


def ev_backtest(df, p, *, thr=0.10, fee=True, min_price=0.05, max_price=0.95):
    """One entry per market: pick the side with max EV over thr, fill at that
    side's ask, pay $1 if it resolves your way. Fee-aware. Returns summary."""
    up_ask = df["up_best_ask"].to_numpy()
    dn_ask = df["down_best_ask"].to_numpy()
    y_up = df["resolved_up"].to_numpy()
    slug = df["market_slug"].to_numpy()
    ev_up = p - up_ask
    ev_dn = (1 - p) - dn_ask
    # choose side
    take_up = (ev_up >= thr) & (ev_up >= ev_dn) & (up_ask > min_price) & (up_ask < max_price)
    take_dn = (ev_dn >= thr) & (ev_dn > ev_up) & (dn_ask > min_price) & (dn_ask < max_price)
    seen = set()
    pnl = []
    wins = 0
    for i in range(len(df)):
        s = slug[i]
        if s in seen:
            continue
        if take_up[i]:
            cost = up_ask[i] + (taker_fee(up_ask[i]) if fee else 0)
            payoff = 1.0 if y_up[i] == 1 else 0.0
            pnl.append(payoff - cost); wins += int(y_up[i] == 1); seen.add(s)
        elif take_dn[i]:
            cost = dn_ask[i] + (taker_fee(dn_ask[i]) if fee else 0)
            payoff = 1.0 if y_up[i] == 0 else 0.0
            pnl.append(payoff - cost); wins += int(y_up[i] == 0); seen.add(s)
    pnl = np.array(pnl)
    n = len(pnl)
    return {
        "trades": n,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "total_pnl": round(float(pnl.sum()), 2),
        "avg_pnl_per_trade": round(float(pnl.mean()), 4) if n else 0.0,
        "roi_on_notional": round(float(pnl.sum() / n), 4) if n else 0.0,
    }


def main():
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2026-05-13")
    ap.add_argument("--test-start", default="2026-05-14")
    ap.add_argument("--test-end", default="2026-12-31")
    ap.add_argument("--min-rtds-cov-day", type=float, default=0.0)  # placeholder
    ap.add_argument("--ev-thr", type=float, default=0.10)
    args = ap.parse_args()

    train = load_days("2000-01-01", args.train_end)
    test = load_days(args.test_start, args.test_end)
    if train.empty or test.empty:
        print(f"train rows={len(train)} test rows={len(test)} — need both. Abort.")
        return
    feats = feature_cols(train)
    print(f"train={len(train):,} rows  test={len(test):,} rows  features={len(feats)}")

    # time-separated calibration/early-stop val = last 2 train dates
    train = train.copy()
    train["date"] = (train["now_ns"] // 86_400_000_000_000)
    dates = sorted(train["date"].unique())
    val_dates = set(dates[-2:])
    tr = train[~train["date"].isin(val_dates)]
    va = train[train["date"].isin(val_dates)]

    def XY(d):
        X = d[feats].astype(float).values
        y = d["resolved_up"].astype(int).values
        init = _logit(d["implied_p_up"].astype(float).values)
        return X, y, init

    Xtr, ytr, itr = XY(tr)
    Xva, yva, iva = XY(va)

    dtr = lgb.Dataset(Xtr, label=ytr, init_score=itr, feature_name=feats)
    dva = lgb.Dataset(Xva, label=yva, init_score=iva, reference=dtr)
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, max_depth=-1, verbose=-1, metric="binary_logloss")
    booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])

    def predict(d):
        X = d[feats].astype(float).values
        init = _logit(d["implied_p_up"].astype(float).values)
        raw = booster.predict(X, raw_score=True)
        return 1.0 / (1.0 + np.exp(-(init + raw)))

    # isotonic calibrator fit on val (time-separated from both train-core and test)
    p_va = predict(va)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_va, yva)

    # ---- forward-walk evaluation on TEST ----
    p_test_raw = predict(test)
    p_test = iso.transform(p_test_raw)
    y_test = test["resolved_up"].astype(int).values
    p_book = test["implied_p_up"].astype(float).values
    p_bs = test["p_bs"].astype(float).values

    print("\n=== FORWARD-WALK (test) probability quality ===")
    print(f"{'model':22s}{'brier':>9s}{'logloss':>9s}{'auc':>8s}")
    for name, p in (("fair_value (cal)", p_test), ("fair_value (raw)", p_test_raw),
                    ("baseline p_bs", p_bs), ("baseline book", p_book)):
        print(f"{name:22s}{brier(p,y_test):>9.4f}{logloss(p,y_test):>9.4f}{auc(p,y_test):>8.3f}")

    print("\n=== FORWARD-WALK EV backtest (test, $1 notional, taker fee) ===")
    for label, p, fee in (("model cal + fee", p_test, True),
                          ("model cal NOfee", p_test, False),
                          ("book-only + fee", p_book, True)):
        r = ev_backtest(test.assign(_p=p), p, thr=args.ev_thr, fee=fee)
        print(f"  {label:18s} {r}")

    # by ttc bucket (model cal + fee)
    print("\n  by ttc bucket (model cal + fee):")
    tcol = test["ttc_s"].values
    for lo, hi in ((3, 15), (15, 30), (30, 60), (60, 90)):
        mask = (tcol >= lo) & (tcol < hi)
        if mask.sum() == 0:
            continue
        r = ev_backtest(test[mask], p_test[mask], thr=args.ev_thr, fee=True)
        print(f"    ttc[{lo:>2d},{hi:>2d})  {r}")

    # save
    ART.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(ART / "model.txt"))
    import joblib
    joblib.dump(iso, ART / "calibrator.pkl")
    (ART / "features.json").write_text(json.dumps(feats, indent=1))
    imp = sorted(zip(feats, booster.feature_importance(importance_type="gain")),
                 key=lambda x: -x[1])
    (ART / "feature_importance.json").write_text(json.dumps(imp, indent=1))
    print(f"\nTop features by gain: {[f for f,_ in imp[:12]]}")
    print(f"Saved -> {ART}")


if __name__ == "__main__":
    main()
