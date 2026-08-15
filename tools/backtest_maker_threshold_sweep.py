"""Fine-grained sweep of edge_threshold for the maker (POST_ONLY at midpoint)
strategy on OOS data, to find the threshold that maximises trade volume
while keeping per-fill PnL positive.

The earlier coarse sweep (tools/backtest_maker_oos.py) showed that midpoint
maker is profitable at every threshold from 0.00 to 0.30, with the
high-volume sweet spot somewhere around 0.05. This script zooms in with
0.01 steps so we can pick the lowest threshold that still has good
per-fill economics (= more trades per day).

Uses the SAME data load and predict pipeline as backtest_maker_oos.py;
just sweeps more thresholds and reports a few more derived metrics.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from feature_cleanup import clean_features  # noqa: E402

DATASET = REPO / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close"
ART = REPO / "artifacts_cleaned" / "model_02_fair_resolution" / "dense_close" / "lightgbm" / "model.pkl"

OOS_DATES = ["2026-05-21", "2026-05-22", "2026-05-26", "2026-05-29"]
TTC_MIN_S, TTC_MAX_S = 10.0, 60.0
SIZE = 1.0


def load_data(dates: list[str]) -> pl.DataFrame:
    parts = []
    for d in dates:
        p = DATASET / f"{d}.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        df = df.filter(pl.col("resolved_side_label").is_not_null())
        df = df.filter(pl.col("up_token_best_bid").is_not_null())
        df = df.filter(pl.col("up_token_best_ask").is_not_null())
        df = df.filter(pl.col("down_token_best_bid").is_not_null())
        df = df.filter(pl.col("down_token_best_ask").is_not_null())
        df = df.filter(pl.col("up_token_best_ask") > 0.01)
        df = df.filter(pl.col("up_token_best_bid") < 0.99)
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
    return pl.concat(parts, how="diagonal").sort("snapshot_ts_ns")


def predict(model, feats, df):
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    cols = set(df.columns)
    for i, f in enumerate(feats):
        if f not in cols:
            continue
        s = df.get_column(f)
        if not s.dtype.is_numeric():
            continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    raw = model.predict_proba(X)[:, 1]
    cal = getattr(model, "_calibrator", None)
    if cal is not None:
        raw = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
    return raw


def simulate(df, p_up, *, edge_threshold, max_pos_per_market=2, min_gap_s=10.0):
    """Same maker-fill logic as backtest_maker_oos.py, with `midpoint`
    pricing and the new per-market policy (2 fills max, 10s apart)."""
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()

    edge_dn = up_bid - p_up

    # Per-market future-ask lookup
    fut_ask: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i in range(len(df)):
        fut_ask[str(slugs[i])].append((int(snap_ts[i]), float(dn_ask[i])))

    pos_per_mkt: dict[str, int] = defaultdict(int)
    last_entry_ns: dict[str, int] = {}
    min_gap_ns = int(min_gap_s * 1e9)

    n_sig = n_fill = n_win = n_loss = 0
    pnl = 0.0

    for i in range(len(df)):
        if not (TTC_MIN_S <= ttc[i] <= TTC_MAX_S):
            continue
        if edge_dn[i] < edge_threshold:
            continue
        if dn_ask[i] <= 0.02 or dn_ask[i] >= 0.98:
            continue
        slug = str(slugs[i])
        if pos_per_mkt[slug] >= max_pos_per_market:
            continue
        # Gap rule
        last = last_entry_ns.get(slug)
        if last is not None and (int(snap_ts[i]) - last) < min_gap_ns:
            continue
        # Midpoint pricing
        limit = round((dn_bid[i] + dn_ask[i]) / 2.0, 2)
        if limit <= 0 or limit >= 1:
            continue
        n_sig += 1
        pos_per_mkt[slug] += 1
        last_entry_ns[slug] = int(snap_ts[i])
        # Fill: any future snapshot of same market with ask <= limit, before close
        filled = False
        for t, a in fut_ask[slug]:
            if t <= int(snap_ts[i]):
                continue
            if t >= int(close_ts[i]):
                break
            if a <= limit:
                filled = True
                break
        if not filled:
            continue
        n_fill += 1
        won = (resolved[i] == 0)   # DOWN wins iff label == 0
        pnl_i = (1.0 if won else 0.0) - limit
        pnl += pnl_i * SIZE
        if won: n_win += 1
        else:   n_loss += 1

    return {
        "thr": edge_threshold,
        "n_sig": n_sig, "n_fill": n_fill,
        "fill_rate": n_fill / max(1, n_sig),
        "win_rate": n_win / max(1, n_fill),
        "pnl": pnl,
        "per_fill": pnl / max(1, n_fill),
        "fills_per_day": n_fill / 4.0,   # 4 OOS days
        "pnl_per_day": pnl / 4.0,
    }


def main():
    print(f"loading OOS data: {OOS_DATES}")
    df = load_data(OOS_DATES)
    print(f"  {len(df)} snapshots, {df['market_slug'].n_unique()} markets")
    df = clean_features(df)
    model = joblib.load(ART)
    feats = list(json.loads((ART.parent / "feature_importance.json").read_text()).keys())
    p_up = predict(model, feats, df)

    # FINE sweep -- 0.01 step from -0.05 (model agrees with market or worse)
    # up to 0.30 (high-conviction signals only)
    thresholds = [round(x, 3) for x in np.arange(-0.05, 0.31, 0.01)]

    print(f"\nfine-grain sweep (midpoint pricing, 2-per-market, 10 s gap):")
    print("=" * 100)
    print(f"{'thr':>6} | {'signals':>7} | {'fills':>5} | {'fill%':>5} | {'win%':>5}"
          f" | {'PnL_$':>8} | {'$/fill':>7} | {'fills/day':>9} | {'$/day':>7}")
    print("-" * 100)
    rows = []
    for thr in thresholds:
        r = simulate(df, p_up, edge_threshold=thr)
        rows.append(r)
        print(f"{thr:>6.2f} | {r['n_sig']:>7d} | {r['n_fill']:>5d} | {r['fill_rate']:>5.1%}"
              f" | {r['win_rate']:>5.1%} | {r['pnl']:>+8.2f} | {r['per_fill']:>+7.4f}"
              f" | {r['fills_per_day']:>9.1f} | {r['pnl_per_day']:>+7.2f}")
    print("=" * 100)

    # Recommend a few "sweet spots"
    creditable = [r for r in rows if r["n_fill"] >= 20]   # need decent sample
    if creditable:
        # Best by total PnL
        b_pnl = max(creditable, key=lambda r: r["pnl"])
        # Best by per-fill PnL
        b_pf = max(creditable, key=lambda r: r["per_fill"])
        # Highest fills/day with PnL still positive
        b_vol = max([r for r in creditable if r["pnl"] > 0],
                    key=lambda r: r["fills_per_day"], default=None)
        # Best Sharpe-ish: pnl / sqrt(n_fill)
        b_sharpe = max(creditable, key=lambda r: r["pnl"] / max(1, np.sqrt(r["n_fill"])))

        print(f"\nRECOMMENDATIONS (>= 20 fills):")
        print(f"  max total PnL      : thr={b_pnl['thr']:.2f}  pnl=${b_pnl['pnl']:+.2f}"
              f"  fills/day={b_pnl['fills_per_day']:.1f}  win={b_pnl['win_rate']:.1%}")
        print(f"  max per-fill PnL   : thr={b_pf['thr']:.2f}  pnl=${b_pf['pnl']:+.2f}"
              f"  fills/day={b_pf['fills_per_day']:.1f}  win={b_pf['win_rate']:.1%}")
        print(f"  max volume (PnL>0) : thr={b_vol['thr']:.2f}  pnl=${b_vol['pnl']:+.2f}"
              f"  fills/day={b_vol['fills_per_day']:.1f}  win={b_vol['win_rate']:.1%}")
        print(f"  best risk-adj      : thr={b_sharpe['thr']:.2f}  pnl=${b_sharpe['pnl']:+.2f}"
              f"  fills/day={b_sharpe['fills_per_day']:.1f}  win={b_sharpe['win_rate']:.1%}")


if __name__ == "__main__":
    main()
