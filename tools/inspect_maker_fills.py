"""Inspect every fill from the maker-OOS backtest at a specific threshold.
We dump entry price, won/lost, payoff, PnL so we can audit the headline
win-rate / per-fill PnL numbers."""
from __future__ import annotations

import json
import sys
from collections import defaultdict, Counter
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


def load_data(dates):
    parts = []
    for d in dates:
        p = DATASET / f"{d}.parquet"
        if not p.exists(): continue
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
        if f not in cols: continue
        s = df.get_column(f)
        if not s.dtype.is_numeric(): continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    raw = model.predict_proba(X)[:, 1]
    cal = getattr(model, "_calibrator", None)
    if cal is not None:
        raw = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
    return raw


def run(threshold: float):
    df = load_data(OOS_DATES)
    df = clean_features(df)
    model = joblib.load(ART)
    feats = list(json.loads((ART.parent / "feature_importance.json").read_text()).keys())
    p_up = predict(model, feats, df)

    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()

    edge_dn = up_bid - p_up
    fut_ask = defaultdict(list)
    for i in range(len(df)):
        fut_ask[str(slugs[i])].append((int(snap_ts[i]), float(dn_ask[i])))

    pos_per_mkt = defaultdict(int)
    last_entry = {}
    min_gap_ns = int(10.0 * 1e9)

    fills = []   # list of dicts per fill
    for i in range(len(df)):
        if not (TTC_MIN_S <= ttc[i] <= TTC_MAX_S): continue
        if edge_dn[i] < threshold: continue
        if dn_ask[i] <= 0.02 or dn_ask[i] >= 0.98: continue
        slug = str(slugs[i])
        if pos_per_mkt[slug] >= 2: continue
        last = last_entry.get(slug)
        if last is not None and (int(snap_ts[i]) - last) < min_gap_ns: continue
        limit = round((dn_bid[i] + dn_ask[i]) / 2.0, 2)
        if limit <= 0 or limit >= 1: continue
        pos_per_mkt[slug] += 1
        last_entry[slug] = int(snap_ts[i])

        # fill check
        filled = False
        for t, a in fut_ask[slug]:
            if t <= int(snap_ts[i]): continue
            if t >= int(close_ts[i]): break
            if a <= limit:
                filled = True; break
        if not filled: continue

        won = (resolved[i] == 0)
        fills.append({
            "slug": slug,
            "limit": limit,
            "won": won,
            "payoff": 1.0 if won else 0.0,
            "pnl": (1.0 if won else 0.0) - limit,
            "edge_dn": float(edge_dn[i]),
            "p_up": float(p_up[i]),
            "dn_bid": float(dn_bid[i]),
            "dn_ask": float(dn_ask[i]),
            "ttc": float(ttc[i]),
        })

    if not fills:
        print(f"No fills at threshold {threshold}.")
        return

    n = len(fills)
    wins = sum(1 for f in fills if f["won"])
    avg_entry = np.mean([f["limit"] for f in fills])
    median_entry = np.median([f["limit"] for f in fills])
    avg_pnl = np.mean([f["pnl"] for f in fills])
    total_pnl = sum(f["pnl"] for f in fills)

    print(f"=== threshold = {threshold} ===")
    print(f"fills: {n}   wins: {wins}   win_rate: {wins/n:.1%}   total_pnl: ${total_pnl:+.2f}")
    print(f"avg entry price : ${avg_entry:.3f}")
    print(f"med entry price : ${median_entry:.3f}")
    print(f"avg PnL/fill    : ${avg_pnl:+.4f}")
    print()
    # Entry price histogram
    bucket_edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    print(f"{'price_bucket':>12} | {'n_fills':>7} | {'n_wins':>6} | {'win_rate':>8} | {'avg_pnl':>8}")
    for j in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[j], bucket_edges[j+1]
        in_b = [f for f in fills if lo <= f["limit"] < hi]
        if not in_b: continue
        w = sum(1 for f in in_b if f["won"])
        avg = np.mean([f["pnl"] for f in in_b])
        print(f"  [{lo:.2f},{hi:.2f}) | {len(in_b):>7d} | {w:>6d} | {w/len(in_b):>7.1%}"
              f" | ${avg:>+7.4f}")

    # Show first 10 fills as raw rows
    print()
    print(f"first 10 fills:")
    print(f"  {'slug':<32} {'limit':>6} {'won':>4} {'pnl':>7} {'p_up':>6} {'dn_ask':>7}")
    for f in fills[:10]:
        print(f"  {f['slug']:<32} {f['limit']:>6.3f} {str(f['won']):>4} "
              f"{f['pnl']:>+7.3f} {f['p_up']:>6.3f} {f['dn_ask']:>7.3f}")

    # Sanity: how would win rate distribute with NO edge?
    # If model is useless, win_rate ~= 1 - avg_entry (since DOWN priced X% should win X% of the time)
    print()
    expected_naive_win = 1 - avg_entry   # if maker fills are at midpoint and market is fair
    print(f"naive expected win rate if model has zero edge: {expected_naive_win:.1%}")
    print(f"actual win rate                                : {wins/n:.1%}")
    print(f"excess (model edge contribution)               : {(wins/n) - expected_naive_win:+.1%}")


if __name__ == "__main__":
    for t in (0.07, 0.15, 0.30):
        run(t); print()
