"""Re-run the maker OOS threshold sweep with MIN_DOWN_PRICE = 0.30
applied to BOTH gates (decision-engine: skip if down_ask < 0.30; maker:
skip if midpoint limit < 0.30). Reports day-by-day PnL so we can see
how the floor reshapes per-day distribution.

Same data, same model, same per-market caps (2 / 10s gap) as the live
maker -- only the price floor changes vs the earlier sweep.
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
MIN_DOWN_PRICE = 0.30           # the new floor
MAX_POS_PER_MARKET = 2
MIN_GAP_S = 10.0


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


def simulate(df, p_up, *, edge_threshold):
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()
    dates = df["date"].to_numpy()

    edge_dn = up_bid - p_up

    fut_ask = defaultdict(list)
    for i in range(len(df)):
        fut_ask[str(slugs[i])].append((int(snap_ts[i]), float(dn_ask[i])))

    pos_per_mkt = defaultdict(int)
    last_entry_ns: dict[str, int] = {}
    min_gap_ns = int(MIN_GAP_S * 1e9)

    fills_by_date = defaultdict(int)
    wins_by_date = defaultdict(int)
    pnl_by_date = defaultdict(float)
    n_sig_total = n_fill_total = n_win_total = 0
    pnl_total = 0.0
    # Counters for the gates (to show what the floor is filtering)
    n_skip_ask_floor = 0      # decision-engine gate: down_ask < 0.30
    n_skip_lim_floor = 0      # maker gate: midpoint < 0.30

    for i in range(len(df)):
        if not (TTC_MIN_S <= ttc[i] <= TTC_MAX_S):
            continue
        if edge_dn[i] < edge_threshold:
            continue
        # decision-engine MIN_DOWN_PRICE gate (uses ask)
        if dn_ask[i] < MIN_DOWN_PRICE:
            n_skip_ask_floor += 1
            continue
        if dn_ask[i] <= 0.02 or dn_ask[i] >= 0.98:
            continue
        slug = str(slugs[i])
        if pos_per_mkt[slug] >= MAX_POS_PER_MARKET:
            continue
        last = last_entry_ns.get(slug)
        if last is not None and (int(snap_ts[i]) - last) < min_gap_ns:
            continue
        limit = round((dn_bid[i] + dn_ask[i]) / 2.0, 2)
        if not (0.02 <= limit <= 0.98):
            continue
        # maker gate: midpoint < 0.30
        if limit < MIN_DOWN_PRICE:
            n_skip_lim_floor += 1
            continue
        n_sig_total += 1
        pos_per_mkt[slug] += 1
        last_entry_ns[slug] = int(snap_ts[i])

        filled = False
        for t, a in fut_ask[slug]:
            if t <= int(snap_ts[i]): continue
            if t >= int(close_ts[i]): break
            if a <= limit:
                filled = True; break
        if not filled:
            continue

        n_fill_total += 1
        won = (resolved[i] == 0)
        pnl = (1.0 if won else 0.0) - limit
        date = str(dates[i])
        pnl_total += pnl
        fills_by_date[date] += 1
        pnl_by_date[date] += pnl
        if won:
            n_win_total += 1
            wins_by_date[date] += 1

    return {
        "thr": edge_threshold,
        "n_sig": n_sig_total, "n_fill": n_fill_total, "n_win": n_win_total,
        "fill_rate": n_fill_total / max(1, n_sig_total),
        "win_rate": n_win_total / max(1, n_fill_total),
        "pnl": pnl_total,
        "per_fill": pnl_total / max(1, n_fill_total),
        "by_date": {
            d: {"fills": fills_by_date[d], "wins": wins_by_date[d],
                "pnl": pnl_by_date[d]}
            for d in OOS_DATES
        },
        "n_skip_ask_floor": n_skip_ask_floor,
        "n_skip_lim_floor": n_skip_lim_floor,
    }


def main():
    print(f"loading OOS data: {OOS_DATES}")
    df = load_data(OOS_DATES)
    print(f"  {len(df)} snapshots, {df['market_slug'].n_unique()} markets")
    print(f"  MIN_DOWN_PRICE = {MIN_DOWN_PRICE}, MAX_POS_PER_MARKET = {MAX_POS_PER_MARKET}, "
          f"MIN_GAP_S = {MIN_GAP_S}")
    df = clean_features(df)
    model = joblib.load(ART)
    feats = list(json.loads((ART.parent / "feature_importance.json").read_text()).keys())
    p_up = predict(model, feats, df)

    thresholds = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]

    rows = []
    print()
    header = (f"{'thr':>5} | {'fills':>5} | {'win%':>5} | {'$/fill':>7} | {'pnl':>8}"
              f" | {'skip<asK':>8} | {'skip<lim':>8}")
    print(header); print("-" * len(header))
    for thr in thresholds:
        r = simulate(df, p_up, edge_threshold=thr)
        rows.append(r)
        print(f"{thr:>5.2f} | {r['n_fill']:>5d} | {r['win_rate']:>5.1%} |"
              f" {r['per_fill']:>+7.4f} | {r['pnl']:>+8.2f} |"
              f" {r['n_skip_ask_floor']:>8d} | {r['n_skip_lim_floor']:>8d}")
    print()

    # Detailed per-day breakdown at each threshold
    for r in rows:
        print(f"\n--- thr = {r['thr']:.2f}  total fills={r['n_fill']}  total PnL=${r['pnl']:+.2f}"
              f"  win={r['win_rate']:.1%}  $/fill={r['per_fill']:+.4f} ---")
        print(f"  {'date':<12} {'fills':>6} {'wins':>5} {'loss':>5} {'win%':>6} {'pnl_$':>8} {'$/fill':>8}")
        for d in OOS_DATES:
            v = r["by_date"][d]
            f, w, p = v["fills"], v["wins"], v["pnl"]
            if f == 0:
                print(f"  {d:<12} {0:>6d} {0:>5d} {0:>5d} {'-':>6} {0.0:>+8.2f} {'-':>8}")
                continue
            print(f"  {d:<12} {f:>6d} {w:>5d} {f-w:>5d} {w/f:>5.1%}"
                  f"  {p:>+8.2f}  {p/f:>+8.4f}")


if __name__ == "__main__":
    main()
