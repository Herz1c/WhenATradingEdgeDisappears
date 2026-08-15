"""Analyse the no-bias model's OOS fills to find tight entry rules that
yield high PnL/fill without collapsing trade count to zero.

Approach:
  1. Simulate at a permissive threshold (0.02) on TEST days, capturing
     every potential fill plus rich context (edge, entry price, ttc,
     btc_return, vol, side, p_up_calibrated, etc.)
  2. Bin by each axis to find where PnL is great vs terrible.
  3. Combine the best filters into composite rules and re-evaluate.

Output: a recommended rule set + before/after numbers.
"""
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

ART_DIR = REPO / "artifacts_cleaned" / "model_no_bias_v1"
DC = REPO / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close_no_bias"
TEST_DAYS = ["2026-05-21", "2026-05-22", "2026-05-26", "2026-05-29"]

TTC_MIN_S, TTC_MAX_S = 10.0, 60.0
SIZE = 1.0


def load_split(dates):
    parts = []
    for d in dates:
        p = DC / f"{d}.parquet"
        if not p.exists(): continue
        df = pl.read_parquet(p)
        df = df.filter(pl.col("resolved_side_label").is_not_null())
        df = df.filter(pl.col("up_token_best_bid").is_not_null())
        df = df.filter(pl.col("up_token_best_ask").is_not_null())
        df = df.filter(pl.col("down_token_best_bid").is_not_null())
        df = df.filter(pl.col("down_token_best_ask").is_not_null())
        df = df.filter(pl.col("up_token_best_ask") > 0.01)
        df = df.filter(pl.col("up_token_best_bid") < 0.99)
        df = df.filter((pl.col("t_to_close_s") > TTC_MIN_S) &
                       (pl.col("t_to_close_s") < TTC_MAX_S))
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
    return pl.concat(parts, how="diagonal").sort("snapshot_ts_ns")


def build_X(df, feats):
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    cols = set(df.columns)
    for i, f in enumerate(feats):
        if f not in cols: continue
        s = df.get_column(f)
        if not s.dtype.is_numeric(): continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    return X


def simulate_with_context(df, p_up, *, threshold=0.02, min_price=0.30,
                          max_pos=2, min_gap_s=10.0):
    """Same maker simulation, recording context per fill."""
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    up_ask = df["up_token_best_ask"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    ttc    = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    snap_ts  = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()
    dates = df["date"].to_numpy()
    btc_ret_5s = df["btc_return_5s_raw"].to_numpy().astype(float)
    btc_ret_10s = df["btc_return_10s_raw"].to_numpy().astype(float)
    btc_vol_15s = df["btc_realized_vol_15s_raw"].to_numpy().astype(float)
    up_imb = df["up_token_imbalance"].to_numpy().astype(float)
    dn_imb = df["down_token_imbalance"].to_numpy().astype(float)
    delta_raw = df["delta_to_strike_raw"].to_numpy().astype(float)

    edge_up = p_up - up_ask
    edge_dn = up_bid - p_up

    fut_up_ask = defaultdict(list)
    fut_dn_ask = defaultdict(list)
    for i in range(len(df)):
        fut_up_ask[str(slugs[i])].append((int(snap_ts[i]), float(up_ask[i])))
        fut_dn_ask[str(slugs[i])].append((int(snap_ts[i]), float(dn_ask[i])))

    pos_per_mkt_side = defaultdict(int)
    last_entry_ns = {}
    gap_ns = int(min_gap_s * 1e9)
    fills = []

    for i in range(len(df)):
        eu = edge_up[i]; ed = edge_dn[i]
        if eu < threshold and ed < threshold: continue
        side = "UP" if eu >= ed else "DOWN"
        if side == "UP" and eu < threshold: continue
        if side == "DOWN" and ed < threshold: continue
        bid = up_bid[i] if side == "UP" else dn_bid[i]
        ask = up_ask[i] if side == "UP" else dn_ask[i]
        if not (min_price <= ask < 1.0): continue
        if not (0 < bid < ask): continue
        slug = str(slugs[i])
        key = (slug, side)
        if pos_per_mkt_side[key] >= max_pos: continue
        last = last_entry_ns.get(key)
        if last is not None and (int(snap_ts[i]) - last) < gap_ns: continue
        limit = round((bid + ask) / 2.0, 2)
        if limit < min_price: continue
        if not (0.02 <= limit <= 0.98): continue
        pos_per_mkt_side[key] += 1
        last_entry_ns[key] = int(snap_ts[i])
        fut = fut_up_ask[slug] if side == "UP" else fut_dn_ask[slug]
        filled = False
        for t, a in fut:
            if t <= int(snap_ts[i]): continue
            if t >= int(close_ts[i]): break
            if a <= limit:
                filled = True; break
        if not filled: continue
        won = (resolved[i] == 1) if side == "UP" else (resolved[i] == 0)
        edge_at_entry = eu if side == "UP" else ed
        pnl = (1.0 if won else 0.0) - limit
        fills.append({
            "date": str(dates[i]),
            "slug": slug,
            "side": side,
            "ttc": float(ttc[i]),
            "limit": limit,
            "edge": float(edge_at_entry),
            "p_up": float(p_up[i]),
            "btc_ret_5s": float(btc_ret_5s[i]) if np.isfinite(btc_ret_5s[i]) else 0.0,
            "btc_ret_10s": float(btc_ret_10s[i]) if np.isfinite(btc_ret_10s[i]) else 0.0,
            "btc_vol_15s": float(btc_vol_15s[i]) if np.isfinite(btc_vol_15s[i]) else 0.0,
            "up_imb": float(up_imb[i]) if np.isfinite(up_imb[i]) else 0.0,
            "dn_imb": float(dn_imb[i]) if np.isfinite(dn_imb[i]) else 0.0,
            "delta_raw": float(delta_raw[i]) if np.isfinite(delta_raw[i]) else 0.0,
            "won": int(won),
            "pnl": float(pnl) * SIZE,
        })
    return fills


def bucket_stats(fills, key_fn, label, n_buckets=8):
    """Sort fills by key, split into N buckets, report stats per bucket."""
    fills = [f for f in fills if key_fn(f) is not None and np.isfinite(key_fn(f))]
    if not fills:
        print(f"  no fills with this key"); return
    vals = np.array([key_fn(f) for f in fills])
    cuts = np.quantile(vals, np.linspace(0, 1, n_buckets+1))
    print(f"\n--- {label} ({len(fills)} fills) ---")
    print(f"  {'bucket':<22} {'n':>5} {'win%':>5} {'avg_pnl':>8} {'total':>8}")
    for b in range(n_buckets):
        lo, hi = cuts[b], cuts[b+1]
        if b == n_buckets - 1:
            mask = (vals >= lo) & (vals <= hi)
        else:
            mask = (vals >= lo) & (vals < hi)
        sub = [f for j, f in enumerate(fills) if mask[j]]
        if not sub: continue
        wins = sum(f["won"] for f in sub)
        pnls = np.array([f["pnl"] for f in sub])
        bucket_label = f"[{lo:.4g}, {hi:.4g})"
        print(f"  {bucket_label:<22} {len(sub):>5} {wins/len(sub):>4.1%} "
              f"${pnls.mean():>+7.4f} ${pnls.sum():>+7.2f}")


def evaluate_rule(fills, rule_fn, label):
    """Apply a rule (predicate), report PnL/win/n compared to baseline."""
    kept = [f for f in fills if rule_fn(f)]
    if not kept:
        print(f"  rule={label}: 0 fills kept"); return
    wins = sum(f["won"] for f in kept)
    pnls = np.array([f["pnl"] for f in kept])
    print(f"  rule={label}: n={len(kept):>4} win={wins/len(kept):>5.1%} "
          f"avg=${pnls.mean():>+.4f} total=${pnls.sum():>+.2f}")


def main():
    model = joblib.load(ART_DIR / "model.pkl")
    feats = json.loads((ART_DIR / "experiment_manifest.json").read_text())["features"]
    cal = getattr(model, "_calibrator", None)

    df = load_split(TEST_DAYS)
    df_clean = clean_features(df)
    X = build_X(df_clean, feats)
    p_up = cal.predict(model.predict_proba(X)[:, 1]) if cal else model.predict_proba(X)[:, 1]

    # Simulate at permissive threshold to gather large fill pool
    fills = simulate_with_context(df, p_up, threshold=0.02)
    if not fills:
        print("no fills."); return

    n = len(fills); wins = sum(f["won"] for f in fills)
    tot = sum(f["pnl"] for f in fills)
    print(f"\n=== baseline (threshold=0.02, both sides) ===")
    print(f"  n={n}  win={wins/n:.1%}  total=${tot:+.2f}  per_fill=${tot/n:+.4f}")

    # Bucket-by-axis to see what predicts winning
    bucket_stats(fills, lambda f: f["edge"],           "EDGE")
    bucket_stats(fills, lambda f: f["limit"],          "ENTRY PRICE")
    bucket_stats(fills, lambda f: f["ttc"],            "TTC (sec to close)")
    bucket_stats(fills, lambda f: f["btc_ret_5s"]*1e4, "BTC_RETURN_5s (bps)")
    bucket_stats(fills, lambda f: f["btc_vol_15s"]*1e4, "BTC_VOL_15s (bps)")
    bucket_stats(fills, lambda f: f["up_imb"],         "UP_TOKEN_IMBALANCE")
    bucket_stats(fills, lambda f: f["delta_raw"],      "RAW_DELTA_TO_STRIKE")

    # Combined side-specific lookup
    for side in ("UP", "DOWN"):
        side_fills = [f for f in fills if f["side"] == side]
        if side_fills:
            print(f"\n=== {side} side breakdown ({len(side_fills)} fills) ===")
            bucket_stats(side_fills, lambda f: f["edge"],     f"{side}: EDGE")
            bucket_stats(side_fills, lambda f: f["limit"],    f"{side}: ENTRY PRICE")
            bucket_stats(side_fills, lambda f: f["ttc"],      f"{side}: TTC")

    # Test some candidate composite rules
    print(f"\n=== composite rule tests (refined) ===")
    evaluate_rule(fills, lambda f: True, "no_rule (baseline)")

    # Filter out the explicit BAD zones found in the bucket analysis:
    #  - UP side at entry price [0.36, 0.46) lost money badly
    #  - raw_delta in [51, 86] range: only 27-39% win rate (danger zone)
    #  - mid-range edges 0.07-0.20 with low avg PnL
    evaluate_rule(fills,
        lambda f: not (f["side"] == "UP" and 0.36 <= f["limit"] < 0.46),
        "skip UP at limit in [0.36, 0.46) (the worst bucket)")
    evaluate_rule(fills,
        lambda f: not (51.0 <= f["delta_raw"] <= 86.0),
        "skip raw_delta in danger zone [51, 86]")
    evaluate_rule(fills,
        lambda f: f["up_imb"] >= 0.02 if f["side"] == "UP" else f["up_imb"] <= -0.02,
        "imbalance agrees with side (UP imb>0.02, DOWN imb<-0.02)")

    # Combine the strongest filters
    evaluate_rule(fills,
        lambda f: (not (f["side"] == "UP" and 0.36 <= f["limit"] < 0.46)) and
                  (not (51.0 <= f["delta_raw"] <= 86.0)),
        "skip UP-low-price + skip danger-delta")
    evaluate_rule(fills,
        lambda f: (not (f["side"] == "UP" and 0.36 <= f["limit"] < 0.46)) and
                  (not (51.0 <= f["delta_raw"] <= 86.0)) and
                  f["edge"] >= 0.05,
        "skip UP-low-price + skip danger-delta + edge>=0.05")

    # DOWN-side wins big at low prices (table showed +$6-8 in each bucket from 0.30 to 0.65)
    # UP-side wins big at mid-high prices [0.58, 0.80]
    evaluate_rule(fills,
        lambda f: ((f["side"] == "DOWN") or
                   (f["side"] == "UP" and f["limit"] >= 0.50)),
        "DOWN: any | UP: limit>=0.50")
    evaluate_rule(fills,
        lambda f: ((f["side"] == "DOWN") or
                   (f["side"] == "UP" and f["limit"] >= 0.50)) and
                  (not (51.0 <= f["delta_raw"] <= 86.0)),
        "DOWN: any | UP: limit>=0.50 | skip danger-delta")
    evaluate_rule(fills,
        lambda f: ((f["side"] == "DOWN") or
                   (f["side"] == "UP" and f["limit"] >= 0.50)) and
                  (not (51.0 <= f["delta_raw"] <= 86.0)) and
                  f["edge"] >= 0.05,
        "DOWN any | UP limit>=0.50 | skip danger-delta | edge>=0.05")

    # Original (legacy) rules for comparison
    evaluate_rule(fills, lambda f: f["edge"] >= 0.10, "edge >= 0.10")
    evaluate_rule(fills, lambda f: f["edge"] >= 0.15, "edge >= 0.15")
    # Rule 1: minimum edge of 0.10
    evaluate_rule(fills, lambda f: f["edge"] >= 0.10, "edge >= 0.10")
    # Rule 2: minimum edge of 0.15
    evaluate_rule(fills, lambda f: f["edge"] >= 0.15, "edge >= 0.15")
    # Rule 3: tighter entry-price band (avoid extremes)
    evaluate_rule(fills, lambda f: 0.35 <= f["limit"] <= 0.75, "limit in [0.35, 0.75]")
    # Rule 4: only last 30s of the market
    evaluate_rule(fills, lambda f: f["ttc"] <= 30.0, "ttc <= 30s")
    # Rule 5: skip when momentum disagrees with our side (sign of BTC return matches our bet)
    evaluate_rule(fills,
        lambda f: (f["side"] == "UP"   and f["btc_ret_5s"] >= -0.0001) or
                  (f["side"] == "DOWN" and f["btc_ret_5s"] <= +0.0001),
        "momentum agrees with side")
    # Rule 6: combined edge + price + momentum
    evaluate_rule(fills,
        lambda f: f["edge"] >= 0.10 and 0.35 <= f["limit"] <= 0.75 and
                  ((f["side"] == "UP"   and f["btc_ret_5s"] >= -0.0001) or
                   (f["side"] == "DOWN" and f["btc_ret_5s"] <= +0.0001)),
        "edge>=0.10 + price-band + momentum")
    # Rule 7: aggressive -- only HIGH edge and last 30s
    evaluate_rule(fills,
        lambda f: f["edge"] >= 0.15 and f["ttc"] <= 30.0,
        "edge>=0.15 AND ttc<=30s")
    # Rule 8: ultra-tight -- best of all
    evaluate_rule(fills,
        lambda f: f["edge"] >= 0.12 and 0.35 <= f["limit"] <= 0.75 and
                  f["ttc"] <= 35.0 and
                  ((f["side"] == "UP"   and f["btc_ret_5s"] >= -0.0001) or
                   (f["side"] == "DOWN" and f["btc_ret_5s"] <= +0.0001)),
        "edge>=0.12 + price-band + ttc<=35 + momentum")

    # Per-day check for the best refined rule
    print(f"\n=== per-day for best refined rule: DOWN any | UP limit>=0.50 | skip danger-delta | edge>=0.05 ===")
    best = [f for f in fills if
            ((f["side"] == "DOWN") or
             (f["side"] == "UP" and f["limit"] >= 0.50)) and
            (not (51.0 <= f["delta_raw"] <= 86.0)) and
            f["edge"] >= 0.05]
    by_day = defaultdict(list)
    for f in best:
        by_day[f["date"]].append(f)
    for d in TEST_DAYS:
        sub = by_day.get(d, [])
        if not sub:
            print(f"  {d}: 0 fills"); continue
        wins = sum(f["won"] for f in sub)
        s_pnls = np.array([f["pnl"] for f in sub])
        ups = sum(1 for f in sub if f["side"]=="UP"); dns = sum(1 for f in sub if f["side"]=="DOWN")
        print(f"  {d}: n={len(sub):>3}  win={wins/len(sub):>5.1%}  "
              f"total=${s_pnls.sum():>+7.2f}  avg=${s_pnls.mean():>+.4f}  (UP={ups} DN={dns})")


if __name__ == "__main__":
    main()
