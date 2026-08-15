"""Train a fresh LightGBM on the no-bias dataset with:
  - ~22 features, none requiring synthetic_corrected / rolling_bias.
  - Isotonic calibrator fit on a held-out calibration split,
    SAVED INSIDE the model as model._calibrator.
  - Maker backtest on a final test split, allowing BOTH up-side and
    down-side entries, sweeping thresholds to find the best config.

Data splits:
  TRAIN:  2026-05-04 .. 2026-05-15  (full healthy days)
  CALIB:  2026-05-16 .. 2026-05-20
  TEST:   2026-05-21 .. 2026-05-29

Model lives at:
  artifacts_cleaned/model_no_bias_v1/model.pkl
  artifacts_cleaned/model_no_bias_v1/feature_importance.json
  artifacts_cleaned/model_no_bias_v1/experiment_manifest.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from feature_cleanup import clean_features  # noqa: E402

DC = REPO / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close_no_bias"
OUT_DIR = REPO / "artifacts_cleaned" / "model_no_bias_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DAYS = [
    "2026-05-04","2026-05-05","2026-05-06","2026-05-07","2026-05-08",
    "2026-05-09","2026-05-10","2026-05-11","2026-05-12","2026-05-13",
    "2026-05-14","2026-05-15",
]
CALIB_DAYS = [
    "2026-05-16","2026-05-17","2026-05-18","2026-05-19","2026-05-20",
]
TEST_DAYS = [
    "2026-05-21","2026-05-22","2026-05-26","2026-05-29",
]

# ── Feature list ────────────────────────────────────────────────────────────
# Selected from the top-30 importance ranking of the OLD model, with every
# bias-dependent feature replaced by its _raw counterpart, every calendar
# feature dropped (overfitting traps on small sample), and a few new
# additions (multiple return horizons).
FEATURES = [
    # Time within market
    "t_to_close_s",
    # Raw delta-to-strike (replaces synthetic_corrected-derived versions)
    "delta_to_strike_raw",
    "abs_delta_to_strike_raw",
    "delta_sign_raw",
    "delta_to_strike_over_vol_raw",
    # Raw BTC dynamics
    "btc_return_1s_raw",
    "btc_return_3s_raw",
    "btc_return_5s_raw",
    "btc_return_10s_raw",
    "btc_realized_vol_5s_raw",
    "btc_realized_vol_15s_raw",
    # Time-spent (raw)
    "time_spent_above_strike_recent_s_raw",
    "time_spent_below_strike_recent_s_raw",
    # Raw BTC sources / cross-source signals
    "binance_spot_mid",
    "binance_usdm_mid",
    "hyperliquid_mid",
    "cross_source_spread_usd",
    "spot_perp_divergence_usd",
    # Polymarket book (raw, no bias dependence)
    "up_token_best_bid",
    "up_token_best_ask",
    "up_token_microprice",
    "down_token_microprice",
    "up_token_imbalance",
    "down_token_imbalance",
    "up_token_bid_depth_total",
    "up_token_ask_depth_total",
    "down_token_bid_depth_total",
    "down_token_ask_depth_total",
]

TTC_MIN_S, TTC_MAX_S = 10.0, 60.0


def load_split(dates: list[str]) -> pl.DataFrame:
    parts = []
    for d in dates:
        p = DC / f"{d}.parquet"
        if not p.exists():
            print(f"  skip {d}: parquet missing")
            continue
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


def build_X(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    cols = set(df.columns)
    missing = [f for f in feats if f not in cols]
    if missing:
        print(f"  WARNING missing features: {missing}")
    for i, f in enumerate(feats):
        if f not in cols:
            continue
        s = df.get_column(f)
        if not s.dtype.is_numeric():
            continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    return X


def maker_sim_both_sides(df: pl.DataFrame, p_up: np.ndarray, *,
                         edge_threshold: float, min_price: float = 0.30,
                         max_pos_per_market_per_side: int = 2,
                         min_gap_s: float = 10.0, size: float = 1.0):
    """Maker simulation with BOTH UP and DOWN entries.
       For each snapshot, pick whichever side has the larger positive edge.
       Maker price = midpoint of that side's book.
       Fill check: future best-ask of the chosen side drops to <= limit
                   before market close.
    """
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    up_ask = df["up_token_best_ask"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)  # 1=UP won, 0=DOWN won
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()
    dates = df["date"].to_numpy()

    # edges
    edge_up = p_up - up_ask        # buy UP at ask -- but we're a maker, so this is signal-only
    edge_dn = up_bid - p_up        # buy DOWN side conviction

    # Per-market future ask lookups, one per side
    fut_up_ask = defaultdict(list)
    fut_dn_ask = defaultdict(list)
    for i in range(len(df)):
        fut_up_ask[str(slugs[i])].append((int(snap_ts[i]), float(up_ask[i])))
        fut_dn_ask[str(slugs[i])].append((int(snap_ts[i]), float(dn_ask[i])))

    pos_per_mkt_side: dict[tuple[str, str], int] = defaultdict(int)
    last_entry_ns: dict[tuple[str, str], int] = {}
    gap_ns = int(min_gap_s * 1e9)

    n_sig = n_fill = n_win = 0
    n_sig_up = n_fill_up = n_win_up = 0
    n_sig_dn = n_fill_dn = n_win_dn = 0
    pnl = 0.0
    pnl_up = 0.0; pnl_dn = 0.0
    pnl_by_day = defaultdict(float)
    fills_by_day = defaultdict(int)

    for i in range(len(df)):
        # Pick the side with bigger positive edge
        eu = edge_up[i]; ed = edge_dn[i]
        if eu < edge_threshold and ed < edge_threshold:
            continue
        side = "UP" if eu >= ed else "DOWN"
        if side == "UP" and eu < edge_threshold: continue
        if side == "DOWN" and ed < edge_threshold: continue

        bid = up_bid[i] if side == "UP" else dn_bid[i]
        ask = up_ask[i] if side == "UP" else dn_ask[i]
        if not (min_price <= ask < 1.0): continue
        if not (0 < bid < ask): continue
        slug = str(slugs[i])
        key = (slug, side)
        if pos_per_mkt_side[key] >= max_pos_per_market_per_side: continue
        last = last_entry_ns.get(key)
        if last is not None and (int(snap_ts[i]) - last) < gap_ns: continue

        limit = round((bid + ask) / 2.0, 2)
        if limit < min_price: continue
        if not (0.02 <= limit <= 0.98): continue

        n_sig += 1
        if side == "UP": n_sig_up += 1
        else: n_sig_dn += 1
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
        n_fill += 1
        # Win iff outcome matches the side we bought
        won = (resolved[i] == 1) if side == "UP" else (resolved[i] == 0)
        pnl_i = (1.0 if won else 0.0) - limit
        pnl += pnl_i * size
        date = str(dates[i])
        pnl_by_day[date] += pnl_i * size
        fills_by_day[date] += 1
        if side == "UP":
            n_fill_up += 1; pnl_up += pnl_i * size
            if won: n_win_up += 1
        else:
            n_fill_dn += 1; pnl_dn += pnl_i * size
            if won: n_win_dn += 1
        if won: n_win += 1

    return {
        "edge_threshold": edge_threshold,
        "n_sig": n_sig, "n_fill": n_fill, "n_win": n_win,
        "fill_rate": n_fill / max(1, n_sig),
        "win_rate": n_win / max(1, n_fill),
        "pnl": pnl,
        "pnl_per_fill": pnl / max(1, n_fill),
        "up_side": {"n_sig": n_sig_up, "n_fill": n_fill_up, "n_win": n_win_up,
                    "win_rate": n_win_up / max(1, n_fill_up), "pnl": pnl_up},
        "dn_side": {"n_sig": n_sig_dn, "n_fill": n_fill_dn, "n_win": n_win_dn,
                    "win_rate": n_win_dn / max(1, n_fill_dn), "pnl": pnl_dn},
        "pnl_by_day": dict(pnl_by_day),
        "fills_by_day": dict(fills_by_day),
    }


def main():
    print("=" * 70)
    print(f"FEATURES: {len(FEATURES)}")
    for f in FEATURES: print(f"  - {f}")

    # ── Load splits ───────────────────────────────────────────────────────
    print(f"\nloading TRAIN: {TRAIN_DAYS}")
    train_df = load_split(TRAIN_DAYS)
    print(f"  TRAIN: {len(train_df)} rows, {train_df['market_slug'].n_unique()} markets")
    print(f"loading CALIB: {CALIB_DAYS}")
    calib_df = load_split(CALIB_DAYS)
    print(f"  CALIB: {len(calib_df)} rows, {calib_df['market_slug'].n_unique()} markets")
    print(f"loading TEST: {TEST_DAYS}")
    test_df = load_split(TEST_DAYS)
    print(f"  TEST:  {len(test_df)} rows, {test_df['market_slug'].n_unique()} markets")

    # Clean features (applies the same transforms the runtime uses)
    print("\ncleaning features...")
    train_df_clean = clean_features(train_df)
    calib_df_clean = clean_features(calib_df)
    test_df_clean  = clean_features(test_df)

    X_train = build_X(train_df_clean, FEATURES)
    y_train = train_df["resolved_side_label"].to_numpy().astype(int)
    X_calib = build_X(calib_df_clean, FEATURES)
    y_calib = calib_df["resolved_side_label"].to_numpy().astype(int)
    X_test  = build_X(test_df_clean,  FEATURES)
    y_test  = test_df["resolved_side_label"].to_numpy().astype(int)

    print(f"\n  train shape: {X_train.shape}  UP rate: {y_train.mean():.3f}")
    print(f"  calib shape: {X_calib.shape}  UP rate: {y_calib.mean():.3f}")
    print(f"  test  shape: {X_test.shape}  UP rate: {y_test.mean():.3f}")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\ntraining LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=200,
        objective="binary",
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_calib, y_calib)],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(20, verbose=False)])

    # ── Fit calibrator on CALIB ──────────────────────────────────────────
    print("\nfitting isotonic calibrator on CALIB split...")
    raw_p_calib = model.predict_proba(X_calib)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    cal.fit(raw_p_calib, y_calib)
    cal_p_calib = cal.predict(raw_p_calib)
    print(f"  raw   p_up mean on CALIB: {raw_p_calib.mean():.3f}  (target {y_calib.mean():.3f})")
    print(f"  calib p_up mean on CALIB: {cal_p_calib.mean():.3f}  (target {y_calib.mean():.3f})")

    # Attach + save
    model._calibrator = cal
    joblib.dump(model, OUT_DIR / "model.pkl")
    fi = {FEATURES[i]: float(v) for i, v in enumerate(model.feature_importances_)}
    fi_sorted = dict(sorted(fi.items(), key=lambda kv: -kv[1]))
    (OUT_DIR / "feature_importance.json").write_text(json.dumps(fi_sorted, indent=2))
    manifest = {
        "model_id": "model_no_bias_v1",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "train_days": TRAIN_DAYS,
        "calib_days": CALIB_DAYS,
        "test_days": TEST_DAYS,
        "train_rows": int(len(X_train)),
        "calib_rows": int(len(X_calib)),
        "test_rows":  int(len(X_test)),
        "hyperparams": {
            "n_estimators": int(model.n_estimators_),
            "max_depth": 6, "num_leaves": 31, "learning_rate": 0.05,
        },
    }
    (OUT_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  saved to {OUT_DIR}")

    # ── OOS evaluation ────────────────────────────────────────────────────
    print(f"\nevaluating on TEST split ({len(X_test)} rows)...")
    raw_p_test = model.predict_proba(X_test)[:, 1]
    cal_p_test = cal.predict(raw_p_test)
    print(f"  raw   p_up mean on TEST: {raw_p_test.mean():.3f}  (actual UP {y_test.mean():.3f})")
    print(f"  calib p_up mean on TEST: {cal_p_test.mean():.3f}  (actual UP {y_test.mean():.3f})")

    # ── Maker backtest, both sides, threshold sweep ──────────────────────
    print(f"\nMaker backtest (both sides, midpoint, min_price=0.30):")
    print(f"{'thr':>5} {'sig':>5} {'fill':>5} {'fill%':>5} {'win%':>5} {'PnL':>8} {'$/fill':>7}"
          f" | {'UP_n':>4} {'UP_fill':>7} {'UP_win%':>7} {'UP_pnl':>7}"
          f" | {'DN_n':>4} {'DN_fill':>7} {'DN_win%':>7} {'DN_pnl':>7}")
    print("-" * 130)
    results = []
    for thr in [0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]:
        r = maker_sim_both_sides(test_df, cal_p_test, edge_threshold=thr)
        results.append(r)
        u, d = r["up_side"], r["dn_side"]
        print(f"{thr:>5.2f} {r['n_sig']:>5d} {r['n_fill']:>5d} {r['fill_rate']:>5.1%}"
              f" {r['win_rate']:>5.1%} {r['pnl']:>+8.2f} {r['pnl_per_fill']:>+7.4f}"
              f" | {u['n_sig']:>4d} {u['n_fill']:>7d} {u['win_rate']:>6.1%} {u['pnl']:>+7.2f}"
              f" | {d['n_sig']:>4d} {d['n_fill']:>7d} {d['win_rate']:>6.1%} {d['pnl']:>+7.2f}")

    # Pick best by total PnL with >= 20 fills
    creditable = [r for r in results if r["n_fill"] >= 20]
    if creditable:
        best = max(creditable, key=lambda r: r["pnl"])
        print(f"\nBEST threshold: {best['edge_threshold']:.2f}")
        print(f"  fills/day = {best['n_fill']/len(TEST_DAYS):.1f}   "
              f"daily PnL = ${best['pnl']/len(TEST_DAYS):.2f}   "
              f"win = {best['win_rate']:.1%}")
        print(f"  by day:")
        for d in TEST_DAYS:
            f_ = best["fills_by_day"].get(d, 0)
            p_ = best["pnl_by_day"].get(d, 0.0)
            print(f"    {d}: {f_:>4d} fills, ${p_:>+7.2f}")

    print("\nFeature importance (top 15):")
    for i, (k, v) in enumerate(list(fi_sorted.items())[:15], 1):
        print(f"  {i:>2}. {k:<42} imp={v:.1f}")


if __name__ == "__main__":
    main()
