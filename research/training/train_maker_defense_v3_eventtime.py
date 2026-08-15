"""Maker-Defense v3: Event-time microstructure features from event64 dataset.

Replaces 1Hz aggregates with sub-millisecond L2 event sequences.  For each
anchor we have the last 64 L2 events (book_state, top_of_book, trade) with
full microstructure: best_bid/ask, mid, microprice, depth, imbalance, trade
size & side, event timestamps.

Engineered features (~40 per row):

  Microprice dynamics (the directional signal):
    - micro_first, micro_last (deltas vs mid)
    - micro_change_total       (last - first)
    - micro_change_q1..q4      (per-quartile micro changes within window)
    - micro_realized_vol       (std of micro diffs)
    - micro_max_excursion      (max abs deviation from first)

  Imbalance trajectory:
    - imb_first, imb_last
    - imb_mean, imb_std, imb_min, imb_max
    - imb_change_total

  Depth (order book stress):
    - bid_depth_first / last / mean / min
    - ask_depth_first / last / mean / min
    - depth_asymmetry_last (ask-bid)/(ask+bid)
    - depth_collapse (min/first ratio for each side)

  Trade flow / toxicity:
    - n_trades_in_window
    - trade_to_event_ratio
    - signed_trade_size_sum (buyer-maker is sell flow, etc.)
    - abs_trade_size_sum
    - large_trade_indicator (any single trade > threshold)

  Event timing / burstiness:
    - mean_dt_between_events
    - max_dt_gap (longest pause)
    - n_events_in_last_1s, last_500ms, last_100ms
    - event_acceleration (recent rate / overall rate)

  Spread dynamics:
    - spread_mean, spread_max, spread_volatility
    - spread_widening (last - first)

Target: maker-defense extreme_10c_5s = |mid_move_5s| >= 0.10
"""
import sys, json, os
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import polars as pl
import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from model_factory.trainers.logistic_regression_trainer import _to_float_array


DATASET_PATH = Path("data/datasets/microstructure_sequence_dataset_v1_event64")
TRAIN_DATES = [f"2026-04-{d:02d}" for d in range(19, 31)] + [f"2026-05-{d:02d}" for d in range(1, 5)]
VAL_DATES   = ["2026-05-05"]
TEST_DATES  = ["2026-05-06"]

# Targets
TARGETS = [
    {"name": "extreme_10c_5s",  "move_col": "mid_move_5s",  "threshold": 0.10},
    {"name": "severe_7c_3s",    "move_col": "mid_move_3s",  "threshold": 0.07},
    {"name": "moderate_5c_5s",  "move_col": "mid_move_5s",  "threshold": 0.05},
]

# Aggregate features from the 1Hz layer (kept as baseline reference)
AGG_FEATURES = [
    "anchor_source",
    "event_count_1s", "event_count_3s", "event_count_5s", "event_count_15s",
    "spread_mean_5s", "spread_min_5s", "spread_max_5s",
    "imbalance_mean_5s", "imbalance_last", "imbalance_slope_5s",
    "mid_return_1s", "mid_return_3s", "mid_return_5s",
    "quote_churn_rate_5s", "depth_change_rate_5s",
    "one_sided_book", "crossed_book", "stale_book",
    "sequence_completeness_rate",
    "price_to_beat", "live_btc_usd",
    "delta_to_strike", "abs_delta_to_strike", "delta_sign",
    "t_since_open_s", "t_to_close_s", "phase_bucket",
    "live_reference_trusted", "live_bias_mode", "live_applied_bias_age_seconds",
    "sequence_length_actual",
]


def build_event_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute event-time features from the 64-event lists.

    All operations use polars list-eval expressions for speed.  Output adds
    ~40 new columns to the frame.
    """
    # Helper: per-quartile slicing for the microprice trajectory
    # quartiles: 0:16, 16:32, 32:48, 48:64

    df = df.with_columns([
        # === MICROPRICE DYNAMICS ===
        pl.col("events_microprice").list.first().alias("evt_micro_first"),
        pl.col("events_microprice").list.last().alias("evt_micro_last"),
        pl.col("events_microprice").list.mean().alias("evt_micro_mean"),
        pl.col("events_microprice").list.std().alias("evt_micro_std"),
        pl.col("events_microprice").list.min().alias("evt_micro_min"),
        pl.col("events_microprice").list.max().alias("evt_micro_max"),

        # Quartile slices for microprice
        pl.col("events_microprice").list.slice(0, 16).list.mean().alias("evt_micro_q1_mean"),
        pl.col("events_microprice").list.slice(16, 16).list.mean().alias("evt_micro_q2_mean"),
        pl.col("events_microprice").list.slice(32, 16).list.mean().alias("evt_micro_q3_mean"),
        pl.col("events_microprice").list.slice(48, 16).list.mean().alias("evt_micro_q4_mean"),

        # Realized vol of microprice diffs (within the window)
        pl.col("events_microprice").list.eval(pl.element().diff().abs()).list.mean().alias("evt_micro_mean_abs_diff"),
        pl.col("events_microprice").list.eval(pl.element().diff().abs()).list.max().alias("evt_micro_max_abs_diff"),
        pl.col("events_microprice").list.eval(pl.element().diff()).list.std().alias("evt_micro_diff_std"),

        # === MID DYNAMICS (discrete tick steps) ===
        pl.col("events_mid").list.first().alias("evt_mid_first"),
        pl.col("events_mid").list.last().alias("evt_mid_last"),
        pl.col("events_mid").list.eval(pl.element().diff().abs()).list.sum().alias("evt_mid_total_move"),
        pl.col("events_mid").list.n_unique().alias("evt_mid_n_unique_levels"),

        # === IMBALANCE TRAJECTORY ===
        pl.col("events_imbalance").list.first().alias("evt_imb_first"),
        pl.col("events_imbalance").list.last().alias("evt_imb_last"),
        pl.col("events_imbalance").list.mean().alias("evt_imb_mean"),
        pl.col("events_imbalance").list.std().alias("evt_imb_std"),
        pl.col("events_imbalance").list.min().alias("evt_imb_min"),
        pl.col("events_imbalance").list.max().alias("evt_imb_max"),
        pl.col("events_imbalance").list.slice(48, 16).list.mean().alias("evt_imb_q4_mean"),  # most recent quartile

        # === DEPTH (ORDER BOOK STRESS) ===
        pl.col("events_bid_depth_total").list.first().alias("evt_bid_depth_first"),
        pl.col("events_bid_depth_total").list.last().alias("evt_bid_depth_last"),
        pl.col("events_bid_depth_total").list.mean().alias("evt_bid_depth_mean"),
        pl.col("events_bid_depth_total").list.min().alias("evt_bid_depth_min"),
        pl.col("events_ask_depth_total").list.first().alias("evt_ask_depth_first"),
        pl.col("events_ask_depth_total").list.last().alias("evt_ask_depth_last"),
        pl.col("events_ask_depth_total").list.mean().alias("evt_ask_depth_mean"),
        pl.col("events_ask_depth_total").list.min().alias("evt_ask_depth_min"),

        # === TRADE FLOW ===
        pl.col("events_trade_size").list.sum().alias("evt_trade_size_sum"),
        pl.col("events_trade_size").list.max().alias("evt_trade_size_max"),
        # Count trades = events with nonzero trade_size
        pl.col("events_trade_size").list.eval((pl.element() > 0).cast(pl.Int8)).list.sum().alias("evt_n_trades"),

        # === SPREAD DYNAMICS ===
        pl.col("events_spread").list.first().alias("evt_spread_first"),
        pl.col("events_spread").list.last().alias("evt_spread_last"),
        pl.col("events_spread").list.mean().alias("evt_spread_mean"),
        pl.col("events_spread").list.max().alias("evt_spread_max"),
        pl.col("events_spread").list.std().alias("evt_spread_std"),

        # === EVENT TIMING / BURSTINESS ===
        pl.col("events_dt_from_prev_s").list.mean().alias("evt_dt_mean"),
        pl.col("events_dt_from_prev_s").list.max().alias("evt_dt_max"),
        pl.col("events_dt_from_prev_s").list.std().alias("evt_dt_std"),
        # Window span: -dt_from_anchor[0] = how far back does the 64-event window reach
        pl.col("events_dt_from_anchor_s").list.first().alias("evt_window_span_s"),
        pl.col("events_dt_from_anchor_s").list.last().alias("evt_last_event_dt_s"),

        # Burst: events in last ~1s relative to anchor
        pl.col("events_dt_from_anchor_s")
            .list.eval((pl.element() >= -1.0).cast(pl.Int8)).list.sum().alias("evt_n_events_last_1s"),
        pl.col("events_dt_from_anchor_s")
            .list.eval((pl.element() >= -0.5).cast(pl.Int8)).list.sum().alias("evt_n_events_last_500ms"),
        pl.col("events_dt_from_anchor_s")
            .list.eval((pl.element() >= -0.1).cast(pl.Int8)).list.sum().alias("evt_n_events_last_100ms"),
    ])

    # === Derived features (need columns above to exist) ===
    df = df.with_columns([
        # Microprice change features
        (pl.col("evt_micro_last") - pl.col("evt_micro_first")).alias("evt_micro_change_total"),
        (pl.col("evt_micro_q2_mean") - pl.col("evt_micro_q1_mean")).alias("evt_micro_change_q1_q2"),
        (pl.col("evt_micro_q3_mean") - pl.col("evt_micro_q2_mean")).alias("evt_micro_change_q2_q3"),
        (pl.col("evt_micro_q4_mean") - pl.col("evt_micro_q3_mean")).alias("evt_micro_change_q3_q4"),
        (pl.col("evt_micro_max") - pl.col("evt_micro_min")).alias("evt_micro_range"),

        # Imbalance change
        (pl.col("evt_imb_last") - pl.col("evt_imb_first")).alias("evt_imb_change_total"),
        (pl.col("evt_imb_q4_mean") - pl.col("evt_imb_mean")).alias("evt_imb_recent_vs_mean"),

        # Depth ratios (avoid div by zero)
        (pl.col("evt_bid_depth_last") / pl.max_horizontal([pl.col("evt_bid_depth_first"), pl.lit(1e-6)]))
            .alias("evt_bid_depth_ratio"),
        (pl.col("evt_ask_depth_last") / pl.max_horizontal([pl.col("evt_ask_depth_first"), pl.lit(1e-6)]))
            .alias("evt_ask_depth_ratio"),
        ((pl.col("evt_ask_depth_last") - pl.col("evt_bid_depth_last"))
         / pl.max_horizontal([pl.col("evt_ask_depth_last") + pl.col("evt_bid_depth_last"), pl.lit(1e-6)]))
            .alias("evt_depth_asymmetry_last"),

        # Trade rate
        (pl.col("evt_n_trades").cast(pl.Float64) / pl.col("sequence_length_actual").cast(pl.Float64))
            .alias("evt_trade_rate"),

        # Spread widening
        (pl.col("evt_spread_last") - pl.col("evt_spread_first")).alias("evt_spread_widening"),

        # Acceleration: recent burst rate vs overall rate
        (pl.col("evt_n_events_last_1s").cast(pl.Float64)
            / pl.max_horizontal([pl.col("sequence_length_actual").cast(pl.Float64) / 10.0, pl.lit(1.0)]))
            .alias("evt_acceleration_1s"),
    ])

    return df


def get_event_feature_names() -> list[str]:
    return [
        # Microprice
        "evt_micro_first","evt_micro_last","evt_micro_mean","evt_micro_std",
        "evt_micro_min","evt_micro_max",
        "evt_micro_q1_mean","evt_micro_q2_mean","evt_micro_q3_mean","evt_micro_q4_mean",
        "evt_micro_mean_abs_diff","evt_micro_max_abs_diff","evt_micro_diff_std",
        "evt_micro_change_total","evt_micro_change_q1_q2","evt_micro_change_q2_q3",
        "evt_micro_change_q3_q4","evt_micro_range",
        # Mid
        "evt_mid_first","evt_mid_last","evt_mid_total_move","evt_mid_n_unique_levels",
        # Imbalance
        "evt_imb_first","evt_imb_last","evt_imb_mean","evt_imb_std",
        "evt_imb_min","evt_imb_max","evt_imb_q4_mean",
        "evt_imb_change_total","evt_imb_recent_vs_mean",
        # Depth
        "evt_bid_depth_first","evt_bid_depth_last","evt_bid_depth_mean","evt_bid_depth_min",
        "evt_ask_depth_first","evt_ask_depth_last","evt_ask_depth_mean","evt_ask_depth_min",
        "evt_bid_depth_ratio","evt_ask_depth_ratio","evt_depth_asymmetry_last",
        # Trade flow
        "evt_trade_size_sum","evt_trade_size_max","evt_n_trades","evt_trade_rate",
        # Spread
        "evt_spread_first","evt_spread_last","evt_spread_mean","evt_spread_max",
        "evt_spread_std","evt_spread_widening",
        # Timing
        "evt_dt_mean","evt_dt_max","evt_dt_std",
        "evt_window_span_s","evt_last_event_dt_s",
        "evt_n_events_last_1s","evt_n_events_last_500ms","evt_n_events_last_100ms",
        "evt_acceleration_1s",
    ]


EVENT_FEATURES = get_event_feature_names()
ALL_FEATURES = AGG_FEATURES + EVENT_FEATURES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(dates, split_name, min_seq_len=32):
    """Memory-efficient: process each day individually, drop event lists ASAP."""
    print(f"  [{split_name}] Loading {len(dates)} day(s) from event64...", flush=True)
    # Columns we need from the parquet (lists + aggregates + meta + targets)
    list_cols = [
        "events_recv_ts_ns", "events_dt_from_anchor_s", "events_dt_from_prev_s",
        "events_event_type", "events_best_bid", "events_best_ask", "events_mid",
        "events_spread", "events_microprice", "events_bid_depth_total",
        "events_ask_depth_total", "events_imbalance", "events_trade_size",
    ]
    target_cols = ["mid_move_1s", "mid_move_3s", "mid_move_5s"]
    meta_cols = ["sequence_feature_eligible", "sequence_length_actual"]
    load_cols = list(dict.fromkeys(list_cols + AGG_FEATURES + target_cols + meta_cols))

    daily_frames = []
    for d in dates:
        f = DATASET_PATH / f"{d}.parquet"
        if not f.exists():
            print(f"    SKIP missing: {d}")
            continue
        schema = pl.scan_parquet(str(f)).collect_schema()
        avail = set(schema.names())
        cols_here = [c for c in load_cols if c in avail]
        day = pl.read_parquet(str(f), columns=cols_here)

        # Filter eligibility
        if "sequence_feature_eligible" in day.columns:
            day = day.filter(pl.col("sequence_feature_eligible"))
        day = day.filter(pl.col("sequence_length_actual") >= min_seq_len)
        day = day.filter(
            pl.col("mid_move_5s").is_not_null() & pl.col("mid_move_5s").is_not_nan()
        )

        # Build event features
        day = build_event_features(day)

        # DROP list columns to free memory
        day = day.drop([c for c in list_cols if c in day.columns])

        # Targets
        for t in TARGETS:
            day = day.with_columns(
                (pl.col(t["move_col"]).abs() >= t["threshold"]).cast(pl.Int8).alias(f"target_{t['name']}")
            )

        daily_frames.append(day)
        print(f"    [{d}] processed {day.height:,} rows (cols={day.width})")

    df = pl.concat(daily_frames, how="vertical_relaxed")
    print(f"  [{split_name}] total: {df.height:,} rows  x  {df.width} cols")
    for t in TARGETS:
        pos = float(df[f"target_{t['name']}"].mean())
        print(f"  [{split_name}] target {t['name']}: positive_rate={pos:.4%}")
    return df


# ---------------------------------------------------------------------------
# Train one target
# ---------------------------------------------------------------------------

def fpr_at_recall(y_true, y_score, target_recalls=(0.80, 0.90, 0.95, 0.98, 0.99)):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    out = []
    for target in target_recalls:
        idx = np.searchsorted(tpr, target, side="left")
        idx = max(0, min(idx, len(thresholds) - 1))
        out.append({
            "target_recall": target,
            "achieved_recall": float(tpr[idx]),
            "fpr": float(fpr[idx]),
            "threshold": float(thresholds[idx]),
        })
    return out


def train_target(target_cfg, train_df, val_df, test_df, feature_set, label):
    name = target_cfg["name"]
    print(f"\n{'='*72}")
    print(f"  TRAINING [{label}]: {name}  |  features={len(feature_set)}")
    print(f"{'='*72}")

    X_tr, _ = _to_float_array(train_df.select(feature_set))
    y_tr = train_df[f"target_{name}"].to_numpy().astype(int)
    X_va, _ = _to_float_array(val_df.select(feature_set))
    y_va = val_df[f"target_{name}"].to_numpy().astype(int)
    X_te, _ = _to_float_array(test_df.select(feature_set))
    y_te = test_df[f"target_{name}"].to_numpy().astype(int)

    print(f"  pos: train={y_tr.mean():.4%}  val={y_va.mean():.4%}  test={y_te.mean():.4%}")

    model = lgb.LGBMClassifier(
        objective="binary", metric="auc",
        num_leaves=63, min_child_samples=200, learning_rate=0.05,
        n_estimators=1500, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    print(f"  best_iteration: {model.best_iteration_}")

    proba_te = model.predict_proba(X_te)[:, 1]
    auc_t = roc_auc_score(y_te, proba_te)
    ap_t  = average_precision_score(y_te, proba_te)
    base  = float(y_te.mean())

    print(f"  [TEST] AUC={auc_t:.4f}  AvgPrec={ap_t:.4f}  lift={ap_t/base:.2f}x")

    fpr_tbl = fpr_at_recall(y_te, proba_te)
    print(f"\n  {'Recall':>9s}  {'FPR':>8s}  {'Threshold':>10s}  {'%selected':>10s}")
    for row in fpr_tbl:
        sel_pct = float((proba_te >= row["threshold"]).mean())
        print(f"  {row['target_recall']:>9.2%}  {row['fpr']:>8.4f}  "
              f"{row['threshold']:>10.5f}  {sel_pct:>10.2%}")

    return {
        "label": label,
        "target": name,
        "auc_test": float(auc_t), "ap_test": float(ap_t),
        "pos_rate_test": base,
        "lift_over_random": float(ap_t / base),
        "fpr_at_recall": fpr_tbl,
        "best_iter": int(model.best_iteration_ or 0),
        "feature_importance": dict(sorted(
            zip(feature_set, [float(v) / max(float(model.feature_importances_.sum()), 1.0)
                              for v in model.feature_importances_]),
            key=lambda x: -x[1]
        )),
    }, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "#"*72)
    print("  MAKER-DEFENSE v3: Event-time microstructure features (event64)")
    print("#"*72)

    train_df = load_split(TRAIN_DATES, "train")
    val_df   = load_split(VAL_DATES,   "val")
    test_df  = load_split(TEST_DATES,  "test")

    out_root = Path("artifacts/model_08c_maker_defense_v3_eventtime")
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Three runs per target:
    #   A) aggregates only (baseline, like v1)
    #   B) event-time only
    #   C) aggregates + event-time (full)
    for tgt in TARGETS:
        all_results[tgt["name"]] = {}
        for label, feats in [
            ("agg_only",      AGG_FEATURES),
            ("event_only",    EVENT_FEATURES),
            ("agg_plus_event", ALL_FEATURES),
        ]:
            m, mdl = train_target(tgt, train_df, val_df, test_df, feats, label)
            all_results[tgt["name"]][label] = m

            save_dir = out_root / tgt["name"] / label
            save_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(mdl, save_dir / "model.pkl")
            with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)

    # ----------------------------------------------------------------------
    # Summary comparison
    # ----------------------------------------------------------------------
    print("\n\n" + "#"*72)
    print("  COMPARISON: agg_only  vs  event_only  vs  agg+event")
    print("#"*72)
    for tgt_name, runs in all_results.items():
        print(f"\n  === {tgt_name} ===")
        print(f"  {'Run':<18s}  {'AUC':>6s}  {'AP':>6s}  {'lift':>5s}  "
              f"{'FPR@95':>8s}  {'FPR@99':>8s}")
        print(f"  {'-'*18}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}")
        for label, m in runs.items():
            f95 = next((r["fpr"] for r in m["fpr_at_recall"] if r["target_recall"]==0.95), float("nan"))
            f99 = next((r["fpr"] for r in m["fpr_at_recall"] if r["target_recall"]==0.99), float("nan"))
            print(f"  {label:<18s}  {m['auc_test']:>6.4f}  {m['ap_test']:>6.4f}  "
                  f"{m['lift_over_random']:>5.2f}x  {f95:>8.4f}  {f99:>8.4f}")

    with open(out_root / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== DONE ===")
