#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOS evaluation for model_08b and all model_08c variants on 2026-05-13..16.

Skipped in previous run because they need 6 BTC-derived features not embedded
in the tabular parquet.  This script computes them at runtime from
external_btc_events_v1 and joins by anchor_ts_ns.

08c v3 event_only / agg_plus_event require evt_* statistics from event array
columns — those are skipped here and noted.

Run:  py -3 oos_eval_08bc.py
"""

from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json, joblib
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

sys.path.insert(0, str(Path(".") / "src"))
from model_factory.trainers.logistic_regression_trainer import _to_float_array

# ── Constants ─────────────────────────────────────────────────────────────────

DATES = ["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"]

# Feature list for 08b / 08c-v1 (49 features, from results.json[config][features])
FEAT_49 = json.loads(
    Path("artifacts_cleaned/model_08b_big_move_classifier/tabular_btc/lightgbm/results.json").read_text()
)["config"]["features"]

# BTC-derived features (last 6 in FEAT_49)
BTC_FEATS = ["btc_return_3s", "btc_return_5s", "btc_return_15s",
             "btc_abs_return_5s", "btc_abs_return_15s", "btc_realized_vol_30s"]

assert FEAT_49[-6:] == BTC_FEATS, "BTC features not at expected positions"
TABULAR_FEATS_43 = FEAT_49[:43]   # first 43 come from tabular dataset

# 08c_v2 extends to 58 features: FEAT_49 + 9 extras (in training order from metrics.json)
# abs_mid_return_* are computable; jump/burst/collapse are near-zero importance → zero-filled
EXTRA_9 = [
    "abs_mid_return_1s", "abs_mid_return_3s", "abs_mid_return_5s",
    "past_jump_5s", "past_big_jump_5s",
    "btc_jump_5s", "btc_jump_15s",
    "spread_burst_5s", "depth_collapse_5s",
]
FEAT_58 = FEAT_49 + EXTRA_9

# Feature list for 08c v3 agg_only (32 features, from training script)
AGG_FEATS_32 = [
    "anchor_source", "event_count_1s", "event_count_3s", "event_count_5s", "event_count_15s",
    "spread_mean_5s", "spread_min_5s", "spread_max_5s",
    "imbalance_mean_5s", "imbalance_last", "imbalance_slope_5s",
    "mid_return_1s", "mid_return_3s", "mid_return_5s",
    "quote_churn_rate_5s", "depth_change_rate_5s",
    "one_sided_book", "crossed_book", "stale_book", "sequence_completeness_rate",
    "price_to_beat", "live_btc_usd",
    "delta_to_strike", "abs_delta_to_strike", "delta_sign",
    "t_since_open_s", "t_to_close_s", "phase_bucket",
    "live_reference_trusted", "live_bias_mode", "live_applied_bias_age_seconds",
    "sequence_length_actual",          # only in event64, not tabular
]


# ── BTC feature computation ───────────────────────────────────────────────────

def _load_btc(date: str) -> pl.DataFrame | None:
    """Load BTC mid prices from external_btc_events_v1 (binance_usdm ticker only)."""
    btc_root = Path("data/canonical/external_btc_events_v1")
    matches = sorted(btc_root.glob(f"*{date}*.parquet"))
    if not matches:
        return None
    df = pl.read_parquet(matches[0]).filter(
        (pl.col("venue") == "binance_usdm") & (pl.col("event_type") == "ticker")
    ).select(["recv_ts_ns", "mid_price"]).drop_nulls().sort("recv_ts_ns")
    return df


def _asof_lookup(anchors: pl.Series, btc: pl.DataFrame, offset_ns: int, col_name: str) -> pl.Series:
    """Backward asof-lookup of btc.mid_price at (anchor - offset_ns)."""
    lookup = pl.DataFrame({"anchor_ts_ns": anchors}).with_columns(
        (pl.col("anchor_ts_ns") - offset_ns).alias("lookup_ts")
    ).join_asof(
        btc.rename({"recv_ts_ns": "btc_ts", "mid_price": col_name}),
        left_on="lookup_ts", right_on="btc_ts", strategy="backward",
    ).get_column(col_name)
    return lookup


def add_btc_features(df: pl.DataFrame, date: str) -> pl.DataFrame:
    """Add 6 BTC-derived columns to df (joined on anchor_ts_ns)."""
    btc = _load_btc(date)
    if btc is None or len(btc) == 0:
        for f in BTC_FEATS:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(f))
        return df

    # Unique anchors (many rows share same timestamp — compute once, join back)
    unique_anchors = df.get_column("anchor_ts_ns").unique().sort()

    # Mid at anchor
    mid_at = pl.DataFrame({"anchor_ts_ns": unique_anchors}).join_asof(
        btc.rename({"recv_ts_ns": "btc_ts", "mid_price": "mid_at_anchor"}),
        left_on="anchor_ts_ns", right_on="btc_ts", strategy="backward",
    )

    # Mid at lookbacks
    for lag_s, col in [(3, "mid_3s"), (5, "mid_5s"), (15, "mid_15s")]:
        lag_ns = lag_s * 1_000_000_000
        lagged = pl.DataFrame({"anchor_ts_ns": unique_anchors}).with_columns(
            (pl.col("anchor_ts_ns") - lag_ns).alias("lookup_ts")
        ).join_asof(
            btc.rename({"recv_ts_ns": "btc_ts", "mid_price": col}),
            left_on="lookup_ts", right_on="btc_ts", strategy="backward",
        ).select(["anchor_ts_ns", col])
        mid_at = mid_at.join(lagged, on="anchor_ts_ns", how="left")

    # Log returns
    EPS = 1e-10
    mid_at = mid_at.with_columns([
        ((pl.col("mid_at_anchor") + EPS) / (pl.col("mid_3s")  + EPS)).log().alias("btc_return_3s"),
        ((pl.col("mid_at_anchor") + EPS) / (pl.col("mid_5s")  + EPS)).log().alias("btc_return_5s"),
        ((pl.col("mid_at_anchor") + EPS) / (pl.col("mid_15s") + EPS)).log().alias("btc_return_15s"),
    ]).with_columns([
        pl.col("btc_return_5s").abs().alias("btc_abs_return_5s"),
        pl.col("btc_return_15s").abs().alias("btc_abs_return_15s"),
    ])

    # btc_realized_vol_30s: rolling std of log-returns over 30-second window on BTC series
    WIN_NS = 30_000_000_000
    btc_dt = btc.with_columns(
        pl.col("recv_ts_ns").cast(pl.Datetime("ns")).alias("ts")
    ).with_columns(
        (pl.col("mid_price") / pl.col("mid_price").shift(1)).log().fill_nan(None).alias("logret")
    ).drop_nulls("logret")

    # rolling std within 30 s windows (time-based)
    roll_std = btc_dt.sort("ts").with_columns(
        pl.col("logret")
        .rolling_std(window_size="30s", by="ts", min_periods=2)
        .alias("btc_realized_vol_30s")
    ).select(["recv_ts_ns", "btc_realized_vol_30s"])

    vol_lookup = pl.DataFrame({"anchor_ts_ns": unique_anchors}).join_asof(
        roll_std.rename({"recv_ts_ns": "btc_ts"}),
        left_on="anchor_ts_ns", right_on="btc_ts", strategy="backward",
    ).select(["anchor_ts_ns", "btc_realized_vol_30s"])

    mid_at = mid_at.join(vol_lookup, on="anchor_ts_ns", how="left")
    btc_cols = mid_at.select(["anchor_ts_ns"] + BTC_FEATS)

    return df.join(btc_cols, on="anchor_ts_ns", how="left")


# ── Inference helpers ─────────────────────────────────────────────────────────

def make_X(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    avail = set(df.columns)
    cols = [c for c in features if c in avail]
    X, _ = _to_float_array(df.select(cols))
    return np.nan_to_num(X, nan=0.0)


def predict_clf(model, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)[:, 1]
    cal = getattr(model, "_calibrator", None)
    return np.clip(cal.predict(raw), 1e-6, 1 - 1e-6) if cal else raw


def clf_metrics(y: np.ndarray, p: np.ndarray) -> dict | None:
    y = y.astype(int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return {
        "ll":    float(log_loss(y, p)),
        "auc":   float(roc_auc_score(y, p)),
        "ap":    float(average_precision_score(y, p)),
        "base_rate": float(y.mean()),
        "n":     int(len(y)),
    }


# ── Per-date data loader ──────────────────────────────────────────────────────

_tab_cache:   dict[str, pl.DataFrame] = {}
_e64_cache:   dict[str, pl.DataFrame] = {}


def load_tab(date: str) -> pl.DataFrame:
    if date not in _tab_cache:
        p = Path(f"data/datasets/microstructure_sequence_dataset_v1_tabular/{date}.parquet")
        _tab_cache[date] = pl.read_parquet(p) if p.exists() else pl.DataFrame()
    return _tab_cache[date]


def load_e64(date: str) -> pl.DataFrame:
    if date not in _e64_cache:
        p = Path(f"data/datasets/microstructure_sequence_dataset_v1_event64/{date}.parquet")
        # Only load agg columns (non-list) to keep memory low
        schema = pl.scan_parquet(p).collect_schema()
        scalar_cols = [n for n, t in schema.items() if not isinstance(t, pl.List)]
        _e64_cache[date] = pl.read_parquet(p, columns=scalar_cols) if p.exists() else pl.DataFrame()
    return _e64_cache[date]


# ── Evaluation runners ────────────────────────────────────────────────────────

def _add_v2_extras(df: pl.DataFrame) -> pl.DataFrame:
    """Add the 9 extra columns needed by 08c_v2 (58-feature models)."""
    # Computable from existing mid_return columns
    for lag in [1, 3, 5]:
        col = f"mid_return_{lag}s"
        new = f"abs_mid_return_{lag}s"
        if col in df.columns and new not in df.columns:
            df = df.with_columns(pl.col(col).abs().alias(new))
        elif new not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(new))
    # Near-zero importance — safe to zero-fill
    for c in ["past_jump_5s", "past_big_jump_5s", "btc_jump_5s",
              "btc_jump_15s", "spread_burst_5s", "depth_collapse_5s"]:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    return df


def eval_49feat_model(pkl_path: str, target_expr, dates: list[str],
                      features: list[str] | None = None,
                      extra_prep=None) -> tuple[dict, dict | None]:
    """Evaluate a model that uses tabular+BTC features (49 or 58 cols)."""
    # === FEATURE-CLEANUP for cleaned-artifact eval ===
    import sys as _sys
    _sys.path.insert(0, "src")
    from feature_cleanup import clean_features, cleaned_feature_list
    # =================================================
    model = joblib.load(pkl_path)
    features = features or FEAT_49
    # Translate to cleaned feature names so they match what the cleaned model expects
    features = cleaned_feature_list(features)
    day_results: dict[str, dict | None] = {}
    all_y, all_p = [], []

    for date in dates:
        tab = load_tab(date)
        if len(tab) == 0:
            day_results[date] = None
            continue

        # Add BTC features using the TRAINING-TIME implementation (percent returns,
        # per-market rolling, 30-row std).  Earlier add_btc_features used log returns +
        # asof joins + 30s time-windowed std, which is a train/inference mismatch.
        if not all(f in tab.columns for f in BTC_FEATS):
            # augment_btc_features needs live_btc_usd, market_slug, anchor_ts_ns
            import sys as _sys2
            _sys2.path.insert(0, ".")
            from train_move_size_v2 import augment_btc_features
            tab = augment_btc_features(tab)

        # Additional prep (e.g., v2 extra columns)
        if extra_prep is not None:
            tab = extra_prep(tab)

        # Apply feature cleanup so cleaned columns exist
        tab = clean_features(tab)

        # Compute target
        df = tab.with_columns(target_expr.alias("__target__")).filter(
            pl.col("__target__").is_not_null()
        )
        y = df["__target__"].to_numpy().astype(int)
        X = make_X(df, features)
        p = predict_clf(model, X)
        r = clf_metrics(y, p)
        day_results[date] = r
        if r:
            all_y.append(y); all_p.append(p)

    oos = clf_metrics(np.concatenate(all_y), np.concatenate(all_p)) if all_y else None
    return day_results, oos


def eval_agg32_model(pkl_path: str, target_expr, dates: list[str]) -> tuple[dict, dict | None]:
    """Evaluate an 08c v3 agg_only model (32 features, needs sequence_length_actual from e64)."""
    # === FEATURE-CLEANUP for cleaned-artifact eval ===
    import sys as _sys
    _sys.path.insert(0, "src")
    from feature_cleanup import clean_features, cleaned_feature_list
    # =================================================
    model = joblib.load(pkl_path)
    features = cleaned_feature_list(AGG_FEATS_32)
    day_results: dict[str, dict | None] = {}
    all_y, all_p = [], []

    for date in dates:
        tab = load_tab(date)
        e64 = load_e64(date)
        if len(tab) == 0:
            day_results[date] = None
            continue

        # Join sequence_length_actual from e64
        if "sequence_length_actual" not in tab.columns and "sequence_length_actual" in e64.columns:
            key_cols = ["sequence_id"] if "sequence_id" in tab.columns and "sequence_id" in e64.columns \
                       else ["anchor_ts_ns", "token_asset_id"] if "token_asset_id" in tab.columns else ["anchor_ts_ns"]
            join_cols = key_cols + ["sequence_length_actual"]
            tab = tab.join(e64.select([c for c in join_cols if c in e64.columns]),
                           on=[c for c in key_cols if c in tab.columns and c in e64.columns],
                           how="left")

        # Apply feature cleanup so cleaned columns exist
        tab = clean_features(tab)

        df = tab.with_columns(target_expr.alias("__target__")).filter(
            pl.col("__target__").is_not_null()
        )
        y = df["__target__"].to_numpy().astype(int)
        X = make_X(df, features)
        p = predict_clf(model, X)
        r = clf_metrics(y, p)
        day_results[date] = r
        if r:
            all_y.append(y); all_p.append(p)

    oos = clf_metrics(np.concatenate(all_y), np.concatenate(all_p)) if all_y else None
    return day_results, oos


# ── Model specs ───────────────────────────────────────────────────────────────

TARGET_EXPRS = {
    "big_move_02":      (pl.col("mid_move_5s").abs() >= 0.02).cast(pl.Int8),
    "extreme_10c_5s":   (pl.col("mid_move_5s").abs() >= 0.10).cast(pl.Int8),
    "severe_7c_3s":     (pl.col("mid_move_3s").abs() >= 0.07).cast(pl.Int8),
    "moderate_5c_5s":   (pl.col("mid_move_5s").abs() >= 0.05).cast(pl.Int8),
}

# Test-day reference AUC from stored metrics.json
REF_AUC = {
    "08b":                         None,   # no test AUC stored for 08b
    "08c_v1/extreme_10c_5s":       0.873,
    "08c_v1/severe_7c_3s":         json.loads(Path("artifacts_cleaned/model_08c_maker_defense/severe_7c_3s/metrics.json").read_text()).get("auc_test"),
    "08c_v1/moderate_5c_5s":       json.loads(Path("artifacts_cleaned/model_08c_maker_defense/moderate_5c_5s/metrics.json").read_text()).get("auc_test"),
    "08c_v2/extreme_10c_5s":       json.loads(Path("artifacts_cleaned/model_08c_maker_defense_v2/extreme_10c_5s/metrics.json").read_text()).get("auc_test"),
    "08c_v2/severe_7c_3s":         json.loads(Path("artifacts_cleaned/model_08c_maker_defense_v2/severe_7c_3s/metrics.json").read_text()).get("auc_test"),
    "08c_v2/moderate_5c_5s":       json.loads(Path("artifacts_cleaned/model_08c_maker_defense_v2/moderate_5c_5s/metrics.json").read_text()).get("auc_test"),
    "08c_v3_agg/extreme_10c_5s":   json.loads(Path("artifacts_cleaned/model_08c_maker_defense_v3_eventtime/extreme_10c_5s/agg_only/metrics.json").read_text()).get("auc_test"),
    "08c_v3_agg/severe_7c_3s":     json.loads(Path("artifacts_cleaned/model_08c_maker_defense_v3_eventtime/severe_7c_3s/agg_only/metrics.json").read_text()).get("auc_test"),
    "08c_v3_agg/moderate_5c_5s":   json.loads(Path("artifacts_cleaned/model_08c_maker_defense_v3_eventtime/moderate_5c_5s/agg_only/metrics.json").read_text()).get("auc_test"),
}

SPECS_49 = [
    ("08b", "artifacts_cleaned/model_08b_big_move_classifier/tabular_btc/lightgbm/model.pkl", "big_move_02"),
    *[(f"08c_v1/{t}", f"artifacts_cleaned/model_08c_maker_defense/{t}/model.pkl", t)
      for t in ["extreme_10c_5s", "severe_7c_3s", "moderate_5c_5s"]],
]

# 08c_v2: 58 features = FEAT_49 + EXTRA_9  (abs returns computable; jumps near-zero → 0)
SPECS_58 = [
    *[(f"08c_v2/{t}", f"artifacts_cleaned/model_08c_maker_defense_v2/{t}/model.pkl", t)
      for t in ["extreme_10c_5s", "severe_7c_3s", "moderate_5c_5s"]],
]

SPECS_AGG32 = [
    *[(f"08c_v3_agg/{t}", f"artifacts_cleaned/model_08c_maker_defense_v3_eventtime/{t}/agg_only/model.pkl", t)
      for t in ["extreme_10c_5s", "severe_7c_3s", "moderate_5c_5s"]],
]


# ── Reporting ─────────────────────────────────────────────────────────────────

W = 125

def _f(val, fmt):
    return f"{val:{fmt}}" if val is not None else "  —  "

def _drift(oos_auc, ref_auc):
    if oos_auc is None or ref_auc is None:
        return "   "
    d = oos_auc - ref_auc
    arrow = "+" if d > 0 else ""
    ok = "ok" if d >= -0.01 else "!!"
    return f"{arrow}{d:+.3f} {ok}"

def print_table(all_results):
    sep = "─" * W
    hdr = (
        f"{'MODEL':<30} {'BASE%':^7} │ "
        f"{'May-13':^7} {'May-14':^7} {'May-15':^7} {'May-16':^7} │ "
        f"{'OOS AUC':^8} │ "
        f"{'May-13':^7} {'May-14':^7} {'May-15':^7} {'May-16':^7} │ "
        f"{'OOS AP':^7} │ {'vs test':^10}"
    )
    sub = (
        f"{'':30} {'':7} │ "
        f"{'AUC-ROC':^31} │ "
        f"{'':8} │ "
        f"{'Avg Precision':^31} │ "
        f"{'':7} │ {'(ΔAUC)':^10}"
    )
    print(sep)
    print(hdr)
    print(sub)
    print(sep)
    for label, (day_results, oos) in all_results:
        br = oos["base_rate"] if oos else None
        br_str = f"{br:.2%}" if br is not None else "  —  "
        aucs = [_f(day_results.get(d) and day_results[d].get("auc"), ".4f") for d in DATES]
        aps  = [_f(day_results.get(d) and day_results[d].get("ap"),  ".4f") for d in DATES]
        oos_auc = _f(oos and oos.get("auc"), ".4f")
        oos_ap  = _f(oos and oos.get("ap"),  ".4f")
        drift = _drift(oos and oos.get("auc"), REF_AUC.get(label))
        print(
            f"{label:<30} {br_str:^7} │ "
            f"{aucs[0]:^7} {aucs[1]:^7} {aucs[2]:^7} {aucs[3]:^7} │ "
            f"{oos_auc:^8} │ "
            f"{aps[0]:^7} {aps[1]:^7} {aps[2]:^7} {aps[3]:^7} │ "
            f"{oos_ap:^7} │ {drift:^10}"
        )
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Pre-load and BTC-enrich tabular data (shared across all 49-feat models)
    print("Pre-computing BTC features for each date...", file=sys.stderr)
    btc_enriched: dict[str, pl.DataFrame] = {}
    for date in DATES:
        print(f"  {date} ...", end=" ", file=sys.stderr, flush=True)
        tab = load_tab(date)
        if len(tab) > 0:
            tab = add_btc_features(tab, date)
            _tab_cache[date] = tab    # update cache with enriched df
            print(f"{len(tab):,} rows, BTC cols present: {all(f in tab.columns for f in BTC_FEATS)}", file=sys.stderr)
        else:
            print("empty", file=sys.stderr)

    all_results = []

    print("\n[49-feature models: 08b and 08c v1]", file=sys.stderr)
    for label, pkl, target_key in SPECS_49:
        print(f"  {label} ...", end=" ", file=sys.stderr, flush=True)
        try:
            r = eval_49feat_model(pkl, TARGET_EXPRS[target_key], DATES)
            print("done", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            r = ({d: None for d in DATES}, None)
        all_results.append((label, r))

    print("\n[58-feature models: 08c v2  (abs_mid_return computed; jump cols zero-filled)]", file=sys.stderr)
    for label, pkl, target_key in SPECS_58:
        print(f"  {label} ...", end=" ", file=sys.stderr, flush=True)
        try:
            r = eval_49feat_model(pkl, TARGET_EXPRS[target_key], DATES,
                                  features=FEAT_58, extra_prep=_add_v2_extras)
            print("done", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            r = ({d: None for d in DATES}, None)
        all_results.append((label, r))

    print("\n[32-feature agg_only: 08c v3]", file=sys.stderr)
    for label, pkl, target_key in SPECS_AGG32:
        print(f"  {label} ...", end=" ", file=sys.stderr, flush=True)
        try:
            r = eval_agg32_model(pkl, TARGET_EXPRS[target_key], DATES)
            print("done", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            r = ({d: None for d in DATES}, None)
        all_results.append((label, r))

    print("\n", "=" * W)
    print("OOS EVALUATION — model_08b + model_08c (all variants)  |  2026-05-13 through 2026-05-16")
    print(f"Training test day: 2026-05-06  |  OOS gap: 7+ days")
    print(f"Metric: AUC-ROC and Average Precision (AP).  Base% = OOS positive rate.")
    print(f"vs test: OOS AUC minus known test-day AUC  (+ = improved, - = degraded).")
    print("=" * W)
    print()
    print("49-FEATURE MODELS  [08b + 08c v1 — tabular 43 features + 6 BTC-derived]")
    print()
    print_table([(l, r) for l, r in all_results if "v3" not in l and "v2" not in l])
    print()
    print("58-FEATURE MODELS  [08c v2 — 49 base + abs_mid_return_{1,3,5}s + 6 jump/burst cols zero-filled (near-0 importance)]")
    print()
    print_table([(l, r) for l, r in all_results if "v2" in l])
    print()
    print("08c v3 AGG_ONLY  [32 aggregate features; event_only/agg_plus_event skipped — need evt_* computation]")
    print()
    print_table([(l, r) for l, r in all_results if "v3" in l])
    print()
    print("SKIPPED:")
    print("  08c_v3/*/event_only     (61 features) — require evt_* statistics computed from event array columns")
    print("  08c_v3/*/agg_plus_event (93 features) — same requirement")

    # Save report
    out = Path("docs/oos_eval_08bc_cleaned_report.md")
    import io as _io
    buf = _io.StringIO()
    old_stdout = sys.stdout
    # re-run print_table to buf
    sys.stdout = buf
    print_table([(l, r) for l, r in all_results if "v3" not in l and "v2" not in l])
    print_table([(l, r) for l, r in all_results if "v2" in l])
    print_table([(l, r) for l, r in all_results if "v3" in l])
    sys.stdout = old_stdout
    out.write_text("```\n" + buf.getvalue() + "\n```\n", encoding="utf-8")
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
