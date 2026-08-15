#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOS evaluation for the trojan-immune 08c v3 event_only and agg_plus_event variants.

These models use evt_* features computed from the event-array columns in the
event64 parquet (events_microprice, events_imbalance, events_spread, etc.).
They are STRUCTURALLY immune to the trojan-feature failure mode because they
don't see live_btc_usd, price_to_beat, or live_applied_bias_age_seconds at all
(0% trojan share by construction).

Evaluated:
  - artifacts_cleaned/model_08c_maker_defense_v3_eventtime/{target}/event_only/
  - artifacts_cleaned/model_08c_maker_defense_v3_eventtime/{target}/agg_plus_event/
  for target in {extreme_10c_5s, severe_7c_3s, moderate_5c_5s}

Output: docs/oos_eval_event_only_report.md  +  console table
"""
from __future__ import annotations
import io, sys, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import joblib
import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from model_factory.trainers.logistic_regression_trainer import _to_float_array
from feature_cleanup import clean_features, cleaned_feature_list

# Import the canonical event-feature builder and lists from training script
from train_maker_defense_v3_eventtime import (
    build_event_features,
    EVENT_FEATURES,
    AGG_FEATURES,
)

ALL_FEATURES = AGG_FEATURES + EVENT_FEATURES

OOS_DATES = ["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"]

TARGETS = {
    "extreme_10c_5s": (pl.col("mid_move_5s").abs() >= 0.10).cast(pl.Int8),
    "severe_7c_3s":   (pl.col("mid_move_3s").abs() >= 0.07).cast(pl.Int8),
    "moderate_5c_5s": (pl.col("mid_move_5s").abs() >= 0.05).cast(pl.Int8),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_X(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    avail = set(df.columns)
    cols = [c for c in features if c in avail]
    X, _ = _to_float_array(df.select(cols))
    return np.nan_to_num(X, nan=0.0)


def _predict_clf(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)[:, 1]
        cal = getattr(model, "_calibrator", None)
        if cal is not None:
            return np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
        return raw
    return model.predict(X)


def _metrics(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return {
        "n":   int(len(y)),
        "auc": float(roc_auc_score(y, p)),
        "ll":  float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ap":  float(average_precision_score(y, p)),
        "base_rate": float(y.mean()),
    }


_e64_cache: dict[str, pl.DataFrame] = {}

def load_e64_with_events(date: str) -> pl.DataFrame:
    """Load event64 parquet including event-list columns (heavier than load_e64)."""
    if date in _e64_cache:
        return _e64_cache[date]
    p = Path(f"data/datasets/microstructure_sequence_dataset_v1_event64/{date}.parquet")
    if not p.exists():
        _e64_cache[date] = pl.DataFrame()
        return _e64_cache[date]
    df = pl.read_parquet(p)
    # Apply same eligibility / min-seq-len filters as training
    if "sequence_feature_eligible" in df.columns:
        df = df.filter(pl.col("sequence_feature_eligible"))
    df = df.filter(pl.col("sequence_length_actual") >= 32)
    df = df.filter(pl.col("mid_move_5s").is_not_null() & pl.col("mid_move_5s").is_not_nan())
    # Build evt_* features
    print(f"    [{date}] building event features on {len(df):,} rows ...", file=sys.stderr, flush=True)
    df = build_event_features(df)
    # Apply feature cleanup so cleaned column names exist
    df = clean_features(df)
    _e64_cache[date] = df
    return df


def evaluate(pkl_path: str, features: list[str], target_key: str) -> tuple[dict, dict | None]:
    if not Path(pkl_path).exists():
        return {}, None
    model = joblib.load(pkl_path)
    features = cleaned_feature_list(features)
    target_expr = TARGETS[target_key]
    day_results: dict[str, dict | None] = {}
    all_y, all_p = [], []
    for date in OOS_DATES:
        df = load_e64_with_events(date)
        if len(df) == 0:
            day_results[date] = None
            continue
        df = df.with_columns(target_expr.alias("__target__")).filter(
            pl.col("__target__").is_not_null()
        )
        y = df["__target__"].to_numpy().astype(int)
        X = _make_X(df, features)
        p = _predict_clf(model, X)
        r = _metrics(y, p)
        day_results[date] = r
        if r:
            all_y.append(y)
            all_p.append(p)
    oos = _metrics(np.concatenate(all_y), np.concatenate(all_p)) if all_y else None
    return day_results, oos


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    targets = ["severe_7c_3s", "moderate_5c_5s", "extreme_10c_5s"]
    variants = [
        ("event_only", EVENT_FEATURES),
        ("agg_plus_event", ALL_FEATURES),
    ]

    rows = []
    for variant_name, features in variants:
        for tname in targets:
            label = f"08c_v3_{variant_name}/{tname}"
            pkl = f"artifacts_cleaned/model_08c_maker_defense_v3_eventtime/{tname}/{variant_name}/model.pkl"
            print(f"\n[{label}] {pkl}", file=sys.stderr)
            day_results, oos = evaluate(pkl, features, tname)
            rows.append({
                "label": label, "variant": variant_name, "target": tname,
                "day_results": day_results, "oos": oos,
            })
            print(f"  OOS: {oos}", file=sys.stderr)

    # ── Build report ─────────────────────────────────────────────────────────
    lines = []
    lines.append("# OOS Evaluation — 08c v3 event_only & agg_plus_event (trojan-immune)\n\n")
    lines.append("These models are **structurally immune** to the trojan-feature failure mode:\n")
    lines.append("they use only `evt_*` features computed from the L2 event arrays.\n\n")
    lines.append(f"OOS window: {OOS_DATES[0]} .. {OOS_DATES[-1]}\n\n")

    lines.append("## Results\n\n")
    lines.append("| Variant / Target | OOS N | Base% | OOS AUC | OOS AP | OOS log-loss |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for r in rows:
        oos = r["oos"]
        if oos is None:
            lines.append(f"| `{r['label']}` | — | — | — | — | — |\n")
            continue
        lines.append(
            f"| `{r['label']}` | {oos['n']:,} | {oos['base_rate']*100:.1f}% | "
            f"**{oos['auc']:.4f}** | {oos['ap']:.4f} | {oos['ll']:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## Per-Day Breakdown\n\n")
    lines.append("| Variant / Target | May-13 AUC | May-14 AUC | May-15 AUC | May-16 AUC |\n")
    lines.append("|---|---|---|---|---|\n")
    for r in rows:
        cells = []
        for d in OOS_DATES:
            dr = r["day_results"].get(d)
            cells.append(f"{dr['auc']:.4f}" if dr else "—")
        lines.append(f"| `{r['label']}` | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # Console summary
    print("\n" + "=" * 80)
    print("OOS — 08c v3 event_only / agg_plus_event (trojan-immune)")
    print("=" * 80)
    print(f"{'MODEL':<48} {'N':>9} {'BASE%':>7} {'AUC':>8} {'AP':>8}")
    print("-" * 90)
    for r in rows:
        oos = r["oos"]
        if oos is None:
            print(f"{r['label']:<48} {'—':>9} {'—':>7} {'—':>8} {'—':>8}")
        else:
            print(f"{r['label']:<48} {oos['n']:>9,} {oos['base_rate']*100:>6.1f}% "
                  f"{oos['auc']:>8.4f} {oos['ap']:>8.4f}")
    print()

    out = Path("docs/oos_eval_event_only_report.md")
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Report saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
