#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Out-of-sample evaluation: all trained models on 2026-05-13 through 2026-05-16.

Training window was 2026-04-19..2026-05-06 (test day = May 6).
These four days are fully outside the training+val+test window.

Run:  py -3 oos_eval_may13_16.py
Output saved to:  docs/oos_eval_may13_16_cleaned_report.md
"""

from __future__ import annotations
import io, sys

# Force UTF-8 output on Windows so Unicode chars don't crash cp1250
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import joblib, yaml, json, textwrap
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error,
)

sys.path.insert(0, str(Path(".") / "src"))
from model_factory.trainers.logistic_regression_trainer import _to_float_array

DATES = ["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"]

# ── Test-day reference scores from eval_report.md / metrics.json ─────────────
# Used to compute OOS drift vs known test-day performance
TEST_SCORES: dict[str, dict] = {
    # classifiers: {"ll": float, "auc": float}
    "01_bias/preopen/lgbm":         {"ll": None,  "auc": None},   # GATE_FAIL, no clean test score
    "01_bias/preopen/lr":           {"ll": None,  "auc": None},
    "01_bias/first15s/lgbm":        {"ll": None,  "auc": None},
    "01_bias/first15s/lr":          {"ll": None,  "auc": None},
    "01_bias/first30s/lgbm":        {"ll": None,  "auc": None},
    "01_bias/first30s/lr":          {"ll": None,  "auc": None},
    "02_fair_res/coarse/lgbm":      {"ll": 0.646, "auc": None},
    "02_fair_res/coarse/lr":        {"ll": 0.664, "auc": None},
    "02_fair_res/dense_close/lgbm": {"ll": 0.402, "auc": None},
    "03_fill_prob/lgbm":            {"ll": 0.547, "auc": None},
    "03_fill_prob/lr":              {"ll": 0.617, "auc": None},
    "05_closing_flip/lgbm":         {"ll": None,  "auc": None},
    "05_closing_flip/lr":           {"ll": None,  "auc": None},
    "07_ms_direction/lgbm":         {"ll": None,  "auc": None},   # 69.5% acc on test
    "07_ms_direction/lr":           {"ll": None,  "auc": None},
    # regressors: {"mae": float, "sign_acc": float}
    "04_adverse_sel/lgbm":          {"mae": 0.147, "sign_acc": 0.674},
    "04_adverse_sel/linreg":        {"mae": 0.170, "sign_acc": None},
    "04b_mkt_to_close/lgbm":        {"mae": 1.023, "sign_acc": 0.564},
    "04b_mkt_to_close/linreg":      {"mae": None,  "sign_acc": None},
    "06_mispricing/lgbm":           {"mae": None,  "sign_acc": None},
    "06_mispricing/linreg":         {"mae": None,  "sign_acc": None},
    "08_move_size/lgbm":            {"mae": None,  "sign_acc": None},
    "08_move_size/linreg":          {"mae": None,  "sign_acc": None},
}

# ── Dataset path helpers ──────────────────────────────────────────────────────

def mi_ds(variant: str, date: str) -> str:
    return f"data/datasets/market_interval_dataset_v1_{variant}/{date}.parquet"

def rs_ds(variant: str, date: str) -> str:
    return f"data/datasets/resolution_snapshot_dataset_v1_{variant}/{date}.parquet"

def ms_ds(variant: str, date: str) -> str:
    return f"data/datasets/microstructure_sequence_dataset_v1_{variant}/{date}.parquet"

def ei_ds(date: str) -> str:
    return f"data/datasets/execution_intent_dataset_v1/{date}.parquet"

# ── Model specs ───────────────────────────────────────────────────────────────
# Fields:
#   label          display label (also keys TEST_SCORES)
#   gate           PASS / COND_PASS / GATE_FAIL
#   pkl            path to model.pkl
#   features_yaml  path to feature_columns yaml (None → read from model artifact)
#   dataset_fn     callable(date) -> parquet path
#   target         column name; prefix "COMPUTE:" means derive it
#   kind           "clf" or "reg"
#   row_filter     bool column to filter rows on (e.g. "filled")
#   nonnull_target filter out null target rows

SPECS = [
    # ── Model 01: Bias  (GATE_FAIL — loses to delta-to-strike LR baseline) ──
    *[dict(label=f"01_bias/{v}/lgbm", gate="GATE_FAIL",
           pkl=f"artifacts_cleaned/model_01_bias/{v}/lightgbm/model.pkl",
           features_yaml="config/feature_sets/model_01_bias_features_v1.yaml",
           dataset_fn=lambda d, v=v: mi_ds(v, d),
           target="resolved_side_label", kind="clf")
      for v in ["preopen", "first15s", "first30s"]],
    *[dict(label=f"01_bias/{v}/lr", gate="GATE_FAIL",
           pkl=f"artifacts_cleaned/model_01_bias/{v}/logistic_regression/model.pkl",
           features_yaml="config/feature_sets/model_01_bias_features_v1.yaml",
           dataset_fn=lambda d, v=v: mi_ds(v, d),
           target="resolved_side_label", kind="clf")
      for v in ["preopen", "first15s", "first30s"]],

    # ── Model 02: Fair Resolution ──────────────────────────────────────────
    *[dict(label=f"02_fair_res/{v}/lgbm", gate="PASS",
           pkl=f"artifacts_cleaned/model_02_fair_resolution/{v}/lightgbm/model.pkl",
           features_yaml="config/feature_sets/model_02_fair_resolution_features_v1.yaml",
           dataset_fn=lambda d, v=v: rs_ds(v, d),
           target="resolved_side_label", kind="clf")
      for v in ["coarse", "dense_close"]],
    dict(label="02_fair_res/coarse/lr", gate="PASS",
         pkl="artifacts_cleaned/model_02_fair_resolution/coarse/logistic_regression/model.pkl",
         features_yaml="config/feature_sets/model_02_fair_resolution_features_v1.yaml",
         dataset_fn=lambda d: rs_ds("coarse", d),
         target="resolved_side_label", kind="clf"),

    # ── Model 03: Fill Probability ─────────────────────────────────────────
    dict(label="03_fill_prob/lgbm", gate="PASS",
         pkl="artifacts_cleaned/model_03_fill_probability/lightgbm/model.pkl",
         features_yaml="config/feature_sets/model_03_fill_probability_features_v1.yaml",
         dataset_fn=ei_ds, target="filled", kind="clf"),
    dict(label="03_fill_prob/lr", gate="PASS",
         pkl="artifacts_cleaned/model_03_fill_probability/logistic_regression/model.pkl",
         features_yaml="config/feature_sets/model_03_fill_probability_features_v1.yaml",
         dataset_fn=ei_ds, target="filled", kind="clf"),

    # ── Model 05: Closing Flip ────────────────────────────────────────────
    dict(label="05_closing_flip/lgbm", gate="PASS",
         pkl="artifacts_cleaned/model_05_closing_flip/dense_close/lightgbm/model.pkl",
         features_yaml="config/feature_sets/model_05_closing_flip_features_v1.yaml",
         dataset_fn=lambda d: rs_ds("dense_close", d),
         target="flip_happened_after_t", kind="clf"),
    dict(label="05_closing_flip/lr", gate="PASS",
         pkl="artifacts_cleaned/model_05_closing_flip/dense_close/logistic_regression/model.pkl",
         features_yaml="config/feature_sets/model_05_closing_flip_features_v1.yaml",
         dataset_fn=lambda d: rs_ds("dense_close", d),
         target="flip_happened_after_t", kind="clf"),

    # ── Model 07: Microstructure Direction ───────────────────────────────
    # mid_move_sign_5s is ternary {-1, 0, 1}; model trained on non-zero rows: -1->0, 1->1
    dict(label="07_ms_direction/lgbm", gate="PASS",
         pkl="artifacts_cleaned/model_07_microstructure_direction/tabular/lightgbm/model.pkl",
         features_yaml="config/feature_sets/model_07_microstructure_direction_features_v1.yaml",
         dataset_fn=lambda d: ms_ds("tabular", d),
         target="mid_move_sign_5s", kind="clf", nonnull_target=True, binarize_target=True),
    dict(label="07_ms_direction/lr", gate="PASS",
         pkl="artifacts_cleaned/model_07_microstructure_direction/tabular/logistic_regression/model.pkl",
         features_yaml="config/feature_sets/model_07_microstructure_direction_features_v1.yaml",
         dataset_fn=lambda d: ms_ds("tabular", d),
         target="mid_move_sign_5s", kind="clf", nonnull_target=True, binarize_target=True),

    # ── Model 04: Adverse Selection (filled rows only) ────────────────────
    dict(label="04_adverse_sel/lgbm", gate="PASS",
         pkl="artifacts_cleaned/model_04_adverse_selection/lightgbm/model.pkl",
         features_yaml="config/feature_sets/model_04_adverse_selection_features_v1.yaml",
         dataset_fn=ei_ds, target="markout_1s", kind="reg",
         row_filter="filled", nonnull_target=True),
    dict(label="04_adverse_sel/linreg", gate="PASS",
         pkl="artifacts_cleaned/model_04_adverse_selection/linear_regression/model.pkl",
         features_yaml="config/feature_sets/model_04_adverse_selection_features_v1.yaml",
         dataset_fn=ei_ds, target="markout_1s", kind="reg",
         row_filter="filled", nonnull_target=True),

    # ── Model 04b: Markout-to-Close (filled rows, uses model_04 feature set) ─
    dict(label="04b_mkt_to_close/lgbm", gate="PASS",
         pkl="artifacts_cleaned/model_04b_markout_to_close/lightgbm/model.pkl",
         features_yaml="config/feature_sets/model_04_adverse_selection_features_v1.yaml",
         dataset_fn=ei_ds, target="markout_to_close", kind="reg",
         row_filter="filled", nonnull_target=True),
    dict(label="04b_mkt_to_close/linreg", gate="PASS",
         pkl="artifacts_cleaned/model_04b_markout_to_close/linear_regression/model.pkl",
         features_yaml="config/feature_sets/model_04_adverse_selection_features_v1.yaml",
         dataset_fn=ei_ds, target="markout_to_close", kind="reg",
         row_filter="filled", nonnull_target=True),

    # ── Model 06: Mispricing (dense_close, non-null target) ───────────────
    dict(label="06_mispricing/lgbm", gate="COND_PASS",
         pkl="artifacts_cleaned/model_06_mispricing/dense_close/lightgbm/model.pkl",
         features_yaml="config/feature_sets/model_06_mispricing_features_v1.yaml",
         dataset_fn=lambda d: rs_ds("dense_close", d),
         target="hold_to_close_edge_vs_mid", kind="reg", nonnull_target=True),
    dict(label="06_mispricing/linreg", gate="COND_PASS",
         pkl="artifacts_cleaned/model_06_mispricing/dense_close/linear_regression/model.pkl",
         features_yaml="config/feature_sets/model_06_mispricing_features_v1.yaml",
         dataset_fn=lambda d: rs_ds("dense_close", d),
         target="hold_to_close_edge_vs_mid", kind="reg", nonnull_target=True),

    # ── Model 08: Move Size ───────────────────────────────────────────────
    dict(label="08_move_size/lgbm", gate="COND_PASS",
         pkl="artifacts_cleaned/model_08_move_size/tabular/lightgbm/model.pkl",
         features_yaml="config/feature_sets/model_08_move_size_features_v1.yaml",
         dataset_fn=lambda d: ms_ds("tabular", d),
         target="mid_move_5s", kind="reg", nonnull_target=True),
    dict(label="08_move_size/linreg", gate="COND_PASS",
         pkl="artifacts_cleaned/model_08_move_size/tabular/linear_regression/model.pkl",
         features_yaml="config/feature_sets/model_08_move_size_features_v1.yaml",
         dataset_fn=lambda d: ms_ds("tabular", d),
         target="mid_move_5s", kind="reg", nonnull_target=True),
]

# ── Core evaluation helpers ───────────────────────────────────────────────────

def load_features(yaml_path: str) -> list[str]:
    return yaml.safe_load(Path(yaml_path).read_text())["feature_columns"]


def make_X(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    avail = set(df.columns)
    cols = [c for c in features if c in avail]
    X, _ = _to_float_array(df.select(cols))
    return np.nan_to_num(X, nan=0.0)


def predict_model(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)[:, 1]
        cal = getattr(model, "_calibrator", None)
        return np.clip(cal.predict(raw), 1e-6, 1 - 1e-6) if cal else raw
    return model.predict(X)


def clf_metrics(y: np.ndarray, p: np.ndarray) -> dict | None:
    if len(y) == 0:
        return None
    y = y.astype(int)
    unique = np.unique(y)
    if len(unique) < 2:
        return {"ll": None, "auc": None, "brier": None, "n": int(len(y)),
                "base_rate": float(y.mean())}
    return {
        "ll":    float(log_loss(y, p)),
        "auc":   float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "n":     int(len(y)),
        "base_rate": float(y.mean()),
    }


def reg_metrics(y: np.ndarray, p: np.ndarray) -> dict | None:
    if len(y) == 0:
        return None
    y = y.astype(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return None
    return {
        "mae":      float(mean_absolute_error(y, p)),
        "rmse":     float(np.sqrt(mean_squared_error(y, p))),
        "sign_acc": float(np.mean(np.sign(p) == np.sign(y))),
        "n":        int(len(y)),
    }


def eval_spec(spec: dict) -> tuple[dict, dict | None]:
    # === FEATURE-CLEANUP IMPORT (for cleaned-artifact eval) ===
    import sys as _sys
    _sys.path.insert(0, "src")
    from feature_cleanup import clean_features, cleaned_feature_list
    # ==========================================================
    model    = joblib.load(spec["pkl"])
    features = load_features(spec["features_yaml"])
    # Translate to cleaned feature names so they match what the cleaned model expects
    features = cleaned_feature_list(features)
    target   = spec["target"]
    kind     = spec["kind"]
    row_filter    = spec.get("row_filter")
    nonnull_target  = spec.get("nonnull_target", False)
    binarize_target = spec.get("binarize_target", False)  # ternary {-1,0,1} -> filter 0s, map -1->0

    # Cache schema per parquet to avoid repeated scans
    day_results: dict[str, dict | None] = {}
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []

    for date in DATES:
        path = spec["dataset_fn"](date)
        if not Path(path).exists():
            day_results[date] = None
            continue

        schema_names = set(pl.scan_parquet(path).collect_schema().names())
        needed = list((set(features) | {target}) & schema_names)
        if row_filter and row_filter in schema_names:
            needed.append(row_filter)

        df = pl.read_parquet(path, columns=list(set(needed)))
        df = clean_features(df)  # ensure cleaned columns exist for the cleaned model

        if row_filter and row_filter in df.columns:
            df = df.filter(pl.col(row_filter).cast(pl.Boolean))
        if nonnull_target and target in df.columns:
            df = df.filter(pl.col(target).is_not_null())
        if binarize_target and target in df.columns:
            # Filter out zero-move rows, remap -1->0, 1->1
            df = df.filter(pl.col(target) != 0)
            df = df.with_columns(
                pl.when(pl.col(target) > 0).then(1).otherwise(0).alias(target)
            )

        if len(df) == 0 or target not in df.columns:
            day_results[date] = None
            continue

        X = make_X(df, features)
        p = predict_model(model, X)
        y = df[target].to_numpy()

        result = clf_metrics(y, p) if kind == "clf" else reg_metrics(y, p)
        day_results[date] = result
        if result is not None:
            all_y.append(y)
            all_p.append(p.astype(np.float64))

    oos: dict | None = None
    if all_y:
        Y = np.concatenate(all_y)
        P = np.concatenate(all_p)
        oos = clf_metrics(Y, P) if kind == "clf" else reg_metrics(Y, P)

    return day_results, oos


# ── Formatting helpers ────────────────────────────────────────────────────────

def _f(val, fmt) -> str:
    return f"{val:{fmt}}" if val is not None else "  —  "

def _delta(oos_val, ref_val, higher_better: bool) -> str:
    if oos_val is None or ref_val is None:
        return "   "
    d = oos_val - ref_val
    sign = "▲" if d > 0 else "▼"
    good = (d > 0) == higher_better
    tag = "✓" if good else "✗"
    return f"{sign}{abs(d):.3f}{tag}"


# ── Report builders ───────────────────────────────────────────────────────────

W = 130

def clf_table(results: list) -> list[str]:
    rows = []
    sep = "─" * W
    rows.append(sep)
    rows.append(
        f"{'MODEL':<32} {'GATE':^10} │ "
        f"{'May-13':^8} {'May-14':^8} {'May-15':^8} {'May-16':^8} │ "
        f"{'OOS-all':^8} │ "
        f"{'May-13':^7} {'May-14':^7} {'May-15':^7} {'May-16':^7} │ "
        f"{'OOS AUC':^8} │ {'vs test':^9}"
    )
    rows.append(
        f"{'':32} {'':10} │ "
        f"{'log_loss':^35} │ "
        f"{'':8} │ "
        f"{'AUC-ROC':^31} │ "
        f"{'':8} │ {'(ΔlogLoss)':^9}"
    )
    rows.append(sep)

    for spec, (day_results, oos) in results:
        lbl = spec["label"]
        gate_flag = {"PASS": "PASS", "COND_PASS": "COND◆", "GATE_FAIL": "FAIL★"}[spec["gate"]]
        lls  = [_f(day_results.get(d, {}) and day_results[d] and day_results[d].get("ll"),  ".4f") for d in DATES]
        aucs = [_f(day_results.get(d, {}) and day_results[d] and day_results[d].get("auc"), ".3f") for d in DATES]
        oos_ll  = _f(oos and oos.get("ll"),  ".4f")
        oos_auc = _f(oos and oos.get("auc"), ".4f")
        ref = TEST_SCORES.get(lbl, {})
        drift = _delta(oos and oos.get("ll"), ref.get("ll"), higher_better=False)
        rows.append(
            f"{lbl:<32} {gate_flag:^10} │ "
            f"{lls[0]:^8} {lls[1]:^8} {lls[2]:^8} {lls[3]:^8} │ "
            f"{oos_ll:^8} │ "
            f"{aucs[0]:^7} {aucs[1]:^7} {aucs[2]:^7} {aucs[3]:^7} │ "
            f"{oos_auc:^8} │ {drift:^9}"
        )
    rows.append(sep)
    return rows


def reg_table(results: list) -> list[str]:
    rows = []
    sep = "─" * W
    rows.append(sep)
    rows.append(
        f"{'MODEL':<32} {'GATE':^10} │ "
        f"{'May-13':^8} {'May-14':^8} {'May-15':^8} {'May-16':^8} │ "
        f"{'OOS-all':^8} │ "
        f"{'May-13':^7} {'May-14':^7} {'May-15':^7} {'May-16':^7} │ "
        f"{'OOS sgn%':^8} │ {'vs test':^9}"
    )
    rows.append(
        f"{'':32} {'':10} │ "
        f"{'MAE':^35} │ "
        f"{'':8} │ "
        f"{'sign_accuracy':^31} │ "
        f"{'':8} │ {'(ΔMAE)':^9}"
    )
    rows.append(sep)

    for spec, (day_results, oos) in results:
        lbl = spec["label"]
        gate_flag = {"PASS": "PASS", "COND_PASS": "COND◆", "GATE_FAIL": "FAIL★"}[spec["gate"]]
        maes = [_f(day_results.get(d) and day_results[d].get("mae"), ".4f") for d in DATES]
        sgns = [_f(day_results.get(d) and day_results[d].get("sign_acc"), ".1%") for d in DATES]
        oos_mae = _f(oos and oos.get("mae"), ".4f")
        oos_sgn = _f(oos and oos.get("sign_acc"), ".1%")
        ref = TEST_SCORES.get(lbl, {})
        drift = _delta(oos and oos.get("mae"), ref.get("mae"), higher_better=False)
        rows.append(
            f"{lbl:<32} {gate_flag:^10} │ "
            f"{maes[0]:^8} {maes[1]:^8} {maes[2]:^8} {maes[3]:^8} │ "
            f"{oos_mae:^8} │ "
            f"{sgns[0]:^7} {sgns[1]:^7} {sgns[2]:^7} {sgns[3]:^7} │ "
            f"{oos_sgn:^8} │ {drift:^9}"
        )
    rows.append(sep)
    return rows


def row_counts_table(results_all: list) -> list[str]:
    rows = ["", "ROW COUNTS PER DAY", "─" * 70,
            f"{'MODEL':<40} {'May-13':>8} {'May-14':>8} {'May-15':>8} {'May-16':>8}",
            "─" * 70]
    for spec, (day_results, _) in results_all:
        ns = []
        for d in DATES:
            r = day_results.get(d)
            ns.append(f"{r['n']:>8,}" if r else "       —")
        rows.append(f"{spec['label']:<40} {''.join(ns)}")
    rows.append("─" * 70)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    clf_results = []
    reg_results = []
    all_results = []

    for i, spec in enumerate(SPECS):
        print(f"[{i+1}/{len(SPECS)}] {spec['label']} ...", end=" ", flush=True)
        try:
            day_results, oos = eval_spec(spec)
            print("done")
        except Exception as exc:
            print(f"ERROR: {exc}")
            day_results = {d: None for d in DATES}
            oos = None

        entry = (spec, (day_results, oos))
        all_results.append(entry)
        (clf_results if spec["kind"] == "clf" else reg_results).append(entry)

    # ── Build report ──────────────────────────────────────────────────────
    lines = [
        "=" * W,
        "OUT-OF-SAMPLE EVALUATION REPORT  — 2026-05-13 through 2026-05-16",
        f"Training window: 2026-04-19..2026-05-06  |  OOS gap: 7+ days beyond test day",
        "=" * W,
        "",
        "CLASSIFIER MODELS  (primary metric: log_loss — lower is better)",
        "NOTE: 'vs test' column = OOS_all log_loss minus test-day log_loss;  ▼=improved ✓ / degraded ✗",
        "",
    ]
    lines += clf_table(clf_results)
    lines += [
        "",
        "REGRESSOR MODELS  (primary metric: MAE — lower is better)",
        "NOTE: 'vs test' column = OOS_all MAE minus test-day MAE;  ▼=improved ✓ / degraded ✗",
        "",
    ]
    lines += reg_table(reg_results)
    lines += row_counts_table(all_results)
    lines += [
        "",
        "LEGEND",
        "  PASS      Gate passed on training test-day (May 6)",
        "  COND◆     Conditional pass — use with caveats (see registry)",
        "  FAIL★     Gate FAILED audit — model has no signal beyond simple baseline",
        "  vs test   Δ relative to known test-day score; ▼✓ = degraded; ▼✗ = improved vs test",
        "            (counter-intuitive: lower log_loss/MAE = better, so ▼ = improvement)",
        "",
        "SKIPPED MODELS (require features not embedded in tabular parquet):",
        "  model_08b_big_move_classifier      — needs btc_return_3s/5s/15s, btc_abs_return_5s/15s,",
        "                                       btc_realized_vol_30s (not in tabular dataset)",
        "  model_08c_maker_defense (v1/v2)    — same 49 features incl. 6 BTC-derived cols",
        "  model_08c_maker_defense_v3 (all)   — feature ordering not recoverable from stored artifacts",
        "",
        "  To evaluate these, compute BTC derived features from external_btc_events_v1 and join",
        "  onto the tabular microstructure dataset by anchor_ts_ns.",
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)

    out_path = Path("docs/oos_eval_may13_16_cleaned_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("```\n" + report_text + "\n```\n", encoding="utf-8")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
