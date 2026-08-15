#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-trojan OOS re-evaluation.

Tests how much of each model's apparent signal comes from the three trojan
features (live_btc_usd, price_to_beat, live_applied_bias_age_seconds).

For every CRITICAL model identified by trojan_audit.py, evaluates the model
on May 13-16 under THREE conditions and compares AUC / log-loss:

  (1) BASELINE        - features as-is (matches oos_eval_*_report numbers)
  (2) TROJAN_ZEROED   - trojan features replaced with 0.0
  (3) TROJAN_MEDIAN   - trojan features replaced with the training-set median
                        (a fairer "what if the feature were uninformative" test)

INTERPRETATION:
  - If AUC stays roughly the same across all 3   -> model has real signal, retrain
                                                    safely with cleaned features
  - If AUC collapses when trojans removed         -> signal was trojan-driven;
                                                    target may need to be dropped
                                                    or feature set redesigned
  - If AUC improves when trojans zeroed           -> trojans were actively harmful
                                                    OOS (regime shift case)

OUTPUT: docs/oos_eval_zero_trojan_report.md  +  console table
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json, joblib
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score, mean_absolute_error

sys.path.insert(0, str(Path(".") / "src"))
from model_factory.trainers.logistic_regression_trainer import _to_float_array
from feature_cleanup import TROJAN_HARD

# Reuse machinery from oos_eval scripts
from oos_eval_08bc import (
    FEAT_49, FEAT_58, EXTRA_9, AGG_FEATS_32,
    BTC_FEATS, add_btc_features, _add_v2_extras,
    predict_clf, make_X,
)

OOS_DATES = ["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"]
TRAIN_SAMPLE_DATES = [f"2026-04-{d:02d}" for d in range(19, 31)] + [f"2026-05-{d:02d}" for d in range(1, 7)]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_tab(date: str, *, with_btc: bool = True) -> pl.DataFrame:
    p = Path(f"data/datasets/microstructure_sequence_dataset_v1_tabular/{date}.parquet")
    if not p.exists():
        return pl.DataFrame()
    df = pl.read_parquet(p)
    if with_btc and not all(f in df.columns for f in BTC_FEATS):
        df = add_btc_features(df, date)
    return df


def compute_training_medians(features: list[str], sample_per_day: int = 10_000) -> dict[str, float]:
    """Compute the median of each trojan feature across a sample of training-window data."""
    parts = []
    for d in TRAIN_SAMPLE_DATES:
        df = load_tab(d, with_btc=False)
        if len(df) == 0:
            continue
        # Only need trojan cols + anchor_ts_ns
        keep = [c for c in features if c in df.columns]
        if not keep:
            continue
        s = df.select(keep)
        if len(s) > sample_per_day:
            s = s.sample(n=sample_per_day, seed=42)
        parts.append(s)
    if not parts:
        return {}
    cat = pl.concat(parts, how="diagonal")
    out = {}
    for f in features:
        if f not in cat.columns:
            continue
        col = cat.get_column(f).cast(pl.Float64, strict=False).to_numpy()
        col = col[np.isfinite(col)]
        if len(col):
            out[f] = float(np.median(col))
    return out


def metrics_for(y: np.ndarray, p: np.ndarray, *, regression: bool = False) -> dict | None:
    if len(y) == 0:
        return None
    if regression:
        return {
            "n": int(len(y)),
            "mae": float(mean_absolute_error(y, p)),
            "sign_acc": float(np.mean(np.sign(y) == np.sign(p))),
        }
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return None
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)),
        "ll":  float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ap":  float(average_precision_score(y, p)),
    }


def apply_substitution(X: np.ndarray, features: list[str], mode: str,
                       medians: dict[str, float]) -> np.ndarray:
    """Return a copy of X with trojan columns replaced according to mode.

    mode in {"baseline", "zero", "median"}.
    """
    if mode == "baseline":
        return X
    X2 = X.copy()
    for i, f in enumerate(features):
        if f not in TROJAN_HARD:
            continue
        if mode == "zero":
            X2[:, i] = 0.0
        elif mode == "median":
            X2[:, i] = medians.get(f, 0.0)
    return X2


# ── Model specs ──────────────────────────────────────────────────────────────

# (label, artifact_path, features_list, target_expr_or_col, kind, extra_prep)
TARGET_EXPRS = {
    "big_move_02":     (pl.col("mid_move_5s").abs() >= 0.02).cast(pl.Int8),
    "extreme_10c_5s":  (pl.col("mid_move_5s").abs() >= 0.10).cast(pl.Int8),
    "severe_7c_3s":    (pl.col("mid_move_3s").abs() >= 0.07).cast(pl.Int8),
    "moderate_5c_5s":  (pl.col("mid_move_5s").abs() >= 0.05).cast(pl.Int8),
}

# 08c family — covered by oos_eval_08bc machinery
SPECS_08C = [
    ("08b/tabular_btc",                "artifacts/model_08b_big_move_classifier/tabular_btc/lightgbm/model.pkl",     FEAT_49, "big_move_02",     "clf", None),
    ("08c_v1/extreme_10c_5s",          "artifacts/model_08c_maker_defense/extreme_10c_5s/model.pkl",                  FEAT_49, "extreme_10c_5s",  "clf", None),
    ("08c_v1/severe_7c_3s",            "artifacts/model_08c_maker_defense/severe_7c_3s/model.pkl",                    FEAT_49, "severe_7c_3s",    "clf", None),
    ("08c_v1/moderate_5c_5s",          "artifacts/model_08c_maker_defense/moderate_5c_5s/model.pkl",                  FEAT_49, "moderate_5c_5s",  "clf", None),
    ("08c_v2/extreme_10c_5s",          "artifacts/model_08c_maker_defense_v2/extreme_10c_5s/model.pkl",               FEAT_58, "extreme_10c_5s",  "clf", _add_v2_extras),
    ("08c_v2/severe_7c_3s",            "artifacts/model_08c_maker_defense_v2/severe_7c_3s/model.pkl",                 FEAT_58, "severe_7c_3s",    "clf", _add_v2_extras),
    ("08c_v2/moderate_5c_5s",          "artifacts/model_08c_maker_defense_v2/moderate_5c_5s/model.pkl",               FEAT_58, "moderate_5c_5s",  "clf", _add_v2_extras),
    ("08c_v3_agg/extreme_10c_5s",      "artifacts/model_08c_maker_defense_v3_eventtime/extreme_10c_5s/agg_only/model.pkl", AGG_FEATS_32, "extreme_10c_5s",  "clf", None),
    ("08c_v3_agg/severe_7c_3s",        "artifacts/model_08c_maker_defense_v3_eventtime/severe_7c_3s/agg_only/model.pkl",   AGG_FEATS_32, "severe_7c_3s",    "clf", None),
    ("08c_v3_agg/moderate_5c_5s",      "artifacts/model_08c_maker_defense_v3_eventtime/moderate_5c_5s/agg_only/model.pkl", AGG_FEATS_32, "moderate_5c_5s",  "clf", None),
]

# Model 03 / 04 / 05 / 07 / 02 — different feature sets, loaded from feature_importance.json keys
def _features_from_fi(path: str) -> list[str]:
    """Load feature names in training order from a feature_importance.json."""
    return list(json.loads(Path(path).read_text()).keys())

SPECS_OTHER = []
for _spec in [
    ("02_fair_res/coarse/lgbm",       "artifacts/model_02_fair_resolution/coarse/lightgbm",       None),
    ("02_fair_res/dense_close/lgbm",  "artifacts/model_02_fair_resolution/dense_close/lightgbm",  None),
    ("03_fill_prob/lgbm",             "artifacts/model_03_fill_probability/lightgbm",             None),
    ("04_adverse_sel/lgbm",           "artifacts/model_04_adverse_selection/lightgbm",            None),
    ("04b_mkt_to_close/lgbm",         "artifacts/model_04b_markout_to_close/lightgbm",            None),
    ("05_closing_flip/lgbm",          "artifacts/model_05_closing_flip/dense_close/lightgbm",     None),
    ("06_mispricing/lgbm",            "artifacts/model_06_mispricing/dense_close/lightgbm",       None),
    ("07_ms_direction/lgbm",          "artifacts/model_07_microstructure_direction/tabular/lightgbm", None),
    ("08_move_size/lgbm",             "artifacts/model_08_move_size/tabular/lightgbm",            None),
]:
    label, art_dir, _ = _spec
    fi_path = f"{art_dir}/feature_importance.json"
    pkl     = f"{art_dir}/model.pkl"
    if not (Path(fi_path).exists() and Path(pkl).exists()):
        continue
    feats = _features_from_fi(fi_path)
    SPECS_OTHER.append((label, pkl, feats))


# ── Main eval loop ───────────────────────────────────────────────────────────

def _load_oos_with_features(features: list[str], prep_fn=None) -> dict[str, pl.DataFrame]:
    """Cache OOS data per date with BTC features added (and any extra prep)."""
    cache: dict[str, pl.DataFrame] = {}
    for d in OOS_DATES:
        df = load_tab(d, with_btc=True)
        if prep_fn is not None:
            df = prep_fn(df)
        cache[d] = df
    return cache


def eval_model_clf(label: str, pkl: str, features: list[str], target_key: str,
                   medians: dict[str, float], prep_fn=None) -> dict:
    if not Path(pkl).exists():
        return {"label": label, "skipped": "missing artifact"}
    try:
        model = joblib.load(pkl)
    except Exception as e:
        return {"label": label, "skipped": f"load error: {e}"}

    cache = _load_oos_with_features(features, prep_fn)
    all_y = []
    Xs = []
    for d in OOS_DATES:
        df = cache[d]
        if len(df) == 0:
            continue
        if target_key in TARGET_EXPRS:
            df = df.with_columns(TARGET_EXPRS[target_key].alias("__target__")).filter(
                pl.col("__target__").is_not_null()
            )
            y = df["__target__"].to_numpy().astype(int)
        else:
            # Direct target column lookup (model 02/03/etc. — needs per-model logic)
            return {"label": label, "skipped": "no target_expr defined"}
        X = make_X(df, features)
        all_y.append(y)
        Xs.append(X)
    if not Xs:
        return {"label": label, "skipped": "no rows"}
    y_all = np.concatenate(all_y)
    X_all = np.vstack(Xs)

    out = {"label": label, "n": len(y_all)}
    for mode in ["baseline", "zero", "median"]:
        Xm = apply_substitution(X_all, features, mode, medians)
        p = predict_clf(model, Xm)
        m = metrics_for(y_all, p)
        out[mode] = m
    return out


def main():
    print("Loading training-window medians for trojan features...", file=sys.stderr)
    medians = compute_training_medians(list(TROJAN_HARD))
    print("  Medians:", {k: f"{v:.4g}" for k, v in medians.items()}, file=sys.stderr)

    results = []

    print("\n[08c family — uses tabular + BTC features]", file=sys.stderr)
    for label, pkl, features, target_key, _kind, prep_fn in SPECS_08C:
        print(f"  {label} ...", end=" ", file=sys.stderr, flush=True)
        try:
            r = eval_model_clf(label, pkl, features, target_key, medians, prep_fn)
            print("done" if "baseline" in r else f"skipped ({r.get('skipped')})", file=sys.stderr)
        except Exception as e:
            r = {"label": label, "skipped": f"error: {e}"}
            print(f"ERROR {e}", file=sys.stderr)
        results.append(r)

    # NOTE: Models 02/03/04/04b/05/06/07/08 use different feature sets and target columns.
    # They need per-model target_key mappings (e.g. 02 predicts fair value, 03 predicts fill prob).
    # See `add_other_model_specs` TODO below to extend.
    # For now we emit a notice in the report — the 08c results already prove the methodology.

    # ── Report ────────────────────────────────────────────────────────────────
    lines = []
    lines.append("# Zero-Trojan OOS Re-Evaluation Report\n")
    lines.append("Compares OOS performance under three conditions: features as-is, trojans zeroed,\n")
    lines.append("trojans replaced with training-set median.  Shows how much each model's signal\n")
    lines.append("actually depends on the three contaminated environmental features.\n\n")
    lines.append(f"OOS window: {OOS_DATES[0]} .. {OOS_DATES[-1]}\n\n")
    lines.append("Trojan-HARD features and their training medians:\n")
    for f in TROJAN_HARD:
        lines.append(f"- `{f}` = {medians.get(f, 'n/a')}\n")
    lines.append("\n")

    lines.append("## 08c Family Results (49/58/32 feature variants)\n\n")
    lines.append("| Model | N | Base AUC | Zero AUC | Median AUC | ΔZero | ΔMedian | Verdict |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for r in results:
        if "skipped" in r:
            lines.append(f"| `{r['label']}` | — | — | — | — | — | — | SKIPPED: {r['skipped']} |\n")
            continue
        b = r["baseline"]
        z = r["zero"]
        m = r["median"]
        if not (b and z and m):
            lines.append(f"| `{r['label']}` | {r['n']:,} | — | — | — | — | — | metric NA |\n")
            continue
        dz = z["auc"] - b["auc"]
        dm = m["auc"] - b["auc"]
        # Classification
        if abs(dm) < 0.01 and abs(dz) < 0.02:
            verdict = "ROBUST"
        elif dm < -0.05:
            verdict = "TROJAN-DEPENDENT"
        elif dm > 0.02:
            verdict = "TROJANS HARMFUL"
        else:
            verdict = "MIXED"
        lines.append(
            f"| `{r['label']}` | {r['n']:,} | {b['auc']:.3f} | {z['auc']:.3f} | {m['auc']:.3f} | "
            f"{dz:+.3f} | {dm:+.3f} | **{verdict}** |\n"
        )
    lines.append("\n")

    lines.append("## Verdict Legend\n")
    lines.append("- **ROBUST**: AUC barely moves under either substitution — model signal is real. Safe to retrain with cleaned features.\n")
    lines.append("- **TROJAN-DEPENDENT**: AUC collapses without trojans — model has no real signal. Target likely cannot be predicted with available features.\n")
    lines.append("- **TROJANS HARMFUL**: AUC improves when trojans neutralized — trojans were actively misleading OOS. Definitely retrain without them.\n")
    lines.append("- **MIXED**: Moderate dependence — retraining without trojans will lose some performance but model isn't entirely fake.\n")
    lines.append("\n")

    lines.append("## TODO: Extend to Models 02/03/04/04b/05/06/07/08\n")
    lines.append("Those models use different feature sets and target columns drawn from different\n")
    lines.append("datasets (resolution_snapshot, execution_intent, etc.).  Each needs its own\n")
    lines.append("target_expr and dataset loader.  See `SPECS_OTHER` for the artifact list — the\n")
    lines.append("LightGBM models for these are also CRITICAL per the trojan audit.\n")
    lines.append("\n")
    lines.append(f"Discovered {len(SPECS_OTHER)} additional LightGBM models awaiting eval:\n")
    for label, pkl, feats in SPECS_OTHER:
        n_trojan = sum(1 for f in feats if f in TROJAN_HARD)
        lines.append(f"- `{label}` — {len(feats)} features, {n_trojan} trojans present\n")
    lines.append("\n")

    out = Path("docs/oos_eval_zero_trojan_report.md")
    out.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
