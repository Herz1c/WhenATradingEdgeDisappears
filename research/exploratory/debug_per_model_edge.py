#!/usr/bin/env python3
"""Per-model honest-edge scorecard.

For each of 9 models, load its NATIVE dataset (the one with the features it
was trained on), run inference on the proper feature matrix, and compare
predictions to ground truth.

Output: a single table with calibration / discrimination metrics + a
"trading-edge" column where applicable (per-fill realized PnL when the
model's output is used as a binary gate).
"""
from __future__ import annotations

import io, json, sys, math
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error, brier_score_loss

# evt_* feature builder for Model 08c
sys.path.insert(0, ".")
from train_maker_defense_v3_eventtime_cleaned import build_event_features, get_event_feature_names

from feature_cleanup import clean_features


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_dataset(name, dates, max_per_day=20_000, seed=42, filled_only=False):
    base = {
        "resolution_dense_close": "data/datasets/resolution_snapshot_dataset_v1_dense_close",
        "resolution_coarse":      "data/datasets/resolution_snapshot_dataset_v1_coarse",
        "execution_intent":       "data/datasets/execution_intent_dataset_v1",
        "microstr_tabular":       "data/datasets/microstructure_sequence_dataset_v1_tabular",
        "microstr_event64":       "data/datasets/microstructure_sequence_dataset_v1_event64",
    }[name]
    parts = []
    for d in dates:
        p = Path(f"{base}/{d}.parquet")
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        if filled_only and "filled" in df.columns:
            df = df.filter(pl.col("filled") == True)
        if max_per_day and len(df) > max_per_day:
            df = df.sample(n=max_per_day, seed=seed, with_replacement=False)
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal")


# ── Model loader ─────────────────────────────────────────────────────────────

def load_model(artifact_path):
    p = Path(artifact_path)
    if not p.exists():
        return None, []
    model = joblib.load(p)
    fi_path = p.parent / "feature_importance.json"
    if fi_path.exists():
        feats = list(json.loads(fi_path.read_text()).keys())
    else:
        # Model 08c uses metrics.json
        m_path = p.parent / "metrics.json"
        if m_path.exists():
            feats = list(json.loads(m_path.read_text()).get("feature_importance", {}).keys())
        else:
            feats = []
    return model, feats


def build_X(df: pl.DataFrame, feat_names: list[str]) -> np.ndarray:
    n = len(df)
    X = np.zeros((n, len(feat_names)), dtype=np.float32)
    cols = set(df.columns)
    for i, f in enumerate(feat_names):
        if f not in cols:
            continue
        s = df.get_column(f)
        if not s.dtype.is_numeric():
            continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    return X


def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)[:, 1]
        cal = getattr(model, "_calibrator", None)
        if cal is not None:
            raw = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
        return raw
    return model.predict(X)


def predict_scalar(model, X):
    return model.predict(X)


# ── Metric helpers ───────────────────────────────────────────────────────────

def bin_quality(y_true, y_pred_proba, label):
    """AUC + log loss + Brier."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred_proba).clip(1e-6, 1 - 1e-6)
    mask = np.isfinite(y_pred) & np.isin(y_true, [0, 1])
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 100 or len(set(y_true)) < 2:
        return {"label": label, "n": int(len(y_true)), "error": "insufficient data"}
    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    # Naive baseline log-loss (constant = base rate)
    base = y_true.mean()
    ll_baseline = log_loss(y_true, np.full_like(y_pred, base))
    skill = (ll_baseline - ll) / ll_baseline
    return {
        "label": label, "n": int(len(y_true)),
        "auc": float(auc), "log_loss": float(ll),
        "log_loss_baseline": float(ll_baseline),
        "log_loss_skill": float(skill),    # >0 means better than constant
        "brier": float(brier),
        "pos_rate": float(base),
        "pred_mean": float(y_pred.mean()),
    }


def reg_quality(y_true, y_pred, label):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 100:
        return {"label": label, "n": int(len(y_true)), "error": "insufficient data"}
    mae = mean_absolute_error(y_true, y_pred)
    mae_baseline_zero = mean_absolute_error(y_true, np.zeros_like(y_true))
    mae_baseline_mean = mean_absolute_error(y_true, np.full_like(y_true, y_true.mean()))
    skill_vs_zero = (mae_baseline_zero - mae) / mae_baseline_zero if mae_baseline_zero else 0.0
    skill_vs_mean = (mae_baseline_mean - mae) / mae_baseline_mean if mae_baseline_mean else 0.0
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() > 0 and y_pred.std() > 0 else 0.0
    sign_match = float((np.sign(y_pred) == np.sign(y_true)).mean())
    return {
        "label": label, "n": int(len(y_true)),
        "mae": float(mae),
        "mae_baseline_predict_zero": float(mae_baseline_zero),
        "mae_baseline_predict_mean": float(mae_baseline_mean),
        "skill_vs_zero": float(skill_vs_zero),
        "skill_vs_mean": float(skill_vs_mean),
        "corr": corr, "sign_match": sign_match,
        "y_mean": float(y_true.mean()), "yhat_mean": float(y_pred.mean()),
    }


def trading_edge(y_true_pnl, predictions, threshold_pairs, label):
    """For each (lo, hi) threshold, compute mean realized PnL of accepted rows."""
    out = []
    y_true_pnl = np.asarray(y_true_pnl, dtype=float)
    p = np.asarray(predictions, dtype=float)
    mask = np.isfinite(y_true_pnl) & np.isfinite(p)
    y, p = y_true_pnl[mask], p[mask]
    for lo, hi in threshold_pairs:
        m = (p > lo) & (p <= hi)
        if m.sum() < 50:
            continue
        out.append({
            "lo": lo, "hi": hi, "n": int(m.sum()),
            "mean_pnl": float(y[m].mean()),
            "median_pnl": float(np.median(y[m])),
        })
    return {"label": label, "buckets": out}


# ── Per-model scoring ───────────────────────────────────────────────────────

OOS_DATES = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10",
             "2026-05-11","2026-05-12","2026-05-13","2026-05-14",
             "2026-05-15","2026-05-16"]

report = {}


def score_model_02_fair_resolution(variant):
    """Model 02: P(resolves UP). Native dataset: resolution_*_close."""
    ds_name = f"resolution_{variant}"   # 'dense_close' or 'coarse'
    art = f"artifacts_cleaned/model_02_fair_resolution/{variant}/lightgbm/model.pkl"
    model, feats = load_model(art)
    if not model:
        return {"error": f"missing {art}"}
    df = load_dataset(ds_name, OOS_DATES, max_per_day=10_000)
    if "resolved_side_label" not in df.columns:
        return {"error": "no resolved_side_label"}
    df = clean_features(df)
    X = build_X(df, feats)
    yhat = predict_proba(model, X)
    y = df["resolved_side_label"].to_numpy()
    q = bin_quality(y, yhat, label=f"Model 02 fair_resolution ({variant})")
    return q


def score_model_05_closing_flip():
    art = "artifacts_cleaned/model_05_closing_flip/dense_close/lightgbm/model.pkl"
    model, feats = load_model(art)
    df = load_dataset("resolution_dense_close", OOS_DATES, max_per_day=10_000)
    if "flip_happened_after_t" not in df.columns:
        return {"error": "no flip_happened_after_t"}
    df = clean_features(df)
    X = build_X(df, feats)
    yhat = predict_proba(model, X)
    y = df["flip_happened_after_t"].cast(pl.Int8).to_numpy()
    return bin_quality(y, yhat, label="Model 05 closing_flip")


def score_model_06_mispricing():
    """Model 06: regress mispricing (predicted vs realized hold_to_close_edge_vs_mid)."""
    art = "artifacts_cleaned/model_06_mispricing/dense_close/lightgbm/model.pkl"
    model, feats = load_model(art)
    df = load_dataset("resolution_dense_close", OOS_DATES, max_per_day=10_000)
    df = clean_features(df)
    if "hold_to_close_edge_vs_mid" not in df.columns:
        return {"error": "no hold_to_close_edge_vs_mid"}
    df = df.filter(pl.col("hold_to_close_edge_vs_mid").is_not_null() & pl.col("hold_to_close_edge_vs_mid").is_not_nan())
    X = build_X(df, feats)
    yhat = predict_scalar(model, X)
    y = df["hold_to_close_edge_vs_mid"].to_numpy()
    return reg_quality(y, yhat, label="Model 06 mispricing")


def score_model_03_fill_prob():
    art = "artifacts_cleaned/model_03_fill_probability/lightgbm/model.pkl"
    model, feats = load_model(art)
    df = load_dataset("execution_intent", OOS_DATES, max_per_day=20_000)
    df = clean_features(df)
    X = build_X(df, feats)
    yhat = predict_proba(model, X)
    y = df["filled"].cast(pl.Int8).to_numpy()
    return bin_quality(y, yhat, label="Model 03 fill_probability")


def score_model_04_adverse_selection():
    art = "artifacts_cleaned/model_04_adverse_selection/lightgbm/model.pkl"
    model, feats = load_model(art)
    df = load_dataset("execution_intent", OOS_DATES, max_per_day=20_000, filled_only=True)
    df = clean_features(df)
    df = df.filter(pl.col("markout_1s").is_not_null() & pl.col("markout_1s").is_not_nan())
    X = build_X(df, feats)
    yhat = predict_scalar(model, X)
    y = df["markout_1s"].to_numpy()
    return reg_quality(y, yhat, label="Model 04 adverse_selection (markout_1s)")


def score_model_04b_mkt_to_close():
    art = "artifacts_cleaned/model_04b_markout_to_close/lightgbm/model.pkl"
    model, feats = load_model(art)
    df = load_dataset("execution_intent", OOS_DATES, max_per_day=20_000, filled_only=True)
    df = clean_features(df)
    df = df.filter(pl.col("markout_to_close").is_not_null() & pl.col("markout_to_close").is_not_nan())
    X = build_X(df, feats)
    yhat = predict_scalar(model, X)
    y = df["markout_to_close"].to_numpy()
    reg = reg_quality(y, yhat, label="Model 04b markout_to_close")

    # Trading edge: per-fill realized PnL conditioned on signed-prediction band
    side = df["order_side"].to_numpy()
    direction = np.where(side == "buy", 1, -1)
    signed_real = y * direction
    signed_pred = yhat * direction
    bands = [(-1e9, -0.20), (-0.20, -0.10), (-0.10, -0.05), (-0.05, -0.01),
             (-0.01, +0.01), (+0.01, +0.05), (+0.05, +0.10), (+0.10, +0.20), (+0.20, +1e9)]
    te = trading_edge(signed_real, signed_pred, bands, "signed-pred bucket")
    reg["trading_edge"] = te
    return reg


def score_model_07_direction():
    art = "artifacts_cleaned/model_07_microstructure_direction/tabular/lightgbm/model.pkl"
    model, feats = load_model(art)
    df = load_dataset("microstr_tabular", OOS_DATES, max_per_day=20_000)
    df = clean_features(df)
    if "mid_move_sign_5s" not in df.columns:
        return {"error": "no mid_move_sign_5s"}
    # mid_move_sign_5s is -1/0/+1; convert to binary up vs not-up
    df = df.filter(pl.col("mid_move_5s").is_not_null() & pl.col("mid_move_5s").is_not_nan())
    X = build_X(df, feats)
    yhat = predict_proba(model, X)
    sign = df["mid_move_sign_5s"].cast(pl.Int8).to_numpy()
    y = (sign > 0).astype(int)   # P(up)
    return bin_quality(y, yhat, label="Model 07 microstructure_direction (P up_5s)")


def score_model_08c(level):
    """Model 08c: P(severe/moderate/extreme move) — needs evt_* features."""
    art = f"artifacts_cleaned/model_08c_maker_defense_v3_eventtime/{level}/agg_plus_event/model.pkl"
    model, feats = load_model(art)
    if not model:
        return {"error": f"missing {art}"}
    df = load_dataset("microstr_event64", OOS_DATES, max_per_day=8_000)
    if len(df) == 0:
        return {"error": "no event64 data"}
    df = build_event_features(df)   # adds evt_* columns
    df = clean_features(df)

    # Build the target. 08c targets are configured per level:
    # severe_7c_3s   : abs(mid_move_3s) >= 0.07
    # moderate_5c_5s : abs(mid_move_5s) >= 0.05
    # extreme_10c_5s : abs(mid_move_5s) >= 0.10
    targets = {
        "severe_7c_3s":   ("mid_move_3s", 0.07),
        "moderate_5c_5s": ("mid_move_5s", 0.05),
        "extreme_10c_5s": ("mid_move_5s", 0.10),
    }
    move_col, thr = targets[level]
    df = df.filter(pl.col(move_col).is_not_null() & pl.col(move_col).is_not_nan())
    y = (df[move_col].to_numpy().__abs__() >= thr).astype(int)
    X = build_X(df, feats)
    yhat = predict_proba(model, X)
    return bin_quality(y, yhat, label=f"Model 08c {level}")


# ── Run all ────────────────────────────────────────────────────────────────

print("=" * 78)
print("PER-MODEL HONEST-EDGE SCORECARD")
print("Dates: 2026-05-07 .. 2026-05-16 (pure post-training OOS)")
print("Each model scored on its NATIVE dataset with its trained feature set.")
print("=" * 78)

print("\nScoring Model 02 (dense_close) ...")
report["model_02_fair_dense_close"] = score_model_02_fair_resolution("dense_close")
print("Scoring Model 02 (coarse) ...")
report["model_02_fair_coarse"] = score_model_02_fair_resolution("coarse")
print("Scoring Model 05 (closing_flip) ...")
report["model_05_closing_flip"] = score_model_05_closing_flip()
print("Scoring Model 06 (mispricing) ...")
report["model_06_mispricing"] = score_model_06_mispricing()
print("Scoring Model 03 (fill_prob) ...")
report["model_03_fill_prob"] = score_model_03_fill_prob()
print("Scoring Model 04 (adverse_sel) ...")
report["model_04_adverse_selection"] = score_model_04_adverse_selection()
print("Scoring Model 04b (markout_to_close) ...")
report["model_04b_mkt_to_close"] = score_model_04b_mkt_to_close()
print("Scoring Model 07 (direction) ...")
report["model_07_direction"] = score_model_07_direction()
print("Scoring Model 08c (severe) ...")
report["model_08c_severe"] = score_model_08c("severe_7c_3s")
print("Scoring Model 08c (moderate) ...")
report["model_08c_moderate"] = score_model_08c("moderate_5c_5s")
print("Scoring Model 08c (extreme) ...")
report["model_08c_extreme"] = score_model_08c("extreme_10c_5s")

# Save full
Path("docs").mkdir(exist_ok=True)
Path("docs/per_model_edge.json").write_text(json.dumps(report, indent=2, default=str))


# ── Print summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("SUMMARY — discrimination metrics")
print("=" * 100)
print(f"{'model':<46} {'kind':<8} {'n':>8} {'AUC':>7} {'logloss skill':>15} {'pos rate':>10}")
print("-" * 100)
for key, r in report.items():
    if "error" in r:
        print(f"{key:<46}  ERROR: {r['error']}")
        continue
    if "auc" in r:
        print(f"{r['label']:<46} {'binary':<8} {r['n']:>8} {r['auc']:>7.3f} {r['log_loss_skill']:>14.1%} {r['pos_rate']:>9.1%}")
    elif "mae" in r:
        print(f"{r['label']:<46} {'regr':<8} {r['n']:>8} {'':<7} {'':<15} {'':<10}")
        print(f"{'  ':<46} {'':<8} {'':<8} corr={r['corr']:+.3f}  sign_match={r['sign_match']:.1%}  MAE={r['mae']:.4f} vs naive {r['mae_baseline_predict_zero']:.4f} (skill {r['skill_vs_zero']:+.1%})")

print("\n=== TRADING EDGE BY MODEL 04b SIGNED-PREDICTION BAND (OOS only) ===")
te = report.get("model_04b_mkt_to_close", {}).get("trading_edge", {})
for b in te.get("buckets", []):
    print(f"  signed_pred ({b['lo']:+.2f}, {b['hi']:+.2f}]: n={b['n']:>6}  mean_pnl=${b['mean_pnl']:+7.4f}  median=${b['median_pnl']:+7.4f}")

print(f"\nFull report → docs/per_model_edge.json")
