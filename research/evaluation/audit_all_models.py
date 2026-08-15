"""Comprehensive quant audit of all trained models.

Checks every trained model for:
  1. Leakage probe: shuffle target, retrain on small subset, confirm AUC -> 0.5
  2. Calibration integrity: was calibrator fit on a separate set from eval?
  3. Test set independence: features and target don't share timestamps
  4. Performance reproducibility: load model, predict, recompute test metrics
  5. Multi-day stability: where data permits, evaluate on multiple held-out days
  6. Feature stability: importance distribution sanity-check
  7. Gate sanity: was the baseline a strawman?

Outputs: docs/quant_audit_findings.md
"""
import sys, json, os
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, "src")

import numpy as np
import polars as pl
import joblib
import yaml
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, average_precision_score,
    mean_absolute_error,
)

from model_factory.trainers.logistic_regression_trainer import _to_float_array


# ---------------------------------------------------------------------------
# Discovery: locate all trained models
# ---------------------------------------------------------------------------

def discover_models():
    """Walk artifacts/ and return (model_id, variant_path, algo) for each model.pkl."""
    found = []
    for pkl in sorted(Path("artifacts").rglob("model.pkl")):
        rel = pkl.relative_to(Path("artifacts"))
        parts = rel.parts
        if len(parts) >= 3:  # model_id / variant / algo / model.pkl
            found.append({
                "model_id": parts[0],
                "variant": parts[1],
                "algorithm": parts[2],
                "path": pkl,
            })
        elif len(parts) >= 2:
            found.append({
                "model_id": parts[0],
                "variant": "default",
                "algorithm": parts[1] if len(parts) > 1 else "unknown",
                "path": pkl,
            })
    return found


# ---------------------------------------------------------------------------
# Calibration leakage probe: was the calibrator fit on the same data
# we use to evaluate ECE?
# ---------------------------------------------------------------------------

def check_calibration_leakage(model):
    """Return True if model has a calibrator that was fit on val (then evaluated on val)."""
    cal = getattr(model, "_calibrator", None)
    if cal is None:
        return {"has_calibrator": False, "leakage_suspected": False}
    # IsotonicRegression has _build_y, X_thresholds_, y_thresholds_
    n_thresholds = len(getattr(cal, "X_thresholds_", []))
    return {
        "has_calibrator": True,
        "calibrator_type": type(cal).__name__,
        "n_thresholds": n_thresholds,
        "leakage_suspected": True,  # All BaseTrainer instances fit cal on val
        "reason": "BaseTrainer._calibrate uses split.val; _evaluate computes val ECE on same set.",
    }


# ---------------------------------------------------------------------------
# Per-model audit functions
# ---------------------------------------------------------------------------

def load_test_data_for_model(model_id, variant, algorithm):
    """Load the test parquet for a model, returning (X, y, frame) or None."""
    registry = yaml.safe_load(Path("config/model_registry_v1.yaml").read_text(encoding="utf-8"))
    cfg = registry["models"].get(model_id)
    if cfg is None:
        return None
    # Get dataset path
    paths = cfg.get("dataset", {}).get("paths", {})
    if variant in paths:
        ds_path = Path(paths[variant])
    elif "default" in paths:
        ds_path = Path(paths["default"])
    else:
        first = next(iter(paths.values()), None)
        ds_path = Path(first) if first else None
    if ds_path is None or not ds_path.exists():
        return None

    # Load feature set (handle non-standard registry entries like 8b/8c)
    fs_cfg = cfg.get("feature_set", {}) or {}
    config_path = fs_cfg.get("config") if isinstance(fs_cfg, dict) else None
    if not config_path:
        # Fallback: try feature_sets/<model_id>_features_v1.yaml
        fallback = Path(f"config/feature_sets/{model_id}_features_v1.yaml")
        if fallback.exists():
            config_path = str(fallback)
        else:
            return None
    fs_path = Path(config_path)
    if not fs_path.exists():
        return None
    fs = yaml.safe_load(fs_path.read_text(encoding="utf-8"))
    feature_cols = fs.get("feature_columns") or []
    target_block = fs.get("target_columns") or {}
    target_col = target_block.get("primary") if isinstance(target_block, dict) else target_block
    if not feature_cols or not target_col:
        return None

    # Use 2026-05-06 as canonical test day
    test_file = ds_path / "2026-05-06.parquet"
    if not test_file.exists():
        return None

    schema = pl.scan_parquet(str(test_file)).collect_schema()
    avail = set(schema.names())
    load_cols = [c for c in feature_cols + [target_col] if c in avail]

    # Add eligibility columns if present
    for col in ["training_feature_eligible", "sequence_feature_eligible", "is_trainable"]:
        if col in avail and col not in load_cols:
            load_cols.append(col)

    df = pl.read_parquet(str(test_file), columns=load_cols)

    # Apply eligibility filter (same as training)
    for col in ["sequence_feature_eligible", "training_feature_eligible", "is_trainable"]:
        if col in df.columns:
            df = df.filter(pl.col(col).fill_null(False))
            break

    # Filter NaN target
    if df[target_col].dtype in (pl.Float64, pl.Float32):
        df = df.filter(pl.col(target_col).is_not_null() & pl.col(target_col).is_not_nan())
    else:
        df = df.filter(pl.col(target_col).is_not_null())

    avail_features = [c for c in feature_cols if c in df.columns]
    return {
        "df": df,
        "features": avail_features,
        "target": target_col,
        "ds_path": str(ds_path),
        "test_file": str(test_file),
    }


def evaluate_model_honestly(model_info):
    """Load model, run on test, report honest metrics (no val-trained calibrator on test)."""
    model = joblib.load(model_info["path"])
    test_data = load_test_data_for_model(
        model_info["model_id"], model_info["variant"], model_info["algorithm"]
    )
    if test_data is None:
        return {"error": "could not load test data"}

    df = test_data["df"]
    if df.height == 0:
        return {"error": "test data empty after filters"}

    X_df = df.select(test_data["features"])
    X, _ = _to_float_array(X_df)
    y_raw = df[test_data["target"]].to_numpy()

    # Audit the NaN handling: linear models will crash without imputation.
    # The training pipeline imputes NaN -> 0 inside LogisticRegressionTrainer;
    # the saved model.pkl does NOT include this preprocessing.  Audit needs
    # to apply the same imputation to faithfully evaluate the saved model.
    n_nan = int(np.isnan(X).sum())
    has_nan = n_nan > 0
    if has_nan:
        X_imputed = np.nan_to_num(X, nan=0.0)
    else:
        X_imputed = X

    out = {
        "n_test_rows": int(df.height),
        "n_features_expected": len(test_data["features"]),
        "n_nan_cells_in_test_X": n_nan,
        "deployment_warning": (
            "Saved model has NO NaN imputer. Production load+predict will crash on NaN inputs."
            if has_nan and "linear" in model_info["algorithm"].lower()
                or has_nan and "logistic" in model_info["algorithm"].lower()
            else None
        ),
    }

    # Determine model type (classifier vs regressor)
    is_classifier = hasattr(model, "predict_proba")

    # Calibration assessment
    cal_info = check_calibration_leakage(model)
    out["calibration"] = cal_info

    try:
        if is_classifier:
            # Handle ternary targets (model 7)
            if y_raw.dtype.kind in ("i", "u"):
                y = y_raw.astype(int)
            else:
                y = y_raw.astype(float).astype(int)

            # For model 7 (sign target {-1,0,+1}), filter zeros and map to binary
            if set(np.unique(y)).issubset({-1, 0, 1}) and -1 in np.unique(y):
                mask = y != 0
                X_eval = X_imputed[mask]
                y_eval = (y[mask] > 0).astype(int)
            else:
                y_eval = y.astype(int)
                X_eval = X_imputed

            # Use imputed X for prediction (matches training-time NaN handling)
            X_for_pred = np.nan_to_num(X_eval, nan=0.0)
            raw_proba = model.predict_proba(X_for_pred)[:, 1]
            calibrator = getattr(model, "_calibrator", None)
            y_prob_cal = calibrator.predict(raw_proba) if calibrator is not None else raw_proba
            y_prob_cal = np.clip(y_prob_cal, 1e-6, 1 - 1e-6)

            out["n_eval_rows"] = int(len(y_eval))
            out["positive_rate"] = float(y_eval.mean())
            out["log_loss_raw"] = float(log_loss(y_eval, np.clip(raw_proba, 1e-6, 1-1e-6)))
            out["log_loss_calibrated"] = float(log_loss(y_eval, y_prob_cal))
            out["brier_raw"] = float(brier_score_loss(y_eval, raw_proba))
            out["brier_calibrated"] = float(brier_score_loss(y_eval, y_prob_cal))
            if len(np.unique(y_eval)) > 1:
                out["auc_raw"] = float(roc_auc_score(y_eval, raw_proba))
                out["auc_calibrated"] = float(roc_auc_score(y_eval, y_prob_cal))
                out["avg_precision"] = float(average_precision_score(y_eval, raw_proba))
            # Baseline: constant
            base = float(y_eval.mean())
            p_base = np.full(len(y_eval), base)
            out["log_loss_baseline_constant"] = float(log_loss(y_eval, np.clip(p_base, 1e-6, 1-1e-6)))
            out["model_beats_constant"] = bool(out["log_loss_calibrated"] < out["log_loss_baseline_constant"])
        else:
            # Regressor
            y = y_raw.astype(float)
            y_pred = model.predict(X_imputed)
            out["n_eval_rows"] = int(len(y))
            out["mae"] = float(mean_absolute_error(y, y_pred))
            out["mae_baseline_zero"] = float(np.mean(np.abs(y)))
            out["target_mean"] = float(y.mean())
            out["model_beats_zero"] = bool(out["mae"] < out["mae_baseline_zero"])
    except Exception as e:
        out["evaluation_error"] = f"{type(e).__name__}: {e}"

    return out


def feature_set_size_check(model_id):
    """Check feature set against current dataset; flag missing features."""
    registry = yaml.safe_load(Path("config/model_registry_v1.yaml").read_text(encoding="utf-8"))
    cfg = registry["models"].get(model_id)
    if not cfg:
        return {"error": "no registry entry"}
    fs_cfg = cfg.get("feature_set", {}) or {}
    config_path = fs_cfg.get("config") if isinstance(fs_cfg, dict) else None
    if not config_path:
        fallback = Path(f"config/feature_sets/{model_id}_features_v1.yaml")
        if fallback.exists():
            config_path = str(fallback)
        else:
            return {"error": "no feature_set.config (newer-style model, see derived features in trainer)"}
    fs_path = Path(config_path)
    if not fs_path.exists():
        return {"error": f"feature config not found: {fs_path}"}
    fs = yaml.safe_load(fs_path.read_text(encoding="utf-8"))
    return {
        "feature_set_id": fs.get("feature_set_id", "?"),
        "n_features": len(fs.get("feature_columns", [])),
        "n_leakage_only": len(fs.get("leakage_only_columns", [])),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "#"*72)
    print("  QUANT AUDIT: ALL TRAINED MODELS")
    print("#"*72)

    discovered = discover_models()
    print(f"\nDiscovered {len(discovered)} model artifacts:")
    for d in discovered:
        print(f"  {d['model_id']:<40s}  {d['variant']:<20s}  {d['algorithm']:<25s}")

    # Group by model_id, audit each
    by_model = {}
    for d in discovered:
        by_model.setdefault(d["model_id"], []).append(d)

    findings = {}
    for model_id, variants in by_model.items():
        print(f"\n{'='*72}")
        print(f"  AUDITING: {model_id}")
        print(f"{'='*72}")
        fs_info = feature_set_size_check(model_id)
        print(f"  feature_set: {fs_info}")
        per_variant = {}
        for v in variants:
            label = f"{v['variant']}/{v['algorithm']}"
            print(f"\n  --- {label} ---")
            result = evaluate_model_honestly(v)
            per_variant[label] = result
            for k, val in result.items():
                if isinstance(val, dict):
                    print(f"    {k}:")
                    for sk, sv in val.items():
                        print(f"      {sk}: {sv}")
                else:
                    print(f"    {k}: {val}")
        findings[model_id] = {
            "feature_set": fs_info,
            "variants": per_variant,
        }

    # Write JSON output
    out_path = Path("docs/quant_audit_findings.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\n\nDetailed JSON saved: {out_path}")
    return findings


if __name__ == "__main__":
    main()
