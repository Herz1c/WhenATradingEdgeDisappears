from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import yaml
from sklearn.isotonic import IsotonicRegression

from model_factory.artifact_writer import ArtifactWriter
from model_factory.dataset_loader import DatasetSplit, LeakageSafeLoader
from model_factory.evaluation.metrics import (
    compute_brier_score,
    compute_ece,
    compute_edge_by_time_to_close,
    compute_edge_by_session,
    compute_stability_by_phase,
    compute_log_loss,
)


class BaseTrainer(ABC):
    """
    Abstract base for all model trainers.

    Enforces:
    1. Dataset loaded through LeakageSafeLoader.
    2. Experiment manifest written before training begins.
    3. Model artifacts written through ArtifactWriter.
    4. No model is trained without dataset hash recorded.
    5. Calibration hook after training.
    6. Eval report generated after calibration.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.loader = LeakageSafeLoader()
        self.writer = ArtifactWriter()

    @abstractmethod
    def _build_model(self, hyperparams: dict[str, Any]):
        """Instantiate the model with given hyperparams."""

    @abstractmethod
    def _fit(self, model: Any, X_train: Any, y_train: Any) -> None:
        """Fit the model. No return."""

    @abstractmethod
    def _predict_proba(self, model: Any, X: Any) -> np.ndarray:
        """Return probability predictions. Shape: (n_samples,) for binary."""

    def train(
        self,
        dataset_variant: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        target_override: str | None = None,
    ) -> dict[str, Any]:
        # Merge subclass DEFAULT_HYPERPARAMS with any caller-supplied overrides
        defaults = getattr(self.__class__, "DEFAULT_HYPERPARAMS", {})
        hyperparams = {**defaults, **(hyperparams or {})}
        split = self.loader.load(
            self.model_id,
            dataset_variant=dataset_variant,
            target_override=target_override,
            verbose=True,
        )
        artifact_path = self._get_artifact_path(dataset_variant=dataset_variant)
        manifest = self.writer.write_experiment_manifest(
            model_id=self.model_id,
            dataset_split=split,
            algorithm=self.__class__.__name__,
            hyperparams=hyperparams,
            artifact_path=artifact_path,
        )
        model = self._build_model(hyperparams)
        X_train = split.train.drop(split.target_column)
        y_train = split.train[split.target_column]
        self._fit(model, X_train, y_train)
        calibrated_model = self._calibrate(model, split)
        metrics = self._evaluate(calibrated_model, split)
        self._save_artifacts(calibrated_model, split, metrics, artifact_path)
        self.writer.append_experiment_log(
            artifact_path=artifact_path,
            event_type="training_complete",
            data={"metrics": metrics, "manifest_hash": manifest["dataset_provenance"]["dataset_hash"][:16]},
        )
        return metrics

    # AUDIT FIX (2026-05-12):
    # The previous calibration logic fitted IsotonicRegression on split.val
    # and was then evaluated on split.val (val ECE always = 0 artifact),
    # AND on small val sets it overfit so badly that test log_loss got worse
    # after calibration (verified on Model 01).
    #
    # New policy:
    #   1. Skip calibration entirely if val < MIN_VAL_FOR_CAL rows OR
    #      either class has < MIN_POS_FOR_CAL examples.  Predictions will
    #      pass through raw -- documented in metrics.
    #   2. Otherwise: split val 50/50 into cal_set and eval_set; fit
    #      isotonic on cal_set, evaluate val metrics on eval_set.  This
    #      eliminates val-on-val leakage and gives an honest val ECE.
    #   3. Always log the calibration decision into model._calibration_meta
    MIN_VAL_FOR_CAL = 1000
    MIN_POS_FOR_CAL = 100

    def _calibrate(self, model: Any, split: DatasetSplit) -> Any:
        """Isotonic regression post-hoc calibration with held-out cal split."""
        if split.val.is_empty():
            model._calibration_meta = {"calibrated": False, "reason": "empty_val"}
            return model

        y_val_full = split.val[split.target_column].to_numpy().astype(int)
        n_val = len(y_val_full)
        n_pos = int(y_val_full.sum())
        n_neg = n_val - n_pos

        # Guardrail: refuse to calibrate on small val
        if n_val < self.MIN_VAL_FOR_CAL or min(n_pos, n_neg) < self.MIN_POS_FOR_CAL:
            model._calibration_meta = {
                "calibrated": False,
                "reason": f"val too small (n={n_val}, n_pos={n_pos}, n_neg={n_neg}); "
                          f"required n>={self.MIN_VAL_FOR_CAL}, min_class>={self.MIN_POS_FOR_CAL}",
            }
            return model

        # 50/50 split of val into cal_set (fit calibrator) and eval_set (kept for honest val ECE)
        X_val = split.val.drop(split.target_column)
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(n_val)
        cal_idx = perm[: n_val // 2]
        cal_mask = np.zeros(n_val, dtype=bool)
        cal_mask[cal_idx] = True

        raw_probs_val = self._predict_proba(model, X_val)
        raw_probs_cal = raw_probs_val[cal_mask]
        y_cal = y_val_full[cal_mask]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs_cal, y_cal)
        model._calibrator = iso
        model._calibration_meta = {
            "calibrated": True,
            "calibration_set": "val_random_half",
            "n_cal": int(cal_mask.sum()),
            "n_val_eval": int((~cal_mask).sum()),
            "n_pos_cal": int(y_cal.sum()),
            "n_thresholds": int(len(getattr(iso, "X_thresholds_", []))),
        }
        # Stash the eval-set mask so _evaluate can report honest val metrics
        model._val_eval_mask = ~cal_mask
        return model

    # AUDIT FIX (2026-05-12): chunk-aware predict for memory bounds.
    # When the eval set is huge (millions of rows), feeding it whole to
    # the model can OOM via sklearn pipeline transform copies.  We chunk
    # the input frame into CHUNK_SIZE rows and concatenate predictions.
    PREDICT_CHUNK_SIZE = 1_000_000

    def _calibrated_predict(self, model: Any, X: Any) -> np.ndarray:
        # If X is a polars DataFrame larger than CHUNK_SIZE, chunk it.
        if isinstance(X, pl.DataFrame) and X.height > self.PREDICT_CHUNK_SIZE:
            chunks = []
            for start in range(0, X.height, self.PREDICT_CHUNK_SIZE):
                end = min(start + self.PREDICT_CHUNK_SIZE, X.height)
                chunk = X[start:end]
                raw = self._predict_proba(model, chunk)
                iso = getattr(model, "_calibrator", None)
                out = iso.predict(raw) if iso is not None else raw
                chunks.append(np.asarray(out, dtype=np.float32))
            return np.concatenate(chunks)
        raw = self._predict_proba(model, X)
        iso = getattr(model, "_calibrator", None)
        return iso.predict(raw) if iso is not None else raw

    def _evaluate(self, model: Any, split: DatasetSplit) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        # Compute baselines once from train set
        train_rate = float(split.train[split.target_column].cast(pl.Float64).mean())
        baselines = self._compute_baselines(split)

        # AUDIT FIX (2026-05-12): record calibration decision in metrics
        cal_meta = getattr(model, "_calibration_meta", None)
        if cal_meta is not None:
            metrics["_calibration_meta"] = cal_meta

        val_eval_mask = getattr(model, "_val_eval_mask", None)

        for name, frame in {"train": split.train, "val": split.val, "test": split.test}.items():
            if frame.is_empty():
                continue
            y_true = frame[split.target_column].to_numpy().astype(int)
            X = frame.drop(split.target_column)
            y_prob = self._calibrated_predict(model, X)

            # AUDIT FIX: if calibrator was fit on a random half of val, evaluate
            # val metrics only on the held-out half so val ECE is honest.
            # All stratification arrays must use the SAME mask to stay aligned.
            row_mask = None
            if name == "val" and val_eval_mask is not None and len(val_eval_mask) == len(y_true):
                row_mask = val_eval_mask
                y_true = y_true[row_mask]
                y_prob = y_prob[row_mask]
            ll = compute_log_loss(y_true, y_prob)
            bs = compute_brier_score(y_true, y_prob)
            ece = compute_ece(y_true, y_prob)
            entry: dict[str, Any] = {
                "log_loss": ll,
                "brier_score": bs,
                "ece": ece,
                "n_rows": len(y_true),
                "target_mean": float(y_true.mean()),
            }
            # Compare against baselines on val/test
            if name in ("val", "test") and name in baselines:
                entry["baselines"] = baselines[name]
                for bl_name, bl_metrics in baselines[name].items():
                    entry[f"vs_{bl_name}_log_loss_delta"] = ll - bl_metrics["log_loss"]

            # Stratified evaluation -- apply row_mask consistently to all arrays
            def _aligned(col_data):
                return col_data[row_mask] if row_mask is not None else col_data

            if "t_to_close_s" in frame.columns:
                t = _aligned(frame["t_to_close_s"].to_numpy().astype(float))
                entry["edge_by_time_to_close"] = compute_edge_by_time_to_close(y_true, y_prob, t)
            if "phase_bucket" in frame.columns:
                phase = _aligned(frame["phase_bucket"].to_numpy())
                entry["log_loss_by_phase"] = compute_stability_by_phase(
                    y_true, y_prob, phase, compute_log_loss
                )
            if "snapshot_hour_utc" in frame.columns:
                hour = _aligned(frame["snapshot_hour_utc"].to_numpy())
                entry["brier_by_session"] = compute_edge_by_session(y_true, y_prob, hour)

            metrics[name] = entry

        return metrics

    def _compute_baselines(self, split: DatasetSplit) -> dict[str, dict[str, Any]]:
        """Compute constant-up, streak, and delta-to-strike baseline metrics."""
        train_rate = float(split.train[split.target_column].cast(pl.Float64).mean())
        result: dict[str, dict[str, Any]] = {}

        for name, frame in {"val": split.val, "test": split.test}.items():
            if frame.is_empty():
                continue
            y_true = frame[split.target_column].to_numpy().astype(int)
            n = len(y_true)
            entry: dict[str, Any] = {}

            # Baseline 1: constant train base rate
            p_const = np.full(n, train_rate)
            entry["constant_base_rate"] = {
                "log_loss": compute_log_loss(y_true, p_const),
                "brier_score": compute_brier_score(y_true, p_const),
            }

            # Baseline 2: always predict UP (p=0.99)
            p_always_up = np.full(n, 0.99)
            entry["always_up"] = {
                "log_loss": compute_log_loss(y_true, p_always_up),
                "brier_score": compute_brier_score(y_true, p_always_up),
            }

            # Baseline 3: streak-based (resolved_up_ratio_last_12 if present)
            if "resolved_up_ratio_last_12" in frame.columns:
                p_streak = frame["resolved_up_ratio_last_12"].fill_null(train_rate).to_numpy()
                p_streak = np.clip(p_streak, 0.01, 0.99)
                entry["streak_ratio_last_12"] = {
                    "log_loss": compute_log_loss(y_true, p_streak),
                    "brier_score": compute_brier_score(y_true, p_streak),
                }

            # Baseline 4: delta-to-strike heuristic (logistic regression fit on train)
            #
            # AUDIT FIX (2026-05-12): the previous baseline used a hand-rolled
            # `sigmoid(delta * 0.5)` which is wildly out-of-scale for dollar-
            # denominated deltas (typical |delta| = $5..$200, so sigmoid output
            # was ~1.0 for almost every row, giving log_loss > 1.3).  Any model
            # would beat this trivially, making the gate meaningless.
            #
            # New baseline: properly fitted univariate logistic regression on
            # `delta_to_strike` only.  This is the honest "is your model
            # better than just using BTC's distance to strike?" baseline.
            if "delta_to_strike" in frame.columns and "delta_to_strike" in split.train.columns:
                from sklearn.linear_model import LogisticRegression as _LR
                delta_train = split.train["delta_to_strike"].fill_null(0.0).to_numpy().reshape(-1, 1)
                y_train_raw = split.train[split.target_column].to_numpy().astype(int)
                # Need at least both classes present
                if len(np.unique(y_train_raw)) >= 2:
                    try:
                        lr_baseline = _LR(C=1.0, max_iter=200).fit(delta_train, y_train_raw)
                        delta_eval = frame["delta_to_strike"].fill_null(0.0).to_numpy().reshape(-1, 1)
                        p_delta = np.clip(lr_baseline.predict_proba(delta_eval)[:, 1], 1e-6, 1 - 1e-6)
                        entry["delta_to_strike_fitted_lr"] = {
                            "log_loss": compute_log_loss(y_true, p_delta),
                            "brier_score": compute_brier_score(y_true, p_delta),
                        }
                    except Exception:
                        pass  # silently skip if LR fit fails

            result[name] = entry

        return result

    def _save_artifacts(self, model: Any, split: DatasetSplit, metrics: dict[str, Any], artifact_path: str) -> None:
        path = Path(artifact_path)
        path.mkdir(parents=True, exist_ok=True)

        # Persist model
        joblib.dump(model, path / "model.pkl")

        # Feature importance (LightGBM only; skip gracefully for others)
        importance: dict[str, float] = {}
        if hasattr(self, "get_feature_importance"):
            try:
                importance = self.get_feature_importance(model)
            except Exception:
                pass
        if importance:
            (path / "feature_importance.json").write_text(
                json.dumps(importance, indent=2), encoding="utf-8"
            )

        self.writer.write_eval_report(
            artifact_path=path,
            model_id=self.model_id,
            metrics=metrics,
            dataset_split=split,
            algorithm=self.__class__.__name__,
        )

    def _get_artifact_path(self, dataset_variant: str | None = None) -> str:
        import os
        registry = yaml.safe_load(Path("config/model_registry_v1.yaml").read_text(encoding="utf-8")) or {}
        base_path_str = registry["models"][self.model_id]["artifacts"]["base_path"]

        # === ARTIFACT ROOT OVERRIDE ===
        # When MODEL_ARTIFACT_ROOT_OVERRIDE is set (e.g. "artifacts_cleaned"),
        # the leading "artifacts" prefix in the registered base_path is
        # replaced.  Used by the feature-cleanup retrain to keep cleaned
        # artifacts separate from the contaminated baseline.
        override = os.environ.get("MODEL_ARTIFACT_ROOT_OVERRIDE")
        if override:
            override = override.rstrip("/").rstrip("\\")
            for prefix in ("artifacts/", "artifacts\\"):
                if base_path_str.startswith(prefix):
                    base_path_str = override + "/" + base_path_str[len(prefix):]
                    break
        # === END OVERRIDE ===

        base = Path(base_path_str)
        algorithm = getattr(self, "ALGORITHM_NAME", self.__class__.__name__.lower())
        if dataset_variant:
            base = base / dataset_variant
        base = base / algorithm
        return str(base)

