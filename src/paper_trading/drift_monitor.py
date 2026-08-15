from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from model_factory.evaluation.metrics import compute_brier_score, compute_ece


class DriftMonitor:
    """Rolling model drift monitor for deployed prediction streams."""

    def __init__(
        self,
        model_id: str,
        baseline_metrics_path: str,
        window_size: int = 200,
    ):
        self.model_id = model_id
        self.window_size = int(window_size)
        self.predictions = deque(maxlen=self.window_size)
        self.actuals = deque(maxlen=self.window_size)
        self.features = deque(maxlen=self.window_size)
        self.baseline = self._load_baseline(baseline_metrics_path)

    def _load_baseline(self, path: str) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {"brier_score": 0.25, "ece": 0.05, "feature_baselines": {}}
        with p.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        val = data.get("val", data)
        return {
            "brier_score": float(val.get("brier_score", 0.25)),
            "ece": float(val.get("ece", 0.05)),
            "feature_baselines": data.get("feature_baselines", {}),
        }

    def update(self, y_pred: float, y_true: int, features: dict):
        self.predictions.append(float(y_pred))
        self.actuals.append(int(y_true))
        self.features.append(dict(features))

    def check_drift(self) -> dict:
        if not self.predictions:
            return {"model_id": self.model_id, "status": "INSUFFICIENT_DATA", "alerts": []}

        y_prob = np.asarray(self.predictions, dtype=float)
        y_true = np.asarray(self.actuals, dtype=int)
        brier = compute_brier_score(y_true, y_prob)
        ece = compute_ece(y_true, y_prob, n_bins=10)
        alerts = []
        if brier > float(self.baseline.get("brier_score", 0.25)) + 0.03:
            alerts.append("BRIER_DRIFT")
        if ece > 0.08:
            alerts.append("ECE_DRIFT")
        feature_shift = self._feature_shift_status()
        if feature_shift["shifted_features"]:
            alerts.append("FEATURE_DISTRIBUTION_SHIFT")
        return {
            "model_id": self.model_id,
            "status": "DRIFT" if alerts else "OK",
            "brier_score": brier,
            "ece": ece,
            "feature_shift": feature_shift,
            "alerts": alerts,
        }

    def _feature_shift_status(self) -> dict:
        baselines = self.baseline.get("feature_baselines", {}) or {}
        shifted = []
        if not baselines:
            return {"shifted_features": shifted}
        for name, baseline in baselines.items():
            values = [float(row[name]) for row in self.features if name in row and row[name] is not None]
            if len(values) < 20:
                continue
            current_mean = float(np.mean(values))
            baseline_mean = float(baseline.get("mean", current_mean))
            baseline_std = max(float(baseline.get("std", 1.0)), 1e-9)
            if abs(current_mean - baseline_mean) > 3.0 * baseline_std:
                shifted.append(name)
        return {"shifted_features": shifted}

