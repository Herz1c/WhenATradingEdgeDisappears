from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_factory.dataset_loader import DatasetSplit
from model_factory.evaluation.gates import evaluate_gate
from model_factory.evaluation.metrics import compute_brier_score, compute_ece, compute_log_loss


class ModelEvalReportGenerator:
    """Generates standard eval report for any model."""

    REPORT_SECTIONS = [
        "dataset_summary",
        "split_summary",
        "probabilistic_metrics",
        "market_aware_metrics",
        "trading_aware_metrics",
        "stability_metrics",
        "calibration_plot_data",
        "feature_importance",
        "promote_kill_decision",
    ]

    def generate(
        self,
        model_id: str,
        split: DatasetSplit,
        predictions: dict,
        actuals: dict,
        feature_importance: dict,
        artifact_path: str,
        extra_cols: dict | None = None,
    ) -> dict:
        del extra_cols
        metrics: dict[str, Any] = {}
        for split_name, y_prob in predictions.items():
            y_true = actuals[split_name]
            metrics[split_name] = {
                "log_loss": compute_log_loss(y_true, y_prob),
                "brier_score": compute_brier_score(y_true, y_prob),
                "ece": compute_ece(y_true, y_prob),
            }
        metrics["promote_kill_decision"] = self._promote_kill_decision(metrics, model_id)
        path = Path(artifact_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        (path / "feature_importance.json").write_text(json.dumps(feature_importance, indent=2, sort_keys=True), encoding="utf-8")
        (path / "calibration_data.json").write_text(json.dumps({"predictions": {k: list(map(float, v)) for k, v in predictions.items()}}, indent=2), encoding="utf-8")
        lines = [
            f"# Model Evaluation: {model_id}",
            "",
            f"- Dataset hash: `{split.dataset_hash}`",
            f"- Decision: `{metrics['promote_kill_decision']}`",
            "",
            "| split | log_loss | brier_score | ece |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, values in metrics.items():
            if not isinstance(values, dict):
                continue
            lines.append(f"| {name} | {values['log_loss']:.6f} | {values['brier_score']:.6f} | {values['ece']:.6f} |")
        (path / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return metrics

    def _promote_kill_decision(self, metrics: dict, model_id: str) -> str:
        return evaluate_gate(model_id, metrics)

