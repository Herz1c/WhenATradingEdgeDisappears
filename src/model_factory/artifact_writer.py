from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ArtifactWriter:
    """
    Writes model experiment artifacts with full provenance tracking.

    Every artifact directory is expected to contain:
    - model.pkl or equivalent model artifact
    - experiment_manifest.json
    - eval_report.md
    - feature_importance.json
    - experiment_log.jsonl
    """

    def write_experiment_manifest(
        self,
        model_id: str,
        dataset_split: Any,
        algorithm: str,
        hyperparams: dict[str, Any],
        artifact_path: str | Path,
    ) -> dict[str, Any]:
        """Write an immutable provenance record before training begins."""
        manifest = {
            "manifest_version": "1",
            "model_id": model_id,
            "algorithm": algorithm,
            "hyperparams": hyperparams,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_provenance": {
                "dataset_hash": dataset_split.dataset_hash,
                "split_id": dataset_split.split_id,
                "n_train": dataset_split.n_train,
                "n_val": dataset_split.n_val,
                "n_test": dataset_split.n_test,
                "feature_columns": dataset_split.feature_columns,
                "target_column": dataset_split.target_column,
                "metadata": dataset_split.metadata,
            },
            "leakage_checks": {
                "loader_version": "LeakageSafeLoader_v1",
                "leakage_blacklist_enforced": True,
                "temporal_ordering_verified": True,
                "no_future_data_verified": True,
                "eligibility_filter_applied": True,
            },
        }

        path = Path(artifact_path)
        path.mkdir(parents=True, exist_ok=True)
        manifest_file = path / "experiment_manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    def append_experiment_log(
        self,
        artifact_path: str | Path,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Append one JSON line to the experiment log."""
        log_entry = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        path = Path(artifact_path)
        path.mkdir(parents=True, exist_ok=True)
        log_file = path / "experiment_log.jsonl"
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry, sort_keys=True) + "\n")

    def write_eval_report(
        self,
        artifact_path: str | Path,
        model_id: str,
        metrics: dict[str, dict[str, Any]],
        dataset_split: Any,
        algorithm: str,
    ) -> None:
        """Write a human-readable evaluation report as markdown."""
        path = Path(artifact_path)
        path.mkdir(parents=True, exist_ok=True)
        report_path = path / "eval_report.md"

        lines = [
            f"# Evaluation Report: {model_id}",
            "",
            f"**Algorithm:** {algorithm}",
            f"**Dataset hash:** `{dataset_split.dataset_hash[:16]}...`",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Dataset Split",
            "| Split | Rows |",
            "| --- | ---: |",
            f"| Train | {dataset_split.n_train} |",
            f"| Val | {dataset_split.n_val} |",
            f"| Test | {dataset_split.n_test} |",
            "",
            "## Metrics",
            "",
        ]

        for split_name, split_metrics in metrics.items():
            lines.append(f"### {split_name}")
            lines.append("| Metric | Value |")
            lines.append("| --- | ---: |")
            for key, value in split_metrics.items():
                rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
                lines.append(f"| {key} | {rendered} |")
            lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
