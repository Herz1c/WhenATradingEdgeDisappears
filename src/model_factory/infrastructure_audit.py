from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Finding:
    severity: str
    check: str
    message: str


DELIVERABLES = [
    "src/canonical/polymarket_ws_events_builder.py",
    "src/canonical/external_btc_events_builder.py",
    "src/canonical/secondary_context_events_builder.py",
    "src/canonical/asof_join_lib.py",
    "src/canonical/time_alignment_report.py",
    "tests/test_polymarket_ws_events_builder.py",
    "tests/test_external_btc_events_builder.py",
    "tests/test_secondary_context_events_builder.py",
    "tests/test_asof_join_lib.py",
    "src/polymarket_recorder/event_triggered_sampler.py",
    "tests/test_event_triggered_sampler.py",
    "src/orchestration/monthly_build.py",
    "src/orchestration/coverage_matrix.py",
    "src/orchestration/reproducibility_report.py",
    "config/orchestration_config.yaml",
    "docs/source_policy_ledger.md",
    "src/monitoring/book_quality_monitor.py",
    "docs/execution_intent_coverage_gaps.md",
    "data/canonical/execution_intent_coverage_gaps.parquet",
    "config/fee_schedules/fee_schedule_registry.yaml",
    "src/execution_simulator/fee_schedule_registry.py",
    "tests/test_fee_schedule_registry.py",
    "src/model_factory/trainers/base_trainer.py",
    "src/model_factory/trainers/logistic_regression_trainer.py",
    "src/model_factory/trainers/lightgbm_trainer.py",
    "src/model_factory/trainers/linear_regression_trainer.py",
    "src/model_factory/evaluation/walkforward.py",
    "src/model_factory/evaluation/metrics.py",
    "src/model_factory/evaluation/report_generator.py",
    "src/model_factory/evaluation/gates.py",
    "tests/test_walkforward.py",
    "tests/test_evaluation_metrics.py",
    "src/policy/score_aggregator.py",
    "src/policy/threshold_policy.py",
    "src/policy/inventory_controller.py",
    "config/policy/threshold_policy_v1.yaml",
    "config/policy/risk_caps_v1.yaml",
    "config/schemas/execution_intent_dataset_v2_enriched_schema.yaml",
    "src/polymarket_recorder/execution_intent_dataset_v2.py",
    "src/paper_trading/replay_paper.py",
    "src/paper_trading/observability.py",
    "src/paper_trading/drift_monitor.py",
    "config/monitoring/alert_thresholds_v1.yaml",
    "src/live/kill_switch.py",
    "src/live/daily_audit.py",
    "roadmap_btc_5m_updown.txt",
    "docs/decision_log.md",
    "src/model_factory/infrastructure_audit.py",
]

MODEL_SPECIFIC_TRAINERS = [
    "src/model_factory/trainers/model_specific/model_01_bias_trainer.py",
    "src/model_factory/trainers/model_specific/model_02_fair_resolution_trainer.py",
    "src/model_factory/trainers/model_specific/model_03_fill_probability_trainer.py",
    "src/model_factory/trainers/model_specific/model_04_adverse_selection_trainer.py",
    "src/model_factory/trainers/model_specific/model_05_closing_flip_trainer.py",
    "src/model_factory/trainers/model_specific/model_06_mispricing_trainer.py",
    "src/model_factory/trainers/model_specific/model_07_microstructure_direction_trainer.py",
    "src/model_factory/trainers/model_specific/model_08_move_size_trainer.py",
    "src/model_factory/trainers/model_specific/model_09_regime_encoder_trainer.py",
    "src/model_factory/trainers/model_specific/model_10_action_policy_trainer.py",
]


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _has_function(path: str | Path, function_name: str) -> bool:
    tree = ast.parse(_read(path))
    return any(isinstance(node, ast.FunctionDef) and node.name == function_name for node in ast.walk(tree))


def _check_cli_help(command: str, flag: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "polymarket_recorder.cli", command, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    return result.returncode == 0 and flag in result.stdout


def _add(findings: list[Finding], severity: str, check: str, message: str) -> None:
    findings.append(Finding(severity=severity, check=check, message=message))


def run_infrastructure_audit(output_path: str) -> dict[str, Any]:
    """Comprehensive infrastructure audit."""
    findings: list[Finding] = []

    for path in [*DELIVERABLES, *MODEL_SPECIFIC_TRAINERS]:
        if not Path(path).exists():
            _add(findings, "CRITICAL", "deliverable_exists", f"Missing deliverable: {path}")

    if Path("src/canonical/asof_join_lib.py").exists():
        for fn in ["asof_join_left_causal", "asof_join_live_reference", "verify_causal_integrity"]:
            if not _has_function("src/canonical/asof_join_lib.py", fn):
                _add(findings, "CRITICAL", "asof_join_lib", f"Missing function: {fn}")
        try:
            import polars as pl
            from canonical.asof_join_lib import asof_join_left_causal

            asof_join_left_causal(
                pl.DataFrame({"recv_ts_ns": [1]}),
                pl.DataFrame({"recv_ts_ns": [1]}),
                "recv_ts_ns",
                "recv_ts_ns",
                strategy="forward",
            )
            _add(findings, "CRITICAL", "asof_join_lib", "Forward join did not raise ValueError")
        except ValueError:
            pass

    if Path("src/polymarket_recorder/event_triggered_sampler.py").exists():
        code = _read("src/polymarket_recorder/event_triggered_sampler.py")
        for trigger in [
            "strike_cross",
            "spread_burst",
            "trade_burst",
            "imbalance_change",
            "tick_size_update",
            "cross_source_divergence",
        ]:
            if trigger not in code:
                _add(findings, "CRITICAL", "event_triggered_sampler", f"Missing trigger: {trigger}")

    schema_path = Path("config/schemas/resolution_snapshot_dataset_v1_coarse_schema.yaml")
    if schema_path.exists():
        schema_text = _read(schema_path)
        if "trigger_type" not in schema_text:
            _add(findings, "CRITICAL", "schema_contract", "trigger_type missing from coarse schema")
        if "trigger_magnitude" not in schema_text:
            _add(findings, "CRITICAL", "schema_contract", "trigger_magnitude missing from coarse schema")

    if Path("src/orchestration/monthly_build.py").exists():
        code = _read("src/orchestration/monthly_build.py")
        for step in [
            "build-registries",
            "build-live-reference-events",
            "build-phase3-datasets",
            "build-phase8-datasets",
            "build-execution-intent-dataset-v1",
            "generate-coverage-matrix",
        ]:
            if step not in code:
                _add(findings, "CRITICAL", "monthly_build", f"Missing step text: {step}")

    for command in ["build-phase3-datasets", "build-phase8-datasets", "build-execution-intent-dataset-v1"]:
        try:
            if not _check_cli_help(command, "--dry-run"):
                _add(findings, "CRITICAL", "cli_dry_run", f"{command} missing --dry-run")
            if not _check_cli_help(command, "--status"):
                _add(findings, "CRITICAL", "cli_status", f"{command} missing --status")
        except Exception as exc:
            _add(findings, "CRITICAL", "cli_help", f"{command} help failed: {exc}")

    trainer_path = Path("src/model_factory/trainers/base_trainer.py")
    if trainer_path.exists():
        code = _read(trainer_path)
        for method in ["_build_model", "_fit", "_predict_proba", "train", "_calibrate", "_evaluate", "_save_artifacts"]:
            if f"def {method}" not in code:
                _add(findings, "CRITICAL", "base_trainer", f"Missing method: {method}")
        if "LeakageSafeLoader" not in code:
            _add(findings, "CRITICAL", "base_trainer", "LeakageSafeLoader not enforced")
        if code.find("write_experiment_manifest") > code.find("self._fit"):
            _add(findings, "CRITICAL", "base_trainer", "Manifest written after fit")

    if Path("src/model_factory/evaluation/walkforward.py").exists():
        code = _read("src/model_factory/evaluation/walkforward.py")
        if "no_future" not in code.lower() and "max_train" not in code:
            _add(findings, "CRITICAL", "walkforward", "Temporal ordering verification marker missing")

    if Path("src/model_factory/evaluation/metrics.py").exists():
        code = _read("src/model_factory/evaluation/metrics.py")
        required_metrics = [
            "compute_log_loss",
            "compute_brier_score",
            "compute_ece",
            "compute_calibration_curve",
            "compute_implied_edge",
            "compute_edge_by_time_to_close",
            "compute_fee_adjusted_edge",
            "compute_fill_adjusted_edge",
            "compute_markout_stats",
            "compute_inventory_stress",
            "compute_stability_by_day",
            "compute_flip_rate_by_delta_bucket",
            "compute_adverse_flag_precision_recall",
        ]
        for fn in required_metrics:
            if f"def {fn}" not in code:
                _add(findings, "CRITICAL", "metrics", f"Missing metric: {fn}")

    if Path("src/model_factory/evaluation/gates.py").exists():
        code = _read("src/model_factory/evaluation/gates.py")
        for i in range(1, 11):
            if f"model_{i:02d}" not in code:
                _add(findings, "CRITICAL", "gates", f"Missing promote/kill gate for model_{i:02d}")

    if Path("src/policy/score_aggregator.py").exists():
        code = _read("src/policy/score_aggregator.py")
        for action in [
            "quote_both",
            "quote_yes_only",
            "quote_no_only",
            "cancel_yes",
            "cancel_no",
            "cancel_all",
            "taker_emergency_only",
        ]:
            if action not in code:
                _add(findings, "CRITICAL", "policy_action_space", f"Missing action: {action}")
        if "placeholder_mode" not in code and "PLACEHOLDER" not in code:
            _add(findings, "CRITICAL", "policy_placeholder", "No placeholder-mode protection")

    if Path("src/polymarket_recorder/execution_intent_dataset_v2.py").exists():
        if "NotImplementedError" not in _read("src/polymarket_recorder/execution_intent_dataset_v2.py"):
            _add(findings, "CRITICAL", "execution_intent_v2", "Stub does not raise NotImplementedError")

    if Path("src/paper_trading/replay_paper.py").exists():
        if "ExecutionSimulatorV1" not in _read("src/paper_trading/replay_paper.py"):
            _add(findings, "CRITICAL", "replay_paper", "ReplayPaperTrader does not use ExecutionSimulatorV1")
    if Path("src/paper_trading/drift_monitor.py").exists():
        if "brier" not in _read("src/paper_trading/drift_monitor.py").lower():
            _add(findings, "CRITICAL", "drift_monitor", "Brier drift tracking missing")

    if Path("src/live/kill_switch.py").exists():
        code = _read("src/live/kill_switch.py")
        if "threading" not in code or "Event" not in code:
            _add(findings, "CRITICAL", "kill_switch", "KillSwitch is not thread-safe")
        if "is_killed" not in code:
            _add(findings, "CRITICAL", "kill_switch", "is_killed method missing")

    risk_caps_path = Path("config/policy/risk_caps_v1.yaml")
    if risk_caps_path.exists():
        risk_caps = _load_yaml(risk_caps_path)
        text = str(risk_caps)
        for cap in [
            "per_market_size_cap_usdc",
            "per_day_loss_cap_usdc",
            "max_total_exposure_usdc",
            "latency_threshold_ms",
            "freshness_threshold_s",
        ]:
            if cap not in text:
                _add(findings, "CRITICAL", "risk_caps", f"Missing cap: {cap}")

    fee_path = Path("config/fee_schedules/fee_schedule_registry.yaml")
    if fee_path.exists():
        text = str(_load_yaml(fee_path))
        for schedule in ["polymarket_crypto_pre_20260330", "polymarket_crypto_post_20260330"]:
            if schedule not in text:
                _add(findings, "CRITICAL", "fee_schedule_registry", f"Missing schedule: {schedule}")

    decision_log_path = Path("docs/decision_log.md")
    if decision_log_path.exists():
        log = _read(decision_log_path).lower()
        for entry in [
            "fng excluded",
            "rtds excluded",
            "bybit excluded",
            "fee_schedule",
            "markout",
            "training_feature_eligible",
            "leakage",
            "asof_join",
            "event_triggered",
            "infrastructure_sprint",
        ]:
            if entry not in log:
                _add(findings, "CRITICAL", "decision_log", f"Missing decision-log entry marker: {entry}")

    roadmap = Path("roadmap_btc_5m_updown.txt")
    if roadmap.exists() and "INFRASTRUCTURE SPRINT STATUS: COMPLETE" not in _read(roadmap):
        _add(findings, "CRITICAL", "roadmap", "Infrastructure sprint status missing from roadmap")

    critical_count = sum(1 for finding in findings if finding.severity == "CRITICAL")
    warning_count = sum(1 for finding in findings if finding.severity == "WARNING")
    verdict = "PASS" if critical_count == 0 and warning_count == 0 else "FAIL"
    report = {
        "verdict": verdict,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "findings": [finding.__dict__ for finding in findings],
    }
    _write_report(Path(output_path), report)
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Infrastructure Audit 2026-04-28",
        "",
        f"- Verdict: {report['verdict']}",
        f"- CRITICAL findings: {report['critical_count']}",
        f"- WARNING findings: {report['warning_count']}",
        "",
        "## Findings",
    ]
    if report["findings"]:
        lines.extend(["| severity | check | message |", "| --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['severity']} | {finding['check']} | {finding['message']} |")
    else:
        lines.append("No findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
