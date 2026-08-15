from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .dataset_loader import LeakageSafeLoader

EXPECTED_MODEL_IDS = [
    "model_01_bias",
    "model_02_fair_resolution",
    "model_03_fill_probability",
    "model_04_adverse_selection",
    "model_05_closing_flip",
    "model_06_mispricing",
    "model_07_microstructure_direction",
    "model_08_move_size",
    "model_09_regime_encoder",
    "model_10_action_policy",
]

LOADABLE_MODELS = [
    ("model_01_bias", "preopen"),
    ("model_02_fair_resolution", "coarse"),
    ("model_03_fill_probability", None),
    ("model_04_adverse_selection", None),
    ("model_05_closing_flip", "dense_close"),
    ("model_06_mispricing", "dense_close"),
    ("model_07_microstructure_direction", "tabular"),
    ("model_08_move_size", "tabular"),
]

REQUIRED_REGISTRY_FIELDS = [
    "display_name",
    "model_id",
    "phase",
    "status",
    "purpose",
    "dataset",
    "feature_set",
    "target",
    "split",
    "algorithms",
    "evaluation",
    "artifacts",
]

REQUIRED_FEATURE_FIELDS = [
    "feature_set_id",
    "model_id",
    "version",
    "feature_columns",
    "leakage_only_columns",
    "target_columns",
]

REQUIRED_SPLIT_FIELDS = [
    "split_id",
    "model_id",
    "method",
    "ordering_column",
    "train",
    "validation",
    "test",
    "purge_days",
]

UNIVERSAL_LEAKAGE = [
    "resolved_side",
    "flip_happened_after_t",
    "terminal_delta_to_strike",
    "terminal_margin_to_strike",
    "hold_to_close_edge_vs_mid",
    "resolved_ts_ns",
]


@dataclass
class Finding:
    severity: str
    check: str
    detail: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def audit_registry(registry_path: str | Path = "config/model_registry_v1.yaml") -> list[Finding]:
    findings: list[Finding] = []
    registry = load_yaml(registry_path)
    models = registry.get("models", {})
    for model_id in EXPECTED_MODEL_IDS:
        if model_id not in models:
            findings.append(Finding("CRITICAL", "registry_completeness", f"{model_id} missing"))

    for model_id, config in models.items():
        for field in REQUIRED_REGISTRY_FIELDS:
            if field not in config:
                findings.append(Finding("CRITICAL", "registry_required_fields", f"{model_id} missing {field}"))
        if config.get("model_id") != model_id:
            findings.append(
                Finding(
                    "CRITICAL",
                    "registry_model_id_match",
                    f"{model_id} key has model_id={config.get('model_id')}",
                )
            )
    return findings


def audit_feature_configs(feature_sets_dir: str | Path = "config/feature_sets") -> list[Finding]:
    findings: list[Finding] = []
    feature_sets_dir = Path(feature_sets_dir)
    for model_id in EXPECTED_MODEL_IDS:
        path = feature_sets_dir / f"{model_id}_features_v1.yaml"
        if not path.exists():
            findings.append(Finding("CRITICAL", "feature_config_exists", f"Missing {path}"))
            continue
        config = load_yaml(path)
        for field in REQUIRED_FEATURE_FIELDS:
            if field not in config:
                findings.append(Finding("CRITICAL", "feature_required_fields", f"{path} missing {field}"))

        feature_columns = set(config.get("feature_columns") or [])
        leakage_columns = set(config.get("leakage_only_columns") or [])
        overlap = feature_columns & leakage_columns
        if overlap:
            findings.append(
                Finding(
                    "CRITICAL",
                    "feature_leakage_overlap",
                    f"{model_id} has columns in both feature and leakage lists: {sorted(overlap)}",
                )
            )

        target_columns = config.get("target_columns") or {}
        targets = {target_columns.get("primary"), *(target_columns.get("auxiliary") or [])}
        targets.discard(None)
        target_overlap = feature_columns & targets
        if target_overlap:
            findings.append(
                Finding(
                    "CRITICAL",
                    "feature_target_overlap",
                    f"{model_id} target column(s) in feature list: {sorted(target_overlap)}",
                )
            )

        if config.get("status") != "placeholder" and not feature_columns:
            findings.append(Finding("WARNING", "feature_nonempty", f"{model_id} has empty feature_columns"))

        for column in UNIVERSAL_LEAKAGE:
            if column not in leakage_columns:
                findings.append(
                    Finding(
                        "WARNING",
                        "feature_universal_leakage_blacklist",
                        f"{model_id} missing universal leakage column {column}",
                    )
                )
    return findings


def audit_split_configs(splits_dir: str | Path = "config/splits") -> list[Finding]:
    findings: list[Finding] = []
    splits_dir = Path(splits_dir)
    for model_id in EXPECTED_MODEL_IDS:
        path = splits_dir / f"{model_id}_split_v1.yaml"
        if not path.exists():
            findings.append(Finding("CRITICAL", "split_config_exists", f"Missing {path}"))
            continue
        config = load_yaml(path)
        for field in REQUIRED_SPLIT_FIELDS:
            if field not in config:
                findings.append(Finding("CRITICAL", "split_required_fields", f"{path} missing {field}"))
        try:
            train_end = date.fromisoformat(config["train"]["end_day"])
            val_start = date.fromisoformat(config["validation"]["start_day"])
            val_end = date.fromisoformat(config["validation"]["end_day"])
            test_start = date.fromisoformat(config["test"]["start_day"])
            if train_end >= val_start:
                findings.append(Finding("CRITICAL", "split_temporal_order", f"{model_id}: train_end >= val_start"))
            if val_end >= test_start:
                findings.append(Finding("CRITICAL", "split_temporal_order", f"{model_id}: val_end >= test_start"))
            if val_start > val_end:
                findings.append(Finding("CRITICAL", "split_temporal_order", f"{model_id}: val_start > val_end"))
        except Exception as exc:
            findings.append(Finding("CRITICAL", "split_date_parse", f"{model_id}: {exc}"))

        if int(config.get("purge_days", 0)) < 0:
            findings.append(Finding("CRITICAL", "split_purge_days", f"{model_id}: purge_days < 0"))
    return findings


def run_loader_integration() -> dict[str, dict[str, Any]]:
    loader = LeakageSafeLoader()
    results: dict[str, dict[str, Any]] = {}
    for model_id, variant in LOADABLE_MODELS:
        try:
            split = loader.load(model_id, dataset_variant=variant, verbose=False)
            results[model_id] = {
                "status": "OK",
                "variant": variant or "default",
                "n_train": split.n_train,
                "n_val": split.n_val,
                "n_test": split.n_test,
                "n_features": len(split.feature_columns),
                "dataset_hash": split.dataset_hash[:16],
            }
        except ValueError as exc:
            results[model_id] = {"status": "CRITICAL", "variant": variant or "default", "error": str(exc)}
        except Exception as exc:
            results[model_id] = {"status": "ERROR", "variant": variant or "default", "error": str(exc)}
    return results


def run_leakage_injection_test() -> dict[str, Any]:
    loader = LeakageSafeLoader()
    broken_config = {
        "feature_set_id": "broken_test",
        "model_id": "model_01_bias",
        "version": "test",
        "feature_columns": ["btc_return_1s", "resolved_side", "flip_happened_after_t"],
        "leakage_only_columns": ["resolved_side", "flip_happened_after_t", "terminal_delta_to_strike"],
        "target_columns": {"primary": "resolved_side_label", "auxiliary": []},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "broken.yaml"
        path.write_text(yaml.safe_dump(broken_config, sort_keys=False), encoding="utf-8")
        try:
            loader._load_feature_config_from_path(path)
        except ValueError as exc:
            if "LEAKAGE" in str(exc) or "leakage" in str(exc).lower():
                return {"status": "OK", "message": str(exc)}
            return {"status": "WARNING", "message": str(exc)}
    return {
        "status": "CRITICAL",
        "message": "Leakage injection was not caught by the loader.",
    }


def run_dataset_hash_stability_test() -> dict[str, Any]:
    loader = LeakageSafeLoader()
    first = loader.load("model_01_bias", dataset_variant="preopen", verbose=False)
    second = loader.load("model_01_bias", dataset_variant="preopen", verbose=False)
    if first.dataset_hash != second.dataset_hash:
        return {
            "status": "CRITICAL",
            "message": "Same dataset produced different hashes on repeated loads.",
        }
    if len(first.dataset_hash) != 64 or first.dataset_hash == "0" * 64:
        return {"status": "CRITICAL", "message": "Dataset hash is not a valid SHA256 digest."}
    return {"status": "OK", "dataset_hash": first.dataset_hash[:16]}


def run_model_factory_audit(output_path: str | Path = "docs/model_factory_sprint1_audit.md") -> dict[str, Any]:
    findings: list[Finding] = []
    findings.extend(audit_registry())
    findings.extend(audit_feature_configs())
    findings.extend(audit_split_configs())

    integration = run_loader_integration()
    for model_id, result in integration.items():
        if result["status"] in {"CRITICAL", "ERROR"}:
            findings.append(
                Finding(
                    "CRITICAL",
                    "loader_integration",
                    f"{model_id}: {result.get('error', result['status'])}",
                )
            )

    leakage_injection = run_leakage_injection_test()
    if leakage_injection["status"] == "CRITICAL":
        findings.append(Finding("CRITICAL", "leakage_injection", leakage_injection["message"]))
    elif leakage_injection["status"] == "WARNING":
        findings.append(Finding("WARNING", "leakage_injection", leakage_injection["message"]))

    hash_stability = run_dataset_hash_stability_test()
    if hash_stability["status"] == "CRITICAL":
        findings.append(Finding("CRITICAL", "dataset_hash_stability", hash_stability["message"]))

    critical_count = sum(1 for finding in findings if finding.severity == "CRITICAL")
    warning_count = sum(1 for finding in findings if finding.severity == "WARNING")
    report = {
        "verdict": "PASS" if critical_count == 0 else "FAIL",
        "critical_count": critical_count,
        "warning_count": warning_count,
        "findings": [finding.__dict__ for finding in findings],
        "loader_integration": integration,
        "leakage_injection": leakage_injection,
        "dataset_hash_stability": hash_stability,
    }
    write_audit_report(output_path, report)
    return report


def write_audit_report(output_path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Factory Sprint 1 Audit",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Summary",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- CRITICAL findings: `{report['critical_count']}`",
        f"- WARNING findings: `{report['warning_count']}`",
        "- Scope: infrastructure only; no model training was run.",
        "",
        "## Loader Integration",
        "",
        "| model_id | variant | status | train | val | test | features | dataset_hash |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model_id, result in report["loader_integration"].items():
        lines.append(
            "| {model_id} | {variant} | {status} | {train} | {val} | {test} | {features} | {hash} |".format(
                model_id=model_id,
                variant=result.get("variant", ""),
                status=result["status"],
                train=result.get("n_train", ""),
                val=result.get("n_val", ""),
                test=result.get("n_test", ""),
                features=result.get("n_features", ""),
                hash=result.get("dataset_hash", result.get("error", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Leakage Injection",
            "",
            f"- Status: `{report['leakage_injection']['status']}`",
            f"- Message: {report['leakage_injection'].get('message', '')}",
            "",
            "## Dataset Hash Stability",
            "",
            f"- Status: `{report['dataset_hash_stability']['status']}`",
            f"- Hash prefix: `{report['dataset_hash_stability'].get('dataset_hash', '')}`",
            "",
            "## Findings",
            "",
            "| severity | check | detail |",
            "| --- | --- | --- |",
        ]
    )
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(f"| {finding['severity']} | {finding['check']} | {finding['detail']} |")
    else:
        lines.append("| OK | all_checks | No findings. |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def list_registered_models(registry_path: str | Path = "config/model_registry_v1.yaml") -> list[dict[str, Any]]:
    registry = load_yaml(registry_path)
    models = []
    for model_id, config in registry.get("models", {}).items():
        models.append(
            {
                "model_id": model_id,
                "display_name": config.get("display_name"),
                "phase": config.get("phase"),
                "status": config.get("status"),
                "dataset_family": config.get("dataset", {}).get("family"),
                "dataset_version": config.get("dataset", {}).get("version"),
            }
        )
    return models


def audit_summary_json(output_path: str | Path) -> str:
    report = run_model_factory_audit(output_path=output_path)
    return json.dumps(
        {
            "verdict": report["verdict"],
            "critical_count": report["critical_count"],
            "warning_count": report["warning_count"],
            "output_path": str(output_path),
        },
        indent=2,
    )
