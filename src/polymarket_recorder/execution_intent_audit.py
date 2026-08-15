from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from .execution_audit import FORBIDDEN_PATTERNS, audit_execution_results
from .execution_intent_dataset import DATASET_COLUMNS, DATASET_NAME, DEFAULT_DATASET_ROOT, DEFAULT_SCHEMA_PATH, PHASE8_TARGET_COLUMNS
from .execution_ledger import reconcile_execution_results
from .execution_simulator import RESULT_COLUMNS


def _finding(severity: str, check: str, message: str) -> dict[str, str]:
    return {"severity": severity, "check": check, "message": message}


def _date_range(date_from: date, date_to: date) -> list[date]:
    if date_to < date_from:
        raise ValueError("date_to must be >= date_from")
    return [date_from + timedelta(days=offset) for offset in range((date_to - date_from).days + 1)]


def _read_schema_columns(schema_path: Path) -> list[str]:
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    return list(payload["columns"].keys())


def read_execution_intent_shards(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    date_from: date,
    date_to: date,
) -> tuple[pd.DataFrame, list[Path]]:
    paths = [dataset_root / DATASET_NAME / f"{target_day.isoformat()}.parquet" for target_day in _date_range(date_from, date_to)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {DATASET_NAME} shards: {[str(path) for path in missing]}")
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame(columns=DATASET_COLUMNS), paths
    return pd.concat(frames, axis=0, ignore_index=True), paths


def audit_execution_intent_frame(*, frame: pd.DataFrame, schema_path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    expected_columns = _read_schema_columns(schema_path)
    actual_columns = list(frame.columns)
    missing = [column for column in expected_columns if column not in actual_columns]
    extra = [column for column in actual_columns if column not in expected_columns]
    if missing or extra:
        findings.append(_finding("CRITICAL", "schema", f"Schema mismatch missing={missing} extra={extra}"))

    forbidden_columns = sorted(set(actual_columns) & PHASE8_TARGET_COLUMNS)
    forbidden_patterns = [column for column in actual_columns if any(pattern in column.lower() for pattern in FORBIDDEN_PATTERNS)]
    if forbidden_columns:
        findings.append(_finding("CRITICAL", "phase8_target_leakage", f"Forbidden Phase 8 target columns present: {forbidden_columns}"))
    if forbidden_patterns:
        findings.append(_finding("CRITICAL", "forbidden_source_columns", f"Forbidden source-like columns present: {forbidden_patterns}"))

    rows = int(len(frame))
    if rows == 0:
        findings.append(_finding("WARNING", "empty_dataset", f"{DATASET_NAME} has zero rows"))
        return _summary(frame, findings, execution_audit=None, ledger={"balances": True, "empty_dataset": True})

    execution_frame = frame[[column for column in RESULT_COLUMNS if column in frame.columns]].copy()
    if list(execution_frame.columns) == RESULT_COLUMNS:
        execution_audit = audit_execution_results(execution_frame)
        for audit_finding in execution_audit["findings"]:
            severity = "CRITICAL" if audit_finding["severity"] == "CRITICAL" else "WARNING"
            findings.append(
                _finding(
                    severity,
                    f"execution_result_{audit_finding['check']}",
                    audit_finding["message"],
                )
            )
        ledger = reconcile_execution_results(execution_frame)
        if not ledger["balances"]:
            findings.append(
                _finding(
                    "CRITICAL",
                    "ledger_reconciliation",
                    f"Ledger settlement differs from row net PnL by {ledger['difference']}",
                )
            )
    else:
        execution_audit = None
        ledger = None
        findings.append(_finding("CRITICAL", "execution_result_schema", "Cannot run execution audit because RESULT_COLUMNS are incomplete"))

    if {"source_anchor_dataset", "input_policy_compliant"}.issubset(frame.columns):
        bad_source = int((frame["source_anchor_dataset"].astype(str) != "microstructure_sequence_dataset_v1_tabular").sum())
        if bad_source:
            findings.append(_finding("CRITICAL", "anchor_source", f"{bad_source} rows use an unapproved anchor dataset"))
        non_policy = int((~frame["input_policy_compliant"].astype(bool)).sum())
        if non_policy:
            findings.append(_finding("CRITICAL", "input_policy", f"{non_policy} rows are not input-policy compliant"))

    if {"anchor_feature_eligible", "execution_training_eligible"}.issubset(frame.columns):
        bad_training_anchor = int((frame["execution_training_eligible"].astype(bool) & ~frame["anchor_feature_eligible"].astype(bool)).sum())
        if bad_training_anchor:
            findings.append(_finding("CRITICAL", "training_anchor_policy", f"{bad_training_anchor} training rows use ineligible anchors"))

    if {"execution_training_eligible", "live_price_degraded"}.issubset(frame.columns):
        degraded_training = int((frame["execution_training_eligible"].astype(bool) & frame["live_price_degraded"].astype(bool)).sum())
        if degraded_training:
            findings.append(_finding("CRITICAL", "degraded_training_rows", f"{degraded_training} degraded live-reference rows are training eligible"))

    if {"execution_training_eligible", "execution_exclusion_reason"}.issubset(frame.columns):
        excluded_training = int(
            (
                frame["execution_training_eligible"].astype(bool)
                & frame["execution_exclusion_reason"].fillna("").astype(str).ne("")
            ).sum()
        )
        if excluded_training:
            findings.append(_finding("CRITICAL", "excluded_training_rows", f"{excluded_training} excluded rows are training eligible"))

    if {"state_recv_ts_ns", "order_ts_ns", "submitted"}.issubset(frame.columns):
        future_state = int(
            (
                frame["submitted"].astype(bool)
                & (frame["state_recv_ts_ns"].astype("int64") >= frame["order_ts_ns"].astype("int64"))
            ).sum()
        )
        if future_state:
            findings.append(_finding("CRITICAL", "state_causality", f"{future_state} submitted rows have state_recv_ts_ns >= order_ts_ns"))

    if {"live_reference_recv_ts_ns", "order_ts_ns"}.issubset(frame.columns):
        future_live = int((frame["live_reference_recv_ts_ns"].astype("int64") > frame["order_ts_ns"].astype("int64")).sum())
        if future_live:
            findings.append(_finding("CRITICAL", "live_reference_causality", f"{future_live} rows have future live reference"))

    if {"filled", "fill_ts_ns", "cancel_effective_ts_ns", "market_close_ts_ns", "order_ts_ns"}.issubset(frame.columns):
        filled = frame["filled"].astype(bool)
        at_or_before_order = int((filled & (frame["fill_ts_ns"].astype("int64") <= frame["order_ts_ns"].astype("int64"))).sum())
        at_or_after_cancel = int(
            (
                filled
                & (frame["last_fill_ts_ns"].astype("int64") >= frame["cancel_effective_ts_ns"].astype("int64"))
            ).sum()
        )
        at_or_after_close = int(
            (
                filled
                & (frame["last_fill_ts_ns"].astype("int64") >= frame["market_close_ts_ns"].astype("int64"))
            ).sum()
        )
        if at_or_before_order:
            findings.append(_finding("CRITICAL", "fill_causality", f"{at_or_before_order} fills are at/before order"))
        if at_or_after_cancel:
            findings.append(_finding("CRITICAL", "cancel_boundary", f"{at_or_after_cancel} fills are at/after cancel effective"))
        if at_or_after_close:
            findings.append(_finding("CRITICAL", "close_boundary", f"{at_or_after_close} fills are at/after close"))

    if {"execution_training_eligible", "submitted"}.issubset(frame.columns):
        training_without_submission = int(
            (frame["execution_training_eligible"].astype(bool) & ~frame["submitted"].astype(bool)).sum()
        )
        if training_without_submission:
            findings.append(
                _finding("CRITICAL", "training_submission", f"{training_without_submission} training rows were not submitted")
            )

    if {"execution_training_eligible", "live_reference_trusted"}.issubset(frame.columns):
        training_without_live = int(
            (frame["execution_training_eligible"].astype(bool) & ~frame["live_reference_trusted"].astype(bool)).sum()
        )
        if training_without_live:
            findings.append(
                _finding("CRITICAL", "training_live_reference", f"{training_without_live} training rows lack trusted live reference")
            )

    duplicated_intents = int(frame["intent_id"].duplicated().sum()) if "intent_id" in frame.columns else 0
    if duplicated_intents:
        findings.append(_finding("CRITICAL", "intent_id_unique", f"{duplicated_intents} duplicate intent_id values"))

    return _summary(frame, findings, execution_audit=execution_audit, ledger=ledger)


def _summary(
    frame: pd.DataFrame,
    findings: list[dict[str, str]],
    *,
    execution_audit: dict[str, Any] | None,
    ledger: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = int(len(frame))
    critical_count = sum(1 for finding in findings if finding["severity"] == "CRITICAL")
    warning_count = sum(1 for finding in findings if finding["severity"] == "WARNING")
    verdict = "PASS" if critical_count == 0 else "FAIL"
    return {
        "dataset": DATASET_NAME,
        "rows": rows,
        "submitted_rows": int(frame["submitted"].sum()) if "submitted" in frame.columns and rows else 0,
        "filled_rows": int(frame["filled"].sum()) if "filled" in frame.columns and rows else 0,
        "training_eligible_rows": int(frame["execution_training_eligible"].sum())
        if "execution_training_eligible" in frame.columns and rows
        else 0,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "findings": findings,
        "execution_audit": execution_audit,
        "ledger": ledger,
        "verdict": verdict,
    }


def write_execution_intent_audit_report(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    date_from: date,
    date_to: date,
    output_path: Path,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    paths = [dataset_root / DATASET_NAME / f"{target_day.isoformat()}.parquet" for target_day in _date_range(date_from, date_to)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {DATASET_NAME} shards: {[str(path) for path in missing]}")

    shard_reports: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    seen_intent_ids: set[str] = set()
    global_duplicate_intents = 0
    for path in paths:
        frame = pd.read_parquet(path)
        shard_report = audit_execution_intent_frame(frame=frame, schema_path=schema_path)
        shard_report["path"] = str(path)
        shard_report["rows"] = int(len(frame))
        shard_reports.append(shard_report)
        for finding in shard_report["findings"]:
            findings.append(
                _finding(
                    finding["severity"],
                    f"{path.name}:{finding['check']}",
                    finding["message"],
                )
            )
        if "intent_id" in frame.columns:
            for intent_id in frame["intent_id"].dropna().astype(str).tolist():
                if intent_id in seen_intent_ids:
                    global_duplicate_intents += 1
                else:
                    seen_intent_ids.add(intent_id)
        del frame

    if global_duplicate_intents:
        findings.append(_finding("CRITICAL", "global_intent_id_unique", f"{global_duplicate_intents} duplicate intent_id values across shards"))
    rows = sum(int(report["rows"]) for report in shard_reports)
    submitted_rows = sum(int(report["submitted_rows"]) for report in shard_reports)
    filled_rows = sum(int(report["filled_rows"]) for report in shard_reports)
    training_eligible_rows = sum(int(report["training_eligible_rows"]) for report in shard_reports)
    critical_count = sum(1 for finding in findings if finding["severity"] == "CRITICAL")
    warning_count = sum(1 for finding in findings if finding["severity"] == "WARNING")
    ledger_balances = all(bool((report.get("ledger") or {}).get("balances")) for report in shard_reports)
    report = {
        "dataset": DATASET_NAME,
        "rows": int(rows),
        "submitted_rows": int(submitted_rows),
        "filled_rows": int(filled_rows),
        "training_eligible_rows": int(training_eligible_rows),
        "critical_count": int(critical_count),
        "warning_count": int(warning_count),
        "findings": findings,
        "shard_reports": shard_reports,
        "execution_audit": None,
        "ledger": {"balances": ledger_balances},
        "verdict": "PASS" if critical_count == 0 else "FAIL",
    }
    lines = [
        "# execution_intent_dataset_v1 Audit Report",
        "",
        f"- Date range: `{date_from.isoformat()}` to `{date_to.isoformat()}`",
        f"- Shards: `{len(paths)}`",
        f"- Rows: `{report['rows']}`",
        f"- Submitted rows: `{report['submitted_rows']}`",
        f"- Filled rows: `{report['filled_rows']}`",
        f"- Training-eligible rows: `{report['training_eligible_rows']}`",
        f"- Critical findings: `{report['critical_count']}`",
        f"- Warning findings: `{report['warning_count']}`",
        f"- Verdict: `{report['verdict']}`",
        "",
        "## Shards",
        "",
    ]
    lines.extend(
        [
            "| shard | rows | submitted | filled | training eligible | critical | warning | verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for shard_report in shard_reports:
        lines.append(
            f"| `{Path(shard_report['path']).name}` | {shard_report['rows']} | {shard_report['submitted_rows']} | "
            f"{shard_report['filled_rows']} | {shard_report['training_eligible_rows']} | "
            f"{shard_report['critical_count']} | {shard_report['warning_count']} | {shard_report['verdict']} |"
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for idx, finding in enumerate(report["findings"], start=1):
            lines.append(f"- `{finding['severity']}-{idx:03d}` `{finding['check']}`: {finding['message']}")
    else:
        lines.append("- No findings.")
    lines.extend(
        [
            "",
            "## Audit Gates",
            "",
            f"- Schema matches: `{not any(f['check'] == 'schema' for f in report['findings'])}`",
            f"- No future live-reference rows: `{not any(f['check'] == 'live_reference_causality' for f in report['findings'])}`",
            f"- No future L2 state rows: `{not any(f['check'] == 'state_causality' for f in report['findings'])}`",
            f"- Ledger balances: `{bool(report['ledger'].get('balances')) if report['ledger'] else False}`",
            f"- Policy-compliant inputs: `{not any(f['check'] == 'input_policy' for f in report['findings'])}`",
            f"- No forbidden source columns: `{not any(f['check'] in {'forbidden_source_columns', 'phase8_target_leakage'} for f in report['findings'])}`",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
