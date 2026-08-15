from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from .microstructure_sequence_builder import DEFAULT_SCHEMA_PATH, SCHEMA_COLUMNS, TABULAR_COLUMNS, TARGET_COLUMNS


EVENT_COLUMNS = [column for column in SCHEMA_COLUMNS if column not in TABULAR_COLUMNS]


def _load_schema(schema_path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))


def _finding(severity: str, check: str, message: str) -> dict[str, str]:
    return {"severity": severity, "check": check, "message": message}


def _phase3_anchor_count(dataset_root: Path, date_str: str) -> int:
    total = 0
    for name in ("resolution_snapshot_dataset_v1_coarse", "resolution_snapshot_dataset_v1_dense_close"):
        path = dataset_root / name / f"{date_str}.parquet"
        if path.exists():
            total += pq.ParquetFile(path).metadata.num_rows
    return total


def _scan_dataset(path: Path, schema: dict[str, Any], dataset_root: Path, dataset_name: str) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    if not path.exists():
        return {"path": str(path), "rows": 0, "findings": [_finding("CRITICAL", "exists", f"Missing dataset path: {path}")]}
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    expected_columns = TABULAR_COLUMNS if dataset_name == "tabular" else SCHEMA_COLUMNS
    if columns != expected_columns:
        missing = sorted(set(expected_columns) - set(columns))
        extra = sorted(set(columns) - set(expected_columns))
        findings.append(_finding("CRITICAL", "schema", f"Schema mismatch missing={missing} extra={extra}"))
    schema_columns = schema.get("columns") or {}
    for column in TARGET_COLUMNS:
        if not schema_columns.get(column, {}).get("leakage_only"):
            findings.append(_finding("CRITICAL", "target_leakage_contract", f"Target {column} not marked leakage_only"))
        if schema_columns.get(column, {}).get("training_eligible"):
            findings.append(_finding("CRITICAL", "target_training_contract", f"Target {column} marked training_eligible"))
    if any("fng" in column.lower() for column in columns):
        findings.append(_finding("CRITICAL", "fng_absence", "FNG column found in Phase 8 dataset"))

    rows = parquet.metadata.num_rows
    eligible = 0
    complete = 0
    target_counts = {h: 0 for h in (1, 3, 5, 15, 30)}
    short_sequences = 0
    crossed = 0
    stale = 0
    one_sided = 0
    max_abs_returns = {h: 0.0 for h in (1, 3, 5)}
    token_side_bad = 0
    future_events = 0
    target_unavailable_populated = {h: 0 for h in (1, 3, 5, 15, 30)}
    target_available_past_close = {h: 0 for h in (1, 3, 5, 15, 30)}
    negative_excursions = {("mfe", 5): 0, ("mae", 5): 0, ("mfe", 15): 0, ("mae", 15): 0}
    invalid_event_types: set[str] = set()
    lengths: list[int] = []
    batch_columns = [
        "anchor_ts_ns",
        "events_recv_ts_ns",
        "events_event_type",
        "token_side_normalized",
        "sequence_feature_eligible",
        "sequence_complete_eligible",
        "sequence_length_actual",
        "t_to_close_s",
        "mid_move_1s",
        "mid_move_3s",
        "mid_move_5s",
        "mid_move_15s",
        "mid_move_30s",
        "max_favorable_excursion_5s",
        "max_adverse_excursion_5s",
        "max_favorable_excursion_15s",
        "max_adverse_excursion_15s",
        "target_1s_available",
        "target_3s_available",
        "target_5s_available",
        "target_15s_available",
        "target_30s_available",
        "crossed_book",
        "stale_book",
        "one_sided_book",
        "mid_return_1s",
        "mid_return_3s",
        "mid_return_5s",
    ]
    for batch in parquet.iter_batches(columns=[column for column in batch_columns if column in columns], batch_size=8192):
        frame = batch.to_pandas()
        eligible += int(frame["sequence_feature_eligible"].sum())
        complete += int(frame["sequence_complete_eligible"].sum())
        for h in (1, 3, 5, 15, 30):
            target_counts[h] += int(frame[f"target_{h}s_available"].sum())
            unavailable = ~frame[f"target_{h}s_available"].astype(bool)
            target_unavailable_populated[h] += int((unavailable & frame[f"mid_move_{h}s"].notna()).sum())
            target_available_past_close[h] += int(
                (frame[f"target_{h}s_available"].astype(bool) & (frame["t_to_close_s"] <= h)).sum()
            )
        if "sequence_length_actual" in frame.columns:
            short_sequences += int((frame["sequence_length_actual"] < 32).sum())
            lengths.extend(frame["sequence_length_actual"].astype(int).tolist())
        crossed += int(frame["crossed_book"].sum())
        stale += int(frame["stale_book"].sum())
        one_sided += int(frame["one_sided_book"].sum())
        token_side_bad += int((~frame["token_side_normalized"].isin(["UP", "DOWN"])).sum())
        if "events_recv_ts_ns" in frame.columns:
            for anchor_ts, event_times in zip(frame["anchor_ts_ns"].tolist(), frame["events_recv_ts_ns"].tolist(), strict=False):
                for event_ts in event_times:
                    if event_ts and int(event_ts) >= int(anchor_ts):
                        future_events += 1
                        break
        if "events_event_type" in frame.columns:
            valid_event_types = {"book_state", "top_of_book", "trade", "tick_update", "other", "pad", ""}
            for event_types in frame["events_event_type"].tolist():
                for event_type in event_types:
                    value = "" if event_type is None else str(event_type)
                    if value not in valid_event_types:
                        invalid_event_types.add(value)
        for h in (1, 3, 5):
            values = frame[f"mid_return_{h}s"].dropna().abs()
            if len(values):
                max_abs_returns[h] = max(max_abs_returns[h], float(values.max()))
        for h in (5, 15):
            negative_excursions[("mfe", h)] += int((frame[f"max_favorable_excursion_{h}s"].dropna() < 0).sum())
            negative_excursions[("mae", h)] += int((frame[f"max_adverse_excursion_{h}s"].dropna() < 0).sum())
    if future_events:
        findings.append(_finding("CRITICAL", "causal_integrity", f"{future_events} rows contain future events"))
    if invalid_event_types:
        findings.append(_finding("CRITICAL", "event_type", f"Invalid event_type values: {sorted(invalid_event_types)}"))
    if token_side_bad:
        findings.append(_finding("CRITICAL", "token_mapping", f"{token_side_bad} rows have invalid token_side_normalized"))
    for h in (1, 3, 5, 15, 30):
        if target_unavailable_populated[h]:
            findings.append(_finding("CRITICAL", f"target_{h}s_contract", f"{target_unavailable_populated[h]} rows have mid_move_{h}s populated while target_{h}s_available=False"))
        if target_available_past_close[h]:
            findings.append(_finding("CRITICAL", f"target_{h}s_boundary", f"{target_available_past_close[h]} rows have target_{h}s_available=True but t_to_close_s <= {h}"))
    for h in (5, 15):
        neg_mfe = negative_excursions[("mfe", h)]
        neg_mae = negative_excursions[("mae", h)]
        if neg_mfe:
            findings.append(_finding("CRITICAL", f"mfe_{h}s", f"{neg_mfe} negative MFE values at {h}s"))
        if neg_mae:
            findings.append(_finding("CRITICAL", f"mae_{h}s", f"{neg_mae} negative MAE values at {h}s"))
    eligible_rate = eligible / rows if rows else 0.0
    complete_rate = complete / rows if rows else 0.0
    if eligible_rate <= 0.65:
        findings.append(_finding("WARNING", "eligibility_rate", f"sequence_feature_eligible {eligible_rate:.3f} below threshold"))
    if rows and short_sequences / rows > 0.20:
        findings.append(_finding("WARNING", "sequence_length", ">20% of sequences have fewer than 32 events"))
    for flag, count in (("crossed_book", crossed), ("stale_book", stale), ("one_sided_book", one_sided)):
        rate = count / rows if rows else 0.0
        if rate > 0.03:
            findings.append(_finding("WARNING", flag, f"{flag} rate {rate:.4f} above 3% threshold"))
    for h, value in max_abs_returns.items():
        if value >= 0.5:
            findings.append(_finding("WARNING", f"mid_return_{h}s", f"Tail mid_return_{h}s max_abs={value:.4f}; retained as trainable event"))
    date_str = path.stem
    phase3_count = _phase3_anchor_count(dataset_root, date_str)
    coverage = rows / max(phase3_count, 1)
    if coverage <= 0.85:
        findings.append(_finding("WARNING", "phase3_coverage", f"Coverage {coverage:.3f} below 0.85 threshold"))
    return {
        "path": str(path),
        "rows": rows,
        "eligible": eligible,
        "eligible_rate": eligible_rate,
        "complete": complete,
        "complete_rate": complete_rate,
        "target_counts": target_counts,
        "sequence_length_p10": float(pd.Series(lengths).quantile(0.10)) if lengths else 0.0,
        "sequence_length_p50": float(pd.Series(lengths).median()) if lengths else 0.0,
        "crossed_book_rate": crossed / rows if rows else 0.0,
        "stale_book_rate": stale / rows if rows else 0.0,
        "one_sided_book_rate": one_sided / rows if rows else 0.0,
        "coverage_vs_phase3": coverage,
        "findings": findings,
    }


def write_phase8_audit_report(
    *,
    tabular_path: Path,
    event64_path: Path,
    output_path: Path,
    event128_path: Path | None = None,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    dataset_root: Path = Path("data/datasets"),
) -> dict[str, Any]:
    schema = _load_schema(schema_path)
    paths = {"tabular": tabular_path, "event64": event64_path}
    if event128_path is not None:
        paths["event128"] = event128_path
    datasets = {name: _scan_dataset(path, schema, dataset_root, name) for name, path in paths.items()}
    findings = [finding for dataset in datasets.values() for finding in dataset["findings"]]
    critical_count = sum(1 for finding in findings if finding["severity"] == "CRITICAL")
    lines = [
        "# Phase 8 Microstructure Dataset Audit",
        "",
        f"- Critical findings: `{critical_count}`",
        f"- Total findings: `{len(findings)}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | rows | eligible_rate | complete_rate | coverage_vs_phase3 | seq_len_p10 | seq_len_p50 | crossed | stale | one_sided |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, dataset in datasets.items():
        lines.append(
            f"| {name} | {dataset['rows']} | {dataset['eligible_rate']:.4f} | {dataset['complete_rate']:.4f} | "
            f"{dataset['coverage_vs_phase3']:.4f} | {dataset['sequence_length_p10']:.1f} | {dataset['sequence_length_p50']:.1f} | "
            f"{dataset['crossed_book_rate']:.4f} | {dataset['stale_book_rate']:.4f} | {dataset['one_sided_book_rate']:.4f} |"
        )
    lines.extend(["", "## Target Availability", "", "| dataset | 1s | 3s | 5s | 15s | 30s |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name, dataset in datasets.items():
        rows = max(int(dataset["rows"]), 1)
        target_counts = dataset["target_counts"]
        lines.append(
            f"| {name} | {target_counts[1] / rows:.4f} | {target_counts[3] / rows:.4f} | "
            f"{target_counts[5] / rows:.4f} | {target_counts[15] / rows:.4f} | {target_counts[30] / rows:.4f} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(
            f"- `{finding['severity']}` `{finding['check']}`: {finding['message']}"
            for finding in findings
        )
    else:
        lines.append("- No findings.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"datasets": datasets, "findings": findings, "critical_count": critical_count, "output_path": str(output_path)}


def _date_range(date_from: date, date_to: date) -> list[date]:
    values: list[date] = []
    current = date_from
    while current <= date_to:
        values.append(current)
        current += timedelta(days=1)
    return values


def _variant_path(dataset_root: Path, variant: str, date_str: str) -> Path:
    names = {
        "tabular": "microstructure_sequence_dataset_v1_tabular",
        "event64": "microstructure_sequence_dataset_v1_event64",
        "event128": "microstructure_sequence_dataset_v1_event128",
    }
    return dataset_root / names[variant] / f"{date_str}.parquet"


def _scan_positive_dt(path: Path, expected_len: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["events_dt_from_anchor_s"], batch_size=8192):
        frame = batch.to_pandas()
        for row_idx, values in enumerate(frame["events_dt_from_anchor_s"].tolist()):
            seq_values = [] if values is None else list(values)
            if len(seq_values) != expected_len:
                examples.append({"row": row_idx, "reason": "wrong_length", "length": len(seq_values)})
            positives = [float(value) for value in seq_values if value is not None and not pd.isna(value) and float(value) > 0]
            if positives:
                examples.append({"row": row_idx, "positive_dt": positives[:3]})
            if len(examples) >= 5:
                return examples
    return examples


def _sequence_stats(path: Path) -> dict[str, float]:
    columns = pq.read_schema(path).names
    if "sequence_length_actual" not in columns:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "frac_lt_32": 0.0}
    values = pd.read_parquet(path, columns=["sequence_length_actual"])["sequence_length_actual"].astype(float)
    return {
        "p10": float(values.quantile(0.10)),
        "p50": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "frac_lt_32": float((values < 32).mean()),
    }


def write_phase8_full_audit_report(
    *,
    date_from: date,
    date_to: date,
    output_path: Path,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    dataset_root: Path = Path("data/datasets"),
) -> dict[str, Any]:
    schema = _load_schema(schema_path)
    dates = [value.isoformat() for value in _date_range(date_from, date_to)]
    variants = ("tabular", "event64", "event128")
    findings: list[dict[str, str]] = []
    datasets: dict[tuple[str, str], dict[str, Any]] = {}
    rows_by_date: dict[str, dict[str, int]] = {}
    sequence_by_date: dict[tuple[str, str], dict[str, float]] = {}
    target_rates: dict[str, dict[int, float]] = {}
    phase3_coverage: dict[str, float] = {}

    for date_str in dates:
        rows_by_date[date_str] = {}
        for variant in variants:
            path = _variant_path(dataset_root, variant, date_str)
            dataset = _scan_dataset(path, schema, dataset_root, variant)
            datasets[(date_str, variant)] = dataset
            rows_by_date[date_str][variant] = int(dataset["rows"])
            findings.extend(
                {
                    **finding,
                    "location": f"{variant} {date_str}",
                }
                for finding in dataset["findings"]
            )
            sequence_by_date[(date_str, variant)] = _sequence_stats(path)
            if variant in {"event64", "event128"} and path.exists():
                expected_len = 64 if variant == "event64" else 128
                positive_examples = _scan_positive_dt(path, expected_len)
                if positive_examples:
                    findings.append(
                        _finding("CRITICAL", "dt_from_anchor", f"{variant} {date_str} positive dt_from_anchor_s or wrong sequence length: {positive_examples}")
                    )
        tabular = pd.read_parquet(
            _variant_path(dataset_root, "tabular", date_str),
            columns=[
                "sequence_id",
                "market_slug",
                "token_asset_id",
                "anchor_ts_ns",
                "anchor_source",
                "target_1s_available",
                "target_3s_available",
                "target_5s_available",
                "target_15s_available",
                "target_30s_available",
                "phase_bucket",
            ],
        )
        target_rates[date_str] = {
            horizon: float(tabular[f"target_{horizon}s_available"].mean())
            for horizon in (1, 3, 5, 15, 30)
        }
        phase3_count = _phase3_anchor_count(dataset_root, date_str)
        phase3_coverage[date_str] = len(tabular) / max(phase3_count * 2, 1)
        ids_by_variant: dict[str, set[str]] = {}
        keys_by_variant: dict[str, set[tuple[Any, ...]]] = {}
        for variant in variants:
            frame = pd.read_parquet(
                _variant_path(dataset_root, variant, date_str),
                columns=["sequence_id", "market_slug", "token_asset_id", "anchor_ts_ns", "anchor_source"],
            )
            ids_by_variant[variant] = set(frame["sequence_id"].tolist())
            keys_by_variant[variant] = set(
                map(tuple, frame[["market_slug", "token_asset_id", "anchor_ts_ns", "anchor_source"]].itertuples(index=False, name=None))
            )
            duplicate_ids = len(frame) - frame["sequence_id"].nunique()
            if duplicate_ids:
                findings.append(_finding("CRITICAL", "sequence_id", f"{variant} {date_str} has {duplicate_ids} duplicate sequence_id values"))
        for variant in ("event64", "event128"):
            only_tabular = len(ids_by_variant["tabular"] - ids_by_variant[variant])
            only_variant = len(ids_by_variant[variant] - ids_by_variant["tabular"])
            if only_tabular or only_variant:
                findings.append(
                    _finding(
                        "CRITICAL",
                        "sequence_id_cross_variant",
                        f"{date_str} tabular vs {variant} mismatch: only_tabular={only_tabular}, only_{variant}={only_variant}",
                    )
                )
            key_only_tabular = len(keys_by_variant["tabular"] - keys_by_variant[variant])
            key_only_variant = len(keys_by_variant[variant] - keys_by_variant["tabular"])
            if key_only_tabular or key_only_variant:
                findings.append(
                    _finding(
                        "CRITICAL",
                        "anchor_key_cross_variant",
                        f"{date_str} tabular vs {variant} anchor-key mismatch: only_tabular={key_only_tabular}, only_{variant}={key_only_variant}",
                    )
                )
        if len(set(rows_by_date[date_str].values())) != 1:
            findings.append(_finding("CRITICAL", "row_count_cross_variant", f"{date_str} row counts differ: {rows_by_date[date_str]}"))

    criticals = [finding for finding in findings if finding["severity"] == "CRITICAL"]
    warnings = [finding for finding in findings if finding["severity"] == "WARNING"]
    verdict = "FAIL" if criticals else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    lines = [
        "# Phase 8 Full Audit Post-Fix",
        "",
        f"- Date range: `{date_from.isoformat()}` to `{date_to.isoformat()}`",
        f"- Overall verdict: **{verdict}**",
        f"- Critical findings: `{len(criticals)}`",
        f"- Warning findings: `{len(warnings)}`",
        "- Scope: cross-date schema, causality, target, sequence, eligibility, and variant consistency audit.",
        "",
        "## 1. EXECUTIVE SUMMARY",
        "",
        "| date | tabular rows | event64 rows | event128 rows | target_5s_avail | coverage_vs_phase3x2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for date_str in dates:
        rows = rows_by_date[date_str]
        lines.append(
            f"| {date_str} | {rows['tabular']} | {rows['event64']} | {rows['event128']} | "
            f"{target_rates[date_str][5]:.4f} | {phase3_coverage[date_str]:.4f} |"
        )
    lines.extend(["", "## 2. CRITICAL FINDINGS", ""])
    if criticals:
        for idx, finding in enumerate(criticals, 1):
            lines.append(f"- CRIT-{idx:03d} `{finding['check']}`: {finding['message']}")
    else:
        lines.append("- No critical findings.")
    lines.extend(["", "## 3. WARNING FINDINGS", ""])
    if warnings:
        for idx, finding in enumerate(warnings, 1):
            location = finding.get("location", "")
            prefix = f"{location}: " if location else ""
            lines.append(f"- WARN-{idx:03d} `{finding['check']}`: {prefix}{finding['message']}")
    else:
        lines.append("- No warning findings.")
    lines.extend(
        [
            "",
            "## 4. PER-DAY QUALITY TABLE",
            "",
            "| date | rows | seq_eligible | seq_complete | target_1s | target_3s | target_5s | target_15s | target_30s | criticals | warnings |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for date_str in dates:
        dataset = datasets[(date_str, "event64")]
        rows = max(int(dataset["rows"]), 1)
        date_findings = [finding for finding in findings if date_str in finding.get("location", "") or date_str in finding.get("message", "")]
        date_criticals = sum(1 for finding in date_findings if finding["severity"] == "CRITICAL")
        date_warnings = sum(1 for finding in date_findings if finding["severity"] == "WARNING")
        targets = dataset["target_counts"]
        lines.append(
            f"| {date_str} | {dataset['rows']} | {dataset['eligible_rate']:.4f} | {dataset['complete_rate']:.4f} | "
            f"{targets[1] / rows:.4f} | {targets[3] / rows:.4f} | {targets[5] / rows:.4f} | "
            f"{targets[15] / rows:.4f} | {targets[30] / rows:.4f} | {date_criticals} | {date_warnings} |"
        )
    lines.extend(
        [
            "",
            "## 5. SEQUENCE COMPLETENESS ANALYSIS",
            "",
            "| date | variant | p10 | p50 | p90 | frac_lt_32 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for date_str in dates:
        for variant in ("event64", "event128"):
            stats = sequence_by_date[(date_str, variant)]
            lines.append(
                f"| {date_str} | {variant} | {stats['p10']:.0f} | {stats['p50']:.0f} | "
                f"{stats['p90']:.0f} | {stats['frac_lt_32']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 6. TARGET AVAILABILITY ANALYSIS",
            "",
            "| date | 1s | 3s | 5s | 15s | 30s |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for date_str in dates:
        rates = target_rates[date_str]
        lines.append(f"| {date_str} | {rates[1]:.4f} | {rates[3]:.4f} | {rates[5]:.4f} | {rates[15]:.4f} | {rates[30]:.4f} |")
    lines.extend(["", "## 7. READINESS FOR PHASE 8 MODEL TRAINING", ""])
    if verdict == "FAIL":
        lines.append("Phase 8 dataset is **not ready for model training** because critical findings remain.")
    elif verdict == "PASS_WITH_WARNINGS":
        lines.append("Phase 8 dataset is ready for model training when sufficient data volume is accumulated, with non-blocking warnings to monitor.")
    else:
        lines.append("Phase 8 dataset factory is complete and training can proceed when sufficient data volume is accumulated.")
    lines.extend(["", "## 8. DECISION LOG ENTRIES", ""])
    lines.append("- See `docs/decision_log.md` for the Phase 8 post-fix decisions and completion entry.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"critical_count": len(criticals), "warning_count": len(warnings), "verdict": verdict, "output_path": str(output_path)}
