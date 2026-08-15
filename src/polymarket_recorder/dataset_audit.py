from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

FULL_NULL_RATIO = 1.0
NEAR_FULL_NULL_RATIO = 0.99
COUPLING_ANOMALY_THRESHOLD = 0.05
EXTREME_CROSS_SOURCE_SPREAD_USD = 100.0
DEAD_COLUMNS = (
    "btc_realized_vol_1s",
    "up_token_min_order_size",
    "down_token_min_order_size",
)
SCHEMA_FILE_BY_DATASET = {
    "resolution_snapshot_dataset_v1_coarse": Path("config/schemas/resolution_snapshot_dataset_v1_coarse_schema.yaml"),
    "resolution_snapshot_dataset_v1_dense_close": Path("config/schemas/resolution_snapshot_dataset_v1_dense_close_schema.yaml"),
    "market_interval_dataset_v1_preopen": Path("config/schemas/market_interval_dataset_v1_schema.yaml"),
    "market_interval_dataset_v1_first15s": Path("config/schemas/market_interval_dataset_v1_schema.yaml"),
    "market_interval_dataset_v1_first30s": Path("config/schemas/market_interval_dataset_v1_schema.yaml"),
}

KNOWN_LEAKAGE_COLUMNS = (
    "flip_happened_after_t",
    "terminal_delta_to_strike",
    "terminal_margin_to_strike",
    "hold_to_close_edge_vs_mid",
)

TRAINING_EXCLUSION_COLUMNS = (
    "training_feature_eligible",
    "feature_matrix_eligible",
    "row_quarantined",
    "quarantined_for_training",
    "training_excluded",
)

KNOWN_LIVE_REFERENCE_COLUMNS = (
    "live_btc_usd",
    "binance_spot_mid",
    "binance_usdm_mid",
    "hyperliquid_mid",
    "spot_perp_divergence_usd",
)

KNOWN_TOKEN_PRICE_COLUMNS = (
    "up_token_best_bid",
    "up_token_best_ask",
    "up_token_mid",
    "down_token_best_bid",
    "down_token_best_ask",
    "down_token_mid",
)

MANUAL_UPSTREAM_CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Policy compliance",
        (
            "Confirm the build consumed only finalized-clean, training-grade inputs and kept degraded slices in quarantine.",
            "Confirm verification-only layers stayed out of training joins and RTDS did not re-enter the truth path.",
            "Cross-check the build report and dataset policy rather than trusting parquet-level symptoms alone.",
        ),
    ),
    (
        "Label correctness",
        (
            "Confirm `price_to_beat` came only from `label_safe=true` strike rows.",
            "Confirm `resolved_side` came only from matched canonical resolution rows.",
            "Confirm market matching stayed on `market_slug` first with `condition_id` fallback only when needed.",
        ),
    ),
    (
        "Temporal / causal integrity",
        (
            "Confirm all cross-source joins were performed on local `recv_ts_ns` / canonical `ts_seconds` rather than source-native event timestamps.",
            "Confirm rows with missing live reference stayed `NaN` for `delta_to_strike` and did not silently fall back to another price feed.",
        ),
    ),
    (
        "Leakage boundary",
        (
            "Confirm future-derived columns remain diagnostic-only and never enter any training feature matrix.",
            "If auto-feature selection exists downstream, add an explicit denylist for leakage-sensitive columns.",
        ),
    ),
)


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("boolean").fillna(False).astype(bool)


def _resolve_schema_contract_path(dataset_path: Path) -> Path | None:
    dataset_name = dataset_path.parent.name
    relative = SCHEMA_FILE_BY_DATASET.get(dataset_name)
    if relative is None:
        return None
    for parent in dataset_path.parents:
        candidate = parent / relative
        if candidate.exists():
            return candidate
    return dataset_path.parents[0] / relative


def _load_schema_contract(dataset_path: Path) -> dict[str, Any] | None:
    schema_path = _resolve_schema_contract_path(dataset_path)
    if schema_path is None or not schema_path.exists():
        return None
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _hour_bucket(frame: pd.DataFrame) -> pd.Series:
    if "snapshot_ts_iso" in frame.columns:
        return pd.to_datetime(frame["snapshot_ts_iso"], utc=True).dt.floor("h")
    if "snapshot_ts_ns" in frame.columns:
        return pd.to_datetime(frame["snapshot_ts_ns"], utc=True, unit="ns").dt.floor("h")
    raise KeyError("Dataset does not contain snapshot_ts_iso or snapshot_ts_ns")


def _null_profile(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_null: list[dict[str, Any]] = []
    near_full_null: list[dict[str, Any]] = []
    for column in frame.columns:
        ratio = float(frame[column].isna().mean())
        item = {"column": column, "null_ratio": ratio}
        if ratio >= FULL_NULL_RATIO:
            full_null.append(item)
        elif ratio >= NEAR_FULL_NULL_RATIO:
            near_full_null.append(item)
    full_null.sort(key=lambda item: item["column"])
    near_full_null.sort(key=lambda item: (-item["null_ratio"], item["column"]))
    return full_null, near_full_null


def _live_reference_profile(frame: pd.DataFrame) -> dict[str, Any]:
    if "live_price_valid" not in frame.columns or "live_source_count" not in frame.columns:
        return {
            "available": False,
            "column_null_rates": [],
            "source_count_distribution": [],
            "degraded_true_count": 0,
            "dead_hours": [],
            "partial_invalid_hours": [],
            "all_invalid_markets": [],
        }

    training_eligible = _bool_series(frame["training_feature_eligible"]) if "training_feature_eligible" in frame.columns else pd.Series(False, index=frame.index)
    hourly = frame.assign(_hour=_hour_bucket(frame), _training_eligible=training_eligible).groupby("_hour", dropna=False).agg(
        rows=("live_source_count", "size"),
        live_source_zero=("live_source_count", lambda series: int((series.fillna(0) == 0).sum())),
        live_price_invalid=("live_price_valid", lambda series: int((~_bool_series(series)).sum())),
        training_eligible_rows=("_training_eligible", lambda series: int(_bool_series(series).sum())),
    )
    dead_hours = []
    partial_invalid_hours = []
    for hour, row in hourly.iterrows():
        item = {
            "hour_utc": str(hour),
            "rows": int(row["rows"]),
            "live_source_zero_rows": int(row["live_source_zero"]),
            "live_price_invalid_rows": int(row["live_price_invalid"]),
            "training_eligible_rows": int(row["training_eligible_rows"]),
        }
        if item["live_source_zero_rows"] == item["rows"]:
            dead_hours.append(item)
        elif item["live_source_zero_rows"] > 0 or item["live_price_invalid_rows"] > 0:
            partial_invalid_hours.append(item)

    market_invalid = frame.groupby("market_slug", dropna=False).agg(
        rows=("market_slug", "size"),
        invalid_rows=("live_price_valid", lambda series: int((~_bool_series(series)).sum())),
        first_snapshot=("snapshot_ts_iso", "min" if "snapshot_ts_iso" in frame.columns else "size"),
        last_snapshot=("snapshot_ts_iso", "max" if "snapshot_ts_iso" in frame.columns else "size"),
    )
    market_invalid["invalid_fraction"] = market_invalid["invalid_rows"] / market_invalid["rows"]
    market_invalid["training_eligible_rows"] = training_eligible.groupby(frame["market_slug"], dropna=False).sum().astype("int64")
    all_invalid_markets = [
        {
            "market_slug": str(index),
            "rows": int(row["rows"]),
            "invalid_rows": int(row["invalid_rows"]),
            "first_snapshot": str(row["first_snapshot"]),
            "last_snapshot": str(row["last_snapshot"]),
            "training_eligible_rows": int(row["training_eligible_rows"]),
        }
        for index, row in market_invalid[market_invalid["invalid_fraction"] >= 1.0].sort_index().iterrows()
    ]

    return {
        "available": True,
        "column_null_rates": [
            {"column": column, "null_ratio": float(frame[column].isna().mean())}
            for column in KNOWN_LIVE_REFERENCE_COLUMNS
            if column in frame.columns
        ],
        "source_count_distribution": [
            {"source_count": int(key), "rows": int(value)}
            for key, value in sorted(frame["live_source_count"].fillna(0).astype("int64").value_counts().items())
        ],
        "degraded_true_count": int(_bool_series(frame.get("live_price_degraded", pd.Series(False, index=frame.index))).sum()),
        "dead_hours": dead_hours,
        "partial_invalid_hours": partial_invalid_hours,
        "all_invalid_markets": all_invalid_markets,
    }


def _microstructure_profile(frame: pd.DataFrame) -> dict[str, Any]:
    training_eligible = _bool_series(frame["training_feature_eligible"]) if "training_feature_eligible" in frame.columns else pd.Series(False, index=frame.index)
    price_range_violations = []
    for column in KNOWN_TOKEN_PRICE_COLUMNS:
        if column not in frame.columns:
            continue
        valid = frame[column].dropna()
        out_of_range = int(((valid < 0.0) | (valid > 1.0)).sum())
        price_range_violations.append({"column": column, "out_of_range_rows": out_of_range})

    crossed_book_rows = []
    for prefix in ("up_token", "down_token"):
        bid_column = f"{prefix}_best_bid"
        ask_column = f"{prefix}_best_ask"
        if bid_column not in frame.columns or ask_column not in frame.columns:
            continue
        mask = frame[bid_column].notna() & frame[ask_column].notna() & (frame[bid_column] >= frame[ask_column])
        sample = frame.loc[mask, ["market_slug", "snapshot_ts_iso", bid_column, ask_column]].head(10)
        crossed_book_rows.append(
            {
                "prefix": prefix,
                "count": int(mask.sum()),
                "training_eligible_rows": int(training_eligible.loc[mask].sum()),
                "examples": sample.to_dict(orient="records"),
            }
        )

    coupling_examples: list[dict[str, Any]] = []
    coupling_count = 0
    flagged_coupling_count = 0
    if {"up_token_mid", "down_token_mid"}.issubset(frame.columns):
        coupling_delta = frame["up_token_mid"] + frame["down_token_mid"] - 1.0
        mask = coupling_delta.abs() > COUPLING_ANOMALY_THRESHOLD
        coupling_count = int(mask.sum())
        if "coupling_anomaly" in frame.columns:
            flagged_coupling_count = int(_bool_series(frame["coupling_anomaly"]).loc[mask].sum())
        if coupling_count:
            sample_columns = [
                column
                for column in (
                    "market_slug",
                    "snapshot_ts_iso",
                    "up_token_mid",
                    "down_token_mid",
                    "yes_no_coupling_residual",
                )
                if column in frame.columns
            ]
            sample = frame.loc[mask, sample_columns].copy()
            if "yes_no_coupling_residual" not in sample.columns:
                sample["yes_no_coupling_residual"] = coupling_delta[mask]
            sample = sample.assign(_abs_residual=sample["yes_no_coupling_residual"].abs()).sort_values(
                "_abs_residual",
                ascending=False,
                kind="stable",
            )
            coupling_examples = sample.drop(columns="_abs_residual").head(10).to_dict(orient="records")

    extreme_cross_source_count = 0
    if "cross_source_spread_usd" in frame.columns:
        values = frame["cross_source_spread_usd"].dropna()
        extreme_cross_source_count = int((values >= EXTREME_CROSS_SOURCE_SPREAD_USD).sum())

    return {
        "price_range_violations": price_range_violations,
        "crossed_book_rows": crossed_book_rows,
        "coupling_anomaly_count": coupling_count,
        "flagged_coupling_rows": flagged_coupling_count,
        "coupling_anomaly_examples": coupling_examples,
        "extreme_cross_source_spread_rows": extreme_cross_source_count,
    }


def _consistency_profile(frame: pd.DataFrame) -> dict[str, Any]:
    duplicate_count = 0
    if {"market_slug", "snapshot_ts_ns"}.issubset(frame.columns):
        duplicate_count = int(frame.duplicated(["market_slug", "snapshot_ts_ns"]).sum())

    market_resolved_side_inconsistent = 0
    if {"market_slug", "resolved_side"}.issubset(frame.columns):
        market_resolved_side_inconsistent = int(
            (frame.groupby("market_slug")["resolved_side"].nunique(dropna=True) > 1).sum()
        )

    market_price_to_beat_inconsistent = 0
    if {"market_slug", "price_to_beat"}.issubset(frame.columns):
        market_price_to_beat_inconsistent = int(
            (frame.groupby("market_slug")["price_to_beat"].nunique(dropna=True) > 1).sum()
        )

    delta_max_abs_error = None
    if {"live_btc_usd", "price_to_beat", "delta_to_strike"}.issubset(frame.columns):
        expected = frame["live_btc_usd"] - frame["price_to_beat"]
        mask = ~(frame["delta_to_strike"].isna() & expected.isna())
        if bool(mask.any()):
            delta_max_abs_error = float((frame.loc[mask, "delta_to_strike"] - expected[mask]).abs().max())
        else:
            delta_max_abs_error = 0.0

    abs_delta_max_abs_error = None
    if {"delta_to_strike", "abs_delta_to_strike"}.issubset(frame.columns):
        expected = frame["delta_to_strike"].abs()
        mask = ~(frame["abs_delta_to_strike"].isna() & expected.isna())
        if bool(mask.any()):
            abs_delta_max_abs_error = float((frame.loc[mask, "abs_delta_to_strike"] - expected[mask]).abs().max())
        else:
            abs_delta_max_abs_error = 0.0

    return {
        "duplicate_market_snapshot_count": duplicate_count,
        "market_resolved_side_inconsistent_count": market_resolved_side_inconsistent,
        "market_price_to_beat_inconsistent_count": market_price_to_beat_inconsistent,
        "delta_to_strike_max_abs_error": delta_max_abs_error,
        "abs_delta_to_strike_max_abs_error": abs_delta_max_abs_error,
    }


def _build_finding(
    *,
    severity: str,
    problem: str,
    evidence: str,
    why_it_matters: str,
    exact_fix: str,
    acceptance_criteria: str,
) -> dict[str, str]:
    return {
        "severity": severity,
        "problem": problem,
        "evidence": evidence,
        "why_it_matters": why_it_matters,
        "exact_fix": exact_fix,
        "acceptance_criteria": acceptance_criteria,
    }


def _derive_findings(report: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    schema_columns = dict((report.get("schema_contract") or {}).get("columns") or {})

    forbidden_fng_columns = [column for column in report["policy"]["fng_columns_present"]]
    if forbidden_fng_columns:
        findings.append(
            _build_finding(
                severity="high",
                problem="FNG columns are still present in the Phase 3 parquet",
                evidence=", ".join(f"`{column}`" for column in forbidden_fng_columns),
                why_it_matters="The approved source stack excludes FNG entirely, so any surviving `fng_*` field means the rebuild contract was not respected.",
                exact_fix="Remove FNG from the dataset factory and abort the build whenever any `fng_*` column appears in output.",
                acceptance_criteria="The next audit shows zero `fng_*` columns in the dataset.",
            )
        )

    full_null_columns = [item["column"] for item in report["null_profile"]["full_null_columns"]]
    dead_columns_present = [column for column in DEAD_COLUMNS if column in report["summary"]["column_names"]]
    if dead_columns_present:
        findings.append(
            _build_finding(
                severity="medium",
                problem="Dead columns that should have been removed are still present in the parquet",
                evidence=", ".join(f"`{column}`" for column in dead_columns_present),
                why_it_matters="These columns were explicitly removed from the Phase 3 contract, so their presence means the dataset was not rebuilt to the current schema.",
                exact_fix="Remove the dead columns from the builder output and schema contracts, then rebuild the dataset.",
                acceptance_criteria="The next audit shows zero dead columns in the parquet.",
            )
        )
    trainable_full_null_columns = [
        column
        for column in full_null_columns
        if bool((schema_columns.get(column) or {}).get("training_eligible", True))
    ]
    if trainable_full_null_columns:
        findings.append(
            _build_finding(
                severity="medium",
                problem="Fully null feature columns are present in the parquet",
                evidence=", ".join(f"`{column}`" for column in trainable_full_null_columns),
                why_it_matters="Dead columns create fake feature surface area, confuse feature selection, and hide builder bugs or design mismatches.",
                exact_fix="Either remove these columns from the schema for this dataset view or backfill them from the intended source with a documented causal contract.",
                acceptance_criteria="The next audit shows zero fully-null columns that are still advertised as trainable features.",
            )
        )

    live_reference = report["live_reference"]
    dead_hours_with_training_rows = [item for item in live_reference["dead_hours"] if int(item["training_eligible_rows"]) > 0]
    if live_reference["dead_hours"] and (dead_hours_with_training_rows or not report["leakage"]["has_explicit_training_exclusion_field"]):
        dead_hours = ", ".join(f"`{item['hour_utc']}`" for item in live_reference["dead_hours"])
        findings.append(
            _build_finding(
                severity="high",
                problem="Entire live-reference-dead hour(s) reached the modeling dataset",
                evidence=dead_hours,
                why_it_matters="Rows with `live_source_count=0` across a whole hour cannot support `delta_to_strike`-driven modeling and should not silently survive as trainable rows.",
                exact_fix="Quarantine dead hours from training-eligible rows or add an explicit row-level training exclusion flag that downstream code must honor.",
                acceptance_criteria="Training-eligible rows contain zero dead live-reference hours, and the audit report names any excluded hours explicitly.",
            )
        )

    invalid_markets_with_training_rows = [
        item for item in live_reference["all_invalid_markets"] if int(item["training_eligible_rows"]) > 0
    ]
    if live_reference["all_invalid_markets"] and (
        invalid_markets_with_training_rows or not report["leakage"]["has_explicit_training_exclusion_field"]
    ):
        findings.append(
            _build_finding(
                severity="high",
                problem="Some markets are fully present in the shard but have no valid live BTC reference in any row",
                evidence=f"`{len(live_reference['all_invalid_markets'])}` markets are 100% invalid for `live_price_valid`.",
                why_it_matters="These rows cannot support strike-relative features and need explicit quarantine or downstream exclusion to avoid accidental training leakage through missingness patterns.",
                exact_fix="Either exclude fully-invalid markets from the trainable output or add a first-class eligibility flag with a documented exclusion rule.",
                acceptance_criteria="All trainable markets have at least one valid live reference row or are explicitly marked non-trainable.",
            )
        )

    leakage_contract_ok = bool(report["leakage"]["schema_contract_ok"])
    if report["leakage"]["leakage_columns_present"] and not leakage_contract_ok:
        findings.append(
            _build_finding(
                severity="high",
                problem="Future-derived diagnostic columns are present without an explicit feature-matrix exclusion field",
                evidence=", ".join(f"`{column}`" for column in report["leakage"]["leakage_columns_present"]),
                why_it_matters="These columns are valid for diagnostics, but they become dangerous if downstream feature selection ever auto-includes all numeric columns.",
                exact_fix="Introduce an explicit training feature eligibility contract or a hard feature denylist for leakage-sensitive columns in every downstream model pipeline.",
                acceptance_criteria="Leakage-sensitive columns are either absent from the trainable matrix by contract or a first-class exclusion flag is present and enforced in tests.",
            )
        )

    micro = report["microstructure"]
    unflagged_coupling_rows = micro["coupling_anomaly_count"] - micro["flagged_coupling_rows"]
    if unflagged_coupling_rows > 0:
        findings.append(
            _build_finding(
                severity="medium",
                problem="Cross-side coupling anomalies remain in the parquet",
                evidence=f"`{unflagged_coupling_rows}` row(s) exceed the |up_mid + down_mid - 1| > {COUPLING_ANOMALY_THRESHOLD:.2f} threshold without being fully flagged out of training.",
                why_it_matters="Large residuals usually indicate stale cross-side state, snapshot mismatch, or malformed market reconstruction and can distort pricing-derived targets.",
                exact_fix="Flag these rows as coupling anomalies and either quarantine them from training or document a clipping / exclusion rule.",
                acceptance_criteria="No unflagged coupling anomalies above the configured threshold remain in trainable rows.",
            )
        )

    crossed_book_training_rows = sum(item["training_eligible_rows"] for item in micro["crossed_book_rows"])
    if crossed_book_training_rows > 0:
        findings.append(
            _build_finding(
                severity="medium",
                problem="Crossed token books remain training-eligible in the shard",
                evidence=f"`{crossed_book_training_rows}` row(s) have best_bid >= best_ask across the token books while still marked training-eligible.",
                why_it_matters="Crossed books can be real for a moment, but repeated or unflagged crossed states can poison microstructure features and mid-price construction.",
                exact_fix="Audit whether these are legitimate crossed snapshots or stale-state mismatches, then either preserve them with an anomaly flag or quarantine them from training.",
                acceptance_criteria="Every crossed-book row is either justified as a real market state or excluded / flagged in the trainable dataset.",
            )
        )

    return findings


def audit_phase3_dataset(dataset_path: Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    frame = pd.read_parquet(dataset_path)
    schema_contract = _load_schema_contract(dataset_path)
    full_null_columns, near_full_null_columns = _null_profile(frame)
    live_reference = _live_reference_profile(frame)
    microstructure = _microstructure_profile(frame)
    consistency = _consistency_profile(frame)

    summary = {
        "dataset_path": str(dataset_path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "unique_markets": int(frame["market_slug"].nunique(dropna=True)) if "market_slug" in frame.columns else 0,
        "sample_family_values": sorted(frame["sample_family"].dropna().astype(str).unique().tolist())
        if "sample_family" in frame.columns
        else [],
        "sample_bucket_ms_values": sorted(frame["sample_bucket_ms"].dropna().astype("int64").unique().tolist())
        if "sample_bucket_ms" in frame.columns
        else [],
        "snapshot_ts_iso_min": str(frame["snapshot_ts_iso"].min()) if "snapshot_ts_iso" in frame.columns else "",
        "snapshot_ts_iso_max": str(frame["snapshot_ts_iso"].max()) if "snapshot_ts_iso" in frame.columns else "",
        "column_names": list(frame.columns),
    }

    leakage_columns_present = [column for column in KNOWN_LEAKAGE_COLUMNS if column in frame.columns]
    has_explicit_training_exclusion_field = any(column in frame.columns for column in TRAINING_EXCLUSION_COLUMNS)
    schema_columns = dict((schema_contract or {}).get("columns") or {})
    schema_leakage_columns = [column for column, meta in schema_columns.items() if bool((meta or {}).get("leakage_only"))]
    schema_training_columns = [column for column, meta in schema_columns.items() if bool((meta or {}).get("training_eligible"))]
    schema_leakage_overlap = sorted(set(schema_leakage_columns) & set(schema_training_columns))
    fng_columns_present = [column for column in frame.columns if "fng" in column.lower()]

    passes: list[str] = []
    if consistency["duplicate_market_snapshot_count"] == 0:
        passes.append("`market_slug + snapshot_ts_ns` has no duplicate rows.")
    if consistency["market_resolved_side_inconsistent_count"] == 0:
        passes.append("`resolved_side` is internally consistent within each market.")
    if consistency["market_price_to_beat_inconsistent_count"] == 0:
        passes.append("`price_to_beat` is internally consistent within each market.")
    if consistency["delta_to_strike_max_abs_error"] in (0.0, None):
        passes.append("`delta_to_strike` matches `live_btc_usd - price_to_beat` exactly where both are available.")
    if consistency["abs_delta_to_strike_max_abs_error"] in (0.0, None):
        passes.append("`abs_delta_to_strike` matches `abs(delta_to_strike)` exactly where both are available.")
    if all(item["out_of_range_rows"] == 0 for item in microstructure["price_range_violations"]):
        passes.append("All token price-like columns stay inside the `[0, 1]` range.")
    if not fng_columns_present:
        passes.append("No forbidden `fng_*` columns are present in the dataset.")
    if not any(column in summary["column_names"] for column in DEAD_COLUMNS):
        passes.append("All requested dead columns were removed from the dataset schema.")
    if leakage_columns_present and not schema_leakage_overlap and has_explicit_training_exclusion_field:
        passes.append("Leakage-sensitive columns remain present only as diagnostic fields outside the training feature contract.")
    if all(item["training_eligible_rows"] == 0 for item in live_reference["dead_hours"]):
        passes.append("Dead live-reference hours, when present, are fully excluded from training-eligible rows.")
    if all(item["training_eligible_rows"] == 0 for item in live_reference["all_invalid_markets"]):
        passes.append("Markets without any valid live reference rows are fully excluded from training-eligible rows.")
    if microstructure["coupling_anomaly_count"] == microstructure["flagged_coupling_rows"]:
        passes.append("All detected coupling anomalies are explicitly flagged.")
    if all(item["training_eligible_rows"] == 0 for item in microstructure["crossed_book_rows"]):
        passes.append("All crossed-book rows are excluded from the training-eligible subset.")

    report = {
        "summary": summary,
        "schema_contract": schema_contract,
        "consistency": consistency,
        "null_profile": {
            "full_null_columns": full_null_columns,
            "near_full_null_columns": near_full_null_columns,
        },
        "policy": {
            "fng_columns_present": fng_columns_present,
            "dead_columns_present": [column for column in DEAD_COLUMNS if column in summary["column_names"]],
        },
        "live_reference": live_reference,
        "microstructure": microstructure,
        "leakage": {
            "leakage_columns_present": leakage_columns_present,
            "has_explicit_training_exclusion_field": has_explicit_training_exclusion_field,
            "schema_contract_ok": not schema_leakage_overlap and all(
                not bool((schema_columns.get(column) or {}).get("training_eligible"))
                for column in leakage_columns_present
            ) if schema_columns else False,
            "schema_leakage_overlap": schema_leakage_overlap,
            "training_exclusion_columns_present": [
                column for column in TRAINING_EXCLUSION_COLUMNS if column in frame.columns
            ],
        },
        "passes": passes,
        "manual_upstream_checks": [
            {"area": area, "checks": list(checks)} for area, checks in MANUAL_UPSTREAM_CHECKS
        ],
    }
    report["findings"] = _derive_findings(report)
    return report


def render_phase3_audit_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Phase 3 Dataset Audit",
        "",
        f"- Dataset: `{summary['dataset_path']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Columns: `{summary['columns']}`",
        f"- Unique markets: `{summary['unique_markets']}`",
        f"- Sample family: `{', '.join(summary['sample_family_values']) or 'n/a'}`",
        f"- Sample bucket ms: `{', '.join(str(value) for value in summary['sample_bucket_ms_values']) or 'n/a'}`",
        f"- Snapshot range: `{summary['snapshot_ts_iso_min']}` -> `{summary['snapshot_ts_iso_max']}`",
        "",
        "## Automated Passes",
        "",
    ]
    if report["passes"]:
        lines.extend(f"- {item}" for item in report["passes"])
    else:
        lines.append("- No automatic green flags were recorded.")

    lines.extend(
        [
            "",
            "## Findings",
            "",
        ]
    )
    if report["findings"]:
        for index, finding in enumerate(report["findings"], start=1):
            lines.extend(
                [
                    f"### {index}. {finding['problem']} [{finding['severity'].upper()}]",
                    "",
                    f"- Evidence: {finding['evidence']}",
                    f"- Why it matters: {finding['why_it_matters']}",
                    f"- Exact fix: {finding['exact_fix']}",
                    f"- Acceptance criteria: {finding['acceptance_criteria']}",
                    "",
                ]
            )
    else:
        lines.append("- No audit findings were generated.")

    lines.extend(
        [
            "## Null Profile",
            "",
        ]
    )
    if report["null_profile"]["full_null_columns"]:
        lines.append("### Fully Null Columns")
        lines.append("")
        for item in report["null_profile"]["full_null_columns"]:
            lines.append(f"- `{item['column']}`")
        lines.append("")
    if report["null_profile"]["near_full_null_columns"]:
        lines.append("### Near-Fully Null Columns")
        lines.append("")
        for item in report["null_profile"]["near_full_null_columns"][:20]:
            lines.append(f"- `{item['column']}` -> `{_pct(item['null_ratio'])}` null")
        lines.append("")
    if not report["null_profile"]["full_null_columns"] and not report["null_profile"]["near_full_null_columns"]:
        lines.append("- No fully-null or near-fully-null columns were detected.")
        lines.append("")

    lines.extend(
        [
            "## Live Reference Coverage",
            "",
        ]
    )
    live_reference = report["live_reference"]
    if live_reference["available"]:
        lines.append("### Column Null Rates")
        lines.append("")
        for item in live_reference["column_null_rates"]:
            lines.append(f"- `{item['column']}` -> `{_pct(item['null_ratio'])}` null")
        lines.append("")
        lines.append("### Source Count Distribution")
        lines.append("")
        for item in live_reference["source_count_distribution"]:
            lines.append(f"- `source_count={item['source_count']}` -> `{item['rows']}` rows")
        lines.append(f"- `live_price_degraded=True` rows -> `{live_reference['degraded_true_count']}`")
        lines.append("")
        if live_reference["dead_hours"]:
            lines.append("### Dead Hours")
            lines.append("")
            for item in live_reference["dead_hours"]:
                lines.append(
                    f"- `{item['hour_utc']}` -> `{item['live_source_zero_rows']}` / `{item['rows']}` rows with `live_source_count=0`"
                )
            lines.append("")
        if live_reference["partial_invalid_hours"]:
            lines.append("### Partial Invalid Hours")
            lines.append("")
            for item in live_reference["partial_invalid_hours"][:20]:
                lines.append(
                    f"- `{item['hour_utc']}` -> `{item['live_source_zero_rows']}` zero-source rows, `{item['live_price_invalid_rows']}` invalid-price rows out of `{item['rows']}`"
                )
            lines.append("")
        if live_reference["all_invalid_markets"]:
            lines.append("### Fully Invalid Markets")
            lines.append("")
            for item in live_reference["all_invalid_markets"][:20]:
                lines.append(
                    f"- `{item['market_slug']}` -> `{item['invalid_rows']}` / `{item['rows']}` invalid rows (`{item['first_snapshot']}` -> `{item['last_snapshot']}`)"
                )
            lines.append("")
    else:
        lines.append("- No live reference coverage columns were present in the dataset.")
        lines.append("")

    lines.extend(
        [
            "## Leakage Boundary",
            "",
            f"- Leakage-sensitive columns present: `{', '.join(report['leakage']['leakage_columns_present']) or 'none'}`",
            f"- Leakage columns present in schema contract: `{', '.join(report['leakage']['schema_leakage_overlap']) or 'none'}`",
            f"- Explicit training exclusion columns present: `{', '.join(report['leakage']['training_exclusion_columns_present']) or 'none'}`",
            f"- FNG columns present: `{', '.join(report['policy']['fng_columns_present']) or 'none'}`",
            f"- Dead columns present: `{', '.join(report['policy']['dead_columns_present']) or 'none'}`",
            "",
            "## Microstructure Sanity",
            "",
        ]
    )
    for item in report["microstructure"]["price_range_violations"]:
        lines.append(f"- `{item['column']}` out-of-range rows -> `{item['out_of_range_rows']}`")
    for item in report["microstructure"]["crossed_book_rows"]:
        lines.append(
            f"- `{item['prefix']}` crossed-book rows -> `{item['count']}` total, `{item['training_eligible_rows']}` still training-eligible"
        )
    lines.append(
        f"- Coupling anomaly rows above threshold -> `{report['microstructure']['coupling_anomaly_count']}` total, `{report['microstructure']['flagged_coupling_rows']}` flagged"
    )
    lines.append(
        f"- Extreme cross-source spread rows >= `{EXTREME_CROSS_SOURCE_SPREAD_USD}` USD -> `{report['microstructure']['extreme_cross_source_spread_rows']}`"
    )
    lines.append("")
    if report["microstructure"]["coupling_anomaly_examples"]:
        lines.append("### Coupling Anomaly Examples")
        lines.append("")
        for item in report["microstructure"]["coupling_anomaly_examples"][:10]:
            lines.append(f"- `{json.dumps(item, sort_keys=True)}`")
        lines.append("")
    crossed_examples = [
        example
        for group in report["microstructure"]["crossed_book_rows"]
        for example in group["examples"]
    ]
    if crossed_examples:
        lines.append("### Crossed-Book Examples")
        lines.append("")
        for item in crossed_examples[:10]:
            lines.append(f"- `{json.dumps(item, sort_keys=True)}`")
        lines.append("")

    lines.extend(
        [
            "## Manual Upstream Checks",
            "",
            "These checks are required to close the audit, but they cannot be proven from the parquet alone.",
            "",
        ]
    )
    for area in report["manual_upstream_checks"]:
        lines.append(f"### {area['area']}")
        lines.append("")
        for item in area["checks"]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_phase3_audit_report(*, dataset_path: Path, output_path: Path) -> dict[str, Any]:
    report = audit_phase3_dataset(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_phase3_audit_markdown(report), encoding="utf-8")
    return report
