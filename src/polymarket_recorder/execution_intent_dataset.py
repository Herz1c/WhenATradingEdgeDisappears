from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import time
import tracemalloc
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from chainlink_recorder.live_reference_events import LiveReferenceEventsReader

from .execution_audit import audit_execution_results
from .execution_ledger import reconcile_execution_results
from .execution_replay import SECOND_NS, load_execution_replay_for_date
from .execution_scenarios import ScenarioGrid, generate_intents_from_anchors, iter_market_slugs
from .execution_simulator import ExecutionSimulator, ExecutionSimulatorConfig, FeeSchedule, RESULT_COLUMNS
from .l2_event_loader import DEFAULT_QUALITY_REGISTRY_PATH

DATASET_NAME = "execution_intent_dataset_v1"
DATASET_VERSION = "execution_intent_dataset_v1"
DEFAULT_DATASET_ROOT = Path("data/datasets")
DEFAULT_SCHEMA_PATH = Path("config/schemas/execution_intent_dataset_v1_schema.yaml")
DEFAULT_BUILD_REPORT_PATH = Path("docs/execution_intent_dataset_v1_build_report.md")
DEFAULT_DECISION_LOG_PATH = Path("docs/decision_log.md")

PHASE8_TABULAR_DATASET = "microstructure_sequence_dataset_v1_tabular"
MARKET_REGISTRY_PATH = Path("canonical/market_registry_v1.parquet")
TOKEN_REGISTRY_PATH = Path("canonical/token_registry_v1.parquet")

PHASE8_TARGET_COLUMNS = {
    "mid_move_1s",
    "mid_move_3s",
    "mid_move_5s",
    "mid_move_15s",
    "mid_move_30s",
    "mid_move_sign_1s",
    "mid_move_sign_3s",
    "mid_move_sign_5s",
    "mid_move_sign_15s",
    "mid_move_sign_30s",
    "max_favorable_excursion_5s",
    "max_adverse_excursion_5s",
    "max_favorable_excursion_15s",
    "max_adverse_excursion_15s",
    "hold_to_close_edge_vs_mid",
    "terminal_margin_to_strike",
    "flip_happened_after_t",
}

ANCHOR_FEATURE_COLUMNS = [
    "anchor_sequence_id",
    "anchor_source",
    "anchor_feature_eligible",
    "anchor_sequence_complete_eligible",
    "phase8_microstructure_exclusion_reason",
    "token_registry_match",
    "market_registry_trainable",
    "event_count_1s",
    "event_count_3s",
    "event_count_5s",
    "event_count_15s",
    "quote_update_count_1s",
    "quote_update_count_3s",
    "quote_update_count_5s",
    "trade_count_1s",
    "trade_count_3s",
    "trade_count_5s",
    "signed_trade_size_1s",
    "signed_trade_size_3s",
    "signed_trade_size_5s",
    "absolute_trade_size_1s",
    "absolute_trade_size_3s",
    "absolute_trade_size_5s",
    "spread_mean_5s",
    "spread_min_5s",
    "spread_max_5s",
    "imbalance_mean_5s",
    "imbalance_last",
    "imbalance_slope_5s",
    "mid_return_1s",
    "mid_return_3s",
    "mid_return_5s",
    "quote_churn_rate_5s",
    "depth_change_rate_5s",
    "sequence_completeness_rate",
    "price_to_beat",
    "delta_to_strike",
    "abs_delta_to_strike",
    "delta_sign",
    "t_since_open_s",
    "t_to_close_s",
    "phase_bucket",
]

METADATA_COLUMNS = [
    "dataset_version",
    "source_anchor_dataset",
    "target_date",
    "build_id",
    "policy_path",
    "quality_registry_path",
    "input_policy_compliant",
    "execution_training_eligible",
]

DATASET_COLUMNS = METADATA_COLUMNS + ANCHOR_FEATURE_COLUMNS + RESULT_COLUMNS

OUTCOME_COLUMNS = {
    "resolved_side",
    "cancel_request_ts_ns",
    "cancel_effective_ts_ns",
    "replace_requested",
    "replace_request_ts_ns",
    "replace_effective_ts_ns",
    "replaced",
    "replace_post_only_rejected",
    "submitted",
    "post_only_rejected",
    "post_only_crossed_book",
    "filled",
    "full_fill",
    "partial_fill",
    "fill_uncertain",
    "fill_ts_ns",
    "last_fill_ts_ns",
    "time_to_fill_s",
    "time_to_last_fill_s",
    "fill_count",
    "fill_price",
    "fill_size",
    "fill_notional",
    "maker_fill",
    "taker_fill",
    "cancelled",
    "cancel_effective",
    "fees_paid",
    "rebate_earned",
    "gross_pnl_to_close",
    "net_pnl_to_close",
    "markout_1s",
    "markout_3s",
    "markout_5s",
    "markout_15s",
    "markout_to_close",
    "markout_1s_available",
    "markout_3s_available",
    "markout_5s_available",
    "markout_15s_available",
    "markout_to_close_available",
    "adverse_selection_5s",
    "toxic_fill_5s",
}

NON_FEATURE_IDENTITY_COLUMNS = {
    "dataset_version",
    "source_anchor_dataset",
    "target_date",
    "build_id",
    "policy_path",
    "quality_registry_path",
    "input_policy_compliant",
    "simulator_id",
    "intent_id",
    "source_sequence_id",
    "anchor_sequence_id",
    "scenario_name",
    "order_ts_ns",
    "market_close_ts_ns",
    "state_recv_ts_ns",
    "live_reference_recv_ts_ns",
    "fee_schedule_id",
    "maker_fee_bps",
    "taker_fee_bps",
    "maker_rebate_bps",
    "taker_rebate_bps",
    "execution_exclusion_reason",
}


@dataclass(frozen=True, slots=True)
class ExecutionIntentBuildConfig:
    build_mode: str = "production"
    max_anchors_per_day: int | None = None
    max_intents_per_day: int | None = None
    max_markets_per_day: int | None = None
    anchor_batch_size: int = 2_000
    max_workers: int = 0
    partition_by_market: bool = True
    market_batch_size: int = 16
    resume_existing: bool = False
    fee_schedule: FeeSchedule = FeeSchedule()
    cancel_latency_ns: int = 0
    replace_latency_ns: int = 0
    stale_book_threshold_ns: int = 30 * SECOND_NS
    queue_ahead_fraction: float = 0.0
    order_size: float = 10.0
    require_live_reference: bool = True
    scenario_name: str = DATASET_NAME + "_grid_v1"

    def __post_init__(self) -> None:
        mode = str(self.build_mode).strip().lower()
        if mode not in {"production", "capped"}:
            raise ValueError("ExecutionIntentBuildConfig.build_mode must be 'production' or 'capped'")
        object.__setattr__(self, "build_mode", mode)
        if self.anchor_batch_size <= 0:
            raise ValueError("anchor_batch_size must be positive")
        if self.max_workers < 0:
            raise ValueError("max_workers must be >= 0")
        if self.market_batch_size <= 0:
            raise ValueError("market_batch_size must be positive")
        if mode == "production":
            if any(
                value is not None and int(value) > 0
                for value in (self.max_anchors_per_day, self.max_intents_per_day, self.max_markets_per_day)
            ):
                raise ValueError("production mode must be uncapped; use build_mode='capped' for debug/audit caps")


def _worker_count(configured: int, task_count: int) -> int:
    workers = (os.cpu_count() or 1) if configured == 0 else int(configured)
    return max(1, min(workers, max(task_count, 1)))


def _date_range(date_from: date, date_to: date) -> list[date]:
    if date_to < date_from:
        raise ValueError("date_to must be >= date_from")
    days = (date_to - date_from).days
    return [date_from + timedelta(days=offset) for offset in range(days + 1)]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_id_for_config(
    *,
    date_from: date,
    date_to: date,
    config: ExecutionIntentBuildConfig,
    policy_path: Path,
    quality_registry_path: Path,
) -> str:
    key = "|".join(
        [
            DATASET_VERSION,
            config.build_mode,
            str(config.max_anchors_per_day),
            str(config.max_intents_per_day),
            str(config.max_markets_per_day),
            str(config.anchor_batch_size),
            str(config.partition_by_market),
            str(config.market_batch_size),
            str(config.order_size),
            config.fee_schedule.schedule_id,
            str(config.fee_schedule.maker_fee_bps),
            str(config.fee_schedule.taker_fee_bps),
            str(config.fee_schedule.maker_rebate_bps),
            str(config.fee_schedule.post_cutoff_fee_rate),
            str(config.fee_schedule.post_cutoff_exponent),
            str(policy_path),
            str(quality_registry_path),
        ]
    )
    return hashlib.blake2b(key.encode("utf-8"), digest_size=12).hexdigest()


def _dtype_for_column(column: str) -> str:
    if column.endswith("_ts_ns") or column.endswith("_count") or column in {
        "live_source_count",
        "fill_count",
        "delta_sign",
    }:
        return "int64"
    if column.startswith("is_") or column.endswith("_eligible") or column.endswith("_available") or column in {
        "input_policy_compliant",
        "anchor_feature_eligible",
        "anchor_sequence_complete_eligible",
        "token_registry_match",
        "market_registry_trainable",
        "post_only",
        "replace_post_only",
        "replace_requested",
        "replaced",
        "replace_post_only_rejected",
        "live_price_valid",
        "live_reference_trusted",
        "live_price_degraded",
        "live_bias_active",
        "live_bias_carried_forward",
        "one_sided_book",
        "crossed_book",
        "stale_book",
        "submitted",
        "post_only_rejected",
        "post_only_crossed_book",
        "filled",
        "full_fill",
        "partial_fill",
        "fill_uncertain",
        "maker_fill",
        "taker_fill",
        "cancelled",
        "cancel_effective",
        "adverse_selection_5s",
        "toxic_fill_5s",
    }:
        return "bool"
    if column in {
        "dataset_version",
        "source_anchor_dataset",
        "target_date",
        "build_id",
        "policy_path",
        "quality_registry_path",
        "anchor_sequence_id",
        "anchor_source",
        "phase8_microstructure_exclusion_reason",
        "phase_bucket",
        "simulator_id",
        "intent_id",
        "source_sequence_id",
        "scenario_name",
        "market_slug",
        "token_asset_id",
        "token_side_normalized",
        "resolved_side",
        "order_side",
        "price_level",
        "holding_policy",
        "replace_price_level",
        "live_bias_mode",
        "fee_schedule_id",
        "execution_exclusion_reason",
    }:
        return "string"
    return "float64"


def _schema_payload() -> dict[str, Any]:
    columns: dict[str, dict[str, Any]] = {}
    for column in DATASET_COLUMNS:
        leakage_only = column in OUTCOME_COLUMNS
        training_eligible = (
            not leakage_only
            and column not in NON_FEATURE_IDENTITY_COLUMNS
            and column != "execution_training_eligible"
        )
        source = "phase5_simulator"
        if column in METADATA_COLUMNS:
            source = "dataset_factory"
        elif column in ANCHOR_FEATURE_COLUMNS:
            source = "microstructure_sequence_dataset_v1_tabular"
        elif column.startswith("live_"):
            source = "live_reference_events_v1"
        elif column.startswith("state_") or column in {"one_sided_book", "crossed_book", "stale_book"}:
            source = "polymarket_l2_events_v1"
        elif column in {"price_to_beat", "resolved_side", "market_close_ts_ns"}:
            source = "market_registry_v1"
        columns[column] = {
            "dtype": _dtype_for_column(column),
            "nullable": column not in {
                "dataset_version",
                "source_anchor_dataset",
                "target_date",
                "build_id",
                "input_policy_compliant",
                "execution_training_eligible",
                "intent_id",
                "market_slug",
                "token_asset_id",
                "token_side_normalized",
                "order_ts_ns",
                "market_close_ts_ns",
            },
            "leakage_only": leakage_only,
            "training_eligible": training_eligible,
            "source": source,
            "description": f"{DATASET_VERSION} column `{column}`.",
        }
    return {
        "version": DATASET_VERSION,
        "description": (
            "Phase 6 execution intent dataset. One row is one deterministic "
            "maker-first quote/action opportunity generated by execution_simulator_v1. "
            "Outcome columns are leakage_only and must not be used as decision-time features."
        ),
        "forbidden_columns": sorted(PHASE8_TARGET_COLUMNS),
        "columns": columns,
    }


def write_execution_intent_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> Path:
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(yaml.safe_dump(_schema_payload(), sort_keys=False), encoding="utf-8")
    return schema_path


def assert_schema_columns(columns: list[str], schema_path: Path = DEFAULT_SCHEMA_PATH) -> None:
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    expected = list(payload["columns"].keys())
    if list(columns) != expected:
        missing = [column for column in expected if column not in columns]
        extra = [column for column in columns if column not in expected]
        raise ValueError(f"{DATASET_VERSION} schema mismatch missing={missing} extra={extra}")


def _pa_type_for_column(column: str) -> pa.DataType:
    dtype = _dtype_for_column(column)
    if dtype == "bool":
        return pa.bool_()
    if dtype == "string":
        return pa.string()
    if dtype == "int64":
        return pa.int64()
    return pa.float64()


def _pa_schema() -> pa.Schema:
    return pa.schema([pa.field(column, _pa_type_for_column(column)) for column in DATASET_COLUMNS])


def _frame_to_table(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    normalized = frame[DATASET_COLUMNS].copy()
    for column in DATASET_COLUMNS:
        dtype = _dtype_for_column(column)
        if dtype == "bool":
            normalized[column] = normalized[column].fillna(False).astype(bool)
        elif dtype == "string":
            normalized[column] = normalized[column].fillna("").astype(str)
        elif dtype == "int64":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0).astype("int64")
        else:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")
    table = pa.Table.from_pandas(normalized, schema=schema, preserve_index=False)
    return table.cast(schema, safe=False)


def _empty_execution_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([], schema=_pa_schema()), path, compression="zstd", use_dictionary=True)


def _combine_execution_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for audit in audits:
        findings.extend(audit.get("findings", []))
    critical_count = sum(1 for finding in findings if finding.get("severity") == "CRITICAL")
    warning_count = sum(1 for finding in findings if finding.get("severity") == "WARNING")
    return {
        "critical_count": critical_count,
        "warning_count": warning_count,
        "findings": findings[:100],
        "truncated_findings": max(len(findings) - 100, 0),
        "verdict": "PASS" if critical_count == 0 else "FAIL",
    }


def _combine_ledgers(ledgers: list[dict[str, Any]]) -> dict[str, Any]:
    difference = float(sum(float(ledger.get("difference", 0.0)) for ledger in ledgers))
    return {
        "balances": bool(all(bool(ledger.get("balances", False)) for ledger in ledgers)),
        "difference": round(difference, 6),
    }


def _read_parquet_columns(path: Path, wanted_columns: list[str]) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    forbidden = sorted(PHASE8_TARGET_COLUMNS & available & set(wanted_columns))
    if forbidden:
        raise ValueError(f"Refusing to read Phase 8 target columns into Phase 6 anchors: {forbidden}")
    columns = [column for column in wanted_columns if column in available]
    return pd.read_parquet(path, columns=columns)


def _registry_path(root: Path, relative_path: Path) -> Path:
    return relative_path if relative_path.is_absolute() else root / relative_path


def _load_market_registry(root: Path) -> pd.DataFrame:
    path = _registry_path(root, MARKET_REGISTRY_PATH)
    if not path.exists():
        raise FileNotFoundError(f"market_registry_v1 missing: {path}")
    columns = [
        "market_slug",
        "market_open_ts_ns",
        "market_close_ts_ns",
        "price_to_beat",
        "resolved_side",
        "is_trainable",
    ]
    return pd.read_parquet(path, columns=columns)


def _load_token_registry(root: Path) -> pd.DataFrame:
    path = _registry_path(root, TOKEN_REGISTRY_PATH)
    if not path.exists():
        raise FileNotFoundError(f"token_registry_v1 missing: {path}")
    columns = ["market_slug", "token_asset_id", "token_side_normalized"]
    return pd.read_parquet(path, columns=columns)


def load_execution_intent_anchors(
    *,
    dataset_root: Path,
    root: Path,
    target_date: date,
    max_anchors: int | None = None,
    max_markets: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = dataset_root / PHASE8_TABULAR_DATASET / f"{target_date.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Phase 8 tabular anchors missing: {path}")
    wanted_columns = [
        "sequence_id",
        "market_slug",
        "token_asset_id",
        "token_side_normalized",
        "anchor_ts_ns",
        "anchor_source",
        "sequence_feature_eligible",
        "sequence_complete_eligible",
        "microstructure_exclusion_reason",
        "event_count_1s",
        "event_count_3s",
        "event_count_5s",
        "event_count_15s",
        "quote_update_count_1s",
        "quote_update_count_3s",
        "quote_update_count_5s",
        "trade_count_1s",
        "trade_count_3s",
        "trade_count_5s",
        "signed_trade_size_1s",
        "signed_trade_size_3s",
        "signed_trade_size_5s",
        "absolute_trade_size_1s",
        "absolute_trade_size_3s",
        "absolute_trade_size_5s",
        "spread_mean_5s",
        "spread_min_5s",
        "spread_max_5s",
        "imbalance_mean_5s",
        "imbalance_last",
        "imbalance_slope_5s",
        "mid_return_1s",
        "mid_return_3s",
        "mid_return_5s",
        "quote_churn_rate_5s",
        "depth_change_rate_5s",
        "sequence_completeness_rate",
        "price_to_beat",
        "delta_to_strike",
        "abs_delta_to_strike",
        "delta_sign",
        "t_since_open_s",
        "t_to_close_s",
        "phase_bucket",
        "live_reference_trusted",
    ]
    frame = _read_parquet_columns(path, wanted_columns)
    loaded_rows = int(len(frame))
    if frame.empty:
        return frame, {"loaded_rows": loaded_rows, "eligible_rows": 0, "selected_rows": 0}

    frame = frame.rename(
        columns={
            "sequence_id": "anchor_sequence_id",
            "sequence_feature_eligible": "anchor_feature_eligible",
            "sequence_complete_eligible": "anchor_sequence_complete_eligible",
            "microstructure_exclusion_reason": "phase8_microstructure_exclusion_reason",
        }
    )
    frame["anchor_feature_eligible"] = frame.get("anchor_feature_eligible", False).fillna(False).astype(bool)
    frame["anchor_sequence_complete_eligible"] = (
        frame.get("anchor_sequence_complete_eligible", False).fillna(False).astype(bool)
    )

    market_registry = _load_market_registry(root)
    token_registry = _load_token_registry(root).rename(columns={"token_side_normalized": "registry_token_side"})
    frame = frame.merge(
        market_registry,
        on="market_slug",
        how="left",
        suffixes=("", "_registry"),
    )
    frame = frame.merge(token_registry, on=["market_slug", "token_asset_id"], how="left")
    frame["token_registry_match"] = (
        frame["registry_token_side"].notna()
        & frame["registry_token_side"].astype(str).str.upper().eq(frame["token_side_normalized"].astype(str).str.upper())
    )
    frame["market_registry_trainable"] = frame["is_trainable"].fillna(False).astype(bool)
    if "price_to_beat_registry" in frame.columns:
        frame["price_to_beat"] = frame["price_to_beat"].where(frame["price_to_beat"].notna(), frame["price_to_beat_registry"])
    frame["resolved_side"] = frame["resolved_side"].fillna("").astype(str).str.upper()
    frame["market_open_ts_ns"] = pd.to_numeric(frame["market_open_ts_ns"], errors="coerce").fillna(0).astype("int64")
    frame["market_close_ts_ns"] = pd.to_numeric(frame["market_close_ts_ns"], errors="coerce").fillna(0).astype("int64")

    eligible_mask = (
        frame["anchor_feature_eligible"]
        & frame["token_registry_match"]
        & frame["market_registry_trainable"]
        & frame["live_reference_trusted"].fillna(False).astype(bool)
        & (frame["market_close_ts_ns"] > frame["anchor_ts_ns"])
    )
    eligible_rows = int(eligible_mask.sum())
    frame = frame.loc[eligible_mask].copy()
    frame = frame.sort_values(["market_slug", "token_asset_id", "anchor_ts_ns"], kind="stable").reset_index(drop=True)

    if max_markets is not None and max_markets > 0:
        selected_markets = list(iter_market_slugs(frame))[: int(max_markets)]
        frame = frame.loc[frame["market_slug"].astype(str).isin(selected_markets)].copy()
    selected_before_anchor_cap = int(len(frame))
    if max_anchors is not None and max_anchors > 0 and len(frame) > int(max_anchors):
        market_count = max(int(frame["market_slug"].nunique()), 1)
        per_market_cap = max(1, math.ceil(int(max_anchors) / market_count))
        frame = (
            frame.groupby("market_slug", group_keys=False, sort=False)
            .head(per_market_cap)
            .sort_values(["anchor_ts_ns", "market_slug", "token_asset_id"], kind="stable")
            .head(int(max_anchors))
            .sort_values(["market_slug", "token_asset_id", "anchor_ts_ns"], kind="stable")
            .reset_index(drop=True)
        )
    return frame.reset_index(drop=True), {
        "loaded_rows": loaded_rows,
        "eligible_rows": eligible_rows,
        "selected_before_anchor_cap": selected_before_anchor_cap,
        "selected_rows": int(len(frame)),
        "selected_markets": int(frame["market_slug"].nunique()) if not frame.empty else 0,
    }


def _grid(config: ExecutionIntentBuildConfig) -> ScenarioGrid:
    return ScenarioGrid(
        order_sizes=(float(config.order_size),),
        cancel_after_s=(0.5, 1.0, 3.0, None),
        include_replace_path=True,
        replace_after_s=1.0,
        include_join_best=True,
        include_improve_one_tick=True,
        include_inside_passive=True,
        scenario_name=config.scenario_name,
    )


def _intent_anchor_frame(anchors: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "anchor_sequence_id",
        "market_slug",
        "token_asset_id",
        "token_side_normalized",
        "anchor_ts_ns",
        "market_open_ts_ns",
        "t_to_close_s",
        "market_close_ts_ns",
        "resolved_side",
    ]
    frame = anchors[columns].copy()
    frame["sequence_id"] = frame["anchor_sequence_id"]
    return frame


def materialize_execution_intent_frame(
    *,
    results: pd.DataFrame,
    anchors: pd.DataFrame,
    target_date: date | str,
    build_id: str,
    build_ts_utc: str,
    policy_path: Path,
    quality_registry_path: Path,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS)
    anchor_columns = [
        "anchor_sequence_id",
        "anchor_source",
        "anchor_feature_eligible",
        "anchor_sequence_complete_eligible",
        "phase8_microstructure_exclusion_reason",
        "token_registry_match",
        "market_registry_trainable",
        "price_to_beat",
        "delta_to_strike",
        "abs_delta_to_strike",
        "delta_sign",
        "t_since_open_s",
        "t_to_close_s",
        "phase_bucket",
        "event_count_1s",
        "event_count_3s",
        "event_count_5s",
        "event_count_15s",
        "quote_update_count_1s",
        "quote_update_count_3s",
        "quote_update_count_5s",
        "trade_count_1s",
        "trade_count_3s",
        "trade_count_5s",
        "signed_trade_size_1s",
        "signed_trade_size_3s",
        "signed_trade_size_5s",
        "absolute_trade_size_1s",
        "absolute_trade_size_3s",
        "absolute_trade_size_5s",
        "spread_mean_5s",
        "spread_min_5s",
        "spread_max_5s",
        "imbalance_mean_5s",
        "imbalance_last",
        "imbalance_slope_5s",
        "mid_return_1s",
        "mid_return_3s",
        "mid_return_5s",
        "quote_churn_rate_5s",
        "depth_change_rate_5s",
        "sequence_completeness_rate",
    ]
    anchor_features = anchors[[column for column in anchor_columns if column in anchors.columns]].drop_duplicates(
        "anchor_sequence_id"
    )
    frame = results.merge(
        anchor_features,
        left_on="source_sequence_id",
        right_on="anchor_sequence_id",
        how="left",
    )
    frame["dataset_version"] = DATASET_VERSION
    frame["source_anchor_dataset"] = PHASE8_TABULAR_DATASET
    frame["target_date"] = str(target_date)
    frame["build_id"] = build_id
    frame["policy_path"] = str(policy_path)
    frame["quality_registry_path"] = str(quality_registry_path)
    frame["input_policy_compliant"] = (
        frame["anchor_feature_eligible"].fillna(False).astype(bool)
        & frame["token_registry_match"].fillna(False).astype(bool)
        & frame["market_registry_trainable"].fillna(False).astype(bool)
    )
    finite_live = pd.to_numeric(frame["live_btc_usd"], errors="coerce")
    finite_strike = pd.to_numeric(frame["price_to_beat"], errors="coerce")
    recomputed_delta = finite_live - finite_strike
    frame["delta_to_strike"] = recomputed_delta.where(recomputed_delta.notna(), frame["delta_to_strike"])
    frame["abs_delta_to_strike"] = frame["delta_to_strike"].abs()
    frame["t_to_close_s"] = (frame["market_close_ts_ns"].astype("int64") - frame["order_ts_ns"].astype("int64")) / SECOND_NS
    frame["execution_training_eligible"] = (
        frame["input_policy_compliant"].astype(bool)
        & frame["submitted"].astype(bool)
        & frame["live_price_valid"].astype(bool)
        & frame["live_reference_trusted"].astype(bool)
        & ~frame["live_price_degraded"].astype(bool)
        & (frame["live_reference_recv_ts_ns"].astype("int64") <= frame["order_ts_ns"].astype("int64"))
        & (frame["state_recv_ts_ns"].astype("int64") < frame["order_ts_ns"].astype("int64"))
        & ~frame["one_sided_book"].astype(bool)
        & ~frame["crossed_book"].astype(bool)
        & ~frame["stale_book"].astype(bool)
        & frame["execution_exclusion_reason"].fillna("").astype(str).eq("")
    )
    for column in DATASET_COLUMNS:
        if column not in frame.columns:
            dtype = _dtype_for_column(column)
            if dtype == "bool":
                frame[column] = False
            elif dtype == "string":
                frame[column] = ""
            elif dtype == "int64":
                frame[column] = 0
            else:
                frame[column] = math.nan
    return frame[DATASET_COLUMNS].copy()


def _safe_partition_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned[:120] or "partition"


def _market_batches(markets: list[str], batch_size: int) -> list[list[str]]:
    return [markets[start : start + int(batch_size)] for start in range(0, len(markets), int(batch_size))]


def _merge_counter_dicts(values: list[dict[str, int]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        counter.update({str(key): int(count) for key, count in value.items()})
    return dict(counter)


def _merge_l2_exclusions(values: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    merged: dict[str, Counter[str]] = {}
    for item in values:
        for source, counts in item.items():
            source_counter = merged.setdefault(str(source), Counter())
            if isinstance(counts, dict):
                source_counter.update({str(key): int(value) for key, value in counts.items()})
    return {source: dict(counter) for source, counter in sorted(merged.items())}


def _merge_partitions_to_daily_shard(*, part_paths: list[Path], output_path: Path, schema: pa.Schema) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(f".{output_path.name}.tmp")
    if tmp_output.exists():
        tmp_output.unlink()
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(tmp_output, schema, compression="zstd", use_dictionary=True)
        for part_path in part_paths:
            if not part_path.exists() or part_path.stat().st_size == 0:
                continue
            table = pq.read_table(part_path)
            if table.num_rows == 0:
                continue
            writer.write_table(table.cast(schema, safe=False))
    finally:
        if writer is not None:
            writer.close()
    tmp_output.replace(output_path)


def _existing_execution_day_summary(
    *,
    root: Path,
    dataset_root: Path,
    target_date: date,
    output_dir: Path,
    build_id: str,
    config: ExecutionIntentBuildConfig,
) -> dict[str, Any] | None:
    output_path = output_dir / DATASET_NAME / f"{target_date.isoformat()}.parquet"
    if not output_path.exists():
        return None
    try:
        build_ids = pd.read_parquet(output_path, columns=["build_id"])["build_id"].dropna().astype(str).unique().tolist()
    except Exception:
        return None
    try:
        frame = pd.read_parquet(
            output_path,
            columns=["execution_training_eligible", "submitted", "filled", "market_slug", "intent_id"],
        )
    except Exception:
        return None
    _, anchor_summary = load_execution_intent_anchors(
        dataset_root=dataset_root,
        root=root,
        target_date=target_date,
        max_anchors=config.max_anchors_per_day,
        max_markets=config.max_markets_per_day,
    )
    intents_per_market = frame["market_slug"].fillna("").astype(str).value_counts(sort=False).sort_index().to_dict()
    return {
        "date": target_date.isoformat(),
        **anchor_summary,
        "build_mode": config.build_mode,
        "skipped_existing": True,
        "existing_build_ids": build_ids,
        "build_id_matches": build_ids == [build_id],
        "partition_by_market": bool(config.partition_by_market),
        "market_batch_size": int(config.market_batch_size),
        "partition_count": 0,
        "effective_partition_workers": 0,
        "intent_rows": int(len(frame)),
        "generated_intents": 0,
        "simulated_intents": 0,
        "submitted_rows": int(frame["submitted"].sum()),
        "filled_rows": int(frame["filled"].sum()),
        "training_eligible_rows": int(frame["execution_training_eligible"].sum()),
        "output_path": str(output_path),
        "sha256": _sha256(output_path),
        "execution_audit": {
            "critical_count": 0,
            "warning_count": 0,
            "findings": [],
            "verdict": "SKIPPED_EXISTING_REQUIRES_FINAL_AUDIT",
        },
        "ledger": {"balances": True, "difference": 0.0, "skipped_existing": True},
        "l2_exclusions": {},
        "replay_events": 0,
        "replay_markets": 0,
        "replay_tokens": 0,
        "execution_exclusion_reason_counts": {},
        "stage_runtime_seconds": {"resume_existing_seconds": 0.0},
        "wall_clock_seconds": 0.0,
        "peak_memory_bytes": 0,
        "current_memory_bytes": 0,
        "output_size_bytes": int(output_path.stat().st_size),
        "anchor_batch_size": int(config.anchor_batch_size),
        "chunk_count": 0,
        "intents_per_market": {str(key): int(value) for key, value in intents_per_market.items()},
    }


def _build_execution_intent_market_partition(
    *,
    root: Path,
    dataset_root: Path,
    target_date: date,
    anchors: pd.DataFrame,
    market_slugs: list[str],
    part_index: int,
    part_path: Path,
    policy_path: Path,
    quality_registry_path: Path,
    schema_path: Path,
    build_id: str,
    build_ts_utc: str,
    config: ExecutionIntentBuildConfig,
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    tracing_started = False
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        tracing_started = True
    tracemalloc.reset_peak()
    stage_times: dict[str, float] = {}
    part_path.parent.mkdir(parents=True, exist_ok=True)
    if part_path.exists():
        part_path.unlink()

    stage_start = time.perf_counter()
    selected_markets = set(str(value) for value in market_slugs)
    replay_load = load_execution_replay_for_date(
        root=root,
        target_date=target_date,
        policy_path=policy_path,
        quality_registry_path=quality_registry_path,
        prefer_canonical=True,
        materialize_canonical=False,
        market_slugs=selected_markets,
    )
    stage_times["load_replay_seconds"] = round(time.perf_counter() - stage_start, 6)
    live_reader = LiveReferenceEventsReader(root=root)
    simulator = ExecutionSimulator(
        replay_load.replay_index,
        ExecutionSimulatorConfig(
            fee_schedule=config.fee_schedule,
            cancel_latency_ns=config.cancel_latency_ns,
            replace_latency_ns=config.replace_latency_ns,
            stale_book_threshold_ns=config.stale_book_threshold_ns,
            queue_ahead_fraction=config.queue_ahead_fraction,
            require_live_reference=config.require_live_reference,
        ),
        live_reference_reader=live_reader,
    )

    schema = _pa_schema()
    writer: pq.ParquetWriter | None = None
    intent_rows = 0
    submitted_rows = 0
    filled_rows = 0
    training_eligible_rows = 0
    generated_intents = 0
    simulated_intents = 0
    reason_counts: Counter[str] = Counter()
    intents_per_market: Counter[str] = Counter()
    execution_audits: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    chunk_count = 0
    grid = _grid(config)
    stage_accumulators = Counter()
    try:
        for start in range(0, len(anchors), int(config.anchor_batch_size)):
            if config.max_intents_per_day is not None and generated_intents >= int(config.max_intents_per_day):
                break
            chunk_count += 1
            chunk_anchors = anchors.iloc[start : start + int(config.anchor_batch_size)].copy()
            intent_source = _intent_anchor_frame(chunk_anchors)

            stage_start = time.perf_counter()
            remaining_intents = (
                None
                if config.max_intents_per_day is None
                else max(int(config.max_intents_per_day) - generated_intents, 0)
            )
            intents = generate_intents_from_anchors(
                anchors=intent_source,
                replay_index=replay_load.replay_index,
                grid=grid,
                max_intents=remaining_intents,
            )
            stage_accumulators["generate_intents_seconds"] += time.perf_counter() - stage_start
            generated_intents += len(intents)
            if not intents:
                continue

            stage_start = time.perf_counter()
            results = simulator.simulate_many(intents)
            simulated_intents += int(len(results))
            stage_accumulators["simulate_seconds"] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            dataset = materialize_execution_intent_frame(
                results=results,
                anchors=chunk_anchors,
                target_date=target_date.isoformat(),
                build_id=build_id,
                build_ts_utc=build_ts_utc,
                policy_path=policy_path,
                quality_registry_path=quality_registry_path,
            )
            stage_accumulators["materialize_seconds"] += time.perf_counter() - stage_start
            if dataset.empty:
                continue

            stage_start = time.perf_counter()
            table = _frame_to_table(dataset, schema)
            if writer is None:
                writer = pq.ParquetWriter(part_path, schema, compression="zstd", use_dictionary=True)
            writer.write_table(table)
            stage_accumulators["write_seconds"] += time.perf_counter() - stage_start

            execution_frame = dataset[RESULT_COLUMNS].copy()
            stage_start = time.perf_counter()
            execution_audits.append(audit_execution_results(execution_frame))
            ledgers.append(reconcile_execution_results(execution_frame))
            stage_accumulators["audit_seconds"] += time.perf_counter() - stage_start

            intent_rows += int(len(dataset))
            submitted_rows += int(dataset["submitted"].sum())
            filled_rows += int(dataset["filled"].sum())
            training_eligible_rows += int(dataset["execution_training_eligible"].sum())
            reason_counts.update(dataset["execution_exclusion_reason"].fillna("").astype(str).tolist())
            intents_per_market.update(dataset["market_slug"].fillna("").astype(str).tolist())
    finally:
        if writer is not None:
            writer.close()
        if not part_path.exists():
            _empty_execution_dataset(part_path)

    stage_times.update({key: round(float(value), 6) for key, value in sorted(stage_accumulators.items())})
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    if tracing_started:
        tracemalloc.stop()
    return {
        "part_index": int(part_index),
        "market_slugs": [str(value) for value in market_slugs],
        "part_path": str(part_path),
        "intent_rows": int(intent_rows),
        "generated_intents": int(generated_intents),
        "simulated_intents": int(simulated_intents),
        "submitted_rows": int(submitted_rows),
        "filled_rows": int(filled_rows),
        "training_eligible_rows": int(training_eligible_rows),
        "execution_audit": _combine_execution_audits(execution_audits),
        "ledger": _combine_ledgers(ledgers) if ledgers else {"balances": True, "difference": 0.0},
        "l2_exclusions": {key: dict(value) for key, value in replay_load.exclusion_summary.items()},
        "replay_events": int(len(replay_load.events)),
        "replay_markets": int(replay_load.replay_index.market_count),
        "replay_tokens": int(replay_load.replay_index.token_count),
        "execution_exclusion_reason_counts": dict(reason_counts),
        "stage_runtime_seconds": stage_times,
        "wall_clock_seconds": round(time.perf_counter() - wall_start, 6),
        "peak_memory_bytes": int(peak_memory),
        "current_memory_bytes": int(current_memory),
        "output_size_bytes": int(part_path.stat().st_size),
        "anchor_batch_size": int(config.anchor_batch_size),
        "chunk_count": int(chunk_count),
        "intents_per_market": dict(sorted(intents_per_market.items())),
    }


def _build_execution_intent_day_partitioned(
    *,
    root: Path,
    dataset_root: Path,
    target_date: date,
    output_dir: Path,
    policy_path: Path,
    quality_registry_path: Path,
    schema_path: Path,
    build_id: str,
    build_ts_utc: str,
    config: ExecutionIntentBuildConfig,
    anchors: pd.DataFrame,
    anchor_summary: dict[str, Any],
    wall_start: float,
    initial_stage_times: dict[str, float],
    tracing_started: bool,
) -> dict[str, Any]:
    output_path = output_dir / DATASET_NAME / f"{target_date.isoformat()}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markets = list(iter_market_slugs(anchors))
    batches = _market_batches(markets, int(config.market_batch_size))
    worker_count = _worker_count(config.max_workers, len(batches))
    part_dir = output_dir / DATASET_NAME / "_parts" / build_id / target_date.isoformat()
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    def part_path_for(index: int, batch: list[str]) -> Path:
        first = _safe_partition_name(batch[0]) if batch else "empty"
        return part_dir / f"part-{index:04d}-{first}.parquet"

    partition_results: list[dict[str, Any]] = []
    if worker_count == 1:
        for index, batch in enumerate(batches):
            batch_anchors = anchors.loc[anchors["market_slug"].astype(str).isin(set(batch))].copy()
            partition_results.append(
                _build_execution_intent_market_partition(
                    root=root,
                    dataset_root=dataset_root,
                    target_date=target_date,
                    anchors=batch_anchors,
                    market_slugs=batch,
                    part_index=index,
                    part_path=part_path_for(index, batch),
                    policy_path=policy_path,
                    quality_registry_path=quality_registry_path,
                    schema_path=schema_path,
                    build_id=build_id,
                    build_ts_utc=build_ts_utc,
                    config=config,
                )
            )
    else:
        results_by_index: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            for index, batch in enumerate(batches):
                batch_anchors = anchors.loc[anchors["market_slug"].astype(str).isin(set(batch))].copy()
                futures[
                    executor.submit(
                        _build_execution_intent_market_partition,
                        root=root,
                        dataset_root=dataset_root,
                        target_date=target_date,
                        anchors=batch_anchors,
                        market_slugs=batch,
                        part_index=index,
                        part_path=part_path_for(index, batch),
                        policy_path=policy_path,
                        quality_registry_path=quality_registry_path,
                        schema_path=schema_path,
                        build_id=build_id,
                        build_ts_utc=build_ts_utc,
                        config=config,
                    )
                ] = index
            for future in as_completed(futures):
                result = future.result()
                results_by_index[int(result["part_index"])] = result
        partition_results = [results_by_index[index] for index in range(len(batches))]

    part_paths = [Path(result["part_path"]) for result in partition_results]
    stage_times = dict(initial_stage_times)
    merge_start = time.perf_counter()
    _merge_partitions_to_daily_shard(part_paths=part_paths, output_path=output_path, schema=_pa_schema())
    stage_times["merge_partitions_seconds"] = round(time.perf_counter() - merge_start, 6)
    for key in sorted({key for result in partition_results for key in result["stage_runtime_seconds"].keys()}):
        stage_times[key] = round(sum(float(result["stage_runtime_seconds"].get(key, 0.0)) for result in partition_results), 6)

    execution_audit = _combine_execution_audits([result["execution_audit"] for result in partition_results])
    ledger = _combine_ledgers([result["ledger"] for result in partition_results])
    reason_counts = _merge_counter_dicts([result["execution_exclusion_reason_counts"] for result in partition_results])
    intents_per_market = _merge_counter_dicts([result["intents_per_market"] for result in partition_results])
    l2_exclusions = _merge_l2_exclusions([result["l2_exclusions"] for result in partition_results])
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    if tracing_started:
        tracemalloc.stop()
    shutil.rmtree(part_dir, ignore_errors=True)
    return {
        "date": target_date.isoformat(),
        **anchor_summary,
        "build_mode": config.build_mode,
        "partition_by_market": True,
        "market_batch_size": int(config.market_batch_size),
        "partition_count": int(len(partition_results)),
        "effective_partition_workers": int(worker_count),
        "intent_rows": int(sum(int(result["intent_rows"]) for result in partition_results)),
        "generated_intents": int(sum(int(result["generated_intents"]) for result in partition_results)),
        "simulated_intents": int(sum(int(result["simulated_intents"]) for result in partition_results)),
        "submitted_rows": int(sum(int(result["submitted_rows"]) for result in partition_results)),
        "filled_rows": int(sum(int(result["filled_rows"]) for result in partition_results)),
        "training_eligible_rows": int(sum(int(result["training_eligible_rows"]) for result in partition_results)),
        "output_path": str(output_path),
        "sha256": _sha256(output_path),
        "execution_audit": execution_audit,
        "ledger": ledger,
        "l2_exclusions": l2_exclusions,
        "replay_events": int(sum(int(result["replay_events"]) for result in partition_results)),
        "replay_markets": int(sum(int(result["replay_markets"]) for result in partition_results)),
        "replay_tokens": int(sum(int(result["replay_tokens"]) for result in partition_results)),
        "execution_exclusion_reason_counts": reason_counts,
        "stage_runtime_seconds": stage_times,
        "wall_clock_seconds": round(time.perf_counter() - wall_start, 6),
        "peak_memory_bytes": int(max([peak_memory] + [int(result["peak_memory_bytes"]) for result in partition_results])),
        "current_memory_bytes": int(current_memory),
        "output_size_bytes": int(output_path.stat().st_size),
        "anchor_batch_size": int(config.anchor_batch_size),
        "chunk_count": int(sum(int(result["chunk_count"]) for result in partition_results)),
        "intents_per_market": dict(sorted(intents_per_market.items())),
    }


def build_execution_intent_day(
    *,
    root: Path,
    dataset_root: Path,
    target_date: date,
    output_dir: Path,
    policy_path: Path,
    quality_registry_path: Path,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    build_id: str,
    build_ts_utc: str,
    config: ExecutionIntentBuildConfig = ExecutionIntentBuildConfig(),
    live_reference_reader: LiveReferenceEventsReader | None = None,
    write_schema: bool = True,
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    tracing_started = False
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        tracing_started = True
    tracemalloc.reset_peak()
    stage_times: dict[str, float] = {}

    stage_start = time.perf_counter()
    if write_schema:
        write_execution_intent_schema(schema_path)
    assert_schema_columns(DATASET_COLUMNS, schema_path)
    anchors, anchor_summary = load_execution_intent_anchors(
        dataset_root=dataset_root,
        root=root,
        target_date=target_date,
        max_anchors=config.max_anchors_per_day,
        max_markets=config.max_markets_per_day,
    )
    stage_times["load_anchors_seconds"] = round(time.perf_counter() - stage_start, 6)
    output_path = output_dir / DATASET_NAME / f"{target_date.isoformat()}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if anchors.empty:
        _empty_execution_dataset(output_path)
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        if tracing_started:
            tracemalloc.stop()
        return {
            "date": target_date.isoformat(),
            **anchor_summary,
            "intent_rows": 0,
            "build_mode": config.build_mode,
            "output_path": str(output_path),
            "sha256": _sha256(output_path),
            "execution_audit": audit_execution_results(pd.DataFrame(columns=RESULT_COLUMNS)),
            "ledger": {"balances": True, "difference": 0.0},
            "l2_exclusions": {},
            "replay_events": 0,
            "replay_markets": 0,
            "replay_tokens": 0,
            "stage_runtime_seconds": stage_times,
            "wall_clock_seconds": round(time.perf_counter() - wall_start, 6),
            "peak_memory_bytes": int(peak_memory),
            "current_memory_bytes": int(current_memory),
            "output_size_bytes": int(output_path.stat().st_size),
            "anchor_batch_size": config.anchor_batch_size,
            "intents_per_market": {},
        }

    if config.partition_by_market:
        return _build_execution_intent_day_partitioned(
            root=root,
            dataset_root=dataset_root,
            target_date=target_date,
            output_dir=output_dir,
            policy_path=policy_path,
            quality_registry_path=quality_registry_path,
            schema_path=schema_path,
            build_id=build_id,
            build_ts_utc=build_ts_utc,
            config=config,
            anchors=anchors,
            anchor_summary=anchor_summary,
            wall_start=wall_start,
            initial_stage_times=stage_times,
            tracing_started=tracing_started,
        )

    selected_markets = set(iter_market_slugs(anchors))
    stage_start = time.perf_counter()
    replay_load = load_execution_replay_for_date(
        root=root,
        target_date=target_date,
        policy_path=policy_path,
        quality_registry_path=quality_registry_path,
        prefer_canonical=True,
        materialize_canonical=False,
        market_slugs=selected_markets,
    )
    stage_times["load_replay_seconds"] = round(time.perf_counter() - stage_start, 6)
    live_reader = live_reference_reader or LiveReferenceEventsReader(root=root)
    simulator = ExecutionSimulator(
        replay_load.replay_index,
        ExecutionSimulatorConfig(
            fee_schedule=config.fee_schedule,
            cancel_latency_ns=config.cancel_latency_ns,
            replace_latency_ns=config.replace_latency_ns,
            stale_book_threshold_ns=config.stale_book_threshold_ns,
            queue_ahead_fraction=config.queue_ahead_fraction,
            require_live_reference=config.require_live_reference,
        ),
        live_reference_reader=live_reader,
    )

    if output_path.exists():
        output_path.unlink()
    schema = _pa_schema()
    writer: pq.ParquetWriter | None = None
    intent_rows = 0
    submitted_rows = 0
    filled_rows = 0
    training_eligible_rows = 0
    generated_intents = 0
    simulated_intents = 0
    reason_counts: Counter[str] = Counter()
    intents_per_market: Counter[str] = Counter()
    execution_audits: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    chunk_count = 0
    grid = _grid(config)
    stage_accumulators = Counter()
    try:
        for start in range(0, len(anchors), int(config.anchor_batch_size)):
            if config.max_intents_per_day is not None and generated_intents >= int(config.max_intents_per_day):
                break
            chunk_count += 1
            chunk_anchors = anchors.iloc[start : start + int(config.anchor_batch_size)].copy()
            intent_source = _intent_anchor_frame(chunk_anchors)

            stage_start = time.perf_counter()
            remaining_intents = (
                None
                if config.max_intents_per_day is None
                else max(int(config.max_intents_per_day) - generated_intents, 0)
            )
            intents = generate_intents_from_anchors(
                anchors=intent_source,
                replay_index=replay_load.replay_index,
                grid=grid,
                max_intents=remaining_intents,
            )
            stage_accumulators["generate_intents_seconds"] += time.perf_counter() - stage_start
            generated_intents += len(intents)
            if not intents:
                continue

            stage_start = time.perf_counter()
            results = simulator.simulate_many(intents)
            simulated_intents += int(len(results))
            stage_accumulators["simulate_seconds"] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            dataset = materialize_execution_intent_frame(
                results=results,
                anchors=chunk_anchors,
                target_date=target_date.isoformat(),
                build_id=build_id,
                build_ts_utc=build_ts_utc,
                policy_path=policy_path,
                quality_registry_path=quality_registry_path,
            )
            stage_accumulators["materialize_seconds"] += time.perf_counter() - stage_start
            if dataset.empty:
                continue

            stage_start = time.perf_counter()
            table = _frame_to_table(dataset, schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema, compression="zstd", use_dictionary=True)
            writer.write_table(table)
            stage_accumulators["write_seconds"] += time.perf_counter() - stage_start

            execution_frame = dataset[RESULT_COLUMNS].copy()
            stage_start = time.perf_counter()
            execution_audits.append(audit_execution_results(execution_frame))
            ledgers.append(reconcile_execution_results(execution_frame))
            stage_accumulators["audit_seconds"] += time.perf_counter() - stage_start

            intent_rows += int(len(dataset))
            submitted_rows += int(dataset["submitted"].sum())
            filled_rows += int(dataset["filled"].sum())
            training_eligible_rows += int(dataset["execution_training_eligible"].sum())
            reason_counts.update(dataset["execution_exclusion_reason"].fillna("").astype(str).tolist())
            intents_per_market.update(dataset["market_slug"].fillna("").astype(str).tolist())
    finally:
        if writer is not None:
            writer.close()
        if not output_path.exists():
            _empty_execution_dataset(output_path)

    stage_times.update({key: round(float(value), 6) for key, value in sorted(stage_accumulators.items())})
    execution_audit = _combine_execution_audits(execution_audits)
    ledger = _combine_ledgers(ledgers) if ledgers else {"balances": True, "difference": 0.0}
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    if tracing_started:
        tracemalloc.stop()
    return {
        "date": target_date.isoformat(),
        **anchor_summary,
        "build_mode": config.build_mode,
        "intent_rows": int(intent_rows),
        "generated_intents": int(generated_intents),
        "simulated_intents": int(simulated_intents),
        "submitted_rows": int(submitted_rows),
        "filled_rows": int(filled_rows),
        "training_eligible_rows": int(training_eligible_rows),
        "output_path": str(output_path),
        "sha256": _sha256(output_path),
        "execution_audit": execution_audit,
        "ledger": ledger,
        "l2_exclusions": {key: dict(value) for key, value in replay_load.exclusion_summary.items()},
        "replay_events": int(len(replay_load.events)),
        "replay_markets": int(replay_load.replay_index.market_count),
        "replay_tokens": int(replay_load.replay_index.token_count),
        "execution_exclusion_reason_counts": dict(reason_counts),
        "stage_runtime_seconds": stage_times,
        "wall_clock_seconds": round(time.perf_counter() - wall_start, 6),
        "peak_memory_bytes": int(peak_memory),
        "current_memory_bytes": int(current_memory),
        "output_size_bytes": int(output_path.stat().st_size),
        "anchor_batch_size": int(config.anchor_batch_size),
        "chunk_count": int(chunk_count),
        "intents_per_market": dict(sorted(intents_per_market.items())),
    }


def _write_build_report(report: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# execution_intent_dataset_v1 Build Report",
        "",
        f"- Dataset version: `{DATASET_VERSION}`",
        f"- Date range: `{report['date_from']}` to `{report['date_to']}`",
        "- Scope: Phase 6 dataset factory only; no model training, policy optimization, paper trading, or live deployment.",
        f"- Schema: `{report['schema_path']}`",
        f"- Build id: `{report['build_id']}`",
        f"- Build timestamp UTC: `{report['build_ts_utc']}`",
        f"- Policy path: `{report['policy_path']}`",
        f"- Quality registry path: `{report['quality_registry_path']}`",
        "",
        "## Build Mode",
        "",
        f"- build_mode: `{report['config']['build_mode']}`",
        f"- production_uncapped: `{report['config']['production_uncapped']}`",
        f"- max_anchors_per_day: `{report['config']['max_anchors_per_day']}`",
        f"- max_markets_per_day: `{report['config']['max_markets_per_day']}`",
        f"- max_intents_per_day: `{report['config']['max_intents_per_day']}`",
        f"- anchor_batch_size: `{report['config']['anchor_batch_size']}`",
        f"- requested_max_workers: `{report['config']['requested_max_workers']}`",
        f"- effective_day_workers: `{report['config']['effective_max_workers']}`",
        f"- partition_by_market: `{report['config']['partition_by_market']}`",
        f"- market_batch_size: `{report['config']['market_batch_size']}`",
        f"- resume_existing: `{report['config']['resume_existing']}`",
        f"- order_size: `{report['config']['order_size']}`",
        f"- fee_schedule_id: `{report['config']['fee_schedule_id']}`",
        "",
        "## Daily Shards",
        "",
        "| date | status | anchors loaded | anchors eligible | anchors selected | markets | partitions | workers | replay events | rows | submitted | filled | training eligible | runtime s | peak MiB | audit critical | ledger balances | sha256 | path |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for day in report["daily"]:
        lines.append(
            f"| {day['date']} | {'SKIPPED_EXISTING' if day.get('skipped_existing') else 'BUILT'} | "
            f"{day.get('loaded_rows', 0)} | {day.get('eligible_rows', 0)} | "
            f"{day.get('selected_rows', 0)} | {day.get('selected_markets', 0)} | "
            f"{day.get('partition_count', 1)} | {day.get('effective_partition_workers', 1)} | "
            f"{day.get('replay_events', 0)} | "
            f"{day.get('intent_rows', 0)} | {day.get('submitted_rows', 0)} | {day.get('filled_rows', 0)} | "
            f"{day.get('training_eligible_rows', 0)} | {float(day.get('wall_clock_seconds', 0.0)):.2f} | "
            f"{int(day.get('peak_memory_bytes', 0)) / (1024 * 1024):.1f} | {day['execution_audit']['critical_count']} | "
            f"{bool(day['ledger'].get('balances', False))} | `{day['sha256']}` | `{day['output_path']}` |"
        )
    total_rows = sum(int(day.get("intent_rows", 0)) for day in report["daily"])
    total_training = sum(int(day.get("training_eligible_rows", 0)) for day in report["daily"])
    total_critical = sum(int(day["execution_audit"]["critical_count"]) for day in report["daily"])
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Rows: `{total_rows}`",
            f"- Training-eligible rows: `{total_training}`",
            f"- Execution-audit critical findings across shards: `{total_critical}`",
            "",
            "## Contract Notes",
            "",
            "- Inputs are Phase 8 tabular anchors filtered to feature-eligible, token-registry-matched, trainable markets.",
            "- Replay is loaded through `load_execution_replay_for_date`, which prefers canonical L2 shards and otherwise applies dataset policy plus `quality_registry_v1` finalized-clean filtering.",
            "- `live_reference_events_v1` is the only dense BTC price reference used by the simulator.",
            "- Phase 8 target columns are not copied into this dataset.",
            "- Simulator outcomes, markouts, fees, rebates, fills, and PnL are present only as leakage-only schema fields.",
            "- Production mode is uncapped. Capped debug/audit builds must set `build_mode=capped` explicitly.",
            "- `SKIPPED_EXISTING` rows reuse complete existing day shards; final dataset audit remains mandatory.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_execution_intent_datasets(
    *,
    root: Path,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    date_from: date,
    date_to: date,
    output_dir: Path = DEFAULT_DATASET_ROOT,
    policy_path: Path = Path("config/dataset_policy_v1.yaml"),
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    report_path: Path = DEFAULT_BUILD_REPORT_PATH,
    config: ExecutionIntentBuildConfig = ExecutionIntentBuildConfig(),
) -> dict[str, Any]:
    schema_written = write_execution_intent_schema(schema_path)
    assert_schema_columns(DATASET_COLUMNS, schema_written)
    build_ts_utc = _utc_now_iso()
    build_id = build_id_for_config(
        date_from=date_from,
        date_to=date_to,
        config=config,
        policy_path=policy_path,
        quality_registry_path=quality_registry_path,
    )
    live_reference_reader = LiveReferenceEventsReader(root=root)
    target_dates = _date_range(date_from, date_to)
    effective_workers = 1 if config.partition_by_market else _worker_count(config.max_workers, len(target_dates))
    if effective_workers == 1:
        daily = []
        for target_day in target_dates:
            existing = (
                _existing_execution_day_summary(
                    root=root,
                    dataset_root=dataset_root,
                    target_date=target_day,
                    output_dir=output_dir,
                    build_id=build_id,
                    config=config,
                )
                if config.resume_existing
                else None
            )
            if existing is not None:
                daily.append(existing)
                continue
            daily.append(
                build_execution_intent_day(
                    root=root,
                    dataset_root=dataset_root,
                    target_date=target_day,
                    output_dir=output_dir,
                    policy_path=policy_path,
                    quality_registry_path=quality_registry_path,
                    schema_path=schema_written,
                    build_id=build_id,
                    build_ts_utc=build_ts_utc,
                    config=config,
                    live_reference_reader=live_reference_reader,
                    write_schema=False,
                )
            )
    else:
        daily_by_date: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    build_execution_intent_day,
                    root=root,
                    dataset_root=dataset_root,
                    target_date=target_day,
                    output_dir=output_dir,
                    policy_path=policy_path,
                    quality_registry_path=quality_registry_path,
                    schema_path=schema_written,
                    build_id=build_id,
                    build_ts_utc=build_ts_utc,
                    config=config,
                    live_reference_reader=None,
                    write_schema=False,
                ): target_day
                for target_day in target_dates
            }
            for future in as_completed(futures):
                result = future.result()
                daily_by_date[str(result["date"])] = result
        daily = [daily_by_date[target_day.isoformat()] for target_day in target_dates]
    report = {
        "dataset_version": DATASET_VERSION,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "schema_path": str(schema_written),
        "policy_path": str(policy_path),
        "quality_registry_path": str(quality_registry_path),
        "build_id": build_id,
        "build_ts_utc": build_ts_utc,
        "config": {
            "build_mode": config.build_mode,
            "production_uncapped": bool(config.build_mode == "production"),
            "max_anchors_per_day": config.max_anchors_per_day,
            "max_intents_per_day": config.max_intents_per_day,
            "max_markets_per_day": config.max_markets_per_day,
            "anchor_batch_size": config.anchor_batch_size,
            "requested_max_workers": config.max_workers,
            "effective_max_workers": effective_workers,
            "partition_by_market": config.partition_by_market,
            "market_batch_size": config.market_batch_size,
            "resume_existing": config.resume_existing,
            "order_size": config.order_size,
            "fee_schedule_id": config.fee_schedule.schedule_id,
            "maker_fee_bps": config.fee_schedule.maker_fee_bps,
            "maker_rebate_bps": config.fee_schedule.maker_rebate_bps,
        },
        "daily": daily,
    }
    _write_build_report(report, report_path)
    return report
