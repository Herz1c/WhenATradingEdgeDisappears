from __future__ import annotations

import hashlib
import io
import math
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .microstructure_sequence_builder import EVENT128_DATASET, EVENT64_DATASET, TARGET_COLUMNS

TENSOR_DATASET = "microstructure_sequence_dataset_v1_tensor"
DEFAULT_DATASET_ROOT = Path("data/datasets")
DEFAULT_SCHEMA_PATH = Path("config/schemas/microstructure_sequence_dataset_v1_tensor_schema.yaml")
DEFAULT_BUILD_REPORT_PATH = Path("docs/microstructure_tensor_export_build_report.md")
DEFAULT_AUDIT_REPORT_PATH = Path("docs/microstructure_tensor_export_audit_report.md")
DEFAULT_REPRO_REPORT_PATH = Path("docs/microstructure_tensor_export_reproducibility_report.md")

VARIANT_SPECS = {
    "event64": {"source_dataset": EVENT64_DATASET, "sequence_length": 64},
    "event128": {"source_dataset": EVENT128_DATASET, "sequence_length": 128},
}

EVENT_NUMERIC_CHANNELS = [
    "events_dt_from_anchor_s",
    "events_dt_from_prev_s",
    "events_best_bid",
    "events_best_ask",
    "events_mid",
    "events_spread",
    "events_microprice",
    "events_bid_depth_total",
    "events_ask_depth_total",
    "events_imbalance",
    "events_last_trade_price",
    "events_trade_size",
    "events_tick_size",
]
EVENT_DERIVED_CHANNELS = ["events_event_type_code", "events_trade_side_code"]
EVENT_CHANNELS = EVENT_NUMERIC_CHANNELS + EVENT_DERIVED_CHANNELS

CONTEXT_COLUMNS = [
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
    "one_sided_book",
    "crossed_book",
    "stale_book",
    "sequence_completeness_rate",
    "price_to_beat",
    "live_btc_usd",
    "delta_to_strike",
    "abs_delta_to_strike",
    "delta_sign",
    "t_since_open_s",
    "t_to_close_s",
    "live_reference_trusted",
    "live_applied_bias_age_seconds",
]

HORIZONS = (1, 3, 5, 15, 30)
EXCURSION_HORIZONS = (5, 15)
EVENT_TYPE_CODES = {"pad": 0, "book_state": 1, "top_of_book": 2, "trade": 3, "tick_update": 4, "other": 5, "": 5}
TRADE_SIDE_CODES = {"none": 0, "buy": 1, "sell": 2, "unknown": 3, "": 0}


@dataclass(frozen=True, slots=True)
class TensorExportConfig:
    variants: tuple[str, ...] = ("event64", "event128")
    rows_per_shard: int = 25_000
    compression_level: int = 9
    max_workers: int = 0
    skip_existing: bool = True

    def __post_init__(self) -> None:
        unknown = sorted(set(self.variants) - set(VARIANT_SPECS))
        if unknown:
            raise ValueError(f"Unknown tensor variants: {unknown}")
        if self.rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        if not (0 <= self.compression_level <= 9):
            raise ValueError("compression_level must be between 0 and 9")
        if self.max_workers < 0:
            raise ValueError("max_workers must be >= 0")


def _date_range(date_from: date, date_to: date) -> list[date]:
    if date_to < date_from:
        raise ValueError("date_to must be >= date_from")
    return [date_from + timedelta(days=offset) for offset in range((date_to - date_from).days + 1)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        result = float(value)
    except Exception:
        return math.nan
    return result if math.isfinite(result) else math.nan


def _fixed_list(value: Any, sequence_length: int, *, default: Any) -> list[Any]:
    values = [] if value is None else list(value)
    if len(values) < sequence_length:
        values = [default] * (sequence_length - len(values)) + values
    elif len(values) > sequence_length:
        values = values[-sequence_length:]
    return values


def _npy_payload(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asanyarray(array), allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray], compression_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if compression_level > 0 else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, mode="w", compression=compression, compresslevel=compression_level) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.external_attr = 0o644 << 16
            archive.writestr(info, _npy_payload(arrays[name]))


def _string_array(values: pd.Series, width: int) -> np.ndarray:
    return values.fillna("").astype(str).to_numpy(dtype=f"<U{width}")


def frame_to_tensor_arrays(frame: pd.DataFrame, *, sequence_length: int, variant: str) -> dict[str, np.ndarray]:
    rows = int(len(frame))
    x_events = np.full((rows, sequence_length, len(EVENT_CHANNELS)), np.nan, dtype=np.float32)
    events_recv_ts_ns = np.zeros((rows, sequence_length), dtype=np.int64)
    event_mask = np.zeros((rows, sequence_length), dtype=np.bool_)

    event_type_values: list[list[str]] = []
    trade_side_values: list[list[str]] = []
    for row_idx, row in enumerate(frame.itertuples(index=False)):
        row_data = row._asdict()
        recv_values = _fixed_list(row_data.get("events_recv_ts_ns"), sequence_length, default=0)
        events_recv_ts_ns[row_idx, :] = np.asarray([int(value or 0) for value in recv_values], dtype=np.int64)

        event_types = [str(value or "") for value in _fixed_list(row_data.get("events_event_type"), sequence_length, default="pad")]
        trade_sides = [str(value or "") for value in _fixed_list(row_data.get("events_trade_side"), sequence_length, default="none")]
        event_type_values.append(event_types)
        trade_side_values.append(trade_sides)
        event_mask[row_idx, :] = np.asarray(
            [event_type != "pad" and int(recv_values[pos] or 0) > 0 for pos, event_type in enumerate(event_types)],
            dtype=np.bool_,
        )
        for channel_idx, column in enumerate(EVENT_NUMERIC_CHANNELS):
            values = _fixed_list(row_data.get(column), sequence_length, default=math.nan)
            x_events[row_idx, :, channel_idx] = np.asarray([_safe_float(value) for value in values], dtype=np.float32)
        event_type_idx = EVENT_CHANNELS.index("events_event_type_code")
        trade_side_idx = EVENT_CHANNELS.index("events_trade_side_code")
        x_events[row_idx, :, event_type_idx] = np.asarray(
            [EVENT_TYPE_CODES.get(value, EVENT_TYPE_CODES["other"]) for value in event_types],
            dtype=np.float32,
        )
        x_events[row_idx, :, trade_side_idx] = np.asarray(
            [TRADE_SIDE_CODES.get(value, TRADE_SIDE_CODES["unknown"]) for value in trade_sides],
            dtype=np.float32,
        )

    context = np.zeros((rows, len(CONTEXT_COLUMNS)), dtype=np.float32)
    for idx, column in enumerate(CONTEXT_COLUMNS):
        if column in frame.columns:
            if frame[column].dtype == bool:
                context[:, idx] = frame[column].fillna(False).astype(bool).astype("int8").to_numpy(dtype=np.float32)
            else:
                context[:, idx] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)

    y_mid_move = np.column_stack(
        [pd.to_numeric(frame[f"mid_move_{horizon}s"], errors="coerce").to_numpy(dtype=np.float32) for horizon in HORIZONS]
    )
    y_mid_move_sign = np.column_stack(
        [pd.to_numeric(frame[f"mid_move_sign_{horizon}s"], errors="coerce").fillna(0).to_numpy(dtype=np.int8) for horizon in HORIZONS]
    )
    y_target_available = np.column_stack(
        [frame[f"target_{horizon}s_available"].fillna(False).astype(bool).to_numpy(dtype=np.bool_) for horizon in HORIZONS]
    )
    y_mfe = np.column_stack(
        [pd.to_numeric(frame[f"max_favorable_excursion_{horizon}s"], errors="coerce").to_numpy(dtype=np.float32) for horizon in EXCURSION_HORIZONS]
    )
    y_mae = np.column_stack(
        [pd.to_numeric(frame[f"max_adverse_excursion_{horizon}s"], errors="coerce").to_numpy(dtype=np.float32) for horizon in EXCURSION_HORIZONS]
    )

    return {
        "x_events": x_events,
        "event_mask": event_mask,
        "events_recv_ts_ns": events_recv_ts_ns,
        "x_context": context,
        "y_mid_move": y_mid_move,
        "y_mid_move_sign": y_mid_move_sign,
        "y_target_available": y_target_available,
        "y_max_favorable_excursion": y_mfe,
        "y_max_adverse_excursion": y_mae,
        "y_hold_to_close_edge_vs_mid": pd.to_numeric(frame["hold_to_close_edge_vs_mid"], errors="coerce").to_numpy(dtype=np.float32),
        "y_terminal_margin_to_strike": pd.to_numeric(frame["terminal_margin_to_strike"], errors="coerce").to_numpy(dtype=np.float32),
        "y_flip_happened_after_t": frame["flip_happened_after_t"].fillna(False).astype(bool).to_numpy(dtype=np.bool_),
        "sequence_id": _string_array(frame["sequence_id"], 32),
        "market_slug": _string_array(frame["market_slug"], 256),
        "token_asset_id": _string_array(frame["token_asset_id"], 128),
        "token_side_normalized": _string_array(frame["token_side_normalized"], 8),
        "anchor_source": _string_array(frame["anchor_source"], 32),
        "anchor_ts_ns": pd.to_numeric(frame["anchor_ts_ns"], errors="coerce").fillna(0).to_numpy(dtype=np.int64),
        "sequence_length_actual": pd.to_numeric(frame["sequence_length_actual"], errors="coerce").fillna(0).to_numpy(dtype=np.int16),
        "sequence_feature_eligible": frame["sequence_feature_eligible"].fillna(False).astype(bool).to_numpy(dtype=np.bool_),
        "sequence_complete_eligible": frame["sequence_complete_eligible"].fillna(False).astype(bool).to_numpy(dtype=np.bool_),
        "variant": np.asarray([variant] * rows, dtype="<U16"),
        "event_channel_names": np.asarray(EVENT_CHANNELS, dtype="<U64"),
        "context_feature_names": np.asarray(CONTEXT_COLUMNS, dtype="<U64"),
        "target_horizons_s": np.asarray(HORIZONS, dtype=np.int16),
        "excursion_horizons_s": np.asarray(EXCURSION_HORIZONS, dtype=np.int16),
        "event_type_vocab": np.asarray([f"{key}:{value}" for key, value in sorted(EVENT_TYPE_CODES.items(), key=lambda item: item[1])], dtype="<U32"),
        "trade_side_vocab": np.asarray([f"{key}:{value}" for key, value in sorted(TRADE_SIDE_CODES.items(), key=lambda item: item[1])], dtype="<U32"),
    }


def write_tensor_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> Path:
    payload = {
        "dataset_name": TENSOR_DATASET,
        "lineage": "Derived from audited microstructure_sequence_dataset_v1_event64/event128 parquet shards.",
        "sample_unit": "One Phase 8 sequence sample: market_slug, token_asset_id, anchor_ts_ns, anchor_source.",
        "shard_naming": "data/datasets/microstructure_sequence_dataset_v1_tensor/{variant}/YYYY-MM-DD.partNNN.npz",
        "variants": VARIANT_SPECS,
        "arrays": {
            "x_events": {"dtype": "float32", "shape": ["n", "sequence_length", len(EVENT_CHANNELS)], "channels": EVENT_CHANNELS},
            "event_mask": {"dtype": "bool", "shape": ["n", "sequence_length"]},
            "events_recv_ts_ns": {"dtype": "int64", "shape": ["n", "sequence_length"], "causal_rule": "masked event recv_ts_ns < anchor_ts_ns"},
            "x_context": {"dtype": "float32", "shape": ["n", len(CONTEXT_COLUMNS)], "features": CONTEXT_COLUMNS},
            "y_mid_move": {"dtype": "float32", "shape": ["n", len(HORIZONS)], "horizons_s": list(HORIZONS), "leakage_only": True},
            "y_mid_move_sign": {"dtype": "int8", "shape": ["n", len(HORIZONS)], "horizons_s": list(HORIZONS), "leakage_only": True},
            "y_target_available": {"dtype": "bool", "shape": ["n", len(HORIZONS)], "horizons_s": list(HORIZONS)},
            "y_max_favorable_excursion": {"dtype": "float32", "shape": ["n", len(EXCURSION_HORIZONS)], "horizons_s": list(EXCURSION_HORIZONS), "leakage_only": True},
            "y_max_adverse_excursion": {"dtype": "float32", "shape": ["n", len(EXCURSION_HORIZONS)], "horizons_s": list(EXCURSION_HORIZONS), "leakage_only": True},
            "sequence_id": {"dtype": "unicode", "shape": ["n"]},
            "anchor_ts_ns": {"dtype": "int64", "shape": ["n"]},
        },
        "padding": "Pre-anchor padding positions use event_mask=False and event_type_code=0.",
        "missing_numeric": "NaN is preserved for unavailable numeric event/context/target values.",
    }
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return schema_path


def _source_path(dataset_root: Path, variant: str, target_date: date) -> Path:
    return dataset_root / VARIANT_SPECS[variant]["source_dataset"] / f"{target_date.isoformat()}.parquet"


def _tensor_output_path(output_dir: Path, variant: str, target_date: date, part_index: int) -> Path:
    return output_dir / TENSOR_DATASET / variant / f"{target_date.isoformat()}.part{part_index:03d}.npz"


def _columns_for_export(path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow.names
    wanted = [
        "sequence_id",
        "market_slug",
        "token_asset_id",
        "token_side_normalized",
        "anchor_ts_ns",
        "anchor_source",
        "events_recv_ts_ns",
        "events_event_type",
        "events_trade_side",
        "sequence_length_actual",
        "sequence_feature_eligible",
        "sequence_complete_eligible",
        "hold_to_close_edge_vs_mid",
        "terminal_margin_to_strike",
        "flip_happened_after_t",
        *EVENT_NUMERIC_CHANNELS,
        *CONTEXT_COLUMNS,
        *TARGET_COLUMNS,
        *[f"target_{horizon}s_available" for horizon in HORIZONS],
    ]
    selected: list[str] = []
    seen: set[str] = set()
    for column in wanted:
        if column in schema and column not in seen:
            selected.append(column)
            seen.add(column)
    return selected


def _build_tensor_day_variant(
    *,
    dataset_root: Path,
    output_dir: Path,
    target_day: date,
    variant: str,
    config: TensorExportConfig,
) -> dict[str, Any]:
    source = _source_path(dataset_root, variant, target_day)
    if not source.exists():
        return {
            "date": target_day.isoformat(),
            "variant": variant,
            "source_path": str(source),
            "status": "MISSING_SOURCE",
            "rows": 0,
            "parts": [],
        }
    if config.skip_existing:
        existing = _existing_tensor_day_variant_summary(
            dataset_root=dataset_root,
            output_dir=output_dir,
            target_day=target_day,
            variant=variant,
        )
        if existing is not None:
            return existing
    for old_part in _iter_tensor_parts(output_dir, variant, target_day):
        old_part.unlink()
    parquet = pq.ParquetFile(source)
    sequence_length = int(VARIANT_SPECS[variant]["sequence_length"])
    source_rows = int(parquet.metadata.num_rows)
    rows = 0
    parts: list[dict[str, Any]] = []
    columns = _columns_for_export(source)
    if source_rows == 0:
        frame = pq.read_table(source, columns=columns).to_pandas()
        arrays = frame_to_tensor_arrays(frame, sequence_length=sequence_length, variant=variant)
        output_path = _tensor_output_path(output_dir, variant, target_day, 0)
        _write_deterministic_npz(output_path, arrays, config.compression_level)
        parts.append(
            {
                "part": 0,
                "path": str(output_path),
                "rows": 0,
                "sha256": _sha256(output_path),
                "size_bytes": int(output_path.stat().st_size),
            }
        )
        return {
            "date": target_day.isoformat(),
            "variant": variant,
            "source_path": str(source),
            "source_rows": source_rows,
            "rows": 0,
            "parts": parts,
            "status": "PASS",
        }
    for part_index, batch in enumerate(parquet.iter_batches(columns=columns, batch_size=int(config.rows_per_shard))):
        frame = batch.to_pandas()
        arrays = frame_to_tensor_arrays(frame, sequence_length=sequence_length, variant=variant)
        output_path = _tensor_output_path(output_dir, variant, target_day, part_index)
        _write_deterministic_npz(output_path, arrays, config.compression_level)
        rows += int(len(frame))
        parts.append(
            {
                "part": int(part_index),
                "path": str(output_path),
                "rows": int(len(frame)),
                "sha256": _sha256(output_path),
                "size_bytes": int(output_path.stat().st_size),
            }
        )
    return {
        "date": target_day.isoformat(),
        "variant": variant,
        "source_path": str(source),
        "source_rows": source_rows,
        "rows": rows,
        "parts": parts,
        "status": "PASS" if rows == source_rows else "FAIL",
    }


def _existing_tensor_day_variant_summary(
    *,
    dataset_root: Path,
    output_dir: Path,
    target_day: date,
    variant: str,
) -> dict[str, Any] | None:
    source = _source_path(dataset_root, variant, target_day)
    parts = _iter_tensor_parts(output_dir, variant, target_day)
    if not parts:
        return None
    try:
        source_rows = int(pq.ParquetFile(source).metadata.num_rows)
        rows = 0
        part_rows: list[dict[str, Any]] = []
        for index, part in enumerate(parts):
            with np.load(part, allow_pickle=False) as payload:
                count = int(payload["x_events"].shape[0])
            rows += count
            part_rows.append(
                {
                    "part": int(index),
                    "path": str(part),
                    "rows": count,
                    "sha256": _sha256(part),
                    "size_bytes": int(part.stat().st_size),
                }
            )
    except Exception:
        return None
    return {
        "date": target_day.isoformat(),
        "variant": variant,
        "source_path": str(source),
        "source_rows": source_rows,
        "rows": rows,
        "parts": part_rows,
        "status": "SKIPPED_EXISTING" if rows == source_rows else "FAIL",
        "skipped_existing": True,
    }


def build_microstructure_tensor_exports(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_DATASET_ROOT,
    date_from: date,
    date_to: date,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    report_path: Path = DEFAULT_BUILD_REPORT_PATH,
    config: TensorExportConfig = TensorExportConfig(),
) -> dict[str, Any]:
    schema_written = write_tensor_schema(schema_path)
    target_dates = _date_range(date_from, date_to)
    tasks = [(target_day, variant) for target_day in target_dates for variant in config.variants]
    worker_count = (os.cpu_count() or 1) if config.max_workers == 0 else int(config.max_workers)
    worker_count = max(1, min(worker_count, len(tasks) or 1))
    if worker_count == 1:
        daily = [
            _build_tensor_day_variant(
                dataset_root=dataset_root,
                output_dir=output_dir,
                target_day=target_day,
                variant=variant,
                config=config,
            )
            for target_day, variant in tasks
        ]
    else:
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _build_tensor_day_variant,
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    target_day=target_day,
                    variant=variant,
                    config=config,
                ): (target_day, variant)
                for target_day, variant in tasks
            }
            for future in as_completed(futures):
                target_day, variant = futures[future]
                by_key[(target_day.isoformat(), variant)] = future.result()
        daily = [by_key[(target_day.isoformat(), variant)] for target_day, variant in tasks]
    report = {
        "dataset": TENSOR_DATASET,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "schema_path": str(schema_written),
        "config": {
            "variants": list(config.variants),
            "rows_per_shard": int(config.rows_per_shard),
            "compression_level": int(config.compression_level),
            "max_workers": int(worker_count),
            "skip_existing": bool(config.skip_existing),
        },
        "daily": daily,
    }
    _write_tensor_build_report(report, report_path)
    return report


def _write_tensor_build_report(report: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Microstructure Tensor Export Build Report",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Date range: `{report['date_from']}` to `{report['date_to']}`",
        f"- Schema: `{report['schema_path']}`",
        f"- Rows per tensor shard: `{report['config']['rows_per_shard']}`",
        "- Scope: dataset-factory tensor export only; no model training.",
        "",
        "## Lineage",
        "",
        "- Source of truth: audited `microstructure_sequence_dataset_v1_event64` and `microstructure_sequence_dataset_v1_event128` parquet shards.",
        "- Output format: deterministic shard-level NPZ with fixed ZIP metadata and stable array ordering.",
        "",
        "## Shards",
        "",
        "| date | variant | source rows | tensor rows | parts | status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for day in report["daily"]:
        lines.append(
            f"| {day['date']} | {day['variant']} | {day.get('source_rows', 0)} | {day.get('rows', 0)} | "
            f"{len(day.get('parts', []))} | {day['status']} |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _iter_tensor_parts(output_dir: Path, variant: str, target_day: date) -> list[Path]:
    directory = output_dir / TENSOR_DATASET / variant
    return sorted(directory.glob(f"{target_day.isoformat()}.part*.npz"))


def audit_microstructure_tensor_exports(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    tensor_root: Path = DEFAULT_DATASET_ROOT,
    date_from: date,
    date_to: date,
    output_path: Path = DEFAULT_AUDIT_REPORT_PATH,
    variants: tuple[str, ...] = ("event64", "event128"),
) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    daily: list[dict[str, Any]] = []
    for target_day in _date_range(date_from, date_to):
        for variant in variants:
            source = _source_path(dataset_root, variant, target_day)
            parts = _iter_tensor_parts(tensor_root, variant, target_day)
            expected_len = int(VARIANT_SPECS[variant]["sequence_length"])
            if not source.exists():
                findings.append({"severity": "CRITICAL", "check": "source_exists", "message": f"Missing source parquet {source}"})
                continue
            if not parts:
                findings.append({"severity": "CRITICAL", "check": "tensor_exists", "message": f"Missing tensor parts for {variant} {target_day}"})
                continue
            source_rows = int(pq.ParquetFile(source).metadata.num_rows)
            rows = 0
            for part in parts:
                with np.load(part, allow_pickle=False) as payload:
                    x_events = payload["x_events"]
                    event_mask = payload["event_mask"]
                    recv_ts = payload["events_recv_ts_ns"]
                    anchor_ts = payload["anchor_ts_ns"]
                    target_available = payload["y_target_available"]
                    y_mid_move = payload["y_mid_move"]
                    if x_events.ndim != 3 or x_events.shape[1] != expected_len or x_events.shape[2] != len(EVENT_CHANNELS):
                        findings.append({"severity": "CRITICAL", "check": "shape", "message": f"{part} invalid x_events shape {x_events.shape}"})
                    if event_mask.shape != (x_events.shape[0], expected_len):
                        findings.append({"severity": "CRITICAL", "check": "mask_shape", "message": f"{part} invalid event_mask shape {event_mask.shape}"})
                    if recv_ts.shape != event_mask.shape:
                        findings.append({"severity": "CRITICAL", "check": "recv_ts_shape", "message": f"{part} invalid events_recv_ts_ns shape {recv_ts.shape}"})
                    future_mask = event_mask & (recv_ts >= anchor_ts.reshape(-1, 1))
                    if bool(future_mask.any()):
                        findings.append({"severity": "CRITICAL", "check": "causality", "message": f"{part} contains future or anchor-time events"})
                    unavailable_populated = (~target_available) & ~np.isnan(y_mid_move)
                    if bool(unavailable_populated.any()):
                        findings.append({"severity": "CRITICAL", "check": "target_alignment", "message": f"{part} has populated y_mid_move where target unavailable"})
                    rows += int(x_events.shape[0])
            if rows != source_rows:
                findings.append({"severity": "CRITICAL", "check": "row_count", "message": f"{variant} {target_day}: tensor rows {rows} != source rows {source_rows}"})
            daily.append(
                {
                    "date": target_day.isoformat(),
                    "variant": variant,
                    "source_rows": source_rows,
                    "tensor_rows": rows,
                    "parts": len(parts),
                    "status": "PASS" if rows == source_rows else "FAIL",
                }
            )
    critical_count = sum(1 for finding in findings if finding["severity"] == "CRITICAL")
    report = {
        "dataset": TENSOR_DATASET,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "critical_count": critical_count,
        "warning_count": sum(1 for finding in findings if finding["severity"] == "WARNING"),
        "verdict": "PASS" if critical_count == 0 else "FAIL",
        "findings": findings,
        "daily": daily,
    }
    _write_tensor_audit_report(report, output_path)
    return report


def _write_tensor_audit_report(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Microstructure Tensor Export Audit Report",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Date range: `{report['date_from']}` to `{report['date_to']}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Critical findings: `{report['critical_count']}`",
        f"- Warning findings: `{report['warning_count']}`",
        "",
        "## Daily Summary",
        "",
        "| date | variant | source rows | tensor rows | parts | status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for day in report["daily"]:
        lines.append(
            f"| {day['date']} | {day['variant']} | {day['source_rows']} | {day['tensor_rows']} | {day['parts']} | {day['status']} |"
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for idx, finding in enumerate(report["findings"], 1):
            lines.append(f"- `{finding['severity']}-{idx:03d}` `{finding['check']}`: {finding['message']}")
    else:
        lines.append("- No findings.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_microstructure_tensor_reproducibility_report(
    *,
    tensor_root: Path = DEFAULT_DATASET_ROOT,
    date_from: date,
    date_to: date,
    output_path: Path = DEFAULT_REPRO_REPORT_PATH,
    variants: tuple[str, ...] = ("event64", "event128"),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target_day in _date_range(date_from, date_to):
        for variant in variants:
            for part in _iter_tensor_parts(tensor_root, variant, target_day):
                rows.append(
                    {
                        "date": target_day.isoformat(),
                        "variant": variant,
                        "path": str(part),
                        "sha256": _sha256(part),
                        "size_bytes": int(part.stat().st_size),
                    }
                )
    report = {
        "dataset": TENSOR_DATASET,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "parts": rows,
        "verdict": "PASS" if rows else "FAIL",
    }
    lines = [
        "# Microstructure Tensor Export Reproducibility Report",
        "",
        f"- Dataset: `{TENSOR_DATASET}`",
        f"- Date range: `{date_from.isoformat()}` to `{date_to.isoformat()}`",
        f"- Result: `{report['verdict']}`",
        "- Determinism contract: NPZ archives are written with fixed ZIP timestamps, sorted array names, stable sample order inherited from source parquet, and SHA-256 content hashes.",
        "",
        "| date | variant | size bytes | sha256 | path |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['date']} | {row['variant']} | {row['size_bytes']} | `{row['sha256']}` | `{row['path']}` |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
