from __future__ import annotations

import math
import random
import heapq
from bisect import bisect_right
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from binance_recorder.compression import iter_zstd_jsonl
from binance_recorder.utils import ns_to_iso
from market_recorders.dataset_policy import (
    SCRIPTURE_EXPLICIT_SOURCE_HOUR_QUARANTINES,
    DatasetPolicy,
    load_dataset_policy,
)
from market_recorders.unified_reader import UnifiedRawReader

SECOND_NS = 1_000_000_000
STALE_SECONDS = 30.0
SPOT_PREMIUM_MARKUP_BPS = 2.9
DELAYED_RESIDUAL_LOOKBACK_SECONDS = 3_600
DELAYED_RESIDUAL_DELAY_SECONDS = 1_800
DELAYED_RESIDUAL_WINDOW_SECONDS = DELAYED_RESIDUAL_LOOKBACK_SECONDS - DELAYED_RESIDUAL_DELAY_SECONDS
DELAYED_RESIDUAL_SANITY_LIMIT_USD = 250.0
DEFAULT_DATE_FROM = date(2026, 4, 19)
DEFAULT_DATE_TO = date(2026, 4, 22)
DEFAULT_OUTPUT_PATH = Path("canonical/live_reference_events_v1")
DEFAULT_VALIDATION_PATH = Path("docs/live_reference_events_v1_validation.md")
DEFAULT_SCHEMA_PATH = Path("config/schemas/live_reference_events_v1_schema.yaml")
SOURCE_ORDER = ("binance_spot", "binance_usdm", "hyperliquid")


@dataclass(slots=True, frozen=True)
class SourceSpec:
    source_name: str
    relative_prefix: Path
    kind: str
    parser_kind: str
    role: str


@dataclass(slots=True, frozen=True)
class SourceUpdate:
    effective_ts_seconds: int
    recv_ts_ns: int
    mid: float


@dataclass(slots=True, frozen=True)
class CalibrationObservation:
    chainlink_ts_seconds: int
    recv_ts_ns: int
    available_ts_seconds: int
    price: float


@dataclass(slots=True, frozen=True)
class DelayedBiasResidual:
    chainlink_ts_seconds: int
    active_start_seconds: int
    active_end_seconds: int
    residual: float


@dataclass(slots=True, frozen=True)
class SourceFileDecision:
    source_name: str
    date_str: str
    hour_str: str
    raw_path: str
    manifest_path: str
    eligible: bool
    reason_category: str | None
    quality_class: str | None
    file_state: str | None
    source_tier: str | None
    valid_row_count: int
    rows: tuple[Any, ...]


SOURCE_SPECS = {
    "binance_spot": SourceSpec(
        source_name="binance_spot",
        relative_prefix=Path("binance") / "spot" / "BTCUSDT",
        kind="ws",
        parser_kind="binance_ws",
        role="primary",
    ),
    "binance_usdm": SourceSpec(
        source_name="binance_usdm",
        relative_prefix=Path("binance") / "usdm" / "BTCUSDT",
        kind="ws",
        parser_kind="binance_ws",
        role="primary",
    ),
    "hyperliquid": SourceSpec(
        source_name="hyperliquid",
        relative_prefix=Path("hyperliquid") / "linear" / "BTC",
        kind="ws",
        parser_kind="hyperliquid_ws",
        role="primary",
    ),
    "chainlink_public_delayed": SourceSpec(
        source_name="chainlink_public_delayed",
        relative_prefix=Path("chainlink_public_delayed") / "public_stream_page" / "BTCUSD",
        kind="page",
        parser_kind="public_delayed_page",
        role="calibration",
    ),
}


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    if lower == upper:
        return lower_value
    return lower_value + ((upper_value - lower_value) * (rank - lower))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _spot_only_premium_price(*, spot_mid: float | None) -> float | None:
    if not _is_finite(spot_mid):
        return None
    spot = float(spot_mid)
    return spot * (1.0 + (SPOT_PREMIUM_MARKUP_BPS / 10_000.0))


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _iter_dates(date_from: date, date_to: date) -> list[date]:
    values: list[date] = []
    current = date_from
    while current <= date_to:
        values.append(current)
        current += timedelta(days=1)
    return values


def _ceil_second(ts_ns: int) -> int:
    return (ts_ns + SECOND_NS - 1) // SECOND_NS


def _hour_key(ts_seconds: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(ts_seconds, tz=UTC)
    return dt.date().isoformat(), f"{dt.hour:02d}"


def _discover_earliest_raw_source_date(root: Path, source_names: tuple[str, ...]) -> date | None:
    dates: list[date] = []
    for source_name in source_names:
        spec = SOURCE_SPECS[source_name]
        source_root = root / "raw" / spec.relative_prefix
        if not source_root.exists():
            continue
        for child in source_root.iterdir():
            if not child.is_dir():
                continue
            try:
                dates.append(date.fromisoformat(child.name))
            except ValueError:
                continue
    return min(dates) if dates else None


def _manifest_path_for_raw_path(root: Path, raw_path: Path) -> Path:
    relative = raw_path.relative_to(root / "raw")
    return (root / "manifests" / relative).with_suffix("").with_suffix(".manifest.json")


def _parse_binance_ws_rows(path: Path) -> tuple[SourceUpdate, ...]:
    by_second: dict[int, SourceUpdate] = {}
    for record in iter_zstd_jsonl(path):
        normalized = record.get("normalized") or {}
        bid = _to_float(normalized.get("best_bid_price"))
        ask = _to_float(normalized.get("best_ask_price"))
        recv_ts_ns = _to_int(record.get("recv_ts_ns"))
        if bid is None or ask is None or recv_ts_ns is None:
            continue
        effective_ts_seconds = _ceil_second(recv_ts_ns)
        update = SourceUpdate(
            effective_ts_seconds=effective_ts_seconds,
            recv_ts_ns=recv_ts_ns,
            mid=(bid + ask) / 2.0,
        )
        current = by_second.get(effective_ts_seconds)
        if current is None or update.recv_ts_ns >= current.recv_ts_ns:
            by_second[effective_ts_seconds] = update
    return tuple(by_second[key] for key in sorted(by_second))


def _parse_hyperliquid_ws_rows(path: Path) -> tuple[SourceUpdate, ...]:
    by_second: dict[int, SourceUpdate] = {}
    for record in iter_zstd_jsonl(path):
        if str(record.get("topic")) != "l2Book":
            continue
        payload = record.get("payload") or {}
        data = payload.get("data") or {}
        levels = data.get("levels")
        recv_ts_ns = _to_int(record.get("recv_ts_ns"))
        if recv_ts_ns is None or not isinstance(levels, list) or len(levels) < 2:
            continue
        bids = levels[0]
        asks = levels[1]
        if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
            continue
        bid_prices = [_to_float(level.get("px")) for level in bids]
        ask_prices = [_to_float(level.get("px")) for level in asks]
        bid_prices = [value for value in bid_prices if value is not None]
        ask_prices = [value for value in ask_prices if value is not None]
        if not bid_prices or not ask_prices:
            continue
        best_bid = max(bid_prices)
        best_ask = min(ask_prices)
        effective_ts_seconds = _ceil_second(recv_ts_ns)
        update = SourceUpdate(
            effective_ts_seconds=effective_ts_seconds,
            recv_ts_ns=recv_ts_ns,
            mid=(best_bid + best_ask) / 2.0,
        )
        current = by_second.get(effective_ts_seconds)
        if current is None or update.recv_ts_ns >= current.recv_ts_ns:
            by_second[effective_ts_seconds] = update
    return tuple(by_second[key] for key in sorted(by_second))


def _parse_public_delayed_rows(path: Path, availability_offset_seconds: int) -> tuple[CalibrationObservation, ...]:
    earliest_by_chainlink_second: dict[int, CalibrationObservation] = {}
    for record in iter_zstd_jsonl(path):
        recv_ts_ns = _to_int(record.get("recv_ts_ns"))
        display_ts = _to_int(record.get("chainlink_display_ts"))
        price = _to_float(record.get("btc_usd_price"))
        if recv_ts_ns is None or display_ts is None or price is None:
            continue
        chainlink_ts_seconds = display_ts if abs(display_ts) < 10_000_000_000 else display_ts // SECOND_NS
        available_ts_seconds = max(
            _ceil_second(recv_ts_ns),
            int(chainlink_ts_seconds) + availability_offset_seconds,
        )
        observation = CalibrationObservation(
            chainlink_ts_seconds=int(chainlink_ts_seconds),
            recv_ts_ns=recv_ts_ns,
            available_ts_seconds=available_ts_seconds,
            price=price,
        )
        current = earliest_by_chainlink_second.get(observation.chainlink_ts_seconds)
        if current is None or observation.recv_ts_ns < current.recv_ts_ns:
            earliest_by_chainlink_second[observation.chainlink_ts_seconds] = observation
    return tuple(earliest_by_chainlink_second[key] for key in sorted(earliest_by_chainlink_second))


def _parse_rows_for_spec(path: Path, spec: SourceSpec, availability_offset_seconds: int) -> tuple[Any, ...]:
    if spec.parser_kind == "binance_ws":
        return _parse_binance_ws_rows(path)
    if spec.parser_kind == "hyperliquid_ws":
        return _parse_hyperliquid_ws_rows(path)
    if spec.parser_kind == "public_delayed_page":
        return _parse_public_delayed_rows(path, availability_offset_seconds)
    raise ValueError(f"Unsupported parser kind: {spec.parser_kind}")


def _primary_reason_for_file(
    *,
    policy: DatasetPolicy,
    source_name: str,
    date_str: str,
    hour_str: str,
    quality_class: str | None,
    file_state: str | None,
) -> str | None:
    if (source_name, date_str, hour_str) in SCRIPTURE_EXPLICIT_SOURCE_HOUR_QUARANTINES:
        return "source_specific_integrity_check"
    resolved_quality = quality_class or "unknown"
    if resolved_quality not in policy.allowed_manifest_quality_classes:
        return f"manifest_quality_class:{resolved_quality}"
    resolved_file_state = file_state or "unknown"
    if resolved_file_state not in policy.allowed_file_states:
        return f"file_state:{resolved_file_state}"
    return None


def _process_source_file(task: dict[str, Any]) -> SourceFileDecision:
    root = Path(task["root"])
    raw_path = Path(task["raw_path"])
    spec = SOURCE_SPECS[str(task["source_name"])]
    policy = load_dataset_policy(Path(task["policy_path"]))
    manifest_path = _manifest_path_for_raw_path(root, raw_path)
    reader = UnifiedRawReader(root=root)
    metadata = reader.file_metadata(raw_path)
    date_str = str(task["date_str"])
    hour_str = str(task["hour_str"])
    quality_class = str(metadata.get("quality_class")) if metadata.get("quality_class") is not None else None
    file_state = str(metadata.get("file_state")) if metadata.get("file_state") is not None else None
    source_tier = str(metadata.get("source_tier")) if metadata.get("source_tier") is not None else None
    primary_reason = _primary_reason_for_file(
        policy=policy,
        source_name=spec.source_name,
        date_str=date_str,
        hour_str=hour_str,
        quality_class=quality_class,
        file_state=file_state,
    )
    if primary_reason is not None:
        return SourceFileDecision(
            source_name=spec.source_name,
            date_str=date_str,
            hour_str=hour_str,
            raw_path=str(raw_path),
            manifest_path=str(manifest_path),
            eligible=False,
            reason_category=primary_reason,
            quality_class=quality_class,
            file_state=file_state,
            source_tier=source_tier,
            valid_row_count=0,
            rows=(),
        )

    try:
        rows = _parse_rows_for_spec(
            raw_path,
            spec,
            int(task["availability_offset_seconds"]),
        )
    except Exception:
        return SourceFileDecision(
            source_name=spec.source_name,
            date_str=date_str,
            hour_str=hour_str,
            raw_path=str(raw_path),
            manifest_path=str(manifest_path),
            eligible=False,
            reason_category="parse_failure",
            quality_class=quality_class,
            file_state=file_state,
            source_tier=source_tier,
            valid_row_count=0,
            rows=(),
        )

    if not rows:
        return SourceFileDecision(
            source_name=spec.source_name,
            date_str=date_str,
            hour_str=hour_str,
            raw_path=str(raw_path),
            manifest_path=str(manifest_path),
            eligible=False,
            reason_category="missing_required_fields",
            quality_class=quality_class,
            file_state=file_state,
            source_tier=source_tier,
            valid_row_count=0,
            rows=(),
        )

    return SourceFileDecision(
        source_name=spec.source_name,
        date_str=date_str,
        hour_str=hour_str,
        raw_path=str(raw_path),
        manifest_path=str(manifest_path),
        eligible=True,
        reason_category=None,
        quality_class=quality_class,
        file_state=file_state,
        source_tier=source_tier,
        valid_row_count=len(rows),
        rows=rows,
    )


def _run_parallel(tasks: list[dict[str, Any]], *, max_workers: int) -> list[SourceFileDecision]:
    if not tasks:
        return []
    if max_workers <= 1:
        return [_process_source_file(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_process_source_file, tasks))


def _list_source_tasks(
    *,
    root: Path,
    policy: DatasetPolicy,
    policy_path: Path,
    source_name: str,
    date_from: date,
    date_to: date,
) -> list[dict[str, Any]]:
    spec = SOURCE_SPECS[source_name]
    reader = UnifiedRawReader(root=root)
    tasks: list[dict[str, Any]] = []
    availability_offset_seconds = 0
    if spec.role == "calibration":
        availability_offset_seconds = int(
            policy.live_reference_calibration_inputs[source_name]["causal_availability_offset_seconds"]
        )
    for target_date in _iter_dates(date_from, date_to):
        for raw_path in reader.list_files_for_date(
            relative_prefix=spec.relative_prefix,
            target_date=target_date,
            kind=spec.kind,
            finalized_only=False,
        ):
            tasks.append(
                {
                    "root": str(root),
                    "raw_path": str(raw_path),
                    "source_name": source_name,
                    "date_str": target_date.isoformat(),
                    "hour_str": raw_path.name.split(".")[0],
                    "policy_path": str(policy_path),
                    "availability_offset_seconds": availability_offset_seconds,
                }
            )
    return tasks


DEFAULT_POLICY_PATH = Path("config/dataset_policy_v1.yaml")


def _load_source_hours(
    *,
    root: Path,
    policy: DatasetPolicy,
    policy_path: Path,
    source_name: str,
    date_from: date,
    date_to: date,
    max_workers: int,
) -> list[SourceFileDecision]:
    tasks = _list_source_tasks(
        root=root,
        policy=policy,
        policy_path=policy_path,
        source_name=source_name,
        date_from=date_from,
        date_to=date_to,
    )
    return _run_parallel(tasks, max_workers=max_workers)


def _rolling_volatility_bucket(values: list[float | None]) -> list[float | None]:
    realized_vol: list[float | None] = [None] * len(values)
    rolling_returns: deque[float] = deque()
    rolling_sum = 0.0
    previous_mid: float | None = None
    for index, mid in enumerate(values):
        if mid is None or previous_mid is None or previous_mid <= 0 or mid <= 0:
            rolling_returns.clear()
            rolling_sum = 0.0
            previous_mid = mid
            realized_vol[index] = None
            continue
        log_return = math.log(mid / previous_mid)
        squared = log_return * log_return
        rolling_returns.append(squared)
        rolling_sum += squared
        if len(rolling_returns) > 300:
            rolling_sum -= rolling_returns.popleft()
        previous_mid = mid
        realized_vol[index] = math.sqrt(max(rolling_sum, 0.0))
    return realized_vol


def _terciles(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return _percentile(values, 1.0 / 3.0), _percentile(values, 2.0 / 3.0)


def _vol_bucket(value: float | None, *, lower: float | None, upper: float | None) -> str:
    if value is None or lower is None or upper is None:
        return "spot_missing"
    if value <= lower:
        return "low"
    if value <= upper:
        return "medium"
    return "high"


def _build_quarantine_summary(decisions: list[SourceFileDecision]) -> tuple[dict[str, dict[str, int]], int]:
    summary: dict[str, dict[str, int]] = {}
    excluded = 0
    for decision in decisions:
        if decision.eligible:
            continue
        excluded += 1
        bucket = summary.setdefault(decision.source_name, {})
        reason = decision.reason_category or "unknown"
        bucket[reason] = bucket.get(reason, 0) + 1
    return summary, excluded


def _build_delayed_bias_residuals(
    *,
    calibration_rows: list[CalibrationObservation],
    synthetic_raw_history: dict[int, float],
) -> tuple[list[DelayedBiasResidual], int]:
    residuals: list[DelayedBiasResidual] = []
    anomaly_count = 0
    for observation in calibration_rows:
        base_price = synthetic_raw_history.get(observation.chainlink_ts_seconds)
        if not _is_finite(base_price) or not _is_finite(observation.price):
            continue
        residual = float(observation.price) - float(base_price)
        if not math.isfinite(residual) or abs(residual) > DELAYED_RESIDUAL_SANITY_LIMIT_USD:
            anomaly_count += 1
            continue
        active_start = max(
            int(observation.available_ts_seconds),
            int(observation.chainlink_ts_seconds + DELAYED_RESIDUAL_DELAY_SECONDS),
        )
        active_end = int(observation.chainlink_ts_seconds + DELAYED_RESIDUAL_LOOKBACK_SECONDS)
        if active_start > active_end:
            continue
        residuals.append(
            DelayedBiasResidual(
                chainlink_ts_seconds=int(observation.chainlink_ts_seconds),
                active_start_seconds=active_start,
                active_end_seconds=active_end,
                residual=residual,
            )
        )
    residuals.sort(key=lambda row: (row.active_start_seconds, row.chainlink_ts_seconds))
    return residuals, anomaly_count


def _apply_delayed_residual_bias(
    *,
    rows: list[dict[str, Any]],
    residuals: list[DelayedBiasResidual],
) -> None:
    if not rows:
        return
    first_ts_seconds = int(rows[0]["ts_seconds"])
    starts: dict[int, list[DelayedBiasResidual]] = {}
    removals: dict[int, list[DelayedBiasResidual]] = {}
    for residual in residuals:
        if residual.active_end_seconds < first_ts_seconds:
            continue
        starts.setdefault(max(residual.active_start_seconds, first_ts_seconds), []).append(residual)
        removals.setdefault(residual.active_end_seconds + 1, []).append(residual)

    active_sum = 0.0
    active_count = 0
    active_latest_heap: list[tuple[int, int]] = []
    latest_usable_chainlink_ts: int | None = None
    last_valid_bias: float | None = None
    last_valid_bias_chainlink_ts: int | None = None

    for row in rows:
        ts_seconds = int(row["ts_seconds"])
        for residual in starts.get(ts_seconds, ()):
            active_sum += residual.residual
            active_count += 1
            latest_usable_chainlink_ts = (
                residual.chainlink_ts_seconds
                if latest_usable_chainlink_ts is None
                else max(latest_usable_chainlink_ts, residual.chainlink_ts_seconds)
            )
            heapq.heappush(active_latest_heap, (-residual.chainlink_ts_seconds, residual.active_end_seconds))
        for residual in removals.get(ts_seconds, ()):
            active_sum -= residual.residual
            active_count -= 1
        while active_latest_heap and active_latest_heap[0][1] < ts_seconds:
            heapq.heappop(active_latest_heap)

        bias_active = active_count > 0
        bias_bootstrapped = latest_usable_chainlink_ts is not None
        active_window_bias = float(active_sum / active_count) if bias_active else math.nan
        latest_active_chainlink_ts = -active_latest_heap[0][0] if bias_active and active_latest_heap else None
        if bias_active:
            last_valid_bias = float(active_window_bias)
            last_valid_bias_chainlink_ts = latest_active_chainlink_ts
        bias_carried_forward = bool(
            not bias_active
            and last_valid_bias is not None
            and last_valid_bias_chainlink_ts is not None
        )
        if bias_active:
            rolling_bias = float(active_window_bias)
            bias_mode = "active_window"
            applied_bias_age_seconds = (
                float(ts_seconds - latest_active_chainlink_ts)
                if latest_active_chainlink_ts is not None
                else math.nan
            )
        elif bias_carried_forward:
            rolling_bias = float(last_valid_bias)
            bias_mode = "carried_forward"
            applied_bias_age_seconds = float(ts_seconds - int(last_valid_bias_chainlink_ts))
        else:
            rolling_bias = 0.0
            bias_mode = "unavailable"
            applied_bias_age_seconds = math.nan
        bias_state_age_seconds = (
            float(ts_seconds - latest_active_chainlink_ts)
            if latest_active_chainlink_ts is not None
            else math.nan
        )
        last_bias_observation_age_seconds = (
            float(ts_seconds - latest_usable_chainlink_ts)
            if latest_usable_chainlink_ts is not None
            else math.nan
        )
        price_valid = bool(row["price_valid"])
        synthetic_raw = float(row["synthetic_raw"]) if _is_finite(row.get("synthetic_raw")) else math.nan

        row["rolling_bias"] = rolling_bias
        row["active_window_bias"] = active_window_bias
        row["bias_state_age_seconds"] = bias_state_age_seconds
        row["applied_bias_age_seconds"] = applied_bias_age_seconds
        row["last_bias_observation_age_seconds"] = last_bias_observation_age_seconds
        row["bias_mode"] = bias_mode
        row["bias_bootstrapped"] = bool(bias_bootstrapped)
        row["bias_active"] = bool(bias_active)
        row["bias_carried_forward"] = bool(bias_carried_forward)
        row["bias_state_stale"] = bool(bias_bootstrapped and not bias_active)
        row["bias_neutralized"] = bool(bias_bootstrapped and not bias_active and not bias_carried_forward)
        row["bias_observation_count"] = int(active_count)
        row["synthetic_corrected"] = (
            float(synthetic_raw + rolling_bias)
            if price_valid and math.isfinite(synthetic_raw)
            else math.nan
        )


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows_"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        rendered: list[str] = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                rendered.append(_format_float(value))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _default_validation_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    sampler = random.Random(42)
    indices = sorted(sampler.sample(range(len(frame)), k=min(10, len(frame))))
    columns = [
        "ts_seconds",
        "synthetic_raw",
        "synthetic_corrected",
        "rolling_bias",
        "active_window_bias",
        "bias_state_age_seconds",
        "applied_bias_age_seconds",
        "last_bias_observation_age_seconds",
        "bias_mode",
        "bias_bootstrapped",
        "bias_active",
        "bias_carried_forward",
        "bias_state_stale",
        "bias_neutralized",
        "bias_observation_count",
        "source_count",
        "price_valid",
        "degraded",
        "binance_spot_mid",
        "binance_spot_staleness_s",
        "binance_usdm_mid",
        "binance_usdm_staleness_s",
        "hyperliquid_mid",
        "hyperliquid_staleness_s",
        "max_cross_source_spread",
    ]
    rows: list[dict[str, Any]] = []
    for index in indices:
        raw = frame.iloc[index]
        row = {column: raw[column] for column in columns}
        rows.append(row)
    return rows


def _write_parquet(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("recv_ts_ns", pa.int64()),
            ("ts_seconds", pa.int64()),
            ("synthetic_raw", pa.float64()),
            ("synthetic_corrected", pa.float64()),
            ("rolling_bias", pa.float64()),
            ("active_window_bias", pa.float64()),
            ("bias_state_age_seconds", pa.float64()),
            ("applied_bias_age_seconds", pa.float64()),
            ("last_bias_observation_age_seconds", pa.float64()),
            ("bias_mode", pa.string()),
            ("bias_bootstrapped", pa.bool_()),
            ("bias_active", pa.bool_()),
            ("bias_carried_forward", pa.bool_()),
            ("bias_state_stale", pa.bool_()),
            ("bias_neutralized", pa.bool_()),
            ("bias_observation_count", pa.int16()),
            ("source_count", pa.int8()),
            ("price_valid", pa.bool_()),
            ("degraded", pa.bool_()),
            ("binance_spot_mid", pa.float64()),
            ("binance_spot_staleness_s", pa.float64()),
            ("binance_usdm_mid", pa.float64()),
            ("binance_usdm_staleness_s", pa.float64()),
            ("hyperliquid_mid", pa.float64()),
            ("hyperliquid_staleness_s", pa.float64()),
            ("max_cross_source_spread", pa.float64()),
        ]
    )
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(
        table,
        output_path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )


def _coerce_output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[
        [
            "recv_ts_ns",
            "ts_seconds",
            "synthetic_raw",
            "synthetic_corrected",
            "rolling_bias",
            "active_window_bias",
            "bias_state_age_seconds",
            "applied_bias_age_seconds",
            "last_bias_observation_age_seconds",
            "bias_mode",
            "bias_bootstrapped",
            "bias_active",
            "bias_carried_forward",
            "bias_state_stale",
            "bias_neutralized",
            "bias_observation_count",
            "source_count",
            "price_valid",
            "degraded",
            "binance_spot_mid",
            "binance_spot_staleness_s",
            "binance_usdm_mid",
            "binance_usdm_staleness_s",
            "hyperliquid_mid",
            "hyperliquid_staleness_s",
            "max_cross_source_spread",
        ]
    ].copy()

    frame["recv_ts_ns"] = frame["recv_ts_ns"].astype("int64")
    frame["ts_seconds"] = frame["ts_seconds"].astype("int64")
    frame["source_count"] = frame["source_count"].astype("int8")
    for column in [
        "synthetic_raw",
        "synthetic_corrected",
        "rolling_bias",
        "active_window_bias",
        "bias_state_age_seconds",
        "applied_bias_age_seconds",
        "last_bias_observation_age_seconds",
        "binance_spot_mid",
        "binance_spot_staleness_s",
        "binance_usdm_mid",
        "binance_usdm_staleness_s",
        "hyperliquid_mid",
        "hyperliquid_staleness_s",
        "max_cross_source_spread",
    ]:
        frame[column] = frame[column].astype("float64")
    frame["bias_mode"] = frame["bias_mode"].astype("string").fillna("unavailable").astype(str)
    frame["bias_observation_count"] = frame["bias_observation_count"].astype("int16")
    for column in [
        "bias_bootstrapped",
        "bias_active",
        "bias_carried_forward",
        "bias_state_stale",
        "bias_neutralized",
        "price_valid",
        "degraded",
    ]:
        frame[column] = frame[column].astype("bool")
    return frame


def _write_schema_contract(schema_path: Path) -> str:
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": "live_reference_events_v1",
        "description": "Synthetic Chainlink BTC/USD price reference built on a 1-second recv_ts_ns grid.",
        "columns": {
            "recv_ts_ns": {
                "dtype": "int64",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "canonical_grid",
                "description": "Canonical 1-second local receive-clock timestamp in nanoseconds.",
            },
            "ts_seconds": {
                "dtype": "int64",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "canonical_grid",
                "description": "UTC second bucket corresponding to recv_ts_ns.",
            },
            "synthetic_raw": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "binance_spot",
                "description": "Leak-free spot-only base proxy: Binance spot mid * (1 + 2.9bps). Binance USD-M and Hyperliquid remain diagnostics only and do not enter the official price.",
            },
            "synthetic_corrected": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "synthetic_raw+chainlink_public_delayed_bias",
                "description": "Official fake live Chainlink BTC/USD value: synthetic_raw plus the applied delayed Chainlink residual bias. The applied bias is fresh when bias_mode=active_window and carried when bias_mode=carried_forward.",
            },
            "rolling_bias": {
                "dtype": "float64",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "Applied residual bias. It equals active_window_bias when the [t-60m, t-30m] window is non-empty, carries the last valid active-window bias when bias_mode=carried_forward, and is 0 only when no bias is available.",
            },
            "active_window_bias": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "Fresh mean residual from delayed Chainlink calibration observations whose event timestamps are at least 30 minutes old and no more than 60 minutes old; NaN when that strict window is empty.",
            },
            "bias_state_age_seconds": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "Age in seconds of the newest Chainlink event currently inside the [t-60m, t-30m] residual window; NaN when the window is empty.",
            },
            "applied_bias_age_seconds": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "Age in seconds of the Chainlink event timestamp backing the applied bias. It continues increasing while the last valid bias is carried forward.",
            },
            "last_bias_observation_age_seconds": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "Age in seconds of the newest delayed Chainlink event that has become causally usable for calibration.",
            },
            "bias_mode": {
                "dtype": "string",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "Applied bias state: active_window for a fresh [t-60m,t-30m] residual mean, carried_forward for the last valid active-window bias after the delayed source goes stale, unavailable before any usable bias exists.",
            },
            "bias_bootstrapped": {
                "dtype": "bool",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "True once at least one delayed Chainlink residual has become causally usable under the 1800s floor.",
            },
            "bias_active": {
                "dtype": "bool",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "Strict freshness flag. True only when the [t-60m, t-30m] delayed residual window is non-empty.",
            },
            "bias_carried_forward": {
                "dtype": "bool",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "True when the strict residual window is empty but the last valid active-window bias is still applied.",
            },
            "bias_state_stale": {
                "dtype": "bool",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "True after calibration has bootstrapped but no delayed Chainlink residual is currently inside [t-60m, t-30m].",
            },
            "bias_neutralized": {
                "dtype": "bool",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "True when calibration has bootstrapped but no active or carried bias is applied. Under the carried-bias policy this should normally stay false after bootstrap.",
            },
            "bias_observation_count": {
                "dtype": "int16",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "chainlink_public_delayed",
                "description": "Number of delayed Chainlink residual observations currently inside [t-60m, t-30m].",
            },
            "source_count": {
                "dtype": "int8",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "binance_spot|binance_usdm|hyperliquid",
                "description": "Number of healthy non-stale primary mids available at the second.",
            },
            "price_valid": {
                "dtype": "bool",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "binance_spot",
                "description": "True when a healthy non-stale Binance spot mid is available for the official spot-only synthetic price.",
            },
            "degraded": {
                "dtype": "bool",
                "nullable": False,
                "leakage_only": False,
                "training_eligible": True,
                "source": "binance_spot|binance_usdm|hyperliquid",
                "description": "Diagnostic flag for official price-valid rows where fewer than all three recorded primary/context mids are available. Non-spot mids do not affect price validity.",
            },
            "binance_spot_mid": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "binance_spot",
                "description": "Healthy Binance spot mid used by the official spot-only synthetic calculation when not stale.",
            },
            "binance_spot_staleness_s": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "binance_spot",
                "description": "Seconds since the latest Binance spot update used for the current second.",
            },
            "binance_usdm_mid": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "binance_usdm",
                "description": "Healthy Binance USD-M mid retained as diagnostic context when not stale and not quarantined; it does not enter the official spot-only synthetic price.",
            },
            "binance_usdm_staleness_s": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "binance_usdm",
                "description": "Seconds since the latest Binance USD-M update used for the current second.",
            },
            "hyperliquid_mid": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "hyperliquid",
                "description": "Healthy Hyperliquid mid retained as diagnostic context when not stale; it does not enter the official spot-only synthetic price.",
            },
            "hyperliquid_staleness_s": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "hyperliquid",
                "description": "Seconds since the latest Hyperliquid update used for the current second.",
            },
            "max_cross_source_spread": {
                "dtype": "float64",
                "nullable": True,
                "leakage_only": False,
                "training_eligible": True,
                "source": "binance_spot|binance_usdm|hyperliquid",
                "description": "Max minus min across available healthy primary mids for the second.",
            },
        },
    }
    schema_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(schema_path)


def _write_daily_parquets(frame: pd.DataFrame, output_dir: Path, *, skip_existing: bool = True) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dates = pd.to_datetime(frame["ts_seconds"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    written_paths: list[Path] = []
    for shard_date in sorted(shard_dates.unique()):
        shard_frame = frame.loc[shard_dates == shard_date].copy()
        shard_path = output_dir / f"{shard_date}.parquet"
        if not (skip_existing and shard_path.exists()):
            _write_parquet(shard_frame, shard_path)
        written_paths.append(shard_path)
    return written_paths


def _resolve_root_relative_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts[:1] in {("docs",), ("config",)}:
        return (root.parent / path) if root.name == "data" else (root / path)
    if path.parts[:1] == ("data",):
        return (root.parent / path) if root.name == "data" else (root / Path(*path.parts[1:]))
    return root / path


def build_live_reference_events(
    *,
    root: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
    date_from: date = DEFAULT_DATE_FROM,
    date_to: date = DEFAULT_DATE_TO,
    max_workers: int = 1,
    skip_existing: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    output_path = _resolve_root_relative_path(root, Path(output_path))
    validation_path = _resolve_root_relative_path(root, Path(validation_path))
    schema_path = _resolve_root_relative_path(root, Path(schema_path))
    policy = load_dataset_policy(policy_path)
    source_names = (*policy.live_reference_primary_inputs, *policy.live_reference_calibration_inputs.keys())
    earliest_raw_source_date = _discover_earliest_raw_source_date(root, source_names)
    default_load_date_from = date_from - timedelta(days=1)
    load_date_from = min(default_load_date_from, earliest_raw_source_date) if earliest_raw_source_date else default_load_date_from
    source_decisions = {
        source_name: _load_source_hours(
            root=root,
            policy=policy,
            policy_path=policy_path,
            source_name=source_name,
            date_from=load_date_from,
            date_to=date_to,
            max_workers=max_workers,
        )
        for source_name in source_names
    }

    quarantine_summary: dict[str, dict[str, int]] = {}
    quarantined_source_hours = 0
    for decisions in source_decisions.values():
        requested_decisions = [
            decision
            for decision in decisions
            if date_from.isoformat() <= decision.date_str <= date_to.isoformat()
        ]
        source_summary, source_excluded = _build_quarantine_summary(requested_decisions)
        quarantined_source_hours += source_excluded
        for source_name, bucket in source_summary.items():
            summary_bucket = quarantine_summary.setdefault(source_name, {})
            for reason, count in bucket.items():
                summary_bucket[reason] = summary_bucket.get(reason, 0) + count

    eligible_updates: dict[str, list[SourceUpdate]] = {}
    eligible_hours: dict[str, set[tuple[str, str]]] = {}
    for source_name in policy.live_reference_primary_inputs:
        decisions = source_decisions[source_name]
        rows = [
            row
            for decision in decisions
            if decision.eligible
            for row in decision.rows
        ]
        rows.sort(key=lambda row: (row.effective_ts_seconds, row.recv_ts_ns))
        eligible_updates[source_name] = rows
        eligible_hours[source_name] = {
            (decision.date_str, decision.hour_str)
            for decision in decisions
            if decision.eligible
        }

    calibration_rows = [
        row
        for decision in source_decisions["chainlink_public_delayed"]
        if decision.eligible
        for row in decision.rows
    ]
    calibration_rows.sort(key=lambda row: (row.available_ts_seconds, row.recv_ts_ns))

    if not any(eligible_updates[source_name] for source_name in policy.live_reference_primary_inputs):
        raise ValueError("No healthy primary live-reference input rows were available for the requested date range.")

    start_ts_seconds = min(
        row.effective_ts_seconds
        for source_name in policy.live_reference_primary_inputs
        for row in eligible_updates[source_name]
    )
    end_ts_seconds = max(
        row.effective_ts_seconds
        for source_name in policy.live_reference_primary_inputs
        for row in eligible_updates[source_name]
    )
    requested_start_ts_seconds = int(datetime(date_from.year, date_from.month, date_from.day, tzinfo=UTC).timestamp())
    requested_end_ts_seconds = int(
        (
            datetime(date_to.year, date_to.month, date_to.day, tzinfo=UTC) + timedelta(days=1)
        ).timestamp()
    ) - 1
    bootstrap_start_ts_seconds = int(
        datetime(load_date_from.year, load_date_from.month, load_date_from.day, tzinfo=UTC).timestamp()
    )
    start_ts_seconds = max(start_ts_seconds, bootstrap_start_ts_seconds)
    end_ts_seconds = min(end_ts_seconds, requested_end_ts_seconds)
    if start_ts_seconds > end_ts_seconds:
        raise ValueError("No eligible live reference seconds fall inside the requested UTC date window.")

    source_indices = {source_name: 0 for source_name in policy.live_reference_primary_inputs}
    current_updates: dict[str, SourceUpdate | None] = {source_name: None for source_name in policy.live_reference_primary_inputs}
    synthetic_raw_history: dict[int, float] = {}

    rows: list[dict[str, Any]] = []
    no_future_violations = 0

    for ts_seconds in range(start_ts_seconds, end_ts_seconds + 1):
        current_hour = _hour_key(ts_seconds)
        available_mids: list[float] = []
        output_row: dict[str, Any] = {
            "recv_ts_ns": int(ts_seconds * SECOND_NS),
            "ts_seconds": ts_seconds,
            "rolling_bias": 0.0,
            "active_window_bias": math.nan,
            "bias_state_age_seconds": math.nan,
            "applied_bias_age_seconds": math.nan,
            "last_bias_observation_age_seconds": math.nan,
            "bias_mode": "unavailable",
            "bias_bootstrapped": False,
            "bias_active": False,
            "bias_carried_forward": False,
            "bias_state_stale": False,
            "bias_neutralized": False,
            "bias_observation_count": 0,
        }

        for source_name in policy.live_reference_primary_inputs:
            updates = eligible_updates[source_name]
            while source_indices[source_name] < len(updates) and updates[source_indices[source_name]].effective_ts_seconds <= ts_seconds:
                current_updates[source_name] = updates[source_indices[source_name]]
                source_indices[source_name] += 1

            update = current_updates[source_name]
            if current_hour not in eligible_hours[source_name]:
                output_row[f"{source_name}_mid"] = math.nan
                output_row[f"{source_name}_staleness_s"] = math.nan
                continue

            if update is None:
                output_row[f"{source_name}_mid"] = math.nan
                output_row[f"{source_name}_staleness_s"] = math.nan
                continue

            if update.recv_ts_ns > ts_seconds * SECOND_NS:
                no_future_violations += 1
                raise AssertionError(f"Future data leakage detected for {source_name} at second {ts_seconds}")

            staleness_s = max(0.0, (ts_seconds * SECOND_NS - update.recv_ts_ns) / SECOND_NS)
            output_row[f"{source_name}_staleness_s"] = float(staleness_s)
            if staleness_s <= STALE_SECONDS:
                output_row[f"{source_name}_mid"] = float(update.mid)
                available_mids.append(float(update.mid))
            else:
                output_row[f"{source_name}_mid"] = math.nan

        source_count = len(available_mids)
        spot_mid = output_row.get("binance_spot_mid")
        synthetic_raw_value = _spot_only_premium_price(
            spot_mid=float(spot_mid) if _is_finite(spot_mid) else None,
        )
        price_valid = bool(_is_finite(synthetic_raw_value))
        degraded = bool(price_valid and source_count < len(policy.live_reference_primary_inputs))
        synthetic_raw = float(synthetic_raw_value) if price_valid and synthetic_raw_value is not None else math.nan
        synthetic_corrected = float(synthetic_raw) if price_valid else math.nan
        max_cross_source_spread = (
            float(max(available_mids) - min(available_mids))
            if source_count >= 2
            else math.nan
        )
        synthetic_raw_history[ts_seconds] = float(synthetic_raw)
        output_row.update(
            {
                "synthetic_raw": float(synthetic_raw),
                "synthetic_corrected": float(synthetic_corrected),
                "source_count": int(source_count),
                "price_valid": bool(price_valid),
                "degraded": bool(degraded),
                "max_cross_source_spread": float(max_cross_source_spread),
            }
        )
        rows.append(output_row)

    if no_future_violations:
        raise AssertionError("Future data leakage detected while building live_reference_events_v1")

    delayed_residuals, bias_anomaly_count = _build_delayed_bias_residuals(
        calibration_rows=calibration_rows,
        synthetic_raw_history=synthetic_raw_history,
    )
    _apply_delayed_residual_bias(rows=rows, residuals=delayed_residuals)
    rows = [
        row
        for row in rows
        if requested_start_ts_seconds <= int(row["ts_seconds"]) <= requested_end_ts_seconds
    ]
    if not rows:
        raise ValueError("No eligible live reference seconds fall inside the requested UTC date window.")

    frame = _coerce_output_frame(pd.DataFrame(rows))
    shard_paths = _write_daily_parquets(frame, output_path, skip_existing=skip_existing)
    schema_contract_path = _write_schema_contract(schema_path)

    daily_rows = [
        {
            "date": shard_path.stem,
            "rows": int(
                len(frame.loc[pd.to_datetime(frame["ts_seconds"], unit="s", utc=True).dt.strftime("%Y-%m-%d") == shard_path.stem])
            ),
            "parquet_path": shard_path.as_posix(),
        }
        for shard_path in shard_paths
    ]

    source_count_ge_2 = float((frame["source_count"] >= 2).mean())
    source_count_eq_1 = float((frame["source_count"] == 1).mean())
    source_count_eq_0 = float((frame["source_count"] == 0).mean())
    staleness_rows = []
    for source_name in policy.live_reference_primary_inputs:
        values = [float(value) for value in frame[f"{source_name}_staleness_s"].dropna().tolist()]
        staleness_rows.append(
            {
                "source": source_name,
                "median_staleness_s": _percentile(values, 0.5),
                "p95_staleness_s": _percentile(values, 0.95),
                "p99_staleness_s": _percentile(values, 0.99),
            }
        )

    bias_values = [float(value) for value in frame["rolling_bias"].tolist()]
    active_bias_values = [
        float(value)
        for value in frame.loc[frame["bias_active"].astype("bool"), "rolling_bias"].tolist()
    ]
    carried_bias_values = [
        float(value)
        for value in frame.loc[frame["bias_carried_forward"].astype("bool"), "rolling_bias"].tolist()
    ]
    bias_age_values = [float(value) for value in frame["bias_state_age_seconds"].dropna().tolist()]
    applied_bias_age_values = [float(value) for value in frame["applied_bias_age_seconds"].dropna().tolist()]
    last_bias_age_values = [
        float(value) for value in frame["last_bias_observation_age_seconds"].dropna().tolist()
    ]
    bias_active_rate = float(frame["bias_active"].astype("bool").mean())
    bias_carried_rate = float(frame["bias_carried_forward"].astype("bool").mean())
    bias_neutralized_rate = float(frame["bias_neutralized"].astype("bool").mean())
    bias_stale_rate = float(frame["bias_state_stale"].astype("bool").mean())
    bias_mode_rows = [
        {
            "bias_mode": str(mode),
            "rows": int(count),
            "fraction": _format_pct(float(count) / len(frame) if len(frame) else 0.0),
        }
        for mode, count in frame["bias_mode"].value_counts(sort=False).items()
    ]
    quarantine_rows = [
        {
            "source": source_name,
            "reason": reason,
            "excluded_source_hours": count,
        }
        for source_name, bucket in sorted(quarantine_summary.items())
        for reason, count in sorted(bucket.items())
    ]
    spot_check_rows = _default_validation_rows(frame)

    validation_lines = [
        "# live_reference_events_v1 Validation",
        "",
        f"- Covered seconds: `{len(frame)}`",
        f"- Date range: `{date_from.isoformat()}` to `{date_to.isoformat()}`",
        f"- Output directory: `{output_path.as_posix()}`",
        "- Official price method: `binance_spot_mid * 1.00029 + applied_bias`.",
        "- `applied_bias` is the fresh `mean(delayed_chainlink_residual[t-60m,t-30m])` when available, otherwise the last valid active-window bias is carried forward.",
        "- Causal rule: delayed Chainlink residuals are usable only after `chainlink_ts + 1800s` and never before local `recv_ts_ns`.",
        "",
        "## Daily Shards",
        "",
        _markdown_table(daily_rows),
        "",
        "## Source Count Distribution",
        "",
        _markdown_table(
            [
                {"metric": "fraction_source_count_ge_2", "value": _format_pct(source_count_ge_2)},
                {"metric": "fraction_source_count_eq_1", "value": _format_pct(source_count_eq_1)},
                {"metric": "fraction_source_count_eq_0", "value": _format_pct(source_count_eq_0)},
            ]
        ),
        "",
        "## Staleness Summary",
        "",
        _markdown_table(staleness_rows),
        "",
        "## Delayed Residual Bias Summary",
        "",
        _markdown_table(
            [
                {"metric": "applied_mean", "value": _format_float(_mean(bias_values))},
                {"metric": "applied_median", "value": _format_float(_percentile(bias_values, 0.5))},
                {"metric": "applied_p5", "value": _format_float(_percentile(bias_values, 0.05))},
                {"metric": "applied_p95", "value": _format_float(_percentile(bias_values, 0.95))},
                {"metric": "active_median", "value": _format_float(_percentile(active_bias_values, 0.5))},
                {"metric": "active_p95", "value": _format_float(_percentile(active_bias_values, 0.95))},
                {"metric": "carried_median", "value": _format_float(_percentile(carried_bias_values, 0.5))},
                {"metric": "carried_p95", "value": _format_float(_percentile(carried_bias_values, 0.95))},
            ]
        ),
        "",
        "## Bias Mode Distribution",
        "",
        _markdown_table(bias_mode_rows),
        "",
        "## Delayed Residual Bias Guardrail Summary",
        "",
        _markdown_table(
            [
                {"metric": "fraction_bias_active", "value": _format_pct(bias_active_rate)},
                {"metric": "fraction_bias_carried_forward", "value": _format_pct(bias_carried_rate)},
                {"metric": "fraction_bias_neutralized", "value": _format_pct(bias_neutralized_rate)},
                {"metric": "fraction_bias_state_stale", "value": _format_pct(bias_stale_rate)},
                {"metric": "max_abs_applied_bias", "value": _format_float(max((abs(value) for value in bias_values), default=math.nan))},
                {"metric": "max_bias_observation_count", "value": int(frame["bias_observation_count"].max()) if len(frame) else 0},
            ]
        ),
        "",
        "## Delayed Residual Bias State Age Summary",
        "",
        _markdown_table(
            [
                {"metric": "median", "value": _format_float(_percentile(bias_age_values, 0.5))},
                {"metric": "p95", "value": _format_float(_percentile(bias_age_values, 0.95))},
                {"metric": "p99", "value": _format_float(_percentile(bias_age_values, 0.99))},
                {"metric": "applied_p95", "value": _format_float(_percentile(applied_bias_age_values, 0.95))},
                {"metric": "applied_p99", "value": _format_float(_percentile(applied_bias_age_values, 0.99))},
                {"metric": "last_observation_p95", "value": _format_float(_percentile(last_bias_age_values, 0.95))},
            ]
        ),
        "",
        f"- bias_anomaly_events: `{bias_anomaly_count}`",
        "",
        "## Quarantine Summary",
        "",
        f"- Excluded source-hours by policy: `{quarantined_source_hours}`",
        "",
        _markdown_table(quarantine_rows),
        "",
        "## Spot Check",
        "",
        _markdown_table(spot_check_rows),
    ]
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text("\n".join(validation_lines).strip() + "\n", encoding="utf-8")

    return {
        "output_path": str(output_path),
        "validation_path": str(validation_path),
        "schema_path": schema_contract_path,
        "row_count": int(len(frame)),
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "daily_parquets": [str(path) for path in shard_paths],
        "fraction_source_count_ge_2": source_count_ge_2,
        "fraction_source_count_eq_1": source_count_eq_1,
        "fraction_source_count_eq_0": source_count_eq_0,
        "bias_anomaly_events": bias_anomaly_count,
        "fraction_bias_active": bias_active_rate,
        "fraction_bias_carried_forward": bias_carried_rate,
        "fraction_bias_neutralized": bias_neutralized_rate,
        "fraction_bias_state_stale": bias_stale_rate,
        "bias_mode_counts": {str(mode): int(count) for mode, count in frame["bias_mode"].value_counts(sort=False).items()},
        "quarantined_source_hours": quarantined_source_hours,
        "quarantine_summary": quarantine_summary,
    }


class LiveReferenceEventsReader:
    def __init__(self, *, root: Path, parquet_path: Path = DEFAULT_OUTPUT_PATH) -> None:
        self.root = Path(root)
        self.parquet_path = _resolve_root_relative_path(self.root, Path(parquet_path))
        frame = self._load_frame(self.parquet_path)
        frame = frame.sort_values("ts_seconds", kind="stable").reset_index(drop=True)
        self._frame = frame
        self._ts_seconds = frame["ts_seconds"].astype("int64").tolist()

    @staticmethod
    def _load_frame(parquet_path: Path) -> pd.DataFrame:
        if parquet_path.is_dir():
            shard_paths = sorted(parquet_path.glob("*.parquet"))
            if not shard_paths:
                raise FileNotFoundError(f"No live_reference_events_v1 shard parquet files found in {parquet_path}")
            frames = [pd.read_parquet(path) for path in shard_paths]
            return pd.concat(frames, axis=0, ignore_index=True)
        return pd.read_parquet(parquet_path)

    def latest_asof(self, *, asof_ts_ns: int) -> dict[str, Any] | None:
        asof_second = int(asof_ts_ns // SECOND_NS)
        index = bisect_right(self._ts_seconds, asof_second) - 1
        if index < 0:
            return None
        row = self._frame.iloc[index]
        recv_ts_ns = int(row["recv_ts_ns"]) if "recv_ts_ns" in row.index else int(row["ts_seconds"]) * SECOND_NS
        price_valid = bool(row["price_valid"])
        synthetic_price = None
        if price_valid and not math.isnan(float(row["synthetic_corrected"])):
            synthetic_price = float(row["synthetic_corrected"])
        return {
            "record_type": "chainlink_live_reference",
            "source": "synthetic_chainlink_v1",
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "chainlink_ts_ns": recv_ts_ns,
            "chainlink_ts_iso": ns_to_iso(recv_ts_ns),
            "btc_usd_price": synthetic_price,
            "price_valid": price_valid,
            "price_unavailable": not price_valid,
            "degraded": bool(row["degraded"]),
            "source_count": int(row["source_count"]),
            "bias_mode": str(row["bias_mode"]) if "bias_mode" in row.index else "unavailable",
            "bias_active": bool(row["bias_active"]) if "bias_active" in row.index else False,
            "bias_carried_forward": bool(row["bias_carried_forward"]) if "bias_carried_forward" in row.index else False,
            "applied_bias_age_seconds": (
                float(row["applied_bias_age_seconds"])
                if "applied_bias_age_seconds" in row.index and not math.isnan(float(row["applied_bias_age_seconds"]))
                else None
            ),
        }
