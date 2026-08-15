from __future__ import annotations

import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from binance_recorder.utils import ns_to_iso

LOCAL_RECV_TS_SOURCE = "python_time_time_ns_epoch_wall_clock"

_ISO8601_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[T ](?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?(?P<tz>Z|[+-]\d{2}:\d{2})?$"
)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<!^)(?=[A-Z])")


def recv_timestamp_source() -> str:
    return LOCAL_RECV_TS_SOURCE


def utc_now_ns() -> int:
    return time.time_ns()


def infer_epoch_unit(value: int) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000_000_000_000_000:
        return "nanosecond"
    if magnitude >= 1_000_000_000_000_000:
        return "microsecond"
    if magnitude >= 1_000_000_000_000:
        return "millisecond"
    return "second"


def normalize_epoch_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def resolve_epoch_unit(value: int, *, unit_hint: str | None = None) -> str:
    if abs(value) >= 100_000_000_000:
        return infer_epoch_unit(value)
    if unit_hint:
        return unit_hint
    return infer_epoch_unit(value)


def epoch_value_to_ns(value: int, *, unit: str) -> int:
    if unit == "nanosecond":
        return value
    if unit == "microsecond":
        return value * 1_000
    if unit == "millisecond":
        return value * 1_000_000
    if unit == "second":
        return value * 1_000_000_000
    raise ValueError(f"Unsupported epoch unit: {unit}")


def epoch_unit_suffix(unit: str) -> str:
    if unit == "nanosecond":
        return "ns"
    if unit == "microsecond":
        return "us"
    if unit == "millisecond":
        return "ms"
    if unit == "second":
        return "s"
    raise ValueError(f"Unsupported epoch unit: {unit}")


def build_epoch_timestamp_fields(
    field_name: str,
    value: Any,
    *,
    unit_hint: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_epoch_value(value)
    if normalized is None:
        return {}
    unit = resolve_epoch_unit(normalized, unit_hint=unit_hint)
    suffix = epoch_unit_suffix(unit)
    ts_ns = epoch_value_to_ns(normalized, unit=unit)
    return {
        f"{field_name}_{suffix}": normalized,
        f"{field_name}_iso": ns_to_iso(ts_ns),
    }


def parse_iso8601_to_ns(value: str | None) -> int | None:
    if value is None:
        return None
    match = _ISO8601_RE.match(value.strip())
    if not match:
        return None
    fraction = match.group("fraction") or ""
    fraction_ns = int(fraction.ljust(9, "0")) if fraction else 0
    naive = datetime.fromisoformat(f"{match.group('date')}T{match.group('time')}")
    tz_text = match.group("tz")
    if tz_text in (None, "Z"):
        aware = naive.replace(tzinfo=UTC)
    else:
        sign = 1 if tz_text.startswith("+") else -1
        hours, minutes = tz_text[1:].split(":")
        offset = timedelta(hours=int(hours), minutes=int(minutes))
        aware = naive.replace(tzinfo=UTC) - (sign * offset)
    return int(aware.timestamp()) * 1_000_000_000 + fraction_ns


def build_iso_timestamp_fields(field_name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    ts_ns = parse_iso8601_to_ns(value)
    if ts_ns is None:
        return {}
    return {
        f"{field_name}_iso": ns_to_iso(ts_ns),
        f"{field_name}_ns": ts_ns,
    }


def build_timestamp_fields(
    field_name: str,
    value: Any,
    *,
    unit_hint: str | None = None,
) -> dict[str, Any]:
    normalized_epoch = normalize_epoch_value(value)
    if normalized_epoch is not None:
        return build_epoch_timestamp_fields(field_name, normalized_epoch, unit_hint=unit_hint)
    return build_iso_timestamp_fields(field_name, value)


def normalize_timestamp_field_name(value: str) -> str:
    snake = _CAMEL_BOUNDARY_RE.sub("_", value).replace("-", "_").replace(".", "_")
    snake = re.sub(r"[^a-zA-Z0-9_]+", "_", snake)
    snake = re.sub(r"_+", "_", snake).strip("_")
    return snake.lower()


def build_recursive_epoch_timestamp_fields(
    payload: Any,
    *,
    candidate_field_names: tuple[str, ...],
    min_epoch_value: int = 1_000_000_000,
) -> dict[str, Any]:
    candidates = {normalize_timestamp_field_name(name) for name in candidate_field_names}
    timestamps: dict[str, Any] = {}

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for raw_key, nested in value.items():
                normalized_key = normalize_timestamp_field_name(str(raw_key))
                next_path = (*path, normalized_key)
                normalized_epoch = normalize_epoch_value(nested)
                if normalized_key in candidates and normalized_epoch is not None and abs(normalized_epoch) >= min_epoch_value:
                    field_name = "_".join(next_path)
                    timestamps.update(build_epoch_timestamp_fields(field_name, normalized_epoch))
                walk(nested, next_path)
            return
        if isinstance(value, list):
            for item in value:
                walk(item, path)

    walk(payload, ())
    return timestamps


def build_capture_timing(*, request_start_ns: int, response_end_ns: int) -> dict[str, Any]:
    lower = min(request_start_ns, response_end_ns)
    upper = max(request_start_ns, response_end_ns)
    midpoint = lower + ((upper - lower) // 2)
    latency_ns = upper - lower
    return {
        "request_start_ts_ns": lower,
        "request_start_ts_iso": ns_to_iso(lower),
        "response_end_ts_ns": upper,
        "response_end_ts_iso": ns_to_iso(upper),
        "response_midpoint_ts_ns": midpoint,
        "response_midpoint_ts_iso": ns_to_iso(midpoint),
        "response_latency_ns": latency_ns,
        "response_latency_ms": round(latency_ns / 1_000_000, 3),
    }


def build_clock_probe(
    field_name: str,
    value: Any,
    *,
    request_start_ns: int,
    response_end_ns: int,
    unit_hint: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_epoch_value(value)
    if normalized is None:
        return {}
    unit = resolve_epoch_unit(normalized, unit_hint=unit_hint)
    server_ts_ns = epoch_value_to_ns(normalized, unit=unit)
    midpoint = min(request_start_ns, response_end_ns) + (abs(response_end_ns - request_start_ns) // 2)
    timestamp_fields = build_epoch_timestamp_fields(field_name, normalized, unit_hint=unit)
    return {
        "timestamp_field": field_name,
        "timestamp_unit": unit,
        "timestamp_epoch_ns": server_ts_ns,
        "timestamp_iso": timestamp_fields.get(f"{field_name}_iso"),
        "request_start_minus_server_ms": round((request_start_ns - server_ts_ns) / 1_000_000, 3),
        "response_end_minus_server_ms": round((response_end_ns - server_ts_ns) / 1_000_000, 3),
        "midpoint_minus_server_ms": round((midpoint - server_ts_ns) / 1_000_000, 3),
    }


def iso8601_now_plus(*, seconds: int = 0) -> str:
    return (datetime.now(tz=UTC) + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
