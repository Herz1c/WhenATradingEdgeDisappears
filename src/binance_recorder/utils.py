from __future__ import annotations

import signal
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def utc_now_ns() -> int:
    return time.time_ns()


def ns_to_utc_datetime(ts_ns: int) -> datetime:
    seconds = ts_ns / 1_000_000_000
    return datetime.fromtimestamp(seconds, tz=UTC)


def dt_to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1_000_000_000)


def isoformat_utc(value: datetime | None = None) -> str:
    target = value or utc_now()
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    return target.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def ns_to_iso(ts_ns: int) -> str:
    return isoformat_utc(ns_to_utc_datetime(ts_ns))


def floor_to_hour(value: datetime) -> datetime:
    target = value.astimezone(UTC)
    return target.replace(minute=0, second=0, microsecond=0)


def hour_filename(value: datetime) -> str:
    return floor_to_hour(value).strftime("%H")


def date_folder(value: datetime | date) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%Y-%m-%d")
    return value.isoformat()


def parse_utc_date(value: str) -> date:
    return date.fromisoformat(value)


def iter_hours_for_day(day: date) -> Iterable[datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    for hour in range(24):
        yield start + timedelta(hours=hour)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_connection_id(prefix: str = "conn") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def safe_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("Boolean cannot be coerced to int safely")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value.strip())
    raise TypeError(f"Unsupported int-like value: {type(value)!r}")


def get_record_timestamp_ns(record: dict[str, Any]) -> int:
    if "recv_ts_ns" in record:
        return safe_int(record["recv_ts_ns"])
    if "ts_ns" in record:
        return safe_int(record["ts_ns"])
    raise KeyError("Record does not contain recv_ts_ns or ts_ns")


def maybe_import_pandas() -> Any:
    import pandas

    return pandas


def install_signal_handlers(stop_callback: Callable[[int], None]) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda received, _frame, callback=stop_callback: callback(received))
        except ValueError:
            continue
