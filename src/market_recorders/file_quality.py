from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

ACTIVE_TAIL_TOLERANT = "active_tail_tolerant"
FINALIZED_CLEAN = "finalized_clean"
RECOVERED = "recovered"
CONTINUITY_COMPROMISED = "continuity_compromised"
STALE_CONTEXT = "stale_context"

_QUALITY_PRIORITY = {
    FINALIZED_CLEAN: 0,
    ACTIVE_TAIL_TOLERANT: 1,
    RECOVERED: 2,
    STALE_CONTEXT: 3,
    CONTINUITY_COMPROMISED: 4,
}


def default_quality_class(*, file_state: str) -> str:
    if file_state == "active":
        return ACTIVE_TAIL_TOLERANT
    if file_state == "recovered":
        return RECOVERED
    return FINALIZED_CLEAN


def manifest_quality_class(manifest_payload: dict[str, Any] | None) -> str | None:
    if not manifest_payload:
        return None
    quality_class = manifest_payload.get("quality_class")
    if isinstance(quality_class, str) and quality_class:
        return quality_class
    file_state = manifest_payload.get("file_state")
    if isinstance(file_state, str) and file_state:
        return default_quality_class(file_state=file_state)
    return None


def _hour_key(hour_start: datetime) -> str:
    return hour_start.astimezone(UTC).replace(minute=0, second=0, microsecond=0).isoformat()


def _utc_hour_from_ts_ns(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC).replace(minute=0, second=0, microsecond=0)


@dataclass(slots=True)
class HourlyQualityTracker:
    _quality_by_hour: dict[str, str] = field(default_factory=dict)
    _flags_by_hour: dict[str, set[str]] = field(default_factory=dict)
    _details_by_hour: dict[str, dict[str, Any]] = field(default_factory=dict)

    def mark(
        self,
        *,
        ts_ns: int | None = None,
        hour_start: datetime | None = None,
        quality_class: str | None = None,
        flags: tuple[str, ...] = (),
        details: dict[str, Any] | None = None,
    ) -> None:
        if hour_start is None:
            if ts_ns is None:
                raise ValueError("Either ts_ns or hour_start is required")
            hour_start = _utc_hour_from_ts_ns(ts_ns)
        key = _hour_key(hour_start)
        if quality_class is not None:
            current = self._quality_by_hour.get(key)
            if current is None or _QUALITY_PRIORITY.get(quality_class, 0) >= _QUALITY_PRIORITY.get(current, 0):
                self._quality_by_hour[key] = quality_class
        if flags:
            self._flags_by_hour.setdefault(key, set()).update(flag for flag in flags if flag)
        if details:
            self._details_by_hour.setdefault(key, {}).update(details)

    def manifest_extra_fields(self, hour_start: datetime, finalized: bool) -> dict[str, Any]:
        del finalized
        key = _hour_key(hour_start)
        payload: dict[str, Any] = {}
        quality_class = self._quality_by_hour.get(key)
        if quality_class is not None:
            payload["quality_class"] = quality_class
        flags = sorted(self._flags_by_hour.get(key, ()))
        if flags:
            payload["quality_flags"] = flags
        details = self._details_by_hour.get(key)
        if details:
            payload["quality_details"] = dict(details)
        return payload
