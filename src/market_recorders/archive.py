from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable

from binance_recorder.compression import (
    ZstdJsonlAppender,
    iter_zstd_jsonl,
    iter_zstd_jsonl_with_options,
    quarantine_corrupt_archive,
)
from binance_recorder.manifests import write_manifest
from binance_recorder.utils import (
    date_folder,
    ensure_directory,
    get_record_timestamp_ns,
    isoformat_utc,
    ns_to_utc_datetime,
)
from market_recorders.file_quality import default_quality_class
from market_recorders.source_registry import source_registry_metadata

DEFAULT_MANIFEST_COUNTER_FIELDS = ("record_type", "stream", "topic", "endpoint", "poller")
ACTIVE_FILE_STATE = "active"
FINALIZED_FILE_STATE = "finalized"
RECOVERED_FILE_STATE = "recovered"
RECOVERY_STATIC_FIELDS = (
    "source",
    "venue",
    "symbol",
    "product_id",
    "market_family",
    "selected_market_slug",
    "selected_market_id",
    "selected_condition_id",
    "service",
    "backend",
    "component",
    "market_type",
    "topic",
    "subscribed_symbol",
    "recorder_service",
    "runtime_session_id",
    "runtime_started_at_ns",
    "runtime_started_at_iso",
    "workspace_git_sha",
    "workspace_version",
    "config_fingerprint",
    "config_snapshot_redacted",
    "config_snapshot",
)


@dataclass(slots=True, frozen=True)
class HourlyArchivePathBuilder:
    root: Path
    namespace: tuple[str, ...]

    def raw_day_dir(self, day: date | datetime) -> Path:
        return ensure_directory(self.root / "raw" / Path(*self.namespace) / date_folder(day))

    def manifest_day_dir(self, day: date | datetime) -> Path:
        return ensure_directory(self.root / "manifests" / Path(*self.namespace) / date_folder(day))

    def raw_file_path(self, ts: datetime, kind: str) -> Path:
        filename = f"{ts.astimezone(UTC).strftime('%H')}.{kind}.jsonl.zst"
        return self.raw_day_dir(ts) / filename

    def manifest_file_path(self, ts: datetime, kind: str) -> Path:
        filename = f"{ts.astimezone(UTC).strftime('%H')}.{kind}.manifest.json"
        return self.manifest_day_dir(ts) / filename

    def daily_manifest_path(self, day: date | datetime) -> Path:
        target_day = day.date() if isinstance(day, datetime) else day
        filename = f"{date_folder(target_day)}.manifest.json"
        return self.manifest_day_dir(target_day) / filename


@dataclass(slots=True)
class GenericManifestAccumulator:
    kind: str
    created_at: str
    static_fields: dict[str, Any] = field(default_factory=dict)
    counter_fields: tuple[str, ...] = DEFAULT_MANIFEST_COUNTER_FIELDS
    record_count: int = 0
    first_recv_ts_ns: int | None = None
    last_recv_ts_ns: int | None = None
    field_counts: dict[str, Counter[str]] = field(default_factory=dict)

    def observe(self, record: dict[str, Any]) -> None:
        self.record_count += 1
        ts_ns = get_record_timestamp_ns(record)
        if self.first_recv_ts_ns is None or ts_ns < self.first_recv_ts_ns:
            self.first_recv_ts_ns = ts_ns
        if self.last_recv_ts_ns is None or ts_ns > self.last_recv_ts_ns:
            self.last_recv_ts_ns = ts_ns
        for field_name in self.counter_fields:
            value = record.get(field_name)
            if value is None:
                continue
            if field_name not in self.field_counts:
                self.field_counts[field_name] = Counter()
            self.field_counts[field_name][str(value)] += 1

    def to_manifest(
        self,
        *,
        path: Path,
        finalized: bool,
        recovered_from_raw: bool = False,
        manifest_written_at: str | None = None,
        readable_record_count: int | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        written_at = manifest_written_at or isoformat_utc()
        file_state = manifest_file_state(finalized=finalized, recovered_from_raw=recovered_from_raw)
        record_count = readable_record_count if readable_record_count is not None else self.record_count
        payload: dict[str, Any] = {
            "file_path": str(path).replace("\\", "/"),
            "file_size": path.stat().st_size if path.exists() else 0,
            "compression": "zstd",
            "record_kind": self.kind,
            "record_count": record_count,
            "first_recv_ts_ns": self.first_recv_ts_ns,
            "last_recv_ts_ns": self.last_recv_ts_ns,
            "first_recv_ts_iso": isoformat_utc(ns_to_utc_datetime(self.first_recv_ts_ns))
            if self.first_recv_ts_ns is not None
            else None,
            "last_recv_ts_iso": isoformat_utc(ns_to_utc_datetime(self.last_recv_ts_ns))
            if self.last_recv_ts_ns is not None
            else None,
            "created_at": self.created_at,
            "manifest_written_at": written_at,
            "finalized": finalized,
            "recovered_from_raw": recovered_from_raw,
            "file_state": file_state,
            "tail_status": manifest_tail_status(file_state),
            "quality_class": default_quality_class(file_state=file_state),
            "manifest_record_count_source": "readable_rows" if readable_record_count is not None else "accumulator",
            "field_counts": {name: dict(counter) for name, counter in self.field_counts.items()},
        }
        if readable_record_count is not None:
            payload["readable_record_count"] = readable_record_count
            if readable_record_count != self.record_count:
                payload["observed_record_count"] = self.record_count
        payload.update(self.static_fields)
        if extra_fields:
            payload.update(extra_fields)
        return payload


@dataclass(slots=True)
class _WriterState:
    hour_start: datetime
    path: Path
    appender: ZstdJsonlAppender
    accumulator: GenericManifestAccumulator
    last_flush_ns: int
    last_manifest_write_ns: int


class HourlyRotatingJsonlWriter:
    def __init__(
        self,
        *,
        root: Path,
        namespace: tuple[str, ...],
        kind: str,
        flush_interval_seconds: float = 2.0,
        manifest_static_fields: dict[str, Any] | None = None,
        manifest_counter_fields: tuple[str, ...] | None = None,
        manifest_enabled: bool = True,
        manifest_checkpoint_interval_seconds: float = 60.0,
        manifest_extra_fields_provider: Callable[[datetime, bool], dict[str, Any]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.path_builder = HourlyArchivePathBuilder(root=root, namespace=namespace)
        self.kind = kind
        self.flush_interval_ns = int(flush_interval_seconds * 1_000_000_000)
        self.manifest_static_fields = manifest_static_fields or {}
        self.manifest_counter_fields = manifest_counter_fields
        self.manifest_enabled = manifest_enabled
        self.manifest_checkpoint_interval_ns = int(manifest_checkpoint_interval_seconds * 1_000_000_000)
        self.manifest_extra_fields_provider = manifest_extra_fields_provider
        self.logger = logger or logging.getLogger(__name__)
        self._state: _WriterState | None = None

    def write(self, record: dict[str, Any]) -> None:
        ts = ns_to_utc_datetime(get_record_timestamp_ns(record)).replace(minute=0, second=0, microsecond=0)
        state = self._ensure_writer(ts)
        state.appender.write_json(record)
        state.accumulator.observe(record)
        self._flush_if_needed(state, force_manifest_checkpoint=state.accumulator.record_count == 1)

    def close(self) -> None:
        if self._state is None:
            return
        self._finalize_current(self._state)
        self._state = None

    def advance_time(self, *, now_ns: int | None = None) -> bool:
        if self._state is None:
            return False
        target_hour = ns_to_utc_datetime(now_ns or time.time_ns()).replace(minute=0, second=0, microsecond=0)
        if target_hour <= self._state.hour_start:
            return False
        self._finalize_current(self._state)
        self._state = None
        return True

    def _ensure_writer(self, ts: datetime) -> _WriterState:
        if self._state is None:
            self._state = self._open_state(ts)
            return self._state
        if self._state.hour_start != ts:
            self._finalize_current(self._state)
            self._state = self._open_state(ts)
        return self._state

    def _open_state(self, hour_start: datetime) -> _WriterState:
        raw_path = self.path_builder.raw_file_path(hour_start, self.kind)
        appender = ZstdJsonlAppender(raw_path)
        created_at = isoformat_utc()
        manifest_path = self.path_builder.manifest_file_path(hour_start, self.kind)
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest_payload.get("created_at"), str):
                    created_at = manifest_payload["created_at"]
            except Exception:
                self.logger.warning(
                    "Failed to read existing manifest while reopening hourly writer",
                    extra={"manifest_path": str(manifest_path)},
                )
        accumulator = GenericManifestAccumulator(
            kind=self.kind,
            created_at=created_at,
            static_fields=self.manifest_static_fields,
            counter_fields=self.manifest_counter_fields or DEFAULT_MANIFEST_COUNTER_FIELDS,
        )
        if raw_path.exists() and raw_path.stat().st_size > 0:
            try:
                for record in iter_zstd_jsonl_with_options(
                    raw_path,
                    allow_truncated_stream=True,
                    allow_partial_final_line=True,
                ):
                    accumulator.observe(record)
            except Exception as exc:
                self.logger.warning(
                    "Failed to resume hourly writer state from existing raw file",
                    extra={"raw_path": str(raw_path), "kind": self.kind, "error": repr(exc)},
                )
        return _WriterState(
            hour_start=hour_start,
            path=raw_path,
            appender=appender,
            accumulator=accumulator,
            last_flush_ns=time.time_ns(),
            last_manifest_write_ns=0,
        )

    def _flush_if_needed(self, state: _WriterState, *, force_manifest_checkpoint: bool = False) -> None:
        now_ns = time.time_ns()
        if force_manifest_checkpoint or now_ns - state.last_flush_ns >= self.flush_interval_ns:
            state.appender.flush()
            state.last_flush_ns = now_ns
        if force_manifest_checkpoint or (
            self.manifest_enabled and now_ns - state.last_manifest_write_ns >= self.manifest_checkpoint_interval_ns
        ):
            self._write_manifest(state, finalized=False)
            state.last_manifest_write_ns = now_ns

    def _finalize_current(self, state: _WriterState) -> None:
        try:
            state.appender.close()
        finally:
            if not self.manifest_enabled or not state.appender.has_written:
                return
            self._write_manifest(state, finalized=True)

    def _write_manifest(self, state: _WriterState, *, finalized: bool) -> None:
        relative = state.path.relative_to(self.path_builder.root / "raw")
        extra_fields = source_registry_metadata(relative, kind=self.kind)
        if self.manifest_extra_fields_provider is not None:
            extra_fields.update(self.manifest_extra_fields_provider(state.hour_start, finalized))
        readable_record_count = None
        if finalized and state.path.exists():
            readable_record_count = count_jsonl_records(state.path)
        manifest = state.accumulator.to_manifest(
            path=state.path,
            finalized=finalized,
            readable_record_count=readable_record_count,
            extra_fields=extra_fields,
        )
        manifest_path = self.path_builder.manifest_file_path(state.hour_start, self.kind)
        write_manifest(manifest_path, manifest)


def rebuild_hour_manifest(
    *,
    root: Path,
    namespace: tuple[str, ...],
    kind: str,
    target_hour: datetime,
    static_fields: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    logger = logger or logging.getLogger(__name__)
    path_builder = HourlyArchivePathBuilder(root=root, namespace=namespace)
    raw_path = path_builder.raw_file_path(target_hour, kind)
    manifest_path = path_builder.manifest_file_path(target_hour, kind)
    accumulator = GenericManifestAccumulator(
        kind=kind,
        created_at=isoformat_utc(),
        static_fields=static_fields or {},
    )
    try:
        for record in iter_zstd_jsonl(raw_path):
            accumulator.observe(record)
    except Exception as exc:
        quarantine_corrupt_archive(raw_path, logger=logger, reason=exc)
        raise
    readable_record_count = count_jsonl_records(raw_path)
    write_manifest(
        manifest_path,
        accumulator.to_manifest(
            path=raw_path,
            finalized=False,
            recovered_from_raw=True,
            readable_record_count=readable_record_count,
            extra_fields=source_registry_metadata(raw_path.relative_to(root / "raw"), kind=kind),
        ),
    )
    return manifest_path


def manifest_path_for_raw_path(*, root: Path, raw_path: Path) -> Path:
    relative = raw_path.relative_to(root / "raw")
    return root / "manifests" / relative.parent / relative.name.replace(".jsonl.zst", ".manifest.json")


def infer_manifest_static_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        field_name: record[field_name]
        for field_name in RECOVERY_STATIC_FIELDS
        if record.get(field_name) is not None
    }


def rebuild_manifest_for_raw_path(*, root: Path, raw_path: Path, logger: logging.Logger | None = None) -> Path:
    logger = logger or logging.getLogger(__name__)
    accumulator: GenericManifestAccumulator | None = None
    static_fields: dict[str, Any] = {}
    kind = raw_path.name.removesuffix(".jsonl.zst").rsplit(".", 1)[-1]
    try:
        for record in iter_zstd_jsonl(raw_path):
            if accumulator is None:
                static_fields = infer_manifest_static_fields(record)
                accumulator = GenericManifestAccumulator(
                    kind=kind,
                    created_at=isoformat_utc(),
                    static_fields=static_fields,
                )
            accumulator.observe(record)
    except Exception as exc:
        quarantine_corrupt_archive(raw_path, logger=logger, reason=exc)
        raise
    if accumulator is None:
        raise ValueError(f"Cannot rebuild manifest for empty raw file: {raw_path}")
    manifest_path = manifest_path_for_raw_path(root=root, raw_path=raw_path)
    readable_record_count = count_jsonl_records(raw_path)
    write_manifest(
        manifest_path,
        accumulator.to_manifest(
            path=raw_path,
            finalized=False,
            recovered_from_raw=True,
            readable_record_count=readable_record_count,
            extra_fields=source_registry_metadata(raw_path.relative_to(root / "raw"), kind=kind),
        ),
    )
    return manifest_path


def rebuild_missing_manifests(*, root: Path, logger: logging.Logger | None = None) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    raw_root = root / "raw"
    report = {
        "rebuilt_manifest_files": [],
        "skipped_empty_raw_files": [],
        "corrupt_or_unreadable_files": [],
    }
    for raw_path in sorted(raw_root.rglob("*.jsonl.zst")):
        manifest_path = manifest_path_for_raw_path(root=root, raw_path=raw_path)
        if manifest_path.exists():
            continue
        if raw_path.stat().st_size == 0:
            report["skipped_empty_raw_files"].append(str(raw_path.relative_to(raw_root)).replace("\\", "/"))
            continue
        try:
            rebuilt = rebuild_manifest_for_raw_path(root=root, raw_path=raw_path, logger=logger)
        except Exception as exc:
            report["corrupt_or_unreadable_files"].append(
                {
                    "file": str(raw_path.relative_to(raw_root)).replace("\\", "/"),
                    "error": repr(exc),
                }
            )
            continue
        report["rebuilt_manifest_files"].append(str(rebuilt.relative_to(root / "manifests")).replace("\\", "/"))
    return report


def manifest_file_state(*, finalized: bool, recovered_from_raw: bool) -> str:
    if finalized:
        return FINALIZED_FILE_STATE
    if recovered_from_raw:
        return RECOVERED_FILE_STATE
    return ACTIVE_FILE_STATE


def manifest_tail_status(file_state: str) -> str:
    if file_state == FINALIZED_FILE_STATE:
        return "tail_clean"
    if file_state == RECOVERED_FILE_STATE:
        return "recovered_requires_review"
    return "tail_may_be_partial"


def count_jsonl_records(
    path: Path,
    *,
    allow_truncated_stream: bool = False,
    allow_partial_final_line: bool = False,
) -> int:
    return sum(
        1
        for _ in iter_zstd_jsonl_with_options(
            path,
            allow_truncated_stream=allow_truncated_stream,
            allow_partial_final_line=allow_partial_final_line,
        )
    )
