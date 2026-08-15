from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

from binance_recorder.compression import iter_zstd_jsonl, iter_zstd_jsonl_with_options
from binance_recorder.utils import parse_utc_date
from market_recorders.file_quality import manifest_quality_class
from market_recorders.archive import manifest_path_for_raw_path
from market_recorders.source_registry import infer_kind_from_path, source_tier_for_relative_path


class ActiveFileReadRequiresTailTolerantError(ValueError):
    """Raised when a caller tries to read a non-finalized file without explicit tail tolerance."""


def _record_matches(
    record: dict,
    *,
    record_type: str | None = None,
    topic: str | None = None,
    message_type: str | None = None,
    symbol: str | None = None,
    stream: str | None = None,
    channel: str | None = None,
    endpoint: str | None = None,
    token_outcome: str | None = None,
) -> bool:
    if record_type is not None and str(record.get("record_type")) != record_type:
        return False
    if topic is not None and str(record.get("topic")) != topic:
        return False
    if message_type is not None and str(record.get("message_type")) != message_type:
        return False
    if symbol is not None and str(record.get("symbol")) != symbol:
        return False
    if stream is not None and str(record.get("stream")) != stream:
        return False
    if channel is not None and str(record.get("channel")) != channel:
        return False
    if endpoint is not None and str(record.get("endpoint")) != endpoint:
        return False
    if token_outcome is not None and str(record.get("token_outcome")) != token_outcome:
        return False
    return True


class UnifiedRawReader:
    def __init__(self, *, root: Path) -> None:
        self.root = root

    def _manifest_payload_for_raw_path(self, path: Path) -> dict | None:
        manifest_path = manifest_path_for_raw_path(root=self.root, raw_path=path)
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _manifest_is_finalized(self, path: Path) -> bool:
        manifest_payload = self._manifest_payload_for_raw_path(path)
        return bool(manifest_payload and manifest_payload.get("finalized") is True)

    def file_metadata(self, path: Path) -> dict[str, object]:
        manifest_payload = self._manifest_payload_for_raw_path(path) or {}
        relative = path.relative_to(self.root / "raw")
        kind = infer_kind_from_path(path) or "unknown"
        return {
            "relative_path": str(relative).replace("\\", "/"),
            "kind": kind,
            "finalized": bool(manifest_payload.get("finalized") is True),
            "file_state": manifest_payload.get("file_state"),
            "tail_status": manifest_payload.get("tail_status"),
            "quality_class": manifest_quality_class(manifest_payload),
            "quality_flags": manifest_payload.get("quality_flags", []),
            "source_tier": manifest_payload.get("source_tier")
            or source_tier_for_relative_path(relative, kind=kind),
            "manifest": manifest_payload,
        }

    def iter_file(
        self,
        path: Path,
        *,
        record_type: str | None = None,
        topic: str | None = None,
        message_type: str | None = None,
        symbol: str | None = None,
        stream: str | None = None,
        channel: str | None = None,
        endpoint: str | None = None,
        token_outcome: str | None = None,
        finalized_only: bool = False,
        tail_tolerant: bool = False,
    ) -> Iterator[dict]:
        manifest_payload = self._manifest_payload_for_raw_path(path)
        manifest_is_finalized = bool(manifest_payload and manifest_payload.get("finalized") is True)
        if finalized_only and not manifest_is_finalized:
            return
        if manifest_payload and not manifest_is_finalized and not tail_tolerant:
            raise ActiveFileReadRequiresTailTolerantError(
                f"Refusing to read non-finalized file without tail_tolerant=True: {path}"
            )
        iterator = (
            iter_zstd_jsonl_with_options(
                path,
                allow_truncated_stream=tail_tolerant,
                allow_partial_final_line=tail_tolerant,
            )
            if tail_tolerant
            else iter_zstd_jsonl(path)
        )
        for record in iterator:
            if _record_matches(
                record,
                record_type=record_type,
                topic=topic,
                message_type=message_type,
                symbol=symbol,
                stream=stream,
                channel=channel,
                endpoint=endpoint,
                token_outcome=token_outcome,
            ):
                yield record

    def iter_date(
        self,
        *,
        relative_prefix: Path,
        target_date: date | str,
        kind: str | None = None,
        record_type: str | None = None,
        topic: str | None = None,
        message_type: str | None = None,
        symbol: str | None = None,
        stream: str | None = None,
        channel: str | None = None,
        endpoint: str | None = None,
        token_outcome: str | None = None,
        finalized_only: bool = False,
        tail_tolerant: bool = False,
    ) -> Iterator[dict]:
        day = parse_utc_date(target_date) if isinstance(target_date, str) else target_date
        for file_path in self.list_files_for_date(
            relative_prefix=relative_prefix,
            target_date=day,
            kind=kind,
            finalized_only=finalized_only,
        ):
            yield from self.iter_file(
                file_path,
                record_type=record_type,
                topic=topic,
                message_type=message_type,
                symbol=symbol,
                stream=stream,
                channel=channel,
                endpoint=endpoint,
                token_outcome=token_outcome,
                finalized_only=finalized_only,
                tail_tolerant=tail_tolerant,
            )

    def list_files_for_date(
        self,
        *,
        relative_prefix: Path,
        target_date: date | str,
        kind: str | None = None,
        finalized_only: bool = False,
    ) -> list[Path]:
        day = parse_utc_date(target_date) if isinstance(target_date, str) else target_date
        folder = self.root / "raw" / relative_prefix / day.isoformat()
        if not folder.exists():
            return []
        files = sorted(folder.glob("*.jsonl.zst"))
        if finalized_only:
            files = [path for path in files if self._manifest_is_finalized(path)]
        if kind is None:
            return files
        suffix = f".{kind}.jsonl.zst"
        return [path for path in files if path.name.endswith(suffix)]

    def list_file_metadata_for_date(
        self,
        *,
        relative_prefix: Path,
        target_date: date | str,
        kind: str | None = None,
        finalized_only: bool = False,
    ) -> list[dict[str, object]]:
        return [
            self.file_metadata(path)
            for path in self.list_files_for_date(
                relative_prefix=relative_prefix,
                target_date=target_date,
                kind=kind,
                finalized_only=finalized_only,
            )
        ]

    @staticmethod
    def date_from_record(record: dict) -> date:
        if "recv_ts_iso" in record:
            return datetime.fromisoformat(record["recv_ts_iso"].replace("Z", "+00:00")).astimezone(UTC).date()
        if "recv_ts_ns" in record:
            return datetime.fromtimestamp(record["recv_ts_ns"] / 1_000_000_000, tz=UTC).date()
        raise KeyError("Record does not include recv_ts_iso or recv_ts_ns")
