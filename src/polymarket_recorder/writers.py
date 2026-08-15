from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

from binance_recorder.compression import ZstdJsonlAppender
from binance_recorder.manifests import write_manifest
from binance_recorder.utils import isoformat_utc
from market_recorders.archive import GenericManifestAccumulator, count_jsonl_records
from market_recorders.source_registry import source_registry_metadata

from .utils import SelectedMarket, market_active_window_fields, session_basename


@dataclass(slots=True)
class _StaticWriter:
    path: Path
    kind: str
    static_fields: dict[str, Any]
    flush_interval_seconds: float = 2.0
    manifest_enabled: bool = True
    flush_every_write: bool = False
    frame_flush_every_write: bool = False
    verify_readable_count_on_finalize: bool = False
    _appender: ZstdJsonlAppender = field(init=False)
    _accumulator: GenericManifestAccumulator = field(init=False)
    _flush_interval_ns: int = field(init=False)
    _last_flush_ns: int = field(init=False)
    _last_manifest_write_ns: int = field(init=False)

    def __post_init__(self) -> None:
        self._appender = ZstdJsonlAppender(self.path)
        self._accumulator = GenericManifestAccumulator(
            kind=self.kind,
            created_at=isoformat_utc(),
            static_fields=self.static_fields,
        )
        self._flush_interval_ns = int(self.flush_interval_seconds * 1_000_000_000)
        self._last_flush_ns = time.time_ns()
        self._last_manifest_write_ns = 0

    def write(self, record: dict[str, Any]) -> None:
        self._appender.write_json(record)
        self._accumulator.observe(record)
        if self.flush_every_write:
            self._appender.flush(frame=self.frame_flush_every_write)
            self._last_flush_ns = time.time_ns()
        self._flush_if_needed(force_manifest_checkpoint=self._accumulator.record_count == 1)

    def close(self) -> None:
        try:
            self._appender.close()
        finally:
            if not self._appender.has_written or not self.manifest_enabled:
                return
            self._write_manifest(finalized=True)

    def _flush_if_needed(self, *, force_manifest_checkpoint: bool = False) -> None:
        now_ns = time.time_ns()
        if force_manifest_checkpoint or now_ns - self._last_flush_ns >= self._flush_interval_ns:
            self._appender.flush()
            self._last_flush_ns = now_ns
        if self.manifest_enabled and (
            force_manifest_checkpoint or now_ns - self._last_manifest_write_ns >= 60_000_000_000
        ):
            self._write_manifest(finalized=False)
            self._last_manifest_write_ns = now_ns

    def _write_manifest(self, *, finalized: bool) -> None:
        manifest_path = Path(str(self.path).replace("/raw/", "/manifests/").replace("\\raw\\", "\\manifests\\"))
        manifest_path = manifest_path.with_suffix("").with_suffix(".manifest.json")
        readable_record_count = None
        extra_fields: dict[str, Any] = {}
        if "raw" in self.path.parts:
            raw_index = self.path.parts.index("raw")
            relative = Path(*self.path.parts[raw_index + 1 :])
            extra_fields.update(source_registry_metadata(relative, kind=self.kind))
        if finalized and self.verify_readable_count_on_finalize and self.path.exists():
            readable_record_count = count_jsonl_records(self.path)
        write_manifest(
            manifest_path,
            self._accumulator.to_manifest(
                path=self.path,
                finalized=finalized,
                readable_record_count=readable_record_count,
                extra_fields=extra_fields,
            ),
        )


@dataclass(slots=True)
class PolymarketSessionWriters:
    root: Path
    market: SelectedMarket
    started_at: Any
    kinds: tuple[str, ...] = ("ws", "discovery", "lifecycle")
    token_kinds: tuple[str, ...] = ()
    flush_interval_seconds: float = 2.0
    manifest_enabled: bool = True
    static_fields_extra: dict[str, Any] = field(default_factory=dict)
    _writers: dict[str, _StaticWriter] = field(init=False)
    token_writers: dict[str, dict[str, _StaticWriter]] = field(init=False)

    def __post_init__(self) -> None:
        day_folder = self.root / "raw" / "polymarket" / "btc_updown_5m" / self.started_at.astimezone(UTC).strftime(
            "%Y-%m-%d"
        )
        base = session_basename(self.started_at, self.market)
        static_fields = {
            "source": "polymarket",
            "market_family": "btc_updown_5m",
            "selected_market_slug": self.market.market_slug,
            "selected_market_id": self.market.condition_id or self.market.market_id,
            **market_active_window_fields(self.market, interval_seconds=300),
            **self.static_fields_extra,
        }
        self._writers: dict[str, _StaticWriter] = {
            kind: _StaticWriter(
                day_folder / f"{base}.{kind}.jsonl.zst",
                kind,
                static_fields,
                flush_interval_seconds=self.flush_interval_seconds,
                manifest_enabled=self.manifest_enabled,
                flush_every_write=kind == "lifecycle",
                frame_flush_every_write=kind == "lifecycle",
                verify_readable_count_on_finalize=kind == "lifecycle",
            )
            for kind in self.kinds
        }
        self.token_writers: dict[str, dict[str, _StaticWriter]] = {}
        for asset_id, outcome in self.market.outcome_map.items():
            suffix = outcome.strip().lower().replace(" ", "_")
            self.token_writers[asset_id] = {}
            if "token_ws" in self.token_kinds:
                self.token_writers[asset_id]["token_ws"] = _StaticWriter(
                    day_folder / f"{base}__{suffix}.token_ws.jsonl.zst",
                    "token_ws",
                    {**static_fields, "token_outcome": outcome, "token_asset_id": asset_id},
                    flush_interval_seconds=self.flush_interval_seconds,
                    manifest_enabled=self.manifest_enabled,
                )
            if "token_rest" in self.token_kinds:
                self.token_writers[asset_id]["token_rest"] = _StaticWriter(
                    day_folder / f"{base}__{suffix}.token_rest.jsonl.zst",
                    "token_rest",
                    {**static_fields, "token_outcome": outcome, "token_asset_id": asset_id},
                    flush_interval_seconds=self.flush_interval_seconds,
                    manifest_enabled=self.manifest_enabled,
                )

    def writer(self, kind: str) -> _StaticWriter:
        return self._writers[kind]

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        for writer_group in self.token_writers.values():
            for writer in writer_group.values():
                writer.close()
