from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import polars as pl

from market_recorders.file_quality import FINALIZED_CLEAN
from market_recorders.unified_reader import UnifiedRawReader

POLYMARKET_WS_EVENTS_V1_SCHEMA = {
    "recv_ts_ns": pl.Int64,
    "recv_ts_iso": pl.Utf8,
    "market_slug": pl.Utf8,
    "event_type": pl.Utf8,
    "market_scope": pl.Utf8,
    "token_modeling_safe": pl.Boolean,
    "matched_asset_ids": pl.List(pl.Utf8),
    "matched_outcomes": pl.List(pl.Utf8),
    "payload_summary": pl.Utf8,
}

PARQUET_CHUNK_ROWS = 25_000

_EVENT_TYPE_MAP = {
    "market": "book_update",
    "book": "book_update",
    "orderbook": "book_update",
    "price_change": "price_change",
    "last_trade_price": "trade",
    "trade": "trade",
    "tick_size_change": "tick_update",
    "tick_size_update": "tick_update",
    "open": "market_open",
    "closed": "market_close",
    "market_open": "market_open",
    "market_close": "market_close",
    "system": "system",
    "subscribed": "system",
}


def _date_range(date_from: date, date_to: date) -> list[date]:
    return [date_from + timedelta(days=i) for i in range((date_to - date_from).days + 1)]


def normalize_event_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in _EVENT_TYPE_MAP:
        return _EVENT_TYPE_MAP[raw]
    if "price" in raw and "change" in raw:
        return "price_change"
    if "trade" in raw:
        return "trade"
    if "tick" in raw:
        return "tick_update"
    if "open" in raw:
        return "market_open"
    if "close" in raw:
        return "market_close"
    if not raw:
        return "other"
    return "other"


def matched_asset_ids(record: dict[str, Any]) -> list[str]:
    values = record.get("matched_asset_ids") or record.get("selected_asset_ids") or record.get("asset_ids")
    if values is None and record.get("asset_id") is not None:
        values = [record.get("asset_id")]
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if value is not None]


def matched_outcomes(record: dict[str, Any]) -> list[str]:
    values = record.get("matched_outcomes") or record.get("selected_outcomes") or record.get("outcomes")
    if values is None and record.get("token_outcome") is not None:
        values = [record.get("token_outcome")]
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if value is not None]


def token_modeling_safe(record: dict[str, Any]) -> bool:
    event_type = normalize_event_type(record.get("event_type") or record.get("record_type"))
    assets = matched_asset_ids(record)
    if event_type not in {"price_change", "book_update", "trade", "tick_update"}:
        return False
    return len(assets) == 1 or record.get("token_asset_id") is not None or record.get("asset_id") is not None


def market_scope(record: dict[str, Any]) -> str:
    if token_modeling_safe(record):
        return "token-level"
    if record.get("market_slug") or record.get("selected_market_slug"):
        return "market-level"
    return "system"


def payload_summary(record: dict[str, Any]) -> str:
    keys = [
        "event_type",
        "record_type",
        "asset_id",
        "token_asset_id",
        "side",
        "price",
        "size",
        "best_bid",
        "best_ask",
        "tick_size",
    ]
    summary = {key: record.get(key) for key in keys if key in record}
    return json.dumps(summary, sort_keys=True, separators=(",", ":"))


def normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    recv_ts_ns = record.get("recv_ts_ns")
    if recv_ts_ns is None:
        return None
    return {
        "recv_ts_ns": int(recv_ts_ns),
        "recv_ts_iso": str(record.get("recv_ts_iso") or ""),
        "market_slug": str(record.get("market_slug") or record.get("selected_market_slug") or ""),
        "event_type": normalize_event_type(record.get("event_type") or record.get("record_type")),
        "market_scope": market_scope(record),
        "token_modeling_safe": token_modeling_safe(record),
        "matched_asset_ids": matched_asset_ids(record),
        "matched_outcomes": matched_outcomes(record),
        "payload_summary": payload_summary(record),
    }


def _quality_allowed(reader: UnifiedRawReader, path: Path) -> bool:
    metadata = reader.file_metadata(path)
    return metadata.get("finalized") is True and metadata.get("quality_class") == FINALIZED_CLEAN


def _write_rows_streaming(
    rows: Iterable[dict[str, Any]],
    *,
    output_path: Path,
    schema: dict[str, pl.DataType],
    chunk_rows: int = PARQUET_CHUNK_ROWS,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    total_rows = 0

    def flush() -> None:
        nonlocal writer, total_rows, buffer
        frame = pl.DataFrame(buffer, schema=schema, orient="row")
        table = frame.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema)
        writer.write_table(table)
        total_rows += frame.height
        buffer = []

    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) >= chunk_rows:
                flush()
        if buffer or writer is None:
            flush()
    except Exception:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    if writer is not None:
        writer.close()
    tmp_path.replace(output_path)
    return total_rows


def _iter_polymarket_ws_rows(reader: UnifiedRawReader, target_day: date) -> Iterable[dict[str, Any]]:
    prefix = Path("polymarket") / "btc_updown_5m"
    files = reader.list_files_for_date(
        relative_prefix=prefix,
        target_date=target_day,
        kind="ws",
        finalized_only=True,
    )
    for path in files:
        if not _quality_allowed(reader, path):
            continue
        for record in reader.iter_file(path, finalized_only=True):
            normalized = normalize_record(record)
            if normalized is not None:
                yield normalized


def build_polymarket_ws_events(
    *,
    root: Path = Path("data"),
    date_from: date,
    date_to: date,
    output_dir: Path = Path("data/canonical/polymarket_ws_events_v1"),
    quality_registry_path: Path | None = None,
) -> dict[str, Any]:
    del quality_registry_path
    reader = UnifiedRawReader(root=root)
    daily: list[dict[str, Any]] = []
    for target_day in _date_range(date_from, date_to):
        output_path = output_dir / f"{target_day.isoformat()}.parquet"
        rows_written = _write_rows_streaming(
            _iter_polymarket_ws_rows(reader, target_day),
            output_path=output_path,
            schema=POLYMARKET_WS_EVENTS_V1_SCHEMA,
        )
        daily.append({"date": target_day.isoformat(), "rows": rows_written, "path": str(output_path)})
    return {"dataset": "polymarket_ws_events_v1", "daily": daily}


def status_polymarket_ws_events(
    *,
    date_from: date,
    date_to: date,
    output_dir: Path = Path("data/canonical/polymarket_ws_events_v1"),
) -> dict[str, Any]:
    daily = []
    for target_day in _date_range(date_from, date_to):
        path = output_dir / f"{target_day.isoformat()}.parquet"
        daily.append({"date": target_day.isoformat(), "exists": path.exists(), "path": str(path)})
    return {"dataset": "polymarket_ws_events_v1", "daily": daily}
