from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import polars as pl

from market_recorders.file_quality import FINALIZED_CLEAN
from market_recorders.unified_reader import UnifiedRawReader

EXTERNAL_BTC_EVENTS_V1_SCHEMA = {
    "recv_ts_ns": pl.Int64,
    "source": pl.Utf8,
    "venue": pl.Utf8,
    "symbol": pl.Utf8,
    "event_type": pl.Utf8,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "mid_price": pl.Float64,
    "last_price": pl.Float64,
    "trade_price": pl.Float64,
    "trade_size": pl.Float64,
    "trade_side": pl.Utf8,
    "order_book_status": pl.Utf8,
    "continuity_state": pl.Utf8,
    "source_timing_summary": pl.Utf8,
    "quarantine_flag": pl.Boolean,
}

BINANCE_USDM_QUARANTINE_START_NS = int(
    datetime(2026, 4, 20, 7, 0, tzinfo=UTC).timestamp() * 1_000_000_000
)
BINANCE_USDM_QUARANTINE_END_NS = int(
    datetime(2026, 4, 20, 8, 0, tzinfo=UTC).timestamp() * 1_000_000_000
)

SOURCE_SPECS = [
    ("binance_spot", Path("binance") / "spot" / "BTCUSDT", "binance", "BTCUSDT"),
    ("binance_usdm", Path("binance") / "usdm" / "BTCUSDT", "binance_usdm", "BTCUSDT"),
    ("hyperliquid", Path("hyperliquid") / "linear" / "BTC", "hyperliquid", "BTC"),
]

FORBIDDEN_SOURCES = {"bybit", "coinbase", "deribit", "fng", "rtds"}
PARQUET_CHUNK_ROWS = 50_000


def _date_range(date_from: date, date_to: date) -> list[date]:
    return [date_from + timedelta(days=i) for i in range((date_to - date_from).days + 1)]


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_float(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(record.get(key))
        if value is not None:
            return value
    return None


def _levels_mid(record: dict[str, Any]) -> tuple[float | None, float | None]:
    bids = record.get("bids")
    asks = record.get("asks")
    bid = ask = None
    if isinstance(bids, list) and bids:
        first = bids[0]
        if isinstance(first, dict):
            bid = _to_float(first.get("price") or first.get("px") or first.get("p"))
        elif isinstance(first, (list, tuple)) and first:
            bid = _to_float(first[0])
    if isinstance(asks, list) and asks:
        first = asks[0]
        if isinstance(first, dict):
            ask = _to_float(first.get("price") or first.get("px") or first.get("p"))
        elif isinstance(first, (list, tuple)) and first:
            ask = _to_float(first[0])
    return bid, ask


def normalize_event_type(record: dict[str, Any]) -> str:
    raw = str(record.get("event_type") or record.get("record_type") or record.get("stream") or "").lower()
    if "trade" in raw:
        return "trade"
    if "ticker" in raw or "mark" in raw:
        return "ticker"
    return "book_update"


def is_binance_usdm_quarantined(recv_ts_ns: int) -> bool:
    return BINANCE_USDM_QUARANTINE_START_NS <= int(recv_ts_ns) < BINANCE_USDM_QUARANTINE_END_NS


def continuity_state(record: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    flags = []
    if metadata:
        flags.extend(str(flag).lower() for flag in metadata.get("quality_flags", []) or [])
    text = " ".join(flags + [str(record.get("continuity_state") or ""), str(record.get("event") or "")]).lower()
    if "reconnect" in text:
        return "reconnect"
    if "gap" in text or "compromised" in text:
        return "gap"
    return "ok"


def normalize_record(
    record: dict[str, Any],
    *,
    source: str,
    venue: str,
    symbol: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if source.lower() in FORBIDDEN_SOURCES:
        raise ValueError(f"Forbidden source cannot enter external_btc_events_v1: {source}")
    recv_ts_ns = record.get("recv_ts_ns")
    if recv_ts_ns is None:
        return None
    recv_ts_ns = int(recv_ts_ns)
    level_bid, level_ask = _levels_mid(record)
    best_bid = _first_float(record, "best_bid", "best_bid_price", "bid", "bid_price") or level_bid
    best_ask = _first_float(record, "best_ask", "best_ask_price", "ask", "ask_price") or level_ask
    mid_price = _first_float(record, "mid_price", "mid")
    if mid_price is None and best_bid is not None and best_ask is not None:
        mid_price = (best_bid + best_ask) / 2.0
    trade_price = _first_float(record, "trade_price", "price", "last_trade_price")
    last_price = _first_float(record, "last_price", "mark_price", "index_price") or trade_price
    lag_ms = None
    exchange_ts = record.get("event_time_ms") or record.get("exchange_ts_ms") or record.get("E")
    if exchange_ts is not None:
        try:
            lag_ms = (recv_ts_ns / 1_000_000) - float(exchange_ts)
        except Exception:
            lag_ms = None
    quarantine = source == "binance_usdm" and is_binance_usdm_quarantined(recv_ts_ns)
    order_book_status = "healthy" if best_bid is not None and best_ask is not None and best_bid <= best_ask else "degraded"
    if metadata and metadata.get("quality_class") != FINALIZED_CLEAN:
        order_book_status = "stale"
    timing = {"exchange_ts": exchange_ts, "recv_ts_ns": recv_ts_ns, "lag_ms": lag_ms}
    return {
        "recv_ts_ns": recv_ts_ns,
        "source": source,
        "venue": venue,
        "symbol": symbol,
        "event_type": normalize_event_type(record),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "last_price": last_price,
        "trade_price": trade_price,
        "trade_size": _first_float(record, "trade_size", "quantity", "qty", "size"),
        "trade_side": str(record.get("trade_side") or record.get("side") or ""),
        "order_book_status": order_book_status,
        "continuity_state": continuity_state(record, metadata),
        "source_timing_summary": json.dumps(timing, sort_keys=True, separators=(",", ":")),
        "quarantine_flag": quarantine,
    }


def _quality_allowed(reader: UnifiedRawReader, path: Path) -> bool:
    metadata = reader.file_metadata(path)
    return metadata.get("finalized") is True and metadata.get("quality_class") == FINALIZED_CLEAN


def _write_rows_streaming(
    rows,
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
        nonlocal writer, buffer, total_rows
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


def _iter_external_btc_rows(reader: UnifiedRawReader, target_day: date):
    for source, prefix, venue, symbol in SOURCE_SPECS:
        for path in reader.list_files_for_date(
            relative_prefix=prefix,
            target_date=target_day,
            kind="ws",
            finalized_only=True,
        ):
            metadata = reader.file_metadata(path)
            if not _quality_allowed(reader, path):
                continue
            for record in reader.iter_file(path, finalized_only=True):
                normalized = normalize_record(
                    record,
                    source=source,
                    venue=venue,
                    symbol=symbol,
                    metadata=metadata,
                )
                if normalized is None:
                    continue
                if normalized["quarantine_flag"]:
                    continue
                yield normalized


def build_external_btc_events(
    *,
    root: Path = Path("data"),
    date_from: date,
    date_to: date,
    output_dir: Path = Path("data/canonical/external_btc_events_v1"),
    quality_registry_path: Path | None = None,
) -> dict[str, Any]:
    del quality_registry_path
    reader = UnifiedRawReader(root=root)
    daily: list[dict[str, Any]] = []
    for target_day in _date_range(date_from, date_to):
        output_path = output_dir / f"{target_day.isoformat()}.parquet"
        rows_written = _write_rows_streaming(
            _iter_external_btc_rows(reader, target_day),
            output_path=output_path,
            schema=EXTERNAL_BTC_EVENTS_V1_SCHEMA,
        )
        daily.append({"date": target_day.isoformat(), "rows": rows_written, "path": str(output_path)})
    return {"dataset": "external_btc_events_v1", "daily": daily}
