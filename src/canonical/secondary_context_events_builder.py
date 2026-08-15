from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from chainlink_recorder.live_reference import normalize_public_delayed_record
from chainlink_recorder.reader import ChainlinkPublicDelayedRawReader

SECONDARY_CONTEXT_EVENTS_V1_SCHEMA = {
    "recv_ts_ns": pl.Int64,
    "source": pl.Utf8,
    "endpoint": pl.Utf8,
    "semantic_role": pl.Utf8,
    "causal_available_ts_ns": pl.Int64,
    "extracted_price": pl.Float64,
    "extracted_ts": pl.Int64,
    "payload_summary": pl.Utf8,
}

CAUSAL_FLOOR_NS = 1_800_000_000_000


def _date_range(date_from: date, date_to: date) -> list[date]:
    return [date_from + timedelta(days=i) for i in range((date_to - date_from).days + 1)]


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    normalized = normalize_public_delayed_record(record)
    recv_ts_ns = normalized.get("recv_ts_ns")
    if recv_ts_ns is None:
        return None
    extracted = record.get("extracted", {}) if isinstance(record.get("extracted"), dict) else {}
    price = _to_float(record.get("btc_usd_price") or normalized.get("btc_usd_price"))
    payload = {
        "record_type": record.get("record_type"),
        "chainlink_display_ts": record.get("chainlink_display_ts"),
        "price_source": extracted.get("price_source") or extracted.get("primary_price_source"),
    }
    return {
        "recv_ts_ns": int(recv_ts_ns),
        "source": "chainlink_public_delayed",
        "endpoint": str(record.get("endpoint") or "public_stream_page"),
        "semantic_role": "btc_price_calibration",
        "causal_available_ts_ns": int(recv_ts_ns) + CAUSAL_FLOOR_NS,
        "extracted_price": price,
        "extracted_ts": int(normalized.get("chainlink_ts_ns") or 0),
        "payload_summary": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }


def build_secondary_context_events(
    *,
    root: Path = Path("data"),
    date_from: date,
    date_to: date,
    output_dir: Path = Path("data/canonical/secondary_context_events_v1"),
    quality_registry_path: Path | None = None,
) -> dict[str, Any]:
    del quality_registry_path
    reader = ChainlinkPublicDelayedRawReader(root=root, symbol="BTCUSD")
    daily = []
    for target_day in _date_range(date_from, date_to):
        rows = []
        for record in reader.iter_kind(target_date=target_day.isoformat(), kind="page", finalized_only=True):
            normalized = normalize_record(record)
            if normalized is not None:
                rows.append(normalized)
        frame = pl.DataFrame(rows, schema=SECONDARY_CONTEXT_EVENTS_V1_SCHEMA, orient="row")
        output_path = output_dir / f"{target_day.isoformat()}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output_path)
        daily.append({"date": target_day.isoformat(), "rows": frame.height, "path": str(output_path)})
    return {"dataset": "secondary_context_events_v1", "daily": daily}
