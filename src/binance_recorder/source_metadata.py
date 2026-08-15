from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from market_recorders.time_utils import build_timestamp_fields

_BINANCE_TIME_FIELDS = {
    "E": "event_time",
    "T": "trade_time",
    "transactTime": "transaction_time",
    "closeTime": "close_time",
    "openTime": "open_time",
    "nextFundingTime": "next_funding_time",
}

_STREAM_EVENT_SUFFIXES = (
    ("@bookTicker", "bookTicker"),
    ("@depth", "depthUpdate"),
    ("@trade", "trade"),
    ("@aggTrade", "aggTrade"),
    ("@markPrice", "markPriceUpdate"),
    ("@forceOrder", "forceOrder"),
)


def _to_decimal_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return format(Decimal(str(value)), "f")
    except (InvalidOperation, ValueError):
        return None


def infer_binance_event_type(payload: dict[str, Any], *, stream: str | None = None) -> str | None:
    event_type = payload.get("e")
    if isinstance(event_type, str) and event_type:
        return event_type
    if not isinstance(stream, str):
        return None
    for suffix, inferred in _STREAM_EVENT_SUFFIXES:
        if suffix in stream:
            return inferred
    return None


def extract_binance_source_timestamps(
    payload: dict[str, Any],
    *,
    unit_hint: str | None = None,
    stream: str | None = None,
) -> dict[str, Any]:
    event_type = str(infer_binance_event_type(payload, stream=stream) or "")
    extracted: dict[str, Any] = {}
    for key, field_name in _BINANCE_TIME_FIELDS.items():
        if key in payload:
            if event_type == "markPriceUpdate" and key == "T":
                field_name = "next_funding_time"
            extracted.update(build_timestamp_fields(field_name, payload[key], unit_hint=unit_hint))
    return extracted


def normalize_binance_payload(payload: dict[str, Any], *, stream: str | None = None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    event_type = infer_binance_event_type(payload, stream=stream)
    if payload.get("s"):
        normalized["symbol"] = payload.get("s")
    if event_type:
        normalized["normalized_event_type"] = event_type
    if event_type in {"trade", "aggTrade"}:
        normalized["price"] = _to_decimal_string(payload.get("p"))
        normalized["quantity"] = _to_decimal_string(payload.get("q"))
        normalized["buyer_is_maker"] = payload.get("m")
        normalized["trade_id"] = payload.get("t") or payload.get("a")
    elif event_type == "bookTicker":
        normalized["best_bid_price"] = _to_decimal_string(payload.get("b"))
        normalized["best_bid_qty"] = _to_decimal_string(payload.get("B"))
        normalized["best_ask_price"] = _to_decimal_string(payload.get("a"))
        normalized["best_ask_qty"] = _to_decimal_string(payload.get("A"))
        normalized["update_id"] = payload.get("u")
    elif event_type == "depthUpdate":
        normalized["first_update_id"] = payload.get("U")
        normalized["final_update_id"] = payload.get("u")
        normalized["previous_final_update_id"] = payload.get("pu")
        normalized["bid_level_count"] = len(payload.get("b", []))
        normalized["ask_level_count"] = len(payload.get("a", []))
    elif event_type == "markPriceUpdate":
        normalized["mark_price"] = _to_decimal_string(payload.get("p"))
        normalized["index_price"] = _to_decimal_string(payload.get("i"))
        normalized["estimated_settle_price"] = _to_decimal_string(payload.get("P"))
        normalized["funding_rate"] = _to_decimal_string(payload.get("r"))
        normalized["next_funding_time"] = payload.get("T")
        if payload.get("T") is not None:
            normalized.update(build_timestamp_fields("next_funding_time", payload.get("T"), unit_hint="millisecond"))
    return {key: value for key, value in normalized.items() if value is not None}
