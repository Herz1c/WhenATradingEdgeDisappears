from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Any

from binance_recorder.utils import ns_to_iso
from market_recorders.time_utils import build_timestamp_fields

from .datastreams_full_report import decode_full_report


_DECIMAL_SCALE = 10**18


def _decimal_string(value: Any, *, scale: int | None = None) -> str | None:
    if value is None:
        return None
    try:
        dec = Decimal(str(value))
        if scale:
            dec = dec / Decimal(scale)
        return format(dec, "f")
    except Exception:
        return None


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _build_freshness(extracted: dict[str, Any], *, recv_ts_ns: int) -> dict[str, Any]:
    reference_ts_ns = extracted.get("last_seen_timestamp_ns")
    if reference_ts_ns is None and extracted.get("observations_timestamp_s") is not None:
        reference_ts_ns = int(extracted["observations_timestamp_s"]) * 1_000_000_000
    if reference_ts_ns is None and extracted.get("last_update_timestamp_s") is not None:
        reference_ts_ns = int(extracted["last_update_timestamp_s"]) * 1_000_000_000
    if reference_ts_ns is None and extracted.get("timestamp_s") is not None:
        reference_ts_ns = int(extracted["timestamp_s"]) * 1_000_000_000
    if reference_ts_ns is None:
        return {}
    freshness = {
        "reference_ts_ns": int(reference_ts_ns),
        "reference_ts_iso": ns_to_iso(int(reference_ts_ns)),
        "source_age_ms": round((recv_ts_ns - int(reference_ts_ns)) / 1_000_000, 3),
    }
    expires_at_s = extracted.get("expires_at_s")
    if expires_at_s is not None:
        expires_at_ns = int(expires_at_s) * 1_000_000_000
        freshness["expires_in_ms"] = round((expires_at_ns - recv_ts_ns) / 1_000_000, 3)
        freshness["is_expired"] = recv_ts_ns > expires_at_ns
    return freshness


def decode_datastreams_report(payload: dict[str, Any], *, recv_ts_ns: int | None = None) -> dict[str, Any]:
    report = payload.get("report") if isinstance(payload.get("report"), dict) else payload
    decoded_full_report: dict[str, Any] = {}
    full_report = report.get("fullReport")
    if isinstance(full_report, str) and full_report.startswith("0x"):
        try:
            decoded_full_report = decode_full_report(
                full_report,
                feed_id=report.get("feedID") or report.get("feedId") or payload.get("feedID"),
            )
        except Exception as exc:
            decoded_full_report = {"full_report_decode_error": repr(exc)}

    feed_id = _first_value(
        decoded_full_report.get("feed_id"),
        report.get("feedID"),
        report.get("feedId"),
        payload.get("feedID"),
    )
    extracted: dict[str, Any] = {
        "feed_id": feed_id,
        "display_name": _first_value(report.get("displayName"), payload.get("displayName"), "BTC / USD"),
        "full_report_present": isinstance(full_report, str),
    }
    if feed_id is not None:
        extracted["feed_version_hex"] = f"0x{str(feed_id)[2:6].lower()}"
    schema_version = _first_value(report.get("schemaVersion"), payload.get("schemaVersion"), decoded_full_report.get("schema_version"))
    if schema_version is not None:
        extracted["schema_version"] = schema_version
    if decoded_full_report.get("report_context") is not None:
        extracted["report_context"] = decoded_full_report["report_context"]
    if decoded_full_report.get("signature_count") is not None:
        extracted["signature_count"] = decoded_full_report["signature_count"]
    if decoded_full_report.get("raw_vs") is not None:
        extracted["raw_vs"] = decoded_full_report["raw_vs"]
    if isinstance(full_report, str):
        extracted["full_report_sha256"] = hashlib.sha256(full_report.encode("utf-8")).hexdigest()
        if decoded_full_report.get("full_report_decode_error") is None:
            extracted["full_report_decode_status"] = "decoded"
        else:
            extracted["full_report_decode_status"] = "decode_error"
            extracted["full_report_decode_error"] = decoded_full_report["full_report_decode_error"]

    price_candidates = (
        ("benchmark_price", "decoded_v3_benchmark_price"),
        ("benchmarkPrice", "reported_primary_price"),
        ("mid_price", "decoded_mid_price"),
        ("midPrice", "reported_mid_price"),
        ("price", "reported_price"),
        ("primaryPrice", "reported_primary_price"),
        ("mid", "decoded_mid"),
        ("last_traded_price", "decoded_last_traded_price"),
        ("lastTradedPrice", "reported_last_traded_price"),
        ("exchange_rate", "decoded_exchange_rate"),
    )
    selected_price_raw = None
    selected_price_source = None
    for key, source_name in price_candidates:
        value = _first_value(decoded_full_report.get(key), report.get(key), payload.get(key))
        if value is not None:
            selected_price_raw = str(value)
            selected_price_source = source_name
            break

    bid_value = _first_value(decoded_full_report.get("bid"), report.get("bid"), payload.get("bid"), decoded_full_report.get("best_bid"))
    ask_value = _first_value(decoded_full_report.get("ask"), report.get("ask"), payload.get("ask"), decoded_full_report.get("best_ask"))
    if bid_value is not None:
        extracted["bid_raw"] = str(bid_value)
        extracted["bid"] = _decimal_string(bid_value, scale=_DECIMAL_SCALE)
    if ask_value is not None:
        extracted["ask_raw"] = str(ask_value)
        extracted["ask"] = _decimal_string(ask_value, scale=_DECIMAL_SCALE)

    if selected_price_raw is None and bid_value is not None and ask_value is not None:
        midpoint_raw = (Decimal(str(bid_value)) + Decimal(str(ask_value))) / Decimal(2)
        selected_price_raw = format(midpoint_raw, "f")
        selected_price_source = "derived_midpoint_from_bid_ask"

    if selected_price_raw is not None:
        extracted["live_price_raw"] = selected_price_raw
        extracted["live_price"] = _decimal_string(selected_price_raw, scale=_DECIMAL_SCALE)
        extracted["price_raw"] = selected_price_raw
        extracted["price"] = extracted["live_price"]
        extracted["mid_price_raw"] = selected_price_raw
        extracted["mid_price"] = extracted["live_price"]
        extracted["primary_price_source"] = selected_price_source

    for field_name, value in (
        ("valid_from_timestamp", _first_value(decoded_full_report.get("valid_from_timestamp"), report.get("validFromTimestamp"))),
        ("observations_timestamp", _first_value(decoded_full_report.get("observations_timestamp"), report.get("observationsTimestamp"), report.get("observationTimestamp"))),
        ("expires_at", _first_value(decoded_full_report.get("expires_at"), report.get("expiresAt"))),
        ("last_update_timestamp", _first_value(decoded_full_report.get("last_update_timestamp"), report.get("lastUpdateTimestamp"))),
        ("last_seen_timestamp", decoded_full_report.get("last_seen_timestamp_ns")),
        ("timestamp", _first_value(decoded_full_report.get("timestamp"), report.get("timestamp"))),
    ):
        extracted.update(build_timestamp_fields(field_name, value))

    for raw_name, output_name in (
        ("native_fee", "native_fee_raw"),
        ("link_fee", "link_fee_raw"),
        ("market_status", "market_status"),
        ("bid_volume", "bid_volume_raw"),
        ("ask_volume", "ask_volume_raw"),
        ("tokenized_price", "tokenized_price_raw"),
        ("current_multiplier", "current_multiplier_raw"),
        ("new_multiplier", "new_multiplier_raw"),
        ("ripcord", "ripcord"),
    ):
        value = decoded_full_report.get(raw_name)
        if value is not None:
            extracted[output_name] = str(value) if output_name.endswith("_raw") else value

    if report.get("baseAsset") is not None:
        extracted["base_asset"] = report["baseAsset"]
    if report.get("quoteAsset") is not None:
        extracted["quote_asset"] = report["quoteAsset"]
    if report.get("productName") is not None:
        extracted["product_name"] = report["productName"]
    if report.get("reportContext") is not None and "report_context" not in extracted:
        extracted["report_context"] = report["reportContext"]

    if recv_ts_ns is not None:
        extracted.update(_build_freshness(extracted, recv_ts_ns=recv_ts_ns))
        if extracted.get("source_age_ms") is not None:
            extracted["age_ms"] = extracted["source_age_ms"]
            extracted["freshness_ms"] = extracted["source_age_ms"]

    return {key: value for key, value in extracted.items() if value is not None}
