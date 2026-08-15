from __future__ import annotations

from typing import Any

from eth_abi import decode as abi_decode


_FULL_REPORT_TYPES = ("bytes32[3]", "bytes", "bytes32[]", "bytes32[]", "bytes32")

_REPORT_SCHEMAS: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {
    "0002": (
        2,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192"),
        ("feed_id", "valid_from_timestamp", "observations_timestamp", "native_fee", "link_fee", "expires_at", "price"),
    ),
    "0003": (
        3,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192", "int192", "int192"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "benchmark_price",
            "bid",
            "ask",
        ),
    ),
    "0004": (
        4,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192", "uint8"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "price",
            "market_status",
        ),
    ),
    "0005": (
        5,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192", "uint32", "uint32"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "rate",
            "timestamp",
            "duration",
        ),
    ),
    "0006": (
        6,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192", "int192", "int192", "int192", "int192"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "price",
            "price2",
            "price3",
            "price4",
            "price5",
        ),
    ),
    "0007": (
        7,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "exchange_rate",
        ),
    ),
    "0008": (
        8,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "uint64", "int192", "uint32"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "last_update_timestamp",
            "mid_price",
            "market_status",
        ),
    ),
    "0009": (
        9,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192", "uint64", "int192", "uint32"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "nav_per_share",
            "nav_date",
            "aum",
            "ripcord",
        ),
    ),
    "000a": (
        10,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "uint64", "int192", "uint32", "int192", "int192", "uint32", "int192"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "last_update_timestamp",
            "price",
            "market_status",
            "current_multiplier",
            "new_multiplier",
            "activation_date_time",
            "tokenized_price",
        ),
    ),
    "000b": (
        11,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192", "uint64", "int192", "int192", "int192", "int192", "int192", "uint32"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "mid",
            "last_seen_timestamp_ns",
            "bid",
            "bid_volume",
            "ask",
            "ask_volume",
            "last_traded_price",
            "market_status",
        ),
    ),
    "000c": (
        12,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192", "int192", "uint64", "uint32"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "nav_per_share",
            "next_nav_per_share",
            "nav_date",
            "ripcord",
        ),
    ),
    "000d": (
        13,
        ("bytes32", "uint32", "uint32", "uint192", "uint192", "uint32", "int192", "int192", "uint64", "uint64", "int192"),
        (
            "feed_id",
            "valid_from_timestamp",
            "observations_timestamp",
            "native_fee",
            "link_fee",
            "expires_at",
            "best_ask",
            "best_bid",
            "ask_volume",
            "bid_volume",
            "last_traded_price",
        ),
    ),
}


def _to_hex(value: Any) -> str:
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if isinstance(value, bytearray):
        return "0x" + bytes(value).hex()
    return str(value)


def _version_from_feed_id(feed_id: str) -> str:
    normalized = feed_id.lower()
    if normalized.startswith("0x") and len(normalized) >= 6:
        return normalized[2:6]
    raise ValueError(f"Invalid feed id: {feed_id}")


def decode_full_report(full_report_hex: str, *, feed_id: str | None = None) -> dict[str, Any]:
    raw_hex = full_report_hex[2:] if full_report_hex.startswith("0x") else full_report_hex
    outer = abi_decode(_FULL_REPORT_TYPES, bytes.fromhex(raw_hex))
    report_context, report_blob, raw_rs, raw_ss, raw_vs = outer
    inferred_feed_id = feed_id or _to_hex(report_blob[:32])
    version = _version_from_feed_id(inferred_feed_id)
    if version not in _REPORT_SCHEMAS:
        raise ValueError(f"Unsupported Chainlink Data Streams report version: 0x{version}")
    schema_version, types, names = _REPORT_SCHEMAS[version]
    decoded_values = abi_decode(types, report_blob)
    payload = {
        name: (_to_hex(value) if name == "feed_id" else value)
        for name, value in zip(names, decoded_values, strict=True)
    }
    payload["schema_version"] = schema_version
    payload["feed_version_hex"] = f"0x{version}"
    payload["report_context"] = [_to_hex(item) for item in report_context]
    payload["signature_count"] = min(len(raw_rs), len(raw_ss))
    payload["raw_vs"] = _to_hex(raw_vs)
    return payload
