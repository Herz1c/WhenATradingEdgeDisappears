from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from market_recorders.time_utils import epoch_value_to_ns

_PRIORITY_FIELDS = (
    "cts_ns",
    "event_time_ns",
    "updates_event_time_ns",
    "item_t_ns",
    "trade_time_ns",
    "trades_time_ns",
    "heartbeat_current_time_ns",
    "message_timestamp_ns",
    "ts_ns",
    "timestamp_ns",
    "time_ns",
    "created_at_ns",
    "updated_at_ns",
)

_SKIP_REFERENCE_SEMANTICS = {
    "collection_snapshot",
    "delayed_display",
    "historical_series",
    "metadata_only",
    "recv_only",
    "source_metadata",
    "verification_only",
}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    if lower == upper:
        return lower_value
    return lower_value + ((upper_value - lower_value) * (rank - lower))


def _suffix_to_unit(name: str) -> str | None:
    if name.endswith("_ns"):
        return "nanosecond"
    if name.endswith("_us"):
        return "microsecond"
    if name.endswith("_ms"):
        return "millisecond"
    if name.endswith("_s"):
        return "second"
    return None


def _normalize_candidate_timestamp_ns(name: str, value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    unit = _suffix_to_unit(name)
    if unit is None:
        return None
    return epoch_value_to_ns(value, unit=unit)


def _record_timestamp_semantics(record: dict[str, Any]) -> str:
    value = record.get("timestamp_semantics") or record.get("source_timestamp_role") or ""
    return str(value).strip().lower()


def _source_key_from_record(record: dict[str, Any]) -> str:
    source = str(record.get("source") or "")
    venue = str(record.get("venue") or record.get("market_type") or record.get("channel") or "")
    symbol = str(record.get("symbol") or record.get("product_id") or "")
    if source == "binance":
        market = str(record.get("venue") or record.get("market") or "")
        return f"binance/{market}/{symbol}".rstrip("/")
    if source == "polymarket":
        return "polymarket/btc_updown_5m"
    if source == "polymarket_rtds_chainlink":
        return "polymarket/rtds/crypto_prices_chainlink/btc_usd"
    if source == "chainlink_public_delayed":
        return f"chainlink_public_delayed/public_stream_page/{symbol}".rstrip("/")
    return "/".join(part for part in (source, venue, symbol) if part)


def gap_warning_threshold_ms(source_key: str) -> float:
    if source_key.startswith("binance/spot"):
        return 2_000.0
    if source_key.startswith("binance/usdm"):
        return 2_000.0
    if source_key == "polymarket/btc_updown_5m":
        return 5_000.0
    if source_key.startswith("polymarket/rtds"):
        return 20_000.0
    if source_key.startswith("bybit/linear"):
        return 2_000.0
    if source_key.startswith("coinbase/advanced"):
        return 3_000.0
    if source_key.startswith("hyperliquid/linear"):
        return 3_000.0
    if source_key.startswith("deribit/options"):
        return 5_000.0
    return 5_000.0


def lag_warning_threshold_ms(source_key: str) -> float:
    if source_key.startswith("binance/"):
        return 1_500.0
    if source_key == "polymarket/btc_updown_5m":
        return 2_000.0
    if source_key.startswith("polymarket/rtds"):
        return 20_000.0
    if source_key.startswith("bybit/linear"):
        return 2_000.0
    if source_key.startswith("coinbase/advanced"):
        return 2_000.0
    if source_key.startswith("hyperliquid/linear"):
        return 2_500.0
    if source_key.startswith("deribit/options"):
        return 5_000.0
    return 3_000.0


def reference_timestamp_ns(record: dict[str, Any]) -> int | None:
    semantics = _record_timestamp_semantics(record)
    if semantics in _SKIP_REFERENCE_SEMANTICS or semantics.startswith("metadata_only"):
        return None
    recv_minus_chainlink_ms = record.get("recv_minus_chainlink_ms")
    recv_ts_ns = record.get("recv_ts_ns")
    if isinstance(recv_minus_chainlink_ms, (int, float)) and isinstance(recv_ts_ns, int):
        return int(recv_ts_ns - (float(recv_minus_chainlink_ms) * 1_000_000))
    source_timestamps = record.get("source_timestamps")
    if not isinstance(source_timestamps, dict):
        return None
    for field_name in _PRIORITY_FIELDS:
        candidate = _normalize_candidate_timestamp_ns(field_name, source_timestamps.get(field_name))
        if candidate is not None:
            return candidate
    candidates: list[tuple[int, str]] = []
    for name, value in source_timestamps.items():
        if name.endswith("_iso") or name.startswith("next_"):
            continue
        candidate = _normalize_candidate_timestamp_ns(name, value)
        if candidate is not None:
            candidates.append((candidate, name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][0]


@dataclass(slots=True)
class TransportTelemetryAccumulator:
    source_key: str
    sample_count: int = 0
    lag_sample_count: int = 0
    lag_values_ms: list[float] = field(default_factory=list)
    negative_lag_count: int = 0
    last_recv_ts_ns: int | None = None
    silent_gap_count: int = 0
    max_gap_ms: float | None = None

    def observe(self, record: dict[str, Any]) -> None:
        recv_ts_ns = record.get("recv_ts_ns")
        if not isinstance(recv_ts_ns, int):
            return
        self.sample_count += 1
        if self.last_recv_ts_ns is not None:
            gap_ms = (recv_ts_ns - self.last_recv_ts_ns) / 1_000_000
            self.max_gap_ms = gap_ms if self.max_gap_ms is None else max(self.max_gap_ms, gap_ms)
            if gap_ms >= gap_warning_threshold_ms(self.source_key):
                self.silent_gap_count += 1
        self.last_recv_ts_ns = recv_ts_ns
        source_ts_ns = reference_timestamp_ns(record)
        if source_ts_ns is None:
            return
        lag_ms = (recv_ts_ns - source_ts_ns) / 1_000_000
        self.lag_sample_count += 1
        self.lag_values_ms.append(lag_ms)
        if lag_ms < 0:
            self.negative_lag_count += 1

    def summary(self) -> dict[str, Any]:
        if not self.sample_count:
            return {}
        warnings: list[str] = []
        p50_lag_ms = median(self.lag_values_ms) if self.lag_values_ms else None
        p95_lag_ms = _percentile(self.lag_values_ms, 0.95)
        max_lag_ms = max(self.lag_values_ms) if self.lag_values_ms else None
        lag_threshold = lag_warning_threshold_ms(self.source_key)
        if p95_lag_ms is not None and p95_lag_ms > lag_threshold:
            warnings.append("lag_p95_elevated")
        if max_lag_ms is not None and max_lag_ms > lag_threshold * 2:
            warnings.append("lag_max_elevated")
        if self.negative_lag_count > 0:
            warnings.append("negative_lag_observed")
        if self.silent_gap_count > 0:
            warnings.append("silent_gap_observed")
        return {
            "sample_count": self.sample_count,
            "lag_sample_count": self.lag_sample_count,
            "recv_minus_source_ms": {
                "p50": p50_lag_ms,
                "p95": p95_lag_ms,
                "max": max_lag_ms,
            },
            "negative_lag_count": self.negative_lag_count,
            "negative_lag_rate": (
                round(self.negative_lag_count / self.lag_sample_count, 6) if self.lag_sample_count else None
            ),
            "silent_gap_count": self.silent_gap_count,
            "max_gap_ms": self.max_gap_ms,
            "warnings": warnings,
        }


def new_transport_accumulator_for_record(record: dict[str, Any]) -> TransportTelemetryAccumulator:
    return TransportTelemetryAccumulator(source_key=_source_key_from_record(record))
