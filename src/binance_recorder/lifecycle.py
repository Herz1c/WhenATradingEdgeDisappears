from __future__ import annotations

from typing import Any

from binance_recorder.utils import ns_to_iso
from market_recorders.time_utils import recv_timestamp_source, utc_now_ns


def lifecycle_event(
    *,
    event: str,
    symbol: str,
    market: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recv_ts_ns = utc_now_ns()
    payload = {
        "record_type": "lifecycle_event",
        "event": event,
        "source": "binance",
        "market": market,
        "symbol": symbol,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
        "ts_source": recv_timestamp_source(),
    }
    if details:
        payload.update(details)
    return payload
