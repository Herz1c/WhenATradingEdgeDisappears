from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from market_recorders.unified_reader import UnifiedRawReader


def _symbol_path_component(symbol: str) -> str:
    return symbol.strip().lower().replace("/", "_")


class PolymarketRtdsRawReader(UnifiedRawReader):
    def __init__(
        self,
        *,
        root: Path,
        topic: str = "crypto_prices_chainlink",
        symbol: str = "btc/usd",
    ) -> None:
        super().__init__(root=root)
        self.topic = topic
        self.symbol = symbol
        self.relative_prefix = Path("polymarket") / "rtds" / topic / _symbol_path_component(symbol)

    def iter_kind(
        self,
        *,
        target_date: str,
        kind: str,
        topic: str | None = None,
        message_type: str | None = None,
        symbol: str | None = None,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict]:
        if topic is not None and topic != self.topic:
            return
        if symbol is not None and symbol != self.symbol:
            return
        if message_type is not None and message_type != "update":
            return
        yield from self.iter_date(
            relative_prefix=self.relative_prefix,
            target_date=target_date,
            kind=kind,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        )

    def iter_normalized(
        self,
        *,
        target_date: str,
        kind: str = "ws",
        topic: str | None = None,
        message_type: str | None = None,
        symbol: str | None = None,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict]:
        for record in self.iter_kind(
            target_date=target_date,
            kind=kind,
            topic=topic,
            message_type=message_type,
            symbol=symbol,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            chainlink_ts_ms = record.get("chainlink_ts_ms")
            if chainlink_ts_ms is None:
                source_timestamps = record.get("source_timestamps", {})
                chainlink_ts_ms = source_timestamps.get("price_timestamp_ms") or source_timestamps.get(
                    "message_timestamp_ms"
                )
            btc_usd_price = record.get("btc_usd_price")
            if btc_usd_price is None:
                extracted = record.get("extracted", {})
                btc_usd_price = record.get("live_price", extracted.get("value"))
            yield {
                "record_type": record.get("record_type", "chainlink_live_reference"),
                "recv_ts_ns": record.get("recv_ts_ns"),
                "recv_ts_iso": record.get("recv_ts_iso"),
                "recv_ts_source": record.get("recv_ts_source"),
                "chainlink_ts_ms": chainlink_ts_ms,
                "chainlink_ts_iso": record.get("chainlink_ts_iso"),
                "btc_usd_price": btc_usd_price,
                "source": record.get("source", "polymarket_rtds_chainlink"),
            }
