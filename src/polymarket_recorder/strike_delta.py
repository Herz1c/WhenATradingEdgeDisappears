from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

from market_recorders.time_utils import utc_now_ns

from chainlink_recorder.live_reference_events import LiveReferenceEventsReader

from .reader import PolymarketRawReader


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _matches_market(
    record: dict[str, Any],
    *,
    market_slug: str | None,
    condition_id: str | None,
) -> bool:
    if market_slug is not None and str(record.get("selected_market_slug")) != market_slug:
        return False
    if condition_id is not None and str(record.get("selected_condition_id")) != condition_id:
        return False
    return True


class PolymarketStrikeDeltaReader:
    def __init__(
        self,
        *,
        root: Path,
        symbol: str = "BTCUSD",
        rtds_symbol: str = "btc/usd",
    ) -> None:
        self.root = root
        self._strike_reader = PolymarketRawReader(root=root)
        del symbol, rtds_symbol
        self._reference_reader = LiveReferenceEventsReader(root=root)

    def iter_strike_records(
        self,
        *,
        target_date: str,
        market_slug: str | None = None,
        condition_id: str | None = None,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict[str, Any]]:
        for record in self._strike_reader.iter_kind(
            target_date=target_date,
            kind="strike",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            if record.get("record_type") != "strike_observation":
                continue
            if record.get("label_safe") is not True:
                continue
            if not _matches_market(record, market_slug=market_slug, condition_id=condition_id):
                continue
            yield record

    @staticmethod
    def build_delta_record(*, strike_record: dict[str, Any], live_reference: dict[str, Any]) -> dict[str, Any] | None:
        strike_price = _to_decimal(strike_record.get("price_to_beat"))
        live_price = _to_decimal(live_reference.get("btc_usd_price") or live_reference.get("live_price"))
        price_valid = bool(live_reference.get("price_valid") is not False and live_price is not None)
        if strike_price is None:
            return None
        if not price_valid:
            return {
                "record_type": "strike_delta_reference",
                "selected_market_slug": strike_record.get("selected_market_slug"),
                "selected_market_id": strike_record.get("selected_market_id"),
                "selected_condition_id": strike_record.get("selected_condition_id"),
                "market_family": strike_record.get("market_family"),
                "strike_recv_ts_ns": strike_record.get("recv_ts_ns"),
                "strike_recv_ts_iso": strike_record.get("recv_ts_iso"),
                "price_to_beat": str(strike_price),
                "live_source": live_reference.get("source"),
                "live_recv_ts_ns": live_reference.get("recv_ts_ns"),
                "live_recv_ts_iso": live_reference.get("recv_ts_iso"),
                "live_chainlink_ts_ns": live_reference.get("chainlink_ts_ns"),
                "live_chainlink_ts_iso": live_reference.get("chainlink_ts_iso"),
                "live_chainlink_btcusd": None,
                "price_unavailable": True,
                "delta_to_strike": float("nan"),
                "absolute_delta_to_strike": float("nan"),
                "delta_percent": float("nan"),
                "delta_bps": float("nan"),
            }
        signed_delta = live_price - strike_price
        absolute_delta = abs(signed_delta)
        delta_pct = (signed_delta / strike_price) * Decimal(100) if strike_price != 0 else None
        delta_bps = (signed_delta / strike_price) * Decimal(10_000) if strike_price != 0 else None
        return {
            "record_type": "strike_delta_reference",
            "selected_market_slug": strike_record.get("selected_market_slug"),
            "selected_market_id": strike_record.get("selected_market_id"),
            "selected_condition_id": strike_record.get("selected_condition_id"),
            "market_family": strike_record.get("market_family"),
            "strike_recv_ts_ns": strike_record.get("recv_ts_ns"),
            "strike_recv_ts_iso": strike_record.get("recv_ts_iso"),
            "price_to_beat": str(strike_price),
            "live_source": live_reference.get("source"),
            "live_recv_ts_ns": live_reference.get("recv_ts_ns"),
            "live_recv_ts_iso": live_reference.get("recv_ts_iso"),
            "live_chainlink_ts_ns": live_reference.get("chainlink_ts_ns"),
            "live_chainlink_ts_iso": live_reference.get("chainlink_ts_iso"),
            "live_chainlink_btcusd": str(live_price),
            "price_unavailable": False,
            "delta_to_strike": str(signed_delta),
            "absolute_delta_to_strike": str(absolute_delta),
            "delta_percent": float(delta_pct) if delta_pct is not None else None,
            "delta_bps": float(delta_bps) if delta_bps is not None else None,
        }

    def iter_historical_delta(
        self,
        *,
        target_date: str,
        market_slug: str | None = None,
        condition_id: str | None = None,
        allow_verification_fallback: bool = False,
        reference_lookback_days: int = 1,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict[str, Any]]:
        del allow_verification_fallback, reference_lookback_days
        for strike_record in self.iter_strike_records(
            target_date=target_date,
            market_slug=market_slug,
            condition_id=condition_id,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            strike_ts_ns = strike_record.get("recv_ts_ns")
            if strike_ts_ns is None:
                continue
            live_reference = self._reference_reader.latest_asof(asof_ts_ns=int(strike_ts_ns))
            if live_reference is None:
                continue
            delta_record = self.build_delta_record(strike_record=strike_record, live_reference=live_reference)
            if delta_record is not None:
                yield delta_record

    def current_live_delta(
        self,
        *,
        target_date: str,
        market_slug: str | None = None,
        condition_id: str | None = None,
        allow_verification_fallback: bool = False,
        reference_lookback_days: int = 1,
        now_ts_ns: int | None = None,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> dict[str, Any] | None:
        del allow_verification_fallback, reference_lookback_days
        strikes = list(
            self.iter_strike_records(
                target_date=target_date,
                market_slug=market_slug,
                condition_id=condition_id,
                finalized_only=finalized_only,
                tail_tolerant=tail_tolerant,
            )
        )
        if not strikes:
            return None
        latest_strike = max(strikes, key=lambda record: int(record.get("recv_ts_ns", 0)))
        live_reference = self._reference_reader.latest_asof(asof_ts_ns=now_ts_ns or utc_now_ns())
        if live_reference is None:
            return None
        return self.build_delta_record(strike_record=latest_strike, live_reference=live_reference)
