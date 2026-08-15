from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from binance_recorder.utils import ns_to_iso
from market_recorders.unified_reader import UnifiedRawReader
from polymarket_recorder.rtds_reader import PolymarketRtdsRawReader
from chainlink_recorder.reader import ChainlinkPublicDelayedRawReader

from .datastreams_decode import decode_datastreams_report


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_price(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _direct_chainlink_ts_ns(record: dict[str, Any]) -> int | None:
    chainlink_ts_ns = _to_int(record.get("chainlink_ts_ns"))
    if chainlink_ts_ns is not None:
        return chainlink_ts_ns
    chainlink_ts = _to_int(record.get("chainlink_ts"))
    if chainlink_ts is not None:
        return chainlink_ts * 1_000_000_000
    extracted = record.get("extracted", {}) if isinstance(record.get("extracted"), dict) else {}
    direct_seconds = _coalesce(
        extracted.get("observations_timestamp_s"),
        extracted.get("valid_from_timestamp_s"),
        extracted.get("timestamp_s"),
    )
    direct_seconds = _to_int(direct_seconds)
    if direct_seconds is not None:
        return direct_seconds * 1_000_000_000
    return _to_int(_coalesce(record.get("reference_ts_ns"), record.get("recv_ts_ns")))


def _rtds_chainlink_ts_ns(record: dict[str, Any]) -> int | None:
    chainlink_ts_ms = _to_int(record.get("chainlink_ts_ms"))
    if chainlink_ts_ms is not None:
        return chainlink_ts_ms * 1_000_000
    source_timestamps = record.get("source_timestamps", {}) if isinstance(record.get("source_timestamps"), dict) else {}
    price_timestamp_ms = _to_int(source_timestamps.get("price_timestamp_ms"))
    if price_timestamp_ms is not None:
        return price_timestamp_ms * 1_000_000
    message_timestamp_ms = _to_int(source_timestamps.get("message_timestamp_ms"))
    if message_timestamp_ms is not None:
        return message_timestamp_ms * 1_000_000
    return _to_int(_coalesce(record.get("reference_ts_ns"), record.get("recv_ts_ns")))


def _normalized_record(
    *,
    record_type: str | None,
    recv_ts_ns: int | None,
    recv_ts_iso: str | None,
    recv_ts_source: str | None,
    chainlink_ts_ns: int | None,
    btc_usd_price: str | None,
    source: str,
) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": recv_ts_iso,
        "recv_ts_source": recv_ts_source,
        "chainlink_ts_ns": chainlink_ts_ns,
        "chainlink_ts_iso": ns_to_iso(chainlink_ts_ns) if chainlink_ts_ns is not None else None,
        "btc_usd_price": btc_usd_price,
        "source": source,
    }


def build_direct_live_record(
    payload: dict[str, Any],
    *,
    backend: str,
    symbol: str,
    recv_ts_ns: int,
    recv_ts_iso: str,
    recv_ts_source: str,
    endpoint_url: str,
    capture_timing: dict[str, Any],
    connection_id: str | None = None,
    local_msg_seq: int | None = None,
) -> dict[str, Any] | None:
    del symbol, endpoint_url, capture_timing, connection_id, local_msg_seq
    extracted = decode_datastreams_report(payload, recv_ts_ns=recv_ts_ns)
    btc_usd_price = _to_price(_coalesce(extracted.get("live_price"), extracted.get("price"), extracted.get("mid_price")))
    chainlink_ts = _to_int(
        _coalesce(
            extracted.get("observations_timestamp_s"),
            extracted.get("valid_from_timestamp_s"),
            extracted.get("timestamp_s"),
        )
    )
    if chainlink_ts is None and extracted.get("reference_ts_ns") is not None:
        chainlink_ts = int(int(extracted["reference_ts_ns"]) // 1_000_000_000)
    if btc_usd_price is None or chainlink_ts is None:
        return None
    source = "chainlink_datastreams_ws" if backend == "datastreams_ws" else "chainlink_datastreams_api"
    return {
        "record_type": "chainlink_live_reference",
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": recv_ts_iso,
        "recv_ts_source": recv_ts_source,
        "chainlink_ts": chainlink_ts,
        "chainlink_ts_iso": ns_to_iso(chainlink_ts * 1_000_000_000),
        "btc_usd_price": btc_usd_price,
        "source": source,
    }


def normalize_direct_live_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("source") in {"chainlink_datastreams_ws", "chainlink_datastreams_api"} and record.get("btc_usd_price") is not None:
        return {
            key: value
            for key, value in _normalized_record(
                record_type=str(record.get("record_type") or "chainlink_live_reference"),
                recv_ts_ns=_to_int(record.get("recv_ts_ns")),
                recv_ts_iso=record.get("recv_ts_iso"),
                recv_ts_source=record.get("recv_ts_source"),
                chainlink_ts_ns=_direct_chainlink_ts_ns(record),
                btc_usd_price=_to_price(record.get("btc_usd_price")),
                source=str(record.get("source")),
            ).items()
            if value is not None
        }

    extracted = record.get("extracted", {}) if isinstance(record.get("extracted"), dict) else {}
    return {
        key: value
        for key, value in _normalized_record(
            record_type=str(record.get("record_type") or "chainlink_live_reference"),
            recv_ts_ns=_to_int(record.get("recv_ts_ns")),
            recv_ts_iso=record.get("recv_ts_iso"),
            recv_ts_source=record.get("recv_ts_source"),
            chainlink_ts_ns=_direct_chainlink_ts_ns(record),
            btc_usd_price=_to_price(
                _coalesce(record.get("btc_usd_price"), record.get("live_price"), extracted.get("live_price"), extracted.get("price"))
            ),
            source=str(record.get("source") or "chainlink_datastreams_ws"),
        ).items()
        if value is not None
    }


def normalize_rtds_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("source") == "polymarket_rtds_chainlink" and record.get("btc_usd_price") is not None:
        return {
            key: value
            for key, value in _normalized_record(
                record_type=str(record.get("record_type") or "chainlink_live_reference"),
                recv_ts_ns=_to_int(record.get("recv_ts_ns")),
                recv_ts_iso=record.get("recv_ts_iso"),
                recv_ts_source=record.get("recv_ts_source"),
                chainlink_ts_ns=_rtds_chainlink_ts_ns(record),
                btc_usd_price=_to_price(record.get("btc_usd_price")),
                source="polymarket_rtds_chainlink",
            ).items()
            if value is not None
        }

    extracted = record.get("extracted", {}) if isinstance(record.get("extracted"), dict) else {}
    return {
        key: value
        for key, value in _normalized_record(
            record_type=str(record.get("record_type") or "chainlink_live_reference"),
            recv_ts_ns=_to_int(record.get("recv_ts_ns")),
            recv_ts_iso=record.get("recv_ts_iso"),
            recv_ts_source=record.get("recv_ts_source"),
            chainlink_ts_ns=_rtds_chainlink_ts_ns(record),
            btc_usd_price=_to_price(_coalesce(record.get("btc_usd_price"), record.get("live_price"), extracted.get("value"))),
            source="polymarket_rtds_chainlink",
        ).items()
        if value is not None
    }


def normalize_onchain_record(record: dict[str, Any]) -> dict[str, Any]:
    round_data = record.get("round_data", {}) if isinstance(record.get("round_data"), dict) else {}
    block = record.get("block", {}) if isinstance(record.get("block"), dict) else {}
    chainlink_ts_s = _to_int(_coalesce(round_data.get("updated_at_s"), block.get("timestamp_s")))
    chainlink_ts_ns = chainlink_ts_s * 1_000_000_000 if chainlink_ts_s is not None else None
    return {
        key: value
        for key, value in _normalized_record(
            record_type="chainlink_verification_reference",
            recv_ts_ns=_to_int(record.get("recv_ts_ns")),
            recv_ts_iso=record.get("recv_ts_iso"),
            recv_ts_source=record.get("recv_ts_source"),
            chainlink_ts_ns=chainlink_ts_ns,
            btc_usd_price=_to_price(round_data.get("answer")),
            source="chainlink_onchain_verification",
        ).items()
        if value is not None
    }


def normalize_page_record(record: dict[str, Any]) -> dict[str, Any]:
    extracted = record.get("extracted", {}) if isinstance(record.get("extracted"), dict) else {}
    chainlink_ts_ns = _to_int(_coalesce(extracted.get("reference_ts_ns"), record.get("recv_ts_ns")))
    return {
        key: value
        for key, value in _normalized_record(
            record_type="chainlink_verification_reference",
            recv_ts_ns=_to_int(record.get("recv_ts_ns")),
            recv_ts_iso=record.get("recv_ts_iso"),
            recv_ts_source=record.get("recv_ts_source"),
            chainlink_ts_ns=chainlink_ts_ns,
            btc_usd_price=_to_price(_coalesce(extracted.get("btc_usd_price"), extracted.get("live_price"), extracted.get("mid_price"))),
            source="chainlink_page_verification",
        ).items()
        if value is not None
    }


def normalize_public_delayed_record(record: dict[str, Any]) -> dict[str, Any]:
    display_ts = _to_int(record.get("chainlink_display_ts"))
    chainlink_ts_ns = None
    if display_ts is not None:
        chainlink_ts_ns = display_ts * 1_000_000_000 if abs(display_ts) < 10_000_000_000 else display_ts
    return {
        key: value
        for key, value in _normalized_record(
            record_type=str(record.get("record_type") or "chainlink_public_delayed_observation"),
            recv_ts_ns=_to_int(record.get("recv_ts_ns")),
            recv_ts_iso=record.get("recv_ts_iso"),
            recv_ts_source=record.get("recv_ts_source"),
            chainlink_ts_ns=chainlink_ts_ns,
            btc_usd_price=_to_price(record.get("btc_usd_price")),
            source="chainlink_public_delayed",
        ).items()
        if value is not None
    }


class ChainlinkLiveReferenceReader:
    def __init__(
        self,
        *,
        root: Path,
        symbol: str = "BTCUSD",
        rtds_symbol: str = "btc/usd",
    ) -> None:
        self.root = root
        self.symbol = symbol
        self.rtds_symbol = rtds_symbol
        self._raw_reader = UnifiedRawReader(root=root)
        self._rtds_reader = PolymarketRtdsRawReader(root=root, symbol=rtds_symbol)
        self._public_delayed_reader = ChainlinkPublicDelayedRawReader(root=root, symbol=symbol)

    def iter_direct_live(
        self,
        *,
        target_date: date | str,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict[str, Any]]:
        relative_prefix = Path("chainlink") / "datastreams_ws" / self.symbol
        for record in self._raw_reader.iter_date(
            relative_prefix=relative_prefix,
            target_date=target_date,
            kind="live",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            normalized = normalize_direct_live_record(record)
            if normalized.get("btc_usd_price") is not None:
                yield normalized

    def iter_rtds_proxy(
        self,
        *,
        target_date: date | str,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict[str, Any]]:
        for record in self._rtds_reader.iter_kind(
            target_date=str(target_date),
            kind="ws",
            topic="crypto_prices_chainlink",
            message_type="update",
            symbol=self.rtds_symbol,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            normalized = normalize_rtds_record(record)
            if normalized.get("btc_usd_price") is not None:
                yield normalized

    def iter_verification_fallback(
        self,
        *,
        target_date: date | str,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict[str, Any]]:
        onchain_prefix = Path("chainlink") / "onchain" / self.symbol
        for record in self._raw_reader.iter_date(
            relative_prefix=onchain_prefix,
            target_date=target_date,
            kind="onchain",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            normalized = normalize_onchain_record(record)
            if normalized.get("btc_usd_price") is not None:
                yield normalized
        for record in self._public_delayed_reader.iter_kind(
            target_date=str(target_date),
            kind="page",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            normalized = normalize_public_delayed_record(record)
            if normalized.get("btc_usd_price") is not None:
                yield normalized
        page_prefix = Path("chainlink") / "public_page" / self.symbol
        for record in self._raw_reader.iter_date(
            relative_prefix=page_prefix,
            target_date=target_date,
            kind="page",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            normalized = normalize_page_record(record)
            if normalized.get("btc_usd_price") is not None:
                yield normalized

    def iter_best_available(
        self,
        *,
        target_date: date | str,
        allow_verification_fallback: bool = False,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict[str, Any]]:
        direct = list(
            self.iter_direct_live(
                target_date=target_date,
                finalized_only=finalized_only,
                tail_tolerant=tail_tolerant,
            )
        )
        if direct:
            yield from direct
            return
        rtds = list(
            self.iter_rtds_proxy(
                target_date=target_date,
                finalized_only=finalized_only,
                tail_tolerant=tail_tolerant,
            )
        )
        if rtds:
            yield from rtds
            return
        if allow_verification_fallback:
            yield from self.iter_verification_fallback(
                target_date=target_date,
                finalized_only=finalized_only,
                tail_tolerant=tail_tolerant,
            )

    @staticmethod
    def _recv_ts_ns(record: dict[str, Any]) -> int | None:
        return _to_int(record.get("recv_ts_ns"))

    @staticmethod
    def _chainlink_ts_ns(record: dict[str, Any]) -> int | None:
        return _to_int(record.get("chainlink_ts_ns"))

    @staticmethod
    def _candidate_dates(asof_ts_ns: int, *, lookback_days: int) -> list[date]:
        anchor = datetime.fromtimestamp(asof_ts_ns / 1_000_000_000, tz=UTC).date()
        return [anchor - timedelta(days=offset) for offset in range(lookback_days + 1)]

    @staticmethod
    def _latest_before(records: Iterable[dict[str, Any]], *, asof_ts_ns: int) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_rank: tuple[int, int] | None = None
        for record in records:
            recv_ts_ns = ChainlinkLiveReferenceReader._recv_ts_ns(record)
            if recv_ts_ns is None or recv_ts_ns > asof_ts_ns:
                continue
            chainlink_ts_ns = ChainlinkLiveReferenceReader._chainlink_ts_ns(record) or -1
            rank = (recv_ts_ns, chainlink_ts_ns)
            if best is None or best_rank is None or rank > best_rank:
                best = record
                best_rank = rank
        return best

    def latest_asof(
        self,
        *,
        asof_ts_ns: int,
        allow_verification_fallback: bool = False,
        lookback_days: int = 1,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> dict[str, Any] | None:
        candidate_dates = self._candidate_dates(asof_ts_ns, lookback_days=lookback_days)
        sources = [
            lambda target_date: self.iter_direct_live(
                target_date=target_date,
                finalized_only=finalized_only,
                tail_tolerant=tail_tolerant,
            ),
            lambda target_date: self.iter_rtds_proxy(
                target_date=target_date,
                finalized_only=finalized_only,
                tail_tolerant=tail_tolerant,
            ),
        ]
        if allow_verification_fallback:
            sources.append(
                lambda target_date: self.iter_verification_fallback(
                    target_date=target_date,
                    finalized_only=finalized_only,
                    tail_tolerant=tail_tolerant,
                )
            )
        for source_iter in sources:
            best = self._latest_before(
                (
                    record
                    for target_date in candidate_dates
                    for record in source_iter(target_date.isoformat())
                ),
                asof_ts_ns=asof_ts_ns,
            )
            if best is not None:
                return best
        return None
