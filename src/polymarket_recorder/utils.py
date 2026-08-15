from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from binance_recorder.utils import isoformat_utc
from market_recorders.time_utils import (
    build_recursive_epoch_timestamp_fields,
    build_timestamp_fields,
)

_INCLUDE_PATTERNS = [
    re.compile(r"\bbtc\b", re.IGNORECASE),
    re.compile(r"bitcoin", re.IGNORECASE),
    re.compile(r"btc-updown-5m-\d+$", re.IGNORECASE),
    re.compile(r"up\s*or\s*down|up/down|higher\s+or\s+lower", re.IGNORECASE),
    re.compile(r"\b5m\b|5-minute|5 minute", re.IGNORECASE),
    re.compile(r"btc-up-or-down-5m", re.IGNORECASE),
]
_EXCLUDE_PATTERNS = [
    re.compile(r"\beth\b", re.IGNORECASE),
    re.compile(r"\bsol\b", re.IGNORECASE),
    re.compile(r"\bxrp\b", re.IGNORECASE),
]
_TS_FIELD_CANDIDATES = (
    "timestamp",
    "ts",
    "created_at",
    "createdAt",
    "updated_at",
    "updatedAt",
    "matchTime",
    "lastTradedAt",
    "time",
)
_DEFAULT_ACTIVE_PREOPEN_SECONDS = 150


def parse_json_array(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def normalize_outcome_side(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "up":
        return "up"
    if normalized == "down":
        return "down"
    return None


def normalize_slug(text: str) -> str:
    lowered = text.strip().lower()
    return re.sub(r"[^a-z0-9\-]+", "-", lowered).strip("-") or "market"


def short_identity(identity: str) -> str:
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return digest[:10]


def generate_target_slugs(
    *,
    now: datetime,
    prefix: str,
    interval_seconds: int,
    lookback_intervals: int,
    lookahead_intervals: int,
) -> list[str]:
    base_epoch = int(now.astimezone(UTC).timestamp())
    floored = base_epoch - (base_epoch % interval_seconds)
    return [
        f"{prefix}-{floored + (offset * interval_seconds)}"
        for offset in range(-lookback_intervals, lookahead_intervals + 1)
    ]


def _slug_epoch(slug: str) -> int | None:
    match = re.search(r"(\d{9,})$", slug)
    if not match:
        return None
    return int(match.group(1))


def _parse_iso_epoch(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).timestamp())
    except ValueError:
        return None


@dataclass(slots=True)
class SelectedMarket:
    market_id: str
    condition_id: str
    market_slug: str
    question: str
    asset_ids: list[str]
    outcomes: list[str]
    series_slug: str | None
    end_time_iso: str | None
    start_time_epoch: int | None
    selection_reason: str
    raw_market: dict[str, Any]

    @property
    def identity(self) -> str:
        return self.condition_id or self.market_id or self.market_slug or ",".join(self.asset_ids)

    @property
    def safe_slug(self) -> str:
        return normalize_slug(self.market_slug)

    @property
    def short_identity(self) -> str:
        return short_identity(self.identity)

    @property
    def outcome_map(self) -> dict[str, str]:
        return {
            asset_id: outcome
            for asset_id, outcome in zip(self.asset_ids, self.outcomes, strict=False)
        }

    @property
    def event_id(self) -> str | None:
        raw_event = self.raw_market.get("events", [None])[0] if isinstance(self.raw_market.get("events"), list) else None
        if isinstance(raw_event, dict) and raw_event.get("id") is not None:
            return str(raw_event["id"])
        if self.raw_market.get("eventId") is not None:
            return str(self.raw_market["eventId"])
        return None

    @property
    def event_slug(self) -> str | None:
        raw_event = self.raw_market.get("events", [None])[0] if isinstance(self.raw_market.get("events"), list) else None
        if isinstance(raw_event, dict) and raw_event.get("slug") is not None:
            return str(raw_event["slug"])
        return None


def market_open_epoch(market: SelectedMarket) -> int | None:
    return market.start_time_epoch or _slug_epoch(market.market_slug)


def market_close_epoch(market: SelectedMarket, *, interval_seconds: int) -> int | None:
    open_epoch = market_open_epoch(market)
    if open_epoch is not None:
        return open_epoch + interval_seconds
    return _parse_iso_epoch(market.end_time_iso)


def market_start_time_iso(market: SelectedMarket) -> str | None:
    for candidate in ("eventStartTime", "startTime"):
        value = market.raw_market.get(candidate)
        if isinstance(value, str) and value:
            return value
    events = market.raw_market.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            for candidate in ("startTime", "eventStartTime"):
                value = event.get(candidate)
                if isinstance(value, str) and value:
                    return value
    open_epoch = market_open_epoch(market)
    if open_epoch is None:
        return None
    return isoformat_utc(datetime.fromtimestamp(open_epoch, tz=UTC))


def market_active_window_fields(
    market: SelectedMarket,
    *,
    interval_seconds: int,
    preopen_seconds: int = _DEFAULT_ACTIVE_PREOPEN_SECONDS,
) -> dict[str, Any]:
    open_epoch = market_open_epoch(market)
    close_epoch = market_close_epoch(market, interval_seconds=interval_seconds)
    active_start_epoch = open_epoch - preopen_seconds if open_epoch is not None else None
    return {
        "market_active_window_start_s": active_start_epoch,
        "market_active_window_start_iso": isoformat_utc(datetime.fromtimestamp(active_start_epoch, tz=UTC))
        if active_start_epoch is not None
        else None,
        "market_open_s": open_epoch,
        "market_open_iso": isoformat_utc(datetime.fromtimestamp(open_epoch, tz=UTC)) if open_epoch is not None else None,
        "market_close_s": close_epoch,
        "market_close_iso": isoformat_utc(datetime.fromtimestamp(close_epoch, tz=UTC)) if close_epoch is not None else None,
        "market_interval_seconds": interval_seconds,
        "market_active_preopen_seconds": preopen_seconds,
    }


def market_is_active(
    market: SelectedMarket,
    *,
    now: datetime,
    interval_seconds: int,
    preopen_seconds: int = _DEFAULT_ACTIVE_PREOPEN_SECONDS,
) -> bool:
    open_epoch = market_open_epoch(market)
    close_epoch = market_close_epoch(market, interval_seconds=interval_seconds)
    if open_epoch is None or close_epoch is None:
        return False
    now_epoch = now.astimezone(UTC).timestamp()
    return open_epoch - preopen_seconds <= now_epoch < close_epoch


def market_has_expired(
    market: SelectedMarket,
    *,
    now: datetime,
    interval_seconds: int,
) -> bool:
    close_epoch = market_close_epoch(market, interval_seconds=interval_seconds)
    if close_epoch is None:
        return False
    return now.astimezone(UTC).timestamp() >= close_epoch


def sort_markets_by_open(markets: list[SelectedMarket]) -> list[SelectedMarket]:
    return sorted(
        markets,
        key=lambda market: (
            market_open_epoch(market) or 0,
            market.market_slug,
            market.identity,
        ),
    )


def newest_market(markets: list[SelectedMarket]) -> SelectedMarket | None:
    ordered = sort_markets_by_open(markets)
    return ordered[-1] if ordered else None


def _candidate_target_markets_with_scores(
    items: list[dict[str, Any]],
    *,
    target_slugs: list[str],
    now: datetime,
    stale_grace_seconds: float,
    allow_stale: bool = False,
) -> list[tuple[tuple[Any, ...], SelectedMarket]]:
    candidates: list[tuple[tuple[Any, ...], SelectedMarket]] = []
    target_slug_set = set(target_slugs)
    now_epoch = int(now.astimezone(UTC).timestamp())
    for item in items:
        slug = str(item.get("slug") or "")
        question = str(item.get("question") or "")
        haystack = f"{slug} {question} {item.get('seriesSlug') or ''}"
        if any(pattern.search(haystack) for pattern in _EXCLUDE_PATTERNS):
            continue
        include_match_count = sum(1 for pattern in _INCLUDE_PATTERNS if pattern.search(haystack))
        slug_exact = slug in target_slug_set
        series_exact = str(item.get("seriesSlug") or "").lower() == "btc-up-or-down-5m"
        has_bitcoin = bool(re.search(r"\bbtc\b|bitcoin", haystack, re.IGNORECASE))
        has_updown = bool(re.search(r"up|down|higher|lower", haystack, re.IGNORECASE))
        has_5m = bool(re.search(r"\b5m\b|5-minute|5 minute", haystack, re.IGNORECASE))
        if not ((has_bitcoin and has_updown and has_5m) or slug_exact or series_exact):
            continue
        asset_ids = [str(value) for value in parse_json_array(item.get("clobTokenIds")) if str(value)]
        if len(asset_ids) != 2:
            continue
        outcomes = [str(value) for value in parse_json_array(item.get("outcomes"))]
        if len(outcomes) != 2:
            outcomes = ["Up", "Down"]
        end_time_iso = item.get("endDate")
        if isinstance(end_time_iso, str):
            end_epoch = _parse_iso_epoch(end_time_iso)
            if not allow_stale and end_epoch is not None and end_epoch < now_epoch - stale_grace_seconds:
                continue
        start_epoch = _slug_epoch(slug)
        closest_start = -abs((start_epoch or 0) - now_epoch)
        score = (
            1 if slug_exact else 0,
            1 if series_exact else 0,
            1 if item.get("active") else 0,
            1 if not item.get("closed") else 0,
            include_match_count,
            closest_start,
            start_epoch or 0,
        )
        candidates.append(
            (
                score,
                SelectedMarket(
                    market_id=str(item.get("id") or ""),
                    condition_id=str(item.get("conditionId") or item.get("condition_id") or ""),
                    market_slug=slug,
                    question=question,
                    asset_ids=asset_ids,
                    outcomes=outcomes,
                    series_slug=item.get("seriesSlug"),
                    end_time_iso=end_time_iso if isinstance(end_time_iso, str) else None,
                    start_time_epoch=start_epoch,
                    selection_reason="slug_exact"
                    if slug_exact
                    else "series_exact"
                    if series_exact
                    else "heuristic_ranked",
                    raw_market=item,
                ),
            )
        )
    return candidates


def candidate_target_markets(
    items: list[dict[str, Any]],
    *,
    target_slugs: list[str],
    now: datetime,
    stale_grace_seconds: float,
    allow_stale: bool = False,
) -> list[SelectedMarket]:
    return sort_markets_by_open(
        [market for _, market in _candidate_target_markets_with_scores(
            items,
            target_slugs=target_slugs,
            now=now,
            stale_grace_seconds=stale_grace_seconds,
            allow_stale=allow_stale,
        )]
    )


def select_target_market(
    items: list[dict[str, Any]],
    *,
    target_slugs: list[str],
    now: datetime,
    stale_grace_seconds: float,
    allow_stale: bool = False,
) -> SelectedMarket | None:
    best: tuple[tuple[Any, ...], SelectedMarket] | None = None
    for score, market in _candidate_target_markets_with_scores(
        items,
        target_slugs=target_slugs,
        now=now,
        stale_grace_seconds=stale_grace_seconds,
        allow_stale=allow_stale,
    ):
        if best is None or score > best[0]:
            best = (score, market)
    return best[1] if best else None


def select_active_markets(
    items: list[dict[str, Any]],
    *,
    target_slugs: list[str],
    now: datetime,
    stale_grace_seconds: float,
    interval_seconds: int,
    preopen_seconds: int = _DEFAULT_ACTIVE_PREOPEN_SECONDS,
) -> list[SelectedMarket]:
    return [
        market
        for market in candidate_target_markets(
            items,
            target_slugs=target_slugs,
            now=now,
            stale_grace_seconds=stale_grace_seconds,
        )
        if market_is_active(
            market,
            now=now,
            interval_seconds=interval_seconds,
            preopen_seconds=preopen_seconds,
        )
    ]


def build_polymarket_source_timestamps(payload: dict[str, Any]) -> dict[str, Any]:
    timestamps: dict[str, Any] = {}
    for field_name in _TS_FIELD_CANDIDATES:
        if field_name in payload:
            timestamps.update(build_timestamp_fields("message_timestamp", payload[field_name]))
            break
    timestamps.update(
        {
            key: value
            for key, value in build_recursive_epoch_timestamp_fields(
                payload,
                candidate_field_names=_TS_FIELD_CANDIDATES,
            ).items()
            if key not in timestamps
        }
    )
    return timestamps


def polymarket_source_ts_ns_from_timestamps(
    source_timestamps: dict[str, Any] | None,
    *,
    fallback_recv_ns: int | None = None,
) -> int | None:
    timestamps = source_timestamps or {}
    for key in ("timestamp_ns", "message_timestamp_ns"):
        try:
            value = int(timestamps.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    for key in ("timestamp_ms", "message_timestamp_ms"):
        try:
            value = int(timestamps.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value * 1_000_000
    return fallback_recv_ns


def polymarket_source_clock_fields(
    source_timestamps: dict[str, Any] | None,
    *,
    recv_ts_ns: int,
) -> dict[str, Any]:
    source_ts_ns = polymarket_source_ts_ns_from_timestamps(source_timestamps, fallback_recv_ns=None)
    if source_ts_ns is None:
        return {
            "source_ts_ns": None,
            "source_lag_ms": None,
            "source_lag_s": None,
            "source_clock": "missing",
        }
    source_lag_ms = round((int(recv_ts_ns) - int(source_ts_ns)) / 1_000_000, 3)
    return {
        "source_ts_ns": int(source_ts_ns),
        "source_lag_ms": source_lag_ms,
        "source_lag_s": round(source_lag_ms / 1000.0, 6),
        "source_clock": "payload",
    }


def session_basename(now: datetime, market: SelectedMarket) -> str:
    return f"{now.astimezone(UTC).strftime('%H-%M-%S')}__{market.safe_slug}__{market.short_identity}"


def session_common_fields(market: SelectedMarket) -> dict[str, Any]:
    return {
        "market_family": "btc_updown_5m",
        "selected_market_id": market.condition_id or market.market_id or market.market_slug,
        "selected_condition_id": market.condition_id,
        "selected_event_id": market.event_id,
        "selected_event_slug": market.event_slug,
        "selected_market_slug": market.market_slug,
        "selected_asset_ids": market.asset_ids,
        "selected_outcomes": market.outcomes,
        "selection_reason": market.selection_reason,
        **market_active_window_fields(
            market,
            interval_seconds=300,
            preopen_seconds=_DEFAULT_ACTIVE_PREOPEN_SECONDS,
        ),
    }


def detection_metadata(keys: dict[str, str]) -> dict[str, Any]:
    return {
        "api_keys_detected": any(
            alias in keys
            for alias in ("api key", "api_key", "key", "api secret", "api_secret", "secret", "api passphrase")
        ),
        "detected_key_aliases": sorted(
            alias
            for alias in ("api key", "api_key", "key", "api secret", "api_secret", "secret", "api passphrase")
            if alias in keys
        ),
    }
