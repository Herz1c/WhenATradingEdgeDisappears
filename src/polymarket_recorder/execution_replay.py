from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import hashlib
import math
import numpy as np
import pandas as pd

from .l2_event_loader import (
    DEFAULT_CANONICAL_L2_EVENTS_PATH,
    DEFAULT_QUALITY_REGISTRY_PATH,
    L2EventLoadResult,
    load_canonical_l2_events_for_date,
    load_l2_events,
    write_canonical_l2_events_for_date,
)

SECOND_NS = 1_000_000_000
DEFAULT_STALE_BOOK_NS = 30 * SECOND_NS


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _safe_tick_size(value: Any, default: float = 0.01) -> float:
    parsed = _finite_float(value)
    if parsed is None or parsed <= 0:
        return default
    return parsed


def _event_type_priority(value: Any) -> int:
    event_type = str(value or "").lower()
    priorities = {
        "book_state": 10,
        "top_of_book": 20,
        "trade": 30,
        "tick_update": 40,
        "other": 90,
    }
    return priorities.get(event_type, 80)


def _stable_event_hash(row: pd.Series) -> str:
    parts = [
        str(row.get("market_slug") or ""),
        str(row.get("token_asset_id") or ""),
        str(row.get("token_side_normalized") or ""),
        str(row.get("recv_ts_ns") or ""),
        str(row.get("event_type") or ""),
        str(row.get("book_state_seq") or ""),
        str(row.get("best_bid") or ""),
        str(row.get("best_ask") or ""),
        str(row.get("last_trade_price") or ""),
        str(row.get("trade_size") or ""),
        str(row.get("trade_side") or ""),
        str(row.get("source_file_path") or ""),
        str(row.get("source_file_index") or ""),
    ]
    return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=12).hexdigest()


@dataclass(frozen=True, slots=True)
class BookState:
    market_slug: str
    token_asset_id: str
    token_side_normalized: str
    recv_ts_ns: int
    best_bid: float | None
    best_ask: float | None
    mid: float | None
    spread: float | None
    bid_depth_total: float | None
    ask_depth_total: float | None
    tick_size: float
    stale_s: float
    one_sided_book: bool
    crossed_book: bool
    valid: bool


class TokenReplay:
    """Causal as-of view over one market/token L2 event stream."""

    def __init__(
        self,
        events: pd.DataFrame,
        *,
        stale_book_ns: int = DEFAULT_STALE_BOOK_NS,
        default_tick_size: float = 0.01,
    ) -> None:
        required = {"market_slug", "token_asset_id", "token_side_normalized", "recv_ts_ns", "event_type"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"TokenReplay missing required columns: {sorted(missing)}")
        if events.empty:
            raise ValueError("TokenReplay requires at least one event")

        frame = events.copy()
        frame["recv_ts_ns"] = frame["recv_ts_ns"].astype("int64")
        if "book_state_seq" not in frame.columns:
            frame["book_state_seq"] = np.nan
        if "source_file_path" not in frame.columns:
            frame["source_file_path"] = ""
        if "source_file_index" not in frame.columns:
            frame["source_file_index"] = 0
        frame["_event_type_priority"] = frame["event_type"].map(_event_type_priority).astype("int16")
        frame["_book_state_seq_sort"] = pd.to_numeric(frame["book_state_seq"], errors="coerce").fillna(-1).astype("int64")
        frame["_source_file_index_sort"] = pd.to_numeric(frame["source_file_index"], errors="coerce").fillna(0).astype("int64")
        frame["_event_sort_hash"] = frame.apply(_stable_event_hash, axis=1)
        frame = frame.sort_values(
            [
                "recv_ts_ns",
                "_event_type_priority",
                "_book_state_seq_sort",
                "source_file_path",
                "_source_file_index_sort",
                "_event_sort_hash",
            ],
            kind="stable",
        ).reset_index(drop=True)
        self.events = frame
        self.market_slug = str(frame["market_slug"].iloc[0])
        self.token_asset_id = str(frame["token_asset_id"].iloc[0])
        self.token_side_normalized = str(frame["token_side_normalized"].iloc[0]).upper()
        self.stale_book_ns = int(stale_book_ns)
        self.default_tick_size = float(default_tick_size)
        self._event_ts = frame["recv_ts_ns"].to_numpy(dtype=np.int64)

        book_mask = frame["best_bid"].notna() & frame["best_ask"].notna()
        self.book_events = frame.loc[book_mask].copy().reset_index(drop=True)
        self._book_ts = (
            self.book_events["recv_ts_ns"].to_numpy(dtype=np.int64)
            if not self.book_events.empty
            else np.array([], dtype=np.int64)
        )

        trade_mask = frame["event_type"].astype(str).eq("trade") & frame["last_trade_price"].notna()
        self.trade_events = frame.loc[trade_mask].copy().reset_index(drop=True)
        self._trade_ts = (
            self.trade_events["recv_ts_ns"].to_numpy(dtype=np.int64)
            if not self.trade_events.empty
            else np.array([], dtype=np.int64)
        )
        if self.trade_events.empty:
            self._trade_prices = np.array([], dtype=np.float64)
            self._trade_sizes = np.array([], dtype=np.float64)
            self._trade_side_codes = np.array([], dtype=np.int8)
        else:
            self._trade_prices = pd.to_numeric(self.trade_events["last_trade_price"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            self._trade_sizes = (
                pd.to_numeric(self.trade_events["trade_size"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            )
            side_values = self.trade_events["trade_side"].fillna("").astype(str).str.lower()
            self._trade_side_codes = np.select(
                [side_values.eq("buy"), side_values.eq("sell"), side_values.eq("unknown")],
                [1, -1, 0],
                default=9,
            ).astype(np.int8)

    def state_before(self, ts_ns: int) -> BookState:
        """Return the latest full book state with recv_ts_ns strictly before ts_ns."""

        if self.book_events.empty:
            return BookState(
                market_slug=self.market_slug,
                token_asset_id=self.token_asset_id,
                token_side_normalized=self.token_side_normalized,
                recv_ts_ns=0,
                best_bid=None,
                best_ask=None,
                mid=None,
                spread=None,
                bid_depth_total=None,
                ask_depth_total=None,
                tick_size=self.default_tick_size,
                stale_s=math.inf,
                one_sided_book=True,
                crossed_book=False,
                valid=False,
            )
        idx = bisect_left(self._book_ts, int(ts_ns)) - 1
        if idx < 0:
            return BookState(
                market_slug=self.market_slug,
                token_asset_id=self.token_asset_id,
                token_side_normalized=self.token_side_normalized,
                recv_ts_ns=0,
                best_bid=None,
                best_ask=None,
                mid=None,
                spread=None,
                bid_depth_total=None,
                ask_depth_total=None,
                tick_size=self.default_tick_size,
                stale_s=math.inf,
                one_sided_book=True,
                crossed_book=False,
                valid=False,
            )
        return self._state_from_row(self.book_events.iloc[idx], int(ts_ns))

    def state_at_or_before(self, ts_ns: int) -> BookState:
        """Return the latest full book state with recv_ts_ns <= ts_ns."""

        if self.book_events.empty:
            return self.state_before(ts_ns)
        idx = bisect_right(self._book_ts, int(ts_ns)) - 1
        if idx < 0:
            return self.state_before(ts_ns)
        return self._state_from_row(self.book_events.iloc[idx], int(ts_ns))

    def _state_from_row(self, row: pd.Series, asof_ts_ns: int) -> BookState:
        best_bid = _finite_float(row.get("best_bid"))
        best_ask = _finite_float(row.get("best_ask"))
        bid_depth = _finite_float(row.get("bid_depth_total"))
        ask_depth = _finite_float(row.get("ask_depth_total"))
        mid = _finite_float(row.get("mid"))
        if mid is None and best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        spread = _finite_float(row.get("spread"))
        if spread is None and best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
        recv_ts_ns = int(row["recv_ts_ns"])
        stale_s = max(0.0, (int(asof_ts_ns) - recv_ts_ns) / SECOND_NS)
        crossed = best_bid is not None and best_ask is not None and best_bid >= best_ask
        one_sided = (
            best_bid is None
            or best_ask is None
            or bid_depth is None
            or ask_depth is None
            or bid_depth <= 0
            or ask_depth <= 0
        )
        return BookState(
            market_slug=str(row.get("market_slug") or self.market_slug),
            token_asset_id=str(row.get("token_asset_id") or self.token_asset_id),
            token_side_normalized=str(row.get("token_side_normalized") or self.token_side_normalized).upper(),
            recv_ts_ns=recv_ts_ns,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=mid,
            spread=spread,
            bid_depth_total=bid_depth,
            ask_depth_total=ask_depth,
            tick_size=_safe_tick_size(row.get("tick_size"), self.default_tick_size),
            stale_s=stale_s,
            one_sided_book=one_sided,
            crossed_book=crossed,
            valid=best_bid is not None and best_ask is not None and mid is not None and not crossed,
        )

    def events_between(self, start_exclusive_ns: int, end_inclusive_ns: int) -> pd.DataFrame:
        left = bisect_right(self._event_ts, int(start_exclusive_ns))
        right = bisect_right(self._event_ts, int(end_inclusive_ns))
        return self.events.iloc[left:right]

    def trades_between(self, start_exclusive_ns: int, end_inclusive_ns: int) -> pd.DataFrame:
        if self.trade_events.empty:
            return self.trade_events.copy()
        left = bisect_right(self._trade_ts, int(start_exclusive_ns))
        right = bisect_right(self._trade_ts, int(end_inclusive_ns))
        return self.trade_events.iloc[left:right]

    def trades_between_open(self, start_exclusive_ns: int, end_exclusive_ns: int) -> pd.DataFrame:
        if self.trade_events.empty or int(end_exclusive_ns) <= int(start_exclusive_ns):
            return self.trade_events.copy().iloc[0:0]
        left = bisect_right(self._trade_ts, int(start_exclusive_ns))
        right = bisect_left(self._trade_ts, int(end_exclusive_ns))
        return self.trade_events.iloc[left:right]

    def mid_at_or_before(self, ts_ns: int) -> float | None:
        state = self.state_at_or_before(int(ts_ns))
        return state.mid if state.valid or state.mid is not None else None

    def terminal_mid_before_close(self, market_close_ts_ns: int) -> float | None:
        state = self.state_before(int(market_close_ts_ns))
        return state.mid if state.valid or state.mid is not None else None


class ExecutionReplayIndex:
    """Token replay registry keyed by (market_slug, token_asset_id)."""

    def __init__(self, token_replays: dict[tuple[str, str], TokenReplay]) -> None:
        self._token_replays = token_replays

    @classmethod
    def from_events(
        cls,
        events: pd.DataFrame,
        *,
        stale_book_ns: int = DEFAULT_STALE_BOOK_NS,
        default_tick_size: float = 0.01,
    ) -> "ExecutionReplayIndex":
        if events.empty:
            return cls({})
        required = {"market_slug", "token_asset_id", "recv_ts_ns"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"ExecutionReplayIndex missing required event columns: {sorted(missing)}")
        token_replays: dict[tuple[str, str], TokenReplay] = {}
        for (market_slug, token_asset_id), group in events.groupby(["market_slug", "token_asset_id"], sort=False):
            token_replays[(str(market_slug), str(token_asset_id))] = TokenReplay(
                group,
                stale_book_ns=stale_book_ns,
                default_tick_size=default_tick_size,
            )
        return cls(token_replays)

    def get(self, market_slug: str, token_asset_id: str) -> TokenReplay | None:
        return self._token_replays.get((str(market_slug), str(token_asset_id)))

    def require(self, market_slug: str, token_asset_id: str) -> TokenReplay:
        replay = self.get(market_slug, token_asset_id)
        if replay is None:
            raise KeyError(f"No replay for market={market_slug} token={token_asset_id}")
        return replay

    @property
    def token_count(self) -> int:
        return len(self._token_replays)

    @property
    def market_count(self) -> int:
        return len({market_slug for market_slug, _ in self._token_replays})


@dataclass(frozen=True, slots=True)
class ExecutionReplayLoadResult:
    replay_index: ExecutionReplayIndex
    events: pd.DataFrame
    exclusion_summary: dict[str, Any]


def load_execution_replay_for_date(
    *,
    root: Path,
    target_date: date,
    policy_path: Path = Path("config/dataset_policy_v1.yaml"),
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
    canonical_l2_path: Path = DEFAULT_CANONICAL_L2_EVENTS_PATH,
    prefer_canonical: bool = True,
    materialize_canonical: bool = False,
    market_slug: str | None = None,
    market_slugs: set[str] | None = None,
) -> ExecutionReplayLoadResult:
    if materialize_canonical:
        write_canonical_l2_events_for_date(
            root=root,
            target_date=target_date,
            output_path=canonical_l2_path,
            policy_path=policy_path,
            quality_registry_path=quality_registry_path,
        )
    if prefer_canonical:
        canonical_events = load_canonical_l2_events_for_date(
            root=root,
            target_date=target_date,
            canonical_path=canonical_l2_path,
            market_slug=market_slug,
            market_slugs=market_slugs,
        )
        if canonical_events is not None:
            return ExecutionReplayLoadResult(
                replay_index=ExecutionReplayIndex.from_events(canonical_events),
                events=canonical_events,
                exclusion_summary={"polymarket_l2": {"canonical_l2_events_v1": int(len(canonical_events))}},
            )
    load_result: L2EventLoadResult = load_l2_events(
        root=root,
        target_date=target_date,
        policy_path=policy_path,
        quality_registry_path=quality_registry_path,
        market_slug=market_slug,
        market_slugs=market_slugs,
    )
    replay_index = ExecutionReplayIndex.from_events(load_result.events)
    return ExecutionReplayLoadResult(
        replay_index=replay_index,
        events=load_result.events,
        exclusion_summary=load_result.exclusion_summary,
    )
