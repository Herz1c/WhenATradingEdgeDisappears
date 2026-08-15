"""Streaming feature-pipeline engine — the production decision path.

Architecture (mirrors what the true live bot will do, except the event
source is swappable):

    [event_source] --> [StreamingFeatureEngine]
        |                  |
        |                  |-- per-market TokenL2Index (up + down)
        |                  |-- per-market RestContextIndex
        |                  |-- global LiveReferenceIndex
        |                  |-- per-market snapshot grid pointer
        |                  |
        |                  +--> at each dense_close grid time, calls
        |                       dataset_factory._build_snapshot_row()
        |                       on the *current* indices and emits the
        |                       resulting feature row.
        |
        +-- file_replay  : reads recorder .jsonl.zst in arrival order
        +-- (TBD) live_ws: subscribes to Polymarket / Binance / etc.

Key design choices:

  1. The indices we maintain are bit-identical in *shape* to the ones
     dataset_factory._load_l2_indices_for_day() produces. They're just
     built incrementally instead of all-at-once.

  2. We reuse dataset_factory._build_snapshot_row() verbatim. This
     guarantees that if the indices match, the feature output matches
     the offline dataset row-for-row.

  3. Snapshot timing follows the exact dense_close schedule:
       250 ms cadence from T-60s to T-10s, then 100 ms from T-10s to T.
     We "fire" a snapshot the first time we observe an event with
     recv_ts_ns >= next_snapshot_ts. This is the closest you can get
     to "live snapshot at exactly time T" — sub-tick lag is bounded by
     the WS event cadence (typically <100 ms in practice).

  4. The MarketLiveWindow (used for above/below-strike features) is
     built *truncated* — only seconds in [market_open_s, snapshot_second]
     are populated. The dataset_factory builds it over the full
     [market_open_s, market_close_s) but the future-leaking columns
     produced from that (future_flip, terminal_*) are stripped by
     clean_features anyway, so the model-input features are identical.

  5. For LIVE markets we don't have resolved_side at snapshot time. The
     `_build_snapshot_row` function uses it for `winning_mid` and
     `hold_to_close_edge_vs_mid` — both leakage-only columns. We pass
     resolved_side="unknown" which makes winning_mid None and the
     downstream columns None — same outcome as the leakage strip.
"""
from __future__ import annotations

import io
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from heapq import merge
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd

from polymarket_recorder.dataset_factory import (
    MILLISECOND_NS, SECOND_NS,
    LiveReferenceIndex, MarketLabel, MarketLiveWindow,
    RestContextIndex, RestContextState, TokenBookState, TokenL2Index,
    _build_snapshot_row, _mid, _to_float, _to_int,
)
from market_recorders.unified_reader import UnifiedRawReader
from market_recorders.dataset_policy import (
    DEFAULT_DATASET_POLICY_PATH, load_dataset_policy, training_metadata_eligibility,
)


# ──────────────────────────────────────────────────────────────────────────────
# Incremental index helpers (append-only, same shape as offline)
# ──────────────────────────────────────────────────────────────────────────────
def _empty_l2_index() -> TokenL2Index:
    return TokenL2Index(
        state_times=[], states=[],
        event_times_by_type={},
        trade_times=[],
        trade_signed_size_prefix=[0.0],
        trade_abs_size_prefix=[0.0],
        trade_count_prefix=[0],
    )


def _empty_rest_index() -> RestContextIndex:
    return RestContextIndex(times=[], states=[])


def _empty_live_reference() -> LiveReferenceIndex:
    return LiveReferenceIndex([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])


def _append_l2_record(idx: TokenL2Index, record: dict) -> None:
    """Mirror of the per-record logic in _load_l2_indices_for_day."""
    if record.get("token_modeling_safe") is not True:
        return
    recv_ts_ns = _to_int(record.get("recv_ts_ns"))
    if recv_ts_ns is None:
        return

    event_type = str(record.get("event_type") or "").strip()
    if event_type:
        idx.event_times_by_type.setdefault(event_type, []).append(recv_ts_ns)

    if record.get("record_type") == "trade_execution":
        size = _to_float(record.get("size")) or 0.0
        side = str(record.get("side") or "").upper()
        signed_size = size if side == "BUY" else -size if side == "SELL" else 0.0
        idx.trade_times.append(recv_ts_ns)
        idx.trade_signed_size_prefix.append(idx.trade_signed_size_prefix[-1] + signed_size)
        idx.trade_abs_size_prefix.append(idx.trade_abs_size_prefix[-1] + abs(size))
        idx.trade_count_prefix.append(idx.trade_count_prefix[-1] + 1)

    if record.get("record_type") != "l2_book_state":
        return

    best_bid = _to_float(record.get("best_bid"))
    best_ask = _to_float(record.get("best_ask"))
    bid_depth_total = _to_float((record.get("depth_totals") or {}).get("bid_size_total"))
    ask_depth_total = _to_float((record.get("depth_totals") or {}).get("ask_size_total"))
    state = TokenBookState(
        recv_ts_ns=recv_ts_ns,
        best_bid=best_bid,
        best_ask=best_ask,
        mid=_mid(best_bid, best_ask),
        spread=_to_float(record.get("spread")),
        bid_depth_total=bid_depth_total,
        ask_depth_total=ask_depth_total,
        tick_size=_to_float(record.get("tick_size")),
        min_order_size=_to_float(record.get("min_order_size")),
        last_trade_price=_to_float(record.get("last_trade_price")),
    )
    idx.states.append(state)
    idx.state_times.append(recv_ts_ns)


def _append_rest_record(idx: RestContextIndex, record: dict) -> None:
    if str(record.get("endpoint") or "") != "gamma_discovery":
        return
    recv_ts_ns = _to_int(record.get("recv_ts_ns"))
    if recv_ts_ns is None:
        return
    selected_market = (record.get("payload") or {}).get("selected_market") or {}
    if not isinstance(selected_market, dict):
        return
    state = RestContextState(
        recv_ts_ns=recv_ts_ns,
        maker_base_fee_bps=_to_float(selected_market.get("makerBaseFee")),
        taker_base_fee_bps=_to_float(selected_market.get("takerBaseFee")),
        rewards_min_size=_to_float(selected_market.get("rewardsMinSize")),
        rewards_max_spread=_to_float(selected_market.get("rewardsMaxSpread")),
        order_min_size=_to_float(selected_market.get("orderMinSize")),
        spread=_to_float(selected_market.get("spread")),
        best_bid=_to_float(selected_market.get("bestBid")),
        best_ask=_to_float(selected_market.get("bestAsk")),
        last_trade_price=_to_float(selected_market.get("lastTradePrice")),
    )
    idx.times.append(recv_ts_ns)
    idx.states.append(state)


# ──────────────────────────────────────────────────────────────────────────────
# Live-reference index — loaded from already-built canonical per-day parquet
# (in true live mode this gets fed by the chainlink/live_reference recorder)
# ──────────────────────────────────────────────────────────────────────────────
def load_live_reference_from_canonical(
    *, root: Path, shard_dates: list[date],
) -> LiveReferenceIndex:
    """Replica of dataset_factory._load_live_reference_index, exported for live use."""
    base = root / "canonical" / "live_reference_events_v1"
    frames: list[pd.DataFrame] = []
    for shard_date in sorted(set(shard_dates)):
        path = base / f"{shard_date.isoformat()}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return _empty_live_reference()
    import math as _math
    frame = pd.concat(frames, axis=0, ignore_index=True).sort_values("ts_seconds", kind="stable")
    for column, default in {
        "bias_bootstrapped": True, "bias_active": False, "bias_carried_forward": False,
        "bias_mode": "unavailable", "active_window_bias": _math.nan,
        "bias_state_stale": True, "bias_neutralized": False,
        "bias_observation_count": 0,
        "bias_state_age_seconds": _math.nan, "applied_bias_age_seconds": _math.nan,
    }.items():
        if column not in frame.columns:
            frame[column] = default
    return LiveReferenceIndex(
        ts_seconds=frame["ts_seconds"].astype("int64").tolist(),
        synthetic_corrected=frame["synthetic_corrected"].astype("float64").tolist(),
        price_valid=frame["price_valid"].astype("bool").tolist(),
        degraded=frame["degraded"].astype("bool").tolist(),
        source_count=frame["source_count"].astype("int64").tolist(),
        bias_bootstrapped=frame["bias_bootstrapped"].astype("bool").tolist(),
        bias_active=frame["bias_active"].astype("bool").tolist(),
        bias_carried_forward=frame["bias_carried_forward"].astype("bool").tolist(),
        bias_mode=frame["bias_mode"].astype("string").fillna("unavailable").astype(str).tolist(),
        active_window_bias=frame["active_window_bias"].astype("float64").tolist(),
        bias_state_stale=frame["bias_state_stale"].astype("bool").tolist(),
        bias_neutralized=frame["bias_neutralized"].astype("bool").tolist(),
        bias_observation_count=frame["bias_observation_count"].astype("int64").tolist(),
        bias_state_age_seconds=frame["bias_state_age_seconds"].astype("float64").tolist(),
        applied_bias_age_seconds=frame["applied_bias_age_seconds"].astype("float64").tolist(),
        binance_spot_mid=frame["binance_spot_mid"].astype("float64").tolist(),
        binance_usdm_mid=frame["binance_usdm_mid"].astype("float64").tolist(),
        hyperliquid_mid=frame["hyperliquid_mid"].astype("float64").tolist(),
        max_cross_source_spread=frame["max_cross_source_spread"].astype("float64").tolist(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot grid — same schedule as dataset_factory._sample_times_for_dense_close
# ──────────────────────────────────────────────────────────────────────────────
def _dense_close_grid(label: MarketLabel) -> list[int]:
    out: list[int] = []
    start_ns = label.market_close_ts_ns - (60 * SECOND_NS)
    mid_ns   = label.market_close_ts_ns - (10 * SECOND_NS)
    if start_ns < label.market_open_ts_ns:
        start_ns = label.market_open_ts_ns
    cur = start_ns
    while cur < mid_ns:
        out.append(cur)
        cur += 250 * MILLISECOND_NS
    cur = mid_ns
    while cur < label.market_close_ts_ns:
        out.append(cur)
        cur += 100 * MILLISECOND_NS
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Incremental MarketLiveWindow — built up to a snapshot second
# ──────────────────────────────────────────────────────────────────────────────
def _build_market_live_window_truncated(
    *, label: MarketLabel, live_reference: LiveReferenceIndex, snapshot_second: int,
) -> MarketLiveWindow:
    """Same logic as dataset_factory._build_market_live_window but only
    covers [market_open_s, snapshot_second]. Forward-looking columns
    (future_flip, terminal_*) get None/sentinel placeholders — they're
    leakage columns stripped before model input anyway."""
    end_second = min(snapshot_second, label.market_close_s - 1)
    seconds = list(range(label.market_open_s, end_second + 1))
    prices: list[float | None] = []
    signs: list[int] = []
    above_prefix = [0]
    below_prefix = [0]
    last_above = 0
    last_below = 0
    for second in seconds:
        idx = bisect_right(live_reference.ts_seconds, second) - 1
        price = live_reference.price_at_index(idx)
        prices.append(price)
        if price is None:
            signs.append(0)
            above_prefix.append(last_above)
            below_prefix.append(last_below)
            continue
        sign = 1 if price - label.price_to_beat >= 0 else -1
        signs.append(sign)
        if sign >= 0: last_above += 1
        else:         last_below += 1
        above_prefix.append(last_above)
        below_prefix.append(last_below)

    # Future-leaking fields are filled with placeholders matching the
    # offline path when those columns are stripped pre-model.
    return MarketLiveWindow(
        seconds=seconds, prices=prices, signs=signs,
        above_prefix=above_prefix, below_prefix=below_prefix,
        future_flip=[False] * len(seconds),
        terminal_price=None, terminal_delta_to_strike=None, terminal_margin_to_strike=None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class _MarketState:
    label: MarketLabel
    up_index:  TokenL2Index   = field(default_factory=_empty_l2_index)
    down_index: TokenL2Index  = field(default_factory=_empty_l2_index)
    rest_index: RestContextIndex = field(default_factory=_empty_rest_index)
    snapshot_grid: list[int]   = field(default_factory=list)
    next_grid_idx: int         = 0  # pointer into snapshot_grid


class StreamingFeatureEngine:
    """Maintains live indices, emits feature rows at each dense_close grid time."""

    def __init__(self, live_reference: LiveReferenceIndex):
        self.live_reference = live_reference
        self.markets: dict[str, _MarketState] = {}
        # markets indexed by asset_id so L2 events route quickly
        self._asset_to_market: dict[str, tuple[str, str]] = {}   # asset_id -> (market_slug, "up"|"down")

    # ── registration ──────────────────────────────────────────────────────────
    def register_market(self, label: MarketLabel) -> None:
        if label.market_slug in self.markets:
            return
        ms = _MarketState(label=label, snapshot_grid=_dense_close_grid(label))
        self.markets[label.market_slug] = ms
        if label.up_asset_id:   self._asset_to_market[label.up_asset_id]   = (label.market_slug, "up")
        if label.down_asset_id: self._asset_to_market[label.down_asset_id] = (label.market_slug, "down")

    # ── event ingestion ──────────────────────────────────────────────────────
    def feed_l2(self, record: dict) -> None:
        slug = str(record.get("selected_market_slug") or "").strip()
        outcome = str(record.get("token_outcome") or "").strip().lower()
        ms = self.markets.get(slug)
        if ms is None or outcome not in {"up", "down"}:
            return
        idx = ms.up_index if outcome == "up" else ms.down_index
        _append_l2_record(idx, record)

    def feed_rest(self, record: dict) -> None:
        slug = str(record.get("selected_market_slug") or "").strip()
        ms = self.markets.get(slug)
        if ms is None: return
        _append_rest_record(ms.rest_index, record)

    # ── snapshot firing ───────────────────────────────────────────────────────
    def tick(self, now_ns: int) -> Iterator[dict]:
        """For each market whose next grid time has passed, emit a snapshot row."""
        for slug, ms in self.markets.items():
            while ms.next_grid_idx < len(ms.snapshot_grid):
                ts_ns = ms.snapshot_grid[ms.next_grid_idx]
                if ts_ns > now_ns:
                    break
                row = self._build_snapshot(ms, ts_ns)
                ms.next_grid_idx += 1
                if row is not None:
                    yield row

    def _build_snapshot(self, ms: _MarketState, snapshot_ts_ns: int) -> dict | None:
        snapshot_second = snapshot_ts_ns // SECOND_NS
        market_live = _build_market_live_window_truncated(
            label=ms.label,
            live_reference=self.live_reference,
            snapshot_second=int(snapshot_second),
        )
        # sample_bucket_ms matches the offline pipeline
        sample_bucket_ms = 250 if snapshot_ts_ns < ms.label.market_close_ts_ns - (10 * SECOND_NS) else 100
        return _build_snapshot_row(
            label=ms.label,
            snapshot_ts_ns=snapshot_ts_ns,
            sample_family="dense_close",
            sample_bucket_ms=sample_bucket_ms,
            live_reference=self.live_reference,
            market_live=market_live,
            up_index=ms.up_index,
            down_index=ms.down_index,
            rest_index=ms.rest_index,
        )


# ──────────────────────────────────────────────────────────────────────────────
# File-based event source — for parity testing
# ──────────────────────────────────────────────────────────────────────────────
def _file_iter(reader: UnifiedRawReader, path: Path, *, kind: str,
               market_slugs: set[str] | None,
               end_ns: int | None) -> Iterator[tuple[int, str, dict]]:
    """Stream (recv_ts_ns, kind, record) from a single file.

    We DELIBERATELY do not filter by record_type here. The offline
    dataset_factory counts events into `event_times_by_type` for every
    record with a non-empty `event_type`, regardless of record_type
    (e.g. `best_bid_ask` updates can ride on records whose record_type
    is not `l2_book_state`). Filtering here would silently miss those
    counts and produce a mismatched `*_best_bid_ask_updates_1s` feature.

    Cheap per-record filters retained:
      - recv_ts_ns valid + monotonic stop at end_ns
      - rest endpoint must be `gamma_discovery` (matches offline path)
      - market_slug must match (defensive — usually filename filter handles it)
    """
    for record in reader.iter_file(path, finalized_only=True, tail_tolerant=False):
        recv = _to_int(record.get("recv_ts_ns"))
        if recv is None:
            continue
        if end_ns is not None and recv > end_ns:
            return
        if kind == "rest":
            if str(record.get("endpoint") or "") != "gamma_discovery":
                continue
        if market_slugs is not None:
            slug = str(record.get("selected_market_slug") or "").strip()
            if slug not in market_slugs:
                continue
        yield (recv, kind, record)


def iter_raw_events_in_arrival_order(
    *, raw_root: Path, shard_dates: list[date], market_slugs: set[str] | None = None,
    policy_path: Path = DEFAULT_DATASET_POLICY_PATH,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> Iterator[tuple[int, str, dict]]:
    """Yield (recv_ts_ns, kind, record) tuples, globally sorted by
    recv_ts_ns across the (l2, rest) files matching market_slugs.

    Implementation: heapq.merge over per-file generators. Each file
    yields its records in append-order (which is recv_ts_ns-ordered in
    practice), so the merge produces a globally sorted stream with O(1)
    peak memory rather than O(N).

    Cheap pre-filters applied BEFORE any decompression:
      1) Filename-embedded HOUR (`HH-MM-SS__...`) is compared against
         [start_hour - 1, end_hour]. A recorder file opened at HH:MM:SS
         contains data starting at HH:MM:SS, so anything strictly before
         the window can't contribute. The 1-hour pad covers files whose
         data spans into the next hour.
      2) Filename-embedded MARKET SLUG (`...__<slug>__...`) is compared
         against `market_slugs`.

    These two filters drop ~99% of files on a typical day (195 markets,
    each with l2/rest/lifecycle/etc. files per hour) without ever
    opening the .zst archive.
    """
    policy = load_dataset_policy(policy_path)
    reader = UnifiedRawReader(root=raw_root)
    relative_prefix = Path("polymarket") / "btc_updown_5m"

    start_hour = (start_ns // (3600 * SECOND_NS)) % 24 if start_ns is not None else None
    end_hour   = (end_ns   // (3600 * SECOND_NS)) % 24 if end_ns   is not None else None
    # Per-date hour bounds — only enforced on dates that fall *inside*
    # [start_date, end_date]. We assume the caller passes a tight
    # `shard_dates` list (single day, or two adjacent days).
    start_date = (datetime.fromtimestamp(start_ns / 1e9, UTC).date()
                  if start_ns is not None else None)
    end_date   = (datetime.fromtimestamp(end_ns / 1e9, UTC).date()
                  if end_ns is not None else None)

    file_iters: list[Iterator[tuple[int, str, dict]]] = []
    files_scanned = 0; files_skipped_slug = 0; files_skipped_hour = 0
    for shard_date in shard_dates:
        for kind in ("l2", "rest"):
            for path in reader.list_files_for_date(
                relative_prefix=relative_prefix,
                target_date=shard_date, kind=kind, finalized_only=False,
            ):
                # (1) Hour filter — only valid when this shard_date matches the
                # window's start/end date (otherwise we can't compare hour to bound).
                if start_hour is not None and end_hour is not None:
                    try:
                        file_hour = int(path.name[:2])
                    except ValueError:
                        file_hour = None
                    if file_hour is not None:
                        if shard_date == start_date and file_hour < max(0, start_hour - 1):
                            files_skipped_hour += 1
                            continue
                        if shard_date == end_date and file_hour > end_hour:
                            files_skipped_hour += 1
                            continue

                # (2) Slug filter
                if market_slugs is not None:
                    name_parts = path.stem.split("__")
                    if len(name_parts) >= 2 and name_parts[1] not in market_slugs:
                        files_skipped_slug += 1
                        continue

                metadata = reader.file_metadata(path)
                eligible, _ = training_metadata_eligibility(
                    policy=policy, metadata=metadata, source_name="polymarket",
                    date_str=shard_date.isoformat(),
                    hour_str=path.name[:5],
                )
                if not eligible:
                    continue
                files_scanned += 1
                file_iters.append(_file_iter(
                    reader, path, kind=kind, market_slugs=market_slugs, end_ns=end_ns,
                ))

    print(f"  [iter] files: {files_scanned} scanned, "
          f"{files_skipped_slug} skipped (slug), {files_skipped_hour} skipped (hour)",
          flush=True)
    # Heap merge — each file is already monotonic in recv_ts_ns.
    yield from merge(*file_iters, key=lambda x: x[0])


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build a parquet from the streaming engine for a window
# ──────────────────────────────────────────────────────────────────────────────
def replay_window_to_rows(
    *, raw_root: Path, labels: list[MarketLabel],
    start_ns: int, end_ns: int,
    live_reference: LiveReferenceIndex,
) -> list[dict]:
    """Replay raw events in arrival order through the engine and collect
    all snapshot rows the engine emits up through end_ns."""
    engine = StreamingFeatureEngine(live_reference=live_reference)
    for label in labels:
        engine.register_market(label)

    market_slugs = {l.market_slug for l in labels}
    # Only the dates the window touches. The streaming engine doesn't
    # need previous-day shards because raw L2/REST events for a 5-minute
    # market always live within that market's own opening day.
    shard_dates: list[date] = sorted({
        datetime.fromtimestamp(start_ns / 1e9, UTC).date(),
        datetime.fromtimestamp(max(start_ns, end_ns - 1) / 1e9, UTC).date(),
    })

    rows: list[dict] = []
    last_now = 0
    n_events = 0
    for recv_ts_ns, kind, record in iter_raw_events_in_arrival_order(
        raw_root=raw_root, shard_dates=shard_dates, market_slugs=market_slugs,
        start_ns=start_ns, end_ns=end_ns,
    ):
        if kind == "l2":
            engine.feed_l2(record)
        else:
            engine.feed_rest(record)
        if recv_ts_ns >= last_now:
            rows.extend(engine.tick(recv_ts_ns))
            last_now = recv_ts_ns
        n_events += 1
        if n_events % 50_000 == 0:
            print(f"  [replay] events processed: {n_events:,}  emitted rows: {len(rows):,}", flush=True)

    # Final flush — fire any snapshots whose grid time is <= end_ns
    rows.extend(engine.tick(end_ns))
    print(f"  [replay] total events: {n_events:,}  total rows: {len(rows):,}", flush=True)
    return rows
