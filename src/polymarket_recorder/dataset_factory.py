from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from binance_recorder.utils import ns_to_iso
from market_recorders.dataset_policy import (
    DEFAULT_DATASET_POLICY_PATH,
    load_dataset_policy,
    training_metadata_eligibility,
)
from market_recorders.unified_reader import UnifiedRawReader

SECOND_NS = 1_000_000_000
MILLISECOND_NS = 1_000_000
DEFAULT_DATE_FROM = date(2026, 4, 19)
DEFAULT_DATE_TO = date(2026, 4, 22)
DEFAULT_DATASET_OUTPUT_DIR = Path("data/datasets")
DEFAULT_REPORT_PATH = Path("docs/phase3_dataset_factory_report.md")
DEFAULT_SCHEMA_DIR = Path("config/schemas")
DEFAULT_DECISION_LOG_PATH = Path("docs/decision_log.md")
DEFAULT_LIVE_REFERENCE_DIR = Path("data/canonical/live_reference_events_v1")
DEFAULT_QUALITY_REGISTRY_PATH = Path("data/canonical/quality_registry_v1/phase3_market_exclusions.parquet")

COARSE_DATASET_NAME = "resolution_snapshot_dataset_v1_coarse"
DENSE_DATASET_NAME = "resolution_snapshot_dataset_v1_dense_close"
MARKET_INTERVAL_DATASETS = {
    "preopen": "market_interval_dataset_v1_preopen",
    "first15s": "market_interval_dataset_v1_first15s",
    "first30s": "market_interval_dataset_v1_first30s",
}
LEAKAGE_ONLY_COLUMNS = {
    "flip_happened_after_t",
    "terminal_delta_to_strike",
    "terminal_margin_to_strike",
    "hold_to_close_edge_vs_mid",
    "resolved_side",
}
DEAD_COLUMNS = {
    "btc_realized_vol_1s",
    "up_token_min_order_size",
    "down_token_min_order_size",
}


@dataclass(slots=True, frozen=True)
class MarketLabel:
    market_slug: str
    market_id: str
    condition_id: str
    strike_recv_ts_ns: int
    price_to_beat: float
    market_open_ts_ns: int
    market_close_ts_ns: int
    market_open_s: int
    market_close_s: int
    up_asset_id: str | None
    down_asset_id: str | None
    resolved_side: str
    resolution_recv_ts_ns: int
    resolution_resolved_ts_ns: int


@dataclass(slots=True, frozen=True)
class TokenBookState:
    recv_ts_ns: int
    best_bid: float | None
    best_ask: float | None
    mid: float | None
    spread: float | None
    bid_depth_total: float | None
    ask_depth_total: float | None
    tick_size: float | None
    min_order_size: float | None
    last_trade_price: float | None


@dataclass(slots=True)
class TokenL2Index:
    state_times: list[int]
    states: list[TokenBookState]
    event_times_by_type: dict[str, list[int]]
    trade_times: list[int]
    trade_signed_size_prefix: list[float]
    trade_abs_size_prefix: list[float]
    trade_count_prefix: list[int]

    def latest_state(self, asof_ts_ns: int) -> TokenBookState | None:
        index = bisect_right(self.state_times, asof_ts_ns) - 1
        if index < 0:
            return None
        return self.states[index]

    def recent_event_count(self, event_type: str, *, asof_ts_ns: int, lookback_ns: int) -> int:
        values = self.event_times_by_type.get(event_type, [])
        if not values:
            return 0
        start = asof_ts_ns - lookback_ns
        right = bisect_right(values, asof_ts_ns)
        left = bisect_right(values, start)
        return max(0, right - left)

    def recent_trade_summary(self, *, asof_ts_ns: int, lookback_ns: int) -> tuple[int, float, float]:
        if not self.trade_times:
            return 0, 0.0, 0.0
        start = asof_ts_ns - lookback_ns
        right = bisect_right(self.trade_times, asof_ts_ns)
        left = bisect_right(self.trade_times, start)
        if right <= left:
            return 0, 0.0, 0.0
        signed = self.trade_signed_size_prefix[right] - self.trade_signed_size_prefix[left]
        absolute = self.trade_abs_size_prefix[right] - self.trade_abs_size_prefix[left]
        count = self.trade_count_prefix[right] - self.trade_count_prefix[left]
        return count, signed, absolute


@dataclass(slots=True, frozen=True)
class RestContextState:
    recv_ts_ns: int
    maker_base_fee_bps: float | None
    taker_base_fee_bps: float | None
    rewards_min_size: float | None
    rewards_max_spread: float | None
    order_min_size: float | None
    spread: float | None
    best_bid: float | None
    best_ask: float | None
    last_trade_price: float | None


@dataclass(slots=True)
class RestContextIndex:
    times: list[int]
    states: list[RestContextState]

    def latest(self, asof_ts_ns: int) -> RestContextState | None:
        index = bisect_right(self.times, asof_ts_ns) - 1
        if index < 0:
            return None
        return self.states[index]


@dataclass(slots=True)
class LiveReferenceIndex:
    ts_seconds: list[int]
    synthetic_corrected: list[float]
    price_valid: list[bool]
    degraded: list[bool]
    source_count: list[int]
    bias_bootstrapped: list[bool]
    bias_active: list[bool]
    bias_carried_forward: list[bool]
    bias_mode: list[str]
    active_window_bias: list[float]
    bias_state_stale: list[bool]
    bias_neutralized: list[bool]
    bias_observation_count: list[int]
    bias_state_age_seconds: list[float]
    applied_bias_age_seconds: list[float]
    binance_spot_mid: list[float]
    binance_usdm_mid: list[float]
    hyperliquid_mid: list[float]
    max_cross_source_spread: list[float]

    def asof_index(self, snapshot_ts_ns: int) -> int:
        asof_second = snapshot_ts_ns // SECOND_NS
        return bisect_right(self.ts_seconds, int(asof_second)) - 1

    def price_at_index(self, index: int) -> float | None:
        if index < 0 or index >= len(self.ts_seconds) or not self.price_valid[index]:
            return None
        value = self.synthetic_corrected[index]
        if math.isnan(value):
            return None
        return float(value)

    def row_at_index(self, index: int) -> dict[str, Any] | None:
        if index < 0 or index >= len(self.ts_seconds):
            return None
        return {
            "ts_seconds": self.ts_seconds[index],
            "synthetic_corrected": self.synthetic_corrected[index],
            "price_valid": self.price_valid[index],
            "degraded": self.degraded[index],
            "source_count": self.source_count[index],
            "bias_bootstrapped": self.bias_bootstrapped[index],
            "bias_active": self.bias_active[index],
            "bias_carried_forward": self.bias_carried_forward[index],
            "bias_mode": self.bias_mode[index],
            "active_window_bias": self.active_window_bias[index],
            "bias_state_stale": self.bias_state_stale[index],
            "bias_neutralized": self.bias_neutralized[index],
            "bias_observation_count": self.bias_observation_count[index],
            "bias_state_age_seconds": self.bias_state_age_seconds[index],
            "applied_bias_age_seconds": self.applied_bias_age_seconds[index],
            "binance_spot_mid": self.binance_spot_mid[index],
            "binance_usdm_mid": self.binance_usdm_mid[index],
            "hyperliquid_mid": self.hyperliquid_mid[index],
            "max_cross_source_spread": self.max_cross_source_spread[index],
        }


@dataclass(slots=True)
class MarketLiveWindow:
    seconds: list[int]
    prices: list[float | None]
    signs: list[int]
    above_prefix: list[int]
    below_prefix: list[int]
    future_flip: list[bool]
    terminal_price: float | None
    terminal_delta_to_strike: float | None
    terminal_margin_to_strike: float | None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = _mean(values)
    if mean_value is None:
        return None
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _mid(best_bid: float | None, best_ask: float | None) -> float | None:
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0


def _microprice(
    *,
    best_bid: float | None,
    best_ask: float | None,
    bid_depth_total: float | None,
    ask_depth_total: float | None,
) -> float | None:
    if None in {best_bid, best_ask, bid_depth_total, ask_depth_total}:
        return None
    total_depth = float(bid_depth_total) + float(ask_depth_total)
    if total_depth <= 0:
        return None
    return (
        float(best_ask) * float(bid_depth_total) + float(best_bid) * float(ask_depth_total)
    ) / total_depth


def _imbalance(bid_depth_total: float | None, ask_depth_total: float | None) -> float | None:
    if bid_depth_total is None or ask_depth_total is None:
        return None
    total = bid_depth_total + ask_depth_total
    if total <= 0:
        return None
    return (bid_depth_total - ask_depth_total) / total


def _ns_to_utc_date(value_ns: int) -> date:
    return datetime.fromtimestamp(value_ns / SECOND_NS, tz=UTC).date()


def _phase_bucket(t_to_close_s: float) -> str:
    if t_to_close_s <= 10.0:
        return "close"
    if t_to_close_s <= 60.0:
        return "late"
    if t_to_close_s <= 180.0:
        return "mid"
    return "early"


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _write_empty_parquet_like_existing(*, dataset_output_dir: Path, dataset_name: str, target_path: Path) -> None:
    template_paths = [
        path
        for path in sorted((dataset_output_dir / dataset_name).glob("*.parquet"))
        if path != target_path and path.exists() and path.stat().st_size > 0
    ]
    if not template_paths:
        raise FileNotFoundError(f"No template parquet available for empty {dataset_name} shard")
    schema = pq.ParquetFile(template_paths[0]).schema_arrow
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_batches([], schema=schema), target_path, compression="zstd")


def _write_empty_phase3_day_outputs(*, dataset_output_dir: Path, target_date: date) -> dict[str, Any]:
    paths = _phase3_output_paths(dataset_output_dir, target_date)
    hashes: dict[str, str] = {}
    written_paths: dict[str, str] = {}
    for dataset_name, path in paths.items():
        if not path.exists() or path.stat().st_size == 0:
            _write_empty_parquet_like_existing(
                dataset_output_dir=dataset_output_dir,
                dataset_name=dataset_name,
                target_path=path,
            )
        hashes[dataset_name] = _sha256(path)
        written_paths[dataset_name] = str(path)
    return {
        "hashes": hashes,
        "paths": written_paths,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _date_range(date_from: date, date_to: date) -> list[date]:
    current = date_from
    values: list[date] = []
    while current <= date_to:
        values.append(current)
        current += timedelta(days=1)
    return values


def _available_archive_dates(*, root: Path, relative_prefix: Path) -> list[date]:
    folder = root / "raw" / relative_prefix
    if not folder.exists():
        return []
    values: list[date] = []
    for child in folder.iterdir():
        if not child.is_dir():
            continue
        try:
            values.append(date.fromisoformat(child.name))
        except ValueError:
            continue
    return sorted(values)


def _file_hour(path: Path) -> str:
    return path.name.split(".")[0].split("__", 1)[0]


def _iter_training_file_records(
    *,
    root: Path,
    relative_prefix: Path,
    target_date: date,
    kind: str,
    source_name: str,
    policy_path: Path,
    exclusion_summary: dict[str, Counter],
) -> Any:
    policy = load_dataset_policy(policy_path)
    reader = UnifiedRawReader(root=root)
    for path in reader.list_files_for_date(
        relative_prefix=relative_prefix,
        target_date=target_date,
        kind=kind,
        finalized_only=False,
    ):
        metadata = reader.file_metadata(path)
        eligible, reason = training_metadata_eligibility(
            policy=policy,
            metadata=metadata,
            source_name=source_name,
            date_str=target_date.isoformat(),
            hour_str=_file_hour(path),
        )
        if not eligible:
            exclusion_summary[source_name][reason or "unknown"] += 1
            continue
        yield from reader.iter_file(
            path,
            finalized_only=True,
            tail_tolerant=False,
        )


def _canonical_asset_mapping(record: dict[str, Any]) -> tuple[str | None, str | None]:
    asset_ids = [str(value) for value in record.get("selected_asset_ids", []) if value is not None]
    outcomes = [str(value).lower() for value in record.get("selected_outcomes", []) if value is not None]
    mapping = {
        outcome: asset_id
        for asset_id, outcome in zip(asset_ids, outcomes, strict=False)
    }
    return mapping.get("up"), mapping.get("down")


def _load_market_labels(
    *,
    root: Path,
    policy_path: Path,
    date_from: date,
    date_to: date,
) -> tuple[dict[date, list[MarketLabel]], dict[str, Any]]:
    policy = load_dataset_policy(policy_path)
    raw_reader = UnifiedRawReader(root=root)
    strike_prefix = Path("polymarket") / "btc_updown_5m"
    resolution_prefix = Path("polymarket") / "resolution" / "btc_updown_5m"

    strike_exclusions: dict[str, Counter] = defaultdict(Counter)
    resolution_exclusions: dict[str, Counter] = defaultdict(Counter)
    earliest_strikes: dict[str, dict[str, Any]] = {}

    strike_scan_start = date_from - timedelta(days=1)
    strike_scan_dates = [
        shard_date
        for shard_date in _available_archive_dates(root=root, relative_prefix=strike_prefix)
        if strike_scan_start <= shard_date <= date_to
    ]
    if not strike_scan_dates:
        strike_scan_dates = _date_range(strike_scan_start, date_to)

    for target_date in strike_scan_dates:
        for path in raw_reader.list_files_for_date(
            relative_prefix=strike_prefix,
            target_date=target_date,
            kind="strike",
            finalized_only=False,
        ):
            metadata = raw_reader.file_metadata(path)
            eligible, reason = training_metadata_eligibility(
                policy=policy,
                metadata=metadata,
                source_name="polymarket",
                date_str=target_date.isoformat(),
                hour_str=_file_hour(path),
            )
            if not eligible:
                strike_exclusions["polymarket_strike"][reason or "unknown"] += 1
                continue
            for record in raw_reader.iter_file(path, finalized_only=True, tail_tolerant=False):
                if record.get("record_type") != "strike_observation":
                    continue
                if record.get("label_safe") is not True:
                    continue
                strike_price = _to_float(record.get("price_to_beat"))
                market_slug = str(record.get("selected_market_slug") or "").strip()
                market_open_s = _to_int(record.get("market_open_s"))
                market_close_s = _to_int(record.get("market_close_s"))
                recv_ts_ns = _to_int(record.get("recv_ts_ns"))
                if (
                    not market_slug
                    or strike_price is None
                    or market_open_s is None
                    or market_close_s is None
                    or recv_ts_ns is None
                ):
                    continue
                current = earliest_strikes.get(market_slug)
                if current is None or recv_ts_ns < int(current["strike_recv_ts_ns"]):
                    up_asset_id, down_asset_id = _canonical_asset_mapping(record)
                    earliest_strikes[market_slug] = {
                        "market_slug": market_slug,
                        "market_id": str(record.get("selected_market_id") or ""),
                        "condition_id": str(record.get("selected_condition_id") or ""),
                        "strike_recv_ts_ns": recv_ts_ns,
                        "price_to_beat": strike_price,
                        "market_open_s": market_open_s,
                        "market_close_s": market_close_s,
                        "up_asset_id": up_asset_id,
                        "down_asset_id": down_asset_id,
                    }

    earliest_resolutions: dict[str, dict[str, Any]] = {}
    resolution_scan_dates = _available_archive_dates(root=root, relative_prefix=resolution_prefix)
    if not resolution_scan_dates:
        resolution_scan_dates = _date_range(date_from, date_to + timedelta(days=1))
    for target_date in resolution_scan_dates:
        for path in raw_reader.list_files_for_date(
            relative_prefix=resolution_prefix,
            target_date=target_date,
            kind="resolution",
            finalized_only=False,
        ):
            metadata = raw_reader.file_metadata(path)
            eligible, reason = training_metadata_eligibility(
                policy=policy,
                metadata=metadata,
                source_name="polymarket",
                date_str=target_date.isoformat(),
                hour_str=_file_hour(path),
            )
            if not eligible:
                resolution_exclusions["polymarket_resolution"][reason or "unknown"] += 1
                continue
            for record in raw_reader.iter_file(path, finalized_only=True, tail_tolerant=False):
                if record.get("record_type") != "market_resolution":
                    continue
                market_slug = str(record.get("market_slug") or "").strip()
                resolved_side = str(record.get("resolved_side") or "").strip().lower()
                recv_ts_ns = _to_int(record.get("recv_ts_ns"))
                resolved_ts_ns = _to_int(record.get("resolved_ts_ns"))
                if not market_slug or resolved_side not in {"up", "down"} or recv_ts_ns is None or resolved_ts_ns is None:
                    continue
                current = earliest_resolutions.get(market_slug)
                rank = (resolved_ts_ns, recv_ts_ns)
                if current is None or rank < (int(current["resolved_ts_ns"]), int(current["recv_ts_ns"])):
                    earliest_resolutions[market_slug] = {
                        "resolved_side": resolved_side,
                        "recv_ts_ns": recv_ts_ns,
                        "resolved_ts_ns": resolved_ts_ns,
                    }

    labels_by_day: dict[date, list[MarketLabel]] = defaultdict(list)
    missing_resolution_market_slugs: list[str] = []
    for market_slug, strike in sorted(earliest_strikes.items()):
        resolution = earliest_resolutions.get(market_slug)
        if resolution is None:
            missing_resolution_market_slugs.append(market_slug)
            continue
        market_open_ts_ns = int(strike["market_open_s"]) * SECOND_NS
        market_close_ts_ns = int(strike["market_close_s"]) * SECOND_NS
        market_day = datetime.fromtimestamp(int(strike["market_open_s"]), tz=UTC).date()
        if market_day < date_from or market_day > date_to:
            continue
        labels_by_day[market_day].append(
            MarketLabel(
                market_slug=market_slug,
                market_id=strike["market_id"],
                condition_id=strike["condition_id"],
                strike_recv_ts_ns=int(strike["strike_recv_ts_ns"]),
                price_to_beat=float(strike["price_to_beat"]),
                market_open_ts_ns=market_open_ts_ns,
                market_close_ts_ns=market_close_ts_ns,
                market_open_s=int(strike["market_open_s"]),
                market_close_s=int(strike["market_close_s"]),
                up_asset_id=strike["up_asset_id"],
                down_asset_id=strike["down_asset_id"],
                resolved_side=str(resolution["resolved_side"]),
                resolution_recv_ts_ns=int(resolution["recv_ts_ns"]),
                resolution_resolved_ts_ns=int(resolution["resolved_ts_ns"]),
            )
        )

    for labels in labels_by_day.values():
        labels.sort(key=lambda item: (item.market_open_ts_ns, item.market_slug))

    summary = {
        "canonical_label_safe_strikes": len(earliest_strikes),
        "canonical_resolutions_in_scope": len(earliest_resolutions),
        "matched_markets": sum(len(values) for values in labels_by_day.values()),
        "missing_resolution_market_count": len(missing_resolution_market_slugs),
        "missing_resolution_market_slugs": missing_resolution_market_slugs,
        "strike_exclusions": {key: dict(value) for key, value in strike_exclusions.items()},
        "resolution_exclusions": {key: dict(value) for key, value in resolution_exclusions.items()},
    }
    return labels_by_day, summary


def _build_market_history_features(
    labels_by_day: dict[date, list[MarketLabel]],
) -> dict[str, dict[str, Any]]:
    ordered_labels = sorted(
        (label for labels in labels_by_day.values() for label in labels),
        key=lambda item: (item.market_open_ts_ns, item.market_slug),
    )
    history_features: dict[str, dict[str, Any]] = {}
    prior_outcomes: list[int] = []
    up_streak = 0
    down_streak = 0

    for label in ordered_labels:
        previous_side = prior_outcomes[-1] if prior_outcomes else None
        history_features[label.market_slug] = {
            "prior_market_count": len(prior_outcomes),
            "previous_market_resolved_side_label": previous_side if previous_side is not None else -1,
            "resolved_up_streak_before_open": up_streak,
            "resolved_down_streak_before_open": down_streak,
            "resolved_up_ratio_last_12": (
                sum(prior_outcomes[-12:]) / min(len(prior_outcomes), 12)
                if prior_outcomes
                else math.nan
            ),
            "resolved_up_ratio_last_24": (
                sum(prior_outcomes[-24:]) / min(len(prior_outcomes), 24)
                if prior_outcomes
                else math.nan
            ),
        }
        current_label = 1 if label.resolved_side == "up" else 0
        prior_outcomes.append(current_label)
        if current_label == 1:
            up_streak += 1
            down_streak = 0
        else:
            down_streak += 1
            up_streak = 0
    return history_features


def _load_live_reference_index(
    *,
    root: Path,
    shard_dates: list[date],
) -> LiveReferenceIndex:
    base = root / DEFAULT_LIVE_REFERENCE_DIR.relative_to("data")
    frames: list[pd.DataFrame] = []
    for shard_date in sorted(set(shard_dates)):
        path = base / f"{shard_date.isoformat()}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return LiveReferenceIndex([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    frame = pd.concat(frames, axis=0, ignore_index=True).sort_values("ts_seconds", kind="stable")
    for column, default in {
        "bias_bootstrapped": True,
        "bias_active": False,
        "bias_carried_forward": False,
        "bias_mode": "unavailable",
        "active_window_bias": math.nan,
        "bias_state_stale": True,
        "bias_neutralized": False,
        "bias_observation_count": 0,
        "bias_state_age_seconds": math.nan,
        "applied_bias_age_seconds": math.nan,
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


def _load_l2_indices_for_day(
    *,
    root: Path,
    policy_path: Path,
    shard_dates: list[date],
    market_slugs: set[str],
) -> tuple[dict[str, dict[str, TokenL2Index]], dict[str, Counter]]:
    exclusion_summary: dict[str, Counter] = defaultdict(Counter)
    per_market_token_states: dict[str, dict[str, list[TokenBookState]]] = defaultdict(lambda: defaultdict(list))
    per_market_token_state_times: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    per_market_token_event_times: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    per_market_token_trade_times: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    per_market_token_trade_signed_prefix: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0]))
    per_market_token_trade_abs_prefix: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0]))
    per_market_token_trade_count_prefix: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0]))

    relative_prefix = Path("polymarket") / "btc_updown_5m"
    for shard_date in shard_dates:
        for record in _iter_training_file_records(
            root=root,
            relative_prefix=relative_prefix,
            target_date=shard_date,
            kind="l2",
            source_name="polymarket",
            policy_path=policy_path,
            exclusion_summary=exclusion_summary,
        ):
            if record.get("token_modeling_safe") is not True:
                continue
            market_slug = str(record.get("selected_market_slug") or "").strip()
            token_outcome = str(record.get("token_outcome") or "").strip().lower()
            recv_ts_ns = _to_int(record.get("recv_ts_ns"))
            if (
                not market_slug
                or market_slug not in market_slugs
                or token_outcome not in {"up", "down"}
                or recv_ts_ns is None
            ):
                continue
            event_type = str(record.get("event_type") or "").strip()
            if event_type:
                per_market_token_event_times[market_slug][token_outcome][event_type].append(recv_ts_ns)

            if record.get("record_type") == "trade_execution":
                size = _to_float(record.get("size")) or 0.0
                side = str(record.get("side") or "").upper()
                signed_size = size if side == "BUY" else -size if side == "SELL" else 0.0
                per_market_token_trade_times[market_slug][token_outcome].append(recv_ts_ns)
                signed_prefix = per_market_token_trade_signed_prefix[market_slug][token_outcome]
                abs_prefix = per_market_token_trade_abs_prefix[market_slug][token_outcome]
                count_prefix = per_market_token_trade_count_prefix[market_slug][token_outcome]
                signed_prefix.append(signed_prefix[-1] + signed_size)
                abs_prefix.append(abs_prefix[-1] + abs(size))
                count_prefix.append(count_prefix[-1] + 1)

            if record.get("record_type") != "l2_book_state":
                continue

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
            per_market_token_states[market_slug][token_outcome].append(state)
            per_market_token_state_times[market_slug][token_outcome].append(recv_ts_ns)

    indices: dict[str, dict[str, TokenL2Index]] = {}
    for market_slug in sorted(per_market_token_states):
        market_bucket: dict[str, TokenL2Index] = {}
        for token_outcome in sorted(per_market_token_states[market_slug]):
            market_bucket[token_outcome] = TokenL2Index(
                state_times=per_market_token_state_times[market_slug][token_outcome],
                states=per_market_token_states[market_slug][token_outcome],
                event_times_by_type={
                    event_type: times
                    for event_type, times in per_market_token_event_times[market_slug][token_outcome].items()
                },
                trade_times=per_market_token_trade_times[market_slug][token_outcome],
                trade_signed_size_prefix=per_market_token_trade_signed_prefix[market_slug][token_outcome],
                trade_abs_size_prefix=per_market_token_trade_abs_prefix[market_slug][token_outcome],
                trade_count_prefix=per_market_token_trade_count_prefix[market_slug][token_outcome],
            )
        indices[market_slug] = market_bucket
    return indices, {key: Counter(value) for key, value in exclusion_summary.items()}


def _load_rest_indices_for_day(
    *,
    root: Path,
    policy_path: Path,
    shard_dates: list[date],
    market_slugs: set[str],
) -> tuple[dict[str, RestContextIndex], dict[str, Counter]]:
    exclusion_summary: dict[str, Counter] = defaultdict(Counter)
    per_market_states: dict[str, list[RestContextState]] = defaultdict(list)
    per_market_times: dict[str, list[int]] = defaultdict(list)

    relative_prefix = Path("polymarket") / "btc_updown_5m"
    for shard_date in shard_dates:
        for record in _iter_training_file_records(
            root=root,
            relative_prefix=relative_prefix,
            target_date=shard_date,
            kind="rest",
            source_name="polymarket",
            policy_path=policy_path,
            exclusion_summary=exclusion_summary,
        ):
            if str(record.get("endpoint") or "") != "gamma_discovery":
                continue
            market_slug = str(record.get("selected_market_slug") or "").strip()
            if not market_slug or market_slug not in market_slugs:
                continue
            recv_ts_ns = _to_int(record.get("recv_ts_ns"))
            if recv_ts_ns is None:
                continue
            selected_market = (record.get("payload") or {}).get("selected_market") or {}
            if not isinstance(selected_market, dict):
                continue
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
            per_market_states[market_slug].append(state)
            per_market_times[market_slug].append(recv_ts_ns)

    indices: dict[str, RestContextIndex] = {}
    for market_slug in sorted(per_market_states):
        indices[market_slug] = RestContextIndex(
            times=per_market_times[market_slug],
            states=per_market_states[market_slug],
        )
    return indices, {key: Counter(value) for key, value in exclusion_summary.items()}


def _signed_delta_to_label(delta_to_strike: float | None) -> int | None:
    if delta_to_strike is None:
        return None
    return 1 if delta_to_strike >= 0 else -1


def _has_numeric(value: Any) -> bool:
    return value is not None and not math.isnan(float(value))


def _crossed_book(best_bid: float | None, best_ask: float | None) -> bool:
    return best_bid is not None and best_ask is not None and best_bid >= best_ask


def _apply_training_contract(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if frame.empty:
        frame = frame.copy()
        frame["has_canonical_resolution"] = pd.Series(dtype="bool")
        frame["is_trainable"] = pd.Series(dtype="bool")
        frame["market_fully_without_live_reference"] = pd.Series(dtype="bool")
        frame["market_fully_without_trusted_live_reference"] = pd.Series(dtype="bool")
        frame["market_fully_without_token_mid_complete"] = pd.Series(dtype="bool")
        frame["market_5m_training_row_count"] = pd.Series(dtype="int64")
        frame["market_5m_complete_row_count"] = pd.Series(dtype="int64")
        frame["market_5m_complete_rate"] = pd.Series(dtype="float64")
        frame["market_full_matrix_trainable"] = pd.Series(dtype="bool")
        frame["training_feature_eligible"] = pd.Series(dtype="bool")
        frame["exclusion_reason"] = pd.Series(dtype="object")
        frame["complete_feature_matrix_eligible"] = pd.Series(dtype="bool")
        frame["feature_matrix_exclusion_reason"] = pd.Series(dtype="object")
        frame["full_matrix_training_eligible"] = pd.Series(dtype="bool")
        frame["full_matrix_exclusion_reason"] = pd.Series(dtype="object")
        return frame, []

    frame = frame.copy()
    frame["has_canonical_resolution"] = True
    market_fully_without_live_reference = ~frame.groupby("market_slug")["live_price_valid"].transform("any")
    if "live_reference_trusted" not in frame.columns:
        frame["live_reference_trusted"] = False
    market_fully_without_trusted_live_reference = ~frame.groupby("market_slug")["live_reference_trusted"].transform("any")
    if "token_mid_complete" not in frame.columns:
        frame["token_mid_complete"] = False
    if "spot_perp_available" not in frame.columns:
        frame["spot_perp_available"] = False
    market_fully_without_token_mid_complete = ~frame.groupby("market_slug")["token_mid_complete"].transform("any")
    frame["market_fully_without_live_reference"] = market_fully_without_live_reference.astype("bool")
    frame["market_fully_without_trusted_live_reference"] = market_fully_without_trusted_live_reference.astype("bool")
    frame["market_fully_without_token_mid_complete"] = market_fully_without_token_mid_complete.astype("bool")
    frame["is_trainable"] = (
        ~frame["market_fully_without_live_reference"]
        & ~frame["market_fully_without_trusted_live_reference"]
    ).astype("bool")
    frame["training_feature_eligible"] = True
    frame["exclusion_reason"] = ""

    live_invalid = ~frame["live_price_valid"].astype("bool")
    live_untrusted = ~frame["live_reference_trusted"].astype("bool")
    market_not_trainable = ~frame["is_trainable"].astype("bool")
    coupling_mask = frame["coupling_anomaly"].astype("bool")
    crossed_mask = frame["crossed_book"].astype("bool")

    frame.loc[live_invalid, ["training_feature_eligible", "exclusion_reason"]] = [False, "live_reference_unavailable"]
    frame.loc[~live_invalid & live_untrusted, ["training_feature_eligible", "exclusion_reason"]] = [False, "live_reference_untrusted"]
    frame.loc[~live_invalid & ~live_untrusted & market_not_trainable, ["training_feature_eligible", "exclusion_reason"]] = [False, "market_not_trainable"]
    frame.loc[~live_invalid & ~live_untrusted & ~market_not_trainable & coupling_mask, ["training_feature_eligible", "exclusion_reason"]] = [False, "coupling_anomaly"]
    frame.loc[
        ~live_invalid & ~live_untrusted & ~market_not_trainable & ~coupling_mask & crossed_mask,
        ["training_feature_eligible", "exclusion_reason"],
    ] = [False, "crossed_book"]
    frame["training_feature_eligible"] = frame["training_feature_eligible"].astype("bool")
    frame["exclusion_reason"] = frame["exclusion_reason"].astype("string").fillna("").astype(str)
    frame["complete_feature_matrix_eligible"] = (
        frame["training_feature_eligible"].astype("bool")
        & frame["token_mid_complete"].astype("bool")
        & frame["spot_perp_available"].astype("bool")
    )
    frame["feature_matrix_exclusion_reason"] = ""
    frame.loc[
        ~frame["training_feature_eligible"].astype("bool"),
        "feature_matrix_exclusion_reason",
    ] = frame.loc[~frame["training_feature_eligible"].astype("bool"), "exclusion_reason"]
    token_incomplete = frame["training_feature_eligible"].astype("bool") & ~frame["token_mid_complete"].astype("bool")
    spot_perp_missing = (
        frame["training_feature_eligible"].astype("bool")
        & frame["token_mid_complete"].astype("bool")
        & ~frame["spot_perp_available"].astype("bool")
    )
    frame.loc[token_incomplete, "feature_matrix_exclusion_reason"] = "token_mid_incomplete"
    frame.loc[spot_perp_missing, "feature_matrix_exclusion_reason"] = "spot_perp_unavailable"
    frame["complete_feature_matrix_eligible"] = frame["complete_feature_matrix_eligible"].astype("bool")
    frame["feature_matrix_exclusion_reason"] = (
        frame["feature_matrix_exclusion_reason"].astype("string").fillna("").astype(str)
    )
    market_training_row_count = frame.groupby("market_slug")["training_feature_eligible"].transform("sum").astype("int64")
    market_complete_row_count = frame.groupby("market_slug")["complete_feature_matrix_eligible"].transform("sum").astype("int64")
    frame["market_5m_training_row_count"] = market_training_row_count
    frame["market_5m_complete_row_count"] = market_complete_row_count
    market_training_denominator = market_training_row_count.astype("float64").where(
        market_training_row_count > 0,
        math.nan,
    )
    frame["market_5m_complete_rate"] = (
        market_complete_row_count.astype("float64") / market_training_denominator
    ).fillna(0.0)
    frame["market_full_matrix_trainable"] = (
        (frame["market_5m_training_row_count"] > 0)
        & (frame["market_5m_complete_rate"] >= 0.80)
    ).astype("bool")
    frame["full_matrix_training_eligible"] = (
        frame["complete_feature_matrix_eligible"].astype("bool")
        & frame["market_full_matrix_trainable"].astype("bool")
    ).astype("bool")
    frame["full_matrix_exclusion_reason"] = ""
    frame.loc[
        ~frame["complete_feature_matrix_eligible"].astype("bool"),
        "full_matrix_exclusion_reason",
    ] = frame.loc[
        ~frame["complete_feature_matrix_eligible"].astype("bool"),
        "feature_matrix_exclusion_reason",
    ]
    frame.loc[
        frame["complete_feature_matrix_eligible"].astype("bool")
        & ~frame["market_full_matrix_trainable"].astype("bool"),
        "full_matrix_exclusion_reason",
    ] = "market_5m_complete_rate_below_80pct"
    frame["full_matrix_exclusion_reason"] = frame["full_matrix_exclusion_reason"].astype("string").fillna("").astype(str)

    quality_rows = []
    quality_rows.extend([
        {
            "dataset_name": dataset_name,
            "market_slug": str(market_slug),
            "exclusion_reason": "market_fully_without_live_reference",
            "rows": int(group.shape[0]),
            "first_snapshot_ts_ns": int(group["snapshot_ts_ns"].min()),
            "last_snapshot_ts_ns": int(group["snapshot_ts_ns"].max()),
        }
        for market_slug, group in frame.loc[frame["market_fully_without_live_reference"]].groupby("market_slug", sort=True)
    ])
    quality_rows.extend([
        {
            "dataset_name": dataset_name,
            "market_slug": str(market_slug),
            "exclusion_reason": "market_fully_without_trusted_live_reference",
            "rows": int(group.shape[0]),
            "first_snapshot_ts_ns": int(group["snapshot_ts_ns"].min()),
            "last_snapshot_ts_ns": int(group["snapshot_ts_ns"].max()),
        }
        for market_slug, group in frame.loc[
            frame["market_fully_without_trusted_live_reference"]
            & ~frame["market_fully_without_live_reference"]
        ].groupby("market_slug", sort=True)
    ])
    quality_rows.extend([
        {
            "dataset_name": dataset_name,
            "market_slug": str(market_slug),
            "exclusion_reason": "market_fully_without_token_mid_complete",
            "rows": int(group.shape[0]),
            "first_snapshot_ts_ns": int(group["snapshot_ts_ns"].min()),
            "last_snapshot_ts_ns": int(group["snapshot_ts_ns"].max()),
        }
        for market_slug, group in frame.loc[
            frame["market_fully_without_token_mid_complete"]
            & ~frame["market_fully_without_live_reference"]
        ].groupby("market_slug", sort=True)
    ])
    quality_rows.extend([
        {
            "dataset_name": dataset_name,
            "market_slug": str(market_slug),
            "exclusion_reason": "market_5m_complete_rate_below_80pct",
            "rows": int(group.shape[0]),
            "first_snapshot_ts_ns": int(group["snapshot_ts_ns"].min()),
            "last_snapshot_ts_ns": int(group["snapshot_ts_ns"].max()),
        }
        for market_slug, group in frame.loc[
            frame["training_feature_eligible"].astype("bool")
            & ~frame["market_full_matrix_trainable"].astype("bool")
            & ~frame["market_fully_without_live_reference"].astype("bool")
        ].groupby("market_slug", sort=True)
    ])
    return frame, quality_rows


def _assert_no_forbidden_columns(frame: pd.DataFrame) -> None:
    forbidden = [column for column in frame.columns if "fng" in column.lower()]
    if forbidden:
        raise ValueError(f"FNG columns are forbidden in Phase 3 outputs: {sorted(forbidden)}")
    present_dead = [column for column in DEAD_COLUMNS if column in frame.columns]
    if present_dead:
        raise ValueError(f"Dead columns must be removed from Phase 3 outputs: {sorted(present_dead)}")


def _build_market_live_window(
    *,
    label: MarketLabel,
    live_reference: LiveReferenceIndex,
) -> MarketLiveWindow:
    seconds = list(range(label.market_open_s, label.market_close_s))
    prices: list[float | None] = []
    signs: list[int] = []
    above_prefix = [0]
    below_prefix = [0]
    last_above = 0
    last_below = 0
    for second in seconds:
        index = bisect_right(live_reference.ts_seconds, second) - 1
        price = live_reference.price_at_index(index)
        prices.append(price)
        if price is None:
            signs.append(0)
            above_prefix.append(last_above)
            below_prefix.append(last_below)
            continue
        sign = 1 if price - label.price_to_beat >= 0 else -1
        signs.append(sign)
        if sign >= 0:
            last_above += 1
        else:
            last_below += 1
        above_prefix.append(last_above)
        below_prefix.append(last_below)

    future_flip = [False] * len(seconds)
    future_seen_up = False
    future_seen_down = False
    for index in range(len(seconds) - 1, -1, -1):
        sign = signs[index]
        if sign > 0:
            future_flip[index] = future_seen_down
            future_seen_up = True
        elif sign < 0:
            future_flip[index] = future_seen_up
            future_seen_down = True
        else:
            future_flip[index] = future_seen_up or future_seen_down

    terminal_price = None
    for price in reversed(prices):
        if price is not None:
            terminal_price = price
            break
    terminal_delta = terminal_price - label.price_to_beat if terminal_price is not None else None
    terminal_margin = abs(terminal_delta) if terminal_delta is not None else None
    return MarketLiveWindow(
        seconds=seconds,
        prices=prices,
        signs=signs,
        above_prefix=above_prefix,
        below_prefix=below_prefix,
        future_flip=future_flip,
        terminal_price=terminal_price,
        terminal_delta_to_strike=terminal_delta,
        terminal_margin_to_strike=terminal_margin,
    )


def _return_over_horizon(index: int, live_reference: LiveReferenceIndex, horizon_seconds: int) -> float | None:
    if index < 0:
        return None
    previous_index = index - horizon_seconds
    current = live_reference.price_at_index(index)
    if current is None or previous_index < 0:
        return None
    previous = live_reference.price_at_index(previous_index)
    if previous in (None, 0.0):
        return None
    return (current - previous) / previous


def _realized_volatility(index: int, live_reference: LiveReferenceIndex, window_seconds: int) -> float | None:
    if index <= 0:
        return None
    start = max(1, index - window_seconds + 1)
    returns: list[float] = []
    for current_index in range(start, index + 1):
        current = live_reference.price_at_index(current_index)
        previous = live_reference.price_at_index(current_index - 1) if current_index - 1 >= 0 else None
        if current in (None, 0.0) or previous in (None, 0.0):
            continue
        returns.append(math.log(current / previous))
    return _std(returns)


def _window_price_summary(
    *,
    live_reference: LiveReferenceIndex,
    end_index: int,
    window_seconds: int,
) -> dict[str, float | None]:
    if end_index < 0:
        return {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "range": None,
        }
    start_index = max(0, end_index - window_seconds + 1)
    values = [
        value
        for idx in range(start_index, end_index + 1)
        if (value := live_reference.price_at_index(idx)) is not None
    ]
    if not values:
        return {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "range": None,
        }
    return {
        "open": values[0],
        "high": max(values),
        "low": min(values),
        "close": values[-1],
        "range": max(values) - min(values),
    }


def _direction_label(current: float | None, future: float | None) -> int:
    if current is None or future is None:
        return -1
    return 1 if future >= current else 0


def _recent_above_below_seconds(
    *,
    market_live: MarketLiveWindow,
    second: int,
    lookback_seconds: int,
) -> tuple[int, int]:
    if not market_live.seconds:
        return 0, 0
    start_second = max(market_live.seconds[0], second - lookback_seconds + 1)
    start_index = max(0, start_second - market_live.seconds[0])
    end_index = min(len(market_live.seconds) - 1, second - market_live.seconds[0])
    above = market_live.above_prefix[end_index + 1] - market_live.above_prefix[start_index]
    below = market_live.below_prefix[end_index + 1] - market_live.below_prefix[start_index]
    return above, below


def _future_flip_for_second(market_live: MarketLiveWindow, second: int) -> bool:
    if not market_live.seconds:
        return False
    if second < market_live.seconds[0]:
        return market_live.future_flip[0]
    index = min(len(market_live.future_flip) - 1, second - market_live.seconds[0])
    return bool(market_live.future_flip[index])


def _market_interval_context_features(
    *,
    label: MarketLabel,
    snapshot_ts_ns: int,
    live_reference: LiveReferenceIndex,
    market_history_features: dict[str, Any],
) -> dict[str, Any]:
    live_index = live_reference.asof_index(snapshot_ts_ns)
    previous_interval_end_index = live_reference.asof_index(label.market_open_ts_ns - SECOND_NS)
    previous_interval = _window_price_summary(
        live_reference=live_reference,
        end_index=previous_interval_end_index,
        window_seconds=300,
    )
    btc_return_60s = _return_over_horizon(live_index, live_reference, 60)
    btc_return_180s = _return_over_horizon(live_index, live_reference, 180)
    btc_return_300s = _return_over_horizon(live_index, live_reference, 300)
    btc_return_900s = _return_over_horizon(live_index, live_reference, 900)
    btc_return_1800s = _return_over_horizon(live_index, live_reference, 1800)
    btc_realized_vol_60s = _realized_volatility(live_index, live_reference, 60)
    btc_realized_vol_300s = _realized_volatility(live_index, live_reference, 300)
    btc_realized_vol_900s = _realized_volatility(live_index, live_reference, 900)
    return {
        **market_history_features,
        "btc_return_60s": btc_return_60s if btc_return_60s is not None else math.nan,
        "btc_return_180s": btc_return_180s if btc_return_180s is not None else math.nan,
        "btc_return_300s": btc_return_300s if btc_return_300s is not None else math.nan,
        "btc_return_900s": btc_return_900s if btc_return_900s is not None else math.nan,
        "btc_return_1800s": btc_return_1800s if btc_return_1800s is not None else math.nan,
        "btc_realized_vol_60s": btc_realized_vol_60s if btc_realized_vol_60s is not None else math.nan,
        "btc_realized_vol_300s": btc_realized_vol_300s if btc_realized_vol_300s is not None else math.nan,
        "btc_realized_vol_900s": btc_realized_vol_900s if btc_realized_vol_900s is not None else math.nan,
        "prev_interval_btc_open": previous_interval["open"] if previous_interval["open"] is not None else math.nan,
        "prev_interval_btc_high": previous_interval["high"] if previous_interval["high"] is not None else math.nan,
        "prev_interval_btc_low": previous_interval["low"] if previous_interval["low"] is not None else math.nan,
        "prev_interval_btc_close": previous_interval["close"] if previous_interval["close"] is not None else math.nan,
        "prev_interval_btc_range": previous_interval["range"] if previous_interval["range"] is not None else math.nan,
    }


def _book_state_features(state: TokenBookState | None, prefix: str) -> dict[str, Any]:
    if state is None:
        return {
            f"{prefix}_best_bid": math.nan,
            f"{prefix}_best_ask": math.nan,
            f"{prefix}_mid": math.nan,
            f"{prefix}_spread": math.nan,
            f"{prefix}_microprice": math.nan,
            f"{prefix}_bid_depth_total": math.nan,
            f"{prefix}_ask_depth_total": math.nan,
            f"{prefix}_imbalance": math.nan,
            f"{prefix}_last_trade_price": math.nan,
            f"{prefix}_tick_size": math.nan,
        }
    return {
        f"{prefix}_best_bid": state.best_bid if state.best_bid is not None else math.nan,
        f"{prefix}_best_ask": state.best_ask if state.best_ask is not None else math.nan,
        f"{prefix}_mid": state.mid if state.mid is not None else math.nan,
        f"{prefix}_spread": state.spread if state.spread is not None else math.nan,
        f"{prefix}_microprice": (
            microprice
            if (microprice := _microprice(
                best_bid=state.best_bid,
                best_ask=state.best_ask,
                bid_depth_total=state.bid_depth_total,
                ask_depth_total=state.ask_depth_total,
            )) is not None
            else math.nan
        ),
        f"{prefix}_bid_depth_total": state.bid_depth_total if state.bid_depth_total is not None else math.nan,
        f"{prefix}_ask_depth_total": state.ask_depth_total if state.ask_depth_total is not None else math.nan,
        f"{prefix}_imbalance": (
            imbalance
            if (imbalance := _imbalance(state.bid_depth_total, state.ask_depth_total)) is not None
            else math.nan
        ),
        f"{prefix}_last_trade_price": state.last_trade_price if state.last_trade_price is not None else math.nan,
        f"{prefix}_tick_size": state.tick_size if state.tick_size is not None else math.nan,
    }


def _event_count_features(index: TokenL2Index | None, prefix: str, snapshot_ts_ns: int) -> dict[str, Any]:
    if index is None:
        return {
            f"{prefix}_l2_book_updates_1s": 0,
            f"{prefix}_price_change_updates_1s": 0,
            f"{prefix}_best_bid_ask_updates_1s": 0,
            f"{prefix}_trade_updates_1s": 0,
            f"{prefix}_price_change_updates_5s": 0,
            f"{prefix}_trade_count_5s": 0,
            f"{prefix}_signed_trade_size_5s": 0.0,
            f"{prefix}_absolute_trade_size_5s": 0.0,
        }
    trade_count_5s, signed_trade_size_5s, absolute_trade_size_5s = index.recent_trade_summary(
        asof_ts_ns=snapshot_ts_ns,
        lookback_ns=5 * SECOND_NS,
    )
    return {
        f"{prefix}_l2_book_updates_1s": index.recent_event_count("book", asof_ts_ns=snapshot_ts_ns, lookback_ns=SECOND_NS),
        f"{prefix}_price_change_updates_1s": index.recent_event_count(
            "price_change",
            asof_ts_ns=snapshot_ts_ns,
            lookback_ns=SECOND_NS,
        ),
        f"{prefix}_best_bid_ask_updates_1s": index.recent_event_count(
            "best_bid_ask",
            asof_ts_ns=snapshot_ts_ns,
            lookback_ns=SECOND_NS,
        ),
        f"{prefix}_trade_updates_1s": index.recent_event_count(
            "last_trade_price",
            asof_ts_ns=snapshot_ts_ns,
            lookback_ns=SECOND_NS,
        ),
        f"{prefix}_price_change_updates_5s": index.recent_event_count(
            "price_change",
            asof_ts_ns=snapshot_ts_ns,
            lookback_ns=5 * SECOND_NS,
        ),
        f"{prefix}_trade_count_5s": trade_count_5s,
        f"{prefix}_signed_trade_size_5s": signed_trade_size_5s,
        f"{prefix}_absolute_trade_size_5s": absolute_trade_size_5s,
    }


def _sample_times_for_coarse(label: MarketLabel) -> list[int]:
    end_ns = label.market_close_ts_ns - (60 * SECOND_NS)
    if end_ns < label.market_open_ts_ns:
        return []
    return list(range(label.market_open_ts_ns, end_ns + 1, SECOND_NS))


def _sample_times_for_dense_close(label: MarketLabel) -> list[int]:
    values: list[int] = []
    start_ns = label.market_close_ts_ns - (60 * SECOND_NS)
    mid_ns = label.market_close_ts_ns - (10 * SECOND_NS)
    if start_ns < label.market_open_ts_ns:
        start_ns = label.market_open_ts_ns
    current = start_ns
    while current < mid_ns:
        values.append(current)
        current += 250 * MILLISECOND_NS
    current = mid_ns
    while current < label.market_close_ts_ns:
        values.append(current)
        current += 100 * MILLISECOND_NS
    return values


def _sample_times_for_market_interval(label: MarketLabel) -> dict[str, int]:
    return {
        "preopen": label.market_open_ts_ns - SECOND_NS,
        "first15s": label.market_open_ts_ns + (15 * SECOND_NS),
        "first30s": label.market_open_ts_ns + (30 * SECOND_NS),
    }


def _build_snapshot_row(
    *,
    label: MarketLabel,
    snapshot_ts_ns: int,
    sample_family: str,
    sample_bucket_ms: int,
    live_reference: LiveReferenceIndex,
    market_live: MarketLiveWindow,
    up_index: TokenL2Index | None,
    down_index: TokenL2Index | None,
    rest_index: RestContextIndex | None,
) -> dict[str, Any]:
    live_index = live_reference.asof_index(snapshot_ts_ns)
    live_row = live_reference.row_at_index(live_index)
    live_price = live_reference.price_at_index(live_index)
    live_bias_bootstrapped = bool(live_row["bias_bootstrapped"]) if live_row is not None else True
    live_bias_active = bool(live_row["bias_active"]) if live_row is not None else False
    live_bias_carried_forward = bool(live_row["bias_carried_forward"]) if live_row is not None else False
    live_bias_mode = str(live_row["bias_mode"]) if live_row is not None else "unavailable"
    live_bias_state_stale = bool(live_row["bias_state_stale"]) if live_row is not None else True
    live_bias_neutralized = bool(live_row["bias_neutralized"]) if live_row is not None else True
    live_reference_strict_trusted = bool(live_price is not None and live_bias_active)
    live_reference_carried_trusted = bool(live_price is not None and (live_bias_active or live_bias_carried_forward))
    live_reference_trusted = live_reference_carried_trusted
    snapshot_second = snapshot_ts_ns // SECOND_NS
    recent_vol_15s = _realized_volatility(live_index, live_reference, 15) if live_index >= 0 else None
    recent_vol_5s = _realized_volatility(live_index, live_reference, 5) if live_index >= 0 else None
    delta_to_strike = live_price - label.price_to_beat if live_price is not None else None
    abs_delta_to_strike = abs(delta_to_strike) if delta_to_strike is not None else None
    delta_over_recent_vol = _safe_div(delta_to_strike, recent_vol_15s)
    above_60s, below_60s = _recent_above_below_seconds(
        market_live=market_live,
        second=int(snapshot_second),
        lookback_seconds=60,
    )

    btc_return_1s = _return_over_horizon(live_index, live_reference, 1) if live_index >= 0 else None
    btc_return_3s = _return_over_horizon(live_index, live_reference, 3) if live_index >= 0 else None
    btc_return_5s = _return_over_horizon(live_index, live_reference, 5) if live_index >= 0 else None
    btc_return_10s = _return_over_horizon(live_index, live_reference, 10) if live_index >= 0 else None
    acceleration_proxy = (
        btc_return_1s - btc_return_5s
        if btc_return_1s is not None and btc_return_5s is not None
        else None
    )
    jerk_proxy = (
        btc_return_1s - (2.0 * btc_return_3s) + btc_return_5s
        if btc_return_1s is not None and btc_return_3s is not None and btc_return_5s is not None
        else None
    )

    up_state = up_index.latest_state(snapshot_ts_ns) if up_index is not None else None
    down_state = down_index.latest_state(snapshot_ts_ns) if down_index is not None else None
    up_mid = up_state.mid if up_state is not None else None
    down_mid = down_state.mid if down_state is not None else None
    coupling_residual = (up_mid + down_mid - 1.0) if up_mid is not None and down_mid is not None else None

    winning_mid = up_mid if label.resolved_side == "up" else down_mid
    hold_to_close_edge_vs_mid = (1.0 - winning_mid) if winning_mid is not None else None

    rest_state = rest_index.latest(snapshot_ts_ns) if rest_index is not None else None

    row: dict[str, Any] = {
        "snapshot_ts_ns": snapshot_ts_ns,
        "snapshot_ts_iso": ns_to_iso(snapshot_ts_ns),
        "snapshot_ts_seconds": int(snapshot_ts_ns // SECOND_NS),
        "sample_family": sample_family,
        "sample_bucket_ms": sample_bucket_ms,
        "market_slug": label.market_slug,
        "market_id": label.market_id,
        "condition_id": label.condition_id,
        "market_open_ts_ns": label.market_open_ts_ns,
        "market_close_ts_ns": label.market_close_ts_ns,
        "market_open_iso": ns_to_iso(label.market_open_ts_ns),
        "market_close_iso": ns_to_iso(label.market_close_ts_ns),
        "price_to_beat": label.price_to_beat,
        "resolved_side": label.resolved_side,
        "resolved_side_label": 1 if label.resolved_side == "up" else 0,
        "snapshot_hour_utc": int(datetime.fromtimestamp(snapshot_ts_ns / SECOND_NS, tz=UTC).hour),
        "snapshot_minute_utc": int(datetime.fromtimestamp(snapshot_ts_ns / SECOND_NS, tz=UTC).minute),
        "snapshot_day_of_week": int(datetime.fromtimestamp(snapshot_ts_ns / SECOND_NS, tz=UTC).weekday()),
        "snapshot_second_of_day_utc": int(snapshot_second % 86_400),
        "market_interval_seconds": int((label.market_close_ts_ns - label.market_open_ts_ns) / SECOND_NS),
        "t_since_open_s": (snapshot_ts_ns - label.market_open_ts_ns) / SECOND_NS,
        "t_to_close_s": (label.market_close_ts_ns - snapshot_ts_ns) / SECOND_NS,
        "session_progress_fraction": (
            session_progress_fraction
            if (session_progress_fraction := _safe_div(
                float(snapshot_ts_ns - label.market_open_ts_ns),
                float(label.market_close_ts_ns - label.market_open_ts_ns),
            )) is not None
            else math.nan
        ),
        "phase_bucket": _phase_bucket((label.market_close_ts_ns - snapshot_ts_ns) / SECOND_NS),
        "live_btc_usd": live_price if live_price is not None else math.nan,
        "live_price_valid": live_price is not None,
        "live_reference_trusted": live_reference_trusted,
        "live_reference_strict_trusted": live_reference_strict_trusted,
        "live_reference_carried_trusted": live_reference_carried_trusted,
        "live_price_degraded": bool(live_row["degraded"]) if live_row is not None else False,
        "live_source_count": int(live_row["source_count"]) if live_row is not None else 0,
        "live_bias_bootstrapped": live_bias_bootstrapped,
        "live_bias_active": live_bias_active,
        "live_bias_carried_forward": live_bias_carried_forward,
        "live_bias_mode": live_bias_mode,
        "live_bias_state_stale": live_bias_state_stale,
        "live_bias_neutralized": live_bias_neutralized,
        "live_bias_observation_count": int(live_row["bias_observation_count"]) if live_row is not None else 0,
        "live_active_window_bias": (
            float(live_row["active_window_bias"])
            if live_row is not None and not math.isnan(float(live_row["active_window_bias"]))
            else math.nan
        ),
        "live_bias_state_age_seconds": (
            float(live_row["bias_state_age_seconds"])
            if live_row is not None and not math.isnan(float(live_row["bias_state_age_seconds"]))
            else math.nan
        ),
        "live_applied_bias_age_seconds": (
            float(live_row["applied_bias_age_seconds"])
            if live_row is not None and not math.isnan(float(live_row["applied_bias_age_seconds"]))
            else math.nan
        ),
        "binance_spot_mid": (
            float(live_row["binance_spot_mid"])
            if live_row is not None and not math.isnan(float(live_row["binance_spot_mid"]))
            else math.nan
        ),
        "binance_usdm_mid": (
            float(live_row["binance_usdm_mid"])
            if live_row is not None and not math.isnan(float(live_row["binance_usdm_mid"]))
            else math.nan
        ),
        "hyperliquid_mid": (
            float(live_row["hyperliquid_mid"])
            if live_row is not None and not math.isnan(float(live_row["hyperliquid_mid"]))
            else math.nan
        ),
        "delta_to_strike": delta_to_strike if delta_to_strike is not None else math.nan,
        "abs_delta_to_strike": abs_delta_to_strike if abs_delta_to_strike is not None else math.nan,
        "delta_to_strike_over_recent_vol": delta_over_recent_vol if delta_over_recent_vol is not None else math.nan,
        "delta_sign": _signed_delta_to_label(delta_to_strike) or 0,
        "time_spent_above_strike_recent_s": above_60s,
        "time_spent_below_strike_recent_s": below_60s,
        "btc_return_1s": btc_return_1s if btc_return_1s is not None else math.nan,
        "btc_return_3s": btc_return_3s if btc_return_3s is not None else math.nan,
        "btc_return_5s": btc_return_5s if btc_return_5s is not None else math.nan,
        "btc_return_10s": btc_return_10s if btc_return_10s is not None else math.nan,
        "btc_realized_vol_5s": recent_vol_5s if recent_vol_5s is not None else math.nan,
        "btc_realized_vol_15s": recent_vol_15s if recent_vol_15s is not None else math.nan,
        "btc_acceleration_proxy": acceleration_proxy if acceleration_proxy is not None else math.nan,
        "btc_jerk_proxy": jerk_proxy if jerk_proxy is not None else math.nan,
        "cross_source_spread_usd": (
            float(live_row["max_cross_source_spread"])
            if live_row is not None and not math.isnan(float(live_row["max_cross_source_spread"]))
            else math.nan
        ),
        "spot_perp_divergence_usd": (
            float(live_row["binance_spot_mid"]) - float(live_row["binance_usdm_mid"])
            if live_row is not None
            and not math.isnan(float(live_row["binance_spot_mid"]))
            and not math.isnan(float(live_row["binance_usdm_mid"]))
            else math.nan
        ),
        "flip_happened_after_t": _future_flip_for_second(market_live, int(snapshot_second)),
        "terminal_delta_to_strike": (
            market_live.terminal_delta_to_strike if market_live.terminal_delta_to_strike is not None else math.nan
        ),
        "terminal_margin_to_strike": (
            market_live.terminal_margin_to_strike if market_live.terminal_margin_to_strike is not None else math.nan
        ),
        "hold_to_close_edge_vs_mid": hold_to_close_edge_vs_mid if hold_to_close_edge_vs_mid is not None else math.nan,
        "yes_no_coupling_residual": coupling_residual if coupling_residual is not None else math.nan,
        "maker_base_fee_bps": (
            rest_state.maker_base_fee_bps if rest_state and rest_state.maker_base_fee_bps is not None else math.nan
        ),
        "taker_base_fee_bps": (
            rest_state.taker_base_fee_bps if rest_state and rest_state.taker_base_fee_bps is not None else math.nan
        ),
        "market_rewards_min_size": (
            rest_state.rewards_min_size if rest_state and rest_state.rewards_min_size is not None else math.nan
        ),
        "market_rewards_max_spread": (
            rest_state.rewards_max_spread if rest_state and rest_state.rewards_max_spread is not None else math.nan
        ),
        "has_canonical_resolution": True,
        "is_trainable": True,
    }

    row.update(_book_state_features(up_state, "up_token"))
    row.update(_book_state_features(down_state, "down_token"))
    row.update(_event_count_features(up_index, "up_token", snapshot_ts_ns))
    row.update(_event_count_features(down_index, "down_token", snapshot_ts_ns))
    row["coupling_anomaly"] = bool(abs(coupling_residual) > 0.05) if coupling_residual is not None else False
    row["crossed_book"] = bool(
        _crossed_book(row.get("up_token_best_bid"), row.get("up_token_best_ask"))
        or _crossed_book(row.get("down_token_best_bid"), row.get("down_token_best_ask"))
    )
    up_bid_present = _has_numeric(row.get("up_token_best_bid"))
    up_ask_present = _has_numeric(row.get("up_token_best_ask"))
    down_bid_present = _has_numeric(row.get("down_token_best_bid"))
    down_ask_present = _has_numeric(row.get("down_token_best_ask"))
    row["up_token_mid_complete"] = bool(_has_numeric(row.get("up_token_mid")))
    row["down_token_mid_complete"] = bool(_has_numeric(row.get("down_token_mid")))
    row["token_mid_complete"] = bool(row["up_token_mid_complete"] and row["down_token_mid_complete"])
    row["up_token_one_sided_book"] = bool(up_bid_present ^ up_ask_present)
    row["down_token_one_sided_book"] = bool(down_bid_present ^ down_ask_present)
    row["one_sided_book"] = bool(row["up_token_one_sided_book"] or row["down_token_one_sided_book"])
    row["spot_perp_available"] = bool(_has_numeric(row.get("spot_perp_divergence_usd")))
    return row


def _build_market_interval_rows(
    *,
    label: MarketLabel,
    live_reference: LiveReferenceIndex,
    market_live: MarketLiveWindow,
    up_index: TokenL2Index | None,
    down_index: TokenL2Index | None,
    rest_index: RestContextIndex | None,
    market_history_features: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    views: dict[str, list[dict[str, Any]]] = {key: [] for key in MARKET_INTERVAL_DATASETS}
    open_live_price = live_reference.price_at_index(live_reference.asof_index(label.market_open_ts_ns))
    first_15s_price = live_reference.price_at_index(live_reference.asof_index(label.market_open_ts_ns + (15 * SECOND_NS)))
    first_30s_price = live_reference.price_at_index(live_reference.asof_index(label.market_open_ts_ns + (30 * SECOND_NS)))
    for view_name, snapshot_ts_ns in _sample_times_for_market_interval(label).items():
        row = _build_snapshot_row(
            label=label,
            snapshot_ts_ns=snapshot_ts_ns,
            sample_family=view_name,
            sample_bucket_ms=1000,
            live_reference=live_reference,
            market_live=market_live,
            up_index=up_index,
            down_index=down_index,
            rest_index=rest_index,
        )
        row.update(
            _market_interval_context_features(
                label=label,
                snapshot_ts_ns=snapshot_ts_ns,
                live_reference=live_reference,
                market_history_features=market_history_features,
            )
        )
        row.update(
            {
                "market_view": view_name,
                "early_direction_sign": _direction_label(open_live_price, first_15s_price),
                "close_to_open_btc_sign": _direction_label(label.price_to_beat, market_live.terminal_price),
                "first_30s_drift_sign": _direction_label(open_live_price, first_30s_price),
            }
        )
        views[view_name].append(row)
    return views


def _build_day_datasets(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(task["root"])
    target_date = date.fromisoformat(str(task["target_date"]))
    date_from = date.fromisoformat(str(task["date_from"]))
    policy_path = Path(task["policy_path"])
    dataset_output_dir = Path(task["dataset_output_dir"])
    skip_existing = bool(task.get("skip_existing", True))
    labels = [MarketLabel(**payload) for payload in task["labels"]]
    history_features_by_market_slug = {
        str(key): dict(value) for key, value in dict(task.get("history_features_by_market_slug") or {}).items()
    }
    if skip_existing:
        existing = _existing_phase3_day_summary(dataset_output_dir=dataset_output_dir, target_date=target_date)
        if existing is not None:
            return existing
    if not labels:
        empty_outputs = _write_empty_phase3_day_outputs(
            dataset_output_dir=dataset_output_dir,
            target_date=target_date,
        )
        return {
            "target_date": target_date.isoformat(),
            "coarse_rows": 0,
            "dense_rows": 0,
            "market_interval_rows": {key: 0 for key in MARKET_INTERVAL_DATASETS},
            "hashes": empty_outputs["hashes"],
            "paths": empty_outputs["paths"],
            "exclusions": {},
            "quality_registry_rows": [],
        }

    shard_dates = [target_date]
    previous_date = target_date - timedelta(days=1)
    shard_dates.insert(0, previous_date)

    market_slugs = {label.market_slug for label in labels}
    live_reference = _load_live_reference_index(root=root, shard_dates=shard_dates)
    l2_indices, l2_exclusions = _load_l2_indices_for_day(
        root=root,
        policy_path=policy_path,
        shard_dates=shard_dates,
        market_slugs=market_slugs,
    )
    rest_indices, rest_exclusions = _load_rest_indices_for_day(
        root=root,
        policy_path=policy_path,
        shard_dates=shard_dates,
        market_slugs=market_slugs,
    )

    coarse_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    market_interval_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in MARKET_INTERVAL_DATASETS}

    for label in sorted(labels, key=lambda item: (item.market_open_ts_ns, item.market_slug)):
        market_live = _build_market_live_window(label=label, live_reference=live_reference)
        up_index = l2_indices.get(label.market_slug, {}).get("up")
        down_index = l2_indices.get(label.market_slug, {}).get("down")
        rest_index = rest_indices.get(label.market_slug)

        for snapshot_ts_ns in _sample_times_for_coarse(label):
            coarse_rows.append(
                _build_snapshot_row(
                    label=label,
                    snapshot_ts_ns=snapshot_ts_ns,
                    sample_family="coarse",
                    sample_bucket_ms=1000,
                    live_reference=live_reference,
                    market_live=market_live,
                    up_index=up_index,
                    down_index=down_index,
                    rest_index=rest_index,
                )
            )
        for snapshot_ts_ns in _sample_times_for_dense_close(label):
            dense_rows.append(
                _build_snapshot_row(
                    label=label,
                    snapshot_ts_ns=snapshot_ts_ns,
                    sample_family="dense_close",
                    sample_bucket_ms=250 if snapshot_ts_ns < label.market_close_ts_ns - (10 * SECOND_NS) else 100,
                    live_reference=live_reference,
                    market_live=market_live,
                    up_index=up_index,
                    down_index=down_index,
                    rest_index=rest_index,
                )
            )
        interval_views = _build_market_interval_rows(
            label=label,
            live_reference=live_reference,
            market_live=market_live,
            up_index=up_index,
            down_index=down_index,
            rest_index=rest_index,
            market_history_features=history_features_by_market_slug.get(label.market_slug, {}),
        )
        for view_name, rows in interval_views.items():
            market_interval_rows[view_name].extend(rows)

    hashes: dict[str, str] = {}
    written_paths: dict[str, str] = {}
    quality_registry_rows: list[dict[str, Any]] = []
    coarse_row_count = len(coarse_rows)
    dense_row_count = len(dense_rows)
    market_interval_row_counts = {key: len(value) for key, value in market_interval_rows.items()}

    if coarse_rows:
        frame = pd.DataFrame(coarse_rows).sort_values(["snapshot_ts_ns", "market_slug"], kind="stable")
        frame, quality_rows = _apply_training_contract(frame, dataset_name=COARSE_DATASET_NAME)
        _assert_no_forbidden_columns(frame)
        path = dataset_output_dir / COARSE_DATASET_NAME / f"{target_date.isoformat()}.parquet"
        _write_parquet(frame, path)
        hashes[COARSE_DATASET_NAME] = _sha256(path)
        written_paths[COARSE_DATASET_NAME] = str(path)
        quality_registry_rows.extend(quality_rows)

    if dense_rows:
        frame = pd.DataFrame(dense_rows).sort_values(["snapshot_ts_ns", "market_slug"], kind="stable")
        frame, quality_rows = _apply_training_contract(frame, dataset_name=DENSE_DATASET_NAME)
        _assert_no_forbidden_columns(frame)
        path = dataset_output_dir / DENSE_DATASET_NAME / f"{target_date.isoformat()}.parquet"
        _write_parquet(frame, path)
        hashes[DENSE_DATASET_NAME] = _sha256(path)
        written_paths[DENSE_DATASET_NAME] = str(path)
        quality_registry_rows.extend(quality_rows)

    for view_name, dataset_name in MARKET_INTERVAL_DATASETS.items():
        rows = market_interval_rows[view_name]
        if not rows:
            continue
        frame = pd.DataFrame(rows).sort_values(["snapshot_ts_ns", "market_slug"], kind="stable")
        frame, quality_rows = _apply_training_contract(frame, dataset_name=dataset_name)
        _assert_no_forbidden_columns(frame)
        path = dataset_output_dir / dataset_name / f"{target_date.isoformat()}.parquet"
        _write_parquet(frame, path)
        hashes[dataset_name] = _sha256(path)
        written_paths[dataset_name] = str(path)
        quality_registry_rows.extend(quality_rows)

    exclusion_summary: dict[str, Counter] = defaultdict(Counter)
    for source_map in (l2_exclusions, rest_exclusions):
        for source_name, bucket in source_map.items():
            exclusion_summary[source_name].update(bucket)
    return {
        "target_date": target_date.isoformat(),
        "coarse_rows": coarse_row_count,
        "dense_rows": dense_row_count,
        "market_interval_rows": market_interval_row_counts,
        "hashes": hashes,
        "paths": written_paths,
        "exclusions": {key: dict(value) for key, value in exclusion_summary.items()},
        "quality_registry_rows": quality_registry_rows,
    }


def _phase3_output_paths(dataset_output_dir: Path, target_date: date) -> dict[str, Path]:
    day = target_date.isoformat()
    return {
        COARSE_DATASET_NAME: dataset_output_dir / COARSE_DATASET_NAME / f"{day}.parquet",
        DENSE_DATASET_NAME: dataset_output_dir / DENSE_DATASET_NAME / f"{day}.parquet",
        **{
            dataset_name: dataset_output_dir / dataset_name / f"{day}.parquet"
            for dataset_name in MARKET_INTERVAL_DATASETS.values()
        },
    }


def _quality_rows_from_existing_frame(frame: pd.DataFrame, *, dataset_name: str) -> list[dict[str, Any]]:
    try:
        _, quality_rows = _apply_training_contract(frame, dataset_name=dataset_name)
        return quality_rows
    except Exception:
        return []


def _existing_phase3_day_summary(*, dataset_output_dir: Path, target_date: date) -> dict[str, Any] | None:
    output_paths = _phase3_output_paths(dataset_output_dir, target_date)
    if not all(path.exists() and path.stat().st_size > 0 for path in output_paths.values()):
        return None

    row_counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    quality_registry_rows: list[dict[str, Any]] = []
    try:
        for dataset_name, path in output_paths.items():
            frame = pd.read_parquet(path)
            row_counts[dataset_name] = int(len(frame))
            hashes[dataset_name] = _sha256(path)
            quality_registry_rows.extend(_quality_rows_from_existing_frame(frame, dataset_name=dataset_name))
    except Exception:
        return None

    return {
        "target_date": target_date.isoformat(),
        "coarse_rows": row_counts[COARSE_DATASET_NAME],
        "dense_rows": row_counts[DENSE_DATASET_NAME],
        "market_interval_rows": {
            view_name: row_counts[dataset_name]
            for view_name, dataset_name in MARKET_INTERVAL_DATASETS.items()
        },
        "hashes": hashes,
        "paths": {dataset_name: str(path) for dataset_name, path in output_paths.items()},
        "exclusions": {},
        "quality_registry_rows": quality_registry_rows,
        "skipped_existing": True,
    }


def _write_schema_contracts(
    *,
    dataset_output_dir: Path,
    schema_dir: Path,
) -> dict[str, str]:
    schema_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}

    def schema_training_eligible(column: str) -> bool:
        non_feature_columns = {
            "snapshot_ts_ns",
            "snapshot_ts_iso",
            "snapshot_ts_seconds",
            "sample_family",
            "sample_bucket_ms",
            "market_slug",
            "market_id",
            "condition_id",
            "market_open_ts_ns",
            "market_close_ts_ns",
            "market_open_iso",
            "market_close_iso",
            "resolved_side",
            "resolved_side_label",
            "market_view",
            "early_direction_sign",
            "close_to_open_btc_sign",
            "first_30s_drift_sign",
            "has_canonical_resolution",
            "is_trainable",
            "market_fully_without_live_reference",
            "market_fully_without_trusted_live_reference",
            "market_fully_without_token_mid_complete",
            "market_5m_training_row_count",
            "market_5m_complete_row_count",
            "market_5m_complete_rate",
            "market_full_matrix_trainable",
            "training_feature_eligible",
            "exclusion_reason",
            "complete_feature_matrix_eligible",
            "feature_matrix_exclusion_reason",
            "full_matrix_training_eligible",
            "full_matrix_exclusion_reason",
            "coupling_anomaly",
            "crossed_book",
            "live_reference_trusted",
            "live_reference_strict_trusted",
            "live_reference_carried_trusted",
            "live_bias_bootstrapped",
            "live_bias_active",
            "live_bias_carried_forward",
            "live_bias_mode",
            "live_bias_state_stale",
            "live_bias_neutralized",
            "live_active_window_bias",
            "live_bias_state_age_seconds",
            "live_applied_bias_age_seconds",
            "up_token_mid_complete",
            "down_token_mid_complete",
            "token_mid_complete",
            "up_token_one_sided_book",
            "down_token_one_sided_book",
            "one_sided_book",
            "spot_perp_available",
        }
        return column not in non_feature_columns and column not in LEAKAGE_ONLY_COLUMNS

    def schema_source(column: str) -> str:
        if column in {"price_to_beat"}:
            return "polymarket_strike"
        if column in {"resolved_side", "resolved_side_label"}:
            return "polymarket_resolution"
        if column.startswith("up_token_") or column.startswith("down_token_") or column == "yes_no_coupling_residual":
            return "polymarket_l2"
        if column.startswith("maker_") or column.startswith("taker_") or column.startswith("market_rewards_"):
            return "polymarket_rest"
        if (
            column.startswith("live_")
            or column.startswith("binance_")
            or column.startswith("hyperliquid_")
            or column.startswith("btc_")
            or column in {
                "delta_to_strike",
                "abs_delta_to_strike",
                "delta_to_strike_over_recent_vol",
                "delta_sign",
                "time_spent_above_strike_recent_s",
                "time_spent_below_strike_recent_s",
                "cross_source_spread_usd",
                "spot_perp_divergence_usd",
                "live_reference_trusted",
                "live_reference_strict_trusted",
                "live_reference_carried_trusted",
                "live_bias_bootstrapped",
                "live_bias_active",
                "live_bias_carried_forward",
                "live_bias_mode",
                "live_bias_state_stale",
                "live_bias_neutralized",
                "live_bias_observation_count",
                "live_active_window_bias",
                "live_bias_state_age_seconds",
                "live_applied_bias_age_seconds",
                "terminal_delta_to_strike",
                "terminal_margin_to_strike",
                "flip_happened_after_t",
            }
        ):
            return "live_reference_events_v1"
        if column in {
            "snapshot_ts_ns",
            "snapshot_ts_iso",
            "snapshot_ts_seconds",
            "sample_family",
            "sample_bucket_ms",
            "snapshot_hour_utc",
            "snapshot_minute_utc",
            "snapshot_day_of_week",
            "snapshot_second_of_day_utc",
            "market_interval_seconds",
            "t_since_open_s",
            "t_to_close_s",
            "session_progress_fraction",
            "phase_bucket",
            "market_view",
        }:
            return "derived_time"
        if column.startswith("prior_") or column.startswith("previous_") or column.startswith("resolved_") or column.startswith("prev_interval_"):
            return "phase3_history_derived"
        if column in {
            "has_canonical_resolution",
            "is_trainable",
            "market_fully_without_live_reference",
            "market_fully_without_trusted_live_reference",
            "market_fully_without_token_mid_complete",
            "market_5m_training_row_count",
            "market_5m_complete_row_count",
            "market_5m_complete_rate",
            "market_full_matrix_trainable",
            "training_feature_eligible",
            "exclusion_reason",
            "complete_feature_matrix_eligible",
            "feature_matrix_exclusion_reason",
            "full_matrix_training_eligible",
            "full_matrix_exclusion_reason",
            "coupling_anomaly",
            "crossed_book",
            "live_reference_trusted",
            "live_reference_strict_trusted",
            "live_reference_carried_trusted",
            "live_bias_bootstrapped",
            "live_bias_active",
            "live_bias_carried_forward",
            "live_bias_mode",
            "live_bias_state_stale",
            "live_bias_neutralized",
            "up_token_mid_complete",
            "down_token_mid_complete",
            "token_mid_complete",
            "up_token_one_sided_book",
            "down_token_one_sided_book",
            "one_sided_book",
            "spot_perp_available",
            "hold_to_close_edge_vs_mid",
            "early_direction_sign",
            "close_to_open_btc_sign",
            "first_30s_drift_sign",
        }:
            return "phase3_contract"
        return "derived"

    def schema_description(column: str) -> str:
        descriptions = {
            "training_feature_eligible": "True when the row satisfies all Phase 3 feature-quality gates and may enter the training feature matrix.",
            "exclusion_reason": "Primary row-level exclusion reason; empty string when the row is eligible.",
            "complete_feature_matrix_eligible": "True when the row is training-feature eligible and the default full v1 feature matrix has no required missing token-mid or spot/perp fields.",
            "feature_matrix_exclusion_reason": "Primary exclusion reason for the complete default v1 feature matrix; empty string when complete_feature_matrix_eligible is true.",
            "full_matrix_training_eligible": "True when the row is complete_feature_matrix_eligible and its 5-minute market has at least 80% complete rows in this materialized view.",
            "full_matrix_exclusion_reason": "Primary exclusion reason for full_matrix_training_eligible; empty string when full_matrix_training_eligible is true.",
            "coupling_anomaly": "True when |up_token_mid + down_token_mid - 1| exceeds 0.05.",
            "crossed_book": "True when either token book is crossed or locked under the strict >= rule.",
            "live_reference_trusted": "Backward-compatible alias for live_reference_carried_trusted; true when the row has a valid live price with either a fresh active residual window or a carried-forward bias.",
            "live_reference_strict_trusted": "True when the causal live reference price is valid and the official [t-60m, t-30m] delayed Chainlink residual window is active.",
            "live_reference_carried_trusted": "True when the causal live reference price is valid and either the strict residual window is active or the last valid bias is being carried forward.",
            "live_bias_bootstrapped": "True when live_reference_events_v1 has accepted at least one causal delayed Chainlink bias observation.",
            "live_bias_active": "True when live_reference_events_v1 is applying the official [t-60m, t-30m] delayed residual mean at the snapshot.",
            "live_bias_carried_forward": "True when live_reference_events_v1 is applying the last valid active-window bias because the strict residual window is empty.",
            "live_bias_mode": "Applied Phase 2 bias state: active_window, carried_forward, or unavailable.",
            "live_bias_state_stale": "True when live_reference_events_v1 has bootstrapped calibration but the [t-60m, t-30m] residual window is empty.",
            "live_bias_neutralized": "True when live_reference_events_v1 has bootstrapped calibration but no active or carried bias is applied.",
            "live_active_window_bias": "Fresh strict-window delayed Chainlink residual mean; NaN when the strict window is empty.",
            "live_bias_state_age_seconds": "Diagnostic calibration-state age for the newest active delayed Chainlink event; non-trainable because it is NaN whenever the residual window is empty.",
            "live_applied_bias_age_seconds": "Age in seconds of the Chainlink event timestamp backing the applied bias; continues increasing while bias is carried forward.",
            "up_token_mid_complete": "True when the UP token has both best bid and best ask and therefore a valid mid.",
            "down_token_mid_complete": "True when the DOWN token has both best bid and best ask and therefore a valid mid.",
            "token_mid_complete": "True when both UP and DOWN token mids are available.",
            "up_token_one_sided_book": "True when exactly one of UP best bid / best ask is available.",
            "down_token_one_sided_book": "True when exactly one of DOWN best bid / best ask is available.",
            "one_sided_book": "True when either token has a one-sided top of book.",
            "spot_perp_available": "True when spot/perp divergence was computed from both Binance spot and USD-M mids.",
            "has_canonical_resolution": "True when the row is backed by a matched canonical resolution label.",
            "is_trainable": "Market-level trainability flag after Phase 3 market exclusion checks.",
            "market_fully_without_live_reference": "True when every row for the market lacks a valid live BTC reference.",
            "market_fully_without_trusted_live_reference": "True when every row for the market lacks a trusted live reference after bias guardrails.",
            "market_fully_without_token_mid_complete": "True when every row for the market lacks complete UP and DOWN token mids.",
            "market_5m_training_row_count": "Number of training_feature_eligible rows for this market in the current materialized view.",
            "market_5m_complete_row_count": "Number of complete_feature_matrix_eligible rows for this market in the current materialized view.",
            "market_5m_complete_rate": "Complete matrix row count divided by training row count for this 5-minute market in the current materialized view.",
            "market_full_matrix_trainable": "True when market_5m_complete_rate is at least 0.80 and the market has at least one training row.",
            "resolved_side": "Canonical supervised target side for the market.",
            "resolved_side_label": "Binary supervised target for the market: 1=up, 0=down.",
            "flip_happened_after_t": "Diagnostic-only future-derived flag indicating whether strike-relative sign flips after the snapshot.",
            "terminal_delta_to_strike": "Diagnostic-only terminal live BTC delta to strike for the market.",
            "terminal_margin_to_strike": "Diagnostic-only absolute terminal live BTC margin to strike for the market.",
            "hold_to_close_edge_vs_mid": "Diagnostic-only edge-to-close quantity derived from the winning token mid.",
        }
        if column in descriptions:
            return descriptions[column]
        if column.startswith("up_token_") or column.startswith("down_token_"):
            return f"Polymarket token microstructure field `{column}` derived from finalized-clean `.l2` snapshots and recent event summaries."
        if column.startswith("btc_") or column.startswith("live_") or column.startswith("binance_") or column.startswith("hyperliquid_"):
            return f"BTC context field `{column}` derived from canonical `live_reference_events_v1`."
        return f"Phase 3 dataset column `{column}`."

    def write_schema_file(*, dataset_label: str, dataset_name: str, frame: pd.DataFrame, path: Path) -> None:
        payload = {
            "dataset_name": dataset_name,
            "columns": {
                column: {
                    "dtype": str(frame.dtypes[column]),
                    "nullable": bool(frame[column].isna().any()),
                    "leakage_only": column in LEAKAGE_ONLY_COLUMNS,
                    "training_eligible": schema_training_eligible(column),
                    "source": schema_source(column),
                    "description": schema_description(column),
                }
                for column in frame.columns
            },
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        output_paths[dataset_label] = str(path)

    coarse_paths = sorted((dataset_output_dir / COARSE_DATASET_NAME).glob("*.parquet"))
    if coarse_paths:
        write_schema_file(
            dataset_label=COARSE_DATASET_NAME,
            dataset_name=COARSE_DATASET_NAME,
            frame=pd.read_parquet(coarse_paths[0]),
            path=schema_dir / "resolution_snapshot_dataset_v1_coarse_schema.yaml",
        )

    dense_paths = sorted((dataset_output_dir / DENSE_DATASET_NAME).glob("*.parquet"))
    if dense_paths:
        write_schema_file(
            dataset_label=DENSE_DATASET_NAME,
            dataset_name=DENSE_DATASET_NAME,
            frame=pd.read_parquet(dense_paths[0]),
            path=schema_dir / "resolution_snapshot_dataset_v1_dense_close_schema.yaml",
        )

    interval_frame = None
    for dataset_name in MARKET_INTERVAL_DATASETS.values():
        shard_paths = sorted((dataset_output_dir / dataset_name).glob("*.parquet"))
        if shard_paths:
            interval_frame = pd.read_parquet(shard_paths[0])
            break
    if interval_frame is not None:
        write_schema_file(
            dataset_label="market_interval_dataset_v1",
            dataset_name="market_interval_dataset_v1",
            frame=interval_frame,
            path=schema_dir / "market_interval_dataset_v1_schema.yaml",
        )

    return output_paths


def _write_sampling_policy_contract(schema_dir: Path) -> str:
    schema_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sampling_policy_name": "phase3_sampling_policy_v1",
        "market_interval_views": {
            "preopen": {"offset_from_market_open_ms": -1000},
            "first15s": {"offset_from_market_open_ms": 15000},
            "first30s": {"offset_from_market_open_ms": 30000},
        },
        "resolution_snapshot_dataset_v1_coarse": {
            "start": "market_open",
            "end": "t_to_close=60s",
            "step_ms": 1000,
        },
        "resolution_snapshot_dataset_v1_dense_close": [
            {
                "start": "t_to_close=60s",
                "end": "t_to_close=10s",
                "step_ms": 250,
            },
            {
                "start": "t_to_close=10s",
                "end": "strictly_before_close",
                "step_ms": 100,
            },
        ],
        "event_triggered_expansion": {
            "enabled": False,
            "status": "deferred_for_later_phase3_extension",
        },
    }
    path = schema_dir / "phase3_sampling_policy_v1.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _write_quality_registry_exclusions(
    *,
    quality_rows: list[dict[str, Any]],
    quality_registry_path: Path,
) -> str:
    quality_registry_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        quality_rows,
        columns=[
            "dataset_name",
            "market_slug",
            "exclusion_reason",
            "rows",
            "first_snapshot_ts_ns",
            "last_snapshot_ts_ns",
        ],
    )
    if not frame.empty:
        frame = frame.sort_values(["dataset_name", "market_slug"], kind="stable").reset_index(drop=True)
    frame.to_parquet(quality_registry_path, index=False)
    return str(quality_registry_path)


def _assert_schema_leakage_boundaries(schema_paths: dict[str, str]) -> None:
    for path in schema_paths.values():
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        columns = dict(payload.get("columns") or {})
        leakage_cols = [name for name, meta in columns.items() if bool((meta or {}).get("leakage_only"))]
        feature_matrix_cols = [name for name, meta in columns.items() if bool((meta or {}).get("training_eligible"))]
        overlap = sorted(set(leakage_cols) & set(feature_matrix_cols))
        assert not overlap, f"Leakage columns found in feature matrix: {overlap}"


def _validate_phase3_outputs(
    *,
    dataset_output_dir: Path,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    dataset_names = [COARSE_DATASET_NAME, DENSE_DATASET_NAME, *MARKET_INTERVAL_DATASETS.values()]
    for dataset_name in dataset_names:
        shard_paths = sorted((dataset_output_dir / dataset_name).glob("*.parquet"))
        if not shard_paths:
            continue
        frame = pd.concat([pd.read_parquet(path) for path in shard_paths], axis=0, ignore_index=True)
        _assert_no_forbidden_columns(frame)
        required_columns = {
            "training_feature_eligible",
            "complete_feature_matrix_eligible",
            "full_matrix_training_eligible",
            "full_matrix_exclusion_reason",
            "exclusion_reason",
            "feature_matrix_exclusion_reason",
            "coupling_anomaly",
            "crossed_book",
            "live_reference_trusted",
            "live_reference_strict_trusted",
            "live_reference_carried_trusted",
            "token_mid_complete",
            "spot_perp_available",
            "market_5m_complete_rate",
            "market_full_matrix_trainable",
            "has_canonical_resolution",
            "is_trainable",
        }
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise AssertionError(f"{dataset_name} is missing required contract columns: {missing}")
        eligible_rate = float(frame["training_feature_eligible"].astype("bool").mean())
        complete_feature_matrix_rate = float(frame["complete_feature_matrix_eligible"].astype("bool").mean())
        full_matrix_training_rate = float(frame["full_matrix_training_eligible"].astype("bool").mean())
        strict_reference_rate = float(frame["live_reference_strict_trusted"].astype("bool").mean())
        carried_reference_rate = float(frame["live_reference_carried_trusted"].astype("bool").mean())
        coupling_rate = float(frame["coupling_anomaly"].astype("bool").mean())
        crossed_rate = float(frame["crossed_book"].astype("bool").mean())
        if coupling_rate >= 0.01:
            raise AssertionError(f"{dataset_name} coupling_anomaly rate must stay below 1%, got {coupling_rate:.4f}")
        if crossed_rate >= 0.01:
            raise AssertionError(f"{dataset_name} crossed_book rate must stay below 1%, got {crossed_rate:.4f}")
        results[dataset_name] = {
            "rows": int(len(frame)),
            "eligible_rate": eligible_rate,
            "complete_feature_matrix_rate": complete_feature_matrix_rate,
            "full_matrix_training_rate": full_matrix_training_rate,
            "strict_reference_rate": strict_reference_rate,
            "carried_reference_rate": carried_reference_rate,
            "coupling_anomaly_rate": coupling_rate,
            "crossed_book_rate": crossed_rate,
            "excluded_rows": int((~frame["training_feature_eligible"].astype("bool")).sum()),
            "complete_feature_matrix_excluded_rows": int((~frame["complete_feature_matrix_eligible"].astype("bool")).sum()),
            "full_matrix_training_excluded_rows": int((~frame["full_matrix_training_eligible"].astype("bool")).sum()),
        }
    return results


def _append_decision_log_entries(decision_log_path: Path) -> None:
    decision_log_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        (
            "## 2026-04-23 - Phase 3 Sampling Policy Freeze\n\n"
            "- Task: freeze the first supervised snapshot sampling policy for Phase 3 dataset factory.\n"
            "- Decision: `resolution_snapshot_dataset_v1_coarse` uses fixed 1s snapshots from market open through `T-60s`; "
            "`resolution_snapshot_dataset_v1_dense_close` uses fixed 250ms snapshots from `T-60s` to `T-10s` and fixed 100ms snapshots "
            "from `T-10s` to strictly before close. Event-triggered expansion is deferred and raw event streams remain on disk for later augmentation.\n"
            "- Evidence: the roadmap defines this exact multi-resolution progression as the recommended v1 sampling policy; the implemented factory writes both coarse and dense_close datasets using only pre-close causal timestamps.\n"
            "- Alternatives considered: monolithic 1s sampling for the entire market, rejected because the roadmap explicitly warns against downsampling away close microstructure.\n"
        ),
        (
            "## 2026-04-23 - Phase 3 Canonical Label Policy\n\n"
            "- Task: freeze deterministic label pairing for Phase 3 supervised datasets.\n"
            "- Decision: the dataset factory uses the earliest `label_safe=true` finalized-clean strike row as the canonical `price_to_beat` anchor per market slug, "
            "and includes a market in supervised datasets only if a matched finalized-clean canonical resolution row exists within the allowed archive window.\n"
            "- Evidence: this matches the scripture requirement to train only on `label_safe` strikes with matched canonical resolution rows and avoids touching unfinished `2026-04-23` inputs.\n"
            "- Alternatives considered: using later duplicate strike rows or unresolved closed markets, rejected because they weaken causality and violate the training label gate.\n"
        ),
        (
            "## 2026-04-23 - Phase 3 Polymarket Microstructure Source Choice\n\n"
            "- Task: choose the canonical Polymarket event source for Phase 3 v1 token pricing and order-flow features.\n"
            "- Decision: Phase 3 v1 uses finalized-clean Polymarket `.l2` as the canonical microstructure source, because it already materializes book snapshots, reconstructed price changes, top-of-book updates, and trade execution events per token. Raw `.ws` remains preserved for audit and later event-triggered expansion.\n"
            "- Evidence: the `.l2` archive exposes `l2_book_state`, `l2_top_of_book`, and `trade_execution` records with token-level market context, making it sufficient for the v1 snapshot datasets while staying inside the scripture-backed healthy input set.\n"
            "- Alternatives considered: joining raw `.ws` in parallel for Phase 3 v1, rejected as unnecessary duplication for the first canonical dataset factory pass.\n"
        ),
    ]
    existing = decision_log_path.read_text(encoding="utf-8") if decision_log_path.exists() else ""
    with decision_log_path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            header = entry.splitlines()[0]
            if header in existing:
                continue
            if not existing.endswith("\n\n"):
                handle.write("\n")
            handle.write(entry.rstrip() + "\n")
            existing += "\n" + entry


def build_phase3_datasets(
    *,
    root: Path,
    policy_path: Path = DEFAULT_DATASET_POLICY_PATH,
    date_from: date = DEFAULT_DATE_FROM,
    date_to: date = DEFAULT_DATE_TO,
    dataset_output_dir: Path = DEFAULT_DATASET_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    schema_dir: Path = DEFAULT_SCHEMA_DIR,
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
    decision_log_path: Path = DEFAULT_DECISION_LOG_PATH,
    max_workers: int = 1,
    skip_existing: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    policy_path = Path(policy_path)
    dataset_output_dir = Path(dataset_output_dir)
    report_path = Path(report_path)
    schema_dir = Path(schema_dir)
    quality_registry_path = Path(quality_registry_path)
    decision_log_path = Path(decision_log_path)
    if date_to < date_from:
        raise ValueError(f"date_to must be on or after date_from: {date_from} > {date_to}")

    labels_by_day, label_summary = _load_market_labels(
        root=root,
        policy_path=policy_path,
        date_from=date_from,
        date_to=date_to,
    )
    history_features_by_market_slug = _build_market_history_features(labels_by_day)

    dataset_names = [COARSE_DATASET_NAME, DENSE_DATASET_NAME, *MARKET_INTERVAL_DATASETS.values()]
    for dataset_name in dataset_names:
        target_dir = dataset_output_dir / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "root": str(root),
            "target_date": target_date.isoformat(),
            "date_from": date_from.isoformat(),
            "policy_path": str(policy_path),
            "dataset_output_dir": str(dataset_output_dir),
            "labels": [asdict(label) for label in labels_by_day.get(target_date, [])],
            "history_features_by_market_slug": {
                label.market_slug: history_features_by_market_slug.get(label.market_slug, {})
                for label in labels_by_day.get(target_date, [])
            },
            "skip_existing": skip_existing,
        }
        for target_date in _date_range(date_from, date_to)
    ]

    if max_workers <= 1:
        day_summaries = [_build_day_datasets(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
            day_summaries = list(executor.map(_build_day_datasets, tasks))

    quality_registry_rows: list[dict[str, Any]] = []
    for summary in day_summaries:
        quality_registry_rows.extend(list(summary.get("quality_registry_rows") or []))

    schema_paths = _write_schema_contracts(
        dataset_output_dir=dataset_output_dir,
        schema_dir=schema_dir,
    )
    _assert_schema_leakage_boundaries(schema_paths)
    sampling_policy_path = _write_sampling_policy_contract(schema_dir)
    quality_registry_output_path = _write_quality_registry_exclusions(
        quality_rows=quality_registry_rows,
        quality_registry_path=quality_registry_path,
    )
    validation_summary = _validate_phase3_outputs(dataset_output_dir=dataset_output_dir)
    _append_decision_log_entries(decision_log_path)

    hash_rows: list[dict[str, Any]] = []
    total_rows_by_dataset: Counter[str] = Counter()
    for summary in sorted(day_summaries, key=lambda item: item["target_date"]):
        total_rows_by_dataset[COARSE_DATASET_NAME] += int(summary["coarse_rows"])
        total_rows_by_dataset[DENSE_DATASET_NAME] += int(summary["dense_rows"])
        for view_name, dataset_name in MARKET_INTERVAL_DATASETS.items():
            total_rows_by_dataset[dataset_name] += int(summary["market_interval_rows"][view_name])
        for dataset_name, hash_value in sorted(summary["hashes"].items()):
            hash_rows.append(
                {
                    "dataset_name": dataset_name,
                    "date": summary["target_date"],
                    "rows": (
                        summary["coarse_rows"]
                        if dataset_name == COARSE_DATASET_NAME
                        else summary["dense_rows"]
                        if dataset_name == DENSE_DATASET_NAME
                        else summary["market_interval_rows"][next(key for key, value in MARKET_INTERVAL_DATASETS.items() if value == dataset_name)]
                    ),
                    "sha256": hash_value,
                    "path": summary["paths"][dataset_name],
                }
            )

    exclusion_summary: dict[str, Counter] = defaultdict(Counter)
    for summary in day_summaries:
        for source_name, bucket in summary["exclusions"].items():
            for reason, count in bucket.items():
                exclusion_summary[source_name][reason] += int(count)

    report_lines = [
        "# Phase 3 Dataset Factory Report",
        "",
        f"- Date range: `{date_from.isoformat()}` to `{date_to.isoformat()}`",
        f"- Policy: `{policy_path.as_posix()}`",
        f"- Dataset output dir: `{dataset_output_dir.as_posix()}`",
        "",
        "## Label Registry Summary",
        "",
        f"- canonical_label_safe_strikes: `{label_summary['canonical_label_safe_strikes']}`",
        f"- canonical_resolutions_in_scope: `{label_summary['canonical_resolutions_in_scope']}`",
        f"- matched_markets: `{label_summary['matched_markets']}`",
        f"- missing_resolution_market_count: `{label_summary['missing_resolution_market_count']}`",
        "",
        "## Dataset Row Counts",
        "",
        "| dataset_name | total_rows |",
        "| --- | --- |",
    ]
    for dataset_name in dataset_names:
        report_lines.append(f"| {dataset_name} | {int(total_rows_by_dataset[dataset_name])} |")
    report_lines.extend(
        [
            "",
            "## Label Input Exclusion Summary",
            "",
            "| source | reason | excluded_files_or_hours |",
            "| --- | --- | --- |",
        ]
    )
    label_exclusions_present = False
    for exclusion_family in ("strike_exclusions", "resolution_exclusions"):
        for source_name, bucket in sorted(label_summary[exclusion_family].items()):
            for reason, count in sorted(bucket.items()):
                label_exclusions_present = True
                report_lines.append(f"| {source_name} | {reason} | {count} |")
    if not label_exclusions_present:
        report_lines.append("| none | none | 0 |")
    if label_summary["missing_resolution_market_slugs"]:
        report_lines.extend(
            [
                "",
                "## Missing Resolution Market Slugs",
                "",
            ]
        )
        for market_slug in label_summary["missing_resolution_market_slugs"][:50]:
            report_lines.append(f"- `{market_slug}`")
    report_lines.extend(
        [
            "",
            "## Reproducibility Hashes",
            "",
            "| dataset_name | date | rows | sha256 | path |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in hash_rows:
        report_lines.append(
            f"| {row['dataset_name']} | {row['date']} | {row['rows']} | {row['sha256']} | {row['path']} |"
        )
    report_lines.extend(
        [
            "",
            "## Training Input Exclusion Summary",
            "",
            "| source | reason | excluded_files_or_hours |",
            "| --- | --- | --- |",
        ]
    )
    training_exclusions_present = False
    for source_name, bucket in sorted(exclusion_summary.items()):
        for reason, count in sorted(bucket.items()):
            training_exclusions_present = True
            report_lines.append(f"| {source_name} | {reason} | {count} |")
    if not training_exclusions_present:
        report_lines.append("| none | none | 0 |")
    report_lines.extend(
        [
            "",
            "## Schema Contracts",
            "",
            f"- `phase3_sampling_policy_v1` -> `{sampling_policy_path}`",
        ]
    )
    for dataset_name, path in sorted(schema_paths.items()):
        report_lines.append(f"- `{dataset_name}` -> `{path}`")
    report_lines.extend(
        [
            "",
            "## Validation Summary",
            "",
            "| dataset_name | rows | strict_reference_rate | carried_reference_rate | eligible_rate | complete_feature_matrix_rate | full_matrix_training_rate | coupling_anomaly_rate | crossed_book_rate | excluded_rows | complete_feature_matrix_excluded_rows | full_matrix_training_excluded_rows |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for dataset_name in dataset_names:
        if dataset_name not in validation_summary:
            continue
        row = validation_summary[dataset_name]
        report_lines.append(
            f"| {dataset_name} | {row['rows']} | {row['strict_reference_rate']:.4f} | {row['carried_reference_rate']:.4f} | {row['eligible_rate']:.4f} | {row['complete_feature_matrix_rate']:.4f} | {row['full_matrix_training_rate']:.4f} | {row['coupling_anomaly_rate']:.4f} | {row['crossed_book_rate']:.4f} | {row['excluded_rows']} | {row['complete_feature_matrix_excluded_rows']} | {row['full_matrix_training_excluded_rows']} |"
        )
    report_lines.extend(
        [
            "",
            "## quality_registry_v1 Exclusions",
            "",
            f"- Phase 3 market exclusion list -> `{quality_registry_output_path}`",
            f"- excluded_markets_written: `{len(quality_registry_rows)}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    return {
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "matched_markets": int(label_summary["matched_markets"]),
        "missing_resolution_market_count": int(label_summary["missing_resolution_market_count"]),
        "total_rows_by_dataset": {key: int(value) for key, value in total_rows_by_dataset.items()},
        "hashes": hash_rows,
        "schema_paths": schema_paths,
        "sampling_policy_path": sampling_policy_path,
        "quality_registry_path": quality_registry_output_path,
        "validation_summary": validation_summary,
        "report_path": str(report_path),
    }
