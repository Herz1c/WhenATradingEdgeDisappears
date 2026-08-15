from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from binance_recorder.compression import iter_zstd_jsonl
from market_recorders.unified_reader import UnifiedRawReader

SECOND_NS = 1_000_000_000
STALE_SECONDS = 30.0
STALE_NS = int(STALE_SECONDS * SECOND_NS)
BINANCE_USDM_QUARANTINE_HOURS = {("2026-04-20", "07")}
BASE_SYNTHETIC_CHAINLINK_METHOD = "spot_only_premium_v1"
DELAYED_MEAN_RESIDUAL_SYNTHETIC_CHAINLINK_METHOD = "spot_only_premium_delayed_mean_residual_1800s_v1"
SYNTHETIC_CHAINLINK_METHOD = DELAYED_MEAN_RESIDUAL_SYNTHETIC_CHAINLINK_METHOD
SOURCE_NAMES = ("binance_spot", "binance_usdm", "hyperliquid")
SPOT_SOURCE_NAMES = ("binance_spot",)
SPOT_PREMIUM_MARKUP_BPS = 2.9
DELAYED_CHAINLINK_CALIBRATION_DELAY_SECONDS = 1800.0
DELAYED_CHAINLINK_CALIBRATION_DELAY_NS = int(DELAYED_CHAINLINK_CALIBRATION_DELAY_SECONDS * SECOND_NS)
DELAYED_CHAINLINK_EWMA_ALPHA = 0.05
DELAYED_CHAINLINK_RESIDUAL_CLIP_USD = 50.0
SOURCE_DISPLAY_NAMES = {
    "binance_spot": "Binance Spot",
    "binance_usdm": "Binance USD-M",
    "hyperliquid": "Hyperliquid",
}
METHOD_PREFERENCE = {
    DELAYED_MEAN_RESIDUAL_SYNTHETIC_CHAINLINK_METHOD: 0,
    BASE_SYNTHETIC_CHAINLINK_METHOD: 1,
    "median_all": 2,
    "trimmed_mean": 3,
    "volume_weighted_mid": 4,
    "binance_spot_raw": 5,
    "median_spot_only": 6,
}
DECISION_LOG_HEADING = "## 2026-04-22 - Synthetic BTC/USD Chainlink Proxy"


@dataclass(slots=True, frozen=True)
class SecondSourceUpdate:
    second_ts_ns: int
    mid: float
    weight: float | None
    last_recv_ts_ns: int
    raw_mid: float | None = None
    basis_adjusted_mid: float | None = None


@dataclass(slots=True, frozen=True)
class ChainlinkGroundTruthEvent:
    chainlink_ts_ns: int
    chainlink_price: float
    first_seen_recv_ts_ns: int


@dataclass(slots=True)
class SourceState:
    mid: float | None
    raw_mid: float | None
    basis_adjusted_mid: float | None
    weight: float | None
    last_recv_ts_ns: int | None
    staleness_s: float | None
    data_present: bool


@dataclass(slots=True)
class CandidateSnapshot:
    recv_ts_ns: int
    candidate_values: dict[str, float | None]
    source_states: dict[str, SourceState]
    source_count: int
    max_spread_usd: float | None
    outlier_source: str | None


@dataclass(slots=True)
class CandidateStats:
    candidate_name: str
    matched_event_count: int
    event_coverage: float
    mean_error_usd: float | None
    median_abs_error_usd: float | None
    p90_abs_error_usd: float | None
    p99_abs_error_usd: float | None
    fraction_within_1_usd: float | None
    fraction_within_3_usd: float | None
    fraction_within_10_usd: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "matched_event_count": self.matched_event_count,
            "event_coverage": self.event_coverage,
            "mean_error_usd": self.mean_error_usd,
            "median_abs_error_usd": self.median_abs_error_usd,
            "p90_abs_error_usd": self.p90_abs_error_usd,
            "p99_abs_error_usd": self.p99_abs_error_usd,
            "fraction_within_1_usd": self.fraction_within_1_usd,
            "fraction_within_3_usd": self.fraction_within_3_usd,
            "fraction_within_10_usd": self.fraction_within_10_usd,
        }


@dataclass(slots=True)
class DelayedChainlinkBiasCalibrator:
    availability_delay_seconds: float = DELAYED_CHAINLINK_CALIBRATION_DELAY_SECONDS
    ewma_alpha: float = DELAYED_CHAINLINK_EWMA_ALPHA
    residual_clip_usd: float = DELAYED_CHAINLINK_RESIDUAL_CLIP_USD
    bias_usd: float = 0.0
    sample_count: int = 0

    def observe(self, *, base_synthetic_price: float, observed_chainlink_price: float) -> float:
        residual = observed_chainlink_price - base_synthetic_price
        residual = max(-self.residual_clip_usd, min(self.residual_clip_usd, residual))
        if self.sample_count == 0:
            self.bias_usd = residual
        else:
            self.bias_usd = ((1.0 - self.ewma_alpha) * self.bias_usd) + (self.ewma_alpha * residual)
        self.sample_count += 1
        return self.bias_usd

    def export_metadata(self) -> dict[str, Any]:
        return {
            "bias_usd": self.bias_usd,
            "sample_count": self.sample_count,
            "availability_delay_seconds": self.availability_delay_seconds,
            "ewma_alpha": self.ewma_alpha,
            "residual_clip_usd": self.residual_clip_usd,
        }


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


def _parse_target_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _extract_delayed_bias_usd(sources_dict: dict[str, dict[str, Any]]) -> float:
    for key in ("_delayed_chainlink_calibration", "delayed_chainlink_calibration"):
        payload = sources_dict.get(key)
        if isinstance(payload, dict):
            bias = _to_float(payload.get("bias_usd"))
            if bias is not None:
                return bias
    return 0.0


def _apply_delayed_bias(base_price: float | None, bias_usd: float) -> float | None:
    if base_price is None:
        return None
    return base_price + bias_usd


def _iter_dates(date_from: date | str, date_to: date | str) -> list[date]:
    start = _parse_target_date(date_from)
    end = _parse_target_date(date_to)
    dates: list[date] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _ceil_second_ns(ts_ns: int) -> int:
    return ((ts_ns + SECOND_NS - 1) // SECOND_NS) * SECOND_NS


def _valid_clean_files(
    *,
    root: Path,
    relative_prefix: Path,
    kind: str,
    excluded_hours: set[tuple[str, str]] | None = None,
) -> list[tuple[str, str, Path]]:
    reader = UnifiedRawReader(root=root)
    excluded_hours = excluded_hours or set()
    files: list[tuple[str, str, Path]] = []
    base = root / "raw" / relative_prefix
    if not base.exists():
        return files
    for day_dir in sorted(base.iterdir()):
        if not day_dir.is_dir():
            continue
        for path in reader.list_files_for_date(
            relative_prefix=relative_prefix,
            target_date=day_dir.name,
            kind=kind,
            finalized_only=False,
        ):
            metadata = reader.file_metadata(path)
            hour = path.name.split(".")[0]
            if (day_dir.name, hour) in excluded_hours:
                continue
            if metadata["finalized"] is not True:
                continue
            if metadata["quality_class"] != "finalized_clean":
                continue
            files.append((day_dir.name, hour, path))
    return files


def _collapse_updates(rows: dict[int, tuple[float, float | None, int, float | None, float | None]]) -> list[SecondSourceUpdate]:
    return [
        SecondSourceUpdate(
            second_ts_ns=second_ts_ns,
            mid=payload[0],
            weight=payload[1],
            last_recv_ts_ns=payload[2],
            raw_mid=payload[3],
            basis_adjusted_mid=payload[4],
        )
        for second_ts_ns, payload in sorted(rows.items())
    ]


def _parse_binance_ws_file(path_str: str) -> list[SecondSourceUpdate]:
    path = Path(path_str)
    by_second: dict[int, tuple[float, float | None, int, float | None, float | None]] = {}
    for record in iter_zstd_jsonl(path):
        normalized = record.get("normalized") or {}
        bid = _to_float(normalized.get("best_bid_price"))
        ask = _to_float(normalized.get("best_ask_price"))
        recv_ts_ns = _to_int(record.get("recv_ts_ns"))
        if bid is None or ask is None or recv_ts_ns is None:
            continue
        bid_qty = _to_float(normalized.get("best_bid_qty"))
        ask_qty = _to_float(normalized.get("best_ask_qty"))
        quantities = [quantity for quantity in (bid_qty, ask_qty) if quantity is not None and quantity > 0]
        weight = sum(quantities) / len(quantities) if quantities else None
        mid = (bid + ask) / 2.0
        second_ts_ns = _ceil_second_ns(recv_ts_ns)
        current = by_second.get(second_ts_ns)
        if current is None or recv_ts_ns >= current[2]:
            by_second[second_ts_ns] = (mid, weight, recv_ts_ns, mid, None)
    return _collapse_updates(by_second)


def _parse_hyperliquid_ws_file(path_str: str) -> list[SecondSourceUpdate]:
    path = Path(path_str)
    by_second: dict[int, tuple[float, float | None, int, float | None, float | None]] = {}
    for record in iter_zstd_jsonl(path):
        if str(record.get("topic")) != "l2Book":
            continue
        payload = record.get("payload") or {}
        data = payload.get("data") or {}
        levels = data.get("levels")
        recv_ts_ns = _to_int(record.get("recv_ts_ns"))
        if recv_ts_ns is None or not isinstance(levels, list) or len(levels) < 2:
            continue
        bids = levels[0]
        asks = levels[1]
        if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
            continue
        try:
            best_bid = max(_to_float(level.get("px")) for level in bids if _to_float(level.get("px")) is not None)
            best_ask = min(_to_float(level.get("px")) for level in asks if _to_float(level.get("px")) is not None)
        except ValueError:
            continue
        if best_bid is None or best_ask is None:
            continue
        bid_qty = sum(
            _to_float(level.get("sz")) or 0.0
            for level in bids
            if _to_float(level.get("px")) == best_bid
        )
        ask_qty = sum(
            _to_float(level.get("sz")) or 0.0
            for level in asks
            if _to_float(level.get("px")) == best_ask
        )
        positive_weights = [weight for weight in (bid_qty, ask_qty) if weight > 0]
        weight = sum(positive_weights) / len(positive_weights) if positive_weights else None
        mid = (best_bid + best_ask) / 2.0
        second_ts_ns = _ceil_second_ns(recv_ts_ns)
        current = by_second.get(second_ts_ns)
        if current is None or recv_ts_ns >= current[2]:
            by_second[second_ts_ns] = (mid, weight, recv_ts_ns, mid, None)
    return _collapse_updates(by_second)


def _parse_public_delayed_file(path_str: str) -> list[ChainlinkGroundTruthEvent]:
    path = Path(path_str)
    by_chainlink_ts: dict[int, tuple[float, int]] = {}
    for record in iter_zstd_jsonl(path):
        display_ts = _to_int(record.get("chainlink_display_ts"))
        price = _to_float(record.get("btc_usd_price"))
        recv_ts_ns = _to_int(record.get("recv_ts_ns"))
        if display_ts is None or price is None or recv_ts_ns is None:
            continue
        chainlink_ts_ns = display_ts * SECOND_NS if abs(display_ts) < 10_000_000_000 else display_ts
        current = by_chainlink_ts.get(chainlink_ts_ns)
        if current is None:
            by_chainlink_ts[chainlink_ts_ns] = (price, recv_ts_ns)
            continue
        if not math.isclose(current[0], price, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Conflicting public delayed prices for chainlink timestamp {chainlink_ts_ns}: {current[0]} vs {price}"
            )
        if recv_ts_ns < current[1]:
            by_chainlink_ts[chainlink_ts_ns] = (price, recv_ts_ns)
    return [
        ChainlinkGroundTruthEvent(
            chainlink_ts_ns=chainlink_ts_ns,
            chainlink_price=payload[0],
            first_seen_recv_ts_ns=payload[1],
        )
        for chainlink_ts_ns, payload in sorted(by_chainlink_ts.items())
    ]


def _flatten_updates(chunks: list[list[SecondSourceUpdate]]) -> list[SecondSourceUpdate]:
    merged: dict[int, SecondSourceUpdate] = {}
    for chunk in chunks:
        for row in chunk:
            current = merged.get(row.second_ts_ns)
            if current is None or row.last_recv_ts_ns >= current.last_recv_ts_ns:
                merged[row.second_ts_ns] = row
    return [merged[key] for key in sorted(merged)]


def _flatten_ground_truth(chunks: list[list[ChainlinkGroundTruthEvent]]) -> list[ChainlinkGroundTruthEvent]:
    merged: dict[int, ChainlinkGroundTruthEvent] = {}
    for chunk in chunks:
        for row in chunk:
            current = merged.get(row.chainlink_ts_ns)
            if current is None:
                merged[row.chainlink_ts_ns] = row
                continue
            if not math.isclose(current.chainlink_price, row.chainlink_price, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"Conflicting public delayed prices for chainlink timestamp {row.chainlink_ts_ns}: "
                    f"{current.chainlink_price} vs {row.chainlink_price}"
                )
            if row.first_seen_recv_ts_ns < current.first_seen_recv_ts_ns:
                merged[row.chainlink_ts_ns] = row
    return [merged[key] for key in sorted(merged)]


def _run_parallel(loader: Any, paths: list[str], *, max_workers: int) -> list[Any]:
    if not paths:
        return []
    if max_workers <= 1:
        return [loader(path) for path in paths]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(loader, paths))


def load_clean_source_updates(
    *,
    root: Path,
    source_name: str,
    max_workers: int = 1,
) -> list[SecondSourceUpdate]:
    if source_name == "binance_spot":
        prefix = Path("binance") / "spot" / "BTCUSDT"
        files = _valid_clean_files(root=root, relative_prefix=prefix, kind="ws")
        chunks = _run_parallel(_parse_binance_ws_file, [str(path) for _, _, path in files], max_workers=max_workers)
        return _flatten_updates(chunks)
    if source_name == "binance_usdm":
        prefix = Path("binance") / "usdm" / "BTCUSDT"
        files = _valid_clean_files(
            root=root,
            relative_prefix=prefix,
            kind="ws",
            excluded_hours=BINANCE_USDM_QUARANTINE_HOURS,
        )
        chunks = _run_parallel(_parse_binance_ws_file, [str(path) for _, _, path in files], max_workers=max_workers)
        return _flatten_updates(chunks)
    if source_name == "hyperliquid":
        prefix = Path("hyperliquid") / "linear" / "BTC"
        files = _valid_clean_files(root=root, relative_prefix=prefix, kind="ws")
        chunks = _run_parallel(_parse_hyperliquid_ws_file, [str(path) for _, _, path in files], max_workers=max_workers)
        return _flatten_updates(chunks)
    raise ValueError(f"Unsupported source: {source_name}")


def load_chainlink_ground_truth(
    *,
    root: Path,
    max_workers: int = 1,
) -> list[ChainlinkGroundTruthEvent]:
    prefix = Path("chainlink_public_delayed") / "public_stream_page" / "BTCUSD"
    files = _valid_clean_files(root=root, relative_prefix=prefix, kind="page")
    chunks = _run_parallel(_parse_public_delayed_file, [str(path) for _, _, path in files], max_workers=max_workers)
    return _flatten_ground_truth(chunks)


def _fresh_source_state(
    source_payload: dict[str, Any] | None,
    *,
    recv_ts_ns: int,
) -> SourceState:
    payload = source_payload or {}
    mid = _to_float(payload.get("mid"))
    raw_mid = _to_float(payload.get("raw_mid"))
    basis_adjusted_mid = _to_float(payload.get("basis_adjusted_mid"))
    weight = _to_float(payload.get("weight"))
    last_recv_ts_ns = _to_int(payload.get("last_recv_ts_ns"))
    explicit_staleness = _to_float(payload.get("staleness_s"))
    if explicit_staleness is not None:
        staleness_s = explicit_staleness
    elif last_recv_ts_ns is not None:
        staleness_s = max(0.0, (recv_ts_ns - last_recv_ts_ns) / SECOND_NS)
    else:
        staleness_s = None
    data_present = bool(payload.get("data_present") is True)
    if mid is None or staleness_s is None or staleness_s > STALE_SECONDS:
        return SourceState(
            mid=None,
            raw_mid=raw_mid,
            basis_adjusted_mid=basis_adjusted_mid,
            weight=None,
            last_recv_ts_ns=last_recv_ts_ns,
            staleness_s=staleness_s,
            data_present=data_present,
        )
    return SourceState(
        mid=mid,
        raw_mid=raw_mid if raw_mid is not None else mid,
        basis_adjusted_mid=basis_adjusted_mid,
        weight=weight,
        last_recv_ts_ns=last_recv_ts_ns,
        staleness_s=staleness_s,
        data_present=data_present,
    )


def compute_candidate_snapshot(
    sources_dict: dict[str, dict[str, Any]],
    *,
    recv_ts_ns: int,
) -> CandidateSnapshot:
    delayed_bias_usd = _extract_delayed_bias_usd(sources_dict)
    source_states = {
        source_name: _fresh_source_state(sources_dict.get(source_name), recv_ts_ns=recv_ts_ns)
        for source_name in SOURCE_NAMES
    }
    fresh_midpoints = {name: state.mid for name, state in source_states.items() if state.mid is not None}
    candidate_values: dict[str, float | None] = {
        DELAYED_MEAN_RESIDUAL_SYNTHETIC_CHAINLINK_METHOD: None,
        BASE_SYNTHETIC_CHAINLINK_METHOD: None,
        "binance_spot_raw": source_states["binance_spot"].mid,
        "median_spot_only": source_states["binance_spot"].mid,
        "median_all": None,
        "volume_weighted_mid": None,
        "trimmed_mean": None,
    }
    if fresh_midpoints:
        ordered_midpoints = sorted(fresh_midpoints.values())
        candidate_values["median_all"] = float(median(ordered_midpoints))
        if len(ordered_midpoints) >= 3:
            inner = ordered_midpoints[1:-1]
            candidate_values["trimmed_mean"] = sum(inner) / len(inner)
        elif len(ordered_midpoints) >= 2:
            candidate_values["trimmed_mean"] = sum(ordered_midpoints) / len(ordered_midpoints)
        else:
            candidate_values["trimmed_mean"] = ordered_midpoints[0]
        weighted = [
            (state.mid, state.weight)
            for state in source_states.values()
            if state.mid is not None and state.weight is not None and state.weight > 0
        ]
        if weighted:
            total_weight = sum(weight for _, weight in weighted)
            candidate_values["volume_weighted_mid"] = sum(mid * weight for mid, weight in weighted) / total_weight

    spot_mid = source_states["binance_spot"].mid
    if spot_mid is not None:
        candidate_values[BASE_SYNTHETIC_CHAINLINK_METHOD] = spot_mid * (1.0 + (SPOT_PREMIUM_MARKUP_BPS / 10_000.0))
        candidate_values[DELAYED_MEAN_RESIDUAL_SYNTHETIC_CHAINLINK_METHOD] = _apply_delayed_bias(
            candidate_values[BASE_SYNTHETIC_CHAINLINK_METHOD],
            delayed_bias_usd,
        )

    source_count = len(fresh_midpoints)
    max_spread_usd = None
    outlier_source = None
    if source_count >= 2:
        min_mid = min(fresh_midpoints.values())
        max_mid = max(fresh_midpoints.values())
        max_spread_usd = max_mid - min_mid
        median_mid = candidate_values["median_all"]
        if median_mid is not None:
            deviations = {
                name: abs(mid - median_mid)
                for name, mid in fresh_midpoints.items()
            }
            max_deviation = max(deviations.values())
            furthest = [name for name, deviation in deviations.items() if math.isclose(deviation, max_deviation, abs_tol=1e-9)]
            if len(furthest) == 1 and max_deviation > 0:
                outlier_source = furthest[0]
    return CandidateSnapshot(
        recv_ts_ns=recv_ts_ns,
        candidate_values=candidate_values,
        source_states=source_states,
        source_count=source_count,
        max_spread_usd=max_spread_usd,
        outlier_source=outlier_source,
    )


def build_synthetic_chainlink(sources_dict: dict[str, dict[str, Any]], recv_ts_ns: int) -> float:
    """
    Deterministically reconstruct the synthetic BTC/USD Chainlink proxy from forward-filled venue mids.

    Expected per-source payload shape:
    {
        "mid": float | None,
        "raw_mid": float | None,
        "basis_adjusted_mid": float | None,
        "weight": float | None,
        "last_recv_ts_ns": int | None,
        "staleness_s": float | None,
        "data_present": bool,
    }

    Optional synthetic-calibration metadata can be supplied under either
    `"_delayed_chainlink_calibration"` or `"delayed_chainlink_calibration"`:
    {
        "bias_usd": float,
    }

    The function is fully causal: callers must only pass source state built from observations with
    recv_ts_ns <= the requested evaluation timestamp. Any source older than 30 seconds is treated as missing.
    A synthetic price is emitted when fresh Binance spot is available. Binance USD-M and Hyperliquid remain
    diagnostic candidates only and do not enter the selected official price.
    When delayed calibration metadata is absent, the adaptive builder falls back to the base spot-only premium formula.
    """
    snapshot = compute_candidate_snapshot(sources_dict, recv_ts_ns=recv_ts_ns)
    selected = snapshot.candidate_values.get(SYNTHETIC_CHAINLINK_METHOD)
    if selected is None:
        return float("nan")
    return float(selected)


def _empty_bucket_metrics() -> dict[str, dict[str, Any]]:
    return {
        candidate_name: {"errors": [], "matched": 0, "within_1": 0, "within_3": 0, "within_10": 0}
        for candidate_name in METHOD_PREFERENCE
    }


def _stats_from_errors(
    *,
    candidate_name: str,
    errors: list[float],
    total_events: int,
) -> CandidateStats:
    matched = len(errors)
    if matched == 0:
        return CandidateStats(
            candidate_name=candidate_name,
            matched_event_count=0,
            event_coverage=0.0 if total_events else 0.0,
            mean_error_usd=None,
            median_abs_error_usd=None,
            p90_abs_error_usd=None,
            p99_abs_error_usd=None,
            fraction_within_1_usd=None,
            fraction_within_3_usd=None,
            fraction_within_10_usd=None,
        )
    abs_errors = [abs(error) for error in errors]
    return CandidateStats(
        candidate_name=candidate_name,
        matched_event_count=matched,
        event_coverage=matched / total_events if total_events else 0.0,
        mean_error_usd=_mean(errors),
        median_abs_error_usd=_percentile(abs_errors, 0.5),
        p90_abs_error_usd=_percentile(abs_errors, 0.9),
        p99_abs_error_usd=_percentile(abs_errors, 0.99),
        fraction_within_1_usd=sum(abs_error <= 1.0 for abs_error in abs_errors) / matched,
        fraction_within_3_usd=sum(abs_error <= 3.0 for abs_error in abs_errors) / matched,
        fraction_within_10_usd=sum(abs_error <= 10.0 for abs_error in abs_errors) / matched,
    )


def _method_sort_key(stats: CandidateStats) -> tuple[float, float, float, int, int]:
    return (
        stats.median_abs_error_usd if stats.median_abs_error_usd is not None else float("inf"),
        stats.p90_abs_error_usd if stats.p90_abs_error_usd is not None else float("inf"),
        -(stats.fraction_within_3_usd if stats.fraction_within_3_usd is not None else -1.0),
        -stats.matched_event_count,
        METHOD_PREFERENCE.get(stats.candidate_name, 99),
    )


def _format_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _bucket_error_rows(bucket_metrics: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, errors in bucket_metrics.items():
        abs_errors = [abs(error) for error in errors]
        rows.append(
            {
                "bucket": label,
                "count": len(errors),
                "mean_error_usd": _mean(errors),
                "median_abs_error_usd": _percentile(abs_errors, 0.5),
                "p90_abs_error_usd": _percentile(abs_errors, 0.9),
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values: list[str] = []
        for key, _label in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(_format_float(value))
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def _without_existing_decision_entry(existing_text: str, *, heading: str) -> str:
    lines = existing_text.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        if line.strip() == heading:
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept).strip()


def _candidate_table_rows(stats_by_candidate: list[CandidateStats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stats in stats_by_candidate:
        rows.append(
            {
                "candidate": stats.candidate_name,
                "matched_events": stats.matched_event_count,
                "coverage": _format_pct(stats.event_coverage),
                "mean_error_usd": _format_float(stats.mean_error_usd),
                "median_abs_error_usd": _format_float(stats.median_abs_error_usd),
                "p90_abs_error_usd": _format_float(stats.p90_abs_error_usd),
                "p99_abs_error_usd": _format_float(stats.p99_abs_error_usd),
                "within_1_usd": _format_pct(stats.fraction_within_1_usd),
                "within_3_usd": _format_pct(stats.fraction_within_3_usd),
                "within_10_usd": _format_pct(stats.fraction_within_10_usd),
            }
        )
    return rows


def _rolling_volatility_bucket(values: list[float | None]) -> list[float | None]:
    realized_vol: list[float | None] = [None] * len(values)
    rolling_returns: deque[float] = deque()
    rolling_sum = 0.0
    previous_mid: float | None = None
    for index, mid in enumerate(values):
        if mid is None or previous_mid is None or previous_mid <= 0 or mid <= 0:
            rolling_returns.clear()
            rolling_sum = 0.0
            previous_mid = mid
            realized_vol[index] = None
            continue
        log_return = math.log(mid / previous_mid)
        squared = log_return * log_return
        rolling_returns.append(squared)
        rolling_sum += squared
        if len(rolling_returns) > 300:
            rolling_sum -= rolling_returns.popleft()
        previous_mid = mid
        realized_vol[index] = math.sqrt(rolling_sum)
    return realized_vol


def _terciles(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return _percentile(values, 1 / 3), _percentile(values, 2 / 3)


def _vol_bucket(value: float | None, *, lower: float | None, upper: float | None) -> str:
    if value is None or lower is None or upper is None:
        return "spot_missing"
    if value <= lower:
        return "low"
    if value <= upper:
        return "medium"
    return "high"


def _source_grid_payload(update: SecondSourceUpdate | None, *, second_ts_ns: int, data_present: bool) -> dict[str, Any]:
    if update is None:
        return {
            "mid": None,
            "raw_mid": None,
            "basis_adjusted_mid": None,
            "weight": None,
            "last_recv_ts_ns": None,
            "staleness_s": None,
            "data_present": data_present,
        }
    return {
        "mid": update.mid,
        "raw_mid": update.raw_mid,
        "basis_adjusted_mid": update.basis_adjusted_mid,
        "weight": update.weight,
        "last_recv_ts_ns": update.last_recv_ts_ns,
        "staleness_s": max(0.0, (second_ts_ns - update.last_recv_ts_ns) / SECOND_NS),
        "data_present": data_present,
    }


def _delayed_calibration_available_ts_ns(event: ChainlinkGroundTruthEvent) -> int:
    return max(event.first_seen_recv_ts_ns, event.chainlink_ts_ns + DELAYED_CHAINLINK_CALIBRATION_DELAY_NS)


def build_synthetic_chainlink_report(
    *,
    root: Path,
    artifact_dir: Path,
    max_workers: int | None = None,
) -> dict[str, Any]:
    root = Path(root)
    artifact_dir = Path(artifact_dir)
    if max_workers is None:
        max_workers = max(1, min(4, os.cpu_count() or 1))

    source_updates = {
        source_name: load_clean_source_updates(root=root, source_name=source_name, max_workers=max_workers)
        for source_name in SOURCE_NAMES
    }
    ground_truth = load_chainlink_ground_truth(root=root, max_workers=max_workers)
    if not ground_truth:
        raise ValueError("No finalized clean Chainlink delayed ground-truth events were found.")

    source_index = {name: 0 for name in SOURCE_NAMES}
    source_current: dict[str, SecondSourceUpdate | None] = {name: None for name in SOURCE_NAMES}
    event_index = 0
    current_chainlink_event: ChainlinkGroundTruthEvent | None = None
    next_chainlink_event = ground_truth[0] if ground_truth else None

    start_ts_ns = ground_truth[0].chainlink_ts_ns
    end_ts_ns = ground_truth[-1].chainlink_ts_ns
    total_seconds = ((end_ts_ns - start_ts_ns) // SECOND_NS) + 1

    candidate_error_store = {candidate_name: [] for candidate_name in METHOD_PREFERENCE}
    event_rows: list[dict[str, Any]] = []
    best_method_second_errors: list[tuple[int, float, int | None, int | None]] = []
    source_count_rows: Counter[int] = Counter()
    spot_mid_grid: list[float | None] = []
    second_timestamps: list[int] = []
    delayed_calibrator = DelayedChainlinkBiasCalibrator()
    pending_delayed_updates: deque[tuple[int, float, float]] = deque()
    page_arrival_delays_s = [
        max(0.0, (event.first_seen_recv_ts_ns - event.chainlink_ts_ns) / SECOND_NS)
        for event in ground_truth
    ]

    for second_offset in range(total_seconds):
        second_ts_ns = start_ts_ns + (second_offset * SECOND_NS)
        second_timestamps.append(second_ts_ns)

        while pending_delayed_updates and pending_delayed_updates[0][0] <= second_ts_ns:
            _available_ts_ns, base_price, observed_chainlink_price = pending_delayed_updates.popleft()
            delayed_calibrator.observe(
                base_synthetic_price=base_price,
                observed_chainlink_price=observed_chainlink_price,
            )

        for source_name in SOURCE_NAMES:
            updates = source_updates[source_name]
            data_present = False
            while source_index[source_name] < len(updates) and updates[source_index[source_name]].second_ts_ns <= second_ts_ns:
                update = updates[source_index[source_name]]
                source_current[source_name] = update
                source_index[source_name] += 1
                data_present = update.second_ts_ns == second_ts_ns or data_present
            if data_present:
                pass

        source_payload = {
            source_name: _source_grid_payload(
                source_current[source_name],
                second_ts_ns=second_ts_ns,
                data_present=bool(
                    source_current[source_name] is not None
                    and source_current[source_name].second_ts_ns == second_ts_ns
                ),
            )
            for source_name in SOURCE_NAMES
        }
        source_payload["_delayed_chainlink_calibration"] = delayed_calibrator.export_metadata()
        snapshot = compute_candidate_snapshot(source_payload, recv_ts_ns=second_ts_ns)
        spot_mid_grid.append(snapshot.source_states["binance_spot"].mid)

        if next_chainlink_event is not None and next_chainlink_event.chainlink_ts_ns == second_ts_ns:
            current_chainlink_event = next_chainlink_event
            event_index += 1
            next_chainlink_event = ground_truth[event_index] if event_index < len(ground_truth) else None
            row = {
                "chainlink_ts_ns": current_chainlink_event.chainlink_ts_ns,
                "chainlink_price": current_chainlink_event.chainlink_price,
                "source_count": snapshot.source_count,
                "max_spread_usd": snapshot.max_spread_usd,
                "outlier_source": snapshot.outlier_source,
                "candidate_values": dict(snapshot.candidate_values),
                "candidate_errors": {},
            }
            source_count_rows[snapshot.source_count] += 1
            for candidate_name, candidate_value in snapshot.candidate_values.items():
                if candidate_value is None:
                    continue
                error = candidate_value - current_chainlink_event.chainlink_price
                candidate_error_store[candidate_name].append(error)
                row["candidate_errors"][candidate_name] = error
            base_candidate_value = snapshot.candidate_values.get(BASE_SYNTHETIC_CHAINLINK_METHOD)
            if base_candidate_value is not None:
                pending_delayed_updates.append(
                    (
                        _delayed_calibration_available_ts_ns(current_chainlink_event),
                        base_candidate_value,
                        current_chainlink_event.chainlink_price,
                    )
                )
            event_rows.append(row)

        if current_chainlink_event is not None:
            best_candidate_value = snapshot.candidate_values.get(SYNTHETIC_CHAINLINK_METHOD)
            if best_candidate_value is not None:
                seconds_since_last_update = int((second_ts_ns - current_chainlink_event.chainlink_ts_ns) // SECOND_NS)
                seconds_until_next_update = (
                    int((next_chainlink_event.chainlink_ts_ns - second_ts_ns) // SECOND_NS)
                    if next_chainlink_event is not None
                    else None
                )
                best_method_second_errors.append(
                    (
                        second_ts_ns,
                        best_candidate_value - current_chainlink_event.chainlink_price,
                        seconds_since_last_update,
                        seconds_until_next_update,
                    )
                )

    realized_vol = _rolling_volatility_bucket(spot_mid_grid)
    event_vol_samples = [realized_vol[(row["chainlink_ts_ns"] - start_ts_ns) // SECOND_NS] for row in event_rows]
    vol_values = [value for value in event_vol_samples if value is not None]
    low_cut, high_cut = _terciles(vol_values)

    stats_by_candidate = [
        _stats_from_errors(
            candidate_name=candidate_name,
            errors=candidate_error_store[candidate_name],
            total_events=len(ground_truth),
        )
        for candidate_name in METHOD_PREFERENCE
    ]
    stats_by_candidate.sort(key=_method_sort_key)
    winning_candidate = stats_by_candidate[0]

    if winning_candidate.candidate_name != SYNTHETIC_CHAINLINK_METHOD:
        raise ValueError(
            f"Configured synthetic builder method {SYNTHETIC_CHAINLINK_METHOD!r} does not match "
            f"the empirically winning candidate {winning_candidate.candidate_name!r}."
        )

    time_of_day_errors: dict[str, list[float]] = {}
    day_of_week_errors: dict[str, list[float]] = {}
    volatility_bucket_errors: dict[str, list[float]] = {}
    source_availability_errors: dict[str, list[float]] = {}
    selected_event_rows = [
        row
        for row in event_rows
        if row["candidate_errors"].get(SYNTHETIC_CHAINLINK_METHOD) is not None
    ]

    for row in selected_event_rows:
        error = float(row["candidate_errors"][SYNTHETIC_CHAINLINK_METHOD])
        vol_value = realized_vol[(row["chainlink_ts_ns"] - start_ts_ns) // SECOND_NS]
        ts = datetime.fromtimestamp(row["chainlink_ts_ns"] / SECOND_NS, tz=UTC)
        tod_bucket = f"{(ts.hour // 4) * 4:02d}-{((ts.hour // 4) * 4) + 3:02d} UTC"
        day_bucket = ts.strftime("%A")
        vol_bucket = _vol_bucket(vol_value, lower=low_cut, upper=high_cut)
        availability_bucket = f"{row['source_count']} sources"
        time_of_day_errors.setdefault(tod_bucket, []).append(error)
        day_of_week_errors.setdefault(day_bucket, []).append(error)
        volatility_bucket_errors.setdefault(vol_bucket, []).append(error)
        source_availability_errors.setdefault(availability_bucket, []).append(error)

    proximity_errors: dict[str, list[float]] = {}
    for _second_ts_ns, error, since_last, until_next in best_method_second_errors:
        label = "other"
        if until_next is not None and 0 <= until_next < 5:
            label = "0-5s_before_next_update"
        elif since_last is not None and 0 <= since_last < 5:
            label = "0-5s_after_update"
        elif since_last is not None and 5 <= since_last < 10:
            label = "5-10s_after_update"
        elif since_last is not None and 10 <= since_last < 15:
            label = "10-15s_after_update"
        elif since_last is not None and 15 <= since_last < 20:
            label = "15-20s_after_update"
        elif since_last is not None and 20 <= since_last < 25:
            label = "20-25s_after_update"
        elif since_last is not None and 25 <= since_last < 30:
            label = "25-30s_after_update"
        proximity_errors.setdefault(label, []).append(error)

    validation_rows = _candidate_table_rows(stats_by_candidate)
    time_of_day_rows = _bucket_error_rows(dict(sorted(time_of_day_errors.items())))
    day_of_week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_week_rows = _bucket_error_rows({name: day_of_week_errors[name] for name in day_of_week_order if name in day_of_week_errors})
    volatility_bucket_rows = _bucket_error_rows(
        {name: volatility_bucket_errors[name] for name in ("low", "medium", "high", "spot_missing") if name in volatility_bucket_errors}
    )
    source_availability_rows = _bucket_error_rows(
        {name: source_availability_errors[name] for name in sorted(source_availability_errors)}
    )
    proximity_order = [
        "0-5s_after_update",
        "5-10s_after_update",
        "10-15s_after_update",
        "15-20s_after_update",
        "20-25s_after_update",
        "25-30s_after_update",
        "0-5s_before_next_update",
        "other",
    ]
    proximity_rows = _bucket_error_rows({name: proximity_errors[name] for name in proximity_order if name in proximity_errors})

    artifact_dir.mkdir(parents=True, exist_ok=True)
    validation_report_path = artifact_dir / "synthetic_chainlink_validation_report.md"
    known_failure_modes_path = artifact_dir / "known_failure_modes.md"
    decision_log_path = artifact_dir / "decision_log.md"
    json_report_path = artifact_dir / "synthetic_chainlink_validation_report.json"

    validation_md = [
        "# Synthetic Chainlink Validation Report",
        "",
        "## Scope",
        "",
        "- Task: reconstruct a synthetic BTC/USD Chainlink proxy from recorded venue mids only",
        "- Official price input: Binance Spot WS",
        "- Diagnostic sources retained outside the official price: Binance USD-M WS, Hyperliquid WS",
        "- Delayed Chainlink page is used as historical ground truth and delayed calibration input only",
        "- Excluded by policy: RTDS, all Polymarket-side price inputs, active files, non-finalized or non-clean files",
        f"- Builder method frozen in code: `{SYNTHETIC_CHAINLINK_METHOD}`",
        f"- Delayed calibration availability in this report uses a conservative floor of {int(DELAYED_CHAINLINK_CALIBRATION_DELAY_SECONDS)} seconds",
        f"- Empirical page-arrival delay after the displayed Chainlink timestamp: median {_format_float(_percentile(page_arrival_delays_s, 0.5))} s, "
        f"p90 {_format_float(_percentile(page_arrival_delays_s, 0.9))} s, p99 {_format_float(_percentile(page_arrival_delays_s, 0.99))} s",
        "",
        "## Validation Table",
        "",
        _markdown_table(
            validation_rows,
            [
                ("candidate", "Candidate"),
                ("matched_events", "Matched Events"),
                ("coverage", "Coverage"),
                ("mean_error_usd", "Mean Error USD"),
                ("median_abs_error_usd", "Median Abs Error USD"),
                ("p90_abs_error_usd", "P90 Abs Error USD"),
                ("p99_abs_error_usd", "P99 Abs Error USD"),
                ("within_1_usd", "Within +/-1 USD"),
                ("within_3_usd", "Within +/-3 USD"),
                ("within_10_usd", "Within +/-10 USD"),
            ],
        ),
        "",
        "## Selected Candidate",
        "",
        f"- Winner: `{winning_candidate.candidate_name}`",
        f"- Median absolute error: {_format_float(winning_candidate.median_abs_error_usd)} USD",
        f"- P90 absolute error: {_format_float(winning_candidate.p90_abs_error_usd)} USD",
        f"- Matched events: {winning_candidate.matched_event_count}",
        "",
        "## Error Breakdown: Time Of Day",
        "",
        _markdown_table(
            time_of_day_rows,
            [
                ("bucket", "Bucket"),
                ("count", "Count"),
                ("mean_error_usd", "Mean Error USD"),
                ("median_abs_error_usd", "Median Abs Error USD"),
                ("p90_abs_error_usd", "P90 Abs Error USD"),
            ],
        ),
        "",
        "## Error Breakdown: Day Of Week",
        "",
        _markdown_table(
            day_of_week_rows,
            [
                ("bucket", "Bucket"),
                ("count", "Count"),
                ("mean_error_usd", "Mean Error USD"),
                ("median_abs_error_usd", "Median Abs Error USD"),
                ("p90_abs_error_usd", "P90 Abs Error USD"),
            ],
        ),
        "",
        "## Error Breakdown: Binance Spot Realized Volatility",
        "",
        _markdown_table(
            volatility_bucket_rows,
            [
                ("bucket", "Bucket"),
                ("count", "Count"),
                ("mean_error_usd", "Mean Error USD"),
                ("median_abs_error_usd", "Median Abs Error USD"),
                ("p90_abs_error_usd", "P90 Abs Error USD"),
            ],
        ),
        "",
        "## Error Breakdown: Source Availability",
        "",
        _markdown_table(
            source_availability_rows,
            [
                ("bucket", "Bucket"),
                ("count", "Count"),
                ("mean_error_usd", "Mean Error USD"),
                ("median_abs_error_usd", "Median Abs Error USD"),
                ("p90_abs_error_usd", "P90 Abs Error USD"),
            ],
        ),
        "",
        "## Error Breakdown: Proximity To Chainlink Update",
        "",
        _markdown_table(
            proximity_rows,
            [
                ("bucket", "Bucket"),
                ("count", "Count"),
                ("mean_error_usd", "Mean Error USD"),
                ("median_abs_error_usd", "Median Abs Error USD"),
                ("p90_abs_error_usd", "P90 Abs Error USD"),
            ],
        ),
        "",
    ]
    validation_report_path.write_text("\n".join(validation_md), encoding="utf-8")

    failure_mode_lines = [
        "# Known Failure Modes",
        "",
        f"Selected builder: `{winning_candidate.candidate_name}`",
        "",
        "## Structural Limits",
        "",
        "- The synthetic builder never uses RTDS or Polymarket data.",
        "- Any source older than 30 seconds is treated as missing.",
        "- The builder returns `NaN` when Binance spot is missing or older than 30 seconds.",
        f"- Delayed calibration updates are only applied after a conservative {int(DELAYED_CHAINLINK_CALIBRATION_DELAY_SECONDS)} second availability lag.",
        "",
        "## Conditions With Elevated Error",
        "",
    ]
    elevated_sections: list[str] = []
    for label, rows in (
        ("Time of day", time_of_day_rows),
        ("Day of week", day_of_week_rows),
        ("Volatility regime", volatility_bucket_rows),
        ("Source availability", source_availability_rows),
        ("Update proximity", proximity_rows),
    ):
        elevated = [
            row
            for row in rows
            if (row.get("median_abs_error_usd") or 0.0) > 3.0 or (row.get("p90_abs_error_usd") or 0.0) > 3.0
        ]
        if not elevated:
            continue
        elevated_sections.append(f"### {label}")
        elevated_sections.append("")
        for row in elevated:
            elevated_sections.append(
                f"- `{row['bucket']}`: median abs error {_format_float(row['median_abs_error_usd'])} USD, "
                f"p90 {_format_float(row['p90_abs_error_usd'])} USD, count {row['count']}"
            )
        elevated_sections.append("")
    if not elevated_sections:
        elevated_sections.extend(
            [
                "- No bucket crossed the documented ±3 USD median or p90 warning threshold in the current sample.",
                "",
            ]
        )
    failure_mode_lines.extend(elevated_sections)
    failure_mode_lines.extend(
        [
            "## Mitigations",
            "",
            "- Keep the finalized-clean quality gate and Binance USD-M quarantine hour in force.",
            "- Treat missing Binance spot state as a hard builder miss because the selected method is spot-anchored.",
            "- Keep the delayed-bias state clipped and slowly updated; do not replace it with unrestricted tree-model predictions in live trading.",
            "- Keep multi-source candidates available for future research, but do not promote them over the validated winner without a fresh validation run.",
            "- Treat low-source-count seconds as degraded diagnostics, not as a reason to replace the spot-only official price.",
            "",
        ]
    )
    known_failure_modes_path.write_text("\n".join(failure_mode_lines), encoding="utf-8")

    decision_log_lines = []
    alternative_candidates = [
        candidate_name
        for candidate_name in METHOD_PREFERENCE
        if candidate_name != winning_candidate.candidate_name
    ]
    if decision_log_path.exists():
        existing_decision_text = decision_log_path.read_text(encoding="utf-8")
        decision_log_lines.append(
            _without_existing_decision_entry(existing_decision_text, heading=DECISION_LOG_HEADING).rstrip()
        )
        decision_log_lines.append("")
    else:
        decision_log_lines.extend(["# Decision Log", ""])
    decision_log_lines.extend(
        [
            DECISION_LOG_HEADING,
            "",
            "- Task: reconstruct a training-safe and live-safe synthetic BTC/USD Chainlink reference without RTDS.",
            f"- Decision: select `{winning_candidate.candidate_name}` as `synthetic_chainlink_price`.",
            (
                "- Evidence: median abs error "
                f"{_format_float(winning_candidate.median_abs_error_usd)} USD, "
                f"p90 {_format_float(winning_candidate.p90_abs_error_usd)} USD, "
                f"coverage {_format_pct(winning_candidate.event_coverage)} on {winning_candidate.matched_event_count} Chainlink update events."
            ),
            "- Alternatives considered: " + ", ".join(f"`{name}`" for name in alternative_candidates) + ".",
            (
                "- Constraints respected: finalized-clean only, Binance USD-M quarantine hour excluded, causal recv_ts_ns joins only, "
                f"delayed Chainlink used only with a conservative {int(DELAYED_CHAINLINK_CALIBRATION_DELAY_SECONDS)} second availability lag."
            ),
            "",
        ]
    )
    decision_log_path.write_text("\n".join(decision_log_lines).strip() + "\n", encoding="utf-8")

    summary_payload = {
        "builder_method": SYNTHETIC_CHAINLINK_METHOD,
        "winning_candidate": winning_candidate.as_dict(),
        "candidate_stats": [stats.as_dict() for stats in stats_by_candidate],
        "time_of_day_breakdown": time_of_day_rows,
        "day_of_week_breakdown": day_of_week_rows,
        "volatility_breakdown": volatility_bucket_rows,
        "source_availability_breakdown": source_availability_rows,
        "proximity_breakdown": proximity_rows,
        "source_counts_at_events": {str(key): value for key, value in source_count_rows.items()},
        "delayed_chainlink_assumptions": {
            "conservative_availability_delay_seconds": DELAYED_CHAINLINK_CALIBRATION_DELAY_SECONDS,
            "ewma_alpha": DELAYED_CHAINLINK_EWMA_ALPHA,
            "residual_clip_usd": DELAYED_CHAINLINK_RESIDUAL_CLIP_USD,
            "empirical_page_arrival_delay_seconds": {
                "median": _percentile(page_arrival_delays_s, 0.5),
                "p90": _percentile(page_arrival_delays_s, 0.9),
                "p99": _percentile(page_arrival_delays_s, 0.99),
            },
        },
        "artifact_paths": {
            "validation_report_md": str(validation_report_path),
            "known_failure_modes_md": str(known_failure_modes_path),
            "decision_log_md": str(decision_log_path),
        },
    }
    json_report_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    summary_payload["artifact_paths"]["validation_report_json"] = str(json_report_path)
    return summary_payload
