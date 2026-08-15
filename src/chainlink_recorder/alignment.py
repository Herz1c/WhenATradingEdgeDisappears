from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

from binance_recorder.utils import parse_utc_date
from polymarket_recorder.rtds_reader import PolymarketRtdsRawReader

from .live_reference import normalize_public_delayed_record, normalize_rtds_record
from .reader import ChainlinkPublicDelayedRawReader


@dataclass(slots=True)
class AlignmentPoint:
    recv_ts_ns: int
    price: float
    source_ts_ns: int | None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


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


def _iter_dates(date_from: date, date_to: date) -> list[date]:
    current = date_from
    dates: list[date] = []
    while current <= date_to:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def _normalized_record_to_point(record: dict[str, Any]) -> AlignmentPoint | None:
    recv_ts_ns = _to_int(record.get("recv_ts_ns"))
    price = _to_float(record.get("btc_usd_price"))
    if recv_ts_ns is None or price is None:
        return None
    return AlignmentPoint(
        recv_ts_ns=recv_ts_ns,
        price=price,
        source_ts_ns=_to_int(record.get("chainlink_ts_ns")),
    )


def load_rtds_points(
    *,
    root: Path,
    date_from: date | str,
    date_to: date | str,
    rtds_symbol: str = "btc/usd",
    finalized_only: bool = True,
    tail_tolerant: bool = False,
) -> list[AlignmentPoint]:
    reader = PolymarketRtdsRawReader(root=root, symbol=rtds_symbol)
    start = parse_utc_date(date_from) if isinstance(date_from, str) else date_from
    end = parse_utc_date(date_to) if isinstance(date_to, str) else date_to
    points: list[AlignmentPoint] = []
    for target_day in _iter_dates(start, end):
        for record in reader.iter_kind(
            target_date=target_day.isoformat(),
            kind="ws",
            message_type="update",
            topic="crypto_prices_chainlink",
            symbol=rtds_symbol,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            point = _normalized_record_to_point(normalize_rtds_record(record))
            if point is not None:
                points.append(point)
    return sorted(points, key=lambda item: item.recv_ts_ns)


def load_public_delayed_points(
    *,
    root: Path,
    date_from: date | str,
    date_to: date | str,
    symbol: str = "BTCUSD",
    finalized_only: bool = True,
    tail_tolerant: bool = False,
) -> list[AlignmentPoint]:
    reader = ChainlinkPublicDelayedRawReader(root=root, symbol=symbol)
    start = parse_utc_date(date_from) if isinstance(date_from, str) else date_from
    end = parse_utc_date(date_to) if isinstance(date_to, str) else date_to
    points: list[AlignmentPoint] = []
    for target_day in _iter_dates(start, end):
        for record in reader.iter_kind(
            target_date=target_day.isoformat(),
            kind="page",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            point = _normalized_record_to_point(normalize_public_delayed_record(record))
            if point is not None:
                points.append(point)
    return sorted(points, key=lambda item: item.recv_ts_ns)


def _match_points(
    rtds_points: list[AlignmentPoint],
    public_points: list[AlignmentPoint],
    *,
    offset_ns: int,
    tolerance_ns: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    rtds_index = 0
    public_index = 0
    while rtds_index < len(rtds_points) and public_index < len(public_points):
        rtds_point = rtds_points[rtds_index]
        public_point = public_points[public_index]
        shifted_public_ts_ns = public_point.recv_ts_ns - offset_ns
        delta_ns = shifted_public_ts_ns - rtds_point.recv_ts_ns
        if abs(delta_ns) <= tolerance_ns:
            matches.append(
                {
                    "rtds": rtds_point,
                    "public": public_point,
                    "alignment_delta_ms": delta_ns / 1_000_000,
                    "lag_ms": (public_point.recv_ts_ns - rtds_point.recv_ts_ns) / 1_000_000,
                    "price_error": public_point.price - rtds_point.price,
                }
            )
            rtds_index += 1
            public_index += 1
            continue
        if shifted_public_ts_ns < rtds_point.recv_ts_ns:
            public_index += 1
        else:
            rtds_index += 1
    return matches


def compute_alignment_metrics(
    rtds_points: list[AlignmentPoint],
    public_points: list[AlignmentPoint],
    *,
    max_offset_seconds: int = 120,
    tolerance_seconds: int = 2,
    near_match_abs_error: float = 1.0,
) -> dict[str, Any]:
    if not rtds_points or not public_points:
        return {
            "status": "insufficient_data",
            "pair_count": 0,
            "interpretation": "RTDS does not reliably align with the public delayed page.",
        }

    tolerance_ns = tolerance_seconds * 1_000_000_000
    best_offset_ms: int | None = None
    best_matches: list[dict[str, Any]] = []
    best_mae: float | None = None
    best_alignment_residual_mae: float | None = None

    for offset_seconds in range(-max_offset_seconds, max_offset_seconds + 1):
        matches = _match_points(
            rtds_points,
            public_points,
            offset_ns=offset_seconds * 1_000_000_000,
            tolerance_ns=tolerance_ns,
        )
        if len(matches) < 5:
            continue
        mae = mean(abs(match["price_error"]) for match in matches)
        alignment_residual_mae = mean(abs(match["alignment_delta_ms"]) for match in matches)
        if (
            best_mae is None
            or mae < best_mae
            or (
                mae == best_mae
                and (
                    best_alignment_residual_mae is None
                    or alignment_residual_mae < best_alignment_residual_mae
                    or (
                        alignment_residual_mae == best_alignment_residual_mae
                        and len(matches) > len(best_matches)
                    )
                )
            )
        ):
            best_mae = mae
            best_alignment_residual_mae = alignment_residual_mae
            best_offset_ms = offset_seconds * 1_000
            best_matches = matches

    if best_offset_ms is None or not best_matches:
        return {
            "status": "insufficient_data",
            "pair_count": 0,
            "interpretation": "RTDS does not reliably align with the public delayed page.",
        }

    lags_ms = [float(match["lag_ms"]) for match in best_matches]
    alignment_deltas_ms = [float(match["alignment_delta_ms"]) for match in best_matches]
    abs_errors = [abs(float(match["price_error"])) for match in best_matches]
    rounded_exact_matches = [
        round(match["public"].price, 2) == round(match["rtds"].price, 2)
        for match in best_matches
    ]
    near_matches = [error <= near_match_abs_error for error in abs_errors]

    lag_median_ms = median(lags_ms)
    lag_p95_ms = _percentile(lags_ms, 0.95)
    lag_max_ms = max(lags_ms)
    lag_spread_ms = None if lag_p95_ms is None else lag_p95_ms - lag_median_ms
    span_hours = max(
        (best_matches[-1]["rtds"].recv_ts_ns - best_matches[0]["rtds"].recv_ts_ns) / 3_600_000_000_000,
        0.0,
    )
    drift_slope_ms_per_hour = 0.0
    if span_hours > 0:
        drift_slope_ms_per_hour = (lags_ms[-1] - lags_ms[0]) / span_hours

    volatility_pairs: list[float] = []
    calm_pairs: list[float] = []
    step_changes = [
        abs(best_matches[index]["rtds"].price - best_matches[index - 1]["rtds"].price)
        for index in range(1, len(best_matches))
    ]
    volatility_threshold = _percentile(step_changes, 0.75) if step_changes else None
    if volatility_threshold is not None:
        for index in range(1, len(best_matches)):
            bucket = volatility_pairs if step_changes[index - 1] >= volatility_threshold else calm_pairs
            bucket.append(abs_errors[index])

    if lag_spread_ms is not None and lag_spread_ms <= 2_000 and abs(drift_slope_ms_per_hour) <= 500:
        lag_relationship = "stable"
    elif lag_spread_ms is not None and lag_spread_ms <= 5_000 and abs(drift_slope_ms_per_hour) <= 2_000:
        lag_relationship = "mildly_drifting"
    else:
        lag_relationship = "drifting"

    mean_abs_price_error = mean(abs_errors)
    exact_match_rate = sum(rounded_exact_matches) / len(rounded_exact_matches)
    near_match_rate = sum(near_matches) / len(near_matches)
    if near_match_rate >= 0.8 and lag_relationship == "stable":
        interpretation = f"RTDS matches public delayed Chainlink closely after ~{round(lag_median_ms / 1000, 1)} second offset."
        status = "aligned"
    elif near_match_rate >= 0.5:
        interpretation = "RTDS partially aligns with the public delayed page, but the delay relationship is not fully stable."
        status = "partial_alignment"
    else:
        interpretation = "RTDS does not reliably align with the public delayed page."
        status = "mismatch"

    return {
        "status": status,
        "pair_count": len(best_matches),
        "best_offset_ms": best_offset_ms,
        "median_lag_ms": lag_median_ms,
        "p95_lag_ms": lag_p95_ms,
        "max_lag_ms": lag_max_ms,
        "alignment_residual_p95_ms": _percentile(alignment_deltas_ms, 0.95),
        "mean_abs_price_error": mean_abs_price_error,
        "exact_match_rate": exact_match_rate,
        "near_match_rate": near_match_rate,
        "lag_relationship": lag_relationship,
        "drift_slope_ms_per_hour": drift_slope_ms_per_hour,
        "volatile_mean_abs_price_error": mean(volatility_pairs) if volatility_pairs else None,
        "calm_mean_abs_price_error": mean(calm_pairs) if calm_pairs else None,
        "divergence_during_volatile_periods": (
            "higher"
            if volatility_pairs and calm_pairs and mean(volatility_pairs) > mean(calm_pairs) * 1.25
            else "similar"
            if volatility_pairs and calm_pairs
            else "insufficient_data"
        ),
        "interpretation": interpretation,
        "caution": "Alignment is an empirical delay check only and does not prove settlement-feed equivalence.",
    }


def build_chainlink_alignment_report(
    *,
    root: Path,
    date_from: date | str,
    date_to: date | str,
    symbol: str = "BTCUSD",
    rtds_symbol: str = "btc/usd",
    finalized_only: bool = True,
    tail_tolerant: bool = False,
    by_hour: bool = False,
    max_offset_seconds: int = 120,
    tolerance_seconds: int = 2,
    near_match_abs_error: float = 1.0,
) -> dict[str, Any]:
    start = parse_utc_date(date_from) if isinstance(date_from, str) else date_from
    end = parse_utc_date(date_to) if isinstance(date_to, str) else date_to
    rtds_points = load_rtds_points(
        root=root,
        date_from=start,
        date_to=end,
        rtds_symbol=rtds_symbol,
        finalized_only=finalized_only,
        tail_tolerant=tail_tolerant,
    )
    public_points = load_public_delayed_points(
        root=root,
        date_from=start,
        date_to=end,
        symbol=symbol,
        finalized_only=finalized_only,
        tail_tolerant=tail_tolerant,
    )
    report = {
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "rtds_point_count": len(rtds_points),
        "public_delayed_point_count": len(public_points),
        "finalized_only": finalized_only,
        "tail_tolerant": tail_tolerant,
        "summary": compute_alignment_metrics(
            rtds_points,
            public_points,
            max_offset_seconds=max_offset_seconds,
            tolerance_seconds=tolerance_seconds,
            near_match_abs_error=near_match_abs_error,
        ),
    }
    if by_hour:
        hourly_reports: list[dict[str, Any]] = []
        buckets = sorted(
            {
                datetime.fromtimestamp(point.recv_ts_ns / 1_000_000_000, tz=UTC).replace(
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                for point in rtds_points
            }
        )
        for bucket_start in buckets:
            bucket_end = bucket_start + timedelta(hours=1)
            bucket_rtds = [
                point for point in rtds_points if bucket_start <= datetime.fromtimestamp(point.recv_ts_ns / 1_000_000_000, tz=UTC) < bucket_end
            ]
            bucket_public = [
                point for point in public_points if bucket_start <= datetime.fromtimestamp(point.recv_ts_ns / 1_000_000_000, tz=UTC) < bucket_end
            ]
            if not bucket_rtds or not bucket_public:
                continue
            hourly_reports.append(
                {
                    "hour_start_iso": bucket_start.isoformat().replace("+00:00", "Z"),
                    "summary": compute_alignment_metrics(
                        bucket_rtds,
                        bucket_public,
                        max_offset_seconds=max_offset_seconds,
                        tolerance_seconds=tolerance_seconds,
                        near_match_abs_error=near_match_abs_error,
                    ),
                }
            )
        report["hourly"] = hourly_reports
        status_counts: dict[str, int] = {}
        unstable_hours = 0
        for hourly in hourly_reports:
            status = str(hourly["summary"].get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            if str(hourly["summary"].get("lag_relationship")) != "stable":
                unstable_hours += 1
        report["regime_summary"] = {
            "hour_count": len(hourly_reports),
            "status_counts": status_counts,
            "unstable_hour_count": unstable_hours,
            "alignment_stability": (
                "stable"
                if hourly_reports and unstable_hours == 0 and status_counts.get("aligned", 0) == len(hourly_reports)
                else "mixed"
                if hourly_reports
                else "insufficient_data"
            ),
        }
        report["quality_flags"] = [
            flag
            for flag, enabled in (
                ("unstable_alignment_regime", unstable_hours > 0),
                ("hourly_mismatch_detected", status_counts.get("mismatch", 0) > 0),
                ("hourly_partial_alignment_detected", status_counts.get("partial_alignment", 0) > 0),
            )
            if enabled
        ]
    return report
