"""Measure the public delayed Chainlink page's lag against CEX spot, by hour and regime."""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from binance_recorder.compression import iter_zstd_jsonl
from chainlink_recorder.synthetic_chainlink import load_clean_source_updates
from market_recorders.unified_reader import UnifiedRawReader

SECOND_NS = 1_000_000_000
BINANCE_USDM_QUARANTINE_HOURS = {("2026-04-20", "07")}
PRICE_TOLERANCE_USD = 0.01
NEGATIVE_MATCH_WINDOW_SECONDS = 600.0
POSITIVE_MATCH_WINDOW_SECONDS = 3600.0


@dataclass(slots=True, frozen=True)
class OnchainUpdateEvent:
    recv_ts_ns: int
    price: float
    round_id: str


@dataclass(slots=True, frozen=True)
class PageObservation:
    recv_ts_ns: int
    price: float
    display_ts_ns: int | None


@dataclass(slots=True, frozen=True)
class DelayMatch:
    onchain_recv_ts_ns: int
    delayed_recv_ts_ns: int
    onchain_price: float
    delayed_price: float
    delivery_delay_seconds: float
    hour_of_day: int
    day_of_week: str
    volatility_regime: str


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


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _ceil_second_ns(ts_ns: int) -> int:
    return ((ts_ns + SECOND_NS - 1) // SECOND_NS) * SECOND_NS


def _rolling_volatility_bucket(values: list[float | None]) -> list[float | None]:
    realized_vol: list[float | None] = [None] * len(values)
    rolling_returns: list[float] = []
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
            rolling_sum -= rolling_returns.pop(0)
        previous_mid = mid
        realized_vol[index] = math.sqrt(max(rolling_sum, 0.0))
    return realized_vol


def _terciles(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return _percentile(values, 1.0 / 3.0), _percentile(values, 2.0 / 3.0)


def _vol_bucket(value: float | None, *, lower: float | None, upper: float | None) -> str:
    if value is None or lower is None or upper is None:
        return "spot_missing"
    if value <= lower:
        return "low"
    if value <= upper:
        return "medium"
    return "high"


def _valid_clean_files(
    *,
    root: Path,
    relative_prefix: Path,
    kind: str,
    excluded_hours: set[tuple[str, str]] | None = None,
) -> list[Path]:
    reader = UnifiedRawReader(root=root)
    excluded_hours = excluded_hours or set()
    files: list[Path] = []
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
            files.append(path)
    return files


def _run_parallel(loader: Any, paths: list[str], *, max_workers: int) -> list[Any]:
    if not paths:
        return []
    if max_workers <= 1:
        return [loader(path) for path in paths]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(loader, paths))


def _parse_onchain_file(path_str: str) -> list[OnchainUpdateEvent]:
    path = Path(path_str)
    events: list[OnchainUpdateEvent] = []
    last_round_id: str | None = None
    for record in iter_zstd_jsonl(path):
        round_data = record.get("round_data") or {}
        round_id = str(round_data.get("round_id") or "")
        recv_ts_ns = _to_int(record.get("recv_ts_ns"))
        price = _to_float(round_data.get("answer"))
        if not round_id or recv_ts_ns is None or price is None:
            continue
        if round_id == last_round_id:
            continue
        last_round_id = round_id
        events.append(
            OnchainUpdateEvent(
                recv_ts_ns=recv_ts_ns,
                price=price,
                round_id=round_id,
            )
        )
    return events


def _parse_page_file(path_str: str) -> list[PageObservation]:
    path = Path(path_str)
    observations: list[PageObservation] = []
    last_price: float | None = None
    for record in iter_zstd_jsonl(path):
        recv_ts_ns = _to_int(record.get("recv_ts_ns"))
        price = _to_float(record.get("btc_usd_price"))
        if recv_ts_ns is None or price is None:
            continue
        if last_price is not None and abs(price - last_price) <= PRICE_TOLERANCE_USD:
            continue
        last_price = price
        display_ts_ns = _to_int(record.get("chainlink_display_ts"))
        if display_ts_ns is not None and abs(display_ts_ns) < 10_000_000_000:
            display_ts_ns *= SECOND_NS
        observations.append(
            PageObservation(
                recv_ts_ns=recv_ts_ns,
                price=price,
                display_ts_ns=display_ts_ns,
            )
        )
    return observations


def _flatten_onchain(chunks: list[list[OnchainUpdateEvent]]) -> list[OnchainUpdateEvent]:
    merged = sorted(
        (event for chunk in chunks for event in chunk),
        key=lambda event: event.recv_ts_ns,
    )
    flattened: list[OnchainUpdateEvent] = []
    last_round_id: str | None = None
    for event in merged:
        if event.round_id == last_round_id:
            continue
        last_round_id = event.round_id
        flattened.append(event)
    return flattened


def _flatten_page(chunks: list[list[PageObservation]]) -> list[PageObservation]:
    merged = sorted(
        (obs for chunk in chunks for obs in chunk),
        key=lambda obs: obs.recv_ts_ns,
    )
    flattened: list[PageObservation] = []
    last_price: float | None = None
    for observation in merged:
        if last_price is not None and abs(observation.price - last_price) <= PRICE_TOLERANCE_USD:
            continue
        last_price = observation.price
        flattened.append(observation)
    return flattened


def load_onchain_events(*, root: Path, max_workers: int) -> list[OnchainUpdateEvent]:
    prefix = Path("chainlink") / "onchain" / "BTCUSD"
    files = _valid_clean_files(
        root=root,
        relative_prefix=prefix,
        kind="onchain",
        excluded_hours=BINANCE_USDM_QUARANTINE_HOURS,
    )
    chunks = _run_parallel(_parse_onchain_file, [str(path) for path in files], max_workers=max_workers)
    return _flatten_onchain(chunks)


def load_page_observations(*, root: Path, max_workers: int) -> list[PageObservation]:
    prefix = Path("chainlink_public_delayed") / "public_stream_page" / "BTCUSD"
    files = _valid_clean_files(
        root=root,
        relative_prefix=prefix,
        kind="page",
        excluded_hours=BINANCE_USDM_QUARANTINE_HOURS,
    )
    chunks = _run_parallel(_parse_page_file, [str(path) for path in files], max_workers=max_workers)
    return _flatten_page(chunks)


def build_spot_volatility_state(
    *,
    root: Path,
    max_workers: int,
) -> tuple[dict[int, float | None], float | None, float | None]:
    updates = load_clean_source_updates(root=root, source_name="binance_spot", max_workers=max_workers)
    if not updates:
        return {}, None, None
    start_ts_ns = updates[0].second_ts_ns
    end_ts_ns = updates[-1].second_ts_ns
    total_seconds = ((end_ts_ns - start_ts_ns) // SECOND_NS) + 1
    update_index = 0
    current_mid: float | None = None
    spot_mid_grid: list[float | None] = []
    second_keys: list[int] = []
    for second_offset in range(total_seconds):
        second_ts_ns = start_ts_ns + (second_offset * SECOND_NS)
        while update_index < len(updates) and updates[update_index].second_ts_ns <= second_ts_ns:
            current_mid = updates[update_index].mid
            update_index += 1
        spot_mid_grid.append(current_mid)
        second_keys.append(second_ts_ns)
    realized_vol = _rolling_volatility_bucket(spot_mid_grid)
    vol_values = [value for value in realized_vol if value is not None]
    lower, upper = _terciles(vol_values)
    vol_map = {second_ts_ns: realized_vol[index] for index, second_ts_ns in enumerate(second_keys)}
    return vol_map, lower, upper


def match_delays(
    *,
    onchain_events: list[OnchainUpdateEvent],
    page_observations: list[PageObservation],
    spot_vol_map: dict[int, float | None],
    vol_lower: float | None,
    vol_upper: float | None,
    negative_window_seconds: float,
    positive_window_seconds: float,
) -> tuple[list[DelayMatch], int, list[dict[str, Any]]]:
    matches: list[DelayMatch] = []
    diagnostics: list[dict[str, Any]] = []
    negative_window_ns = int(negative_window_seconds * SECOND_NS)
    positive_window_ns = int(positive_window_seconds * SECOND_NS)
    page_index = 0
    unmatched = 0
    for event in onchain_events:
        while page_index < len(page_observations) and page_observations[page_index].recv_ts_ns < event.recv_ts_ns - negative_window_ns:
            page_index += 1
        search_index = page_index
        found_index: int | None = None
        while search_index < len(page_observations):
            observation = page_observations[search_index]
            if observation.recv_ts_ns > event.recv_ts_ns + positive_window_ns:
                break
            if abs(observation.price - event.price) <= PRICE_TOLERANCE_USD:
                found_index = search_index
                break
            search_index += 1
        if found_index is None:
            unmatched += 1
            continue
        observation = page_observations[found_index]
        event_dt = datetime.fromtimestamp(event.recv_ts_ns / SECOND_NS, tz=UTC)
        vol_second = _ceil_second_ns(event.recv_ts_ns)
        vol_value = spot_vol_map.get(vol_second)
        match = DelayMatch(
            onchain_recv_ts_ns=event.recv_ts_ns,
            delayed_recv_ts_ns=observation.recv_ts_ns,
            onchain_price=event.price,
            delayed_price=observation.price,
            delivery_delay_seconds=(observation.recv_ts_ns - event.recv_ts_ns) / SECOND_NS,
            hour_of_day=event_dt.hour,
            day_of_week=event_dt.strftime("%A"),
            volatility_regime=_vol_bucket(vol_value, lower=vol_lower, upper=vol_upper),
        )
        matches.append(match)
        diagnostics.append(
            {
                "onchain_recv_ts_ns": event.recv_ts_ns,
                "onchain_recv_ts_iso": event_dt.isoformat().replace("+00:00", "Z"),
                "onchain_price": event.price,
                "round_id": event.round_id,
                "delayed_recv_ts_ns": observation.recv_ts_ns,
                "delayed_recv_ts_iso": datetime.fromtimestamp(observation.recv_ts_ns / SECOND_NS, tz=UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                "delayed_price": observation.price,
                "delayed_display_ts_ns": observation.display_ts_ns,
                "delayed_display_ts_iso": (
                    datetime.fromtimestamp(observation.display_ts_ns / SECOND_NS, tz=UTC)
                    .isoformat()
                    .replace("+00:00", "Z")
                    if observation.display_ts_ns is not None
                    else None
                ),
                "delivery_delay_seconds": match.delivery_delay_seconds,
                "volatility_regime": match.volatility_regime,
            }
        )
        page_index = found_index + 1
    return matches, unmatched, diagnostics


def _summary_stats(delays: list[float]) -> dict[str, float | None]:
    return {
        "mean_delay_seconds": _mean(delays),
        "median_delay_seconds": _percentile(delays, 0.5),
        "p5_delay_seconds": _percentile(delays, 0.05),
        "p95_delay_seconds": _percentile(delays, 0.95),
        "p99_delay_seconds": _percentile(delays, 0.99),
        "min_delay_seconds": min(delays) if delays else None,
        "max_delay_seconds": max(delays) if delays else None,
    }


def _bucket_label(bucket_start: int) -> str:
    bucket_end = bucket_start + 10
    return f"[{bucket_start}, {bucket_end})"


def _delay_histogram(delays: list[float]) -> list[dict[str, Any]]:
    buckets: dict[int, int] = {}
    for delay in delays:
        bucket_start = math.floor(delay / 10.0) * 10
        buckets[bucket_start] = buckets.get(bucket_start, 0) + 1
    return [
        {"bucket": _bucket_label(bucket_start), "count": buckets[bucket_start]}
        for bucket_start in sorted(buckets)
    ]


def _breakdown_rows(matches: list[DelayMatch], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for match in matches:
        bucket = getattr(match, key)
        grouped.setdefault(str(bucket), []).append(match.delivery_delay_seconds)
    rows: list[dict[str, Any]] = []
    for bucket, delays in sorted(grouped.items()):
        rows.append(
            {
                "bucket": bucket,
                "matched_events": len(delays),
                "mean_delay_seconds": _mean(delays),
                "median_delay_seconds": _percentile(delays, 0.5),
                "p95_delay_seconds": _percentile(delays, 0.95),
                "min_delay_seconds": min(delays),
                "max_delay_seconds": max(delays),
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows_"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(_format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    *,
    root: Path,
    artifact_dir: Path,
    max_workers: int,
    negative_window_seconds: float,
    positive_window_seconds: float,
) -> dict[str, Any]:
    onchain_events = load_onchain_events(root=root, max_workers=max_workers)
    page_observations = load_page_observations(root=root, max_workers=max_workers)
    spot_vol_map, vol_lower, vol_upper = build_spot_volatility_state(root=root, max_workers=max_workers)
    matches, unmatched, diagnostics = match_delays(
        onchain_events=onchain_events,
        page_observations=page_observations,
        spot_vol_map=spot_vol_map,
        vol_lower=vol_lower,
        vol_upper=vol_upper,
        negative_window_seconds=negative_window_seconds,
        positive_window_seconds=positive_window_seconds,
    )
    delays = [match.delivery_delay_seconds for match in matches]
    negative_delays = [delay for delay in delays if delay < 0]
    summary_stats = _summary_stats(delays)
    summary_rows = [
        {
            "metric": "onchain_update_events",
            "value": len(onchain_events),
        },
        {
            "metric": "matched_events",
            "value": len(matches),
        },
        {
            "metric": "match_coverage",
            "value": _format_pct(len(matches) / len(onchain_events) if onchain_events else None),
        },
        {
            "metric": "matching_window",
            "value": f"[{-negative_window_seconds:.0f}s, +{positive_window_seconds:.0f}s]",
        },
        {
            "metric": "price_tolerance_usd",
            "value": f"{PRICE_TOLERANCE_USD:.2f}",
        },
        {
            "metric": "mean_delay_seconds",
            "value": _format_float(summary_stats["mean_delay_seconds"]),
        },
        {
            "metric": "median_delay_seconds",
            "value": _format_float(summary_stats["median_delay_seconds"]),
        },
        {
            "metric": "p5_delay_seconds",
            "value": _format_float(summary_stats["p5_delay_seconds"]),
        },
        {
            "metric": "p95_delay_seconds",
            "value": _format_float(summary_stats["p95_delay_seconds"]),
        },
        {
            "metric": "p99_delay_seconds",
            "value": _format_float(summary_stats["p99_delay_seconds"]),
        },
        {
            "metric": "min_delay_seconds",
            "value": _format_float(summary_stats["min_delay_seconds"]),
        },
        {
            "metric": "max_delay_seconds",
            "value": _format_float(summary_stats["max_delay_seconds"]),
        },
        {
            "metric": "negative_delay_matches",
            "value": len(negative_delays),
        },
    ]

    hour_rows = _breakdown_rows(matches, "hour_of_day")
    day_rows = _breakdown_rows(matches, "day_of_week")
    vol_rows = _breakdown_rows(matches, "volatility_regime")
    histogram_rows = _delay_histogram(delays)

    report = {
        "parameters": {
            "root": str(root),
            "artifact_dir": str(artifact_dir),
            "max_workers": max_workers,
            "negative_window_seconds": negative_window_seconds,
            "positive_window_seconds": positive_window_seconds,
            "price_tolerance_usd": PRICE_TOLERANCE_USD,
            "excluded_hours": sorted([f"{day} {hour}:00 UTC" for day, hour in BINANCE_USDM_QUARANTINE_HOURS]),
        },
        "summary": {
            "onchain_update_events": len(onchain_events),
            "matched_events": len(matches),
            "unmatched_events": unmatched,
            "match_coverage": (len(matches) / len(onchain_events)) if onchain_events else None,
            **summary_stats,
            "negative_delay_matches": len(negative_delays),
        },
        "histogram_10s": histogram_rows,
        "breakdown_by_hour_of_day": hour_rows,
        "breakdown_by_day_of_week": day_rows,
        "breakdown_by_volatility_regime": vol_rows,
        "matched_pairs": diagnostics,
        "safe_causal_offset_recommendation": {
            "status": (
                "insufficient_exact_match_coverage"
                if len(matches) < max(10, math.ceil(len(onchain_events) * 0.25))
                else "supported_by_exact_match_distribution"
            ),
            "recommended_offset_seconds": (
                max(
                    positive_window_seconds / 2.0,
                    summary_stats["p95_delay_seconds"] or 0.0,
                )
                if len(matches) >= max(10, math.ceil(len(onchain_events) * 0.25))
                else 1800.0
            ),
            "reason": (
                "Exact +/- 0.01 USD value matching produced too few plausible pairs to estimate a robust operational "
                "delay distribution. Keep the conservative 1800s availability floor in the dataset factory."
                if len(matches) < max(10, math.ceil(len(onchain_events) * 0.25))
                else "Use at least the matched-sample p95 delay as the causal availability offset."
            ),
        },
    }
    return report


def write_artifacts(report: dict[str, Any], *, artifact_dir: Path) -> tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / "chainlink_public_delay_audit.json"
    md_path = artifact_dir / "chainlink_public_delay_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_table = _markdown_table(
        [{"metric": row["metric"], "value": row["value"]} for row in [
            {"metric": "onchain_update_events", "value": report["summary"]["onchain_update_events"]},
            {"metric": "matched_events", "value": report["summary"]["matched_events"]},
            {"metric": "unmatched_events", "value": report["summary"]["unmatched_events"]},
            {"metric": "match_coverage", "value": _format_pct(report["summary"]["match_coverage"])},
            {"metric": "mean_delay_seconds", "value": report["summary"]["mean_delay_seconds"]},
            {"metric": "median_delay_seconds", "value": report["summary"]["median_delay_seconds"]},
            {"metric": "p5_delay_seconds", "value": report["summary"]["p5_delay_seconds"]},
            {"metric": "p95_delay_seconds", "value": report["summary"]["p95_delay_seconds"]},
            {"metric": "p99_delay_seconds", "value": report["summary"]["p99_delay_seconds"]},
            {"metric": "min_delay_seconds", "value": report["summary"]["min_delay_seconds"]},
            {"metric": "max_delay_seconds", "value": report["summary"]["max_delay_seconds"]},
            {"metric": "negative_delay_matches", "value": report["summary"]["negative_delay_matches"]},
        ]]
    )

    md_lines = [
        "# Chainlink Public Delayed Page Delay Audit",
        "",
        "This audit uses only `finalized_clean` files from `chainlink_public_delayed` and `chainlink_onchain`, "
        "excludes the `2026-04-20 07:00 UTC` quarantine hour, and uses `recv_ts_ns` as the primary clock throughout.",
        "",
        "Exact-value matching is constrained to a generous plausibility window of "
        f"`[-{int(report['parameters']['negative_window_seconds'])}s, +{int(report['parameters']['positive_window_seconds'])}s]` "
        "to avoid falsely pairing unrelated same-price recurrences hours later.",
        "",
        "## Summary",
        "",
        summary_table,
        "",
        "## Delay Histogram (10s buckets)",
        "",
        _markdown_table(report["histogram_10s"]),
        "",
        "## Hour Of Day Breakdown",
        "",
        _markdown_table(report["breakdown_by_hour_of_day"]),
        "",
        "## Day Of Week Breakdown",
        "",
        _markdown_table(report["breakdown_by_day_of_week"]),
        "",
        "## Volatility Regime Breakdown",
        "",
        _markdown_table(report["breakdown_by_volatility_regime"]),
        "",
        "## Recommendation",
        "",
        f"- Status: `{report['safe_causal_offset_recommendation']['status']}`",
        f"- Recommended causal offset: `{_format_float(report['safe_causal_offset_recommendation']['recommended_offset_seconds'])}s`",
        f"- Reason: {report['safe_causal_offset_recommendation']['reason']}",
        "",
        "## Matched Pair Sample",
        "",
        _markdown_table(report["matched_pairs"][:10]),
    ]
    md_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact-delay matching between Chainlink public delayed page and onchain feed.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("docs"))
    parser.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--negative-window-seconds", type=float, default=NEGATIVE_MATCH_WINDOW_SECONDS)
    parser.add_argument("--positive-window-seconds", type=float, default=POSITIVE_MATCH_WINDOW_SECONDS)
    args = parser.parse_args()

    report = build_report(
        root=args.root,
        artifact_dir=args.artifact_dir,
        max_workers=args.max_workers,
        negative_window_seconds=args.negative_window_seconds,
        positive_window_seconds=args.positive_window_seconds,
    )
    json_path, md_path = write_artifacts(report, artifact_dir=args.artifact_dir)
    print(json.dumps({
        "json_path": str(json_path),
        "md_path": str(md_path),
        "summary": report["summary"],
        "safe_causal_offset_recommendation": report["safe_causal_offset_recommendation"],
    }, indent=2))


if __name__ == "__main__":
    main()
