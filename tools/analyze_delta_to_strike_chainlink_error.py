"""Quantify the delta-to-strike error introduced by using a Chainlink proxy instead of the settlement feed."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from binance_recorder.compression import iter_zstd_jsonl  # noqa: E402
from market_recorders.unified_reader import UnifiedRawReader  # noqa: E402

SECOND_NS = 1_000_000_000
DEFAULT_DATASETS = (
    "resolution_snapshot_dataset_v1_coarse",
    "resolution_snapshot_dataset_v1_dense_close",
    "market_interval_dataset_v1_preopen",
    "market_interval_dataset_v1_first15s",
    "market_interval_dataset_v1_first30s",
)
FRESHNESS_WINDOWS_SECONDS = (60, 300, 600, 3_600)


@dataclass(slots=True, frozen=True)
class ChainlinkEvent:
    event_ts_ns: int
    recv_ts_ns: int
    price: float
    source: str
    source_id: str


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _iter_dates(date_from: date, date_to: date) -> list[date]:
    current = date_from
    values: list[date] = []
    while current <= date_to:
        values.append(current)
        current += timedelta(days=1)
    return values


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _valid_clean_files(
    *,
    root: Path,
    relative_prefix: Path,
    kind: str,
    date_from: date,
    date_to: date,
) -> list[Path]:
    reader = UnifiedRawReader(root=root)
    files: list[Path] = []
    for target_date in _iter_dates(date_from, date_to):
        for path in reader.list_files_for_date(
            relative_prefix=relative_prefix,
            target_date=target_date,
            kind=kind,
            finalized_only=False,
        ):
            metadata = reader.file_metadata(path)
            if metadata["finalized"] is True and metadata["quality_class"] == "finalized_clean":
                files.append(path)
    return files


def load_public_delayed_events(*, root: Path, date_from: date, date_to: date) -> tuple[list[ChainlinkEvent], dict[str, Any]]:
    files = _valid_clean_files(
        root=root,
        relative_prefix=Path("chainlink_public_delayed") / "public_stream_page" / "BTCUSD",
        kind="page",
        date_from=date_from,
        date_to=date_to,
    )
    raw_events: list[ChainlinkEvent] = []
    for path in files:
        for record in iter_zstd_jsonl(path):
            recv_ts_ns = _to_int(record.get("recv_ts_ns"))
            display_ts = _to_int(record.get("chainlink_display_ts"))
            price = _to_float(record.get("btc_usd_price"))
            if recv_ts_ns is None or display_ts is None or price is None:
                continue
            event_ts_ns = display_ts if abs(display_ts) >= 10_000_000_000 else display_ts * SECOND_NS
            raw_events.append(
                ChainlinkEvent(
                    event_ts_ns=int(event_ts_ns),
                    recv_ts_ns=recv_ts_ns,
                    price=price,
                    source="chainlink_public_delayed_display_ts",
                    source_id=f"{event_ts_ns}:{price}",
                )
            )
    return _dedupe_events(raw_events), {"files": len(files), "raw_rows": len(raw_events)}


def load_onchain_events(*, root: Path, date_from: date, date_to: date) -> tuple[list[ChainlinkEvent], dict[str, Any]]:
    files = _valid_clean_files(
        root=root,
        relative_prefix=Path("chainlink") / "onchain" / "BTCUSD",
        kind="onchain",
        date_from=date_from,
        date_to=date_to,
    )
    raw_events: list[ChainlinkEvent] = []
    for path in files:
        for record in iter_zstd_jsonl(path):
            round_data = record.get("round_data") if isinstance(record.get("round_data"), dict) else {}
            recv_ts_ns = _to_int(record.get("recv_ts_ns"))
            updated_at_s = _to_int(round_data.get("updated_at_s"))
            price = _to_float(round_data.get("answer"))
            round_id = str(round_data.get("round_id") or "")
            if recv_ts_ns is None or updated_at_s is None or price is None or not round_id:
                continue
            raw_events.append(
                ChainlinkEvent(
                    event_ts_ns=updated_at_s * SECOND_NS,
                    recv_ts_ns=recv_ts_ns,
                    price=price,
                    source="chainlink_onchain_updated_at",
                    source_id=round_id,
                )
            )
    return _dedupe_events(raw_events), {"files": len(files), "raw_rows": len(raw_events)}


def _dedupe_events(events: list[ChainlinkEvent]) -> list[ChainlinkEvent]:
    by_key: dict[tuple[int, str], ChainlinkEvent] = {}
    for event in sorted(events, key=lambda item: (item.event_ts_ns, item.recv_ts_ns, item.source_id)):
        by_key[(event.event_ts_ns, event.source_id)] = event
    by_ts: dict[int, ChainlinkEvent] = {}
    for event in sorted(by_key.values(), key=lambda item: (item.event_ts_ns, item.recv_ts_ns, item.source_id)):
        current = by_ts.get(event.event_ts_ns)
        if current is None or event.recv_ts_ns >= current.recv_ts_ns:
            by_ts[event.event_ts_ns] = event
    return [by_ts[key] for key in sorted(by_ts)]


def _percentile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, q))


def _metric_block(error: np.ndarray, age_s: np.ndarray | None = None) -> dict[str, Any]:
    finite = error[np.isfinite(error)]
    if finite.size == 0:
        return {
            "rows": 0,
            "mean_error_usd": None,
            "median_error_usd": None,
            "median_abs_error_usd": None,
            "p75_abs_error_usd": None,
            "p90_abs_error_usd": None,
            "p95_abs_error_usd": None,
            "p99_abs_error_usd": None,
            "max_abs_error_usd": None,
            "within_1_usd_pct": None,
            "within_2_5_usd_pct": None,
            "within_5_usd_pct": None,
            "within_10_usd_pct": None,
            "chainlink_age_median_s": None,
            "chainlink_age_p95_s": None,
        }
    abs_error = np.abs(finite)
    out = {
        "rows": int(finite.size),
        "mean_error_usd": float(finite.mean()),
        "median_error_usd": float(np.median(finite)),
        "median_abs_error_usd": float(np.median(abs_error)),
        "p75_abs_error_usd": float(np.quantile(abs_error, 0.75)),
        "p90_abs_error_usd": float(np.quantile(abs_error, 0.90)),
        "p95_abs_error_usd": float(np.quantile(abs_error, 0.95)),
        "p99_abs_error_usd": float(np.quantile(abs_error, 0.99)),
        "max_abs_error_usd": float(abs_error.max()),
        "within_1_usd_pct": float((abs_error <= 1.0).mean() * 100.0),
        "within_2_5_usd_pct": float((abs_error <= 2.5).mean() * 100.0),
        "within_5_usd_pct": float((abs_error <= 5.0).mean() * 100.0),
        "within_10_usd_pct": float((abs_error <= 10.0).mean() * 100.0),
    }
    if age_s is not None:
        finite_age = age_s[np.isfinite(error) & np.isfinite(age_s)]
        out["chainlink_age_median_s"] = float(np.median(finite_age)) if finite_age.size else None
        out["chainlink_age_p95_s"] = float(np.quantile(finite_age, 0.95)) if finite_age.size else None
    else:
        out["chainlink_age_median_s"] = None
        out["chainlink_age_p95_s"] = None
    return out


def _load_dataset_frame(root: Path, dataset: str, date_from: date, date_to: date) -> pd.DataFrame:
    directory = root / "datasets" / dataset
    frames: list[pd.DataFrame] = []
    for target_date in _iter_dates(date_from, date_to):
        path = directory / f"{target_date.isoformat()}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    columns = [
        "snapshot_ts_ns",
        "snapshot_ts_iso",
        "market_slug",
        "price_to_beat",
        "live_btc_usd",
        "delta_to_strike",
        "live_bias_active",
        "live_bias_neutralized",
        "training_feature_eligible",
        "complete_feature_matrix_eligible",
        "sample_family",
    ]
    frame = pd.concat(frames, axis=0, ignore_index=True)
    return frame[[column for column in columns if column in frame.columns]].copy()


def _compare_dataset(frame: pd.DataFrame, events: list[ChainlinkEvent]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    event_ts = np.array([event.event_ts_ns for event in events], dtype=np.int64)
    event_prices = np.array([event.price for event in events], dtype=float)
    snapshot_ts = frame["snapshot_ts_ns"].to_numpy(dtype=np.int64)
    indices = np.searchsorted(event_ts, snapshot_ts, side="right") - 1
    matched = indices >= 0
    chainlink_price = np.full(len(frame), np.nan, dtype=float)
    chainlink_age_s = np.full(len(frame), np.nan, dtype=float)
    chainlink_price[matched] = event_prices[indices[matched]]
    chainlink_age_s[matched] = (snapshot_ts[matched] - event_ts[indices[matched]]) / SECOND_NS
    live_price = frame["live_btc_usd"].to_numpy(dtype=float)
    price_to_beat = frame["price_to_beat"].to_numpy(dtype=float)
    stored_delta = frame["delta_to_strike"].to_numpy(dtype=float)
    historical_delta = chainlink_price - price_to_beat
    error = stored_delta - historical_delta
    price_error = live_price - chainlink_price
    if not np.allclose(error[np.isfinite(error)], price_error[np.isfinite(price_error)], equal_nan=True):
        raise AssertionError("Delta error no longer equals live price error; check price_to_beat alignment.")
    return frame, error, chainlink_age_s


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered: list[str] = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                rendered.append(_format_float(value))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def build_report(
    *,
    root: Path,
    date_from: date,
    date_to: date,
    report_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    load_from = date_from - timedelta(days=1)
    public_events, public_meta = load_public_delayed_events(root=root, date_from=load_from, date_to=date_to)
    onchain_events, onchain_meta = load_onchain_events(root=root, date_from=load_from, date_to=date_to)
    sources = {
        "public_delayed_display_ts": (public_events, public_meta),
        "onchain_updated_at": (onchain_events, onchain_meta),
    }
    payload: dict[str, Any] = {
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "source_meta": {
            name: {
                **meta,
                "deduped_events": len(events),
                "first_event_ts": datetime.fromtimestamp(events[0].event_ts_ns / SECOND_NS, tz=UTC).isoformat() if events else None,
                "last_event_ts": datetime.fromtimestamp(events[-1].event_ts_ns / SECOND_NS, tz=UTC).isoformat() if events else None,
            }
            for name, (events, meta) in sources.items()
        },
        "dataset_metrics": [],
        "freshness_metrics": [],
        "date_metrics": [],
        "date_freshness_metrics": [],
        "bias_state_metrics": [],
        "bias_state_freshness_metrics": [],
        "lag_sweep_metrics": [],
        "chainlink_age_bucket_metrics": [],
        "recent_move_bucket_metrics": [],
    }
    live_reference = pd.concat(
        [pd.read_parquet(path) for path in sorted((root / "canonical" / "live_reference_events_v1").glob("*.parquet"))],
        axis=0,
        ignore_index=True,
    ).sort_values("ts_seconds", kind="stable")
    live_reference_ts = live_reference["ts_seconds"].to_numpy(dtype=np.int64)
    live_reference_price = live_reference["synthetic_corrected"].to_numpy(dtype=float)

    for dataset in DEFAULT_DATASETS:
        frame = _load_dataset_frame(root, dataset, date_from, date_to)
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["snapshot_ts_ns"], unit="ns", utc=True).dt.strftime("%Y-%m-%d")
        for source_name, (events, _meta) in sources.items():
            if not events:
                continue
            compared, error, age_s = _compare_dataset(frame, events)
            finite = np.isfinite(error)
            metrics = _metric_block(error, age_s)
            metrics.update(
                {
                    "dataset": dataset,
                    "source": source_name,
                    "total_rows": int(len(compared)),
                    "matched_rows": int(finite.sum()),
                    "matched_pct": float(finite.mean() * 100.0) if len(compared) else 0.0,
                    "training_rows": int(compared["training_feature_eligible"].astype(bool).sum()) if "training_feature_eligible" in compared else None,
                    "complete_matrix_rows": int(compared["complete_feature_matrix_eligible"].astype(bool).sum()) if "complete_feature_matrix_eligible" in compared else None,
                }
            )
            payload["dataset_metrics"].append(metrics)

            for max_age_s in FRESHNESS_WINDOWS_SECONDS:
                fresh_mask = np.isfinite(error) & np.isfinite(age_s) & (age_s <= max_age_s)
                fresh_metrics = _metric_block(error[fresh_mask], age_s[fresh_mask])
                fresh_metrics.update(
                    {
                        "dataset": dataset,
                        "source": source_name,
                        "max_chainlink_age_s": max_age_s,
                        "total_rows": int(len(compared)),
                        "matched_rows": int(fresh_mask.sum()),
                        "matched_pct": float(fresh_mask.mean() * 100.0) if len(compared) else 0.0,
                    }
                )
                payload["freshness_metrics"].append(fresh_metrics)

            if dataset == "resolution_snapshot_dataset_v1_dense_close":
                for date_value, group in compared.groupby("date", sort=True):
                    group_error = error[group.index.to_numpy()]
                    group_age = age_s[group.index.to_numpy()]
                    row = _metric_block(group_error, group_age)
                    row.update({"date": str(date_value), "dataset": dataset, "source": source_name})
                    payload["date_metrics"].append(row)
                    if source_name == "public_delayed_display_ts":
                        fresh_mask = np.isfinite(group_error) & np.isfinite(group_age) & (group_age <= 300)
                        fresh_row = _metric_block(group_error[fresh_mask], group_age[fresh_mask])
                        fresh_row.update(
                            {
                                "date": str(date_value),
                                "dataset": dataset,
                                "source": source_name,
                                "max_chainlink_age_s": 300,
                                "matched_pct": float(fresh_mask.mean() * 100.0) if len(group) else 0.0,
                            }
                        )
                        payload["date_freshness_metrics"].append(fresh_row)
            if dataset == "resolution_snapshot_dataset_v1_dense_close" and source_name == "public_delayed_display_ts":
                fresh_public_mask = np.isfinite(error) & np.isfinite(age_s) & (age_s <= 300)
                for lower_s, upper_s in ((0, 2), (2, 5), (5, 10), (10, 30), (30, 60), (60, 120), (120, 300)):
                    mask = fresh_public_mask & (age_s >= lower_s) & (age_s < upper_s)
                    row = _metric_block(error[mask], age_s[mask])
                    row.update(
                        {
                            "age_bucket": f"{lower_s}-{upper_s}s",
                            "lower_s": lower_s,
                            "upper_s": upper_s,
                            "matched_pct": float(mask.mean() * 100.0) if len(compared) else 0.0,
                        }
                    )
                    payload["chainlink_age_bucket_metrics"].append(row)

                chainlink_price = compared["live_btc_usd"].to_numpy(dtype=float) - error
                snapshot_seconds = (compared["snapshot_ts_ns"].to_numpy(dtype=np.int64) // SECOND_NS).astype(np.int64)
                for lag_s in (0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60):
                    lagged_seconds = snapshot_seconds - lag_s
                    lagged_indices = np.searchsorted(live_reference_ts, lagged_seconds, side="right") - 1
                    mask = fresh_public_mask & (lagged_indices >= 0)
                    lagged_price = np.full(len(compared), np.nan, dtype=float)
                    lagged_price[mask] = live_reference_price[lagged_indices[mask]]
                    lagged_error = lagged_price[mask] - chainlink_price[mask]
                    row = _metric_block(lagged_error, age_s[mask])
                    row.update({"synthetic_lag_s": lag_s, "matched_pct": float(mask.mean() * 100.0) if len(compared) else 0.0})
                    payload["lag_sweep_metrics"].append(row)

                current_indices = np.searchsorted(live_reference_ts, snapshot_seconds, side="right") - 1
                previous_indices = np.searchsorted(live_reference_ts, snapshot_seconds - 10, side="right") - 1
                movement_mask = fresh_public_mask & (current_indices >= 0) & (previous_indices >= 0)
                recent_move_abs = np.full(len(compared), np.nan, dtype=float)
                recent_move_abs[movement_mask] = np.abs(
                    live_reference_price[current_indices[movement_mask]]
                    - live_reference_price[previous_indices[movement_mask]]
                )
                for lower_usd, upper_usd in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, math.inf)):
                    mask = movement_mask & (recent_move_abs >= lower_usd) & (recent_move_abs < upper_usd)
                    row = _metric_block(error[mask], age_s[mask])
                    row.update(
                        {
                            "move_bucket_usd": f"{lower_usd}-{upper_usd if math.isfinite(upper_usd) else 'inf'}",
                            "lower_usd": lower_usd,
                            "upper_usd": upper_usd if math.isfinite(upper_usd) else None,
                            "matched_pct": float(mask.mean() * 100.0) if len(compared) else 0.0,
                        }
                    )
                    payload["recent_move_bucket_metrics"].append(row)

            if dataset == "resolution_snapshot_dataset_v1_dense_close" and source_name == "public_delayed_display_ts":
                for state_name, mask in {
                    "bias_active": compared["live_bias_active"].astype(bool).to_numpy() if "live_bias_active" in compared else np.zeros(len(compared), dtype=bool),
                    "bias_neutralized": compared["live_bias_neutralized"].astype(bool).to_numpy() if "live_bias_neutralized" in compared else np.zeros(len(compared), dtype=bool),
                    "training_feature_eligible": compared["training_feature_eligible"].astype(bool).to_numpy() if "training_feature_eligible" in compared else np.zeros(len(compared), dtype=bool),
                    "complete_feature_matrix_eligible": compared["complete_feature_matrix_eligible"].astype(bool).to_numpy() if "complete_feature_matrix_eligible" in compared else np.zeros(len(compared), dtype=bool),
                }.items():
                    row = _metric_block(error[mask], age_s[mask])
                    row.update({"state": state_name, "dataset": dataset, "source": source_name})
                    payload["bias_state_metrics"].append(row)
                    fresh_mask = mask & np.isfinite(error) & np.isfinite(age_s) & (age_s <= 300)
                    fresh_row = _metric_block(error[fresh_mask], age_s[fresh_mask])
                    fresh_row.update(
                        {
                            "state": state_name,
                            "dataset": dataset,
                            "source": source_name,
                            "max_chainlink_age_s": 300,
                            "matched_pct": float(fresh_mask.mean() * 100.0) if len(compared) else 0.0,
                        }
                    )
                    payload["bias_state_freshness_metrics"].append(fresh_row)

    report_lines = [
        "# delta_to_strike vs Historical Chainlink",
        "",
        "## Scope",
        "",
        f"- Date range: `{date_from.isoformat()}` to `{date_to.isoformat()}`",
        "- Compared value: stored Phase3 `delta_to_strike` versus hypothetical `historical_chainlink_price - price_to_beat`.",
        "- Since `price_to_beat` is identical on both sides, the measured error is exactly `live_btc_usd - historical_chainlink_price`.",
        "- Primary historical source: `chainlink_public_delayed` using its displayed Chainlink event timestamp, joined by as-of `chainlink_ts <= snapshot_ts`.",
        "- Secondary source: `chainlink_onchain` using `round_data.updated_at_s`; included as a sanity check because it is a different / sparse verification feed.",
        "",
        "## Source Coverage",
        "",
    ]
    report_lines.extend(
        _markdown_table(
            [
                {"source": name, **meta}
                for name, meta in payload["source_meta"].items()
            ],
            ["source", "files", "raw_rows", "deduped_events", "first_event_ts", "last_event_ts"],
        )
    )
    report_lines.extend(["", "## Dataset Metrics", ""])
    dataset_rows = [
        {
            "dataset": row["dataset"],
            "source": row["source"],
            "matched_pct": row["matched_pct"],
            "medAE": row["median_abs_error_usd"],
            "p75": row["p75_abs_error_usd"],
            "p90": row["p90_abs_error_usd"],
            "p95": row["p95_abs_error_usd"],
            "p99": row["p99_abs_error_usd"],
            "mean_signed": row["mean_error_usd"],
            "median_signed": row["median_error_usd"],
            "within_5_pct": row["within_5_usd_pct"],
            "within_10_pct": row["within_10_usd_pct"],
            "age_p95_s": row["chainlink_age_p95_s"],
        }
        for row in payload["dataset_metrics"]
    ]
    report_lines.extend(
        _markdown_table(
            dataset_rows,
            [
                "dataset",
                "source",
                "matched_pct",
                "medAE",
                "p75",
                "p90",
                "p95",
                "p99",
                "mean_signed",
                "median_signed",
                "within_5_pct",
                "within_10_pct",
                "age_p95_s",
            ],
        )
    )
    report_lines.extend(["", "## Dataset Metrics With Fresh Historical Chainlink", ""])
    freshness_rows = [
        {
            "dataset": row["dataset"],
            "source": row["source"],
            "max_age_s": row["max_chainlink_age_s"],
            "matched_pct": row["matched_pct"],
            "medAE": row["median_abs_error_usd"],
            "p75": row["p75_abs_error_usd"],
            "p90": row["p90_abs_error_usd"],
            "p95": row["p95_abs_error_usd"],
            "p99": row["p99_abs_error_usd"],
            "mean_signed": row["mean_error_usd"],
            "median_signed": row["median_error_usd"],
            "within_5_pct": row["within_5_usd_pct"],
            "within_10_pct": row["within_10_usd_pct"],
        }
        for row in payload["freshness_metrics"]
        if row["max_chainlink_age_s"] in {300, 3600}
    ]
    report_lines.extend(
        _markdown_table(
            freshness_rows,
            [
                "dataset",
                "source",
                "max_age_s",
                "matched_pct",
                "medAE",
                "p75",
                "p90",
                "p95",
                "p99",
                "mean_signed",
                "median_signed",
                "within_5_pct",
                "within_10_pct",
            ],
        )
    )
    report_lines.extend(["", "## Dense Close By Date vs Public Delayed", ""])
    report_lines.extend(
        _markdown_table(
            payload["date_metrics"],
            ["date", "rows", "median_abs_error_usd", "p90_abs_error_usd", "p95_abs_error_usd", "p99_abs_error_usd", "mean_error_usd", "within_5_usd_pct", "within_10_usd_pct", "chainlink_age_p95_s"],
        )
    )
    report_lines.extend(["", "## Dense Close By Date vs Onchain", ""])
    report_lines.extend(
        _markdown_table(
            [row for row in payload["date_metrics"] if row["source"] == "onchain_updated_at"],
            ["date", "rows", "median_abs_error_usd", "p90_abs_error_usd", "p95_abs_error_usd", "p99_abs_error_usd", "mean_error_usd", "within_5_usd_pct", "within_10_usd_pct", "chainlink_age_p95_s"],
        )
    )
    report_lines.extend(["", "## Dense Close By Date vs Fresh Public Delayed (age <= 300s)", ""])
    report_lines.extend(
        _markdown_table(
            payload["date_freshness_metrics"],
            ["date", "matched_pct", "rows", "median_abs_error_usd", "p90_abs_error_usd", "p95_abs_error_usd", "p99_abs_error_usd", "mean_error_usd", "within_5_usd_pct", "within_10_usd_pct", "chainlink_age_p95_s"],
        )
    )
    report_lines.extend(["", "## Dense Close Bias/Eligibility Slices vs Public Delayed", ""])
    report_lines.extend(
        _markdown_table(
            payload["bias_state_metrics"],
            ["state", "rows", "median_abs_error_usd", "p90_abs_error_usd", "p95_abs_error_usd", "p99_abs_error_usd", "mean_error_usd", "within_5_usd_pct", "within_10_usd_pct", "chainlink_age_p95_s"],
        )
    )
    report_lines.extend(["", "## Dense Close Bias/Eligibility Slices vs Fresh Public Delayed (age <= 300s)", ""])
    report_lines.extend(
        _markdown_table(
            payload["bias_state_freshness_metrics"],
            ["state", "matched_pct", "rows", "median_abs_error_usd", "p90_abs_error_usd", "p95_abs_error_usd", "p99_abs_error_usd", "mean_error_usd", "within_5_usd_pct", "within_10_usd_pct", "chainlink_age_p95_s"],
        )
    )
    report_lines.extend(["", "## Dense Close Error By Chainlink Event Age (Fresh Public Delayed)", ""])
    report_lines.extend(
        _markdown_table(
            payload["chainlink_age_bucket_metrics"],
            ["age_bucket", "rows", "median_abs_error_usd", "p90_abs_error_usd", "p95_abs_error_usd", "p99_abs_error_usd", "mean_error_usd", "within_5_usd_pct", "within_10_usd_pct"],
        )
    )
    report_lines.extend(["", "## Dense Close Synthetic Lag Sweep vs Fresh Public Delayed", ""])
    report_lines.extend(
        _markdown_table(
            payload["lag_sweep_metrics"],
            ["synthetic_lag_s", "rows", "median_abs_error_usd", "p90_abs_error_usd", "p95_abs_error_usd", "p99_abs_error_usd", "mean_error_usd", "within_5_usd_pct", "within_10_usd_pct"],
        )
    )
    report_lines.extend(["", "## Dense Close Error By 10s Synthetic Move", ""])
    report_lines.extend(
        _markdown_table(
            payload["recent_move_bucket_metrics"],
            ["move_bucket_usd", "rows", "median_abs_error_usd", "p90_abs_error_usd", "p95_abs_error_usd", "p99_abs_error_usd", "mean_error_usd", "within_5_usd_pct", "within_10_usd_pct"],
        )
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--date-from", type=_parse_date, default=date(2026, 4, 19))
    parser.add_argument("--date-to", type=_parse_date, default=date(2026, 4, 23))
    parser.add_argument("--report-path", type=Path, default=Path("docs/delta_to_strike_chainlink_error_analysis.md"))
    parser.add_argument("--json-path", type=Path, default=Path("docs/delta_to_strike_chainlink_error_analysis.json"))
    args = parser.parse_args()
    payload = build_report(
        root=args.root,
        date_from=args.date_from,
        date_to=args.date_to,
        report_path=args.report_path,
        json_path=args.json_path,
    )
    primary = next(
        row
        for row in payload["freshness_metrics"]
        if row["dataset"] == "resolution_snapshot_dataset_v1_dense_close"
        and row["source"] == "public_delayed_display_ts"
        and row["max_chainlink_age_s"] == 300
    )
    print(
        json.dumps(
            {
                "report_path": str(args.report_path),
                "json_path": str(args.json_path),
                "dense_close_fresh_public_delayed_max_age_s": primary["max_chainlink_age_s"],
                "dense_close_fresh_public_delayed_matched_pct": primary["matched_pct"],
                "dense_close_fresh_public_delayed_medAE": primary["median_abs_error_usd"],
                "dense_close_fresh_public_delayed_p90": primary["p90_abs_error_usd"],
                "dense_close_fresh_public_delayed_p95": primary["p95_abs_error_usd"],
                "dense_close_fresh_public_delayed_p99": primary["p99_abs_error_usd"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
