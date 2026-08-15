from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _date_range(date_from: date, date_to: date) -> list[date]:
    return [date_from + timedelta(days=i) for i in range((date_to - date_from).days + 1)]


def _source_path(root: Path, source: str, day: date) -> Path:
    if source == "live_reference_events_v1":
        return root / "canonical" / "live_reference_events_v1" / f"{day.isoformat()}.parquet"
    if source == "polymarket_l2_events_v1":
        return root / "canonical" / "polymarket_l2_events_v1" / f"{day.isoformat()}.parquet"
    if source == "external_btc_events_v1":
        return root / "canonical" / "external_btc_events_v1" / f"{day.isoformat()}.parquet"
    raise ValueError(f"Unknown source: {source}")


def _source_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rows": 0, "max_gap_seconds": None, "stale_fraction_gt_30s": None}

    parquet_file = pq.ParquetFile(path)
    rows = int(parquet_file.metadata.num_rows)
    columns = set(parquet_file.schema_arrow.names)
    ts_col = "recv_ts_ns" if "recv_ts_ns" in columns else "ts_seconds" if "ts_seconds" in columns else None
    if rows < 2 or ts_col is None:
        return {"rows": rows, "max_gap_seconds": None, "stale_fraction_gt_30s": None}

    table = pq.read_table(path, columns=[ts_col])
    values = table[ts_col].combine_chunks().to_numpy(zero_copy_only=False)
    values = np.asarray(values)
    if values.dtype.kind in {"f", "c"}:
        values = values[np.isfinite(values)]
    elif values.dtype.kind in {"O", "U", "S"}:
        values = values[values != None]  # noqa: E711
    if values.size < 2:
        return {"rows": rows, "max_gap_seconds": None, "stale_fraction_gt_30s": None}

    if ts_col == "ts_seconds":
        values = (values.astype("float64") * 1_000_000_000).astype("int64")
    else:
        values = values.astype("int64", copy=True)
    values.sort()
    max_gap_ns = int(np.diff(values).max())
    max_gap_seconds = float(max_gap_ns) / 1_000_000_000
    return {
        "rows": rows,
        "max_gap_seconds": max_gap_seconds,
        "stale_fraction_gt_30s": float(max_gap_seconds > 30.0),
    }


def generate_time_alignment_report(
    date_from: str,
    date_to: str,
    output_path: str,
    root: str | Path = "data",
) -> dict[str, Any]:
    """
    Generate time_alignment_report_v1.json for a date range.
    """
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    root_path = Path(root)
    sources = ["polymarket_l2_events_v1", "external_btc_events_v1", "live_reference_events_v1"]
    daily: list[dict[str, Any]] = []
    for day in _date_range(start, end):
        source_stats: dict[str, Any] = {}
        for source in sources:
            source_stats[source] = _source_stats(_source_path(root_path, source, day))
        daily.append({"date": day.isoformat(), "sources": source_stats, "pairwise": []})
    report = {"report_version": "time_alignment_report_v1", "date_from": date_from, "date_to": date_to, "daily": daily}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
