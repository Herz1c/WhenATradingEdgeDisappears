from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl


def _date_range(date_from: date, date_to: date) -> list[date]:
    return [date_from + timedelta(days=i) for i in range((date_to - date_from).days + 1)]


def _rate(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    return float(frame.select(pl.col(column).fill_null(False).cast(pl.Int64).mean()).item() or 0.0)


def _severity(value: float | None) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    if value > 0.03:
        return "CRITICAL"
    if value > 0.01:
        return "WARN"
    return "OK"


def compute_book_quality_report(
    date_from: str,
    date_to: str,
    output_path: str,
    dataset_root: str | Path = "data/datasets",
) -> dict[str, Any]:
    """
    Daily report on crossed_book, stale_book, and one_sided_book rates.
    """
    root = Path(dataset_root)
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    rows: list[dict[str, Any]] = []
    for target_day in _date_range(start, end):
        day = target_day.isoformat()
        frames = []
        for dataset_name in ["resolution_snapshot_dataset_v1_coarse", "microstructure_sequence_dataset_v1_tabular"]:
            path = root / dataset_name / f"{day}.parquet"
            if path.exists():
                frames.append(pl.read_parquet(path))
        frame = pl.concat(frames, how="diagonal") if frames else pl.DataFrame()
        row = {
            "date": day,
            "crossed_book_rate": _rate(frame, "crossed_book"),
            "stale_book_rate": _rate(frame, "stale_book"),
            "one_sided_book_rate": _rate(frame, "one_sided_book"),
        }
        row["severity"] = max(
            [_severity(row["crossed_book_rate"]), _severity(row["stale_book_rate"]), _severity(row["one_sided_book_rate"])],
            key=lambda s: {"OK": 0, "NOT_AVAILABLE": 0, "WARN": 1, "CRITICAL": 2}[s],
        )
        rows.append(row)
    lines = [
        "# Book Quality Monitor",
        "",
        f"- Date range: `{date_from}` to `{date_to}`",
        "- Thresholds: WARN > 1%, CRITICAL > 3%",
        "",
        "| date | crossed_book | stale_book | one_sided_book | severity |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.4%}"

        lines.append(
            f"| {row['date']} | {pct(row['crossed_book_rate'])} | {pct(row['stale_book_rate'])} | "
            f"{pct(row['one_sided_book_rate'])} | {row['severity']} |"
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"date_from": date_from, "date_to": date_to, "rows": rows}

