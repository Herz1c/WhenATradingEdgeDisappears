from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

DATASETS = {
    "live_ref": Path("data/canonical/live_reference_events_v1"),
    "m_interval_preopen": Path("data/datasets/market_interval_dataset_v1_preopen"),
    "m_interval_first15s": Path("data/datasets/market_interval_dataset_v1_first15s"),
    "m_interval_first30s": Path("data/datasets/market_interval_dataset_v1_first30s"),
    "snap_coarse": Path("data/datasets/resolution_snapshot_dataset_v1_coarse"),
    "snap_dense": Path("data/datasets/resolution_snapshot_dataset_v1_dense_close"),
    "snap_event_triggered": Path("data/datasets/resolution_snapshot_dataset_v1_coarse_event_triggered"),
    "micro_tab": Path("data/datasets/microstructure_sequence_dataset_v1_tabular"),
    "micro_event64": Path("data/datasets/microstructure_sequence_dataset_v1_event64"),
    "micro_event128": Path("data/datasets/microstructure_sequence_dataset_v1_event128"),
    "exec_intent": Path("data/datasets/execution_intent_dataset_v1"),
}


def _date_range(date_from: date, date_to: date) -> list[date]:
    return [date_from + timedelta(days=i) for i in range((date_to - date_from).days + 1)]


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _parquet_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "eligible_rows": 0, "eligible_rate": 0.0, "sha256": ""}
    rows = 0
    eligible = 0
    try:
        frame = pl.read_parquet(path)
        rows = frame.height
        for column in [
            "training_feature_eligible",
            "sequence_feature_eligible",
            "execution_training_eligible",
            "price_valid",
        ]:
            if column in frame.columns:
                eligible = int(frame.select(pl.col(column).fill_null(False).cast(pl.Int64).sum()).item())
                break
        else:
            eligible = rows
    except Exception:
        rows = eligible = 0
    return {
        "exists": True,
        "rows": rows,
        "eligible_rows": eligible,
        "eligible_rate": float(eligible / rows) if rows else 0.0,
        "sha256": _sha256(path),
        "last_build_ts": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def generate_coverage_matrix(
    date_from: str,
    date_to: str,
    output_path: str = "docs/coverage_matrix.md",
    json_path: str | None = None,
) -> dict[str, Any]:
    """
    Generate a per-date dataset coverage matrix.
    """
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    rows: list[dict[str, Any]] = []
    for target_day in _date_range(start, end):
        row: dict[str, Any] = {"date": target_day.isoformat(), "datasets": {}}
        audit_status = "PASS"
        for key, folder in DATASETS.items():
            stats = _parquet_stats(folder / f"{target_day.isoformat()}.parquet")
            row["datasets"][key] = stats
            if not stats["exists"]:
                audit_status = "NOT_RUN"
        row["audit_status"] = audit_status
        rows.append(row)

    report = {"date_from": date_from, "date_to": date_to, "rows": rows}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Coverage Matrix",
        "",
        f"- Date range: `{date_from}` to `{date_to}`",
        "",
        "| date | live_ref | m_interval | snap_coarse | snap_dense | snap_event | micro_tab | micro_event64 | micro_event128 | exec_intent | audit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        datasets = row["datasets"]

        def cell(key: str) -> str:
            stats = datasets[key]
            return f"{'OK' if stats['exists'] else 'MISS'} {stats['rows']}"

        m_rows = sum(datasets[key]["rows"] for key in ["m_interval_preopen", "m_interval_first15s", "m_interval_first30s"])
        lines.append(
            f"| {row['date']} | {cell('live_ref')} | OK {m_rows} | {cell('snap_coarse')} | "
            f"{cell('snap_dense')} | {cell('snap_event_triggered')} | {cell('micro_tab')} | "
            f"{cell('micro_event64')} | {cell('micro_event128')} | {cell('exec_intent')} | {row['audit_status']} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if json_path is None:
        json_path = str(out.with_suffix(".json"))
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
