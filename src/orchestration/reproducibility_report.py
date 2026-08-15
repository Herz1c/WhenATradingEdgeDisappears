from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path
from typing import Any

DATASET_DIRS = [
    Path("data/canonical/live_reference_events_v1"),
    Path("data/datasets/market_interval_dataset_v1_preopen"),
    Path("data/datasets/market_interval_dataset_v1_first15s"),
    Path("data/datasets/market_interval_dataset_v1_first30s"),
    Path("data/datasets/resolution_snapshot_dataset_v1_coarse"),
    Path("data/datasets/resolution_snapshot_dataset_v1_dense_close"),
    Path("data/datasets/microstructure_sequence_dataset_v1_tabular"),
    Path("data/datasets/execution_intent_dataset_v1"),
]


def _date_range(date_from: date, date_to: date) -> list[date]:
    return [date_from + timedelta(days=i) for i in range((date_to - date_from).days + 1)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def generate_reproducibility_report(
    date_from: str,
    date_to: str,
    output_path: str = "docs/dataset_factory_reproducibility_report.md",
    hash_root: str = "data/reproducibility_hashes",
) -> dict[str, Any]:
    """
    Hash dataset shards and compare to stored hash files.
    """
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    hash_dir = Path(hash_root)
    hash_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target_day in _date_range(start, end):
        for dataset_dir in DATASET_DIRS:
            dataset_name = dataset_dir.name
            path = dataset_dir / f"{target_day.isoformat()}.parquet"
            hash_path = hash_dir / f"{target_day.isoformat()}_{dataset_name}.sha256"
            if not path.exists():
                rows.append({"date": target_day.isoformat(), "dataset": dataset_name, "status": "MISSING", "sha256": "", "path": str(path)})
                continue
            digest = _sha256(path)
            previous = hash_path.read_text(encoding="utf-8").strip() if hash_path.exists() else ""
            status = "NEW" if not previous else "IDENTICAL" if previous == digest else "CHANGED"
            hash_path.write_text(digest + "\n", encoding="utf-8")
            rows.append({"date": target_day.isoformat(), "dataset": dataset_name, "status": status, "sha256": digest, "path": str(path)})
    verdict = "INCOMPLETE" if any(row["status"] == "MISSING" for row in rows) else "CHANGED" if any(row["status"] == "CHANGED" for row in rows) else "REPRODUCIBLE"
    lines = [
        "# Dataset Factory Reproducibility Report",
        "",
        f"- Date range: `{date_from}` to `{date_to}`",
        f"- Verdict: `{verdict}`",
        "",
        "| date | dataset | status | sha256 | path |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['date']} | {row['dataset']} | {row['status']} | `{row['sha256']}` | `{row['path']}` |")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"date_from": date_from, "date_to": date_to, "verdict": verdict, "rows": rows}

