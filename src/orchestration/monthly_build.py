from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


MONTHLY_BUILD_REQUIRED_STEP_COMMANDS = [
    "build-registries",
    "build-live-reference-events",
    "build-phase3-datasets",
    "build-event-triggered-snapshots",
    "build-phase8-datasets",
    "build-external-btc-events",
    "build-polymarket-ws-events",
    "build-execution-intent-dataset-v1",
    "generate-coverage-matrix",
    "audit-all-datasets",
    "generate-reproducibility-report",
]


@dataclass(slots=True)
class StepResult:
    id: str
    command: str
    required: bool
    status: str
    started_utc: str
    ended_utc: str
    runtime_seconds: float
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _command_for(step_command: str, date_from: str, date_to: str, max_workers: int, dry_run: bool) -> list[str]:
    base = [sys.executable, "-m", "polymarket_recorder.cli", step_command, "--date-from", date_from, "--date-to", date_to]
    if step_command in {"build-phase3-datasets", "build-phase8-datasets", "build-execution-intent-dataset-v1"}:
        base.extend(["--max-workers", str(max_workers)])
    if dry_run:
        base.append("--dry-run")
    return base


def run_monthly_build(
    *,
    date_from: str,
    date_to: str,
    config_path: str = "config/orchestration_config.yaml",
    log_path: str = "logs/orchestration_log.jsonl",
    max_workers: int | None = None,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    """
    Monthly build orchestrator.
    """
    config = _load_config(Path(config_path)).get("orchestration_v1", {})
    workers = int(max_workers if max_workers is not None else config.get("default_max_workers", 1))
    dry = bool(dry_run if dry_run is not None else config.get("dry_run_mode", False))
    fail_fast = bool(config.get("fail_fast", True))
    results: list[StepResult] = []
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    for step in config.get("steps", []):
        started = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        command = _command_for(str(step["command"]), date_from, date_to, workers, dry)
        proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ended = datetime.now(timezone.utc).isoformat()
        result = StepResult(
            id=str(step["id"]),
            command=" ".join(command),
            required=bool(step.get("required", True)),
            status="PASS" if proc.returncode == 0 else "FAIL",
            started_utc=started,
            ended_utc=ended,
            runtime_seconds=round(time.perf_counter() - start, 6),
            returncode=int(proc.returncode),
            stdout_tail=proc.stdout[-2000:],
            stderr_tail=proc.stderr[-2000:],
        )
        results.append(result)
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
        if proc.returncode != 0 and (result.required or fail_fast):
            break
    verdict = "PASS" if all(result.status == "PASS" or not result.required for result in results) else "FAIL"
    return {"date_from": date_from, "date_to": date_to, "dry_run": dry, "verdict": verdict, "steps": [asdict(r) for r in results]}
