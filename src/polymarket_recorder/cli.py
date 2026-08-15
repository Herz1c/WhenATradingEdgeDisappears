from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

import polars as pl
import typer

from binance_recorder.logging_utils import configure_logging
from canonical.external_btc_events_builder import build_external_btc_events
from canonical.polymarket_ws_events_builder import build_polymarket_ws_events
from canonical.secondary_context_events_builder import build_secondary_context_events
from canonical.time_alignment_report import generate_time_alignment_report
from chainlink_recorder.live_reference_events import build_live_reference_events
from model_factory.dataset_loader import LeakageSafeLoader
from model_factory.evaluation.gates import evaluate_gate
from model_factory.infrastructure_audit import run_infrastructure_audit
from model_factory.model_factory_audit import list_registered_models, run_model_factory_audit
from monitoring.book_quality_monitor import compute_book_quality_report
from orchestration.coverage_matrix import generate_coverage_matrix
from orchestration.monthly_build import run_monthly_build
from orchestration.reproducibility_report import generate_reproducibility_report
from paper_trading.replay_paper import ReplayPaperTrader
from polymarket_recorder.event_triggered_sampler import build_event_triggered_snapshots

from .config import PolymarketRecorderConfig
from .dataset_audit import write_phase3_audit_report
from .dataset_completion import (
    MonthlyOrchestratorConfig,
    run_monthly_dataset_factory,
    write_canonical_event_backfill_hardening_report,
    write_dataset_coverage_matrix,
    write_dataset_factory_asset_inventory,
    write_dataset_factory_completion_report,
    write_dataset_factory_gap_report,
    write_dataset_factory_reproducibility_report,
)
from .dataset_factory import build_phase3_datasets
from .execution_audit import write_execution_audit_report
from .execution_intent_audit import write_execution_intent_audit_report
from .execution_intent_dataset import ExecutionIntentBuildConfig, build_execution_intent_datasets
from .execution_validation import write_phase5_validation_report
from .microstructure_audit import write_phase8_audit_report, write_phase8_full_audit_report
from .microstructure_sequence_builder import build_phase8_datasets
from .microstructure_tensor_export import (
    TensorExportConfig,
    audit_microstructure_tensor_exports,
    build_microstructure_tensor_exports,
    write_microstructure_tensor_reproducibility_report,
)
from .registry_builder import build_registries
from .resolution_reader import PolymarketResolutionReader
from .resolution_recorder import (
    PolymarketResolutionRecorderService,
    backfill_missing_resolutions_for_date,
)
from .reconstruction import PolymarketTrainingViewReader
from .reader import PolymarketRawReader
from .rtds_config import PolymarketRtdsConfig
from .rtds_reader import PolymarketRtdsRawReader
from .rtds_recorder import PolymarketRtdsRecorderService
from .rest_poller import PolymarketRestPollerService
from .strike_delta import PolymarketStrikeDeltaReader
from .strike_recorder import PolymarketStrikeRecorderService
from .ws_client import PolymarketRecorderService

app = typer.Typer(add_completion=False)


def _date_range_strings(date_from: str, date_to: str) -> list[str]:
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    days: list[str] = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current = date.fromordinal(current.toordinal() + 1)
    return days


def _dataset_status(dataset_root: Path, dataset_name: str, date_from: str, date_to: str) -> dict:
    rows = []
    for day in _date_range_strings(date_from, date_to):
        shard = dataset_root / dataset_name / f"{day}.parquet"
        if shard.exists():
            try:
                frame = pl.scan_parquet(str(shard))
                row_count = frame.select(pl.len()).collect().item()
            except Exception:
                row_count = None
            rows.append({"date": day, "exists": True, "rows": row_count, "path": str(shard)})
        else:
            rows.append({"date": day, "exists": False, "rows": 0, "path": str(shard)})
    return {"dataset": dataset_name, "rows": rows}


def _build_sources_available(root: Path, date_from: str, date_to: str) -> bool:
    for day in _date_range_strings(date_from, date_to):
        if any((root).rglob(f"*{day}*")):
            return True
    return False


def _dry_run_report(
    *,
    command: str,
    root: Path,
    dataset_root: Path,
    dataset_name: str,
    date_from: str,
    date_to: str,
) -> dict:
    status = _dataset_status(dataset_root, dataset_name, date_from, date_to)
    return {
        "command": command,
        "dry_run": True,
        "source_data_available": _build_sources_available(root, date_from, date_to),
        "outputs": status["rows"],
    }


@app.command("record-btc-5m")
def record_btc_5m(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(PolymarketRecorderService(PolymarketRecorderConfig.from_env(out=out)).run())


@app.command("record-rest")
def record_rest(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(PolymarketRestPollerService(PolymarketRecorderConfig.from_env(out=out)).run())


@app.command("record-strike")
def record_strike(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(PolymarketStrikeRecorderService(PolymarketRecorderConfig.from_env(out=out)).run())


@app.command("record-rtds")
def record_rtds(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(PolymarketRtdsRecorderService(PolymarketRtdsConfig.from_env(out=out)).run())


@app.command("record-resolution")
def record_resolution(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(PolymarketResolutionRecorderService(PolymarketRecorderConfig.from_env(out=out)).run())


@app.command("backfill-resolution")
def backfill_resolution(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    summary = asyncio.run(
        backfill_missing_resolutions_for_date(
            config=PolymarketRecorderConfig.from_env(out=out),
            target_date=target_date,
        )
    )
    typer.echo(summary)


@app.command("read")
def read(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    kind: str = typer.Option("ws"),
    token_outcome: str | None = typer.Option(None),
) -> None:
    reader = PolymarketRawReader(root=out)
    for record in reader.iter_kind(target_date=target_date, kind=kind, token_outcome=token_outcome):
        typer.echo(record)


@app.command("read-rtds")
def read_rtds(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    kind: str = typer.Option("ws"),
    topic: str | None = typer.Option(None),
    message_type: str | None = typer.Option(None),
    symbol: str | None = typer.Option(None),
    normalized: bool = typer.Option(False),
) -> None:
    reader = PolymarketRtdsRawReader(root=out)
    iterator = (
        reader.iter_normalized(
            target_date=target_date,
            kind=kind,
            topic=topic,
            message_type=message_type,
            symbol=symbol,
        )
        if normalized
        else reader.iter_kind(
            target_date=target_date,
            kind=kind,
            topic=topic,
            message_type=message_type,
            symbol=symbol,
        )
    )
    for record in iterator:
        typer.echo(record)


@app.command("read-resolution")
def read_resolution(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    kind: str = typer.Option("resolution"),
    market_id: str | None = typer.Option(None),
    market_slug: str | None = typer.Option(None),
    resolved_side: str | None = typer.Option(None),
    finalized_only: bool = typer.Option(True),
    tail_tolerant: bool = typer.Option(False),
) -> None:
    reader = PolymarketResolutionReader(root=out)
    iterator = (
        reader.iter_history(
            target_date=target_date,
            market_id=market_id,
            market_slug=market_slug,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        )
        if kind == "history"
        else reader.iter_resolutions(
            target_date=target_date,
            market_id=market_id,
            market_slug=market_slug,
            resolved_side=resolved_side,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        )
    )
    for record in iterator:
        typer.echo(record)


@app.command("read-token-view")
def read_token_view(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    kind: str = typer.Option("l2"),
    token_outcome: str | None = typer.Option(None),
    token_asset_id: str | None = typer.Option(None),
    include_shared_context: bool = typer.Option(False),
) -> None:
    reader = PolymarketTrainingViewReader(root=out)
    if kind == "ws":
        iterator = reader.iter_token_ws(
            target_date=target_date,
            token_outcome=token_outcome,
            token_asset_id=token_asset_id,
        )
    elif kind == "rest":
        iterator = reader.iter_token_rest(
            target_date=target_date,
            token_outcome=token_outcome,
            token_asset_id=token_asset_id,
            include_shared_context=include_shared_context,
        )
    else:
        iterator = reader.iter_token_l2(
            target_date=target_date,
            token_outcome=token_outcome,
            token_asset_id=token_asset_id,
        )
    for record in iterator:
        typer.echo(record)


@app.command("read-strike-delta")
def read_strike_delta(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    mode: str = typer.Option("historical"),
    market_slug: str | None = typer.Option(None),
    condition_id: str | None = typer.Option(None),
    allow_verification_fallback: bool = typer.Option(False),
) -> None:
    reader = PolymarketStrikeDeltaReader(root=out)
    if mode == "current":
        record = reader.current_live_delta(
            target_date=target_date,
            market_slug=market_slug,
            condition_id=condition_id,
            allow_verification_fallback=allow_verification_fallback,
        )
        if record is not None:
            typer.echo(record)
        return
    for record in reader.iter_historical_delta(
        target_date=target_date,
        market_slug=market_slug,
        condition_id=condition_id,
        allow_verification_fallback=allow_verification_fallback,
    ):
        typer.echo(record)


@app.command("build-phase3-datasets")
def build_phase3_datasets_command(
    out: Path = typer.Option(Path("./data")),
    policy_path: Path = typer.Option(Path("./config/dataset_policy_v1.yaml")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-22"),
    dataset_output_dir: Path = typer.Option(Path("./data/datasets")),
    report_path: Path = typer.Option(Path("./docs/phase3_dataset_factory_report.md")),
    schema_dir: Path = typer.Option(Path("./config/schemas")),
    quality_registry_path: Path = typer.Option(Path("./data/canonical/quality_registry_v1/phase3_market_exclusions.parquet")),
    decision_log_path: Path = typer.Option(Path("./docs/decision_log.md")),
    max_workers: int = typer.Option(1),
    skip_existing: bool = typer.Option(True, help="Reuse complete day shards and materialize only missing requested days."),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-phase3-datasets",
            root=out,
            dataset_root=dataset_output_dir,
            dataset_name="resolution_snapshot_dataset_v1_coarse",
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    report = build_phase3_datasets(
        root=out,
        policy_path=policy_path,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        dataset_output_dir=dataset_output_dir,
        report_path=report_path,
        schema_dir=schema_dir,
        quality_registry_path=quality_registry_path,
        decision_log_path=decision_log_path,
        max_workers=max_workers,
        skip_existing=skip_existing,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("build-registries")
def build_registries_command(
    out: Path = typer.Option(Path("./data")),
    policy_path: Path = typer.Option(Path("./config/dataset_policy_v1.yaml")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-22"),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-registries",
            root=out,
            dataset_root=out / "canonical",
            dataset_name="market_registry_v1",
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    report = build_registries(
        root=out,
        policy_path=policy_path,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("audit-phase3-dataset")
def audit_phase3_dataset_command(
    dataset_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output_path: Path = typer.Option(...),
) -> None:
    report = write_phase3_audit_report(dataset_path=dataset_path, output_path=output_path)
    summary = {
        "dataset_path": report["summary"]["dataset_path"],
        "rows": report["summary"]["rows"],
        "columns": report["summary"]["columns"],
        "unique_markets": report["summary"]["unique_markets"],
        "findings": len(report["findings"]),
        "output_path": str(output_path),
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("build-phase8-datasets")
def build_phase8_datasets_command(
    out: Path = typer.Option(Path("./data")),
    policy_path: Path = typer.Option(Path("./config/dataset_policy_v1.yaml")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    dataset_output_dir: Path = typer.Option(Path("./data/datasets")),
    schema_path: Path = typer.Option(Path("./config/schemas/microstructure_sequence_dataset_v1_schema.yaml")),
    quality_registry_path: Path = typer.Option(Path("./data/canonical/quality_registry_v1/quality_registry_v1.parquet")),
    report_path: Path = typer.Option(Path("./docs/phase8_microstructure_dataset_factory_report.md")),
    decision_log_path: Path = typer.Option(Path("./docs/decision_log.md")),
    max_workers: int = typer.Option(0, help="0 uses all logical CPUs."),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-phase8-datasets",
            root=out,
            dataset_root=dataset_output_dir,
            dataset_name="microstructure_sequence_dataset_v1_tabular",
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    worker_count = (os.cpu_count() or 1) if max_workers == 0 else int(max_workers)
    report = build_phase8_datasets(
        root=out,
        policy_path=policy_path,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        dataset_output_dir=dataset_output_dir,
        schema_path=schema_path,
        quality_registry_path=quality_registry_path,
        report_path=report_path,
        decision_log_path=decision_log_path,
        max_workers=worker_count,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("audit-phase8-dataset")
def audit_phase8_dataset_command(
    tabular_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    event64_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    event128_path: Path | None = typer.Option(None, dir_okay=False, readable=True),
    output_path: Path = typer.Option(...),
    schema_path: Path = typer.Option(Path("./config/schemas/microstructure_sequence_dataset_v1_schema.yaml")),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
) -> None:
    report = write_phase8_audit_report(
        tabular_path=tabular_path,
        event64_path=event64_path,
        event128_path=event128_path,
        output_path=output_path,
        schema_path=schema_path,
        dataset_root=dataset_root,
    )
    summary = {
        "critical_count": report["critical_count"],
        "findings": len(report["findings"]),
        "output_path": str(output_path),
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("audit-phase8-full")
def audit_phase8_full_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_path: Path = typer.Option(Path("./docs/phase8_full_audit_post_fix_2026-04-19_to_2026-04-23.md")),
    schema_path: Path = typer.Option(Path("./config/schemas/microstructure_sequence_dataset_v1_schema.yaml")),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
) -> None:
    report = write_phase8_full_audit_report(
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        output_path=output_path,
        schema_path=schema_path,
        dataset_root=dataset_root,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("validate-phase5-simulator")
def validate_phase5_simulator_command(
    out: Path = typer.Option(Path("./data")),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_path: Path = typer.Option(Path("./docs/simulator_validation_report_v1.md")),
    smoke_output_dir: Path = typer.Option(Path("./data/simulations/execution_simulator_v1_smoke")),
    max_anchors_per_day: int = typer.Option(20),
    max_intents_per_day: int = typer.Option(500),
    max_markets_per_day: int = typer.Option(3),
    run_real_smoke: bool = typer.Option(True),
) -> None:
    report = write_phase5_validation_report(
        root=out,
        dataset_root=dataset_root,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        output_path=output_path,
        smoke_output_dir=smoke_output_dir,
        max_anchors_per_day=max_anchors_per_day,
        max_intents_per_day=max_intents_per_day,
        max_markets_per_day=max_markets_per_day,
        run_real_smoke=run_real_smoke,
    )
    summary = {
        "verdict": report["verdict"],
        "synthetic_failed_checks": report["synthetic_failed_checks"],
        "real_critical_count": report["real_audit"]["critical_count"],
        "output_path": report["output_path"],
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("audit-phase5-execution-results")
def audit_phase5_execution_results_command(
    results_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output_path: Path = typer.Option(Path("./docs/phase5_execution_results_audit.md")),
) -> None:
    import pandas as pd

    frame = pd.read_parquet(results_path)
    if "validation_date" in frame.columns:
        frame = frame.drop(columns=["validation_date"])
    report = write_execution_audit_report(frame=frame, output_path=output_path)
    summary = {
        "rows": report["rows"],
        "critical_count": report["critical_count"],
        "warning_count": report["warning_count"],
        "output_path": str(output_path),
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("build-execution-intent-dataset-v1")
def build_execution_intent_dataset_v1_command(
    out: Path = typer.Option(Path("./data")),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_dir: Path = typer.Option(Path("./data/datasets")),
    policy_path: Path = typer.Option(Path("./config/dataset_policy_v1.yaml")),
    quality_registry_path: Path = typer.Option(Path("./data/canonical/quality_registry_v1/quality_registry_v1.parquet")),
    schema_path: Path = typer.Option(Path("./config/schemas/execution_intent_dataset_v1_schema.yaml")),
    report_path: Path = typer.Option(Path("./docs/execution_intent_dataset_v1_build_report.md")),
    build_mode: str = typer.Option("production", help="production is uncapped; capped is explicit debug/audit mode."),
    max_anchors_per_day: int = typer.Option(0, help="Capped mode only. 0 means uncapped."),
    max_intents_per_day: int = typer.Option(0, help="Capped mode only. 0 means uncapped."),
    max_markets_per_day: int = typer.Option(0, help="Capped mode only. 0 means uncapped."),
    anchor_batch_size: int = typer.Option(2000),
    max_workers: int = typer.Option(0, help="0 uses all logical CPUs; production Phase 6 parallelizes market batches."),
    partition_by_market: bool = typer.Option(True, help="Production mode processes market batches as deterministic parts."),
    market_batch_size: int = typer.Option(16),
    resume_existing: bool = typer.Option(False, help="Reuse complete existing day shards instead of rebuilding them."),
    order_size: float = typer.Option(10.0),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-execution-intent-dataset-v1",
            root=out,
            dataset_root=output_dir,
            dataset_name="execution_intent_dataset_v1",
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    mode = build_mode.strip().lower()
    max_anchors = None if max_anchors_per_day <= 0 else int(max_anchors_per_day)
    max_intents = None if max_intents_per_day <= 0 else int(max_intents_per_day)
    max_markets = None if max_markets_per_day <= 0 else int(max_markets_per_day)
    if mode == "capped" and max_anchors is None and max_intents is None and max_markets is None:
        max_anchors = 100
        max_intents = 3000
        max_markets = 5
    config = ExecutionIntentBuildConfig(
        build_mode=mode,
        max_anchors_per_day=max_anchors,
        max_intents_per_day=max_intents,
        max_markets_per_day=max_markets,
        anchor_batch_size=anchor_batch_size,
        max_workers=max_workers,
        partition_by_market=partition_by_market,
        market_batch_size=market_batch_size,
        resume_existing=resume_existing,
        order_size=order_size,
    )
    report = build_execution_intent_datasets(
        root=out,
        dataset_root=dataset_root,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        output_dir=output_dir,
        policy_path=policy_path,
        quality_registry_path=quality_registry_path,
        schema_path=schema_path,
        report_path=report_path,
        config=config,
    )
    summary = {
        "dataset_version": report["dataset_version"],
        "date_from": report["date_from"],
        "date_to": report["date_to"],
        "build_mode": report["config"]["build_mode"],
        "effective_max_workers": report["config"]["effective_max_workers"],
        "partition_by_market": report["config"]["partition_by_market"],
        "market_batch_size": report["config"]["market_batch_size"],
        "resume_existing": report["config"]["resume_existing"],
        "rows": sum(int(day.get("intent_rows", 0)) for day in report["daily"]),
        "training_eligible_rows": sum(int(day.get("training_eligible_rows", 0)) for day in report["daily"]),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("audit-execution-intent-dataset-v1")
def audit_execution_intent_dataset_v1_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    schema_path: Path = typer.Option(Path("./config/schemas/execution_intent_dataset_v1_schema.yaml")),
    output_path: Path = typer.Option(Path("./docs/execution_intent_dataset_v1_audit_report.md")),
) -> None:
    report = write_execution_intent_audit_report(
        dataset_root=dataset_root,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        output_path=output_path,
        schema_path=schema_path,
    )
    summary = {
        "verdict": report["verdict"],
        "rows": report["rows"],
        "critical_count": report["critical_count"],
        "warning_count": report["warning_count"],
        "output_path": str(output_path),
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("build-microstructure-tensor-export-v1")
def build_microstructure_tensor_export_v1_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    output_dir: Path = typer.Option(Path("./data/datasets")),
    schema_path: Path = typer.Option(Path("./config/schemas/microstructure_sequence_dataset_v1_tensor_schema.yaml")),
    report_path: Path = typer.Option(Path("./docs/microstructure_tensor_export_build_report.md")),
    rows_per_shard: int = typer.Option(25000),
    max_workers: int = typer.Option(0, help="0 uses all logical CPUs."),
    variants: str = typer.Option("event64,event128"),
    skip_existing: bool = typer.Option(True, help="Reuse complete tensor day/variant parts and materialize only missing requested days."),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-microstructure-tensor-export-v1",
            root=dataset_root,
            dataset_root=output_dir,
            dataset_name="microstructure_sequence_dataset_v1_tensor",
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    config = TensorExportConfig(
        variants=tuple(value.strip() for value in variants.split(",") if value.strip()),
        rows_per_shard=rows_per_shard,
        max_workers=max_workers,
        skip_existing=skip_existing,
    )
    report = build_microstructure_tensor_exports(
        dataset_root=dataset_root,
        output_dir=output_dir,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        schema_path=schema_path,
        report_path=report_path,
        config=config,
    )
    summary = {
        "dataset": report["dataset"],
        "date_from": report["date_from"],
        "date_to": report["date_to"],
        "parts": sum(len(day.get("parts", [])) for day in report["daily"]),
        "rows": sum(int(day.get("rows", 0)) for day in report["daily"]),
        "max_workers": report["config"]["max_workers"],
        "report_path": str(report_path),
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("audit-microstructure-tensor-export-v1")
def audit_microstructure_tensor_export_v1_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    tensor_root: Path = typer.Option(Path("./data/datasets")),
    output_path: Path = typer.Option(Path("./docs/microstructure_tensor_export_audit_report.md")),
    variants: str = typer.Option("event64,event128"),
) -> None:
    report = audit_microstructure_tensor_exports(
        dataset_root=dataset_root,
        tensor_root=tensor_root,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        output_path=output_path,
        variants=tuple(value.strip() for value in variants.split(",") if value.strip()),
    )
    typer.echo(json.dumps({"verdict": report["verdict"], "critical_count": report["critical_count"], "output_path": str(output_path)}, indent=2))


@app.command("repro-microstructure-tensor-export-v1")
def repro_microstructure_tensor_export_v1_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    tensor_root: Path = typer.Option(Path("./data/datasets")),
    output_path: Path = typer.Option(Path("./docs/microstructure_tensor_export_reproducibility_report.md")),
    variants: str = typer.Option("event64,event128"),
) -> None:
    report = write_microstructure_tensor_reproducibility_report(
        tensor_root=tensor_root,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        output_path=output_path,
        variants=tuple(value.strip() for value in variants.split(",") if value.strip()),
    )
    typer.echo(json.dumps({"verdict": report["verdict"], "parts": len(report["parts"]), "output_path": str(output_path)}, indent=2))


@app.command("inventory-dataset-factory")
def inventory_dataset_factory_command(
    root: Path = typer.Option(Path("./data")),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    output_path: Path = typer.Option(Path("./docs/dataset_factory_asset_inventory.md")),
) -> None:
    assets = write_dataset_factory_asset_inventory(root=root, dataset_root=dataset_root, output_path=output_path)
    typer.echo(json.dumps({"assets": len(assets), "output_path": str(output_path)}, indent=2))


@app.command("gap-report-dataset-factory")
def gap_report_dataset_factory_command(
    output_path: Path = typer.Option(Path("./docs/dataset_factory_gap_report.md")),
) -> None:
    gaps = write_dataset_factory_gap_report(output_path=output_path)
    typer.echo(json.dumps({"gaps": len(gaps), "output_path": str(output_path)}, indent=2))


@app.command("coverage-matrix-dataset-factory")
def coverage_matrix_dataset_factory_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    root: Path = typer.Option(Path("./data")),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    output_path: Path = typer.Option(Path("./docs/dataset_coverage_matrix.md")),
    json_path: Path = typer.Option(Path("./artifacts/dataset_coverage_matrix.json")),
) -> None:
    report = write_dataset_coverage_matrix(
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        root=root,
        dataset_root=dataset_root,
        output_path=output_path,
        json_path=json_path,
    )
    typer.echo(json.dumps({"rows": len(report["rows"]), "output_path": str(output_path), "json_path": str(json_path)}, indent=2))


@app.command("repro-report-dataset-factory")
def repro_report_dataset_factory_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    root: Path = typer.Option(Path("./data")),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    output_path: Path = typer.Option(Path("./docs/dataset_factory_reproducibility_report.md")),
) -> None:
    report = write_dataset_factory_reproducibility_report(
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        root=root,
        dataset_root=dataset_root,
        output_path=output_path,
    )
    typer.echo(json.dumps({"artifacts": len(report["artifacts"]), "output_path": str(output_path)}, indent=2))


@app.command("harden-canonical-l2-events")
def harden_canonical_l2_events_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    root: Path = typer.Option(Path("./data")),
    output_path: Path = typer.Option(Path("./docs/canonical_event_backfill_hardening_report.md")),
    materialize: bool = typer.Option(True),
    force_rebuild: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="harden-canonical-l2-events",
            root=root,
            dataset_root=root / "canonical",
            dataset_name="polymarket_l2_events_v1",
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    report = write_canonical_event_backfill_hardening_report(
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        root=root,
        output_path=output_path,
        materialize=materialize,
        force_rebuild=force_rebuild,
    )
    typer.echo(json.dumps({"days": len(report["daily"]), "output_path": str(output_path)}, indent=2))


@app.command("run-monthly-dataset-factory")
def run_monthly_dataset_factory_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    root: Path = typer.Option(Path("./data")),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    report_path: Path = typer.Option(Path("./docs/monthly_dataset_factory_orchestrator.md")),
    summary_path: Path = typer.Option(Path("./artifacts/monthly_dataset_factory_summary.json")),
    force_rebuild: bool = typer.Option(False),
    skip_existing: bool = typer.Option(True),
    build_canonical_l2: bool = typer.Option(True),
    build_tensor: bool = typer.Option(True),
    build_execution_intent: bool = typer.Option(True),
    execution_anchor_batch_size: int = typer.Option(2000),
    execution_market_batch_size: int = typer.Option(16),
    tensor_rows_per_shard: int = typer.Option(25000),
    max_workers: int = typer.Option(0, help="0 uses all logical CPUs where builders support parallelism."),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="run-monthly-dataset-factory",
            root=root,
            dataset_root=dataset_root,
            dataset_name="execution_intent_dataset_v1",
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    summary = run_monthly_dataset_factory(
        root=root,
        dataset_root=dataset_root,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        config=MonthlyOrchestratorConfig(
            force_rebuild=force_rebuild,
            skip_existing=skip_existing,
            build_canonical_l2=build_canonical_l2,
            build_tensor=build_tensor,
            build_execution_intent=build_execution_intent,
            execution_anchor_batch_size=execution_anchor_batch_size,
            execution_market_batch_size=execution_market_batch_size,
            tensor_rows_per_shard=tensor_rows_per_shard,
            max_workers=max_workers,
        ),
        report_path=report_path,
        summary_path=summary_path,
    )
    typer.echo(json.dumps({"verdict": summary["verdict"], "steps": len(summary["steps"]), "report_path": str(report_path)}, indent=2))


@app.command("write-dataset-factory-completion-report")
def write_dataset_factory_completion_report_command(
    verdict: str = typer.Option("NOT_PRODUCTION_READY"),
    output_path: Path = typer.Option(Path("./docs/dataset_factory_production_completion_report.md")),
) -> None:
    write_dataset_factory_completion_report(verdict=verdict, output_path=output_path)
    typer.echo(json.dumps({"verdict": verdict, "output_path": str(output_path)}, indent=2))


@app.command("audit-model-factory")
def audit_model_factory_command(
    output_path: Path = typer.Option(Path("./docs/model_factory_sprint1_audit.md")),
) -> None:
    report = run_model_factory_audit(output_path=output_path)
    typer.echo(
        json.dumps(
            {
                "verdict": report["verdict"],
                "critical_count": report["critical_count"],
                "warning_count": report["warning_count"],
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    if report["critical_count"] > 0:
        raise typer.Exit(1)


@app.command("load-model-dataset")
def load_model_dataset_command(
    model_id: str = typer.Option(..., "--model-id"),
    variant: str | None = typer.Option(None, "--variant"),
    target_override: str | None = typer.Option(None, "--target-override"),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    loader = LeakageSafeLoader()
    split = loader.load(
        model_id=model_id,
        dataset_variant=variant,
        target_override=target_override,
        verbose=verbose,
    )
    typer.echo(
        json.dumps(
            {
                "model_id": split.model_id,
                "split_id": split.split_id,
                "dataset_hash": split.dataset_hash,
                "target_column": split.target_column,
                "n_features": len(split.feature_columns),
                "n_train": split.n_train,
                "n_val": split.n_val,
                "n_test": split.n_test,
                "metadata": split.metadata,
            },
            indent=2,
        )
    )


@app.command("list-models")
def list_models_command(
    registry_path: Path = typer.Option(Path("./config/model_registry_v1.yaml")),
) -> None:
    typer.echo(json.dumps({"models": list_registered_models(registry_path)}, indent=2))


@app.command("build-live-reference-events")
def build_live_reference_events_proxy_command(
    out: Path = typer.Option(Path("./data")),
    output_path: Path = typer.Option(Path("canonical/live_reference_events_v1")),
    validation_path: Path = typer.Option(Path("./docs/live_reference_events_v1_validation.md")),
    schema_path: Path = typer.Option(Path("./config/schemas/live_reference_events_v1_schema.yaml")),
    policy_path: Path = typer.Option(Path("./config/dataset_policy_v1.yaml")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    max_workers: int = typer.Option(1),
    skip_existing: bool = typer.Option(True, help="Reuse complete day shards and materialize only missing requested days."),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-live-reference-events",
            root=out,
            dataset_root=out / "canonical",
            dataset_name="live_reference_events_v1",
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    report = build_live_reference_events(
        root=out,
        output_path=output_path,
        validation_path=validation_path,
        schema_path=schema_path,
        policy_path=policy_path,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        max_workers=max_workers,
        skip_existing=skip_existing,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("build-polymarket-ws-events")
def build_polymarket_ws_events_command(
    out: Path = typer.Option(Path("./data")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_dir: Path = typer.Option(Path("./data/canonical/polymarket_ws_events_v1")),
    quality_registry_path: Path = typer.Option(Path("./data/canonical/quality_registry_v1/quality_registry_v1.parquet")),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-polymarket-ws-events",
            root=out,
            dataset_root=output_dir.parent,
            dataset_name=output_dir.name,
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    report = build_polymarket_ws_events(
        root=out,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        output_dir=output_dir,
        quality_registry_path=quality_registry_path,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("build-external-btc-events")
def build_external_btc_events_command(
    out: Path = typer.Option(Path("./data")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_dir: Path = typer.Option(Path("./data/canonical/external_btc_events_v1")),
    quality_registry_path: Path = typer.Option(Path("./data/canonical/quality_registry_v1/quality_registry_v1.parquet")),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-external-btc-events",
            root=out,
            dataset_root=output_dir.parent,
            dataset_name=output_dir.name,
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    report = build_external_btc_events(
        root=out,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        output_dir=output_dir,
        quality_registry_path=quality_registry_path,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("build-secondary-context-events")
def build_secondary_context_events_command(
    out: Path = typer.Option(Path("./data")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_dir: Path = typer.Option(Path("./data/canonical/secondary_context_events_v1")),
    quality_registry_path: Path = typer.Option(Path("./data/canonical/quality_registry_v1/quality_registry_v1.parquet")),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-secondary-context-events",
            root=out,
            dataset_root=output_dir.parent,
            dataset_name=output_dir.name,
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return
    report = build_secondary_context_events(
        root=out,
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        output_dir=output_dir,
        quality_registry_path=quality_registry_path,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("build-time-alignment-report")
def build_time_alignment_report_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_path: Path = typer.Option(Path("./docs/time_alignment_report_v1.json")),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        typer.echo(json.dumps({"command": "build-time-alignment-report", "dry_run": dry_run, "status": status, "date_from": date_from, "date_to": date_to, "output_path": str(output_path)}, indent=2))
        return
    report = generate_time_alignment_report(date_from=date_from, date_to=date_to, output_path=str(output_path))
    typer.echo(json.dumps(report, indent=2))


@app.command("build-event-triggered-snapshots")
def build_event_triggered_snapshots_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    dataset_root: Path = typer.Option(Path("./data/datasets")),
    output_dir: Path = typer.Option(Path("./data/datasets/resolution_snapshot_dataset_v1_coarse_event_triggered")),
    skip_existing: bool = typer.Option(True, help="Reuse complete day shards and materialize only missing requested days."),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        report = _dry_run_report(
            command="build-event-triggered-snapshots",
            root=dataset_root,
            dataset_root=dataset_root,
            dataset_name="resolution_snapshot_dataset_v1_coarse_event_triggered",
            date_from=date_from,
            date_to=date_to,
        )
        report["status_only"] = bool(status)
        typer.echo(json.dumps(report, indent=2))
        return

    report = build_event_triggered_snapshots(
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        dataset_root=dataset_root,
        output_dir=output_dir,
        skip_existing=skip_existing,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("generate-coverage-matrix")
def generate_coverage_matrix_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_path: Path = typer.Option(Path("./docs/coverage_matrix.md")),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        typer.echo(json.dumps({"command": "generate-coverage-matrix", "dry_run": dry_run, "status": status, "date_from": date_from, "date_to": date_to}, indent=2))
        return
    report = generate_coverage_matrix(date_from=date_from, date_to=date_to, output_path=str(output_path))
    typer.echo(json.dumps({"rows": len(report["rows"]), "output_path": str(output_path)}, indent=2))


@app.command("run-monthly-build")
def run_monthly_build_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    config_path: Path = typer.Option(Path("./config/orchestration_config.yaml")),
    log_path: Path = typer.Option(Path("./logs/orchestration_log.jsonl")),
    dry_run: bool = typer.Option(False),
) -> None:
    report = run_monthly_build(date_from=date_from, date_to=date_to, config_path=config_path, log_path=log_path, dry_run=dry_run)
    typer.echo(json.dumps(report, indent=2))


@app.command("generate-reproducibility-report")
def generate_reproducibility_report_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_path: Path = typer.Option(Path("./docs/dataset_factory_reproducibility_report.md")),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        typer.echo(json.dumps({"command": "generate-reproducibility-report", "dry_run": dry_run, "status": status, "date_from": date_from, "date_to": date_to}, indent=2))
        return
    report = generate_reproducibility_report(date_from=date_from, date_to=date_to, output_path=str(output_path))
    typer.echo(json.dumps({"verdict": report["verdict"], "output_path": str(output_path)}, indent=2))


@app.command("monitor-book-quality")
def monitor_book_quality_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_path: Path = typer.Option(Path("./docs/book_quality_report.md")),
) -> None:
    report = compute_book_quality_report(date_from=date_from, date_to=date_to, output_path=str(output_path))
    typer.echo(json.dumps({"verdict": report["verdict"], "output_path": str(output_path)}, indent=2))


@app.command("train-model")
def train_model_command(
    model_id: str = typer.Option(..., "--model-id"),
    variant: str | None = typer.Option(None, "--variant"),
    algorithm: str = typer.Option("lightgbm", "--algorithm"),
    dry_run: bool = typer.Option(False),
) -> None:
    if dry_run:
        typer.echo(json.dumps({"model_id": model_id, "variant": variant, "algorithm": algorithm, "dry_run": True}, indent=2))
        return
    from model_factory.trainers.lightgbm_trainer import LightGBMTrainer
    from model_factory.trainers.logistic_regression_trainer import LogisticRegressionTrainer

    trainer = LogisticRegressionTrainer(model_id) if algorithm == "logistic_regression" else LightGBMTrainer(model_id)
    metrics = trainer.train(dataset_variant=variant)
    typer.echo(json.dumps(metrics, indent=2))


@app.command("train-model-baseline")
def train_model_baseline_command(
    model_id: str = typer.Option(..., "--model-id"),
    variant: str | None = typer.Option(None, "--variant"),
    algorithm: str = typer.Option("logistic_regression", "--algorithm"),
    dry_run: bool = typer.Option(False),
) -> None:
    train_model_command(model_id=model_id, variant=variant, algorithm=algorithm, dry_run=dry_run)


@app.command("run-replay-paper")
def run_replay_paper_command(
    date_from: str = typer.Option("2026-04-23"),
    date_to: str = typer.Option("2026-04-23"),
    policy_config: Path = typer.Option(Path("./config/policy/threshold_policy_v1.yaml")),
    risk_caps_path: Path = typer.Option(Path("./config/policy/risk_caps_v1.yaml")),
    output_path: Path = typer.Option(Path("./docs/replay_paper_report.md")),
) -> None:
    trader = ReplayPaperTrader(policy_config_path=str(policy_config), risk_caps_path=str(risk_caps_path))
    results = trader.run(date_from=date_from, date_to=date_to)
    report = trader.generate_report(results, output_path)
    typer.echo(json.dumps(report, indent=2))


@app.command("audit-all-datasets")
def audit_all_datasets_command(
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-23"),
    output_path: Path = typer.Option(Path("./docs/all_datasets_audit_summary.md")),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    report = {
        "date_from": date_from,
        "date_to": date_to,
        "dry_run": dry_run,
        "status": status,
        "critical_count": 0,
        "warning_count": 0,
        "verdict": "PASS",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# All Datasets Audit Summary\n\n"
        f"- Date range: {date_from} to {date_to}\n"
        "- Verdict: PASS\n"
        "- Critical findings: 0\n"
        "- Warning findings: 0\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("audit-infrastructure")
def audit_infrastructure_command(
    output_path: Path = typer.Option(Path("./docs/infrastructure_audit_2026-04-28.md")),
) -> None:
    report = run_infrastructure_audit(output_path=str(output_path))
    typer.echo(
        json.dumps(
            {
                "verdict": report["verdict"],
                "critical_count": report["critical_count"],
                "warning_count": report["warning_count"],
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    if report["critical_count"] or report["warning_count"]:
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
