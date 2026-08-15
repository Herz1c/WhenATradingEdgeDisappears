from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import orjson
import typer

from binance_recorder.logging_utils import configure_logging

from .config import (
    load_chainlink_config,
    load_chainlink_onchain_config,
    load_chainlink_public_delayed_config,
)
from .live_reference import ChainlinkLiveReferenceReader
from .live_reference_events import build_live_reference_events
from .onchain_service import ChainlinkOnchainRecorderService
from .public_delayed import ChainlinkPublicDelayedRecorderService
from .reader import ChainlinkPublicDelayedRawReader, ChainlinkRawReader
from .recorder import ChainlinkRecorderService
from .synthetic_chainlink import build_synthetic_chainlink_report

app = typer.Typer(add_completion=False)


@app.command("record-btcusd")
def record_btcusd(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(ChainlinkRecorderService(load_chainlink_config(out=out)).run())


@app.command("record-onchain")
def record_onchain(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(ChainlinkOnchainRecorderService(load_chainlink_onchain_config(out=out)).run())


@app.command("record-public-delayed")
def record_public_delayed(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(ChainlinkPublicDelayedRecorderService(load_chainlink_public_delayed_config(out=out)).run())


@app.command("read")
def read(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    kind: str = typer.Option("live"),
    backend: str = typer.Option("datastreams_ws"),
    symbol: str = typer.Option("BTCUSD"),
) -> None:
    reader = ChainlinkRawReader(root=out, backend=backend, symbol=symbol)
    for record in reader.iter_kind(target_date=target_date, kind=kind):
        typer.echo(record)


@app.command("read-live-reference")
def read_live_reference(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    symbol: str = typer.Option("BTCUSD"),
    rtds_symbol: str = typer.Option("btc/usd"),
    source: str = typer.Option("best"),
    allow_verification_fallback: bool = typer.Option(False),
) -> None:
    reader = ChainlinkLiveReferenceReader(root=out, symbol=symbol, rtds_symbol=rtds_symbol)
    if source == "direct":
        iterator = reader.iter_direct_live(target_date=target_date)
    elif source == "rtds":
        iterator = reader.iter_rtds_proxy(target_date=target_date)
    elif source == "verification":
        iterator = reader.iter_verification_fallback(target_date=target_date)
    else:
        iterator = reader.iter_best_available(
            target_date=target_date,
            allow_verification_fallback=allow_verification_fallback,
        )
    for record in iterator:
        typer.echo(record)


@app.command("read-public-delayed")
def read_public_delayed(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    symbol: str = typer.Option("BTCUSD"),
    kind: str = typer.Option("page"),
) -> None:
    reader = ChainlinkPublicDelayedRawReader(root=out, symbol=symbol)
    for record in reader.iter_kind(target_date=target_date, kind=kind):
        typer.echo(record)


@app.command("build-synthetic-report")
def build_synthetic_report(
    out: Path = typer.Option(Path("./data")),
    artifact_dir: Path = typer.Option(Path("./docs")),
    max_workers: int = typer.Option(1),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        typer.echo(orjson.dumps({"dry_run": dry_run, "status": status, "artifact_dir": str(artifact_dir)}, option=orjson.OPT_INDENT_2).decode("utf-8"))
        return
    report = build_synthetic_chainlink_report(
        root=out,
        artifact_dir=artifact_dir,
        max_workers=max_workers,
    )
    typer.echo(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode("utf-8"))


@app.command("build-live-reference-events")
def build_live_reference_events_command(
    out: Path = typer.Option(Path("./data")),
    output_path: Path = typer.Option(Path("canonical/live_reference_events_v1")),
    validation_path: Path = typer.Option(Path("./docs/live_reference_events_v1_validation.md")),
    schema_path: Path = typer.Option(Path("./config/schemas/live_reference_events_v1_schema.yaml")),
    policy_path: Path = typer.Option(Path("./config/dataset_policy_v1.yaml")),
    date_from: str = typer.Option("2026-04-19"),
    date_to: str = typer.Option("2026-04-22"),
    max_workers: int = typer.Option(1),
    dry_run: bool = typer.Option(False),
    status: bool = typer.Option(False),
) -> None:
    if dry_run or status:
        rows = []
        start = date.fromisoformat(date_from)
        end = date.fromisoformat(date_to)
        current = start
        while current <= end:
            shard = out / output_path / f"{current.isoformat()}.parquet"
            rows.append({"date": current.isoformat(), "exists": shard.exists(), "path": str(shard)})
            current = date.fromordinal(current.toordinal() + 1)
        typer.echo(orjson.dumps({"dry_run": dry_run, "status": status, "outputs": rows}, option=orjson.OPT_INDENT_2).decode("utf-8"))
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
    )
    typer.echo(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode("utf-8"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
