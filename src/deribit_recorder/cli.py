from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from binance_recorder.logging_utils import configure_logging

from .config import DeribitRecorderConfig
from .reader import DeribitRawReader
from .recorder import DeribitRecorderService

app = typer.Typer(add_completion=False)


@app.command("record")
def record(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(DeribitRecorderService(DeribitRecorderConfig.from_env(out=out)).run())


@app.command("read")
def read(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    kind: str = typer.Option("ws"),
    currency: str = typer.Option("BTC"),
) -> None:
    reader = DeribitRawReader(root=out, currency=currency)
    for record in reader.iter_kind(target_date=target_date, kind=kind):
        typer.echo(record)


def main() -> None:
    app()
