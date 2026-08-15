from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from binance_recorder.logging_utils import configure_logging

from .config import BybitRecorderConfig
from .reader import BybitRawReader
from .recorder import BybitRecorderService

app = typer.Typer(add_completion=False)


@app.command("record")
def record(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(BybitRecorderService(BybitRecorderConfig.from_env(out=out)).run())


@app.command("read")
def read(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    kind: str = typer.Option("ws"),
    symbol: str = typer.Option("BTCUSDT"),
) -> None:
    reader = BybitRawReader(root=out, symbol=symbol)
    for record in reader.iter_kind(target_date=target_date, kind=kind):
        typer.echo(record)


def main() -> None:
    app()
