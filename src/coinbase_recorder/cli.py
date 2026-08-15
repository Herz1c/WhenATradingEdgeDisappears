from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from binance_recorder.logging_utils import configure_logging

from .config import CoinbaseAdvancedRecorderConfig
from .reader import CoinbaseAdvancedRawReader
from .recorder import CoinbaseAdvancedRecorderService

app = typer.Typer(add_completion=False)


@app.command("record")
def record(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(CoinbaseAdvancedRecorderService(CoinbaseAdvancedRecorderConfig.from_env(out=out)).run())


@app.command("read")
def read(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    kind: str = typer.Option("ws"),
    product_id: str = typer.Option("BTC-USD"),
) -> None:
    reader = CoinbaseAdvancedRawReader(root=out, product_id=product_id)
    for record in reader.iter_kind(target_date=target_date, kind=kind):
        typer.echo(record)


def main() -> None:
    app()
