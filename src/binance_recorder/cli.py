from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from binance_recorder.archive import download_archive_day
from binance_recorder.config import RecorderConfig
from binance_recorder.logging_utils import configure_logging
from binance_recorder.reader import RawArchiveReader
from binance_recorder.usdm_config import BinanceUsdmRecorderConfig
from binance_recorder.usdm_service import BinanceUsdmRecorderService
from binance_recorder.websocket_client import BinanceRecorderService
from binance_recorder.utils import parse_utc_date

app = typer.Typer(add_completion=False)


@app.command("record")
def record(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(BinanceRecorderService(RecorderConfig.from_env(out=out)).run())


@app.command("record-usdm")
def record_usdm(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure_logging(log_level)
    asyncio.run(BinanceUsdmRecorderService(BinanceUsdmRecorderConfig.from_env(out=out)).run())


@app.command("read")
def read(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    market: str = typer.Option("spot"),
    symbol: str = typer.Option("BTCUSDT"),
    kind: str = typer.Option("ws"),
) -> None:
    reader = RawArchiveReader(root=out, market=market, symbol=symbol)
    for record in reader.iter_kind(target_date=target_date, kind=kind):
        typer.echo(record)


@app.command("download-archive-day")
def download_archive_day_command(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
    market: str = typer.Option("spot"),
    dataset: str = typer.Option("trades"),
    symbol: str = typer.Option("BTCUSDT"),
) -> None:
    metadata = download_archive_day(
        out=out,
        market=market,
        dataset=dataset,
        symbol=symbol,
        target_date=parse_utc_date(target_date),
    )
    typer.echo(metadata)


def main() -> None:
    app()
