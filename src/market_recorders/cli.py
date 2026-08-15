from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import date
from pathlib import Path
from typing import Any

import orjson
import typer

from binance_recorder.cli import main as _unused_binance_main  # noqa: F401
from binance_recorder.config import RecorderConfig
from binance_recorder.logging_utils import configure_logging
from binance_recorder.usdm_config import BinanceUsdmRecorderConfig
from binance_recorder.usdm_service import BinanceUsdmRecorderService
from binance_recorder.utils import parse_utc_date
from binance_recorder.websocket_client import BinanceRecorderService
from bybit_recorder.config import BybitRecorderConfig
from bybit_recorder.recorder import BybitRecorderService
from deribit_recorder.config import DeribitRecorderConfig
from deribit_recorder.recorder import DeribitRecorderService
from chainlink_recorder.config import (
    load_chainlink_config,
    load_chainlink_onchain_config,
    load_chainlink_public_delayed_config,
)
from chainlink_recorder.alignment import build_chainlink_alignment_report
from chainlink_recorder.onchain_service import ChainlinkOnchainRecorderService
from chainlink_recorder.public_delayed import ChainlinkPublicDelayedRecorderService
from chainlink_recorder.recorder import ChainlinkRecorderService
from coinbase_recorder.config import CoinbaseAdvancedRecorderConfig
from coinbase_recorder.recorder import CoinbaseAdvancedRecorderService
from hyperliquid_recorder.config import HyperliquidRecorderConfig
from hyperliquid_recorder.recorder import HyperliquidRecorderService
from market_recorders.archive import rebuild_missing_manifests
from polymarket_recorder.config import PolymarketRecorderConfig
from polymarket_recorder.resolution_recorder import PolymarketResolutionRecorderService
from polymarket_recorder.rtds_config import PolymarketRtdsConfig
from polymarket_recorder.rtds_recorder import PolymarketRtdsRecorderService
from polymarket_recorder.rest_poller import PolymarketRestPollerService
from polymarket_recorder.strike_recorder import PolymarketStrikeRecorderService
from polymarket_recorder.ws_client import PolymarketRecorderService

from .fng_recorder import FearAndGreedRecorderConfig, FearAndGreedRecorderService
from .qc import build_daily_qc_report
from .supervisor import RecorderSupervisor

app = typer.Typer(add_completion=False)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_flag_any(*names: str, default: bool = True) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _build_supervisor(
    *,
    out: Path,
    log_level: str,
    enable_binance_spot: bool = True,
    enable_binance_usdm: bool = True,
    enable_polymarket_ws: bool = True,
    enable_polymarket_rest: bool = True,
    enable_polymarket_strike: bool = True,
    enable_polymarket_resolution: bool = True,
    enable_polymarket_rtds: bool = True,
    enable_bybit: bool = True,
    enable_hyperliquid: bool = True,
    enable_deribit: bool = True,
    enable_coinbase: bool = True,
    enable_chainlink_live: bool = True,
    enable_chainlink_public_delayed: bool = True,
    enable_chainlink_page: bool | None = None,
    enable_chainlink_onchain: bool = True,
    enable_fng: bool = True,
) -> RecorderSupervisor:
    configure_logging(log_level)
    logger = logging.getLogger("market_recorders")
    disabled: list[str] = []
    supervisor = RecorderSupervisor(logger=logger, disabled_components=disabled)
    if enable_chainlink_page is not None:
        enable_chainlink_live = enable_chainlink_page

    if enable_binance_spot and _env_flag("ENABLE_BINANCE_SPOT", True):
        spot_config = RecorderConfig.from_env(out=out)
        supervisor.add_service(
            "binance_spot",
            {"symbol": spot_config.symbol, "mode": "spot_combined_streams"},
            lambda config=spot_config: BinanceRecorderService(config, logger=logger),
        )
    else:
        disabled.append("binance_spot")

    if enable_binance_usdm and _env_flag("ENABLE_BINANCE_USDM", True):
        usdm_config = BinanceUsdmRecorderConfig.from_env(out=out)
        supervisor.add_service(
            "binance_usdm",
            {"symbol": usdm_config.symbol, "mode": "futures_public_ws_rest"},
            lambda config=usdm_config: BinanceUsdmRecorderService(config, logger=logger),
        )
    else:
        disabled.append("binance_usdm")

    poly_config = PolymarketRecorderConfig.from_env(out=out)
    if enable_polymarket_ws and _env_flag("ENABLE_POLYMARKET_WS", True):
        supervisor.add_service(
            "polymarket_ws",
            {"market_family": "btc_updown_5m", "mode": "public_market_ws"},
            lambda config=poly_config: PolymarketRecorderService(config, logger=logger),
        )
    else:
        disabled.append("polymarket_ws")

    if enable_polymarket_rest and _env_flag("ENABLE_POLYMARKET_REST", True):
        supervisor.add_service(
            "polymarket_rest",
            {"market_family": "btc_updown_5m", "mode": "public_rest_polling"},
            lambda config=poly_config: PolymarketRestPollerService(config, logger=logger),
        )
    else:
        disabled.append("polymarket_rest")

    if enable_polymarket_strike and _env_flag("ENABLE_POLYMARKET_STRIKE", True):
        supervisor.add_service(
            "polymarket_strike",
            {"market_family": "btc_updown_5m", "mode": "public_market_strike"},
            lambda config=poly_config: PolymarketStrikeRecorderService(config, logger=logger),
        )
    else:
        disabled.append("polymarket_strike")

    if enable_polymarket_resolution and _env_flag("ENABLE_POLYMARKET_RESOLUTION", True):
        supervisor.add_service(
            "polymarket_resolution",
            {"market_family": "btc_updown_5m", "mode": "post_close_market_resolution"},
            lambda config=poly_config: PolymarketResolutionRecorderService(config, logger=logger),
        )
    else:
        disabled.append("polymarket_resolution")

    if enable_polymarket_rtds and _env_flag("ENABLE_POLYMARKET_RTDS", True):
        polymarket_rtds_config = PolymarketRtdsConfig.from_env(out=out)
        supervisor.add_service(
            "polymarket_rtds",
            {
                "topic": polymarket_rtds_config.topic,
                "symbol": polymarket_rtds_config.symbol,
                "mode": "public_rtds_chainlink_ws",
            },
            lambda config=polymarket_rtds_config: PolymarketRtdsRecorderService(config, logger=logger),
        )
    else:
        disabled.append("polymarket_rtds")

    if enable_bybit and _env_flag("ENABLE_BYBIT", True):
        bybit_config = BybitRecorderConfig.from_env(out=out)
        supervisor.add_service(
            "bybit",
            {"symbol": bybit_config.symbol, "mode": "public_linear_ws_rest"},
            lambda config=bybit_config: BybitRecorderService(config, logger=logger),
        )
    else:
        disabled.append("bybit")

    if enable_hyperliquid and _env_flag("ENABLE_HYPERLIQUID", True):
        hyperliquid_config = HyperliquidRecorderConfig.from_env(out=out)
        supervisor.add_service(
            "hyperliquid",
            {"symbol": hyperliquid_config.symbol, "mode": "public_linear_ws_rest"},
            lambda config=hyperliquid_config: HyperliquidRecorderService(config, logger=logger),
        )
    else:
        disabled.append("hyperliquid")

    if enable_deribit and _env_flag("ENABLE_DERIBIT", True):
        deribit_config = DeribitRecorderConfig.from_env(out=out)
        supervisor.add_service(
            "deribit",
            {"symbol": deribit_config.currency, "mode": "public_options_ws_rest"},
            lambda config=deribit_config: DeribitRecorderService(config, logger=logger),
        )
    else:
        disabled.append("deribit")

    if enable_coinbase and _env_flag("ENABLE_COINBASE", True):
        coinbase_config = CoinbaseAdvancedRecorderConfig.from_env(out=out)
        supervisor.add_service(
            "coinbase_advanced",
            {"product_id": coinbase_config.product_id, "mode": "public_advanced_ws"},
            lambda config=coinbase_config: CoinbaseAdvancedRecorderService(config, logger=logger),
        )
    else:
        disabled.append("coinbase_advanced")

    if enable_chainlink_live and _env_flag_any("ENABLE_CHAINLINK_LIVE", "ENABLE_CHAINLINK_PAGE", default=True):
        chainlink_config = load_chainlink_config(out=out)
        chainlink_issue = chainlink_config.availability_issue()
        if chainlink_issue is None and chainlink_config.selected_backend is not None:
            if chainlink_config.selected_backend == "public_page":
                chainlink_mode = "verification_only_public_page"
            elif chainlink_config.selected_backend == "datastreams_ws":
                chainlink_mode = "signed_ws_stream"
            else:
                chainlink_mode = "signed_rest_latest_report"
            supervisor.add_service(
                "chainlink_live",
                {
                    "symbol": chainlink_config.symbol,
                    "backend": chainlink_config.selected_backend,
                    "mode": chainlink_mode,
                    "datastreams_credentials_configured": bool(chainlink_config.datastreams_api_key and chainlink_config.datastreams_api_secret),
                },
                lambda config=chainlink_config: ChainlinkRecorderService(config, logger=logger),
            )
        else:
            disabled.append("chainlink_live")
            log_message = chainlink_config.availability_message() or "Chainlink live service disabled"
            log_extra = {
                "backend": chainlink_config.selected_backend,
                "availability_issue": chainlink_issue,
            }
            if chainlink_issue == "direct_datastreams_not_configured":
                logger.info(log_message, extra=log_extra)
            else:
                logger.warning(log_message, extra=log_extra)
    else:
        disabled.append("chainlink_live")

    if enable_chainlink_public_delayed and _env_flag("ENABLE_CHAINLINK_PUBLIC_DELAYED", True):
        public_delayed_config = load_chainlink_public_delayed_config(out=out)
        supervisor.add_service(
            "chainlink_public_delayed",
            {
                "symbol": public_delayed_config.symbol,
                "backend": "public_stream_page",
                "mode": "public_delayed_comparison_layer",
            },
            lambda config=public_delayed_config: ChainlinkPublicDelayedRecorderService(config, logger=logger),
        )
    else:
        disabled.append("chainlink_public_delayed")

    if enable_chainlink_onchain and _env_flag("ENABLE_CHAINLINK_ONCHAIN", True):
        onchain_config = load_chainlink_onchain_config(out=out)
        supervisor.add_service(
            "chainlink_onchain",
            {"symbol": onchain_config.symbol, "backend": "onchain", "mode": "public_onchain_price_feed"},
            lambda config=onchain_config: ChainlinkOnchainRecorderService(config, logger=logger),
        )
    else:
        disabled.append("chainlink_onchain")

    if enable_fng and _env_flag("ENABLE_FNG", True):
        fng_config = FearAndGreedRecorderConfig.from_env(out=out)
        supervisor.add_service(
            "fng",
            {"source": "alternative_me", "mode": "public_rest_polling"},
            lambda config=fng_config: FearAndGreedRecorderService(config, logger=logger),
        )
    else:
        disabled.append("fng")

    return supervisor


def _run_supervisor(
    *,
    out: Path,
    log_level: str,
    enable_binance_spot: bool = True,
    enable_binance_usdm: bool = True,
    enable_polymarket_ws: bool = True,
    enable_polymarket_rest: bool = True,
    enable_polymarket_strike: bool = True,
    enable_polymarket_resolution: bool = True,
    enable_polymarket_rtds: bool = True,
    enable_bybit: bool = True,
    enable_hyperliquid: bool = True,
    enable_deribit: bool = True,
    enable_coinbase: bool = True,
    enable_chainlink_live: bool = True,
    enable_chainlink_public_delayed: bool = True,
    enable_chainlink_page: bool | None = None,
    enable_chainlink_onchain: bool = True,
    enable_fng: bool = True,
) -> None:
    rebuild_missing_manifests(root=out, logger=logging.getLogger("market_recorders.recovery"))
    supervisor = _build_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=enable_binance_spot,
        enable_binance_usdm=enable_binance_usdm,
        enable_polymarket_ws=enable_polymarket_ws,
        enable_polymarket_rest=enable_polymarket_rest,
        enable_polymarket_strike=enable_polymarket_strike,
        enable_polymarket_resolution=enable_polymarket_resolution,
        enable_polymarket_rtds=enable_polymarket_rtds,
        enable_bybit=enable_bybit,
        enable_hyperliquid=enable_hyperliquid,
        enable_deribit=enable_deribit,
        enable_coinbase=enable_coinbase,
        enable_chainlink_live=enable_chainlink_live,
        enable_chainlink_public_delayed=enable_chainlink_public_delayed,
        enable_chainlink_page=enable_chainlink_page,
        enable_chainlink_onchain=enable_chainlink_onchain,
        enable_fng=enable_fng,
    )
    previous_handlers: dict[int, Any] = {}

    def _handle_signal(signum: int, _frame: Any) -> None:
        logging.getLogger("market_recorders").info(
            "Received stop signal; requesting graceful shutdown",
            extra={"signal": signum},
        )
        supervisor.request_stop(reason=f"signal_{signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except ValueError:
            continue
    try:
        asyncio.run(supervisor.run())
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except ValueError:
                continue


@app.command("run-all")
def run_all(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(out=out, log_level=log_level)


@app.command("run-core-sources")
def run_core_sources(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=True,
        enable_binance_usdm=False,
        enable_polymarket_ws=True,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=True,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=True,
        enable_chainlink_public_delayed=True,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("run-external-venues")
def run_external_venues(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=True,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=False,
        enable_bybit=True,
        enable_hyperliquid=True,
        enable_deribit=False,
        enable_coinbase=True,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("run-oracle-strike")
def run_oracle_strike(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=True,
        enable_polymarket_strike=True,
        enable_polymarket_resolution=True,
        enable_polymarket_rtds=True,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=True,
        enable_chainlink_public_delayed=True,
        enable_chainlink_onchain=True,
        enable_fng=False,
    )


@app.command("run-binance")
def run_binance(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=True,
        enable_binance_usdm=True,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=False,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("run-polymarket")
def run_polymarket(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=True,
        enable_polymarket_rest=True,
        enable_polymarket_strike=True,
        enable_polymarket_resolution=True,
        enable_polymarket_rtds=True,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=True,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("run-bybit")
def run_bybit(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=False,
        enable_bybit=True,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("run-hyperliquid")
def run_hyperliquid(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=False,
        enable_bybit=False,
        enable_hyperliquid=True,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("run-deribit")
def run_deribit(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=False,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=True,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("run-coinbase")
def run_coinbase(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=False,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=True,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("run-chainlink")
def run_chainlink(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=True,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=True,
        enable_chainlink_public_delayed=True,
        enable_chainlink_onchain=True,
        enable_fng=False,
    )


@app.command("run-fng")
def run_fng(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=False,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=True,
    )


@app.command("run-research-stack")
def run_research_stack(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=True,
        enable_polymarket_ws=False,
        enable_polymarket_rest=True,
        enable_polymarket_strike=True,
        enable_polymarket_resolution=True,
        enable_polymarket_rtds=True,
        enable_bybit=True,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=True,
        enable_chainlink_live=True,
        enable_chainlink_public_delayed=True,
        enable_chainlink_onchain=True,
        enable_fng=True,
    )


@app.command("run-polymarket-rtds")
def run_polymarket_rtds(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=False,
        enable_polymarket_rtds=True,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=True,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("run-polymarket-resolution")
def run_polymarket_resolution(
    out: Path = typer.Option(Path("./data")),
    log_level: str = typer.Option("INFO"),
) -> None:
    _run_supervisor(
        out=out,
        log_level=log_level,
        enable_binance_spot=False,
        enable_binance_usdm=False,
        enable_polymarket_ws=False,
        enable_polymarket_rest=False,
        enable_polymarket_strike=False,
        enable_polymarket_resolution=True,
        enable_polymarket_rtds=False,
        enable_bybit=False,
        enable_hyperliquid=False,
        enable_deribit=False,
        enable_coinbase=False,
        enable_chainlink_live=False,
        enable_chainlink_public_delayed=False,
        enable_chainlink_onchain=False,
        enable_fng=False,
    )


@app.command("qc-day")
def qc_day(
    target_date: str = typer.Option(..., "--date"),
    out: Path = typer.Option(Path("./data")),
) -> None:
    report = build_daily_qc_report(out, parse_utc_date(target_date))
    typer.echo(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode("utf-8"))


@app.command("repair-manifests")
def repair_manifests(
    out: Path = typer.Option(Path("./data")),
) -> None:
    report = rebuild_missing_manifests(root=out, logger=logging.getLogger("market_recorders.recovery"))
    typer.echo(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode("utf-8"))


@app.command("compare-chainlink-rtds")
def compare_chainlink_rtds(
    date_from: str = typer.Option(..., "--date-from"),
    date_to: str = typer.Option(..., "--date-to"),
    out: Path = typer.Option(Path("./data")),
    symbol: str = typer.Option("BTCUSD"),
    rtds_symbol: str = typer.Option("btc/usd"),
    finalized_only: bool = typer.Option(True),
    tail_tolerant: bool = typer.Option(False),
    by_hour: bool = typer.Option(False),
    max_offset_seconds: int = typer.Option(120),
    tolerance_seconds: int = typer.Option(2),
    near_match_abs_error: float = typer.Option(1.0),
    artifact: Path | None = typer.Option(None),
) -> None:
    report = build_chainlink_alignment_report(
        root=out,
        date_from=parse_utc_date(date_from),
        date_to=parse_utc_date(date_to),
        symbol=symbol,
        rtds_symbol=rtds_symbol,
        finalized_only=finalized_only,
        tail_tolerant=tail_tolerant,
        by_hour=by_hour,
        max_offset_seconds=max_offset_seconds,
        tolerance_seconds=tolerance_seconds,
        near_match_abs_error=near_match_abs_error,
    )
    payload = orjson.dumps(report, option=orjson.OPT_INDENT_2)
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(payload)
    typer.echo(payload.decode("utf-8"))


def main() -> None:
    app()
