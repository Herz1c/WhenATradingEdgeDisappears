import asyncio
import json
import logging

from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.cli import _build_supervisor
from market_recorders.supervisor import RecorderSupervisor


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


async def _run_supervisor_once() -> list[str]:
    logger = logging.getLogger("test_supervisor")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = _CaptureHandler()
    logger.addHandler(handler)
    supervisor = RecorderSupervisor(logger=logger)

    async def service() -> None:
        return None

    supervisor.add_service("demo", {"mode": "test"}, lambda: service())
    await supervisor.run()
    return handler.messages


def test_supervisor_runs_service_without_logrecord_name_collision() -> None:
    messages = asyncio.run(_run_supervisor_once())
    assert "Starting recorder supervisor" in messages
    assert "Service run starting" in messages
    assert "Recorder supervisor stopped" in messages


def test_build_supervisor_registers_new_public_services_by_default(tmp_path) -> None:
    supervisor = _build_supervisor(out=tmp_path, log_level="INFO")
    names = {service.name for service in supervisor._services}
    assert {
        "hyperliquid",
        "deribit",
        "fng",
        "polymarket_resolution",
        "polymarket_rtds",
        "polymarket_strike",
        "chainlink_public_delayed",
    } <= names
    assert "chainlink_live" not in names


def test_supervisor_request_stop_allows_writer_finalization(tmp_path) -> None:
    async def run_case() -> None:
        logger = logging.getLogger("test_supervisor_graceful_stop")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        supervisor = RecorderSupervisor(logger=logger)
        writer = HourlyRotatingJsonlWriter(
            root=tmp_path,
            namespace=("tests", "graceful"),
            kind="ws",
            flush_interval_seconds=0.01,
        )

        class _Service:
            async def run(self) -> None:
                try:
                    writer.write({"record_type": "ws_event", "recv_ts_ns": 1})
                    while True:
                        await asyncio.sleep(1)
                finally:
                    writer.close()

        supervisor.add_service("demo", {"mode": "test"}, lambda: _Service())
        task = asyncio.create_task(supervisor.run())
        await asyncio.sleep(0.05)
        supervisor.request_stop(reason="test_stop")
        await task

    asyncio.run(run_case())
    manifest_path = (
        tmp_path
        / "manifests"
        / "tests"
        / "graceful"
        / "1970-01-01"
        / "00.ws.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["finalized"] is True
    assert manifest["record_count"] == 1
