from __future__ import annotations

import asyncio
import json

import pytest

import polymarket_recorder.resolution_recorder as resolution_module
from market_recorders.asyncio_utils import cancel_and_await
from polymarket_recorder.config import PolymarketRecorderConfig
from polymarket_recorder.resolution_recorder import PolymarketResolutionRecorderService


def test_cancel_and_await_allows_sync_cleanup_before_cancellation_propagates() -> None:
    events: list[str] = []

    async def child() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            events.append("child_cleanup")

    async def parent() -> None:
        task = asyncio.create_task(child())
        await asyncio.sleep(0)
        try:
            await asyncio.sleep(10)
        finally:
            cancelled_during_cleanup = await cancel_and_await(task)
            events.append("parent_cleanup")
            if cancelled_during_cleanup:
                raise asyncio.CancelledError

    async def run_case() -> None:
        task = asyncio.create_task(parent())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_case())
    assert events == ["child_cleanup", "parent_cleanup"]


def test_resolution_service_finalizes_manifests_on_cancel(monkeypatch, tmp_path) -> None:
    async def _sleep_forever(*_args, **_kwargs) -> None:
        while True:
            await asyncio.sleep(1)

    async def _discover_once(self, _discovery_client, *, poll_seq: int) -> None:
        self._write_lifecycle("discover_stub", {"poll_seq": poll_seq})

    async def _poll_rest_once(self, _discovery_client, *, page_client=None) -> None:
        recv_ts_ns = resolution_module.utc_now_ns()
        self._history_writer.write(
            {
                "record_type": "market_resolution_state_observation",
                "recv_ts_ns": recv_ts_ns,
            }
        )
        await asyncio.sleep(10)

    monkeypatch.setattr(PolymarketResolutionRecorderService, "_discover_once", _discover_once)
    monkeypatch.setattr(PolymarketResolutionRecorderService, "_poll_rest_once", _poll_rest_once)
    monkeypatch.setattr(PolymarketResolutionRecorderService, "_discovery_loop", _sleep_forever)
    monkeypatch.setattr(PolymarketResolutionRecorderService, "_rest_state_loop", _sleep_forever)
    monkeypatch.setattr(PolymarketResolutionRecorderService, "_websocket_loop", _sleep_forever)

    async def run_case() -> None:
        service = PolymarketResolutionRecorderService(PolymarketRecorderConfig.from_env(out=tmp_path))
        task = asyncio.create_task(service.run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_case())

    history_manifest_path = next((tmp_path / "manifests").rglob("*.history.manifest.json"))
    lifecycle_manifest_path = next((tmp_path / "manifests").rglob("*.lifecycle.manifest.json"))
    history_manifest = json.loads(history_manifest_path.read_text(encoding="utf-8"))
    lifecycle_manifest = json.loads(lifecycle_manifest_path.read_text(encoding="utf-8"))

    assert history_manifest["finalized"] is True
    assert lifecycle_manifest["finalized"] is True
