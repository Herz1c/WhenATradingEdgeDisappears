from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import market_recorders.fng_recorder as fng_module
from market_recorders.fng_recorder import FearAndGreedRecorderConfig, FearAndGreedRecorderService


def test_fng_service_continues_after_transient_http_error(monkeypatch, tmp_path) -> None:
    async def run_case() -> int:
        config = FearAndGreedRecorderConfig(
            out=tmp_path,
            poll_interval_seconds=0.01,
            flush_interval_seconds=0.01,
        )
        service = FearAndGreedRecorderService(config)
        success = asyncio.Event()
        calls = {"count": 0}

        async def fake_poll(_client: httpx.AsyncClient) -> None:
            calls["count"] += 1
            if calls["count"] == 1:
                raise httpx.ReadTimeout("boom")
            recv_ts_ns = fng_module.utc_now_ns()
            service._writer.write(
                {
                    "record_type": "rest_observation",
                    "recv_ts_ns": recv_ts_ns,
                }
            )
            success.set()
            await asyncio.sleep(10)

        monkeypatch.setattr(service, "_poll_once", fake_poll)
        task = asyncio.create_task(service.run())
        await asyncio.wait_for(success.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return calls["count"]

    call_count = asyncio.run(run_case())

    manifest_path = next((tmp_path / "manifests").rglob("*.fng.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert call_count >= 2
    assert manifest["finalized"] is True
    assert manifest["record_count"] == 1
