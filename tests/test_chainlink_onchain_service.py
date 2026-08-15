from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import chainlink_recorder.onchain_service as onchain_module
from binance_recorder.compression import iter_zstd_jsonl
from chainlink_recorder.config import ChainlinkOnchainConfig
from chainlink_recorder.onchain_service import ChainlinkOnchainRecorderService


def test_onchain_service_continues_after_transient_http_error(monkeypatch, tmp_path) -> None:
    async def run_case() -> int:
        config = ChainlinkOnchainConfig(
            out=tmp_path,
            poll_interval_seconds=0.01,
            flush_interval_seconds=0.01,
        )
        service = ChainlinkOnchainRecorderService(config)
        success = asyncio.Event()
        calls = {"count": 0}

        async def fake_poll(_client: httpx.AsyncClient) -> None:
            calls["count"] += 1
            if calls["count"] == 1:
                raise httpx.ReadError("boom")
            recv_ts_ns = onchain_module.utc_now_ns()
            service._writer.write(
                {
                    "record_type": "onchain_record",
                    "recv_ts_ns": recv_ts_ns,
                }
            )
            success.set()
            await asyncio.sleep(10)

        monkeypatch.setattr(service, "_poll", fake_poll)
        task = asyncio.create_task(service.run())
        await asyncio.wait_for(success.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return calls["count"]

    call_count = asyncio.run(run_case())

    onchain_manifest_path = next((tmp_path / "manifests").rglob("*.onchain.manifest.json"))
    lifecycle_manifest_path = next((tmp_path / "manifests").rglob("*.lifecycle.manifest.json"))
    onchain_manifest = json.loads(onchain_manifest_path.read_text(encoding="utf-8"))
    lifecycle_manifest = json.loads(lifecycle_manifest_path.read_text(encoding="utf-8"))
    lifecycle_path = next((tmp_path / "raw").rglob("*.lifecycle.jsonl.zst"))
    lifecycle_events = [record["event"] for record in iter_zstd_jsonl(lifecycle_path)]

    assert call_count >= 2
    assert onchain_manifest["finalized"] is True
    assert onchain_manifest["record_count"] == 1
    assert lifecycle_manifest["finalized"] is True
    assert "service_start" in lifecycle_events
    assert "poll_error" in lifecycle_events
    assert "service_stop" in lifecycle_events
