import asyncio

import httpx
import pytest

from binance_recorder.compression import iter_zstd_jsonl
from binance_recorder.usdm_config import BinanceUsdmRecorderConfig
from binance_recorder.usdm_service import BinanceUsdmRecorderService


@pytest.mark.asyncio
async def test_binance_usdm_rest_loop_recovers_after_error(tmp_path, monkeypatch) -> None:
    config = BinanceUsdmRecorderConfig.from_env(out=tmp_path)
    service = BinanceUsdmRecorderService(config)
    calls = {"server_time": 0, "depth": 0, "sleep": 0}

    async def fake_fetch_server_time(_client):
        calls["server_time"] += 1
        if calls["server_time"] == 1:
            raise httpx.ReadTimeout("temporary")

    async def fake_fetch_depth_snapshot(_client):
        calls["depth"] += 1

    async def fake_fetch_premium_index(_client):
        return None

    async def fake_fetch_open_interest(_client):
        return None

    async def fake_fetch_klines(_client):
        return None

    async def fake_sleep(_seconds: float) -> None:
        calls["sleep"] += 1
        if calls["sleep"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(service, "_fetch_server_time", fake_fetch_server_time)
    monkeypatch.setattr(service, "_fetch_depth_snapshot", fake_fetch_depth_snapshot)
    monkeypatch.setattr(service, "_fetch_premium_index", fake_fetch_premium_index)
    monkeypatch.setattr(service, "_fetch_open_interest", fake_fetch_open_interest)
    monkeypatch.setattr(service, "_fetch_klines", fake_fetch_klines)
    monkeypatch.setattr("binance_recorder.usdm_service.asyncio.sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await service._rest_loop(client=None)  # type: ignore[arg-type]

    service._ws_writer.close()
    service._rest_writer.close()
    service._lifecycle_writer.close()

    lifecycle_path = next((tmp_path / "raw").rglob("*.lifecycle.jsonl.zst"))
    events = [record["event"] for record in iter_zstd_jsonl(lifecycle_path)]
    assert "rest_loop_started" in events
    assert "rest_error" in events
    assert calls["server_time"] >= 2
    assert calls["depth"] >= 1
