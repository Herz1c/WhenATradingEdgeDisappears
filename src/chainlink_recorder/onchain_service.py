from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

import httpx

from binance_recorder.utils import ns_to_iso
from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.http_utils import build_async_http_client, request_with_retries
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import build_capture_timing, recv_timestamp_source, utc_now_ns

from .config import ChainlinkOnchainConfig

_SELECTOR_LATEST_ROUND_DATA = "0xfeaf968c"
_SELECTOR_DECIMALS = "0x313ce567"
_SELECTOR_DESCRIPTION = "0x7284e416"
_TRANSIENT_POLL_EXCEPTIONS = (httpx.HTTPError, RuntimeError, ValueError)


def _lifecycle(event: str, config: ChainlinkOnchainConfig, details: dict[str, Any] | None = None) -> dict[str, Any]:
    recv_ts_ns = utc_now_ns()
    payload = {
        "record_type": "lifecycle_event",
        "event": event,
        "source": "chainlink",
        "backend": "onchain",
        "symbol": config.symbol,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
    }
    if details:
        payload.update(details)
    return payload


def _decode_words(hex_value: str) -> list[int]:
    raw = hex_value.removeprefix("0x")
    return [int(raw[index : index + 64], 16) for index in range(0, len(raw), 64) if raw[index : index + 64]]


def _decode_int256(value: int) -> int:
    if value >= 2**255:
        return value - 2**256
    return value


def _decode_abi_string(hex_value: str) -> str | None:
    raw = hex_value.removeprefix("0x")
    if len(raw) < 128:
        return None
    try:
        length = int(raw[64:128], 16)
        text_hex = raw[128 : 128 + (length * 2)]
        return bytes.fromhex(text_hex).decode("utf-8", errors="ignore")
    except Exception:
        return None


class ChainlinkOnchainRecorderService:
    def __init__(self, config: ChainlinkOnchainConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._runtime = build_runtime_metadata(recorder_service="chainlink_onchain", config=config)
        static_fields = self._runtime.manifest_fields({"source": "chainlink", "backend": "onchain", "symbol": config.symbol})
        self._writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="onchain",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
        )
        self._lifecycle_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind="lifecycle",
            flush_interval_seconds=config.flush_interval_seconds,
            manifest_static_fields=static_fields,
        )

    def _advance_writers(self, now_ns: int | None = None) -> None:
        ts_ns = now_ns or utc_now_ns()
        self._writer.advance_time(now_ns=ts_ns)
        self._lifecycle_writer.advance_time(now_ns=ts_ns)

    async def run(self) -> None:
        async with build_async_http_client(
            timeout_seconds=self.config.http_timeout_seconds,
            max_connections=self.config.http_max_connections,
            max_keepalive_connections=self.config.http_max_keepalive_connections,
            keepalive_expiry_seconds=self.config.http_keepalive_expiry_seconds,
            trust_env=self.config.http_trust_env,
        ) as client:
            self._lifecycle_writer.write(_lifecycle("service_start", self.config, {"mode": "public_onchain_price_feed"}))
            try:
                while True:
                    self._advance_writers()
                    try:
                        await self._poll(client)
                    except asyncio.CancelledError:
                        raise
                    except _TRANSIENT_POLL_EXCEPTIONS as exc:
                        self.logger.warning(
                            "Chainlink onchain poll failed",
                            extra={
                                "service_name": "chainlink_onchain",
                                "symbol": self.config.symbol,
                                "backend": "onchain",
                                "mode": "public_onchain_price_feed",
                                "rpc_url": self.config.rpc_url,
                                "error": repr(exc),
                            },
                        )
                        self._lifecycle_writer.write(
                            _lifecycle(
                                "poll_error",
                                self.config,
                                {
                                    "mode": "public_onchain_price_feed",
                                    "rpc_url": self.config.rpc_url,
                                    "error": repr(exc),
                                    "error_type": type(exc).__name__,
                                },
                            )
                        )
                    await asyncio.sleep(self.config.poll_interval_seconds)
            finally:
                self._lifecycle_writer.write(_lifecycle("service_stop", self.config, {"mode": "public_onchain_price_feed"}))
                self._writer.close()
                self._lifecycle_writer.close()

    async def _poll(self, client: httpx.AsyncClient) -> None:
        request_start_ns = utc_now_ns()
        total_request_attempt_count = 0
        block_number_hex, attempt_count = await self._rpc(client, "eth_blockNumber", [])
        total_request_attempt_count += attempt_count
        block, attempt_count = await self._rpc(client, "eth_getBlockByNumber", [block_number_hex, False])
        total_request_attempt_count += attempt_count
        decimals_hex, attempt_count = await self._rpc(
            client,
            "eth_call",
            [{"to": self.config.proxy_address, "data": _SELECTOR_DECIMALS}, "latest"],
        )
        total_request_attempt_count += attempt_count
        description_hex, attempt_count = await self._rpc(
            client, "eth_call", [{"to": self.config.proxy_address, "data": _SELECTOR_DESCRIPTION}, "latest"]
        )
        total_request_attempt_count += attempt_count
        latest_round_hex, attempt_count = await self._rpc(
            client,
            "eth_call",
            [{"to": self.config.proxy_address, "data": _SELECTOR_LATEST_ROUND_DATA}, "latest"],
        )
        total_request_attempt_count += attempt_count
        response_end_ns = utc_now_ns()
        recv_ts_ns = utc_now_ns()
        words = _decode_words(latest_round_hex)
        decimals = int(decimals_hex, 16) if isinstance(decimals_hex, str) else None
        description = _decode_abi_string(description_hex)
        answer = _decode_int256(words[1]) if len(words) > 1 else None
        record = {
            "record_type": "onchain_record",
            "source": "chainlink",
            "backend": "onchain",
            "symbol": self.config.symbol,
            "network": self.config.network,
            "proxy_address": self.config.proxy_address,
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "capture_timing": build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            ),
            "block": {
                "number": int(block_number_hex, 16),
                "number_hex": block_number_hex,
                "hash": block.get("hash") if isinstance(block, dict) else None,
                "timestamp_s": int(block.get("timestamp"), 16) if isinstance(block, dict) and block.get("timestamp") else None,
                "timestamp_iso": ns_to_iso(int(block.get("timestamp"), 16) * 1_000_000_000)
                if isinstance(block, dict) and block.get("timestamp")
                else None,
            },
            "round_data": {
                "round_id": str(words[0]) if len(words) > 0 else None,
                "answer_raw": str(answer) if answer is not None else None,
                "answer": str(Decimal(answer) / (Decimal(10) ** decimals))
                if answer is not None and decimals is not None
                else None,
                "started_at_s": words[2] if len(words) > 2 else None,
                "updated_at_s": words[3] if len(words) > 3 else None,
                "answered_in_round": str(words[4]) if len(words) > 4 else None,
                "decimals": decimals,
                "description": description or None,
            },
        }
        record["capture_timing"]["request_attempt_count"] = total_request_attempt_count
        self._writer.write(record)

    async def _rpc(self, client: httpx.AsyncClient, method: str, params: list[Any]) -> tuple[Any, int]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        outcome = await request_with_retries(
            client,
            "POST",
            self.config.rpc_url,
            max_retries=self.config.http_max_retries,
            retry_backoff_seconds=self.config.http_retry_backoff_seconds,
            json=payload,
        )
        response = outcome.response
        body = response.json()
        if "error" in body:
            raise RuntimeError(body["error"])
        return body.get("result"), outcome.attempt_count
