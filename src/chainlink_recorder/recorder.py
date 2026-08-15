from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Any

import httpx

from binance_recorder.utils import ns_to_iso
from market_recorders.archive import HourlyRotatingJsonlWriter
from market_recorders.runtime_metadata import build_runtime_metadata
from market_recorders.time_utils import build_capture_timing, recv_timestamp_source, utc_now_ns

from .config import ChainlinkRecorderConfig
from .datastreams_auth import datastreams_latest_report_path
from .datastreams_client import DatastreamsApiClient
from .datastreams_ws_client import DatastreamsWsClient
from .live_reference import build_direct_live_record

_TITLE_RE = re.compile(r"<title[^>]*>(?P<title>.*?)</title>", re.IGNORECASE | re.DOTALL)
_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(?P<json>.*?)</script>',
    re.DOTALL,
)


def _lifecycle(
    event: str,
    backend: str,
    config: ChainlinkRecorderConfig,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recv_ts_ns = utc_now_ns()
    payload = {
        "record_type": "lifecycle_event",
        "event": event,
        "source": "chainlink",
        "service": backend,
        "symbol": config.symbol,
        "backend": backend,
        "recv_ts_ns": recv_ts_ns,
        "recv_ts_iso": ns_to_iso(recv_ts_ns),
        "recv_ts_source": recv_timestamp_source(),
    }
    if details:
        payload.update(details)
    return payload


class ChainlinkRecorderService:
    def __init__(self, config: ChainlinkRecorderConfig, *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.backend = config.selected_backend
        if self.backend is None:
            raise RuntimeError(config.availability_message() or "Direct Chainlink live backend is unavailable")
        self.data_kind = "page" if self.backend == "public_page" else "live"
        self._runtime = build_runtime_metadata(recorder_service="chainlink_live", config=config)
        static_fields = self._runtime.manifest_fields({
            "source": "chainlink",
            "service": self.backend,
            "symbol": config.symbol,
            "backend": self.backend,
        })
        self._data_writer = HourlyRotatingJsonlWriter(
            root=config.out,
            namespace=config.namespace,
            kind=self.data_kind,
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
        self._data_writer.advance_time(now_ns=ts_ns)
        self._lifecycle_writer.advance_time(now_ns=ts_ns)

    async def run(self) -> None:
        if self.backend in {"datastreams_api", "datastreams_ws"} and self.config.availability_issue() is not None:
            raise RuntimeError(self.config.availability_message() or self.config.availability_issue())
        if self.backend == "datastreams_ws":
            try:
                await self._run_datastreams_ws()
            finally:
                self._data_writer.close()
                self._lifecycle_writer.close()
            return
        timeout = httpx.Timeout(self.config.http_timeout_seconds)
        headers = {"User-Agent": self.config.user_agent}
        client_kwargs: dict[str, Any] = {"timeout": timeout, "headers": headers}
        if self.backend == "datastreams_api":
            client_kwargs["base_url"] = self.config.datastreams_rest_url
        async with httpx.AsyncClient(**client_kwargs) as client:
            self._lifecycle_writer.write(_lifecycle("service_start", self.backend, self.config, {"mode": self.backend}))
            try:
                if self.backend == "datastreams_api":
                    await self._run_datastreams(client)
                else:
                    await self._run_public_page(client)
            finally:
                self._lifecycle_writer.write(_lifecycle("shutdown", self.backend, self.config, {"mode": self.backend}))
                self._data_writer.close()
                self._lifecycle_writer.close()

    def datastreams_backend_client(self, client: httpx.AsyncClient | None = None) -> Any:
        if self.backend == "datastreams_api":
            if client is None:
                raise ValueError("httpx client is required for datastreams_api backend")
            return DatastreamsApiClient(client, self.config)
        if self.backend == "datastreams_ws":
            return DatastreamsWsClient(
                self.config,
                data_writer=self._data_writer,
                lifecycle_writer=self._lifecycle_writer,
                logger=self.logger,
            )
        raise ValueError(f"Unsupported datastreams backend: {self.backend}")

    async def _run_public_page(self, client: httpx.AsyncClient) -> None:
        delay = self.config.poll_interval_seconds
        while True:
            self._advance_writers()
            try:
                await self._poll_public_page(client)
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._lifecycle_writer.write(_lifecycle("http_error", self.backend, self.config, {"error": repr(exc)}))
                await asyncio.sleep(
                    min(self.config.reconnect_max_seconds, delay)
                    + random.random() * self.config.reconnect_jitter_seconds
                )

    async def _run_datastreams(self, client: httpx.AsyncClient) -> None:
        api_client = self.datastreams_backend_client(client)
        delay = self.config.poll_interval_seconds
        while True:
            self._advance_writers()
            try:
                await self._poll_datastreams(api_client)
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._lifecycle_writer.write(_lifecycle("http_error", self.backend, self.config, {"error": repr(exc)}))
                await asyncio.sleep(
                    min(self.config.reconnect_max_seconds, delay)
                    + random.random() * self.config.reconnect_jitter_seconds
                )

    async def _run_datastreams_ws(self) -> None:
        ws_client = self.datastreams_backend_client()
        await ws_client.run()

    async def _poll_public_page(self, client: httpx.AsyncClient) -> None:
        request_start_ns = utc_now_ns()
        response = await client.get(self.config.page_url)
        response.raise_for_status()
        response_end_ns = utc_now_ns()
        html = response.text
        recv_ts_ns = utc_now_ns()
        next_data_match = _NEXT_DATA_RE.search(html)
        next_data = next_data_match.group("json") if next_data_match else None
        title_match = _TITLE_RE.search(html)
        record = {
            "record_type": "page_observation",
            "source": "chainlink",
            "service": self.backend,
            "symbol": self.config.symbol,
            "backend": self.backend,
            "page_url": self.config.page_url,
            "page_title": title_match.group("title").strip() if title_match else None,
            "recv_ts_ns": recv_ts_ns,
            "recv_ts_iso": ns_to_iso(recv_ts_ns),
            "recv_ts_source": recv_timestamp_source(),
            "capture_timing": build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            ),
            "raw": {"html": html if self.config.store_raw_html else None, "next_data_json": next_data},
            "extracted": {},
        }
        self._data_writer.write(record)

    async def _poll_datastreams(self, api_client: DatastreamsApiClient) -> None:
        request_start_ns = utc_now_ns()
        payload = await api_client.fetch_latest_report()
        response_end_ns = utc_now_ns()
        recv_ts_ns = utc_now_ns()
        path = datastreams_latest_report_path(str(self.config.datastreams_feed_id_btcusd))
        record = build_direct_live_record(
            payload,
            backend="datastreams_api",
            symbol=self.config.symbol,
            recv_ts_ns=recv_ts_ns,
            recv_ts_iso=ns_to_iso(recv_ts_ns),
            recv_ts_source=recv_timestamp_source(),
            endpoint_url=f"{self.config.datastreams_rest_url}{path}",
            capture_timing=build_capture_timing(
                request_start_ns=request_start_ns,
                response_end_ns=response_end_ns,
            ),
        )
        if record is not None:
            self._data_writer.write(record)
