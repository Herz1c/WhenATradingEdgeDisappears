from __future__ import annotations

from typing import Any

import httpx

from .config import ChainlinkRecorderConfig
from .datastreams_auth import build_auth_headers, datastreams_latest_report_path


class DatastreamsApiClient:
    def __init__(self, client: httpx.AsyncClient, config: ChainlinkRecorderConfig) -> None:
        self.client = client
        self.config = config

    def _headers(self, path: str) -> dict[str, str]:
        return build_auth_headers(
            api_key=self.config.datastreams_api_key,
            api_secret=self.config.datastreams_api_secret,
            method="GET",
            full_path=path,
        )

    async def fetch_latest_report(self) -> dict[str, Any]:
        feed_id = self.config.datastreams_feed_id_btcusd
        path = datastreams_latest_report_path(str(feed_id))
        response = await self.client.get(path, headers=self._headers(path))
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload.setdefault("request_metadata", {})
            payload["request_metadata"]["rest_url"] = self.config.datastreams_rest_url
            payload["request_metadata"]["ws_url"] = self.config.datastreams_ws_url
        return payload if isinstance(payload, dict) else {"response": payload}
