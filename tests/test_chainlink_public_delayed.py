from __future__ import annotations

import json
from typing import Any

import pytest

from chainlink_recorder.config import ChainlinkPublicDelayedConfig
from chainlink_recorder.public_delayed import (
    ChainlinkPublicDelayedRecorderService,
    _parse_page_metadata,
)


class _FakeApiResponse:
    def __init__(self, *, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    async def body(self) -> bytes:
        return self._body


class _FakeApiRequestContext:
    """Stands in for Playwright's `context.request`.

    The recorder issues its data call through the browser context so the
    request carries the Vercel challenge clearance that navigation obtained.
    The parsing logic under test is transport-agnostic, so a fake that records
    the outgoing URL/params and replays a canned body is sufficient.
    """

    def __init__(self, *, status: int = 200, payload: Any = None, raises: Exception | None = None) -> None:
        self.status = status
        self.payload = payload if payload is not None else {}
        self.raises = raises
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def get(self, url: str, *, params: dict[str, str], timeout: float) -> _FakeApiResponse:
        del timeout
        self.calls.append((url, dict(params)))
        if self.raises is not None:
            raise self.raises
        return _FakeApiResponse(status=self.status, body=json.dumps(self.payload).encode())


class _FakeContext:
    def __init__(self, request: _FakeApiRequestContext) -> None:
        self.request = request


class _FakePage:
    """Minimal Playwright page stub returning fixed HTML for any navigation."""

    def __init__(self, html: str) -> None:
        self._html = html
        self.goto_calls: list[str] = []

    async def goto(self, url: str, **kwargs: Any) -> None:
        del kwargs
        self.goto_calls.append(url)

    async def content(self) -> str:
        return self._html


def _service_with_fakes(
    tmp_path,
    *,
    payload: Any = None,
    status: int = 200,
    page_html: str | None = None,
) -> tuple[ChainlinkPublicDelayedRecorderService, _FakeApiRequestContext]:
    config = ChainlinkPublicDelayedConfig(
        out=tmp_path,
        debug_include_html=False,
        # Do not burn wall-clock in the challenge-wait loop under test.
        challenge_max_wait_seconds=0.0,
    )
    service = ChainlinkPublicDelayedRecorderService(config)
    api = _FakeApiRequestContext(status=status, payload=payload)
    service._context = _FakeContext(api)  # type: ignore[assignment]
    service._page = _FakePage(_page_html() if page_html is None else page_html)  # type: ignore[assignment]
    return service, api


def _page_html(*, feed_id: str = "0xfeed", multiply: str = "1000000000000000000") -> str:
    payload = {
        "props": {
            "pageProps": {
                "streamData": {
                    "streamMetadata": {
                        "feedId": feed_id,
                        "multiply": multiply,
                    }
                }
            }
        }
    }
    return (
        "<html><head><title>BTC / USD Data Stream | Chainlink</title></head><body>"
        f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'
        "</body></html>"
    )


def test_parse_public_delayed_page_metadata_extracts_feed_id_and_multiply() -> None:
    metadata = _parse_page_metadata(_page_html(), fetched_at_ns=1, debug_include_html=False)
    assert metadata.feed_id == "0xfeed"
    assert str(metadata.multiply) == "1000000000000000000"
    assert metadata.page_title == "BTC / USD Data Stream | Chainlink"
    assert metadata.html is None


@pytest.mark.asyncio
async def test_public_delayed_fetch_record_success(tmp_path) -> None:
    service, api = _service_with_fakes(
        tmp_path,
        payload={
            "data": {
                "allStreamValuesGenerics": {
                    "nodes": [
                        {
                            "validAfterTs": "2026-04-18T17:38:42+00:00",
                            "valueNumeric": "74120.01",
                            "attributeName": "bid",
                        },
                        {
                            "validAfterTs": "2026-04-18T17:38:43+00:00",
                            "valueNumeric": "74123.45",
                            "attributeName": "benchmark",
                        },
                    ]
                }
            }
        },
    )

    record = await service._fetch_record()

    # The benchmark attribute preserves the old public-delayed mid semantics.
    assert len(api.calls) == 1
    data_url, params = api.calls[0]
    assert data_url.endswith("/api/live-data-engine-stream-data")
    assert params == {
        "feedId": "0xfeed",
        "abiIndex": "0",
        "queryWindow": "1m",
        "attributeName": "benchmark",
    }

    assert record["record_type"] == "chainlink_public_delayed_observation"
    assert record["source"] == "chainlink_public_delayed"
    assert record["backend"] == "public_stream_page"
    assert record["symbol"] == "BTCUSD"
    assert record["feed_id"] == "0xfeed"
    # Latest benchmark node wins over the earlier bid node.
    assert record["chainlink_display_ts"] == 1_776_533_923
    assert record["btc_usd_price"] == "74123.45"
    assert record["data_api"] == "live_data_engine_stream_data"
    assert record["parse_status"] == "success"
    assert record["parse_error"] is None
    assert "capture_timing" in record
    assert "page_hash" in record
    assert "content_hash" in record
    assert "debug" not in record


@pytest.mark.asyncio
async def test_public_delayed_fetch_record_returns_failure_row_for_missing_nodes(tmp_path) -> None:
    service, _api = _service_with_fakes(
        tmp_path,
        payload={"data": {"allStreamValuesGenerics": {"nodes": []}}},
    )

    record = await service._fetch_record()

    assert record["parse_status"] == "missing_live_data_engine_node"
    assert record["btc_usd_price"] is None
    assert record["chainlink_display_ts"] is None
    assert "debug" not in record


@pytest.mark.asyncio
async def test_public_delayed_fetch_record_returns_failure_row_when_data_api_errors(tmp_path) -> None:
    """A 429 means the Vercel challenge came back; metadata must be dropped."""
    service, _api = _service_with_fakes(tmp_path, status=429, payload={})

    record = await service._fetch_record()

    assert record["parse_status"] == "http_error"
    assert record["parse_error"] is not None
    assert "429" in record["parse_error"]
    # Cleared metadata forces a fresh navigation (re-solving the challenge).
    assert service._metadata is None


@pytest.mark.asyncio
async def test_public_delayed_fetch_record_returns_failure_row_for_changed_page_structure(tmp_path) -> None:
    service, api = _service_with_fakes(
        tmp_path,
        page_html="<html><body>missing next data</body></html>",
    )

    record = await service._fetch_record()

    assert record["parse_status"] == "page_metadata_error"
    assert record["parse_error"] is not None
    # Metadata never resolved, so the data API is never reached.
    assert api.calls == []
