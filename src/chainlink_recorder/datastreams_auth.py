from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


EMPTY_BODY_SHA256 = hashlib.sha256(b"").hexdigest()


def build_auth_headers(
    *,
    api_key: str | None,
    api_secret: str | None,
    method: str,
    full_path: str,
    body: bytes = b"",
    timestamp_ms: int | None = None,
) -> dict[str, str]:
    if not (api_key and api_secret):
        return {}
    body_hash = hashlib.sha256(body).hexdigest()
    timestamp_ms = timestamp_ms or int(time.time() * 1000)
    string_to_sign = f"{method.upper()} {full_path} {body_hash} {api_key} {timestamp_ms}"
    signature = hmac.new(
        api_secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "Authorization": api_key,
        "X-Authorization-Timestamp": str(timestamp_ms),
        "X-Authorization-Signature-SHA256": signature,
    }


def datastreams_latest_report_path(feed_id: str) -> str:
    return f"/api/v1/reports/latest?feedID={feed_id}"


def datastreams_websocket_url(base_url: str, *, feed_ids: list[str] | tuple[str, ...]) -> tuple[str, str]:
    parsed = urlparse(base_url)
    path = parsed.path if parsed.path not in {"", "/"} else "/api/v1/ws"
    query_items: list[tuple[str, Any]] = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)]
    query_without_feed_ids = [(key, value) for key, value in query_items if key != "feedIDs"]
    query_without_feed_ids.append(("feedIDs", ",".join(feed_ids)))
    query = urlencode(query_without_feed_ids, safe=",")
    full_path = path if not query else f"{path}?{query}"
    ws_url = urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, query, parsed.fragment))
    return ws_url, full_path
