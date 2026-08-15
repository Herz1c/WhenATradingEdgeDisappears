from __future__ import annotations

import asyncio
from collections.abc import Collection
from dataclasses import dataclass

import httpx

DEFAULT_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524})
RETRYABLE_HTTP_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
)


def build_async_http_client(
    *,
    timeout_seconds: float,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    max_connections: int = 20,
    max_keepalive_connections: int = 5,
    keepalive_expiry_seconds: float = 5.0,
    trust_env: bool = False,
) -> httpx.AsyncClient:
    timeout = httpx.Timeout(timeout_seconds)
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=keepalive_expiry_seconds,
    )
    kwargs: dict[str, object] = {
        "timeout": timeout,
        "follow_redirects": True,
        "limits": limits,
        "trust_env": trust_env,
    }
    if base_url is not None:
        kwargs["base_url"] = base_url
    if headers:
        kwargs["headers"] = headers
    return httpx.AsyncClient(**kwargs)


@dataclass(slots=True)
class RetryingHttpResponse:
    response: httpx.Response
    attempt_count: int


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int = 2,
    retry_backoff_seconds: float = 0.5,
    retry_status_codes: Collection[int] = DEFAULT_RETRYABLE_STATUS_CODES,
    **kwargs: object,
) -> RetryingHttpResponse:
    last_error: Exception | None = None
    attempts = max(1, max_retries + 1)
    for attempt_index in range(attempts):
        attempt_count = attempt_index + 1
        try:
            response = await client.request(method, url, **kwargs)
        except RETRYABLE_HTTP_EXCEPTIONS as exc:
            last_error = exc
            if attempt_count >= attempts:
                raise
            await asyncio.sleep(retry_backoff_seconds * attempt_count)
            continue
        if response.status_code in retry_status_codes and attempt_count < attempts:
            await response.aclose()
            await asyncio.sleep(retry_backoff_seconds * attempt_count)
            continue
        response.raise_for_status()
        return RetryingHttpResponse(response=response, attempt_count=attempt_count)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{method} {url} exhausted retries without a response")


@dataclass(slots=True)
class EndpointBackoffDecision:
    current_state: str
    lifecycle_events: list[dict[str, object]]
    sleep_seconds: float
    skip_request: bool


@dataclass(slots=True)
class _EndpointState:
    current_state: str = "healthy"
    consecutive_failures: int = 0
    suspended_until_ns: int = 0


class HttpEndpointBackoffRegistry:
    def __init__(
        self,
        *,
        degraded_failure_threshold: int = 2,
        suspended_failure_threshold: int = 4,
        degraded_backoff_seconds: float = 5.0,
        suspended_backoff_seconds: float = 30.0,
    ) -> None:
        self.degraded_failure_threshold = max(1, degraded_failure_threshold)
        self.suspended_failure_threshold = max(self.degraded_failure_threshold, suspended_failure_threshold)
        self.degraded_backoff_seconds = degraded_backoff_seconds
        self.suspended_backoff_seconds = suspended_backoff_seconds
        self._states: dict[str, _EndpointState] = {}

    def before_request(self, endpoint: str, *, now_ns: int) -> EndpointBackoffDecision:
        state = self._states.setdefault(endpoint, _EndpointState())
        if state.current_state == "suspended" and now_ns < state.suspended_until_ns:
            remaining_seconds = max((state.suspended_until_ns - now_ns) / 1_000_000_000, 0.0)
            return EndpointBackoffDecision(
                current_state=state.current_state,
                lifecycle_events=[
                    {
                        "event": "http_endpoint_request_skipped",
                        "endpoint": endpoint,
                        "http_endpoint_state": state.current_state,
                        "remaining_suspend_seconds": round(remaining_seconds, 3),
                    }
                ],
                sleep_seconds=min(remaining_seconds, self.suspended_backoff_seconds),
                skip_request=True,
            )
        return EndpointBackoffDecision(
            current_state=state.current_state,
            lifecycle_events=[],
            sleep_seconds=0.0,
            skip_request=False,
        )

    def observe_success(self, endpoint: str) -> EndpointBackoffDecision:
        state = self._states.setdefault(endpoint, _EndpointState())
        previous_state = state.current_state
        state.consecutive_failures = 0
        state.suspended_until_ns = 0
        state.current_state = "healthy"
        lifecycle_events: list[dict[str, object]] = []
        if previous_state in {"degraded", "suspended"}:
            lifecycle_events.extend(
                [
                    {
                        "event": "http_endpoint_state_changed",
                        "endpoint": endpoint,
                        "previous_state": previous_state,
                        "new_state": "recovered",
                    },
                    {
                        "event": "http_endpoint_state_changed",
                        "endpoint": endpoint,
                        "previous_state": "recovered",
                        "new_state": "healthy",
                    },
                ]
            )
        return EndpointBackoffDecision(
            current_state=state.current_state,
            lifecycle_events=lifecycle_events,
            sleep_seconds=0.0,
            skip_request=False,
        )

    def observe_failure(
        self,
        endpoint: str,
        *,
        now_ns: int,
        status_code: int | None,
        reason: str,
    ) -> EndpointBackoffDecision:
        state = self._states.setdefault(endpoint, _EndpointState())
        state.consecutive_failures += 1
        previous_state = state.current_state
        next_state = state.current_state
        sleep_seconds = self.degraded_backoff_seconds
        if status_code == 429 or state.consecutive_failures >= self.suspended_failure_threshold:
            next_state = "suspended"
            sleep_seconds = self.suspended_backoff_seconds
            state.suspended_until_ns = now_ns + int(self.suspended_backoff_seconds * 1_000_000_000)
        elif state.consecutive_failures >= self.degraded_failure_threshold:
            next_state = "degraded"
        state.current_state = next_state
        lifecycle_events: list[dict[str, object]] = []
        if next_state != previous_state:
            lifecycle_events.append(
                {
                    "event": "http_endpoint_state_changed",
                    "endpoint": endpoint,
                    "previous_state": previous_state,
                    "new_state": next_state,
                    "reason": reason,
                    "status_code": status_code,
                    "consecutive_failures": state.consecutive_failures,
                }
            )
        return EndpointBackoffDecision(
            current_state=state.current_state,
            lifecycle_events=lifecycle_events,
            sleep_seconds=sleep_seconds,
            skip_request=False,
        )
