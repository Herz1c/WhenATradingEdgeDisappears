from __future__ import annotations

import httpx

from market_recorders.http_utils import HttpEndpointBackoffRegistry, request_with_retries


async def test_request_with_retries_retries_retryable_status_and_returns_attempt_count() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(503, request=request, json={"error": "busy"})
        return httpx.Response(200, request=request, json={"ok": True})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://example.test", transport=transport) as client:
        outcome = await request_with_retries(
            client,
            "GET",
            "/ping",
            max_retries=2,
            retry_backoff_seconds=0.0,
        )

    assert calls["count"] == 3
    assert outcome.attempt_count == 3
    assert outcome.response.json() == {"ok": True}


def test_http_endpoint_backoff_registry_transitions_through_degraded_and_suspended() -> None:
    registry = HttpEndpointBackoffRegistry(
        degraded_failure_threshold=2,
        suspended_failure_threshold=3,
        degraded_backoff_seconds=5.0,
        suspended_backoff_seconds=30.0,
    )

    first = registry.observe_failure("gamma_discovery", now_ns=1, status_code=500, reason="HTTPStatusError")
    second = registry.observe_failure("gamma_discovery", now_ns=2, status_code=500, reason="HTTPStatusError")
    third = registry.observe_failure("gamma_discovery", now_ns=3, status_code=429, reason="HTTPStatusError")
    gate = registry.before_request("gamma_discovery", now_ns=10)
    recovery = registry.observe_success("gamma_discovery")

    assert first.current_state == "healthy"
    assert second.current_state == "degraded"
    assert second.lifecycle_events[0]["new_state"] == "degraded"
    assert third.current_state == "suspended"
    assert third.lifecycle_events[0]["new_state"] == "suspended"
    assert gate.skip_request is True
    assert recovery.current_state == "healthy"
    assert [event["new_state"] for event in recovery.lifecycle_events] == ["recovered", "healthy"]
