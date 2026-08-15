from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HyperliquidRestHealthTransition:
    current_state: str
    lifecycle_events: list[dict[str, object]]
    sleep_seconds: float
    skip_cycle: bool


class HyperliquidRestCircuitBreaker:
    def __init__(
        self,
        *,
        degraded_failure_threshold: int = 2,
        suspended_failure_threshold: int = 4,
        degraded_backoff_seconds: float = 15.0,
        suspended_backoff_seconds: float = 60.0,
    ) -> None:
        self.degraded_failure_threshold = max(1, degraded_failure_threshold)
        self.suspended_failure_threshold = max(self.degraded_failure_threshold, suspended_failure_threshold)
        self.degraded_backoff_seconds = degraded_backoff_seconds
        self.suspended_backoff_seconds = suspended_backoff_seconds
        self.current_state = "healthy"
        self.consecutive_failures = 0
        self.suspended_until_ns = 0

    def before_cycle(self, *, now_ns: int) -> HyperliquidRestHealthTransition:
        if self.current_state == "suspended" and now_ns < self.suspended_until_ns:
            remaining_seconds = max((self.suspended_until_ns - now_ns) / 1_000_000_000, 0.0)
            return HyperliquidRestHealthTransition(
                current_state=self.current_state,
                lifecycle_events=[
                    {
                        "event": "rest_cycle_skipped",
                        "rest_health_state": self.current_state,
                        "remaining_suspend_seconds": round(remaining_seconds, 3),
                    }
                ],
                sleep_seconds=min(remaining_seconds, self.suspended_backoff_seconds),
                skip_cycle=True,
            )
        return HyperliquidRestHealthTransition(
            current_state=self.current_state,
            lifecycle_events=[],
            sleep_seconds=0.0,
            skip_cycle=False,
        )

    def observe_success(self) -> HyperliquidRestHealthTransition:
        previous_state = self.current_state
        self.consecutive_failures = 0
        self.suspended_until_ns = 0
        if previous_state in {"degraded", "suspended"}:
            self.current_state = "healthy"
            return HyperliquidRestHealthTransition(
                current_state="healthy",
                lifecycle_events=[
                    {
                        "event": "rest_health_state_changed",
                        "previous_state": previous_state,
                        "new_state": "recovered",
                    },
                    {
                        "event": "rest_health_state_changed",
                        "previous_state": "recovered",
                        "new_state": "healthy",
                    },
                ],
                sleep_seconds=0.0,
                skip_cycle=False,
            )
        self.current_state = "healthy"
        return HyperliquidRestHealthTransition(
            current_state=self.current_state,
            lifecycle_events=[],
            sleep_seconds=0.0,
            skip_cycle=False,
        )

    def observe_failure(
        self,
        *,
        now_ns: int,
        status_code: int | None,
        reason: str,
    ) -> HyperliquidRestHealthTransition:
        self.consecutive_failures += 1
        previous_state = self.current_state
        next_state = self.current_state
        sleep_seconds = self.degraded_backoff_seconds
        if status_code == 429 or self.consecutive_failures >= self.suspended_failure_threshold:
            next_state = "suspended"
            sleep_seconds = self.suspended_backoff_seconds
            self.suspended_until_ns = now_ns + int(self.suspended_backoff_seconds * 1_000_000_000)
        elif self.consecutive_failures >= self.degraded_failure_threshold:
            next_state = "degraded"
        self.current_state = next_state
        lifecycle_events: list[dict[str, object]] = []
        if next_state != previous_state:
            lifecycle_events.append(
                {
                    "event": "rest_health_state_changed",
                    "previous_state": previous_state,
                    "new_state": next_state,
                    "reason": reason,
                    "status_code": status_code,
                    "consecutive_failures": self.consecutive_failures,
                }
            )
        return HyperliquidRestHealthTransition(
            current_state=self.current_state,
            lifecycle_events=lifecycle_events,
            sleep_seconds=sleep_seconds,
            skip_cycle=False,
        )
