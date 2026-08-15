from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _heartbeat_counter(payload: dict[str, Any]) -> int | None:
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        return None
    first = events[0]
    if not isinstance(first, dict):
        return None
    return _to_int(first.get("heartbeat_counter"))


@dataclass(slots=True)
class CoinbaseContinuityObservation:
    state: str
    requires_reset: bool
    quality_class: str | None
    quality_flags: tuple[str, ...]
    lifecycle_events: list[dict[str, Any]]


class CoinbaseSequenceTracker:
    def __init__(self, *, heartbeat_stale_seconds: float = 5.0) -> None:
        self.heartbeat_stale_ns = int(heartbeat_stale_seconds * 1_000_000_000)
        self.last_sequence_num: int | None = None
        self.last_heartbeat_counter: int | None = None
        self.last_heartbeat_recv_ts_ns: int | None = None

    def reset(self) -> None:
        self.last_sequence_num = None
        self.last_heartbeat_counter = None
        self.last_heartbeat_recv_ts_ns = None

    def observe(self, payload: dict[str, Any], *, recv_ts_ns: int) -> CoinbaseContinuityObservation:
        lifecycle_events: list[dict[str, Any]] = []
        quality_flags: list[str] = []
        quality_class: str | None = None
        requires_reset = False
        state = "healthy"
        channel = str(payload.get("channel") or "")
        sequence_num = _to_int(payload.get("sequence_num"))

        if (
            self.last_heartbeat_recv_ts_ns is not None
            and recv_ts_ns - self.last_heartbeat_recv_ts_ns > self.heartbeat_stale_ns
        ):
            state = "heartbeat_stale"
            requires_reset = True
            quality_class = "continuity_compromised"
            quality_flags.append("heartbeat_stale")
            lifecycle_events.extend(
                (
                    {
                        "event": "heartbeat_stale_detected",
                        "heartbeat_stale_ms": round(
                            (recv_ts_ns - self.last_heartbeat_recv_ts_ns) / 1_000_000,
                            3,
                        ),
                    },
                    {"event": "snapshot_reset_required", "reason": "heartbeat_stale"},
                    {"event": "channel_resubscribe_triggered", "reason": "heartbeat_stale"},
                )
            )

        if channel == "heartbeats":
            heartbeat_counter = _heartbeat_counter(payload)
            if (
                heartbeat_counter is not None
                and self.last_heartbeat_counter is not None
                and heartbeat_counter > self.last_heartbeat_counter + 1
            ):
                state = "heartbeat_gap"
                requires_reset = True
                quality_class = "continuity_compromised"
                quality_flags.append("heartbeat_gap")
                lifecycle_events.extend(
                    (
                        {
                            "event": "heartbeat_gap_detected",
                            "expected_heartbeat_counter": self.last_heartbeat_counter + 1,
                            "received_heartbeat_counter": heartbeat_counter,
                            "missed_heartbeat_count": heartbeat_counter - self.last_heartbeat_counter - 1,
                        },
                        {"event": "snapshot_reset_required", "reason": "heartbeat_gap"},
                        {"event": "channel_resubscribe_triggered", "reason": "heartbeat_gap"},
                    )
                )
            if heartbeat_counter is not None:
                self.last_heartbeat_counter = heartbeat_counter
            self.last_heartbeat_recv_ts_ns = recv_ts_ns

        if sequence_num is not None and sequence_num > 0:
            if self.last_sequence_num is not None:
                if sequence_num > self.last_sequence_num + 1:
                    state = "sequence_gap"
                    requires_reset = True
                    quality_class = "continuity_compromised"
                    quality_flags.append("sequence_gap")
                    lifecycle_events.extend(
                        (
                            {
                                "event": "sequence_gap_detected",
                                "expected_sequence_num": self.last_sequence_num + 1,
                                "received_sequence_num": sequence_num,
                                "gap_size": sequence_num - self.last_sequence_num - 1,
                            },
                            {"event": "snapshot_reset_required", "reason": "sequence_gap"},
                            {"event": "channel_resubscribe_triggered", "reason": "sequence_gap"},
                        )
                    )
                elif sequence_num < self.last_sequence_num:
                    state = "out_of_order"
                    requires_reset = True
                    quality_class = "continuity_compromised"
                    quality_flags.append("out_of_order")
                    lifecycle_events.extend(
                        (
                            {
                                "event": "out_of_order_detected",
                                "last_sequence_num": self.last_sequence_num,
                                "received_sequence_num": sequence_num,
                            },
                            {"event": "snapshot_reset_required", "reason": "out_of_order"},
                            {"event": "channel_resubscribe_triggered", "reason": "out_of_order"},
                        )
                    )
            if self.last_sequence_num is None or sequence_num > self.last_sequence_num:
                self.last_sequence_num = sequence_num

        return CoinbaseContinuityObservation(
            state=state,
            requires_reset=requires_reset,
            quality_class=quality_class,
            quality_flags=tuple(dict.fromkeys(quality_flags)),
            lifecycle_events=lifecycle_events,
        )
