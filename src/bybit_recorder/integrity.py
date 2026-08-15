from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _topic_family(topic: str) -> str:
    return topic.split(".", 1)[0]


@dataclass(slots=True)
class BybitObservation:
    continuity_state: str
    quality_class: str | None
    quality_flags: tuple[str, ...]
    lifecycle_events: list[dict[str, Any]]


@dataclass(slots=True)
class _TopicState:
    seen_snapshot: bool = False
    last_type: str | None = None
    last_u: int | None = None
    last_seq: int | None = None
    last_cs: int | None = None


@dataclass(slots=True)
class BybitContinuityMonitor:
    _topics: dict[str, _TopicState] = field(default_factory=dict)

    def observe(self, payload: dict[str, Any]) -> BybitObservation:
        topic = payload.get("topic")
        if not isinstance(topic, str) or not topic:
            event_name = payload.get("op") or payload.get("type") or "control"
            return BybitObservation(
                continuity_state="control_row",
                quality_class=None,
                quality_flags=(),
                lifecycle_events=[{"event": "bybit_control_row_observed", "control_type": event_name}],
            )
        state = self._topics.setdefault(topic, _TopicState())
        topic_family = _topic_family(topic)
        snapshot_delta_mode = topic_family in {"orderbook", "tickers"}
        message_type = str(payload.get("type") or "unknown")
        data = payload.get("data", {})
        update_id = _to_int(data.get("u")) if isinstance(data, dict) else None
        seq = _to_int(payload.get("seq")) or (_to_int(data.get("seq")) if isinstance(data, dict) else None)
        cs = _to_int(payload.get("cs"))
        lifecycle_events: list[dict[str, Any]] = []
        quality_class: str | None = None
        quality_flags: list[str] = []
        continuity_state = "healthy"

        if snapshot_delta_mode and message_type == "snapshot":
            continuity_state = "reset_snapshot_transition"
            lifecycle_events.append(
                {
                    "event": "bybit_snapshot_reset_received",
                    "topic": topic,
                    "update_id": update_id,
                    "prior_type": state.last_type,
                    "reason": "service_restart" if update_id == 1 else "snapshot_replace",
                }
            )
            state.seen_snapshot = True
        elif snapshot_delta_mode and message_type == "delta":
            if not state.seen_snapshot:
                continuity_state = "continuity_uncertain"
                quality_class = "continuity_compromised"
                quality_flags.append("delta_before_snapshot")
                lifecycle_events.append(
                    {
                        "event": "bybit_sequence_continuity_break",
                        "topic": topic,
                        "reason": "delta_before_snapshot",
                    }
                )
            if update_id is not None and state.last_u is not None and update_id <= state.last_u:
                continuity_state = "continuity_uncertain"
                quality_class = "continuity_compromised"
                quality_flags.append("update_id_regression")
                lifecycle_events.append(
                    {
                        "event": "bybit_sequence_continuity_break",
                        "topic": topic,
                        "reason": "update_id_regression",
                        "last_update_id": state.last_u,
                        "received_update_id": update_id,
                    }
                )
        if seq is not None and state.last_seq is not None and seq <= state.last_seq:
            continuity_state = "continuity_uncertain"
            quality_class = "continuity_compromised"
            quality_flags.append("seq_regression")
            lifecycle_events.append(
                {
                    "event": "bybit_sequence_continuity_break",
                    "topic": topic,
                    "reason": "seq_regression",
                    "last_seq": state.last_seq,
                    "received_seq": seq,
                }
            )
        if cs is not None and state.last_cs is not None and cs <= state.last_cs:
            continuity_state = "continuity_uncertain"
            quality_class = "continuity_compromised"
            quality_flags.append("cs_regression")
            lifecycle_events.append(
                {
                    "event": "bybit_sequence_continuity_break",
                    "topic": topic,
                    "reason": "cs_regression",
                    "last_cs": state.last_cs,
                    "received_cs": cs,
                }
            )
        if snapshot_delta_mode:
            if message_type not in {"snapshot", "delta"}:
                continuity_state = "control_row"
                lifecycle_events.append(
                    {
                        "event": "bybit_control_row_observed",
                        "topic": topic,
                        "message_type": message_type,
                    }
                )
        elif message_type not in {"snapshot", "delta"}:
            continuity_state = "control_row"
            lifecycle_events.append(
                {
                    "event": "bybit_control_row_observed",
                    "topic": topic,
                    "message_type": message_type,
                }
            )

        if update_id is not None:
            state.last_u = update_id
        if seq is not None:
            state.last_seq = seq
        if cs is not None:
            state.last_cs = cs
        state.last_type = message_type

        return BybitObservation(
            continuity_state=continuity_state,
            quality_class=quality_class,
            quality_flags=tuple(dict.fromkeys(quality_flags)),
            lifecycle_events=lifecycle_events,
        )
