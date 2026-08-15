from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower_value = values[lower_index]
    upper_value = values[upper_index]
    if lower_index == upper_index:
        return lower_value
    weight = rank - lower_index
    return lower_value + ((upper_value - lower_value) * weight)


@dataclass(slots=True)
class RtdsQcThresholds:
    freshness_spike_ms: float = 10_000.0
    silent_gap_ms: float = 10_000.0
    burst_followup_gap_ms: float = 1_500.0
    burst_freshness_improvement_ms: float = 5_000.0
    burst_confirmation_ticks: int = 2


def compute_rtds_metrics(
    *,
    recv_ts_ns: int,
    chainlink_ts_ms: int,
    prev_recv_ts_ns: int | None = None,
    prev_chainlink_ts_ms: int | None = None,
    prev_freshness_ms: float | None = None,
) -> dict[str, float | None]:
    freshness_ms = round((recv_ts_ns - (chainlink_ts_ms * 1_000_000)) / 1_000_000, 3)
    recv_gap_ms = (
        round((recv_ts_ns - prev_recv_ts_ns) / 1_000_000, 3)
        if prev_recv_ts_ns is not None
        else None
    )
    chainlink_gap_ms = (
        round(float(chainlink_ts_ms - prev_chainlink_ts_ms), 3)
        if prev_chainlink_ts_ms is not None
        else None
    )
    freshness_delta_ms = (
        round(freshness_ms - prev_freshness_ms, 3)
        if prev_freshness_ms is not None
        else None
    )
    return {
        "recv_minus_chainlink_ms": freshness_ms,
        "recv_gap_ms": recv_gap_ms,
        "chainlink_gap_ms": chainlink_gap_ms,
        "freshness_delta_ms": freshness_delta_ms,
    }


class RtdsBacklogAnalyzer:
    def __init__(self, thresholds: RtdsQcThresholds | None = None) -> None:
        self.thresholds = thresholds or RtdsQcThresholds()
        self._last_recv_ts_ns: int | None = None
        self._last_chainlink_ts_ms: int | None = None
        self._last_freshness_ms: float | None = None
        self._pending_gap: dict[str, Any] | None = None

    def observe(self, record: dict[str, Any]) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
        recv_ts_ns = _safe_int(record.get("recv_ts_ns"))
        chainlink_ts_ms = _safe_int(record.get("chainlink_ts_ms"))
        if recv_ts_ns is None or chainlink_ts_ms is None:
            return {}, []

        metrics = compute_rtds_metrics(
            recv_ts_ns=recv_ts_ns,
            chainlink_ts_ms=chainlink_ts_ms,
            prev_recv_ts_ns=self._last_recv_ts_ns,
            prev_chainlink_ts_ms=self._last_chainlink_ts_ms,
            prev_freshness_ms=self._last_freshness_ms,
        )
        freshness_ms = metrics.get("recv_minus_chainlink_ms")
        recv_gap_ms = metrics.get("recv_gap_ms")
        events: list[dict[str, Any]] = []

        if freshness_ms is not None and freshness_ms >= self.thresholds.freshness_spike_ms:
            if self._last_freshness_ms is None or self._last_freshness_ms < self.thresholds.freshness_spike_ms:
                events.append(
                    {
                        "event": "freshness_spike_detected",
                        "recv_minus_chainlink_ms": freshness_ms,
                        "threshold_ms": self.thresholds.freshness_spike_ms,
                    }
                )

        if recv_gap_ms is not None and recv_gap_ms >= self.thresholds.silent_gap_ms:
            self._pending_gap = {
                "recv_gap_ms": recv_gap_ms,
                "start_freshness_ms": freshness_ms,
                "best_freshness_improvement_ms": 0.0,
                "confirmation_ticks": 0,
                "burst_emitted": False,
            }
            events.append(
                {
                    "event": "silent_gap_detected",
                    "recv_gap_ms": recv_gap_ms,
                    "threshold_ms": self.thresholds.silent_gap_ms,
                    "recv_minus_chainlink_ms": freshness_ms,
                }
            )
        elif self._pending_gap is not None and recv_gap_ms is not None:
            improvement_ms = None
            start_freshness_ms = self._pending_gap.get("start_freshness_ms")
            if start_freshness_ms is not None and freshness_ms is not None:
                improvement_ms = round(float(start_freshness_ms) - float(freshness_ms), 3)
            if (
                improvement_ms is not None
                and improvement_ms >= self.thresholds.burst_freshness_improvement_ms
                and recv_gap_ms <= self.thresholds.burst_followup_gap_ms
            ):
                self._pending_gap["confirmation_ticks"] += 1
                self._pending_gap["best_freshness_improvement_ms"] = max(
                    float(self._pending_gap["best_freshness_improvement_ms"]),
                    improvement_ms,
                )
                if (
                    self._pending_gap["confirmation_ticks"] >= self.thresholds.burst_confirmation_ticks
                    and not self._pending_gap["burst_emitted"]
                ):
                    events.append(
                        {
                            "event": "backlog_burst_detected",
                            "preceding_silent_gap_ms": self._pending_gap["recv_gap_ms"],
                            "recv_gap_ms": recv_gap_ms,
                            "recv_minus_chainlink_ms": freshness_ms,
                            "freshness_improvement_ms": improvement_ms,
                            "confirmation_ticks": self._pending_gap["confirmation_ticks"],
                        }
                    )
                    self._pending_gap["burst_emitted"] = True
            elif recv_gap_ms > self.thresholds.burst_followup_gap_ms * 2:
                self._pending_gap = None

        self._last_recv_ts_ns = recv_ts_ns
        self._last_chainlink_ts_ms = chainlink_ts_ms
        self._last_freshness_ms = freshness_ms
        return metrics, events


def summarize_rtds_stream(
    records: list[dict[str, Any]],
    *,
    thresholds: RtdsQcThresholds | None = None,
) -> dict[str, Any]:
    analyzer = RtdsBacklogAnalyzer(thresholds)
    freshness_values: list[float] = []
    recv_gap_values: list[float] = []
    chainlink_gap_values: list[float] = []
    events = Counter[str]()

    for record in sorted(records, key=lambda item: _safe_int(item.get("recv_ts_ns")) or 0):
        metrics, emitted_events = analyzer.observe(record)
        freshness_ms = metrics.get("recv_minus_chainlink_ms")
        recv_gap_ms = metrics.get("recv_gap_ms")
        chainlink_gap_ms = metrics.get("chainlink_gap_ms")
        if freshness_ms is not None:
            freshness_values.append(float(freshness_ms))
        if recv_gap_ms is not None:
            recv_gap_values.append(float(recv_gap_ms))
        if chainlink_gap_ms is not None:
            chainlink_gap_values.append(float(chainlink_gap_ms))
        for event in emitted_events:
            events[str(event["event"])] += 1

    sorted_freshness = sorted(freshness_values)
    sorted_recv_gaps = sorted(recv_gap_values)
    sorted_chainlink_gaps = sorted(chainlink_gap_values)
    warning_count = sum(events.values())
    return {
        "sample_count": len(records),
        "freshness_ms_p50": _percentile(sorted_freshness, 0.5),
        "freshness_ms_p95": _percentile(sorted_freshness, 0.95),
        "freshness_ms_max": max(sorted_freshness) if sorted_freshness else None,
        "recv_gap_ms_p95": _percentile(sorted_recv_gaps, 0.95),
        "recv_gap_ms_max": max(sorted_recv_gaps) if sorted_recv_gaps else None,
        "chainlink_gap_ms_p95": _percentile(sorted_chainlink_gaps, 0.95),
        "chainlink_gap_ms_max": max(sorted_chainlink_gaps) if sorted_chainlink_gaps else None,
        "warning_event_counts": dict(events),
        "warning_count": warning_count,
        "status": "warn" if warning_count else "clean",
    }
