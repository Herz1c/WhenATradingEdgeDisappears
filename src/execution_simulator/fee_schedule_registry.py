from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class FeeScheduleEntry:
    schedule_id: str
    effective_from: datetime
    effective_to: datetime | None
    asset_class: str
    fee_rate: float
    exponent: int
    maker_fee_rate: float
    maker_rebate_pct: float
    source: str

    def taker_fee(self, *, price: float, size: float) -> float:
        p = float(price)
        if p < 0.0 or p > 1.0:
            raise ValueError("Polymarket fee price must be in [0, 1].")
        return float(size) * float(self.fee_rate) * ((p * (1.0 - p)) ** int(self.exponent))

    def compute_fee(self, *, size: float, price: float) -> float:
        return self.taker_fee(price=price, size=size)

    def maker_fee(self, *, price: float, size: float) -> float:
        del price, size
        return 0.0

    def maker_rebate(self, *, taker_fee: float) -> float:
        return float(taker_fee) * float(self.maker_rebate_pct) / 100.0


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


class FeeScheduleRegistry:
    def __init__(self, path: str | Path = "config/fee_schedules/fee_schedule_registry.yaml") -> None:
        self.path = Path(path)
        self.entries = self._load()
        self._validate_no_overlap()

    @property
    def schedules(self) -> dict[str, FeeScheduleEntry]:
        return {entry.schedule_id: entry for entry in self.entries}

    def _load(self) -> list[FeeScheduleEntry]:
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        entries = []
        for schedule in (payload.get("schedules") or {}).values():
            entries.append(
                FeeScheduleEntry(
                    schedule_id=str(schedule["schedule_id"]),
                    effective_from=_parse_dt(schedule["effective_from"]) or datetime.min.replace(tzinfo=timezone.utc),
                    effective_to=_parse_dt(schedule.get("effective_to")),
                    asset_class=str(schedule.get("asset_class", "crypto")),
                    fee_rate=float(schedule["fee_rate"]),
                    exponent=int(schedule["exponent"]),
                    maker_fee_rate=float(schedule.get("maker_fee_rate", 0.0)),
                    maker_rebate_pct=float(schedule.get("maker_rebate_pct", 0.0)),
                    source=str(schedule.get("source", "")),
                )
            )
        if not entries:
            raise ValueError("Fee schedule registry is empty.")
        return sorted(entries, key=lambda entry: entry.effective_from)

    def _validate_no_overlap(self) -> None:
        for left, right in zip(self.entries, self.entries[1:], strict=False):
            left_end = left.effective_to or datetime.max.replace(tzinfo=timezone.utc)
            if left_end >= right.effective_from:
                raise ValueError(f"Overlapping fee schedules: {left.schedule_id} and {right.schedule_id}")

    def lookup(self, when: str | datetime, asset_class: str = "crypto") -> FeeScheduleEntry:
        if isinstance(when, str):
            ts = _parse_dt(when)
            if ts is None:
                raise ValueError(f"Invalid datetime: {when}")
        else:
            ts = when.astimezone(timezone.utc)
        for entry in self.entries:
            if entry.asset_class != asset_class:
                continue
            end = entry.effective_to or datetime.max.replace(tzinfo=timezone.utc)
            if entry.effective_from <= ts <= end:
                return entry
        raise ValueError(f"No fee schedule for {asset_class} at {ts.isoformat()}")
