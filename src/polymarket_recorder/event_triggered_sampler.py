from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

TRIGGER_CONFIGS = {
    "strike_cross": {
        "condition": "sign(delta_to_strike) changed vs previous snapshot",
        "min_gap_ms": 250,
    },
    "spread_burst": {
        "condition": "up_token_spread or down_token_spread > 2x rolling_mean_spread",
        "min_gap_ms": 250,
    },
    "trade_burst": {
        "condition": "up_token_trade_count_1s > p90_trade_rate or down_token equivalent",
        "min_gap_ms": 250,
    },
    "imbalance_change": {
        "condition": "abs(imbalance_last - imbalance_prev) > 0.30",
        "min_gap_ms": 250,
    },
    "tick_size_update": {
        "condition": "tick_size changed vs previous snapshot",
        "min_gap_ms": 0,
    },
    "cross_source_divergence": {
        "condition": "abs(binance_spot_mid - live_btc_usd) > 15.0",
        "min_gap_ms": 250,
    },
}


def _date_range(date_from: date, date_to: date) -> list[date]:
    return [date_from + timedelta(days=i) for i in range((date_to - date_from).days + 1)]


def _first_existing(frame: pl.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


@dataclass(slots=True)
class EventTriggeredSampler:
    """Event-triggered snapshot sampler for resolution_snapshot_dataset_v1."""

    min_gap_default_ms: int = 250

    def _base_with_trigger_columns(self, snapshot_df: pl.DataFrame) -> pl.DataFrame:
        if "trigger_type" not in snapshot_df.columns:
            snapshot_df = snapshot_df.with_columns(pl.lit("regular").alias("trigger_type"))
        if "trigger_magnitude" not in snapshot_df.columns:
            snapshot_df = snapshot_df.with_columns(pl.lit(0.0).alias("trigger_magnitude"))
        return snapshot_df

    def detect_triggers(self, snapshot_df: pl.DataFrame) -> pl.DataFrame:
        """Add trigger_type and trigger_magnitude columns to existing snapshot rows."""
        if snapshot_df.is_empty():
            return self._base_with_trigger_columns(snapshot_df)
        ts_col = _first_existing(snapshot_df, ["snapshot_ts_ns", "recv_ts_ns"])
        if ts_col is None:
            raise ValueError("EventTriggeredSampler requires snapshot_ts_ns or recv_ts_ns.")
        frame = snapshot_df.sort(["market_slug", ts_col] if "market_slug" in snapshot_df.columns else [ts_col])
        candidates: list[pl.DataFrame] = [self._regular_rows(frame)]
        for trigger_type, detected in [
            ("strike_cross", self._detect_strike_cross(frame, ts_col)),
            ("spread_burst", self._detect_spread_burst(frame, ts_col)),
            ("trade_burst", self._detect_trade_burst(frame, ts_col)),
            ("imbalance_change", self._detect_imbalance_change(frame, ts_col)),
            ("tick_size_update", self._detect_tick_size_update(frame, ts_col)),
            ("cross_source_divergence", self._detect_cross_source_divergence(frame, ts_col)),
        ]:
            if detected.height:
                candidates.append(self._dedup_trigger(detected, ts_col, trigger_type))
        return pl.concat(candidates, how="diagonal").sort(ts_col)

    def generate_additional_snapshots(
        self,
        market_slug: str,
        l2_events: pl.DataFrame,
        live_ref: pl.DataFrame,
        existing_snapshots: pl.DataFrame,
    ) -> pl.DataFrame:
        """Generate additional triggered snapshots not already in regular sampling."""
        del l2_events, live_ref
        frame = existing_snapshots
        if "market_slug" in frame.columns:
            frame = frame.filter(pl.col("market_slug") == market_slug)
        triggered = self.detect_triggers(frame).filter(pl.col("trigger_type") != "regular")
        return triggered

    def _regular_rows(self, frame: pl.DataFrame) -> pl.DataFrame:
        return self._base_with_trigger_columns(frame).with_columns(
            pl.lit("regular").alias("trigger_type"),
            pl.lit(0.0).alias("trigger_magnitude"),
        )

    def _with_trigger(self, frame: pl.DataFrame, trigger_type: str, magnitude: pl.Expr) -> pl.DataFrame:
        return frame.with_columns(
            pl.lit(trigger_type).alias("trigger_type"),
            magnitude.cast(pl.Float64).fill_null(0.0).alias("trigger_magnitude"),
        )

    def _over_group(self) -> list[str] | None:
        return ["market_slug"] if True else None

    def _detect_strike_cross(self, frame: pl.DataFrame, ts_col: str) -> pl.DataFrame:
        if "delta_to_strike" not in frame.columns:
            return frame.clear()
        group = ["market_slug"] if "market_slug" in frame.columns else None
        sign_expr = pl.when(pl.col("delta_to_strike") > 0).then(1).when(pl.col("delta_to_strike") < 0).then(-1).otherwise(0)
        signed = frame.with_columns(sign_expr.alias("_delta_sign"))
        previous = pl.col("_delta_sign").shift(1).over(group) if group else pl.col("_delta_sign").shift(1)
        detected = signed.filter(previous.is_not_null() & (pl.col("_delta_sign") != previous))
        return self._with_trigger(detected.drop("_delta_sign"), "strike_cross", pl.col("delta_to_strike").abs())

    def _detect_spread_burst(self, frame: pl.DataFrame, ts_col: str) -> pl.DataFrame:
        del ts_col
        spread_cols = [c for c in ["up_token_spread", "down_token_spread", "spread"] if c in frame.columns]
        if not spread_cols:
            return frame.clear()
        spread = pl.max_horizontal([pl.col(c).cast(pl.Float64) for c in spread_cols])
        group = ["market_slug"] if "market_slug" in frame.columns else None
        rolling = spread.rolling_mean(window_size=5).over(group) if group else spread.rolling_mean(window_size=5)
        detected = frame.with_columns(spread.alias("_spread"), rolling.alias("_spread_mean")).filter(
            pl.col("_spread_mean").is_not_null() & (pl.col("_spread") > 2.0 * pl.col("_spread_mean"))
        )
        return self._with_trigger(detected, "spread_burst", pl.col("_spread")).drop(["_spread", "_spread_mean"])

    def _detect_trade_burst(self, frame: pl.DataFrame, ts_col: str) -> pl.DataFrame:
        del ts_col
        cols = [c for c in ["up_token_trade_count_1s", "down_token_trade_count_1s", "trade_count_1s"] if c in frame.columns]
        if not cols:
            return frame.clear()
        count = pl.max_horizontal([pl.col(c).cast(pl.Float64) for c in cols])
        threshold = frame.select(count.quantile(0.90)).item()
        threshold = 1.0 if threshold is None else float(threshold)
        detected = frame.with_columns(count.alias("_trade_count")).filter(pl.col("_trade_count") > threshold)
        return self._with_trigger(detected, "trade_burst", pl.col("_trade_count")).drop("_trade_count")

    def _detect_imbalance_change(self, frame: pl.DataFrame, ts_col: str) -> pl.DataFrame:
        del ts_col
        col = _first_existing(frame, ["imbalance", "up_token_imbalance", "down_token_imbalance"])
        if col is None:
            return frame.clear()
        group = ["market_slug"] if "market_slug" in frame.columns else None
        previous = pl.col(col).shift(1).over(group) if group else pl.col(col).shift(1)
        detected = frame.with_columns((pl.col(col) - previous).abs().alias("_imbalance_delta")).filter(
            pl.col("_imbalance_delta") > 0.30
        )
        return self._with_trigger(detected, "imbalance_change", pl.col("_imbalance_delta")).drop("_imbalance_delta")

    def _detect_tick_size_update(self, frame: pl.DataFrame, ts_col: str) -> pl.DataFrame:
        del ts_col
        cols = [c for c in ["up_token_tick_size", "down_token_tick_size", "tick_size"] if c in frame.columns]
        if not cols:
            return frame.clear()
        tick = pl.max_horizontal([pl.col(c).cast(pl.Float64) for c in cols])
        group = ["market_slug"] if "market_slug" in frame.columns else None
        previous = tick.shift(1).over(group) if group else tick.shift(1)
        detected = frame.with_columns(tick.alias("_tick"), previous.alias("_prev_tick")).filter(
            pl.col("_prev_tick").is_not_null() & (pl.col("_tick") != pl.col("_prev_tick"))
        )
        return self._with_trigger(detected, "tick_size_update", pl.lit(1.0)).drop(["_tick", "_prev_tick"])

    def _detect_cross_source_divergence(self, frame: pl.DataFrame, ts_col: str) -> pl.DataFrame:
        del ts_col
        if "binance_spot_mid" not in frame.columns or "live_btc_usd" not in frame.columns:
            return frame.clear()
        detected = frame.with_columns(
            (pl.col("binance_spot_mid") - pl.col("live_btc_usd")).abs().alias("_divergence")
        ).filter(pl.col("_divergence") > 15.0)
        return self._with_trigger(detected, "cross_source_divergence", pl.col("_divergence")).drop("_divergence")

    def _dedup_trigger(self, frame: pl.DataFrame, ts_col: str, trigger_type: str) -> pl.DataFrame:
        min_gap_ms = int(TRIGGER_CONFIGS[trigger_type]["min_gap_ms"])
        if min_gap_ms <= 0 or frame.is_empty():
            return frame
        bucket_ns = min_gap_ms * 1_000_000
        sort_cols = ["market_slug", ts_col] if "market_slug" in frame.columns else [ts_col]
        kept: list[dict[str, Any]] = []
        last_by_group: dict[str, int] = {}
        for row in frame.sort(sort_cols).iter_rows(named=True):
            group_key = str(row.get("market_slug", "__global__"))
            ts = int(row[ts_col])
            previous = last_by_group.get(group_key)
            if previous is None or ts - previous >= bucket_ns:
                kept.append(row)
                last_by_group[group_key] = ts
        if not kept:
            return frame.clear()
        return pl.DataFrame(kept, schema=frame.schema, orient="row")


def build_event_triggered_snapshots_status(
    *,
    date_from: date,
    date_to: date,
    output_dir: Path = Path("data/datasets/resolution_snapshot_dataset_v1_event_triggered"),
) -> dict[str, Any]:
    daily = []
    for target_day in _date_range(date_from, date_to):
        path = output_dir / f"{target_day.isoformat()}.parquet"
        daily.append({"date": target_day.isoformat(), "exists": path.exists(), "path": str(path)})
    return {"dataset": "resolution_snapshot_dataset_v1_event_triggered", "daily": daily}


def build_event_triggered_snapshots(
    *,
    date_from: date,
    date_to: date,
    dataset_root: Path = Path("data/datasets"),
    output_dir: Path = Path("data/datasets/resolution_snapshot_dataset_v1_coarse_event_triggered"),
    skip_existing: bool = True,
) -> dict[str, Any]:
    sampler = EventTriggeredSampler()
    output_dir.mkdir(parents=True, exist_ok=True)
    daily: list[dict[str, Any]] = []
    for target_day in _date_range(date_from, date_to):
        day = target_day.isoformat()
        source = dataset_root / "resolution_snapshot_dataset_v1_coarse" / f"{day}.parquet"
        target = output_dir / f"{day}.parquet"
        if skip_existing and target.exists() and target.stat().st_size > 0:
            daily.append({"date": day, "status": "SKIPPED_EXISTING", "rows": None, "path": str(target)})
            continue
        if not source.exists():
            daily.append({"date": day, "status": "MISSING_SOURCE", "rows": 0})
            continue
        frame = pl.read_parquet(source)
        triggered = sampler.detect_triggers(frame)
        triggered.write_parquet(target)
        daily.append({"date": day, "status": "BUILT", "rows": len(triggered), "path": str(target)})
    return {"dataset": "resolution_snapshot_dataset_v1_coarse_event_triggered", "daily": daily}
