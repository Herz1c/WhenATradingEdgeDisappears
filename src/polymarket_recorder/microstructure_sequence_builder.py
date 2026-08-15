from __future__ import annotations

import hashlib
import math
import os
import threading
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .l2_event_loader import DEFAULT_QUALITY_REGISTRY_PATH, load_l2_events

SECOND_NS = 1_000_000_000
DEFAULT_DATASET_OUTPUT_DIR = Path("data/datasets")
DEFAULT_SCHEMA_PATH = Path("config/schemas/microstructure_sequence_dataset_v1_schema.yaml")
DEFAULT_REPORT_PATH = Path("docs/phase8_microstructure_dataset_factory_report.md")
DEFAULT_DECISION_LOG_PATH = Path("docs/decision_log.md")

TABULAR_DATASET = "microstructure_sequence_dataset_v1_tabular"
EVENT64_DATASET = "microstructure_sequence_dataset_v1_event64"
EVENT128_DATASET = "microstructure_sequence_dataset_v1_event128"

TARGET_COLUMNS = [
    "mid_move_1s",
    "mid_move_3s",
    "mid_move_5s",
    "mid_move_15s",
    "mid_move_30s",
    "mid_move_sign_1s",
    "mid_move_sign_3s",
    "mid_move_sign_5s",
    "mid_move_sign_15s",
    "mid_move_sign_30s",
    "max_favorable_excursion_5s",
    "max_adverse_excursion_5s",
    "max_favorable_excursion_15s",
    "max_adverse_excursion_15s",
    "hold_to_close_edge_vs_mid",
    "terminal_margin_to_strike",
    "flip_happened_after_t",
]

SEQUENCE_COLUMNS: list[str] = [
    "events_recv_ts_ns",
    "events_dt_from_anchor_s",
    "events_dt_from_prev_s",
    "events_event_type",
    "events_best_bid",
    "events_best_ask",
    "events_mid",
    "events_spread",
    "events_microprice",
    "events_bid_depth_total",
    "events_ask_depth_total",
    "events_imbalance",
    "events_last_trade_price",
    "events_trade_size",
    "events_trade_side",
    "events_tick_size",
    "sequence_length_actual",
    "sequence_padded",
]

SCHEMA_COLUMNS: list[str] = [
    "sequence_id",
    "market_slug",
    "token_asset_id",
    "token_side_normalized",
    "anchor_ts_ns",
    "anchor_source",
    "events_recv_ts_ns",
    "events_dt_from_anchor_s",
    "events_dt_from_prev_s",
    "events_event_type",
    "events_best_bid",
    "events_best_ask",
    "events_mid",
    "events_spread",
    "events_microprice",
    "events_bid_depth_total",
    "events_ask_depth_total",
    "events_imbalance",
    "events_last_trade_price",
    "events_trade_size",
    "events_trade_side",
    "events_tick_size",
    "sequence_length_actual",
    "sequence_padded",
    "event_count_1s",
    "event_count_3s",
    "event_count_5s",
    "event_count_15s",
    "quote_update_count_1s",
    "quote_update_count_3s",
    "quote_update_count_5s",
    "trade_count_1s",
    "trade_count_3s",
    "trade_count_5s",
    "signed_trade_size_1s",
    "signed_trade_size_3s",
    "signed_trade_size_5s",
    "absolute_trade_size_1s",
    "absolute_trade_size_3s",
    "absolute_trade_size_5s",
    "spread_mean_5s",
    "spread_min_5s",
    "spread_max_5s",
    "imbalance_mean_5s",
    "imbalance_last",
    "imbalance_slope_5s",
    "mid_return_1s",
    "mid_return_3s",
    "mid_return_5s",
    "quote_churn_rate_5s",
    "depth_change_rate_5s",
    "one_sided_book",
    "crossed_book",
    "stale_book",
    "sequence_completeness_rate",
    "price_to_beat",
    "live_btc_usd",
    "delta_to_strike",
    "abs_delta_to_strike",
    "delta_sign",
    "t_since_open_s",
    "t_to_close_s",
    "phase_bucket",
    "live_reference_trusted",
    "live_bias_mode",
    "live_applied_bias_age_seconds",
    *TARGET_COLUMNS,
    "sequence_feature_eligible",
    "sequence_complete_eligible",
    "target_1s_available",
    "target_3s_available",
    "target_5s_available",
    "target_15s_available",
    "target_30s_available",
    "microstructure_exclusion_reason",
]

TABULAR_COLUMNS: list[str] = [column for column in SCHEMA_COLUMNS if column not in SEQUENCE_COLUMNS]

PA_SCHEMA = pa.schema(
    [
        pa.field("sequence_id", pa.string()),
        pa.field("market_slug", pa.string()),
        pa.field("token_asset_id", pa.string()),
        pa.field("token_side_normalized", pa.string()),
        pa.field("anchor_ts_ns", pa.int64()),
        pa.field("anchor_source", pa.string()),
        pa.field("events_recv_ts_ns", pa.list_(pa.int64())),
        pa.field("events_dt_from_anchor_s", pa.list_(pa.float64())),
        pa.field("events_dt_from_prev_s", pa.list_(pa.float64())),
        pa.field("events_event_type", pa.list_(pa.string())),
        pa.field("events_best_bid", pa.list_(pa.float64())),
        pa.field("events_best_ask", pa.list_(pa.float64())),
        pa.field("events_mid", pa.list_(pa.float64())),
        pa.field("events_spread", pa.list_(pa.float64())),
        pa.field("events_microprice", pa.list_(pa.float64())),
        pa.field("events_bid_depth_total", pa.list_(pa.float64())),
        pa.field("events_ask_depth_total", pa.list_(pa.float64())),
        pa.field("events_imbalance", pa.list_(pa.float64())),
        pa.field("events_last_trade_price", pa.list_(pa.float64())),
        pa.field("events_trade_size", pa.list_(pa.float64())),
        pa.field("events_trade_side", pa.list_(pa.string())),
        pa.field("events_tick_size", pa.list_(pa.float64())),
        pa.field("sequence_length_actual", pa.int16()),
        pa.field("sequence_padded", pa.bool_()),
        pa.field("event_count_1s", pa.int32()),
        pa.field("event_count_3s", pa.int32()),
        pa.field("event_count_5s", pa.int32()),
        pa.field("event_count_15s", pa.int32()),
        pa.field("quote_update_count_1s", pa.int32()),
        pa.field("quote_update_count_3s", pa.int32()),
        pa.field("quote_update_count_5s", pa.int32()),
        pa.field("trade_count_1s", pa.int32()),
        pa.field("trade_count_3s", pa.int32()),
        pa.field("trade_count_5s", pa.int32()),
        pa.field("signed_trade_size_1s", pa.float64()),
        pa.field("signed_trade_size_3s", pa.float64()),
        pa.field("signed_trade_size_5s", pa.float64()),
        pa.field("absolute_trade_size_1s", pa.float64()),
        pa.field("absolute_trade_size_3s", pa.float64()),
        pa.field("absolute_trade_size_5s", pa.float64()),
        pa.field("spread_mean_5s", pa.float64()),
        pa.field("spread_min_5s", pa.float64()),
        pa.field("spread_max_5s", pa.float64()),
        pa.field("imbalance_mean_5s", pa.float64()),
        pa.field("imbalance_last", pa.float64()),
        pa.field("imbalance_slope_5s", pa.float64()),
        pa.field("mid_return_1s", pa.float64()),
        pa.field("mid_return_3s", pa.float64()),
        pa.field("mid_return_5s", pa.float64()),
        pa.field("quote_churn_rate_5s", pa.float64()),
        pa.field("depth_change_rate_5s", pa.float64()),
        pa.field("one_sided_book", pa.bool_()),
        pa.field("crossed_book", pa.bool_()),
        pa.field("stale_book", pa.bool_()),
        pa.field("sequence_completeness_rate", pa.float64()),
        pa.field("price_to_beat", pa.float64()),
        pa.field("live_btc_usd", pa.float64()),
        pa.field("delta_to_strike", pa.float64()),
        pa.field("abs_delta_to_strike", pa.float64()),
        pa.field("delta_sign", pa.int8()),
        pa.field("t_since_open_s", pa.float64()),
        pa.field("t_to_close_s", pa.float64()),
        pa.field("phase_bucket", pa.string()),
        pa.field("live_reference_trusted", pa.bool_()),
        pa.field("live_bias_mode", pa.string()),
        pa.field("live_applied_bias_age_seconds", pa.float64()),
        pa.field("mid_move_1s", pa.float64()),
        pa.field("mid_move_3s", pa.float64()),
        pa.field("mid_move_5s", pa.float64()),
        pa.field("mid_move_15s", pa.float64()),
        pa.field("mid_move_30s", pa.float64()),
        pa.field("mid_move_sign_1s", pa.int8()),
        pa.field("mid_move_sign_3s", pa.int8()),
        pa.field("mid_move_sign_5s", pa.int8()),
        pa.field("mid_move_sign_15s", pa.int8()),
        pa.field("mid_move_sign_30s", pa.int8()),
        pa.field("max_favorable_excursion_5s", pa.float64()),
        pa.field("max_adverse_excursion_5s", pa.float64()),
        pa.field("max_favorable_excursion_15s", pa.float64()),
        pa.field("max_adverse_excursion_15s", pa.float64()),
        pa.field("hold_to_close_edge_vs_mid", pa.float64()),
        pa.field("terminal_margin_to_strike", pa.float64()),
        pa.field("flip_happened_after_t", pa.bool_()),
        pa.field("sequence_feature_eligible", pa.bool_()),
        pa.field("sequence_complete_eligible", pa.bool_()),
        pa.field("target_1s_available", pa.bool_()),
        pa.field("target_3s_available", pa.bool_()),
        pa.field("target_5s_available", pa.bool_()),
        pa.field("target_15s_available", pa.bool_()),
        pa.field("target_30s_available", pa.bool_()),
        pa.field("microstructure_exclusion_reason", pa.string()),
    ]
)
TABULAR_PA_SCHEMA = pa.schema([field for field in PA_SCHEMA if field.name in set(TABULAR_COLUMNS)])


@dataclass(slots=True)
class AnchorRow:
    market_slug: str
    token_asset_id: str
    token_side_normalized: str
    anchor_ts_ns: int
    anchor_source: str
    market_close_ts_ns: int
    resolved_side: str
    price_to_beat: float
    live_btc_usd: float
    delta_to_strike: float
    abs_delta_to_strike: float
    delta_sign: int
    t_since_open_s: float
    t_to_close_s: float
    phase_bucket: str
    live_reference_trusted: bool
    live_price_valid: bool
    live_bias_mode: str
    live_applied_bias_age_seconds: float
    training_feature_eligible: bool
    exclusion_reason: str
    terminal_margin_to_strike: float
    flip_happened_after_t: bool


class TokenEventIndex:
    def __init__(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            self.frame = frame.copy()
            self.times: list[int] = []
            self.times_np = np.array([], dtype=np.int64)
            self.event_type_np = np.array([], dtype=object)
            self.trade_side_np = np.array([], dtype=object)
            self.best_bid_np = np.array([], dtype=float)
            self.best_ask_np = np.array([], dtype=float)
            self.mid_np = np.array([], dtype=float)
            self.spread_np = np.array([], dtype=float)
            self.microprice_np = np.array([], dtype=float)
            self.bid_depth_np = np.array([], dtype=float)
            self.ask_depth_np = np.array([], dtype=float)
            self.imbalance_np = np.array([], dtype=float)
            self.last_trade_price_np = np.array([], dtype=float)
            self.trade_size_np = np.array([], dtype=float)
            self.tick_size_np = np.array([], dtype=float)
            self.is_quote_np = np.array([], dtype=bool)
            self.is_trade_np = np.array([], dtype=bool)
            self.signed_trade_size_np = np.array([], dtype=float)
            self.abs_trade_size_np = np.array([], dtype=float)
            self.event_prefix = np.array([0], dtype=np.int64)
            self.quote_prefix = np.array([0], dtype=np.int64)
            self.trade_prefix = np.array([0], dtype=np.int64)
            self.signed_trade_prefix = np.array([0.0], dtype=float)
            self.abs_trade_prefix = np.array([0.0], dtype=float)
            self.expected_rate_30s = 1.0
            return
        ordered = frame.sort_values("recv_ts_ns").reset_index(drop=True).copy()
        state_cols = [
            "best_bid",
            "best_ask",
            "mid",
            "spread",
            "microprice",
            "bid_depth_total",
            "ask_depth_total",
            "imbalance",
            "last_trade_price",
            "tick_size",
        ]
        for column in state_cols:
            if column not in ordered.columns:
                ordered[column] = np.nan
            ordered[column] = ordered[column].ffill()
        ordered["trade_size"] = ordered.get("trade_size", 0.0).fillna(0.0)
        ordered["trade_side"] = ordered.get("trade_side", "none").fillna("none").astype(str)
        ordered["event_type"] = ordered.get("event_type", "other").fillna("other").astype(str)
        self.frame = ordered
        self.times = ordered["recv_ts_ns"].astype("int64").tolist()
        self.times_np = ordered["recv_ts_ns"].astype("int64").to_numpy()
        self.event_type_np = ordered["event_type"].astype(str).to_numpy()
        self.trade_side_np = ordered["trade_side"].astype(str).to_numpy()
        self.best_bid_np = ordered["best_bid"].astype(float).to_numpy()
        self.best_ask_np = ordered["best_ask"].astype(float).to_numpy()
        self.mid_np = ordered["mid"].astype(float).to_numpy()
        self.spread_np = ordered["spread"].astype(float).to_numpy()
        self.microprice_np = ordered["microprice"].astype(float).to_numpy()
        self.bid_depth_np = ordered["bid_depth_total"].astype(float).to_numpy()
        self.ask_depth_np = ordered["ask_depth_total"].astype(float).to_numpy()
        self.imbalance_np = ordered["imbalance"].astype(float).to_numpy()
        self.last_trade_price_np = ordered["last_trade_price"].astype(float).to_numpy()
        self.trade_size_np = ordered["trade_size"].astype(float).to_numpy()
        self.tick_size_np = ordered["tick_size"].astype(float).to_numpy()
        self.is_quote_np = np.isin(self.event_type_np, ["book_state", "top_of_book"])
        self.is_trade_np = self.event_type_np == "trade"
        trade_sign = np.where(self.trade_side_np == "buy", 1.0, np.where(self.trade_side_np == "sell", -1.0, 0.0))
        self.signed_trade_size_np = self.trade_size_np * trade_sign
        self.abs_trade_size_np = np.abs(self.trade_size_np)
        self.event_prefix = self._prefix(np.ones(len(self.times_np), dtype=np.int64))
        self.quote_prefix = self._prefix(self.is_quote_np.astype(np.int64))
        self.trade_prefix = self._prefix(self.is_trade_np.astype(np.int64))
        self.signed_trade_prefix = self._prefix(self.signed_trade_size_np)
        self.abs_trade_prefix = self._prefix(self.abs_trade_size_np)
        span_s = max((self.times[-1] - self.times[0]) / SECOND_NS, 1.0) if len(self.times) > 1 else 30.0
        self.expected_rate_30s = max((len(self.times) / span_s) * 30.0, 1.0)

    @staticmethod
    def _prefix(values: np.ndarray) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(values)])

    def _pos_before(self, anchor_ts_ns: int) -> int:
        return bisect_left(self.times, int(anchor_ts_ns))

    def _pos_after(self, anchor_ts_ns: int) -> int:
        return bisect_right(self.times, int(anchor_ts_ns))

    def history_before(self, anchor_ts_ns: int) -> pd.DataFrame:
        return self.frame.iloc[: self._pos_before(anchor_ts_ns)]

    def future_after_until(self, anchor_ts_ns: int, end_ts_ns: int) -> pd.DataFrame:
        start = self._pos_after(anchor_ts_ns)
        end = bisect_right(self.times, int(end_ts_ns))
        return self.frame.iloc[start:end]

    def get_event_sequence(self, anchor_ts_ns: int, n_events: int) -> tuple[pd.DataFrame, bool]:
        end = self._pos_before(anchor_ts_ns)
        start = max(0, end - n_events)
        sequence = self.frame.iloc[start:end].copy()
        return sequence, len(sequence) >= n_events

    def _window_bounds(self, anchor_ts_ns: int, seconds: int) -> tuple[int, int]:
        right = bisect_left(self.times, int(anchor_ts_ns))
        left = bisect_left(self.times, int(anchor_ts_ns - (seconds * SECOND_NS)))
        return left, right

    @staticmethod
    def _prefix_sum(prefix: np.ndarray, left: int, right: int) -> float:
        return float(prefix[right] - prefix[left])

    @staticmethod
    def _last_valid(values: np.ndarray) -> float:
        if len(values) == 0:
            return math.nan
        valid = values[~np.isnan(values)]
        return float(valid[-1]) if len(valid) else math.nan

    def latest_mid_at_or_before(self, ts_ns: int) -> float:
        if len(self.times_np) == 0:
            return math.nan
        pos = bisect_right(self.times, int(ts_ns))
        if pos <= 0:
            return math.nan
        return self._last_valid(self.mid_np[:pos])

    def latest_mid_before(self, ts_ns: int) -> float:
        if len(self.times_np) == 0:
            return math.nan
        pos = bisect_left(self.times, int(ts_ns))
        if pos <= 0:
            return math.nan
        return self._last_valid(self.mid_np[:pos])

    def sequence_payload(self, anchor_ts_ns: int, n_events: int) -> tuple[dict[str, list[Any]], int, bool, float]:
        end = bisect_left(self.times, int(anchor_ts_ns))
        start = max(0, end - n_events)
        actual = end - start
        max_gap_s = 0.0
        if actual > 1:
            max_gap_s = float(np.max(np.diff(self.times_np[start:end])) / SECOND_NS)

        def nums(values: np.ndarray) -> list[float]:
            return [float(value) if not np.isnan(value) else math.nan for value in values[start:end]]

        recv_values = self.times_np[start:end].astype(np.int64)
        dt_anchor = ((recv_values - int(anchor_ts_ns)) / SECOND_NS).astype(float) if actual else np.array([], dtype=float)
        dt_prev = np.concatenate([[0.0], np.diff(recv_values) / SECOND_NS]) if actual else np.array([], dtype=float)
        payload: dict[str, list[Any]] = {
            "events_recv_ts_ns": [int(value) for value in recv_values.tolist()],
            "events_dt_from_anchor_s": [float(value) for value in dt_anchor.tolist()],
            "events_dt_from_prev_s": [float(value) for value in dt_prev.tolist()],
            "events_event_type": [str(value) for value in self.event_type_np[start:end].tolist()],
            "events_best_bid": nums(self.best_bid_np),
            "events_best_ask": nums(self.best_ask_np),
            "events_mid": nums(self.mid_np),
            "events_spread": nums(self.spread_np),
            "events_microprice": nums(self.microprice_np),
            "events_bid_depth_total": nums(self.bid_depth_np),
            "events_ask_depth_total": nums(self.ask_depth_np),
            "events_imbalance": nums(self.imbalance_np),
            "events_last_trade_price": nums(self.last_trade_price_np),
            "events_trade_size": nums(self.trade_size_np),
            "events_trade_side": [str(value) for value in self.trade_side_np[start:end].tolist()],
            "events_tick_size": nums(self.tick_size_np),
        }
        missing = n_events - actual
        if missing > 0:
            payload["events_recv_ts_ns"] = [0] * missing + payload["events_recv_ts_ns"]
            payload["events_dt_from_anchor_s"] = [-1_000_000_000.0] * missing + payload["events_dt_from_anchor_s"]
            payload["events_dt_from_prev_s"] = [0.0] * missing + payload["events_dt_from_prev_s"]
            payload["events_event_type"] = ["pad"] * missing + payload["events_event_type"]
            for column in [
                "events_best_bid",
                "events_best_ask",
                "events_mid",
                "events_spread",
                "events_microprice",
                "events_bid_depth_total",
                "events_ask_depth_total",
                "events_last_trade_price",
                "events_tick_size",
            ]:
                payload[column] = [math.nan] * missing + payload[column]
            payload["events_imbalance"] = [0.0] * missing + payload["events_imbalance"]
            payload["events_trade_size"] = [0.0] * missing + payload["events_trade_size"]
            payload["events_trade_side"] = ["none"] * missing + payload["events_trade_side"]
        return payload, actual, actual >= n_events, max_gap_s

    def tabular_features(self, anchor_ts_ns: int) -> dict[str, Any]:
        features: dict[str, Any] = {}
        for seconds in (1, 3, 5, 15):
            left, right = self._window_bounds(anchor_ts_ns, seconds)
            features[f"event_count_{seconds}s"] = int(self._prefix_sum(self.event_prefix, left, right))
        for seconds in (1, 3, 5):
            left, right = self._window_bounds(anchor_ts_ns, seconds)
            features[f"quote_update_count_{seconds}s"] = int(self._prefix_sum(self.quote_prefix, left, right))
            features[f"trade_count_{seconds}s"] = int(self._prefix_sum(self.trade_prefix, left, right))
            features[f"signed_trade_size_{seconds}s"] = self._prefix_sum(self.signed_trade_prefix, left, right)
            features[f"absolute_trade_size_{seconds}s"] = self._prefix_sum(self.abs_trade_prefix, left, right)
        left5, right = self._window_bounds(anchor_ts_ns, 5)
        spreads = self.spread_np[left5:right]
        spreads = spreads[~np.isnan(spreads)]
        features["spread_mean_5s"] = float(np.mean(spreads)) if len(spreads) else math.nan
        features["spread_min_5s"] = float(np.min(spreads)) if len(spreads) else math.nan
        features["spread_max_5s"] = float(np.max(spreads)) if len(spreads) else math.nan
        imbalances = self.imbalance_np[left5:right]
        imbalance_mask = ~np.isnan(imbalances)
        valid_imbalances = imbalances[imbalance_mask]
        features["imbalance_mean_5s"] = float(np.mean(valid_imbalances)) if len(valid_imbalances) else math.nan
        latest_pos = bisect_left(self.times, int(anchor_ts_ns)) - 1
        if latest_pos < 0:
            features.update(
                {
                    "imbalance_last": math.nan,
                    "imbalance_slope_5s": 0.0,
                    "mid_return_1s": math.nan,
                    "mid_return_3s": math.nan,
                    "mid_return_5s": math.nan,
                    "quote_churn_rate_5s": float(features["quote_update_count_5s"] / 5.0),
                    "depth_change_rate_5s": 0.0,
                    "one_sided_book": True,
                    "crossed_book": False,
                    "stale_book": True,
                    "sequence_completeness_rate": 0.0,
                }
            )
            return features
        features["imbalance_last"] = float(self.imbalance_np[latest_pos]) if not np.isnan(self.imbalance_np[latest_pos]) else math.nan
        if len(valid_imbalances) >= 2:
            valid_times = self.times_np[left5:right][imbalance_mask]
            duration_s = max((int(valid_times[-1]) - int(valid_times[0])) / SECOND_NS, 1e-9)
            features["imbalance_slope_5s"] = float((valid_imbalances[-1] - valid_imbalances[0]) / duration_s)
        else:
            features["imbalance_slope_5s"] = 0.0
        latest_mid = self.latest_mid_before(anchor_ts_ns)
        for seconds in (1, 3, 5):
            past_mid = self.latest_mid_at_or_before(anchor_ts_ns - seconds * SECOND_NS)
            features[f"mid_return_{seconds}s"] = (
                float(latest_mid - past_mid)
                if not math.isnan(latest_mid) and not math.isnan(past_mid)
                else math.nan
            )
        features["quote_churn_rate_5s"] = float(features["quote_update_count_5s"] / 5.0)
        if right - left5 >= 2:
            depth = np.nan_to_num(self.bid_depth_np[left5:right]) + np.nan_to_num(self.ask_depth_np[left5:right])
            features["depth_change_rate_5s"] = float((depth[-1] - depth[0]) / 5.0) if len(depth) >= 2 else 0.0
        else:
            features["depth_change_rate_5s"] = 0.0
        best_bid = self.best_bid_np[latest_pos]
        best_ask = self.best_ask_np[latest_pos]
        bid_depth = self.bid_depth_np[latest_pos]
        ask_depth = self.ask_depth_np[latest_pos]
        features["crossed_book"] = bool(not np.isnan(best_bid) and not np.isnan(best_ask) and best_bid >= best_ask)
        features["stale_book"] = bool(anchor_ts_ns - int(self.times_np[latest_pos]) > 30 * SECOND_NS)
        features["one_sided_book"] = bool(
            np.isnan(best_bid)
            or np.isnan(best_ask)
            or np.isnan(bid_depth)
            or np.isnan(ask_depth)
            or bid_depth <= 0
            or ask_depth <= 0
        )
        left30, right30 = self._window_bounds(anchor_ts_ns, 30)
        features["sequence_completeness_rate"] = float((right30 - left30) / max(self.expected_rate_30s, 1.0))
        return features

    def targets(self, anchor: AnchorRow, anchor_mid: float) -> tuple[dict[str, Any], dict[int, bool]]:
        targets: dict[str, Any] = {}
        availability: dict[int, bool] = {}
        for horizon_s in (1, 3, 5, 15, 30):
            target_ts = anchor.anchor_ts_ns + horizon_s * SECOND_NS
            if target_ts > anchor.market_close_ts_ns or math.isnan(anchor_mid):
                mid_move = math.nan
            else:
                start = bisect_right(self.times, int(anchor.anchor_ts_ns))
                end = bisect_right(self.times, int(target_ts))
                future_mid = self._last_valid(self.mid_np[start:end]) if end > start else math.nan
                mid_move = float(future_mid - anchor_mid) if not math.isnan(future_mid) else math.nan
            available = _target_available(t_to_close_s=anchor.t_to_close_s, mid_move=mid_move, horizon_s=horizon_s)
            if not available:
                mid_move = math.nan
            targets[f"mid_move_{horizon_s}s"] = mid_move
            targets[f"mid_move_sign_{horizon_s}s"] = _sign(mid_move)
            availability[horizon_s] = available
            if not availability[horizon_s]:
                assert math.isnan(targets[f"mid_move_{horizon_s}s"]), (
                    f"target_{horizon_s}s_available=False but mid_move_{horizon_s}s is not NaN"
                )
            if anchor.t_to_close_s <= horizon_s:
                assert not availability[horizon_s], (
                    f"target_{horizon_s}s_available=True but t_to_close_s={anchor.t_to_close_s} <= {horizon_s}"
                )

        for horizon_s in (5, 15):
            end_ts = min(anchor.anchor_ts_ns + horizon_s * SECOND_NS, anchor.market_close_ts_ns)
            start = bisect_right(self.times, int(anchor.anchor_ts_ns))
            end = bisect_right(self.times, int(end_ts))
            mids = self.mid_np[start:end]
            mids = mids[~np.isnan(mids)]
            if len(mids) and not math.isnan(anchor_mid):
                mfe = max(float(np.max(mids - anchor_mid)), 0.0)
                mae = max(float(np.max(anchor_mid - mids)), 0.0)
                assert mfe >= 0.0, f"MFE negative: {mfe}"
                assert mae >= 0.0, f"MAE negative: {mae}"
                targets[f"max_favorable_excursion_{horizon_s}s"] = mfe
                targets[f"max_adverse_excursion_{horizon_s}s"] = mae
            else:
                targets[f"max_favorable_excursion_{horizon_s}s"] = math.nan
                targets[f"max_adverse_excursion_{horizon_s}s"] = math.nan
        payout = 1.0 if anchor.resolved_side == anchor.token_side_normalized else 0.0
        targets["hold_to_close_edge_vs_mid"] = float(payout - anchor_mid) if not math.isnan(anchor_mid) else math.nan
        targets["terminal_margin_to_strike"] = anchor.terminal_margin_to_strike
        targets["flip_happened_after_t"] = anchor.flip_happened_after_t
        return targets, availability


def _date_range(date_from: date, date_to: date) -> list[date]:
    values: list[date] = []
    current = date_from
    while current <= date_to:
        values.append(current)
        current += timedelta(days=1)
    return values


def _nan(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _bool(value: Any) -> bool:
    return bool(value) if not pd.isna(value) else False


def _sign(value: float) -> int:
    if math.isnan(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def _target_available(*, t_to_close_s: float, mid_move: float, horizon_s: int) -> bool:
    if t_to_close_s <= horizon_s:
        return False
    if math.isnan(mid_move):
        return False
    return True


def _sequence_id(*, market_slug: str, token_asset_id: str, anchor_ts_ns: int, anchor_source: str, n_events: int | None = None) -> str:
    digest = hashlib.sha1(f"{market_slug}|{token_asset_id}|{anchor_ts_ns}|{anchor_source}".encode()).hexdigest()
    return digest[:24]


def _column_contract(column: str) -> dict[str, Any]:
    list_columns = {
        "events_recv_ts_ns": "list<int64>",
        "events_dt_from_anchor_s": "list<float64>",
        "events_dt_from_prev_s": "list<float64>",
        "events_event_type": "list<string>",
        "events_best_bid": "list<float64>",
        "events_best_ask": "list<float64>",
        "events_mid": "list<float64>",
        "events_spread": "list<float64>",
        "events_microprice": "list<float64>",
        "events_bid_depth_total": "list<float64>",
        "events_ask_depth_total": "list<float64>",
        "events_imbalance": "list<float64>",
        "events_last_trade_price": "list<float64>",
        "events_trade_size": "list<float64>",
        "events_trade_side": "list<string>",
        "events_tick_size": "list<float64>",
    }
    dtype = list_columns.get(column)
    if dtype is None:
        if column in {"anchor_ts_ns"}:
            dtype = "int64"
        elif column.startswith("event_count") or column.startswith("quote_update_count") or column.startswith("trade_count"):
            dtype = "int32"
        elif column in {"delta_sign", "mid_move_sign_1s", "mid_move_sign_3s", "mid_move_sign_5s", "mid_move_sign_15s", "mid_move_sign_30s"}:
            dtype = "int8"
        elif column == "sequence_length_actual":
            dtype = "int16"
        elif column in {
            "sequence_padded",
            "one_sided_book",
            "crossed_book",
            "stale_book",
            "live_reference_trusted",
            "flip_happened_after_t",
            "sequence_feature_eligible",
            "sequence_complete_eligible",
            "target_1s_available",
            "target_3s_available",
            "target_5s_available",
            "target_15s_available",
            "target_30s_available",
        }:
            dtype = "bool"
        elif column in {
            "sequence_id",
            "market_slug",
            "token_asset_id",
            "token_side_normalized",
            "anchor_source",
            "phase_bucket",
            "live_bias_mode",
            "microstructure_exclusion_reason",
        }:
            dtype = "string"
        else:
            dtype = "float64"
    leakage_only = column in TARGET_COLUMNS
    training_eligible = not leakage_only and column not in {
        "sequence_id",
        "market_slug",
        "token_asset_id",
        "token_side_normalized",
        "anchor_ts_ns",
        "anchor_source",
        "sequence_feature_eligible",
        "sequence_complete_eligible",
        "target_1s_available",
        "target_3s_available",
        "target_5s_available",
        "target_15s_available",
        "target_30s_available",
        "microstructure_exclusion_reason",
    }
    source = "polymarket_l2"
    if column in {
        "price_to_beat",
        "live_btc_usd",
        "delta_to_strike",
        "abs_delta_to_strike",
        "delta_sign",
        "t_since_open_s",
        "t_to_close_s",
        "phase_bucket",
        "live_reference_trusted",
        "live_bias_mode",
        "live_applied_bias_age_seconds",
        "terminal_margin_to_strike",
        "flip_happened_after_t",
    }:
        source = "phase3_anchor"
    elif column in TARGET_COLUMNS:
        source = "future_l2_label"
    elif column.startswith("sequence_") or column.startswith("target_") or column == "microstructure_exclusion_reason":
        source = "derived_quality"
    elif column in {"sequence_id", "market_slug", "token_asset_id", "token_side_normalized", "anchor_ts_ns", "anchor_source"}:
        source = "derived_identity"
    description = f"Phase 8 microstructure sequence dataset column `{column}`."
    if column == "events_event_type":
        description = (
            "Normalized event type for each event in sequence. Valid values: book_state, "
            "top_of_book, trade, tick_update, other, pad. Value 'pad' indicates a padding "
            "position at the start of the sequence for samples with fewer than N actual events."
        )
    elif column == "sequence_id":
        description = (
            "Stable sample identifier shared across tabular, event64, and event128 variants for "
            "the same market_slug, token_asset_id, anchor_ts_ns, and anchor_source."
        )
    return {
        "dtype": dtype,
        "nullable": bool(dtype.startswith("float") or dtype.startswith("list") or column == "microstructure_exclusion_reason"),
        "leakage_only": leakage_only,
        "training_eligible": training_eligible,
        "source": source,
        "description": description,
    }


def write_microstructure_schema(schema_path: Path = DEFAULT_SCHEMA_PATH) -> Path:
    schema_path = Path(schema_path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": "microstructure_sequence_dataset_v1",
        "notes": [
            "sequence_length_actual is int16 because event128 cannot be represented by signed int8.",
            "Target columns are labels only and must remain leakage_only=true.",
            "tabular stores only aggregated tabular columns; event64/event128 store tabular columns plus sequence arrays.",
            "events_event_type uses pad as an explicit padding sentinel for prepended padding positions.",
        ],
        "columns": {column: _column_contract(column) for column in SCHEMA_COLUMNS},
    }
    tmp_path = schema_path.with_name(f"{schema_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    tmp_path.replace(schema_path)
    return schema_path


def load_schema_columns(schema_path: Path = DEFAULT_SCHEMA_PATH) -> list[str]:
    payload = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Schema contract is empty or invalid: {schema_path}")
    return list((payload.get("columns") or {}).keys())


def assert_schema_columns(frame_columns: list[str], schema_path: Path = DEFAULT_SCHEMA_PATH) -> None:
    expected = load_schema_columns(schema_path)
    if frame_columns != expected:
        missing = sorted(set(expected) - set(frame_columns))
        extra = sorted(set(frame_columns) - set(expected))
        raise ValueError(f"Phase 8 schema mismatch. missing={missing} extra={extra}")


def _load_token_registry(root: Path) -> dict[str, list[tuple[str, str]]]:
    frame = pd.read_parquet(root / "canonical" / "token_registry_v1.parquet")
    mapping: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        side = str(row.token_side_normalized).upper()
        if side in {"UP", "DOWN"}:
            mapping[str(row.market_slug)].append((str(row.token_asset_id), side))
    return mapping


def _read_phase3_anchors_for_date(root: Path, target_date: date, dataset_output_dir: Path) -> pd.DataFrame:
    sources = [
        ("coarse", dataset_output_dir / "resolution_snapshot_dataset_v1_coarse" / f"{target_date.isoformat()}.parquet"),
        ("dense_close", dataset_output_dir / "resolution_snapshot_dataset_v1_dense_close" / f"{target_date.isoformat()}.parquet"),
    ]
    columns = [
        "snapshot_ts_ns",
        "market_slug",
        "market_close_ts_ns",
        "resolved_side",
        "price_to_beat",
        "live_btc_usd",
        "delta_to_strike",
        "abs_delta_to_strike",
        "delta_sign",
        "t_since_open_s",
        "t_to_close_s",
        "phase_bucket",
        "live_reference_trusted",
        "live_price_valid",
        "live_bias_mode",
        "live_applied_bias_age_seconds",
        "training_feature_eligible",
        "exclusion_reason",
        "terminal_margin_to_strike",
        "flip_happened_after_t",
    ]
    frames: list[pd.DataFrame] = []
    for source, path in sources:
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=columns)
        frame["anchor_source"] = source
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    anchors = pd.concat(frames, ignore_index=True)
    anchors = anchors.drop_duplicates(["market_slug", "snapshot_ts_ns", "anchor_source"]).reset_index(drop=True)
    return anchors


def _expand_token_anchors(phase3_frame: pd.DataFrame, token_registry: dict[str, list[tuple[str, str]]]) -> list[AnchorRow]:
    anchors: list[AnchorRow] = []
    seen: set[tuple[str, str, int, str]] = set()
    for row in phase3_frame.itertuples(index=False):
        market_slug = str(row.market_slug)
        for token_asset_id, side in token_registry.get(market_slug, []):
            anchor_ts_ns = int(row.snapshot_ts_ns)
            anchor_source = str(row.anchor_source)
            key = (market_slug, token_asset_id, anchor_ts_ns, anchor_source)
            if key in seen:
                continue
            seen.add(key)
            anchors.append(
                AnchorRow(
                    market_slug=market_slug,
                    token_asset_id=token_asset_id,
                    token_side_normalized=side,
                    anchor_ts_ns=anchor_ts_ns,
                    anchor_source=anchor_source,
                    market_close_ts_ns=int(row.market_close_ts_ns),
                    resolved_side=str(row.resolved_side).upper(),
                    price_to_beat=_nan(row.price_to_beat),
                    live_btc_usd=_nan(row.live_btc_usd),
                    delta_to_strike=_nan(row.delta_to_strike),
                    abs_delta_to_strike=_nan(row.abs_delta_to_strike),
                    delta_sign=int(row.delta_sign) if not pd.isna(row.delta_sign) else 0,
                    t_since_open_s=_nan(row.t_since_open_s),
                    t_to_close_s=_nan(row.t_to_close_s),
                    phase_bucket=str(row.phase_bucket),
                    live_reference_trusted=_bool(row.live_reference_trusted),
                    live_price_valid=_bool(row.live_price_valid),
                    live_bias_mode=str(row.live_bias_mode),
                    live_applied_bias_age_seconds=_nan(row.live_applied_bias_age_seconds),
                    training_feature_eligible=_bool(row.training_feature_eligible),
                    exclusion_reason=str(row.exclusion_reason or ""),
                    terminal_margin_to_strike=_nan(row.terminal_margin_to_strike),
                    flip_happened_after_t=_bool(row.flip_happened_after_t),
                )
            )
    return anchors


def _pad_sequence(sequence: pd.DataFrame, *, anchor_ts_ns: int, n_events: int) -> pd.DataFrame:
    missing = n_events - len(sequence)
    if missing <= 0:
        return sequence
    pad_rows = [
        {
            "recv_ts_ns": 0,
            "event_type": "pad",
            "best_bid": math.nan,
            "best_ask": math.nan,
            "mid": math.nan,
            "spread": math.nan,
            "microprice": math.nan,
            "bid_depth_total": math.nan,
            "ask_depth_total": math.nan,
            "imbalance": 0.0,
            "last_trade_price": math.nan,
            "trade_size": 0.0,
            "trade_side": "none",
            "tick_size": math.nan,
        }
        for _ in range(missing)
    ]
    padded = pd.concat([pd.DataFrame(pad_rows), sequence], ignore_index=True)
    return padded


def _sequence_arrays(sequence: pd.DataFrame, *, anchor_ts_ns: int, n_events: int) -> dict[str, list[Any]]:
    actual = len(sequence)
    sequence = sequence.copy()
    if actual:
        sequence["events_dt_from_anchor_s"] = (sequence["recv_ts_ns"].astype("int64") - int(anchor_ts_ns)) / SECOND_NS
        dt_prev = sequence["recv_ts_ns"].astype("int64").diff().fillna(0) / SECOND_NS
        sequence["events_dt_from_prev_s"] = dt_prev
    sequence = _pad_sequence(sequence, anchor_ts_ns=anchor_ts_ns, n_events=n_events)
    pad_count = n_events - actual
    if pad_count > 0:
        sequence.loc[: pad_count - 1, "events_dt_from_anchor_s"] = -1_000_000_000.0
        sequence.loc[: pad_count - 1, "events_dt_from_prev_s"] = 0.0
    for source, target in [
        ("recv_ts_ns", "events_recv_ts_ns"),
        ("event_type", "events_event_type"),
        ("best_bid", "events_best_bid"),
        ("best_ask", "events_best_ask"),
        ("mid", "events_mid"),
        ("spread", "events_spread"),
        ("microprice", "events_microprice"),
        ("bid_depth_total", "events_bid_depth_total"),
        ("ask_depth_total", "events_ask_depth_total"),
        ("imbalance", "events_imbalance"),
        ("last_trade_price", "events_last_trade_price"),
        ("trade_size", "events_trade_size"),
        ("trade_side", "events_trade_side"),
        ("tick_size", "events_tick_size"),
    ]:
        if source not in sequence.columns:
            sequence[source] = math.nan
    return {
        "events_recv_ts_ns": [int(value) if not pd.isna(value) else 0 for value in sequence["recv_ts_ns"].tolist()],
        "events_dt_from_anchor_s": [float(value) if not pd.isna(value) else math.nan for value in sequence["events_dt_from_anchor_s"].tolist()],
        "events_dt_from_prev_s": [float(value) if not pd.isna(value) else 0.0 for value in sequence["events_dt_from_prev_s"].tolist()],
        "events_event_type": [str(value) for value in sequence["event_type"].fillna("other").tolist()],
        "events_best_bid": [_nan(value) for value in sequence["best_bid"].tolist()],
        "events_best_ask": [_nan(value) for value in sequence["best_ask"].tolist()],
        "events_mid": [_nan(value) for value in sequence["mid"].tolist()],
        "events_spread": [_nan(value) for value in sequence["spread"].tolist()],
        "events_microprice": [_nan(value) for value in sequence["microprice"].tolist()],
        "events_bid_depth_total": [_nan(value) for value in sequence["bid_depth_total"].tolist()],
        "events_ask_depth_total": [_nan(value) for value in sequence["ask_depth_total"].tolist()],
        "events_imbalance": [_nan(value) for value in sequence["imbalance"].tolist()],
        "events_last_trade_price": [_nan(value) for value in sequence["last_trade_price"].tolist()],
        "events_trade_size": [_nan(value) for value in sequence["trade_size"].tolist()],
        "events_trade_side": [str(value) for value in sequence["trade_side"].fillna("none").tolist()],
        "events_tick_size": [_nan(value) for value in sequence["tick_size"].tolist()],
    }


def _window(history: pd.DataFrame, anchor_ts_ns: int, seconds: int) -> pd.DataFrame:
    if history.empty:
        return history
    return history.loc[history["recv_ts_ns"] >= anchor_ts_ns - (seconds * SECOND_NS)]


def _latest_mid_at_or_before(history: pd.DataFrame, ts_ns: int) -> float:
    if history.empty:
        return math.nan
    values = history.loc[history["recv_ts_ns"] <= ts_ns, "mid"].dropna()
    return float(values.iloc[-1]) if len(values) else math.nan


def compute_tabular_features(index: TokenEventIndex, history: pd.DataFrame, anchor_ts_ns: int) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for seconds in (1, 3, 5, 15):
        w = _window(history, anchor_ts_ns, seconds)
        features[f"event_count_{seconds}s"] = int(len(w))
    for seconds in (1, 3, 5):
        w = _window(history, anchor_ts_ns, seconds)
        features[f"quote_update_count_{seconds}s"] = int(w["event_type"].isin(["book_state", "top_of_book"]).sum()) if not w.empty else 0
        trades = w.loc[w["event_type"].eq("trade")] if not w.empty else w
        features[f"trade_count_{seconds}s"] = int(len(trades))
        if trades.empty:
            features[f"signed_trade_size_{seconds}s"] = 0.0
            features[f"absolute_trade_size_{seconds}s"] = 0.0
        else:
            signs = trades["trade_side"].map({"buy": 1.0, "sell": -1.0}).fillna(0.0)
            sizes = trades["trade_size"].fillna(0.0).astype(float)
            features[f"signed_trade_size_{seconds}s"] = float((signs * sizes).sum())
            features[f"absolute_trade_size_{seconds}s"] = float(sizes.abs().sum())

    w5 = _window(history, anchor_ts_ns, 5)
    spreads = w5["spread"].dropna() if not w5.empty else pd.Series(dtype="float64")
    features["spread_mean_5s"] = float(spreads.mean()) if len(spreads) else math.nan
    features["spread_min_5s"] = float(spreads.min()) if len(spreads) else math.nan
    features["spread_max_5s"] = float(spreads.max()) if len(spreads) else math.nan
    imbalances = w5["imbalance"].dropna() if not w5.empty else pd.Series(dtype="float64")
    features["imbalance_mean_5s"] = float(imbalances.mean()) if len(imbalances) else math.nan
    features["imbalance_last"] = float(history["imbalance"].dropna().iloc[-1]) if not history.empty and len(history["imbalance"].dropna()) else math.nan
    if len(imbalances) >= 2:
        imbalance_times = w5.loc[w5["imbalance"].notna(), "recv_ts_ns"]
        duration_s = max((int(imbalance_times.iloc[-1]) - int(imbalance_times.iloc[0])) / SECOND_NS, 1e-9)
        features["imbalance_slope_5s"] = float((imbalances.iloc[-1] - imbalances.iloc[0]) / duration_s)
    else:
        features["imbalance_slope_5s"] = 0.0

    latest_mid = _latest_mid_at_or_before(history, anchor_ts_ns)
    for seconds in (1, 3, 5):
        past_mid = _latest_mid_at_or_before(history, anchor_ts_ns - seconds * SECOND_NS)
        features[f"mid_return_{seconds}s"] = (
            float(latest_mid - past_mid)
            if not math.isnan(latest_mid) and not math.isnan(past_mid)
            else math.nan
        )
    features["quote_churn_rate_5s"] = float(features["quote_update_count_5s"] / 5.0)
    if not w5.empty and len(w5[["bid_depth_total", "ask_depth_total"]].dropna()) >= 2:
        depth = (w5["bid_depth_total"].fillna(0.0) + w5["ask_depth_total"].fillna(0.0)).astype(float)
        features["depth_change_rate_5s"] = float((depth.iloc[-1] - depth.iloc[0]) / 5.0)
    else:
        features["depth_change_rate_5s"] = 0.0

    latest = history.tail(1)
    if latest.empty:
        features.update({"one_sided_book": True, "crossed_book": False, "stale_book": True, "sequence_completeness_rate": 0.0})
    else:
        row = latest.iloc[0]
        best_bid = _nan(row.get("best_bid"))
        best_ask = _nan(row.get("best_ask"))
        bid_depth = _nan(row.get("bid_depth_total"))
        ask_depth = _nan(row.get("ask_depth_total"))
        features["crossed_book"] = bool(not math.isnan(best_bid) and not math.isnan(best_ask) and best_bid >= best_ask)
        features["stale_book"] = bool(anchor_ts_ns - int(row["recv_ts_ns"]) > 30 * SECOND_NS)
        features["one_sided_book"] = bool(
            math.isnan(best_bid)
            or math.isnan(best_ask)
            or math.isnan(bid_depth)
            or math.isnan(ask_depth)
            or bid_depth <= 0
            or ask_depth <= 0
        )
        features["sequence_completeness_rate"] = float(len(_window(history, anchor_ts_ns, 30)) / max(index.expected_rate_30s, 1.0))
    return features


def compute_targets(index: TokenEventIndex, anchor: AnchorRow, anchor_mid: float) -> tuple[dict[str, Any], dict[int, bool]]:
    targets: dict[str, Any] = {}
    availability: dict[int, bool] = {}
    for horizon_s in (1, 3, 5, 15, 30):
        target_ts = anchor.anchor_ts_ns + horizon_s * SECOND_NS
        if target_ts > anchor.market_close_ts_ns or math.isnan(anchor_mid):
            mid_move = math.nan
        else:
            future = index.future_after_until(anchor.anchor_ts_ns, target_ts)
            future_mids = future["mid"].dropna() if not future.empty else pd.Series(dtype="float64")
            if len(future_mids):
                mid_move = float(future_mids.iloc[-1] - anchor_mid)
            else:
                mid_move = math.nan
        available = _target_available(t_to_close_s=anchor.t_to_close_s, mid_move=mid_move, horizon_s=horizon_s)
        if not available:
            mid_move = math.nan
        targets[f"mid_move_{horizon_s}s"] = mid_move
        targets[f"mid_move_sign_{horizon_s}s"] = _sign(mid_move)
        availability[horizon_s] = available
        if not availability[horizon_s]:
            assert math.isnan(targets[f"mid_move_{horizon_s}s"]), (
                f"target_{horizon_s}s_available=False but mid_move_{horizon_s}s is not NaN"
            )
        if anchor.t_to_close_s <= horizon_s:
            assert not availability[horizon_s], (
                f"target_{horizon_s}s_available=True but t_to_close_s={anchor.t_to_close_s} <= {horizon_s}"
            )

    for horizon_s in (5, 15):
        end_ts = min(anchor.anchor_ts_ns + horizon_s * SECOND_NS, anchor.market_close_ts_ns)
        window = index.future_after_until(anchor.anchor_ts_ns, end_ts)
        mids = window["mid"].dropna() if not window.empty else pd.Series(dtype="float64")
        if len(mids) and not math.isnan(anchor_mid):
            mfe = max(float((mids - anchor_mid).max()), 0.0)
            mae = max(float((anchor_mid - mids).max()), 0.0)
            assert mfe >= 0.0, f"MFE negative: {mfe}"
            assert mae >= 0.0, f"MAE negative: {mae}"
            targets[f"max_favorable_excursion_{horizon_s}s"] = mfe
            targets[f"max_adverse_excursion_{horizon_s}s"] = mae
        else:
            targets[f"max_favorable_excursion_{horizon_s}s"] = math.nan
            targets[f"max_adverse_excursion_{horizon_s}s"] = math.nan
    payout = 1.0 if anchor.resolved_side == anchor.token_side_normalized else 0.0
    targets["hold_to_close_edge_vs_mid"] = float(payout - anchor_mid) if not math.isnan(anchor_mid) else math.nan
    targets["terminal_margin_to_strike"] = anchor.terminal_margin_to_strike
    targets["flip_happened_after_t"] = anchor.flip_happened_after_t
    return targets, availability


def compute_eligibility(anchor: AnchorRow, sequence_actual: int, tabular: dict[str, Any], max_sequence_gap_s: float) -> tuple[bool, str]:
    if not anchor.training_feature_eligible:
        return False, anchor.exclusion_reason or "phase3_anchor_ineligible"
    if not anchor.token_asset_id:
        return False, "token_mapping_missing"
    if not anchor.live_price_valid:
        return False, "live_reference_unavailable"
    if not anchor.live_reference_trusted:
        return False, "live_reference_untrusted"
    if sequence_actual < 10:
        return False, "insufficient_event_history"
    if max_sequence_gap_s > 60.0:
        return False, "sequence_time_gap_too_large"
    if bool(tabular.get("crossed_book")):
        return False, "crossed_book"
    if bool(tabular.get("stale_book")):
        return False, "stale_book"
    return True, ""


def _row_from_anchor_with_base(
    anchor: AnchorRow,
    index: TokenEventIndex,
    n_events: int,
    *,
    history: pd.DataFrame,
    tabular: dict[str, Any],
    targets: dict[str, Any],
    target_available: dict[int, bool],
) -> dict[str, Any]:
    sequence, complete = index.get_event_sequence(anchor.anchor_ts_ns, n_events)
    if not sequence.empty and not (sequence["recv_ts_ns"].astype("int64") < anchor.anchor_ts_ns).all():
        raise ValueError("Future event in Phase 8 sequence")
    actual = int(len(sequence))
    if actual > 1:
        max_gap_s = float(sequence["recv_ts_ns"].astype("int64").diff().dropna().max() / SECOND_NS)
    else:
        max_gap_s = 0.0
    eligible, reason = compute_eligibility(anchor, actual, tabular, max_gap_s)
    arrays = _sequence_arrays(sequence, anchor_ts_ns=anchor.anchor_ts_ns, n_events=n_events)
    row = {
        "sequence_id": _sequence_id(
            market_slug=anchor.market_slug,
            token_asset_id=anchor.token_asset_id,
            anchor_ts_ns=anchor.anchor_ts_ns,
            anchor_source=anchor.anchor_source,
            n_events=n_events,
        ),
        "market_slug": anchor.market_slug,
        "token_asset_id": anchor.token_asset_id,
        "token_side_normalized": anchor.token_side_normalized,
        "anchor_ts_ns": anchor.anchor_ts_ns,
        "anchor_source": anchor.anchor_source,
        **arrays,
        "sequence_length_actual": actual,
        "sequence_padded": bool(actual < n_events),
        **tabular,
        "price_to_beat": anchor.price_to_beat,
        "live_btc_usd": anchor.live_btc_usd,
        "delta_to_strike": anchor.delta_to_strike,
        "abs_delta_to_strike": anchor.abs_delta_to_strike,
        "delta_sign": int(anchor.delta_sign),
        "t_since_open_s": anchor.t_since_open_s,
        "t_to_close_s": anchor.t_to_close_s,
        "phase_bucket": anchor.phase_bucket,
        "live_reference_trusted": anchor.live_reference_trusted,
        "live_bias_mode": anchor.live_bias_mode,
        "live_applied_bias_age_seconds": anchor.live_applied_bias_age_seconds,
        **targets,
        "sequence_feature_eligible": eligible,
        "sequence_complete_eligible": bool(eligible and complete),
        "target_1s_available": target_available[1],
        "target_3s_available": target_available[3],
        "target_5s_available": target_available[5],
        "target_15s_available": target_available[15],
        "target_30s_available": target_available[30],
        "microstructure_exclusion_reason": reason,
    }
    return {column: row.get(column) for column in SCHEMA_COLUMNS}


def _row_from_anchor_precomputed(
    anchor: AnchorRow,
    index: TokenEventIndex,
    n_events: int,
    *,
    tabular: dict[str, Any],
    targets: dict[str, Any],
    target_available: dict[int, bool],
) -> dict[str, Any]:
    arrays, actual, complete, max_gap_s = index.sequence_payload(anchor.anchor_ts_ns, n_events)
    if any(event_ts and event_ts >= anchor.anchor_ts_ns for event_ts in arrays["events_recv_ts_ns"]):
        raise ValueError("Future event in Phase 8 sequence")
    eligible, reason = compute_eligibility(anchor, actual, tabular, max_gap_s)
    row = {
        "sequence_id": _sequence_id(
            market_slug=anchor.market_slug,
            token_asset_id=anchor.token_asset_id,
            anchor_ts_ns=anchor.anchor_ts_ns,
            anchor_source=anchor.anchor_source,
            n_events=n_events,
        ),
        "market_slug": anchor.market_slug,
        "token_asset_id": anchor.token_asset_id,
        "token_side_normalized": anchor.token_side_normalized,
        "anchor_ts_ns": anchor.anchor_ts_ns,
        "anchor_source": anchor.anchor_source,
        **arrays,
        "sequence_length_actual": actual,
        "sequence_padded": bool(actual < n_events),
        **tabular,
        "price_to_beat": anchor.price_to_beat,
        "live_btc_usd": anchor.live_btc_usd,
        "delta_to_strike": anchor.delta_to_strike,
        "abs_delta_to_strike": anchor.abs_delta_to_strike,
        "delta_sign": int(anchor.delta_sign),
        "t_since_open_s": anchor.t_since_open_s,
        "t_to_close_s": anchor.t_to_close_s,
        "phase_bucket": anchor.phase_bucket,
        "live_reference_trusted": anchor.live_reference_trusted,
        "live_bias_mode": anchor.live_bias_mode,
        "live_applied_bias_age_seconds": anchor.live_applied_bias_age_seconds,
        **targets,
        "sequence_feature_eligible": eligible,
        "sequence_complete_eligible": bool(eligible and complete),
        "target_1s_available": target_available[1],
        "target_3s_available": target_available[3],
        "target_5s_available": target_available[5],
        "target_15s_available": target_available[15],
        "target_30s_available": target_available[30],
        "microstructure_exclusion_reason": reason,
    }
    return {column: row.get(column) for column in SCHEMA_COLUMNS}


def _row_from_anchor(anchor: AnchorRow, index: TokenEventIndex, n_events: int) -> dict[str, Any]:
    tabular = index.tabular_features(anchor.anchor_ts_ns)
    anchor_mid = index.latest_mid_before(anchor.anchor_ts_ns)
    targets, target_available = index.targets(anchor, anchor_mid)
    return _row_from_anchor_precomputed(
        anchor,
        index,
        n_events,
        tabular=tabular,
        targets=targets,
        target_available=target_available,
    )


def _rows_from_anchor(anchor: AnchorRow, index: TokenEventIndex) -> tuple[dict[str, Any], dict[str, Any]]:
    tabular = index.tabular_features(anchor.anchor_ts_ns)
    anchor_mid = index.latest_mid_before(anchor.anchor_ts_ns)
    targets, target_available = index.targets(anchor, anchor_mid)
    row64 = _row_from_anchor_precomputed(
        anchor,
        index,
        64,
        tabular=tabular,
        targets=targets,
        target_available=target_available,
    )
    row128 = _row_from_anchor_precomputed(
        anchor,
        index,
        128,
        tabular=tabular,
        targets=targets,
        target_available=target_available,
    )
    return row64, row128


def _write_rows(writer: pq.ParquetWriter, rows: list[dict[str, Any]], schema: pa.Schema = PA_SCHEMA) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table)


def _empty_dataset(path: Path, schema: pa.Schema = PA_SCHEMA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, path, compression="zstd")


def _market_indices(events: pd.DataFrame) -> dict[str, TokenEventIndex]:
    if events.empty:
        return {}
    return {
        str(token_asset_id): TokenEventIndex(group.copy())
        for token_asset_id, group in events.groupby("token_asset_id", sort=False)
    }


def build_microstructure_day(
    *,
    root: Path,
    target_date: date,
    policy_path: Path,
    dataset_output_dir: Path = DEFAULT_DATASET_OUTPUT_DIR,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
    sequence_lengths: list[int] | None = None,
) -> dict[str, Any]:
    sequence_lengths = sequence_lengths or [64, 128]
    if sequence_lengths != [64, 128]:
        raise ValueError("Phase 8 v1 currently expects sequence_lengths=[64, 128]")
    write_microstructure_schema(schema_path)
    assert_schema_columns(SCHEMA_COLUMNS, schema_path)

    phase3_anchors = _read_phase3_anchors_for_date(root, target_date, dataset_output_dir)
    token_registry = _load_token_registry(root)
    anchors = _expand_token_anchors(phase3_anchors, token_registry)
    anchors_by_market: dict[str, list[AnchorRow]] = defaultdict(list)
    for anchor in anchors:
        anchors_by_market[anchor.market_slug].append(anchor)

    output_paths = {
        "tabular": dataset_output_dir / TABULAR_DATASET / f"{target_date.isoformat()}.parquet",
        "event64": dataset_output_dir / EVENT64_DATASET / f"{target_date.isoformat()}.parquet",
        "event128": dataset_output_dir / EVENT128_DATASET / f"{target_date.isoformat()}.parquet",
    }
    if all(path.exists() for path in output_paths.values()):
        try:
            existing_datasets: dict[str, dict[str, Any]] = {}
            for key, path in output_paths.items():
                columns = [
                    "sequence_feature_eligible",
                    "sequence_complete_eligible",
                    "target_1s_available",
                    "target_3s_available",
                    "target_5s_available",
                    "target_15s_available",
                    "target_30s_available",
                ]
                frame = pd.read_parquet(path, columns=columns)
                existing_datasets[key] = {
                    "rows": int(len(frame)),
                    "eligible": int(frame["sequence_feature_eligible"].sum()),
                    "complete": int(frame["sequence_complete_eligible"].sum()),
                    "target_1s_available": int(frame["target_1s_available"].sum()),
                    "target_3s_available": int(frame["target_3s_available"].sum()),
                    "target_5s_available": int(frame["target_5s_available"].sum()),
                    "target_15s_available": int(frame["target_15s_available"].sum()),
                    "target_30s_available": int(frame["target_30s_available"].sum()),
                    "path": str(path),
                    "reused_existing": True,
                }
            return {
                "date": target_date.isoformat(),
                "phase3_anchor_rows": int(len(phase3_anchors)),
                "token_anchor_rows": int(len(anchors)),
                "datasets": existing_datasets,
                "l2_exclusions": {},
                "reused_existing": True,
            }
        except Exception:
            pass
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    output_schemas = {"tabular": TABULAR_PA_SCHEMA, "event64": PA_SCHEMA, "event128": PA_SCHEMA}
    writers = {
        key: pq.ParquetWriter(path, output_schemas[key], compression="zstd", use_dictionary=True)
        for key, path in output_paths.items()
    }
    summaries = {
        "tabular": Counter(),
        "event64": Counter(),
        "event128": Counter(),
    }
    exclusion_summary: dict[str, Counter] = defaultdict(Counter)
    try:
        for market_slug in sorted(anchors_by_market):
            load_result = load_l2_events(
                root=root,
                target_date=target_date,
                policy_path=policy_path,
                quality_registry_path=quality_registry_path,
                market_slug=market_slug,
            )
            for key, counter in load_result.exclusion_summary.items():
                exclusion_summary[key].update(counter)
            indices = _market_indices(load_result.events)
            rows_by_dataset = {"tabular": [], "event64": [], "event128": []}
            for anchor in anchors_by_market[market_slug]:
                index = indices.get(anchor.token_asset_id, TokenEventIndex(pd.DataFrame()))
                row64, row128 = _rows_from_anchor(anchor, index)
                rows_by_dataset["tabular"].append(row64)
                rows_by_dataset["event64"].append(row64)
                rows_by_dataset["event128"].append(row128)
            for key, rows in rows_by_dataset.items():
                _write_rows(writers[key], rows, output_schemas[key])
                summaries[key]["rows"] += len(rows)
                summaries[key]["eligible"] += sum(1 for row in rows if row["sequence_feature_eligible"])
                summaries[key]["complete"] += sum(1 for row in rows if row["sequence_complete_eligible"])
                for horizon in (1, 3, 5, 15, 30):
                    summaries[key][f"target_{horizon}s_available"] += sum(
                        1 for row in rows if row[f"target_{horizon}s_available"]
                    )
        return {
            "date": target_date.isoformat(),
            "phase3_anchor_rows": int(len(phase3_anchors)),
            "token_anchor_rows": int(len(anchors)),
            "datasets": {
                key: {
                    **dict(counter),
                    "path": str(output_paths[key]),
                }
                for key, counter in summaries.items()
            },
            "l2_exclusions": {key: dict(counter) for key, counter in exclusion_summary.items()},
        }
    finally:
        for writer in writers.values():
            writer.close()
        for key, path in output_paths.items():
            if not path.exists():
                _empty_dataset(path, output_schemas[key])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_factory_report(report: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Phase 8 Microstructure Dataset Factory Report",
        "",
        f"- Date range: `{report['date_from']}` to `{report['date_to']}`",
        "- Scope: dataset factory only; no model training, execution simulator, or policy layer.",
        f"- Schema: `{report['schema_path']}`",
        "",
        "## Dataset Summary",
        "",
        "| date | dataset | rows | eligible | eligible_pct | complete | complete_pct | path | sha256 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for day in report["daily"]:
        for dataset_name, summary in day["datasets"].items():
            rows = int(summary.get("rows", 0))
            eligible = int(summary.get("eligible", 0))
            complete = int(summary.get("complete", 0))
            path = Path(summary["path"])
            lines.append(
                f"| {day['date']} | {dataset_name} | {rows} | {eligible} | "
                f"{(eligible / rows if rows else 0):.4f} | {complete} | {(complete / rows if rows else 0):.4f} | "
                f"`{path}` | `{_sha256(path) if path.exists() else ''}` |"
            )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "- Phase 8 uses Phase 3 coarse and dense_close rows as anchors.",
            "- Each Phase 3 anchor is expanded to UP and DOWN token samples using `token_registry_v1`.",
            "- `event64` and `event128` are materialized; tensor export is deferred.",
            "- `tabular` is materialized as pure aggregated tabular features; `event64` and `event128` carry tabular features plus sequence arrays.",
            "- `sequence_length_actual` is stored as `int16` because signed `int8` cannot represent 128.",
            "- `mid_return_*` features are absolute token-mid changes; large jumps with `abs(value) >= 0.5` are retained because they are real BTC 5m UP/DOWN tail events, not automatic training exclusions.",
            "- Phase 8 model training remains deferred until sufficient data volume accumulates.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_decision_log(decision_log_path: Path) -> None:
    entry = """\n## 2026-04-25 - Phase 8 Microstructure Sequence Dataset Build\n\n- Task: build the dataset factory component of Phase 8 for Model 7 and Model 8 without training models.\n- Decision: `microstructure_sequence_dataset_v1` is built from Phase 3 coarse + dense_close anchors, expanded to UP/DOWN token samples through `token_registry_v1`, and materialized as tabular/event64/event128 daily parquet shards.\n- Decision: tensor export is deferred until the parquet sequence contract is audited on the current five-day window.\n- Decision: `sequence_length_actual` is stored as `int16` instead of signed `int8` because event128 requires the value `128`, which signed int8 cannot represent.\n- Decision: Phase 8 inherits Phase 3 carried-trusted training eligibility instead of re-introducing a strict fresh-bias age gate; carried-vs-fresh state remains available through `live_bias_mode` and `live_applied_bias_age_seconds`.\n- Decision: `mid_return_1s`, `mid_return_3s`, and `mid_return_5s` are absolute token-mid changes. Large jumps with `abs(value) >= 0.5` are retained as trainable BTC 5m UP/DOWN tail events and reported by audit as warnings rather than exclusions.\n- Complete: Phase 8 dataset build complete. `microstructure_sequence_dataset_v1` built for 2026-04-19 to 2026-04-23. Anchor sources: Phase 3 coarse + dense_close snapshots. Sequence lengths: 64, 128. Tensor export deferred. Models 7 + 8 to be trained when sufficient data accumulated. Date: 2026-04-25.\n- Evidence: see `docs/phase8_microstructure_dataset_factory_report.md` and the daily `docs/phase8_microstructure_audit_YYYY-MM-DD.md` reports.\n"""
    decision_log_path.parent.mkdir(parents=True, exist_ok=True)
    current = decision_log_path.read_text(encoding="utf-8") if decision_log_path.exists() else "# Decision Log\n"
    if "Phase 8 Microstructure Sequence Dataset Build" not in current:
        decision_log_path.write_text(current.rstrip() + "\n" + entry, encoding="utf-8")


def build_phase8_datasets(
    *,
    root: Path,
    policy_path: Path,
    date_from: date,
    date_to: date,
    dataset_output_dir: Path = DEFAULT_DATASET_OUTPUT_DIR,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    decision_log_path: Path = DEFAULT_DECISION_LOG_PATH,
    max_workers: int = 1,
) -> dict[str, Any]:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    schema_written = write_microstructure_schema(schema_path)
    target_dates = _date_range(date_from, date_to)
    if max_workers == 1 or len(target_dates) <= 1:
        daily = [
            build_microstructure_day(
                root=root,
                target_date=target_date,
                policy_path=policy_path,
                dataset_output_dir=dataset_output_dir,
                schema_path=schema_written,
                quality_registry_path=quality_registry_path,
            )
            for target_date in target_dates
        ]
    else:
        daily_by_date: dict[str, dict[str, Any]] = {}
        worker_count = min(max_workers, len(target_dates))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    build_microstructure_day,
                    root=root,
                    target_date=target_date,
                    policy_path=policy_path,
                    dataset_output_dir=dataset_output_dir,
                    schema_path=schema_written,
                    quality_registry_path=quality_registry_path,
                ): target_date
                for target_date in target_dates
            }
            for future in as_completed(futures):
                target_date = futures[future]
                daily_by_date[target_date.isoformat()] = future.result()
        daily = [daily_by_date[target_date.isoformat()] for target_date in target_dates]
    report = {
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "schema_path": str(schema_written),
        "daily": daily,
    }
    _write_factory_report(report, report_path)
    _append_decision_log(decision_log_path)
    return report
