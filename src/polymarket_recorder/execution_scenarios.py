from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math
import pandas as pd
import pyarrow.parquet as pq

from .execution_replay import ExecutionReplayIndex, SECOND_NS
from .execution_simulator import OrderIntent


@dataclass(frozen=True, slots=True)
class ScenarioGrid:
    order_sides: tuple[str, ...] = ("buy", "sell")
    order_sizes: tuple[float, ...] = (10.0,)
    cancel_after_s: tuple[float | None, ...] = (0.5, 1.0, 3.0, None)
    include_replace_path: bool = True
    replace_after_s: float = 1.0
    include_join_best: bool = True
    include_improve_one_tick: bool = True
    include_inside_passive: bool = True
    scenario_name: str = "phase5_grid_v1"


def _round_price(price: float, tick_size: float) -> float:
    tick = tick_size if tick_size > 0 else 0.01
    return round(round(float(price) / tick) * tick, 6)


def _market_close_from_anchor(row: pd.Series) -> int:
    if "market_close_ts_ns" in row and not pd.isna(row["market_close_ts_ns"]):
        return int(row["market_close_ts_ns"])
    if "t_to_close_s" not in row:
        raise ValueError("Anchor row needs market_close_ts_ns or t_to_close_s")
    return int(row["anchor_ts_ns"]) + int(round(float(row["t_to_close_s"]) * SECOND_NS))


def _market_deploy_from_anchor(row: pd.Series) -> int:
    if "market_open_ts_ns" in row and not pd.isna(row["market_open_ts_ns"]):
        value = int(row["market_open_ts_ns"])
        if value > 0:
            return value
    return int(row["anchor_ts_ns"])


def _resolved_side(row: pd.Series) -> str:
    value = row.get("resolved_side", "")
    if value is None or pd.isna(value):
        return ""
    return str(value).upper()


def _price_candidates_for_side(
    *,
    order_side: str,
    best_bid: float,
    best_ask: float,
    tick_size: float,
    grid: ScenarioGrid,
) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    if order_side == "buy":
        if grid.include_join_best:
            candidates.append(("join_best_bid", best_bid))
        if grid.include_improve_one_tick:
            candidates.append(("improve_bid_1tick", best_bid + tick_size))
        if grid.include_inside_passive:
            candidates.append(("inside_bid", best_ask - tick_size))
        return [
            (name, _round_price(price, tick_size))
            for name, price in candidates
            if math.isfinite(price) and price > 0 and price < best_ask
        ]

    if grid.include_join_best:
        candidates.append(("join_best_ask", best_ask))
    if grid.include_improve_one_tick:
        candidates.append(("improve_ask_1tick", best_ask - tick_size))
    if grid.include_inside_passive:
        candidates.append(("inside_ask", best_bid + tick_size))
    return [
        (name, _round_price(price, tick_size))
        for name, price in candidates
        if math.isfinite(price) and price > best_bid
    ]


def intents_from_anchor_row(
    *,
    row: pd.Series,
    replay_index: ExecutionReplayIndex,
    grid: ScenarioGrid = ScenarioGrid(),
) -> list[OrderIntent]:
    market_slug = str(row["market_slug"])
    token_asset_id = str(row["token_asset_id"])
    replay = replay_index.get(market_slug, token_asset_id)
    if replay is None:
        return []

    order_ts_ns = int(row["anchor_ts_ns"])
    state = replay.state_before(order_ts_ns)
    if not state.valid or state.best_bid is None or state.best_ask is None:
        return []

    tick_size = state.tick_size or 0.01
    market_close_ts_ns = _market_close_from_anchor(row)
    market_deploy_ts_ns = _market_deploy_from_anchor(row)
    intents: list[OrderIntent] = []
    seen: set[tuple[str, float, float, int | None]] = set()
    for order_side in grid.order_sides:
        side_candidates = _price_candidates_for_side(
            order_side=order_side,
            best_bid=float(state.best_bid),
            best_ask=float(state.best_ask),
            tick_size=float(tick_size),
            grid=grid,
        )
        for price_level, limit_price in side_candidates:
            for order_size in grid.order_sizes:
                for cancel_after_s in grid.cancel_after_s:
                    cancel_after_ns = None if cancel_after_s is None else int(float(cancel_after_s) * SECOND_NS)
                    key = (order_side, limit_price, float(order_size), cancel_after_ns)
                    if key in seen:
                        continue
                    seen.add(key)
                    intents.append(
                        OrderIntent.create(
                            market_slug=market_slug,
                            token_asset_id=token_asset_id,
                            token_side_normalized=str(row["token_side_normalized"]).upper(),
                            order_ts_ns=order_ts_ns,
                            market_close_ts_ns=market_close_ts_ns,
                            market_deploy_ts_ns=market_deploy_ts_ns,
                            order_side=order_side,
                            limit_price=limit_price,
                            order_size=float(order_size),
                            price_level=price_level,
                            post_only=True,
                            cancel_after_ns=cancel_after_ns,
                            holding_policy="cancel_after_or_close"
                            if cancel_after_ns is not None
                            else "hold_to_close",
                            source_sequence_id=str(row.get("sequence_id") or ""),
                            resolved_side=_resolved_side(row),
                            scenario_name=grid.scenario_name,
                        )
                    )
                if grid.include_replace_path and len(side_candidates) >= 2:
                    replace_price_level, replace_limit_price = side_candidates[min(1, len(side_candidates) - 1)]
                    if replace_limit_price != limit_price:
                        replace_cancel_after_ns = int(max(grid.replace_after_s + 1.0, 3.0) * SECOND_NS)
                        key = (f"{order_side}_replace", limit_price, float(order_size), int(grid.replace_after_s * SECOND_NS))
                        if key not in seen:
                            seen.add(key)
                            intents.append(
                                OrderIntent.create(
                                    market_slug=market_slug,
                                    token_asset_id=token_asset_id,
                                    token_side_normalized=str(row["token_side_normalized"]).upper(),
                                    order_ts_ns=order_ts_ns,
                                    market_close_ts_ns=market_close_ts_ns,
                                    market_deploy_ts_ns=market_deploy_ts_ns,
                                    order_side=order_side,
                                    limit_price=limit_price,
                                    order_size=float(order_size),
                                    price_level=f"{price_level}_replace",
                                    post_only=True,
                                    cancel_after_ns=replace_cancel_after_ns,
                                    replace_after_ns=int(grid.replace_after_s * SECOND_NS),
                                    replace_limit_price=replace_limit_price,
                                    replace_order_size=float(order_size),
                                    replace_price_level=replace_price_level,
                                    replace_post_only=True,
                                    holding_policy="replace_then_cancel_or_close",
                                    source_sequence_id=str(row.get("sequence_id") or ""),
                                    resolved_side=_resolved_side(row),
                                    scenario_name=grid.scenario_name,
                                )
                            )
    return intents


def generate_intents_from_anchors(
    *,
    anchors: pd.DataFrame,
    replay_index: ExecutionReplayIndex,
    grid: ScenarioGrid = ScenarioGrid(),
    max_intents: int | None = None,
) -> list[OrderIntent]:
    intents: list[OrderIntent] = []
    required = {"market_slug", "token_asset_id", "token_side_normalized", "anchor_ts_ns"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(f"Anchor frame missing required columns: {sorted(missing)}")
    for _, row in anchors.iterrows():
        intents.extend(intents_from_anchor_row(row=row, replay_index=replay_index, grid=grid))
        if max_intents is not None and len(intents) >= max_intents:
            return intents[:max_intents]
    return intents


def load_phase8_anchors(
    *,
    dataset_root: Path,
    target_date: str,
    max_anchors: int | None = None,
    market_slug: str | None = None,
) -> pd.DataFrame:
    path = dataset_root / "microstructure_sequence_dataset_v1_tabular" / f"{target_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Phase 8 tabular anchors missing: {path}")
    columns = [
        "sequence_id",
        "market_slug",
        "token_asset_id",
        "token_side_normalized",
        "anchor_ts_ns",
        "t_to_close_s",
        "phase_bucket",
        "sequence_feature_eligible",
    ]
    available_columns = set(pq.ParquetFile(path).schema_arrow.names)
    frame = pd.read_parquet(path, columns=[column for column in columns if column in available_columns])
    if "sequence_feature_eligible" in frame.columns:
        frame = frame.loc[frame["sequence_feature_eligible"].astype(bool)].copy()
    if market_slug:
        frame = frame.loc[frame["market_slug"].astype(str).eq(market_slug)].copy()
    frame = frame.sort_values(["market_slug", "token_asset_id", "anchor_ts_ns"], kind="stable")
    if max_anchors is not None:
        frame = frame.head(int(max_anchors))
    return frame.reset_index(drop=True)


def iter_market_slugs(anchors: pd.DataFrame) -> Iterable[str]:
    for market_slug in anchors["market_slug"].dropna().astype(str).drop_duplicates().tolist():
        yield market_slug
