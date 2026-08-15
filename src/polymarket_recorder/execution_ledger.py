from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math
import pandas as pd


@dataclass(slots=True)
class InventoryLedger:
    """Simple deterministic ledger for simulated execution results."""

    cash: float = 0.0
    fees_paid: float = 0.0
    rebates_earned: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)
    fills: int = 0
    notional_traded: float = 0.0

    def apply_result(self, result: dict[str, Any]) -> None:
        if not bool(result.get("filled")):
            return
        token_asset_id = str(result["token_asset_id"])
        side = str(result["order_side"]).lower()
        size = float(result.get("fill_size") or 0.0)
        price = float(result.get("fill_price") or 0.0)
        fees = float(result.get("fees_paid") or 0.0)
        rebate = float(result.get("rebate_earned") or 0.0)
        if size <= 0 or price <= 0:
            return
        self.fills += 1
        self.fees_paid += fees
        self.rebates_earned += rebate
        self.notional_traded += size * price
        self.positions.setdefault(token_asset_id, 0.0)
        if side == "buy":
            self.positions[token_asset_id] += size
            self.cash -= size * price + fees - rebate
        elif side == "sell":
            self.positions[token_asset_id] -= size
            self.cash += size * price - fees + rebate

    def mark_to_settlement(self, payouts_by_token: dict[str, float]) -> float:
        value = self.cash
        for token_asset_id, position in self.positions.items():
            value += position * float(payouts_by_token.get(token_asset_id, 0.0))
        return value

    def summary(self) -> dict[str, Any]:
        gross_abs_position = sum(abs(value) for value in self.positions.values())
        return {
            "cash": self.cash,
            "fees_paid": self.fees_paid,
            "rebates_earned": self.rebates_earned,
            "fills": self.fills,
            "notional_traded": self.notional_traded,
            "open_token_count": sum(1 for value in self.positions.values() if abs(value) > 1e-12),
            "gross_abs_position": gross_abs_position,
        }


def reconcile_execution_results(frame: pd.DataFrame) -> dict[str, Any]:
    """Replay filled simulator rows through the ledger and compare to row net PnL."""

    if frame.empty:
        ledger = InventoryLedger()
        settlement_value = 0.0
        payouts_by_token: dict[str, float] = {}
    else:
        filled = frame["filled"].astype(bool) if "filled" in frame.columns else pd.Series(False, index=frame.index)
        filled_frame = frame.loc[filled].copy()
        payouts_frame = frame.loc[
            frame["token_asset_id"].notna()
            & frame["token_side_normalized"].astype(str).str.upper().isin(["UP", "DOWN"])
            & frame["resolved_side"].astype(str).str.upper().isin(["UP", "DOWN"]),
            ["token_asset_id", "token_side_normalized", "resolved_side"],
        ].drop_duplicates("token_asset_id", keep="last")
        payouts_by_token = {
            str(row.token_asset_id): 1.0
            if str(row.token_side_normalized).upper() == str(row.resolved_side).upper()
            else 0.0
            for row in payouts_frame.itertuples(index=False)
        }
        ledger = InventoryLedger()
        if not filled_frame.empty:
            size = pd.to_numeric(filled_frame["fill_size"], errors="coerce").fillna(0.0)
            price = pd.to_numeric(filled_frame["fill_price"], errors="coerce").fillna(0.0)
            fees = pd.to_numeric(filled_frame["fees_paid"], errors="coerce").fillna(0.0)
            rebates = pd.to_numeric(filled_frame["rebate_earned"], errors="coerce").fillna(0.0)
            valid = (size > 0) & (price > 0)
            side = filled_frame["order_side"].astype(str).str.lower()
            buy = valid & side.eq("buy")
            sell = valid & side.eq("sell")
            notional = size * price
            ledger.cash = float((-notional[buy] - fees[buy] + rebates[buy]).sum() + (notional[sell] - fees[sell] + rebates[sell]).sum())
            ledger.fees_paid = float(fees[valid].sum())
            ledger.rebates_earned = float(rebates[valid].sum())
            ledger.fills = int(valid.sum())
            ledger.notional_traded = float(notional[valid].sum())
            signed_size = size.where(buy, 0.0) - size.where(sell, 0.0)
            positions = signed_size.groupby(filled_frame["token_asset_id"].astype(str), sort=False).sum()
            ledger.positions = {str(token): float(value) for token, value in positions.items() if abs(float(value)) > 0.0}
        settlement_value = ledger.mark_to_settlement(payouts_by_token)
    expected_net = 0.0
    populated = 0
    for value in frame.get("net_pnl_to_close", pd.Series(dtype="float64")).tolist():
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            expected_net += parsed
            populated += 1
    diff = settlement_value - expected_net
    tolerance = max(1e-9, 1e-12 * max(1.0, abs(expected_net), ledger.notional_traded))
    return {
        "ledger": ledger.summary(),
        "payouts_by_token": payouts_by_token,
        "settlement_value": settlement_value,
        "expected_net_pnl": expected_net,
        "populated_net_pnl_rows": populated,
        "difference": diff,
        "tolerance": tolerance,
        "balances": abs(diff) <= tolerance,
    }
