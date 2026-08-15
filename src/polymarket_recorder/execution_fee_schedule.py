from __future__ import annotations

from dataclasses import dataclass

SECOND_NS = 1_000_000_000
POLYMARKET_FEE_SCHEDULE_CUTOFF_NS = 1_774_828_800 * SECOND_NS


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Versioned Polymarket crypto fee schedule for Phase 5/6 execution replay."""

    schedule_id: str = "polymarket_crypto_fee_v2"
    pre_cutoff_fee_rate: float = 0.25
    pre_cutoff_exponent: int = 2
    post_cutoff_fee_rate: float = 0.072
    post_cutoff_exponent: int = 1
    maker_rebate_rate: float = 0.20
    rounding_decimals: int = 4

    # Diagnostic compatibility fields retained in simulator/dataset schemas.
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 720.0
    maker_rebate_bps: float = 2000.0
    taker_rebate_bps: float = 0.0

    def params_for_deploy(self, market_deploy_ts_ns: int) -> tuple[float, int]:
        if int(market_deploy_ts_ns) >= POLYMARKET_FEE_SCHEDULE_CUTOFF_NS:
            return float(self.post_cutoff_fee_rate), int(self.post_cutoff_exponent)
        return float(self.pre_cutoff_fee_rate), int(self.pre_cutoff_exponent)

    def taker_fee(self, price: float, size: float, market_deploy_ts_ns: int) -> float:
        fee_rate, exponent = self.params_for_deploy(market_deploy_ts_ns)
        p = float(price)
        fee = float(size) * p * fee_rate * ((p * (1.0 - p)) ** exponent)
        return round(fee, int(self.rounding_decimals))

    def maker_fee(self, price: float, size: float) -> float:
        return 0.0

    def maker_rebate(self, taker_fee: float, market_deploy_ts_ns: int) -> float:
        return round(float(taker_fee) * float(self.maker_rebate_rate), int(self.rounding_decimals))


def compute_taker_fee(price: float, size: float, market_deploy_ts: int) -> float:
    return FeeSchedule().taker_fee(price, size, market_deploy_ts)


def compute_maker_fee(price: float, size: float) -> float:
    return FeeSchedule().maker_fee(price, size)


def compute_maker_rebate(taker_fee: float, market_deploy_ts: int) -> float:
    return FeeSchedule().maker_rebate(taker_fee, market_deploy_ts)
