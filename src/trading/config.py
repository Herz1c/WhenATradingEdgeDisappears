"""All tunable parameters for the hybrid maker/taker strategy.

Defaults are CONSERVATIVE — designed for paper-trading at first.
Tighten / loosen each parameter based on backtester results.

Naming convention:
  - _ticks_     - in price ticks (typically 0.01 = 1¢ for Polymarket)
  - _frac_      - in fractional probability units (0.0-1.0)
  - _seconds    - in real seconds
  - _ns         - in nanoseconds
  - _usd        - in USD
  - _contracts  - in unit contracts (1 = 1 share of YES or NO)
"""
from __future__ import annotations
from dataclasses import dataclass, field


# ── Risk parameters ──────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Position & loss limits at every aggregation level."""
    # Per quote
    max_size_per_quote_contracts: float = 50.0
    min_edge_required_ticks: float = 0.5      # don't quote if expected edge < 0.5 ticks

    # Per market
    max_position_per_market_contracts: float = 200.0
    max_consecutive_losses_per_market: int = 3
    market_blackout_after_losses_seconds: int = 600    # 10 min
    no_quote_in_last_n_seconds: int = 15               # before close

    # Per session (daily)
    max_daily_loss_usd: float = 500.0
    max_drawdown_from_session_peak_usd: float = 250.0
    max_consecutive_session_losses: int = 8
    session_blackout_after_loss_minutes: int = 30

    # Global
    max_gross_inventory_contracts: float = 2000.0
    max_active_quotes_total: int = 50
    max_trades_per_minute: int = 30
    max_taker_trades_per_hour: int = 10                # taker is expensive

    # Volatility scaler
    use_volatility_scaler: bool = True
    vol_scaler_lookback_seconds: int = 60
    vol_scaler_strength: float = 1.0                   # 0=off, 1=normal, 2=aggressive

    # Kelly
    kelly_fraction: float = 0.10                       # 1/10 Kelly. Steady > fast.

    # If set, bypass Kelly entirely and use this fixed size for every quote.
    # Useful for market-maker style: predictable, uniform sizing.
    uniform_quote_size: float | None = None

    # If True, even in uniform-size mode the strategy gates on expected edge
    # (half_spread - adverse_cost + rebate) >= min_edge_required_ticks * tick.
    # Use this to make uniform mode behave like Kelly in terms of refusing
    # to quote into clearly toxic conditions.
    enforce_edge_check_in_uniform_mode: bool = False


# ── Signal threshold parameters ──────────────────────────────────────────────

@dataclass
class SignalConfig:
    """Thresholds for triggering each model-driven decision."""
    # Defense (08c)
    severe_defense_threshold_frac: float = 0.50        # cancel if p_severe >= this
    extreme_freeze_threshold_frac: float = 0.40        # also: freeze requoting
    moderate_lean_threshold_frac: float = 0.55         # widen quote on weak side

    # After hard defense fires, don't requote for cooldown
    defense_cooldown_seconds: int = 30

    # Fair value
    max_fair_value_drift_ticks: float = 5.0            # if |F - mid| > this → don't quote
    quote_edge_ticks: float = 1.0                      # quote at F ± edge

    # Mispricing (taker)
    taker_min_edge_ticks: float = 1.5                  # required to taker, after fees
    taker_min_fill_prob: float = 0.70                  # only take when likely to fill

    # Adverse selection (post-fill)
    flatten_if_predicted_markout_below_ticks: float = -0.8

    # Direction skew (07)
    direction_skew_threshold_frac: float = 0.60        # |p_up - 0.5| > 0.1 → skew
    direction_skew_ticks: float = 0.5

    # Closing flip (05)
    closing_flip_window_seconds: int = 60              # apply 05 within last 60s
    closing_flip_defense_threshold_frac: float = 0.55

    # Directional maker gate using model 04b (markout-to-close).
    # If E[markout_to_close] >= +threshold_usd, allow only BID quote (long bias).
    # If E[markout_to_close] <= -threshold_usd, allow only ASK quote (short bias).
    # If neither, skip quoting (when gate is enabled).
    # Setting to 0.0 disables the gate (default).
    markout_to_close_directional_gate_usd: float = 0.0


# ── Execution parameters ─────────────────────────────────────────────────────

@dataclass
class ExecutionConfig:
    """Per-venue and per-order execution settings."""
    tick_size: float = 0.01                            # Polymarket: 1¢
    maker_rebate_per_contract_ticks: float = 0.0       # filled in for real venue
    taker_fee_per_contract_ticks: float = 0.5          # placeholder; tier-dependent
    default_quote_time_in_force_ms: int = 5_000
    requote_min_interval_ms: int = 200                 # rate-limit re-quoting
    cancel_on_disconnect_seconds: int = 5


# ── Selection / portfolio parameters ─────────────────────────────────────────

@dataclass
class SelectionConfig:
    """How many markets to quote concurrently, and how to pick them."""
    max_concurrent_markets: int = 8
    min_book_spread_to_quote_ticks: float = 0.0        # don't quote if cross/locked
    max_book_spread_to_quote_ticks: float = 5.0        # too wide → don't bother
    require_min_book_depth_contracts: float = 20.0
    prefer_markets_with_higher_volume: bool = True


# ── Top-level config ─────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    risk: RiskConfig = field(default_factory=RiskConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    # Logging / debug
    log_every_decision: bool = True
    log_skipped_quotes: bool = False
    paranoid_checks: bool = True              # double-check risk before emitting

    @classmethod
    def conservative(cls) -> "StrategyConfig":
        """Extra-cautious defaults for first paper-trading session."""
        cfg = cls()
        cfg.risk.kelly_fraction = 0.05
        cfg.risk.max_daily_loss_usd = 200.0
        cfg.risk.max_position_per_market_contracts = 50.0
        cfg.risk.max_gross_inventory_contracts = 500.0
        cfg.signals.severe_defense_threshold_frac = 0.35   # extra-trigger-happy defense
        cfg.signals.taker_min_edge_ticks = 2.0             # require strong taker edge
        cfg.selection.max_concurrent_markets = 3
        return cfg

    @classmethod
    def standard(cls) -> "StrategyConfig":
        """Defaults for normal operation after backtesting passes."""
        return cls()

    @classmethod
    def aggressive(cls) -> "StrategyConfig":
        """Higher-risk, higher-volume. Not recommended without backtester validation."""
        cfg = cls()
        cfg.risk.kelly_fraction = 0.25
        cfg.risk.max_daily_loss_usd = 1500.0
        cfg.selection.max_concurrent_markets = 20
        cfg.signals.severe_defense_threshold_frac = 0.65
        return cfg

    @classmethod
    def market_maker(cls) -> "StrategyConfig":
        """Proper market-maker config: always quote, signals adjust not gate.

        Philosophy: be present in every market with a book.  Use signals to:
          - widen quotes when 08c moderate fires (size down)
          - cancel when 08c severe/extreme fires (defense)
          - lean quote prices on direction (model 07)
          - flatten post-fill on adverse selection (model 04)
        DO NOT use signals to decide whether to quote in the first place.
        """
        cfg = cls()
        # Sizing: use uniform 1-contract for simplicity and predictability
        cfg.risk.uniform_quote_size = 1.0                # NEW: bypass Kelly
        cfg.risk.max_size_per_quote_contracts = 2.0
        cfg.risk.max_position_per_market_contracts = 10.0
        cfg.risk.max_gross_inventory_contracts = 200.0
        cfg.risk.max_active_quotes_total = 500
        cfg.risk.max_trades_per_minute = 200
        cfg.risk.max_taker_trades_per_hour = 0           # NO taker, maker only
        cfg.risk.kelly_fraction = 0.0                    # disable Kelly
        # Selection: be in everything that has a book
        cfg.selection.max_concurrent_markets = 100
        cfg.selection.max_book_spread_to_quote_ticks = 20.0   # very generous
        cfg.selection.require_min_book_depth_contracts = 1.0
        # Signals: relaxed gating, aggressive defense
        cfg.signals.max_fair_value_drift_ticks = 50.0    # don't gate on FV drift
        cfg.signals.severe_defense_threshold_frac = 0.40
        cfg.signals.extreme_freeze_threshold_frac = 0.35
        cfg.signals.moderate_lean_threshold_frac = 0.50
        # Execution
        cfg.execution.requote_min_interval_ms = 50       # quote often
        cfg.execution.maker_rebate_per_contract_ticks = 0.1   # small but real rebate
        return cfg
