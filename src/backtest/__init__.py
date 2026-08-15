"""Backtest harness for finding profitable strategy configs.

Modules:
  fees                - Polymarket fee/rebate calculator (wraps FeeScheduleRegistry)
  fill_model          - Maker/taker fill simulator with realistic fees
  pnl_tracker         - $-denominated PnL accounting with drawdown tracking
  multi_market_engine - Replays anchors across many markets concurrently
  grading             - Calmar-style strategy grade across N Monte Carlo weeks
  mc_weeks            - Generates reproducible MC week samples
  search_space        - Optuna hyperparameter search space for HybridStrategy
  resource_caps       - Quiet-mode CPU affinity / process priority caps
"""
