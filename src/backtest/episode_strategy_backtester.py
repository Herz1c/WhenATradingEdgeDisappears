"""Canonical episode-strategy backtest engine for BTC 5m up/down markets.

Single shared code path for backtesting locked TCN residual strategies on the
episode datasets (data/datasets/btc_5m_episodes_*). Extracted from
tools/search_tcn_fullmarket_exit_policy.py / search_tcn_exit_policy.py so that
strategy locks, audits, policy searches, and the live gate all run identical
entry/fee/delay/slippage logic instead of one-off copies.

Execution model (matches tcn_double_strategy_v1 lock):
  - decision at step T on the side's best ask; fill at the ask observed at
    T + ceil(delay_s / cadence); candidate dropped if fill > quote + slippage
    or fill is non-finite/degenerate (slippage cap is a *filter*, see
    fill_mode="cap_drop"; fill_mode="fill_worse" fills at the delayed price
    with no drop, as the honest execution-realism variant)
  - fee = 0.072 * p * (1 - p), charged on entry fill always, on sell fill for
    active exits; hold-to-resolution redeems winners at $1.00 with no exit fee
  - one entry per market per strategy_id; UTC-hour filter evaluated on the
    decision snapshot timestamp (now_ns), matching the shadow bot's
    LockedStrategy.allowed_at
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

UP_BID, UP_ASK, DN_BID, DN_ASK = 0, 1, 2, 3
NS = 1_000_000_000


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64, copy=False), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64, copy=False), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def fee(price: float) -> float:
    return float(0.072 * price * (1.0 - price))


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategySpec:
    """One locked strategy slot (entry rules)."""
    strategy_id: str
    beta: float
    ttc_min: float          # exclusive
    ttc_max: float          # inclusive
    ev_min: float
    price_lo: float         # exclusive
    price_hi: float         # exclusive
    utc_hours: frozenset[int] | None = None   # None = all hours
    delay_s: float = 2.0
    slippage: float = 0.03
    shares: float = 5.1

    @classmethod
    def from_lock_row(cls, row: dict[str, Any], execution: dict[str, Any]) -> "StrategySpec":
        hours_raw = row.get("utc_hours_allowed")
        hours = None if hours_raw == "all" else frozenset(int(x) for x in hours_raw)
        return cls(
            strategy_id=str(row["strategy_id"]),
            beta=float(row["beta"]),
            ttc_min=float(row["ttc_min_s_exclusive"]),
            ttc_max=float(row["ttc_max_s_inclusive"]),
            ev_min=float(row["ev_min"]),
            price_lo=float(row["price_min_exclusive"]),
            price_hi=float(row["price_max_exclusive"]),
            utc_hours=hours,
            delay_s=float(execution.get("buy_delay_s", 2.0)),
            slippage=float(execution.get("buy_slippage_cap", 0.03)),
            shares=float(execution.get("shares_per_entry", 5.1)),
        )

    def to_json(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["utc_hours"] = sorted(self.utc_hours) if self.utc_hours is not None else "all"
        return d


@dataclass(frozen=True)
class ExitSpec:
    """Active exit policy; the default is pure hold-to-resolution."""
    label: str = "hold"
    stop_loss: float | None = None
    take_profit: float | None = None
    model_drop: float | None = None
    sell_edge: float | None = None
    time_exit_ttc: float | None = None
    min_exit_ttc: float = 15.0
    delay_s: float = 2.0
    slippage: float = 0.03

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


HOLD = ExitSpec("hold")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class EpisodeData:
    split: str
    quotes: np.ndarray        # (n, T, 5) up_bid, up_ask, dn_bid, dn_ask, implied_p_up
    valid: np.ndarray         # (n, T) bool
    y: np.ndarray             # (n,) int8 resolved_up
    date: np.ndarray          # (n,) str
    market_slug: np.ndarray   # (n,) str
    now_ns: np.ndarray        # (n, T) int64 snapshot timestamps
    open_s: np.ndarray        # (n,) int64
    base_logit: np.ndarray    # (n, T) logit(implied_p_up)
    delta: np.ndarray         # (n, T) TCN residual
    cadence_s: float
    utc_hour: np.ndarray = field(init=False)   # (n, T) int8

    def __post_init__(self) -> None:
        self.utc_hour = (((self.now_ns // NS) % 86_400) // 3_600).astype(np.int8)

    @property
    def n_ep(self) -> int:
        return int(self.y.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.valid.shape[1])

    @property
    def ttc_grid(self) -> np.ndarray:
        t = np.arange(self.seq_len, dtype=np.float32) * self.cadence_s
        return self.seq_len * self.cadence_s - t

    def ttc_at(self, step: int) -> float:
        return float(self.seq_len * self.cadence_s - step * self.cadence_s)


def load_tcn_model(tcn_dir: Path, n_features: int, device: "Any" = None):
    """Load the locked ResidualTCN. Torch imported lazily so numpy-only callers
    (e.g. scoring pre-computed deltas) don't need it."""
    import torch
    from train_episode_tcn import ResidualTCN  # noqa: E402  (tools/ on sys.path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = json.loads((tcn_dir / "tcn_report.json").read_text(encoding="utf-8"))
    cfg = report["model"]
    model = ResidualTCN(
        n_features=n_features,
        channels=int(cfg["channels"]),
        blocks=int(cfg["blocks"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=0.0,
        residual_scale=float(cfg.get("residual_scale", 0.5)),
    ).to(device)
    state = torch.load(tcn_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, report, device


def predict_delta(model, x: np.ndarray, *, batch_size: int = 64, device=None) -> np.ndarray:
    import torch
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(device)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outs, axis=0)


def load_episode_split(dataset: Path, split: str, *, tcn_dir: Path | None = None,
                       cadence_s: float = 0.2, batch_size: int = 64,
                       include_mask_channel: bool = True,
                       delta: np.ndarray | None = None) -> EpisodeData:
    """Load a split npz and attach TCN residual deltas (forwarded over the full
    grid, matching the search scripts). Pass `delta` to reuse precomputed ones."""
    z = np.load(dataset / f"{split}.npz", allow_pickle=False)
    valid = z["valid_mask"].astype(bool, copy=False)
    p_market = z["p_market"].astype(np.float32, copy=False)
    p_market[~np.isfinite(p_market)] = 0.5
    if delta is None:
        if tcn_dir is None:
            raise ValueError("need tcn_dir or precomputed delta")
        x = z["X"].astype(np.float32, copy=False)
        if include_mask_channel:
            x = np.concatenate([x, valid[:, :, None].astype(np.float32)], axis=2)
        model, _report, device = load_tcn_model(tcn_dir, n_features=x.shape[2])
        delta = predict_delta(model, x, batch_size=batch_size, device=device)
    return EpisodeData(
        split=split,
        quotes=z["quotes"].astype(np.float32, copy=False),
        valid=valid,
        y=z["y"].astype(np.int8, copy=False),
        date=z["date"].copy(),
        market_slug=z["market_slug"].copy(),
        now_ns=z["now_ns"].astype(np.int64, copy=False),
        open_s=z["open_s"].astype(np.int64, copy=False),
        base_logit=logit_np(p_market),
        delta=delta,
        cadence_s=cadence_s,
    )


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

def find_entries(data: EpisodeData, spec: StrategySpec, *,
                 fill_mode: str = "cap_drop",
                 delay_s: float | None = None) -> list[dict[str, Any]]:
    """First qualifying entry per market for one strategy slot.

    fill_mode:
      "cap_drop"   candidate dropped if delayed fill > quote + slippage (lock
                   semantics; survivorship toward benign fills)
      "fill_worse" always fill at the delayed ask (honest lower bound)
    """
    if fill_mode not in ("cap_drop", "fill_worse"):
        raise ValueError(f"unknown fill_mode {fill_mode!r}")
    p = sigmoid(data.base_logit + spec.beta * data.delta)
    q = data.quotes
    ttc = data.ttc_grid
    band = (ttc > spec.ttc_min) & (ttc <= spec.ttc_max)
    valid = data.valid & band[None, :]
    if spec.utc_hours is not None:
        hours = np.zeros(24, dtype=bool)
        hours[list(spec.utc_hours)] = True
        valid = valid & hours[data.utc_hour]
    up_ask = q[:, :, UP_ASK]
    dn_ask = q[:, :, DN_ASK]
    ev_up = p - up_ask
    ev_dn = (1.0 - p) - dn_ask
    take_up = (
        valid & (ev_up >= spec.ev_min) & (ev_up >= ev_dn)
        & (up_ask > spec.price_lo) & (up_ask < spec.price_hi)
    )
    take_dn = (
        valid & (ev_dn >= spec.ev_min) & (ev_dn > ev_up)
        & (dn_ask > spec.price_lo) & (dn_ask < spec.price_hi)
    )
    cand_ep, cand_step = np.nonzero(take_up | take_dn)
    use_delay = spec.delay_s if delay_s is None else delay_s
    delay_steps = int(math.ceil(use_delay / data.cadence_s))
    entries: list[dict[str, Any]] = []
    entered = np.zeros(data.n_ep, dtype=bool)
    for ep, step in zip(cand_ep.tolist(), cand_step.tolist()):
        if entered[ep]:
            continue
        fill_step = step + delay_steps
        if fill_step >= data.seq_len:
            continue
        side_up = bool(take_up[ep, step])
        quote = float(up_ask[ep, step] if side_up else dn_ask[ep, step])
        fill = float(q[ep, fill_step, UP_ASK] if side_up else q[ep, fill_step, DN_ASK])
        if not (np.isfinite(fill) and 0.0 < fill < 1.0):
            continue
        if fill_mode == "cap_drop" and fill > quote + spec.slippage:
            continue
        p_up = float(p[ep, step])
        entered[ep] = True
        entries.append({
            "strategy_id": spec.strategy_id,
            "ep": ep,
            "step": step,
            "fill_step": fill_step,
            "side_up": side_up,
            "side": "UP" if side_up else "DOWN",
            "quote": quote,
            "fill": fill,
            "p_up": p_up,
            "p_side": p_up if side_up else 1.0 - p_up,
            "ev": float(ev_up[ep, step] if side_up else ev_dn[ep, step]),
            "ttc_s": data.ttc_at(step),
            "utc_hour": int(data.utc_hour[ep, step]),
        })
    return entries


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _reason_to_exit(spec: ExitSpec, *, side_up: bool, entry_fill: float,
                    entry_p_side: float, p_up: float, bid: float, ttc: float) -> str | None:
    p_side = p_up if side_up else 1.0 - p_up
    if spec.stop_loss is not None and bid <= entry_fill - spec.stop_loss:
        return "stop_loss"
    if spec.take_profit is not None and bid >= entry_fill + spec.take_profit:
        return "take_profit"
    if spec.model_drop is not None and np.isfinite(p_side) and p_side <= entry_p_side - spec.model_drop:
        return "model_drop"
    if spec.sell_edge is not None and np.isfinite(p_side) and bid - p_side >= spec.sell_edge:
        return "sell_edge"
    if spec.time_exit_ttc is not None and ttc <= spec.time_exit_ttc:
        return "time_exit"
    return None


def simulate_trade(data: EpisodeData, entry: dict[str, Any], strategy: StrategySpec,
                   exit_spec: ExitSpec = HOLD, *, shares: float | None = None) -> dict[str, Any]:
    """Simulate one entry to its exit (active sell or hold to resolution)."""
    shares = strategy.shares if shares is None else shares
    ep = int(entry["ep"])
    side_up = bool(entry["side_up"])
    q = data.quotes[ep]
    exit_info: dict[str, Any] | None = None
    wants_active_exit = any(v is not None for v in (
        exit_spec.stop_loss, exit_spec.take_profit, exit_spec.model_drop,
        exit_spec.sell_edge, exit_spec.time_exit_ttc))
    if wants_active_exit:
        p = sigmoid(data.base_logit[ep] + strategy.beta * data.delta[ep])
        delay_steps = int(math.ceil(exit_spec.delay_s / data.cadence_s))
        for step in range(int(entry["fill_step"]) + 1, data.seq_len - delay_steps):
            ttc = data.ttc_at(step)
            if ttc < exit_spec.min_exit_ttc:
                break
            bid = float(q[step, UP_BID] if side_up else q[step, DN_BID])
            if not (np.isfinite(bid) and 0.0 < bid < 1.0):
                continue
            p_up = float(p[step]) if data.valid[ep, step] else float("nan")
            reason = _reason_to_exit(
                exit_spec, side_up=side_up, entry_fill=float(entry["fill"]),
                entry_p_side=float(entry["p_side"]), p_up=p_up, bid=bid, ttc=ttc)
            if reason is None:
                continue
            fill_step = step + delay_steps
            sell_fill = float(q[fill_step, UP_BID] if side_up else q[fill_step, DN_BID])
            if not (np.isfinite(sell_fill) and 0.0 < sell_fill < 1.0):
                continue
            if sell_fill < bid - exit_spec.slippage:
                continue
            pnl = shares * (sell_fill - float(entry["fill"])
                            - fee(float(entry["fill"])) - fee(sell_fill))
            exit_info = {
                "exit_type": "sell", "exit_reason": reason, "exit_step": step,
                "exit_ttc_s": ttc, "exit_bid": bid, "exit_fill": sell_fill,
                "pnl": float(pnl),
            }
            break
    win = bool(data.y[ep] == 1) if side_up else bool(data.y[ep] == 0)
    if exit_info is None:
        pnl = shares * ((1.0 if win else 0.0) - float(entry["fill"]) - fee(float(entry["fill"])))
        exit_info = {
            "exit_type": "resolve", "exit_reason": "hold_to_resolution",
            "exit_step": None, "exit_ttc_s": 0.0, "exit_bid": None,
            "exit_fill": None, "pnl": float(pnl),
        }
    return {
        "split": data.split,
        "strategy_id": entry.get("strategy_id"),
        "date": str(data.date[ep]),
        "market_slug": str(data.market_slug[ep]),
        "ep_idx": ep,
        "side": entry["side"],
        "resolved_win": win,
        "entry_step": int(entry["step"]),
        "entry_ttc_s": float(entry["ttc_s"]),
        "entry_utc_hour": int(entry.get("utc_hour", -1)),
        "entry_quote": float(entry["quote"]),
        "entry_fill": float(entry["fill"]),
        "entry_ev": float(entry["ev"]),
        "p_up_entry": float(entry["p_up"]),
        "p_side_entry": float(entry["p_side"]),
        "shares": shares,
        **exit_info,
    }


def summarize(trades: list[dict[str, Any]], markets: int | None = None) -> dict[str, Any]:
    pnl = np.asarray([t["pnl"] for t in trades], dtype=np.float64)
    by_day: dict[str, float] = {}
    exits: dict[str, int] = {}
    wins = 0
    for t in trades:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + float(t["pnl"])
        exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1
        wins += int(bool(t["resolved_win"]))
    day_vals = np.asarray([by_day[k] for k in sorted(by_day)], dtype=np.float64)
    equity = np.cumsum(day_vals) if day_vals.size else day_vals
    dd = equity - np.maximum.accumulate(equity) if equity.size else equity
    out = {
        "trades": int(len(trades)),
        "unique_markets": len({t["market_slug"] for t in trades}),
        "wins_to_resolution": int(wins),
        "resolution_win_rate": float(wins / len(trades)) if trades else 0.0,
        "total_pnl": float(pnl.sum()) if pnl.size else 0.0,
        "avg_pnl": float(pnl.mean()) if pnl.size else 0.0,
        "median_pnl": float(np.median(pnl)) if pnl.size else 0.0,
        "active_days": int(len(by_day)),
        "positive_days": int(sum(1 for v in by_day.values() if v > 0.0)),
        "worst_day": float(day_vals.min()) if day_vals.size else 0.0,
        "best_day": float(day_vals.max()) if day_vals.size else 0.0,
        "max_drawdown": float(dd.min()) if dd.size else 0.0,
        "exit_counts": dict(sorted(exits.items())),
        "by_day": {k: round(v, 6) for k, v in sorted(by_day.items())},
    }
    if markets is not None:
        out["markets_available"] = int(markets)
    return out


# ---------------------------------------------------------------------------
# Lock-level driver
# ---------------------------------------------------------------------------

def load_lock(lock_path: Path) -> tuple[list[StrategySpec], dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    execution = lock.get("execution_assumptions", {})
    specs = [StrategySpec.from_lock_row(row, execution)
             for row in lock["strategies"] if bool(row.get("enabled", True))]
    return specs, lock


def run_locked_strategies(data: EpisodeData, specs: list[StrategySpec],
                          exit_spec: ExitSpec = HOLD, *,
                          fill_mode: str = "cap_drop",
                          delay_s: float | None = None) -> dict[str, Any]:
    """Backtest every strategy slot independently (one entry per market per
    strategy_id) and a combined portfolio view."""
    per_strategy: dict[str, Any] = {}
    all_trades: list[dict[str, Any]] = []
    for spec in specs:
        entries = find_entries(data, spec, fill_mode=fill_mode, delay_s=delay_s)
        trades = [simulate_trade(data, e, spec, exit_spec) for e in entries]
        per_strategy[spec.strategy_id] = {
            "spec": spec.to_json(),
            "summary": summarize(trades, data.n_ep),
            "trades": trades,
        }
        all_trades.extend(trades)
    slugs_by_strategy = [
        {t["market_slug"] for t in s["trades"]} for s in per_strategy.values()
    ]
    overlap = set.intersection(*slugs_by_strategy) if len(slugs_by_strategy) > 1 else set()
    combined = summarize(all_trades, data.n_ep)
    combined["overlap_markets"] = len(overlap)
    return {
        "per_strategy": per_strategy,
        "combined": {"summary": combined, "trades": all_trades},
    }
