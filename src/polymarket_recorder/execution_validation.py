from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from chainlink_recorder.live_reference_events import LiveReferenceEventsReader

from .execution_audit import audit_execution_results
from .execution_ledger import reconcile_execution_results
from .execution_replay import ExecutionReplayIndex, SECOND_NS, load_execution_replay_for_date
from .execution_scenarios import ScenarioGrid, generate_intents_from_anchors, load_phase8_anchors
from .execution_simulator import ExecutionSimulator, ExecutionSimulatorConfig, FeeSchedule, OrderIntent

DEFAULT_VALIDATION_REPORT_PATH = Path("docs/simulator_validation_report_v1.md")
DEFAULT_SMOKE_OUTPUT_DIR = Path("data/simulations/execution_simulator_v1_smoke")


def _synthetic_events() -> pd.DataFrame:
    def book(ts_s: float, bid: float, ask: float, seq: int) -> dict[str, Any]:
        mid = (bid + ask) / 2.0
        return {
            "market_slug": "synthetic-btc-updown-5m",
            "token_asset_id": "synthetic-up-token",
            "token_side_normalized": "UP",
            "book_state_seq": seq,
            "recv_ts_ns": int(ts_s * SECOND_NS),
            "event_type": "book_state",
            "best_bid": bid,
            "best_ask": ask,
            "mid": mid,
            "spread": ask - bid,
            "microprice": mid,
            "bid_depth_total": 100.0,
            "ask_depth_total": 100.0,
            "imbalance": 0.0,
            "last_trade_price": None,
            "trade_size": 0.0,
            "trade_side": "none",
            "tick_size": 0.01,
        }

    def trade(ts_s: float, price: float, size: float, side: str, seq: int, bid: float, ask: float) -> dict[str, Any]:
        row = book(ts_s, bid, ask, seq)
        row.update(
            {
                "event_type": "trade",
                "last_trade_price": price,
                "trade_size": size,
                "trade_side": side,
            }
        )
        return row

    rows = [
        book(1.0, 0.49, 0.51, 1),
        trade(1.5, 0.49, 20.0, "sell", 2, 0.49, 0.51),
        trade(1.6, 0.49, 5.0, "sell", 3, 0.49, 0.51),
        book(2.0, 0.54, 0.56, 4),
        trade(2.5, 0.56, 15.0, "buy", 5, 0.54, 0.56),
        trade(3.0, 0.54, 12.0, "sell", 6, 0.54, 0.56),
        book(4.0, 0.52, 0.54, 7),
        book(8.0, 0.45, 0.47, 8),
        trade(10.0, 0.45, 20.0, "sell", 9, 0.45, 0.47),
    ]
    return pd.DataFrame(rows)


class _SyntheticLiveReferenceReader:
    def latest_asof(self, *, asof_ts_ns: int) -> dict[str, Any] | None:
        if int(8.2 * SECOND_NS) <= asof_ts_ns < int(8.3 * SECOND_NS):
            return {
                "recv_ts_ns": int(8.0 * SECOND_NS),
                "btc_usd_price": None,
                "price_valid": False,
                "degraded": False,
                "source_count": 0,
                "bias_mode": "unavailable",
                "bias_active": False,
                "bias_carried_forward": False,
                "applied_bias_age_seconds": None,
            }
        return {
            "recv_ts_ns": int(asof_ts_ns // SECOND_NS) * SECOND_NS,
            "btc_usd_price": 100_000.0,
            "price_valid": True,
            "degraded": False,
            "source_count": 3,
            "bias_mode": "active_window",
            "bias_active": True,
            "bias_carried_forward": False,
            "applied_bias_age_seconds": 1800.0,
        }


def run_synthetic_scenario_suite() -> dict[str, Any]:
    replay_index = ExecutionReplayIndex.from_events(_synthetic_events())
    simulator = ExecutionSimulator(
        replay_index,
        ExecutionSimulatorConfig(
            fee_schedule=FeeSchedule(
                schedule_id="synthetic_polymarket_crypto_fee_v2",
            ),
            require_live_reference=True,
        ),
        live_reference_reader=_SyntheticLiveReferenceReader(),
    )
    close_ts_ns = 10 * SECOND_NS
    intents = [
        OrderIntent.create(
            market_slug="synthetic-btc-updown-5m",
            token_asset_id="synthetic-up-token",
            token_side_normalized="UP",
            order_ts_ns=int(1.2 * SECOND_NS),
            market_close_ts_ns=close_ts_ns,
            order_side="buy",
            limit_price=0.49,
            order_size=10.0,
            price_level="join_best_bid",
            cancel_after_ns=3 * SECOND_NS,
            scenario_name="synthetic_buy_fill",
            resolved_side="UP",
        ),
        OrderIntent.create(
            market_slug="synthetic-btc-updown-5m",
            token_asset_id="synthetic-up-token",
            token_side_normalized="UP",
            order_ts_ns=int(1.2 * SECOND_NS),
            market_close_ts_ns=close_ts_ns,
            order_side="buy",
            limit_price=0.51,
            order_size=10.0,
            price_level="crossing_bid",
            cancel_after_ns=3 * SECOND_NS,
            scenario_name="synthetic_post_only_reject",
            resolved_side="UP",
        ),
        OrderIntent.create(
            market_slug="synthetic-btc-updown-5m",
            token_asset_id="synthetic-up-token",
            token_side_normalized="UP",
            order_ts_ns=int(1.2 * SECOND_NS),
            market_close_ts_ns=close_ts_ns,
            order_side="buy",
            limit_price=0.49,
            order_size=10.0,
            price_level="join_best_bid",
            cancel_after_ns=int(0.1 * SECOND_NS),
            scenario_name="synthetic_cancel_before_fill",
            resolved_side="UP",
        ),
        OrderIntent.create(
            market_slug="synthetic-btc-updown-5m",
            token_asset_id="synthetic-up-token",
            token_side_normalized="UP",
            order_ts_ns=int(2.1 * SECOND_NS),
            market_close_ts_ns=close_ts_ns,
            order_side="sell",
            limit_price=0.56,
            order_size=10.0,
            price_level="join_best_ask",
            cancel_after_ns=3 * SECOND_NS,
            scenario_name="synthetic_sell_fill",
            resolved_side="UP",
        ),
        OrderIntent.create(
            market_slug="synthetic-btc-updown-5m",
            token_asset_id="synthetic-up-token",
            token_side_normalized="UP",
            order_ts_ns=int(1.2 * SECOND_NS),
            market_close_ts_ns=close_ts_ns,
            order_side="buy",
            limit_price=0.49,
            order_size=30.0,
            price_level="join_best_bid",
            cancel_after_ns=1 * SECOND_NS,
            scenario_name="synthetic_partial_fill",
            resolved_side="UP",
        ),
        OrderIntent.create(
            market_slug="synthetic-btc-updown-5m",
            token_asset_id="synthetic-up-token",
            token_side_normalized="UP",
            order_ts_ns=int(1.2 * SECOND_NS),
            market_close_ts_ns=close_ts_ns,
            order_side="buy",
            limit_price=0.40,
            order_size=10.0,
            price_level="far_bid_no_fill",
            cancel_after_ns=3 * SECOND_NS,
            scenario_name="synthetic_no_fill",
            resolved_side="UP",
        ),
        OrderIntent.create(
            market_slug="synthetic-btc-updown-5m",
            token_asset_id="synthetic-up-token",
            token_side_normalized="UP",
            order_ts_ns=int(1.2 * SECOND_NS),
            market_close_ts_ns=close_ts_ns,
            order_side="buy",
            limit_price=0.48,
            order_size=10.0,
            price_level="replace_from_far_bid",
            replace_after_ns=1 * SECOND_NS,
            replace_limit_price=0.54,
            replace_order_size=10.0,
            replace_price_level="replace_join_bid",
            cancel_after_ns=5 * SECOND_NS,
            scenario_name="synthetic_replace_fill",
            resolved_side="UP",
        ),
        OrderIntent.create(
            market_slug="synthetic-btc-updown-5m",
            token_asset_id="synthetic-up-token",
            token_side_normalized="UP",
            order_ts_ns=int(9.0 * SECOND_NS),
            market_close_ts_ns=close_ts_ns,
            order_side="buy",
            limit_price=0.45,
            order_size=10.0,
            price_level="near_close_bid",
            scenario_name="synthetic_near_close_no_fill_at_close",
            resolved_side="UP",
        ),
        OrderIntent.create(
            market_slug="synthetic-btc-updown-5m",
            token_asset_id="synthetic-up-token",
            token_side_normalized="UP",
            order_ts_ns=int(8.2 * SECOND_NS),
            market_close_ts_ns=close_ts_ns,
            order_side="buy",
            limit_price=0.45,
            order_size=10.0,
            price_level="live_reference_missing",
            scenario_name="synthetic_live_reference_unavailable",
            resolved_side="UP",
        ),
    ]
    frame = simulator.simulate_many(intents)
    checks = {
        "buy_fill_after_order": bool(frame.loc[frame["scenario_name"].eq("synthetic_buy_fill"), "filled"].iloc[0]),
        "post_only_rejected": bool(
            frame.loc[frame["scenario_name"].eq("synthetic_post_only_reject"), "post_only_rejected"].iloc[0]
        ),
        "cancel_blocks_fill": not bool(
            frame.loc[frame["scenario_name"].eq("synthetic_cancel_before_fill"), "filled"].iloc[0]
        ),
        "sell_fill_after_order": bool(frame.loc[frame["scenario_name"].eq("synthetic_sell_fill"), "filled"].iloc[0]),
        "partial_fill_accumulates": bool(
            frame.loc[frame["scenario_name"].eq("synthetic_partial_fill"), "partial_fill"].iloc[0]
            and frame.loc[frame["scenario_name"].eq("synthetic_partial_fill"), "fill_size"].iloc[0] == 25.0
            and frame.loc[frame["scenario_name"].eq("synthetic_partial_fill"), "fill_count"].iloc[0] == 2
        ),
        "replace_fill_after_replace": bool(
            frame.loc[frame["scenario_name"].eq("synthetic_replace_fill"), "filled"].iloc[0]
            and frame.loc[frame["scenario_name"].eq("synthetic_replace_fill"), "fill_ts_ns"].iloc[0]
            >= frame.loc[frame["scenario_name"].eq("synthetic_replace_fill"), "replace_effective_ts_ns"].iloc[0]
        ),
        "near_close_trade_at_close_does_not_fill": not bool(
            frame.loc[frame["scenario_name"].eq("synthetic_near_close_no_fill_at_close"), "filled"].iloc[0]
        ),
        "live_reference_unavailable_blocks_submission": bool(
            frame.loc[
                frame["scenario_name"].eq("synthetic_live_reference_unavailable"),
                "execution_exclusion_reason",
            ].iloc[0]
            == "live_reference_untrusted"
        ),
        "maker_fee_zero_rebate_positive_on_fills": bool(
            (frame.loc[frame["filled"].astype(bool), "fees_paid"] == 0).all()
            and (frame.loc[frame["filled"].astype(bool), "rebate_earned"] > 0).all()
        ),
        "no_fill_after_cancel": bool(
            (
                ~frame["filled"].astype(bool)
                | (frame["last_fill_ts_ns"].astype("int64") < frame["cancel_effective_ts_ns"].astype("int64"))
            ).all()
        ),
        "non_negative_time_to_fill": bool(
            (frame.loc[frame["filled"].astype(bool), "time_to_fill_s"] >= 0).all()
        ),
    }
    audit = audit_execution_results(frame)
    ledger = reconcile_execution_results(frame)
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "results": frame,
        "checks": checks,
        "failed_checks": failed_checks,
        "audit": audit,
        "ledger": ledger,
    }


def _attach_market_registry_context(root: Path, anchors: pd.DataFrame) -> pd.DataFrame:
    registry_path = root / "canonical" / "market_registry_v1.parquet"
    if not registry_path.exists() or anchors.empty:
        return anchors
    registry = pd.read_parquet(
        registry_path,
        columns=["market_slug", "market_open_ts_ns", "market_close_ts_ns", "resolved_side"],
    )
    merged = anchors.merge(registry, on="market_slug", how="left", suffixes=("", "_registry"))
    if "market_close_ts_ns" in merged.columns:
        merged["t_to_close_s"] = (
            (merged["market_close_ts_ns"] - merged["anchor_ts_ns"]) / SECOND_NS
        ).where(merged["market_close_ts_ns"].notna(), merged.get("t_to_close_s"))
    return merged


def run_real_smoke_validation(
    *,
    root: Path,
    dataset_root: Path,
    date_from: date,
    date_to: date,
    max_anchors_per_day: int = 20,
    max_intents_per_day: int = 500,
    max_markets_per_day: int = 3,
) -> dict[str, Any]:
    all_results: list[pd.DataFrame] = []
    per_day: list[dict[str, Any]] = []
    live_reference_reader = LiveReferenceEventsReader(root=root)
    simulator_config = ExecutionSimulatorConfig(
        fee_schedule=FeeSchedule(
            schedule_id="phase5_validation_polymarket_crypto_fee_v2",
        ),
        require_live_reference=True,
    )
    current = date_from
    while current <= date_to:
        date_str = current.isoformat()
        try:
            anchors = load_phase8_anchors(
                dataset_root=dataset_root,
                target_date=date_str,
                max_anchors=None,
            )
        except FileNotFoundError as exc:
            per_day.append({"date": date_str, "rows": 0, "critical_count": 1, "note": str(exc)})
            current += timedelta(days=1)
            continue
        if anchors.empty:
            per_day.append({"date": date_str, "rows": 0, "critical_count": 0, "note": "no eligible anchors"})
            current += timedelta(days=1)
            continue
        market_slugs = set(anchors["market_slug"].drop_duplicates().head(max_markets_per_day).astype(str).tolist())
        anchors = anchors.loc[anchors["market_slug"].isin(market_slugs)].copy()
        per_market_limit = max(1, (int(max_anchors_per_day) + max(1, int(max_markets_per_day)) - 1) // max(1, int(max_markets_per_day)))
        anchors = anchors.groupby("market_slug", sort=False).head(per_market_limit).reset_index(drop=True)
        anchors = _attach_market_registry_context(root, anchors)
        replay_load = load_execution_replay_for_date(
            root=root,
            target_date=current,
            market_slugs=market_slugs,
            prefer_canonical=True,
            materialize_canonical=False,
        )
        intents = generate_intents_from_anchors(
            anchors=anchors,
            replay_index=replay_load.replay_index,
            grid=ScenarioGrid(),
            max_intents=max_intents_per_day,
        )
        simulator = ExecutionSimulator(
            replay_load.replay_index,
            simulator_config,
            live_reference_reader=live_reference_reader,
        )
        results = simulator.simulate_many(intents)
        results.insert(0, "validation_date", date_str)
        audit = audit_execution_results(results.drop(columns=["validation_date"]))
        ledger = reconcile_execution_results(results.drop(columns=["validation_date"]))
        all_results.append(results)
        per_day.append(
            {
                "date": date_str,
                "rows": int(len(results)),
                "filled": audit["filled"],
                "fill_rate": audit["fill_rate"],
                "critical_count": audit["critical_count"],
                "warning_count": audit["warning_count"],
                "ledger_balances": bool(ledger["balances"]),
                "ledger_difference": float(ledger["difference"]),
                "token_replays": replay_load.replay_index.token_count,
                "markets": replay_load.replay_index.market_count,
            }
        )
        current += timedelta(days=1)
    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined_audit = audit_execution_results(combined.drop(columns=["validation_date"])) if not combined.empty else audit_execution_results(combined)
    combined_ledger = reconcile_execution_results(combined.drop(columns=["validation_date"])) if not combined.empty else {"balances": True, "difference": 0.0}
    return {"results": combined, "per_day": per_day, "audit": combined_audit, "ledger": combined_ledger}


def write_phase5_validation_report(
    *,
    root: Path = Path("data"),
    dataset_root: Path = Path("data/datasets"),
    date_from: date = date(2026, 4, 19),
    date_to: date = date(2026, 4, 23),
    output_path: Path = DEFAULT_VALIDATION_REPORT_PATH,
    smoke_output_dir: Path = DEFAULT_SMOKE_OUTPUT_DIR,
    max_anchors_per_day: int = 20,
    max_intents_per_day: int = 500,
    max_markets_per_day: int = 3,
    run_real_smoke: bool = True,
) -> dict[str, Any]:
    synthetic = run_synthetic_scenario_suite()
    real = (
        run_real_smoke_validation(
            root=root,
            dataset_root=dataset_root,
            date_from=date_from,
            date_to=date_to,
            max_anchors_per_day=max_anchors_per_day,
            max_intents_per_day=max_intents_per_day,
            max_markets_per_day=max_markets_per_day,
        )
        if run_real_smoke
        else {
            "results": pd.DataFrame(),
            "per_day": [],
            "audit": audit_execution_results(pd.DataFrame()),
            "ledger": {"balances": True, "difference": 0.0},
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_output_dir.mkdir(parents=True, exist_ok=True)
    if not real["results"].empty:
        real["results"].to_parquet(smoke_output_dir / "phase5_real_smoke_results.parquet", index=False)
    synthetic["results"].to_parquet(smoke_output_dir / "phase5_synthetic_scenario_results.parquet", index=False)

    synthetic_critical = int(synthetic["audit"]["critical_count"]) + len(synthetic["failed_checks"])
    real_critical = int(real["audit"]["critical_count"])
    ledger_critical = 0 if bool(synthetic["ledger"]["balances"]) and bool(real["ledger"]["balances"]) else 1
    verdict = "PASS" if synthetic_critical == 0 and real_critical == 0 and ledger_critical == 0 else "FAIL"
    lines = [
        "# Phase 5 Execution Simulator Validation Report V1",
        "",
        f"- Verdict: `{verdict}`",
        f"- Date range smoke-tested: `{date_from.isoformat()}` to `{date_to.isoformat()}`",
        "- Scope: execution simulator only; no Phase 6 dataset generation, no model training, no policy layer.",
        "- Primary clock: `recv_ts_ns` only.",
        "",
        "## Synthetic Scenario Suite",
        "",
        f"- Rows: `{len(synthetic['results'])}`",
        f"- Failed checks: `{len(synthetic['failed_checks'])}`",
        f"- Audit criticals: `{synthetic['audit']['critical_count']}`",
        f"- Ledger balances: `{str(bool(synthetic['ledger']['balances'])).upper()}`",
        "",
        "| check | pass |",
        "| --- | ---: |",
    ]
    for check, passed in synthetic["checks"].items():
        lines.append(f"| `{check}` | `{str(bool(passed)).upper()}` |")
    lines.extend(
        [
            "",
            "## Real Data Smoke Validation",
            "",
            "| date | rows | filled | fill_rate | markets | token_replays | criticals | warnings | ledger_balances | ledger_diff |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in real["per_day"]:
        lines.append(
            "| {date} | {rows} | {filled} | {fill_rate:.6f} | {markets} | {token_replays} | {critical_count} | {warning_count} | {ledger_balances} | {ledger_difference:.12f} |".format(
                date=row.get("date", ""),
                rows=int(row.get("rows", 0)),
                filled=int(row.get("filled", 0)),
                fill_rate=float(row.get("fill_rate", 0.0)),
                markets=int(row.get("markets", 0)),
                token_replays=int(row.get("token_replays", 0)),
                critical_count=int(row.get("critical_count", 0)),
                warning_count=int(row.get("warning_count", 0)),
                ledger_balances=str(bool(row.get("ledger_balances", False))).upper(),
                ledger_difference=float(row.get("ledger_difference", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Simulator Assumptions",
            "",
            "- Post-only orders that would cross the current book are rejected.",
            "- Fill simulation uses trade events with `order_ts_ns < recv_ts_ns < min(cancel_effective_ts_ns, market_close_ts_ns)`.",
            "- Queue model v1 is configurable; default smoke validation uses zero queue-ahead because the current normalized L2 contract exposes total depth, not price-level queue position.",
            "- Fees and rebates use the versioned Polymarket crypto fee curve: maker fee zero, taker fee curve by deploy date, and maker rebate as 20% of taker-fee equivalent.",
            "- Causal `live_reference_events_v1` context is required for real smoke submissions.",
            "- Replace v1 is cancel-old-then-submit-new after configured cancel/replace latency.",
            "- Settlement PnL uses canonical `resolved_side` when available; otherwise markout-to-close is used as a diagnostic proxy.",
            "",
            "## Audit Summary",
            "",
            f"- Synthetic audit findings: `{len(synthetic['audit']['findings'])}`",
            f"- Real smoke audit findings: `{len(real['audit']['findings'])}`",
            f"- Real smoke critical findings: `{real['audit']['critical_count']}`",
            f"- Real smoke ledger balances: `{str(bool(real['ledger']['balances'])).upper()}`",
            f"- Real smoke ledger difference: `{float(real['ledger']['difference']):.12f}`",
            "",
            "## Readiness",
            "",
        ]
    )
    if verdict == "PASS":
        lines.append("Phase 5 simulator V1 is ready as an audited simulator layer. Phase 6 execution dataset generation can be built on top of it when desired.")
    else:
        lines.append("Phase 5 simulator V1 is not ready; critical validation checks remain.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "verdict": verdict,
        "synthetic_failed_checks": synthetic["failed_checks"],
        "synthetic_audit": synthetic["audit"],
        "synthetic_ledger": synthetic["ledger"],
        "real_audit": real["audit"],
        "real_ledger": real["ledger"],
        "real_per_day": real["per_day"],
        "output_path": str(output_path),
    }
