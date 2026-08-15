from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import pandas as pd

from .execution_simulator import RESULT_COLUMNS

FORBIDDEN_PATTERNS = (
    "fng",
    "rtds",
    "bybit",
    "coinbase",
    "deribit",
    "chainlink_onchain",
    "chainlink_public",
    "binance_spot_mid",
    "binance_usdm_mid",
)


def _finding(severity: str, check: str, message: str) -> dict[str, str]:
    return {"severity": severity, "check": check, "message": message}


def _nan_populated(series: pd.Series) -> pd.Series:
    return series.notna() & ~series.map(lambda value: isinstance(value, float) and math.isnan(value))


def audit_execution_results(frame: pd.DataFrame) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    columns = list(frame.columns)
    missing = [column for column in RESULT_COLUMNS if column not in columns]
    extra = [column for column in columns if column not in RESULT_COLUMNS]
    if missing or extra:
        findings.append(_finding("CRITICAL", "schema", f"Schema mismatch missing={missing} extra={extra}"))
    forbidden_hits = [column for column in columns if any(pattern in column.lower() for pattern in FORBIDDEN_PATTERNS)]
    if forbidden_hits:
        findings.append(_finding("CRITICAL", "forbidden_sources", f"Forbidden source columns present: {forbidden_hits}"))
    if frame.empty:
        findings.append(_finding("WARNING", "empty_results", "Execution simulator produced no result rows"))
        return _summary(frame, findings)

    duplicate_ids = int(frame["simulator_id"].duplicated().sum()) if "simulator_id" in frame else 0
    if duplicate_ids:
        findings.append(_finding("CRITICAL", "simulator_id_unique", f"{duplicate_ids} duplicate simulator_id values"))

    invalid_side = int((~frame["order_side"].isin(["buy", "sell"])).sum()) if "order_side" in frame else 0
    invalid_token = int((~frame["token_side_normalized"].isin(["UP", "DOWN"])).sum()) if "token_side_normalized" in frame else 0
    if invalid_side:
        findings.append(_finding("CRITICAL", "order_side", f"{invalid_side} rows have invalid order_side"))
    if invalid_token:
        findings.append(_finding("CRITICAL", "token_side", f"{invalid_token} rows have invalid token_side_normalized"))

    filled = frame["filled"].astype(bool)
    if {"fill_ts_ns", "order_ts_ns"}.issubset(frame.columns):
        before_order = int((filled & (frame["fill_ts_ns"] <= frame["order_ts_ns"])).sum())
        if before_order:
            findings.append(_finding("CRITICAL", "fill_causality", f"{before_order} fills occur at/before order_ts_ns"))
    if {"fill_ts_ns", "cancel_effective_ts_ns"}.issubset(frame.columns):
        fill_end_col = "last_fill_ts_ns" if "last_fill_ts_ns" in frame.columns else "fill_ts_ns"
        after_cancel = int((filled & (frame[fill_end_col] >= frame["cancel_effective_ts_ns"])).sum())
        if after_cancel:
            findings.append(_finding("CRITICAL", "cancel_causality", f"{after_cancel} fills occur at/after cancel_effective_ts_ns"))
    if {"fill_ts_ns", "market_close_ts_ns"}.issubset(frame.columns):
        at_or_after_close = int((filled & (frame["fill_ts_ns"] >= frame["market_close_ts_ns"])).sum())
        if at_or_after_close:
            findings.append(_finding("CRITICAL", "close_causality", f"{at_or_after_close} fills occur at/after market_close_ts_ns"))
    if "time_to_fill_s" in frame.columns:
        negative_ttf = int((filled & (frame["time_to_fill_s"] < 0)).sum())
        if negative_ttf:
            findings.append(_finding("CRITICAL", "time_to_fill", f"{negative_ttf} fills have negative time_to_fill_s"))
    if {"live_reference_recv_ts_ns", "order_ts_ns", "live_reference_trusted", "execution_exclusion_reason"}.issubset(frame.columns):
        future_live = int((frame["live_reference_recv_ts_ns"] > frame["order_ts_ns"]).sum())
        if future_live:
            findings.append(_finding("CRITICAL", "live_reference_causality", f"{future_live} live-reference rows are after order_ts_ns"))
        missing_live_submission = int(
            (
                frame["submitted"].astype(bool)
                & ~frame["live_reference_trusted"].astype(bool)
            ).sum()
        )
        if missing_live_submission:
            findings.append(
                _finding(
                    "CRITICAL",
                    "live_reference_policy",
                    f"{missing_live_submission} submitted rows lack trusted live_reference_events_v1 context",
                )
            )
    if {"replace_requested", "replace_effective_ts_ns", "order_ts_ns"}.issubset(frame.columns):
        bad_replace_time = int(
            (
                frame["replace_requested"].astype(bool)
                & (frame["replace_effective_ts_ns"] <= frame["order_ts_ns"])
            ).sum()
        )
        if bad_replace_time:
            findings.append(_finding("CRITICAL", "replace_timing", f"{bad_replace_time} replacement orders are effective at/before order time"))

    if {"post_only", "post_only_crossed_book", "post_only_rejected"}.issubset(frame.columns):
        missed_rejects = int(
            (
                frame["post_only"].astype(bool)
                & frame["post_only_crossed_book"].astype(bool)
                & ~frame["post_only_rejected"].astype(bool)
            ).sum()
        )
        if missed_rejects:
            findings.append(_finding("CRITICAL", "post_only", f"{missed_rejects} crossing post-only rows were not rejected"))

    for horizon_s in (1, 3, 5, 15):
        value_col = f"markout_{horizon_s}s"
        available_col = f"markout_{horizon_s}s_available"
        if value_col in frame.columns and available_col in frame.columns:
            bad_populated = int((~frame[available_col].astype(bool) & _nan_populated(frame[value_col])).sum())
            if bad_populated:
                findings.append(
                    _finding(
                        "CRITICAL",
                        f"markout_{horizon_s}s_contract",
                        f"{bad_populated} rows have {value_col} populated while unavailable",
                    )
                )
            past_close = int(
                (
                    frame[available_col].astype(bool)
                    & (frame["fill_ts_ns"] + horizon_s * 1_000_000_000 > frame["market_close_ts_ns"])
                ).sum()
            )
            if past_close:
                findings.append(
                    _finding(
                        "CRITICAL",
                        f"markout_{horizon_s}s_boundary",
                        f"{past_close} rows have {available_col}=True past market close",
                    )
                )

    if {"gross_pnl_to_close", "net_pnl_to_close", "fees_paid", "rebate_earned"}.issubset(frame.columns):
        pnl = frame.loc[filled & frame["gross_pnl_to_close"].notna()].copy()
        if not pnl.empty:
            expected = pnl["gross_pnl_to_close"] - pnl["fees_paid"] + pnl["rebate_earned"]
            mismatch = int(((pnl["net_pnl_to_close"] - expected).abs() > 1e-9).sum())
            if mismatch:
                findings.append(_finding("CRITICAL", "pnl_reconciliation", f"{mismatch} rows fail PnL reconciliation"))
    if {"fees_paid", "rebate_earned"}.issubset(frame.columns):
        negative_fees = int((frame["fees_paid"] < -1e-12).sum())
        negative_rebates = int((frame["rebate_earned"] < -1e-12).sum())
        if negative_fees:
            findings.append(_finding("CRITICAL", "fee_sign", f"{negative_fees} rows have negative fees_paid"))
        if negative_rebates:
            findings.append(_finding("CRITICAL", "rebate_sign", f"{negative_rebates} rows have negative rebate_earned"))
    if {"maker_fill", "fees_paid"}.issubset(frame.columns):
        maker_nonzero_fees = int(
            (frame["maker_fill"].astype(bool) & (frame["fees_paid"].abs() > 1e-12)).sum()
        )
        if maker_nonzero_fees:
            findings.append(
                _finding(
                    "CRITICAL",
                    "maker_fee_zero",
                    f"{maker_nonzero_fees} maker fills have non-zero fees_paid",
                )
            )
    if {"taker_fill", "filled", "fees_paid"}.issubset(frame.columns):
        taker_zero_fees = int(
            (
                frame["filled"].astype(bool)
                & frame["taker_fill"].astype(bool)
                & (frame["fees_paid"] <= 0)
            ).sum()
        )
        if taker_zero_fees:
            findings.append(
                _finding(
                    "CRITICAL",
                    "taker_fee_positive",
                    f"{taker_zero_fees} taker fills have zero/negative fees_paid",
                )
            )
    if {"partial_fill", "fill_count"}.issubset(frame.columns):
        bad_partial_counts = int((frame["partial_fill"].astype(bool) & (frame["fill_count"] <= 0)).sum())
        if bad_partial_counts:
            findings.append(_finding("CRITICAL", "partial_fill_contract", f"{bad_partial_counts} partial fills have no fill slices"))

    fill_rate = float(filled.mean())
    if fill_rate > 0.80:
        findings.append(_finding("WARNING", "fill_rate", f"Fill rate {fill_rate:.3f} is suspiciously high"))
    if fill_rate < 0.001 and len(frame) >= 100:
        findings.append(_finding("WARNING", "fill_rate", f"Fill rate {fill_rate:.3f} is extremely low"))
    for flag in ("stale_book", "crossed_book", "one_sided_book", "fill_uncertain"):
        if flag in frame.columns:
            rate = float(frame[flag].astype(bool).mean())
            threshold = 0.03 if flag != "fill_uncertain" else 0.10
            if rate > threshold:
                findings.append(_finding("WARNING", flag, f"{flag} rate {rate:.4f} above threshold {threshold:.2f}"))

    return _summary(frame, findings)


def _summary(frame: pd.DataFrame, findings: list[dict[str, str]]) -> dict[str, Any]:
    rows = int(len(frame))
    filled = int(frame["filled"].sum()) if "filled" in frame and rows else 0
    critical_count = sum(1 for finding in findings if finding["severity"] == "CRITICAL")
    warning_count = sum(1 for finding in findings if finding["severity"] == "WARNING")
    return {
        "rows": rows,
        "filled": filled,
        "fill_rate": filled / rows if rows else 0.0,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "findings": findings,
    }


def write_execution_audit_report(*, frame: pd.DataFrame, output_path: Path, title: str = "Phase 5 Execution Audit") -> dict[str, Any]:
    report = audit_execution_results(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Filled rows: `{report['filled']}`",
        f"- Fill rate: `{report['fill_rate']:.6f}`",
        f"- Critical findings: `{report['critical_count']}`",
        f"- Warning findings: `{report['warning_count']}`",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        for idx, finding in enumerate(report["findings"], start=1):
            lines.append(
                f"- `{finding['severity']}-{idx:03d}` `{finding['check']}`: {finding['message']}"
            )
    else:
        lines.append("- No findings.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
