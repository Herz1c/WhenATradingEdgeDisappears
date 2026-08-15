from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"event": "malformed_line"})
    return rows


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def generate_daily_live_audit(date: str, trade_log_path: str, output_path: str) -> dict:
    rows = _load_jsonl(Path(trade_log_path))
    decisions = [row for row in rows if row.get("event") == "decision"]
    orders = [row for row in rows if row.get("event") == "order"]
    fills = [row for row in rows if row.get("event") == "fill"]
    latencies = [float(row.get("round_trip_ms", 0.0) or 0.0) for row in rows if "round_trip_ms" in row]
    freshness = [float(row.get("live_ref_age_s", 0.0) or 0.0) for row in rows if "live_ref_age_s" in row]
    kill_events = [row for row in rows if str(row.get("event", "")).upper() == "KILL"]
    alerts = [row for row in rows if str(row.get("event", "")).upper().startswith("ALERT")]

    gross_pnl = sum(float(row.get("gross_pnl", 0.0) or 0.0) for row in fills)
    net_pnl = sum(float(row.get("net_pnl", 0.0) or 0.0) for row in fills)
    fees = sum(float(row.get("fees_paid", 0.0) or 0.0) for row in fills)
    rebates = sum(float(row.get("rebate_earned", 0.0) or 0.0) for row in fills)
    adverse_fill_rate = (
        sum(1 for row in fills if row.get("adverse_selection_5s")) / len(fills)
        if fills
        else 0.0
    )
    freshness_ok = (
        sum(1 for value in freshness if value <= 30.0) / len(freshness)
        if freshness
        else 1.0
    )
    verdict = "NORMAL"
    if kill_events or net_pnl < -50.0 or freshness_ok < 0.95:
        verdict = "CRITICAL"
    elif alerts or adverse_fill_rate > 0.25:
        verdict = "DEGRADED"

    report = {
        "date": date,
        "total_decisions": len(decisions),
        "total_orders_submitted": len(orders),
        "fills": len(fills),
        "fill_rate": len(fills) / len(orders) if orders else 0.0,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "fees_paid": fees,
        "rebates_earned": rebates,
        "latency": {"p50": _p(latencies, 0.50), "p95": _p(latencies, 0.95), "p99": _p(latencies, 0.99)},
        "live_reference_freshness_ok_rate": freshness_ok,
        "kill_switch_events": len(kill_events),
        "alert_events": len(alerts),
        "markets_traded": sorted({str(row.get("market_slug")) for row in fills if row.get("market_slug")}),
        "adverse_fill_rate": adverse_fill_rate,
        "verdict": verdict,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Live Audit {date}",
        "",
        f"- Verdict: {verdict}",
        f"- Total decisions: {len(decisions)}",
        f"- Orders submitted: {len(orders)}",
        f"- Fills: {len(fills)}",
        f"- Fill rate: {report['fill_rate']:.6f}",
        f"- Gross PnL: {gross_pnl:.6f}",
        f"- Net PnL: {net_pnl:.6f}",
        f"- Fees/Rebates: {fees:.6f} / {rebates:.6f}",
        f"- Latency p50/p95/p99: {report['latency']['p50']:.3f} / {report['latency']['p95']:.3f} / {report['latency']['p99']:.3f}",
        f"- Live reference freshness ok rate: {freshness_ok:.6f}",
        f"- Kill switch events: {len(kill_events)}",
        f"- Alert events: {len(alerts)}",
        f"- Markets traded: {len(report['markets_traded'])}",
        f"- Adverse fill rate: {adverse_fill_rate:.6f}",
    ]
    if latencies:
        lines.append(f"- Mean latency: {mean(latencies):.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["output_path"] = str(path)
    return report

