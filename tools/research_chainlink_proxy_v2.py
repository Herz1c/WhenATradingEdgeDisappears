"""Research the v2 Chainlink proxy: candidate source blends, lag alignment, and residual error."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SECOND_NS = 1_000_000_000
DEFAULT_CACHE_PATH = Path("data/_cache/synthetic_chainlink_experiment.parquet")
DEFAULT_REPORT_PATH = Path("docs/chainlink_proxy_v2_research.md")
DEFAULT_JSON_PATH = Path("docs/chainlink_proxy_v2_research.json")


@dataclass(slots=True)
class CandidateMetrics:
    candidate: str
    delay_seconds: int
    window_seconds: int | None
    aggregation: str
    rows: int
    active_rows: int
    active_fraction: float
    median_abs_error_usd: float
    p90_abs_error_usd: float
    p95_abs_error_usd: float
    mean_error_usd: float
    within_3_usd: float
    leakage_safe: bool
    notes: str


def _finite_metrics(
    *,
    candidate: str,
    delay_seconds: int,
    window_seconds: int | None,
    aggregation: str,
    prediction: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    leakage_safe: bool,
    notes: str,
) -> CandidateMetrics:
    error = prediction - target
    mask = np.isfinite(error)
    finite_error = error[mask]
    finite_abs = np.abs(finite_error)
    if len(finite_error) == 0:
        raise ValueError(f"Candidate {candidate!r} produced no finite predictions.")
    active_mask = active[mask]
    return CandidateMetrics(
        candidate=candidate,
        delay_seconds=delay_seconds,
        window_seconds=window_seconds,
        aggregation=aggregation,
        rows=int(mask.sum()),
        active_rows=int(active_mask.sum()),
        active_fraction=float(active_mask.mean()) if len(active_mask) else 0.0,
        median_abs_error_usd=float(np.median(finite_abs)),
        p90_abs_error_usd=float(np.quantile(finite_abs, 0.90)),
        p95_abs_error_usd=float(np.quantile(finite_abs, 0.95)),
        mean_error_usd=float(finite_error.mean()),
        within_3_usd=float((finite_abs <= 3.0).mean()),
        leakage_safe=leakage_safe,
        notes=notes,
    )


def _aggregate(values: np.ndarray, method: str) -> float:
    if method == "last":
        return float(values[-1])
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(np.median(values))
    raise ValueError(f"Unsupported aggregation method: {method}")


def _delayed_residual_candidate(
    *,
    ts_ns: np.ndarray,
    base_proxy: np.ndarray,
    target: np.ndarray,
    delay_seconds: int,
    window_seconds: int,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray]:
    residual = target - base_proxy
    cutoffs = ts_ns - (delay_seconds * SECOND_NS)
    lefts = np.searchsorted(ts_ns, cutoffs - (window_seconds * SECOND_NS), side="left")
    rights = np.searchsorted(ts_ns, cutoffs, side="right")

    prediction = base_proxy.copy()
    active = np.zeros(len(ts_ns), dtype=bool)
    for index, (left, right) in enumerate(zip(lefts, rights, strict=True)):
        if right <= left:
            continue
        prediction[index] = base_proxy[index] + _aggregate(residual[left:right], aggregation)
        active[index] = True
    return prediction, active


def _last_residual_candidate(
    *,
    ts_ns: np.ndarray,
    base_proxy: np.ndarray,
    target: np.ndarray,
    delay_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    residual = target - base_proxy
    cutoffs = ts_ns - (delay_seconds * SECOND_NS)
    indices = np.searchsorted(ts_ns, cutoffs, side="right") - 1
    prediction = base_proxy.copy()
    active = indices >= 0
    prediction[active] = base_proxy[active] + residual[indices[active]]
    return prediction, active


def _candidate_rows(metrics: Iterable[CandidateMetrics]) -> list[str]:
    rows = [
        "| candidate | delay_s | window_s | agg | medAE | p90 | p95 | within_3 | active | safe |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in metrics:
        window = "n/a" if item.window_seconds is None else str(item.window_seconds)
        rows.append(
            "| "
            + " | ".join(
                [
                    item.candidate,
                    str(item.delay_seconds),
                    window,
                    item.aggregation,
                    f"{item.median_abs_error_usd:.4f}",
                    f"{item.p90_abs_error_usd:.4f}",
                    f"{item.p95_abs_error_usd:.4f}",
                    f"{item.within_3_usd * 100:.2f}%",
                    f"{item.active_fraction * 100:.2f}%",
                    "yes" if item.leakage_safe else "diagnostic",
                ]
            )
            + " |"
        )
    return rows


def _date_breakdown(frame: pd.DataFrame, prediction: np.ndarray, active: np.ndarray) -> list[dict[str, object]]:
    error = np.abs(prediction - frame["chainlink_price"].to_numpy(dtype=float))
    breakdown: list[dict[str, object]] = []
    for date_value, group in pd.DataFrame(
        {
            "date": frame["date"].astype(str).to_numpy(),
            "error": error,
            "active": active,
        }
    ).groupby("date", sort=True):
        values = group["error"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        breakdown.append(
            {
                "date": str(date_value),
                "rows": int(len(values)),
                "active_fraction": float(group["active"].mean()),
                "median_abs_error_usd": float(np.median(values)),
                "p90_abs_error_usd": float(np.quantile(values, 0.90)),
                "within_3_usd": float((values <= 3.0).mean()),
            }
        )
    return breakdown


def build_report(*, cache_path: Path, report_path: Path, json_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(cache_path).sort_values("ts_ns", kind="stable").reset_index(drop=True)
    frame = frame[
        frame["baseline"].notna()
        & frame["chainlink_price"].notna()
        & frame["spot"].notna()
    ].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No usable rows found in {cache_path}")

    ts_ns = frame["ts_ns"].to_numpy(dtype=np.int64)
    target = frame["chainlink_price"].to_numpy(dtype=float)
    baseline = frame["baseline"].to_numpy(dtype=float)

    metrics: list[CandidateMetrics] = []

    metrics.append(
        _finite_metrics(
            candidate="spot_premium_calibrated_v1",
            delay_seconds=0,
            window_seconds=None,
            aggregation="none",
            prediction=baseline,
            target=target,
            active=np.ones(len(frame), dtype=bool),
            leakage_safe=True,
            notes="Causal venue-only baseline from existing synthetic Chainlink experiment cache.",
        )
    )

    last_prediction, last_active = _last_residual_candidate(
        ts_ns=ts_ns,
        base_proxy=baseline,
        target=target,
        delay_seconds=1800,
    )
    metrics.append(
        _finite_metrics(
            candidate="baseline_plus_last_30m_residual",
            delay_seconds=1800,
            window_seconds=None,
            aggregation="last",
            prediction=last_prediction,
            target=target,
            active=last_active,
            leakage_safe=True,
            notes="Uses the latest Chainlink residual whose event timestamp is at least 1800 seconds old.",
        )
    )

    strict_candidates: list[tuple[CandidateMetrics, np.ndarray, np.ndarray]] = []
    for window_seconds in (600, 900, 1200, 1500, 1800, 2100, 2400, 3000, 3600, 7200):
        for aggregation in ("mean", "median", "last"):
            prediction, active = _delayed_residual_candidate(
                ts_ns=ts_ns,
                base_proxy=baseline,
                target=target,
                delay_seconds=1800,
                window_seconds=window_seconds,
                aggregation=aggregation,
            )
            item = _finite_metrics(
                candidate=f"baseline_plus_{aggregation}_residual_{window_seconds}s_ending_30m_old",
                delay_seconds=1800,
                window_seconds=window_seconds,
                aggregation=aggregation,
                prediction=prediction,
                target=target,
                active=active,
                leakage_safe=True,
                notes="Calibration window ends at t-1800s, so no Chainlink event newer than 30 minutes is used.",
            )
            strict_candidates.append((item, prediction, active))

    strict_candidates.sort(key=lambda row: row[0].median_abs_error_usd)
    metrics.extend(item for item, _, _ in strict_candidates[:8])
    best_strict, best_prediction, best_active = strict_candidates[0]

    diagnostic_candidates: list[CandidateMetrics] = []
    for delay_seconds in (0, 5, 10, 30, 60):
        prediction, active = _delayed_residual_candidate(
            ts_ns=ts_ns,
            base_proxy=baseline,
            target=target,
            delay_seconds=delay_seconds,
            window_seconds=300,
            aggregation="median",
        )
        diagnostic_candidates.append(
            _finite_metrics(
                candidate=f"diagnostic_baseline_plus_median_residual_300s_delay_{delay_seconds}s",
                delay_seconds=delay_seconds,
                window_seconds=300,
                aggregation="median",
                prediction=prediction,
                target=target,
                active=active,
                leakage_safe=False,
                notes=(
                    "Diagnostic only. This uses Chainlink observations much newer than the strict 1800s "
                    "calibration floor and must not be promoted without a source-availability decision."
                ),
            )
        )

    date_breakdown = _date_breakdown(frame, best_prediction, best_active)
    payload = {
        "cache_path": str(cache_path),
        "row_count": int(len(frame)),
        "best_strict_candidate": asdict(best_strict),
        "strict_candidates": [asdict(item) for item, _, _ in strict_candidates],
        "diagnostic_candidates": [asdict(item) for item in diagnostic_candidates],
        "best_strict_date_breakdown": date_breakdown,
    }

    report_lines = [
        "# Chainlink Proxy v2 Research",
        "",
        "## Scope",
        "",
        "- Target: minimize synthetic Chainlink BTC/USD error without data leakage.",
        "- Evaluation rows: Chainlink public page event timestamps from the experiment cache.",
        "- Base proxy: `spot_premium_calibrated_v1` from the existing experiment cache.",
        "- Strict calibration rule: Chainlink calibration observations are usable only if their event timestamp is at least `1800s` older than the prediction timestamp.",
        "- Fallback: if no strict calibration window exists yet, the base proxy is emitted unchanged.",
        "",
        "## Strict 1800s-Safe Results",
        "",
        *_candidate_rows([metrics[0], metrics[1], *[item for item, _, _ in strict_candidates[:8]]]),
        "",
        "## Best Strict Candidate By Date",
        "",
        "| date | rows | active | medAE | p90 | within_3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in date_breakdown:
        report_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["date"]),
                    str(row["rows"]),
                    f"{float(row['active_fraction']) * 100:.2f}%",
                    f"{float(row['median_abs_error_usd']):.4f}",
                    f"{float(row['p90_abs_error_usd']):.4f}",
                    f"{float(row['within_3_usd']) * 100:.2f}%",
                ]
            )
            + " |"
        )
    report_lines.extend(
        [
            "",
            "## Diagnostic Only: Non-1800s Availability",
            "",
            *_candidate_rows(diagnostic_candidates),
            "",
            "These diagnostic candidates are useful for understanding the upper bound if the public page is treated as available shortly after the displayed Chainlink timestamp. They are not strict-training-safe under the current roadmap/policy contract.",
            "",
            "## Conclusion",
            "",
            f"- Best strict candidate: `{best_strict.candidate}`.",
            f"- Strict median absolute error: `{best_strict.median_abs_error_usd:.4f} USD`.",
            f"- Strict p90 absolute error: `{best_strict.p90_abs_error_usd:.4f} USD`.",
            "- This improves materially over the current base proxy, but it does not reproduce a sub-2.5 USD median error under the strict 1800s calibration floor.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_report(
        cache_path=args.cache_path,
        report_path=args.report_path,
        json_path=args.json_path,
    )
    best = payload["best_strict_candidate"]
    print(
        json.dumps(
            {
                "row_count": payload["row_count"],
                "best_strict_candidate": best["candidate"],
                "median_abs_error_usd": best["median_abs_error_usd"],
                "p90_abs_error_usd": best["p90_abs_error_usd"],
                "report_path": str(args.report_path),
                "json_path": str(args.json_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
