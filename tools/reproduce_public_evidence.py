"""Recompute the public model-score and selection-bias audit artifacts.

The inputs are deliberately small enough for GitHub: validation/test prediction
arrays for five fixed seeds, the corresponding JSON calibration maps, and the
32-day candidate PnL matrix.  No private raw market data are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts/evaluation_repro_v2/predictions"
REPORTS = ROOT / "artifacts/tcn_v2_eval"
SNOOPING = ROOT / "artifacts/audit_v1/snooping_report.json"
UNIVERSE = ROOT / "artifacts/audit_v1/wrc_universe_daily_pnl.npz"
BRIER_OUT = ROOT / "artifacts/tcn_v2_eval/brier_summary.json"
AUDIT_DIR = ROOT / "artifacts/audit_v2"
AUDIT_OUT = AUDIT_DIR / "selection_audit.json"
NULL_OUT = AUDIT_DIR / "wrc_null_distribution.npz"

SEEDS = (7, 11, 23, 42, 101)
CALIBRATORS = (
    ("identity", "identity (uncalibrated)"),
    ("val_temperature", "temperature; fit on validation"),
    ("val_platt_l2", "Platt L2; fit on validation"),
    ("val_isotonic", "isotonic; fit on validation"),
    ("val_residual_shrink", "residual shrink; fit on validation"),
    ("train_dayblock_temperature", "temperature; train day-block fit"),
    ("train_dayblock_residual_shrink", "residual shrink; train day-block fit"),
)
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_REPS = 20_000
MEAN_BLOCK_DAYS = 3.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x64)
    positive = x64 >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x64[positive]))
    exp_x = np.exp(x64[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _apply_calibrator(cal: dict[str, Any], prediction: Any) -> np.ndarray:
    kind = cal["kind"]
    if kind == "identity":
        return _sigmoid(prediction["logit"])
    if kind == "temperature":
        return _sigmoid(prediction["logit"] / float(cal["temperature"]))
    if kind == "platt":
        return _sigmoid(
            float(cal["coef"]) * prediction["logit"] + float(cal["intercept"])
        )
    if kind == "isotonic":
        x = np.asarray(cal["x_thresholds"], dtype=np.float64)
        y = np.asarray(cal["y_thresholds"], dtype=np.float64)
        return np.interp(_sigmoid(prediction["logit"]), x, y)
    if kind == "residual_shrink":
        return _sigmoid(
            prediction["base_logit"] + float(cal["alpha"]) * prediction["delta"]
        )
    raise ValueError(f"unknown calibrator kind: {kind!r}")


def _assert_aligned(reference: Any, candidate: Any, split: str, seed: int) -> None:
    for key in ("y", "date", "market_slug", "ep_idx", "step_idx", "ttc_s"):
        if not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError(f"{split} seed {seed} is not row-aligned on {key}")
    if not np.allclose(reference["base_logit"], candidate["base_logit"], atol=1e-7):
        raise RuntimeError(f"{split} seed {seed} has a different market baseline")


def _cluster_bootstrap(
    loss_delta: np.ndarray, cluster: np.ndarray, *, seed: int
) -> dict[str, Any]:
    unique, inverse = np.unique(cluster.astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=loss_delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    batch = 1_000
    for start in range(0, BOOTSTRAP_REPS, batch):
        stop = min(start + batch, BOOTSTRAP_REPS)
        indices = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        draws[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {
        "clusters": len(unique),
        "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "bootstrap_probability_delta_leq_zero": float(np.mean(draws <= 0.0)),
        "reps": BOOTSTRAP_REPS,
        "seed": seed,
        "resampling": "cluster pairs; cluster sizes retained",
    }


def build_brier_summary() -> dict[str, Any]:
    reports = {
        seed: _read_json(REPORTS / f"tcn_v2_c64_b7_ttc15_150_seed{seed}.json")
        for seed in SEEDS
    }
    loaded: dict[str, dict[int, Any]] = {}
    for split in ("val", "test"):
        loaded[split] = {
            seed: np.load(PREDICTIONS / f"seed{seed}_{split}.npz", allow_pickle=False)
            for seed in SEEDS
        }
        reference = loaded[split][SEEDS[0]]
        for seed in SEEDS[1:]:
            _assert_aligned(reference, loaded[split][seed], split, seed)

    baseline: dict[str, Any] = {}
    per_calibrator: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        reference = loaded[split][SEEDS[0]]
        y = reference["y"].astype(np.float64)
        p_market = _sigmoid(reference["base_logit"])
        baseline[split] = {
            "brier": float(np.mean((p_market - y) ** 2)),
            "n_loss_band_rows": len(y),
            "markets": len(np.unique(reference["market_slug"])),
            "days": len(np.unique(reference["date"])),
        }
        for name, label in CALIBRATORS:
            losses = []
            scores: dict[str, float] = {}
            for seed in SEEDS:
                prediction = loaded[split][seed]
                probability = _apply_calibrator(reports[seed]["calibrators"][name], prediction)
                seed_loss = (probability - y) ** 2
                losses.append(seed_loss)
                scores[str(seed)] = float(np.mean(seed_loss))
            mean_row_loss = np.mean(np.stack(losses, axis=0), axis=0)
            delta = mean_row_loss - (p_market - y) ** 2
            result: dict[str, Any] = {
                "mean_brier_across_seeds": float(np.mean(mean_row_loss)),
                "min_seed_brier": float(min(scores.values())),
                "max_seed_brier": float(max(scores.values())),
                "per_seed_brier": scores,
                "delta_vs_market": float(np.mean(delta)),
                "seeds_beating_market": int(
                    sum(score < baseline[split]["brier"] for score in scores.values())
                ),
            }
            if split == "test":
                result["market_cluster_bootstrap"] = _cluster_bootstrap(
                    delta, reference["market_slug"], seed=BOOTSTRAP_SEED
                )
                result["day_cluster_bootstrap"] = _cluster_bootstrap(
                    delta, reference["date"], seed=BOOTSTRAP_SEED + 1
                )
                lo, hi = result["market_cluster_bootstrap"]["ci95"]
                result["market_cluster_ci_includes_zero"] = bool(lo <= 0.0 <= hi)
            per_calibrator.setdefault(name, {"calibrator": name, "label": label})[split] = result

    rows = [per_calibrator[name] for name, _ in CALIBRATORS]
    test_best = min(rows, key=lambda row: row["test"]["mean_brier_across_seeds"])
    val_best = min(rows, key=lambda row: row["val"]["mean_brier_across_seeds"])
    return {
        "generated_by": "tools/reproduce_public_evidence.py",
        "architecture": "c64_b7",
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "calibrator_count": len(CALIBRATORS),
        "score": "Brier loss on every eligible 200 ms row with 15-150 s to close",
        "market_baseline": baseline,
        "rows": rows,
        "lowest_observed_test_brier_post_hoc": test_best["calibrator"],
        "lowest_validation_brier": val_best["calibrator"],
        "primary_uncertainty_unit": "test market (388 clusters)",
        "headline": {
            "calibrator": test_best["calibrator"],
            "model_brier": test_best["test"]["mean_brier_across_seeds"],
            "market_brier": baseline["test"]["brier"],
            "delta_vs_market": test_best["test"]["delta_vs_market"],
            "market_cluster_ci95": test_best["test"]["market_cluster_bootstrap"]["ci95"],
            "inference": "no demonstrated improvement; the cluster interval includes zero",
        },
        "caveats": [
            "The test split (2026-06-28 through 2026-07-02) was inspected during selection and is not a pristine holdout.",
            "The five seeds share the same events; seed spread is not a confidence interval.",
            "Validation-fitted maps are evaluated in-sample in the validation column.",
            "Seven reported maps were examined. The lowest test score is explicitly labeled post hoc.",
            "Only five test days are available. The 388-market cluster interval is primary; the day-cluster interval is shown as a weak sensitivity check.",
            "Episodes use source timestamps rather than receive timestamps, so the score is a zero-latency upper bound.",
        ],
        "input_sha256": {
            "predictions": {
                f"seed{seed}_{split}": _sha256(PREDICTIONS / f"seed{seed}_{split}.npz")
                for seed in SEEDS
                for split in ("val", "test")
            },
            "reports": {
                f"seed{seed}": _sha256(
                    REPORTS / f"tcn_v2_c64_b7_ttc15_150_seed{seed}.json"
                )
                for seed in SEEDS
            },
        },
    }


def _stationary_bootstrap_indices(
    n: int, b: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    restart_probability = 1.0 / max(1.0, mean_block)
    starts = rng.integers(0, n, size=(b, n))
    restart = rng.random((b, n)) < restart_probability
    restart[:, 0] = True
    indices = np.zeros((b, n), dtype=np.int64)
    for time_index in range(n):
        if time_index == 0:
            indices[:, time_index] = starts[:, time_index]
        else:
            continuation = (indices[:, time_index - 1] + 1) % n
            indices[:, time_index] = np.where(
                restart[:, time_index], starts[:, time_index], continuation
            )
    return indices


def build_selection_audit() -> tuple[dict[str, Any], np.ndarray]:
    matrix_file = np.load(UNIVERSE, allow_pickle=False)
    matrix = np.asarray(matrix_file["daily_pnl"], dtype=np.float64)
    labels = [str(x) for x in matrix_file["labels"].tolist()]
    source_report = _read_json(SNOOPING)
    days = [str(day) for day in source_report["days"]]
    combined_by_day = source_report["slots"]["combined"]["summary"]["by_day"]
    combined = np.asarray([combined_by_day.get(day, 0.0) for day in days], dtype=np.float64)
    if matrix.shape != (len(labels), len(days)):
        raise RuntimeError("candidate matrix dimensions disagree with labels/dates")

    corrected = np.vstack((matrix, combined))
    corrected_labels = labels + ["locked_combined_two_slot_strategy"]
    means = corrected.mean(axis=1)
    best_index = int(np.argmax(means))
    observed_max = float(means[best_index])
    centered = corrected - means[:, None]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = _stationary_bootstrap_indices(
        corrected.shape[1], BOOTSTRAP_REPS, MEAN_BLOCK_DAYS, rng
    )
    null_max = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPS):
        null_max[replicate] = centered[:, indices[replicate]].mean(axis=1).max()
    exceedances = int(np.sum(null_max >= observed_max))
    p_value = float((exceedances + 1) / (BOOTSTRAP_REPS + 1))

    original = source_report["whites_reality_check"]
    audit = {
        "generated_by": "tools/reproduce_public_evidence.py",
        "method": "White's Reality Check with recentered candidate returns and stationary day bootstrap",
        "days": days,
        "n_days": len(days),
        "mean_block_days": MEAN_BLOCK_DAYS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "original_artifact": {
            "n_candidates": int(matrix.shape[0]),
            "stored_p_value": float(original["wrc_p_value"]),
            "stored_bootstrap_reps": int(original["bootstrap_reps"]),
            "issue": "the searched matrix omitted the subsequently reported combined locked strategy",
        },
        "corrected_public_reproduction": {
            "n_candidates": int(corrected.shape[0]),
            "appended_candidate": corrected_labels[-1],
            "appended_total_pnl": float(combined.sum()),
            "observed_max_mean_daily_pnl": observed_max,
            "best_candidate": corrected_labels[best_index],
            "exceedances": exceedances,
            "finite_sample_corrected_p_value": p_value,
            "passes_10pct": bool(p_value < 0.10),
            "null_quantiles": {
                str(q): float(np.quantile(null_max, q))
                for q in (0.5, 0.9, 0.95, 0.975, 0.99)
            },
        },
        "conclusion": "The apparent backtest winner does not survive the 847-candidate selection adjustment.",
        "limits": [
            "The candidate count is a documented lower bound, not a complete count of every human or computational trial.",
            "The historical PnL matrix uses source-time books and therefore is not live-causal performance evidence.",
            "The original 846-candidate stored p-value is retained for provenance, not treated as reproducible with this corrected universe.",
        ],
        "input_sha256": {
            "wrc_universe_daily_pnl": _sha256(UNIVERSE),
            "snooping_report": _sha256(SNOOPING),
        },
    }
    return audit, null_max


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_canonical(path: Path, text: str) -> None:
    """Write LF regardless of platform.

    Path.write_text would translate LF to CRLF on Windows, which makes the
    generated artifact depend on the machine that produced it and breaks
    _check_canonical on any other platform.
    """
    path.write_bytes(text.encode("utf-8"))


def _read_canonical(path: Path) -> str:
    """Read a generated artifact with newlines normalized.

    Path.read_text applies universal newlines on Windows only, so comparing its
    result against LF text silently passes there and fails on Linux.
    """
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _check_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_file():
        raise SystemExit(f"missing generated artifact: {path.relative_to(ROOT)}")
    if _read_canonical(path) != _canonical_json(payload):
        raise SystemExit(f"stale generated artifact: {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed outputs are stale")
    args = parser.parse_args()

    brier = build_brier_summary()
    audit, null_max = build_selection_audit()
    if args.check:
        _check_json(BRIER_OUT, brier)
        _check_json(AUDIT_OUT, audit)
        if not NULL_OUT.is_file():
            raise SystemExit(f"missing generated artifact: {NULL_OUT.relative_to(ROOT)}")
        stored = np.load(NULL_OUT, allow_pickle=False)
        if not np.array_equal(stored["null_max_mean_daily_pnl"], null_max):
            raise SystemExit(f"stale generated artifact: {NULL_OUT.relative_to(ROOT)}")
        print("public evidence artifacts are current")
        return 0

    _write_canonical(BRIER_OUT, _canonical_json(brier))
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    _write_canonical(AUDIT_OUT, _canonical_json(audit))
    np.savez_compressed(
        NULL_OUT,
        null_max_mean_daily_pnl=null_max,
        observed_max_mean_daily_pnl=np.asarray(
            audit["corrected_public_reproduction"]["observed_max_mean_daily_pnl"]
        ),
    )
    print(
        json.dumps(
            {
                "headline_brier_delta": brier["headline"]["delta_vs_market"],
                "headline_market_cluster_ci95": brier["headline"]["market_cluster_ci95"],
                "selection_candidates": audit["corrected_public_reproduction"]["n_candidates"],
                "selection_p_value": audit["corrected_public_reproduction"][
                    "finite_sample_corrected_p_value"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
