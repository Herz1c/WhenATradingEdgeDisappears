"""Reproducible Monte Carlo week sampling.

A "week" = 7 unique dates drawn (without replacement within a week) from a
pool. Different weeks may share days; within one week no day repeats.

Persists to data/backtest/mc_weeks_<N>.json so every strategy candidate is
evaluated on the same set of weeks (apples-to-apples).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def available_dates(dataset_subdir: str = "microstructure_sequence_dataset_v1_tabular") -> list[str]:
    base = _REPO_ROOT / "data" / "datasets" / dataset_subdir
    files = sorted(base.glob("*.parquet"))
    return [f.stem for f in files if "-" in f.stem]


def generate_mc_weeks(n_weeks: int = 20, days_per_week: int = 7,
                      seed: int = 0, pool: list[str] | None = None,
                      cache_path: Path | None = None) -> list[list[str]]:
    """Generate n_weeks of unique-day samples. Cached to disk for stability."""
    pool = pool or available_dates()
    if len(pool) < days_per_week:
        raise ValueError(f"pool has only {len(pool)} dates, need >= {days_per_week}")

    if cache_path is None:
        cache_path = _REPO_ROOT / "data" / "backtest" / f"mc_weeks_{n_weeks}_seed{seed}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    weeks: list[list[str]] = []
    for i in range(n_weeks):
        rng = random.Random(seed * 10_000 + i)
        week = sorted(rng.sample(pool, days_per_week))
        weeks.append(week)

    cache_path.write_text(json.dumps(weeks, indent=2))
    return weeks


if __name__ == "__main__":
    weeks = generate_mc_weeks(n_weeks=20, seed=0)
    print(f"Generated {len(weeks)} weeks. First 3:")
    for w in weeks[:3]:
        print(f"  {w}")
