"""Feature cleanup — neutralize the trojan-horse features identified by the
diagnose_moderate_extreme analysis.

PROBLEM:
  Three features have catastrophic distribution shift (PSI > 1.0) between
  the training window (2026-04-19 .. 2026-05-06) and the OOS window
  (2026-05-13 .. 2026-05-16):

    live_btc_usd                   PSI 7.19   (BTC went $77.5k -> $79.7k, σ halved)
    price_to_beat                  PSI 7.85   (functionally derivative of live_btc_usd)
    live_applied_bias_age_seconds  PSI 2.43   (operational shift: 14h stale -> 33min stale)

  19 of 22 LightGBM models put 10-22% of importance on these features.
  Two 08c targets (moderate_5c_5s, extreme_10c_5s) collapsed/inverted OOS
  because they over-relied on this contaminated signal.

FIX (applied here):
  1. DROP `live_btc_usd`   - redundant with derived BTC features (btc_return_*s, btc_realized_vol_30s)
  2. DROP `price_to_beat`  - functionally derivative of live_btc_usd (PSI shifts identically)
                             ; signal it carries is already in delta_to_strike + live_btc_usd
  3. TRANSFORM `live_applied_bias_age_seconds`
     -> `live_bias_age_log_clipped` = log1p(min(seconds, 600))
        Bounds the value (so a 25x shift in raw seconds doesn't fly off the
        end of training distribution) and log-compresses so tree splits are
        on a sane scale.  Categorical companion `live_bias_age_bucket` also
        provided for models that prefer discrete signals.

DESIGN:
  This module is the single source of truth for the cleanup.  Both training
  scripts and inference pipelines should call `clean_features(df)` after
  feature augmentation (e.g. after `augment_btc_features`) and use
  `cleaned_feature_list(BASE_FEATURES)` instead of the raw BASE_FEATURES list.

USAGE (training):
    from feature_cleanup import clean_features, cleaned_feature_list

    df = augment_btc_features(df)
    df = clean_features(df)
    feature_cols = cleaned_feature_list(BASE_FEATURES + DERIVED_FEATURES)

USAGE (inference):
    df = add_btc_features(df, date)         # existing
    df = clean_features(df)                  # new
    X = make_X(df, cleaned_feature_list(FEAT_49))
"""
from __future__ import annotations

import polars as pl

# ── Trojan registry ──────────────────────────────────────────────────────────

TROJAN_HARD = ("live_btc_usd", "price_to_beat", "live_applied_bias_age_seconds")

# Features dropped entirely (redundant with derived features)
DROP_FEATURES = ("live_btc_usd", "price_to_beat")

# Features transformed (old_name -> new_name); old col is dropped after transform
TRANSFORM_FEATURES = {
    "live_applied_bias_age_seconds": "live_bias_age_log_clipped",
}

# Companion categorical (optional, kept in df but not required by models)
BIAS_AGE_BUCKETS = {
    "fresh":      (0,      60),       # < 1 min
    "stale":      (60,     300),      # 1-5 min
    "very_stale": (300,    3600),     # 5min - 1h
    "ancient":    (3600,   float("inf")),  # > 1 h
}

MAX_BIAS_AGE_S = 600  # clip at 10 minutes — anything older is "non-functional" anyway


def clean_features(df: pl.DataFrame, *,
                   add_bucket: bool = True,
                   keep_originals: bool = False) -> pl.DataFrame:
    """Apply the trojan-feature cleanup to a polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input frame with raw features (post-augmentation).
    add_bucket : bool, default True
        Add `live_bias_age_bucket` categorical companion column.
    keep_originals : bool, default False
        If True, keep the raw trojan columns alongside cleaned ones (for
        diagnostic comparison). Default False (drop them).
    """
    # 1. Transform bias age -> log-clipped
    # If the source column isn't in the dataset (e.g. v3 event64 doesn't
    # include it), synthesize it at MAX_BIAS_AGE_S so the cleaned columns
    # always exist with sane defaults.  Otherwise downstream feature lists
    # that include `live_bias_age_log_clipped` would error.
    if "live_applied_bias_age_seconds" not in df.columns:
        df = df.with_columns(
            pl.lit(MAX_BIAS_AGE_S).cast(pl.Float64).alias("live_applied_bias_age_seconds")
        )
    df = df.with_columns(
        pl.col("live_applied_bias_age_seconds")
          .fill_null(MAX_BIAS_AGE_S)
          .clip(0, MAX_BIAS_AGE_S)
          .log1p()
          .alias("live_bias_age_log_clipped")
    )
    if add_bucket:
        df = df.with_columns(
            pl.when(pl.col("live_applied_bias_age_seconds") < 60).then(0)
              .when(pl.col("live_applied_bias_age_seconds") < 300).then(1)
              .when(pl.col("live_applied_bias_age_seconds") < 3600).then(2)
              .otherwise(3).cast(pl.Int8)
              .alias("live_bias_age_bucket")
        )
    if not keep_originals:
        df = df.drop("live_applied_bias_age_seconds")

    # 2. Drop redundant trojans
    if not keep_originals:
        for c in DROP_FEATURES:
            if c in df.columns:
                df = df.drop(c)

    return df


def cleaned_feature_list(features: list[str], *, include_bucket: bool = False) -> list[str]:
    """Translate a raw feature list to its post-cleanup equivalent.

    Maintains order; substitutes transformed names; drops removed names.
    """
    out: list[str] = []
    for f in features:
        if f in DROP_FEATURES:
            continue
        out.append(TRANSFORM_FEATURES.get(f, f))
    if include_bucket and "live_bias_age_bucket" not in out:
        # Insert after the log_clipped feature, or at end if not present
        try:
            idx = out.index("live_bias_age_log_clipped")
            out.insert(idx + 1, "live_bias_age_bucket")
        except ValueError:
            out.append("live_bias_age_bucket")
    return out


def trojan_indices(features: list[str]) -> list[int]:
    """Return indices of trojan-HARD features in a feature ordering.
    Useful for the zero-out re-evaluation experiment.
    """
    return [i for i, f in enumerate(features) if f in TROJAN_HARD]


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test: build a tiny df and run cleanup
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    import numpy as np
    rng = np.random.default_rng(0)
    n = 100
    df = pl.DataFrame({
        "anchor_ts_ns": np.arange(n) * 1_000_000_000,
        "live_btc_usd": rng.normal(78000, 1000, n),
        "price_to_beat": rng.normal(78000, 1000, n),
        "live_applied_bias_age_seconds": rng.exponential(50_000, n),
        "spread_mean_5s": rng.uniform(0.001, 0.02, n),
        "imbalance_last": rng.uniform(-1, 1, n),
    })
    print("BEFORE columns:", df.columns)
    cleaned = clean_features(df)
    print("AFTER columns:", cleaned.columns)
    print("\nSample cleaned values:")
    print(cleaned.select([
        "live_bias_age_log_clipped", "live_bias_age_bucket",
        "spread_mean_5s", "imbalance_last"
    ]).head(5))

    raw_feats = ["anchor_source", "spread_mean_5s", "price_to_beat", "live_btc_usd",
                 "live_applied_bias_age_seconds", "imbalance_last"]
    print("\nRaw feature list:    ", raw_feats)
    print("Cleaned feature list:", cleaned_feature_list(raw_feats))
    print("Trojan indices in raw:", trojan_indices(raw_feats))
