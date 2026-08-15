"""Build sub-second BTC tick features for each microstructure anchor.

For each anchor_ts_ns in the microstructure dataset, compute features from the
sub-millisecond BTC tick stream (binance_usdm + binance + hyperliquid, ~37M
events/day):

  Mid-price returns over multiple horizons (backward-looking):
    btc_tick_return_100ms / 500ms / 1s / 3s / 5s / 10s / 30s

  Realized volatility (max abs return in past window):
    btc_tick_max_abs_return_1s / 5s

  Trade flow (signed volume from BTC trades):
    btc_tick_signed_vol_1s / 5s

  Jump indicators:
    btc_tick_jump_100ms (|btc_return_100ms| > 0.0005 = 5bp)
    btc_tick_jump_500ms (|btc_return_500ms| > 0.001  = 10bp)
    btc_tick_jump_1s    (|btc_return_1s|    > 0.002  = 20bp)

Output: data/derived/btc_tick_features_v1/<date>.parquet
  columns: anchor_ts_ns (Int64) + the 14 features above
"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path

import polars as pl
import numpy as np


BTC_DIR = Path("data/canonical/external_btc_events_v1")
MICRO_DIR = Path("data/datasets/microstructure_sequence_dataset_v1_tabular")
OUT_DIR = Path("data/derived/btc_tick_features_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATES = [f"2026-04-{d:02d}" for d in range(19, 31)] + [f"2026-05-{d:02d}" for d in range(1, 7)]

LOOKBACKS_NS = {
    "100ms": int(0.1   * 1e9),
    "500ms": int(0.5   * 1e9),
    "1s":    int(1.0   * 1e9),
    "3s":    int(3.0   * 1e9),
    "5s":    int(5.0   * 1e9),
    "10s":   int(10.0  * 1e9),
    "30s":   int(30.0  * 1e9),
}


def build_one_day(date_str: str):
    btc_path = BTC_DIR / f"{date_str}.parquet"
    micro_path = MICRO_DIR / f"{date_str}.parquet"
    out_path = OUT_DIR / f"{date_str}.parquet"

    if not btc_path.exists() or not micro_path.exists():
        print(f"  [{date_str}] MISSING source -- skip")
        return

    if out_path.exists():
        print(f"  [{date_str}] already exists -- skip (delete to rebuild)")
        return

    print(f"  [{date_str}] building...")

    # 1) Load BTC ticker (mid_price) stream from primary venue (binance_usdm)
    btc = pl.read_parquet(
        str(btc_path),
        columns=["recv_ts_ns", "venue", "event_type", "mid_price", "trade_price", "trade_size", "trade_side"]
    )
    # Tickers carry mid_price; trades carry trade_price + size
    # Use binance_usdm tickers as the canonical BTC price stream (most liquid)
    btc_mid = (
        btc.filter((pl.col("venue") == "binance_usdm") & (pl.col("event_type") == "ticker"))
           .select(["recv_ts_ns", "mid_price"])
           .filter(pl.col("mid_price").is_not_null() & pl.col("mid_price").is_not_nan())
           .sort("recv_ts_ns")
    )
    print(f"    btc_mid ticker rows: {btc_mid.height:,}")

    # 2) Load anchor timestamps from microstructure dataset (unique per anchor)
    anchors = (
        pl.read_parquet(str(micro_path), columns=["anchor_ts_ns"])
          .unique()
          .sort("anchor_ts_ns")
    )
    print(f"    unique anchors: {anchors.height:,}")

    # 3) Asof-join BTC mid at anchor
    out = anchors.join_asof(
        btc_mid, left_on="anchor_ts_ns", right_on="recv_ts_ns",
        strategy="backward",
    ).rename({"mid_price": "btc_mid_at_anchor"}).drop("recv_ts_ns")

    # 4) For each lookback, asof-join the BTC mid at (anchor - dt)
    for label, dt_ns in LOOKBACKS_NS.items():
        lookback_anchor = anchors.with_columns(
            (pl.col("anchor_ts_ns") - dt_ns).alias("lookback_ts")
        )
        joined = lookback_anchor.join_asof(
            btc_mid, left_on="lookback_ts", right_on="recv_ts_ns",
            strategy="backward",
        ).select(["anchor_ts_ns", pl.col("mid_price").alias(f"btc_mid_lag_{label}")])
        out = out.join(joined, on="anchor_ts_ns", how="left")

    # 5) Compute log returns
    for label in LOOKBACKS_NS:
        out = out.with_columns(
            (pl.col("btc_mid_at_anchor").log() - pl.col(f"btc_mid_lag_{label}").log())
                .alias(f"btc_tick_return_{label}")
        )

    # 6) Drop the intermediate lag columns
    out = out.drop([f"btc_mid_lag_{label}" for label in LOOKBACKS_NS])

    # 7) Absolute returns (jump magnitude)
    for label in ["1s", "5s"]:
        out = out.with_columns(
            pl.col(f"btc_tick_return_{label}").abs().alias(f"btc_tick_abs_return_{label}")
        )

    # 8) Jump indicators (binary)
    out = out.with_columns([
        (pl.col("btc_tick_return_100ms").abs() >= 0.0005).cast(pl.Int8).alias("btc_tick_jump_100ms"),
        (pl.col("btc_tick_return_500ms").abs() >= 0.0010).cast(pl.Int8).alias("btc_tick_jump_500ms"),
        (pl.col("btc_tick_return_1s").abs()    >= 0.0020).cast(pl.Int8).alias("btc_tick_jump_1s"),
    ])

    # 9) Trade flow features (signed trade volume from binance_usdm trades)
    btc_trades = (
        btc.filter((pl.col("venue") == "binance_usdm") & (pl.col("event_type") == "trade"))
           .filter(pl.col("trade_size").is_not_null())
           .with_columns(
                pl.when(pl.col("trade_side") == "buy")
                  .then(pl.col("trade_size"))
                  .when(pl.col("trade_side") == "sell")
                  .then(-pl.col("trade_size"))
                  .otherwise(0.0)
                  .alias("signed_size")
            )
           .select(["recv_ts_ns", "signed_size", "trade_size"])
           .sort("recv_ts_ns")
    )
    print(f"    btc_trades rows: {btc_trades.height:,}")

    # For each anchor, sum signed trades in past 1s and 5s
    # Use rolling window via group_by_dynamic on anchor times.
    # Easier approach: for each anchor, asof-find the prefix-sum at anchor and
    # at anchor - dt, then difference them.
    btc_trades = btc_trades.with_columns([
        pl.col("signed_size").cum_sum().alias("cum_signed"),
        pl.col("trade_size").cum_sum().alias("cum_abs"),
    ])

    for label, dt_ns in [("1s", LOOKBACKS_NS["1s"]), ("5s", LOOKBACKS_NS["5s"])]:
        # cum at anchor
        at_anchor = anchors.join_asof(
            btc_trades.select(["recv_ts_ns", "cum_signed", "cum_abs"]),
            left_on="anchor_ts_ns", right_on="recv_ts_ns", strategy="backward",
        ).select([
            "anchor_ts_ns",
            pl.col("cum_signed").alias(f"_cum_signed_anchor"),
            pl.col("cum_abs").alias(f"_cum_abs_anchor"),
        ])
        # cum at (anchor - dt)
        at_lag = (
            anchors.with_columns((pl.col("anchor_ts_ns") - dt_ns).alias("lookback_ts"))
                   .join_asof(
                        btc_trades.select(["recv_ts_ns", "cum_signed", "cum_abs"]),
                        left_on="lookback_ts", right_on="recv_ts_ns", strategy="backward",
                    )
                   .select([
                        "anchor_ts_ns",
                        pl.col("cum_signed").alias(f"_cum_signed_lag"),
                        pl.col("cum_abs").alias(f"_cum_abs_lag"),
                    ])
        )
        out = (
            out.join(at_anchor, on="anchor_ts_ns", how="left")
               .join(at_lag,    on="anchor_ts_ns", how="left")
               .with_columns([
                   (pl.col("_cum_signed_anchor").fill_null(0) - pl.col("_cum_signed_lag").fill_null(0))
                        .alias(f"btc_tick_signed_vol_{label}"),
                   (pl.col("_cum_abs_anchor").fill_null(0) - pl.col("_cum_abs_lag").fill_null(0))
                        .alias(f"btc_tick_abs_vol_{label}"),
               ])
               .drop(["_cum_signed_anchor", "_cum_abs_anchor", "_cum_signed_lag", "_cum_abs_lag"])
        )

    # 10) Final fill: null returns (very start of day) -> 0
    feature_cols = [c for c in out.columns if c.startswith("btc_tick_")]
    out = out.with_columns([
        pl.col(c).fill_null(0.0).fill_nan(0.0) for c in feature_cols
    ])

    print(f"    final: {out.height:,} rows  x  {out.width} cols ({len(feature_cols)} features)")
    out.write_parquet(str(out_path))
    print(f"    saved: {out_path}")


if __name__ == "__main__":
    print("\n" + "#"*72)
    print("  Building sub-second BTC tick features for each anchor")
    print("#"*72)
    for d in DATES:
        build_one_day(d)
    print("\n=== DONE ===")
