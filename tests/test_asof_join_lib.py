import polars as pl
import pytest

from canonical.asof_join_lib import (
    asof_join_left_causal,
    asof_join_live_reference,
    verify_causal_integrity,
)

pytestmark = pytest.mark.filterwarnings("ignore:Sortedness of columns cannot be checked:UserWarning")


def test_forward_join_raises_value_error():
    with pytest.raises(ValueError):
        asof_join_left_causal(
            pl.DataFrame({"recv_ts_ns": [1]}),
            pl.DataFrame({"recv_ts_ns": [1]}),
            "recv_ts_ns",
            "recv_ts_ns",
            strategy="forward",
        )


def test_causal_integrity_no_future_rows():
    left = pl.DataFrame({"recv_ts_ns": [10, 20, 30]})
    right = pl.DataFrame({"recv_ts_ns": [5, 20], "right_recv_ts_ns": [5, 20], "value": [1, 2]})
    joined = asof_join_left_causal(left, right, "recv_ts_ns", "recv_ts_ns")
    assert joined["value"].to_list() == [1, 2, 2]
    report = verify_causal_integrity(joined, "recv_ts_ns", "right_recv_ts_ns")
    assert report["n_violations"] == 0


def test_by_group_join_correct():
    left = pl.DataFrame({"market_slug": ["a", "b"], "recv_ts_ns": [10, 10]})
    right = pl.DataFrame(
        {"market_slug": ["a", "b"], "recv_ts_ns": [9, 9], "right_recv_ts_ns": [9, 9], "value": [1, 2]}
    )
    joined = asof_join_left_causal(left, right, "recv_ts_ns", "recv_ts_ns", by=["market_slug"])
    assert joined.sort("market_slug")["value"].to_list() == [1, 2]


def test_tolerance_enforced_and_missing_rows_null():
    left = pl.DataFrame({"recv_ts_ns": [10, 100]})
    right = pl.DataFrame({"recv_ts_ns": [1], "value": [42]})
    joined = asof_join_left_causal(left, right, "recv_ts_ns", "recv_ts_ns", tolerance_ns=5)
    assert joined["value"].to_list() == [None, None]


def test_empty_right_produces_null_not_raise():
    left = pl.DataFrame({"recv_ts_ns": [10]})
    right = pl.DataFrame(schema={"recv_ts_ns": pl.Int64, "value": pl.Int64})
    joined = asof_join_left_causal(left, right, "recv_ts_ns", "recv_ts_ns")
    assert joined["value"].to_list() == [None]


def test_verify_causal_integrity_detects_violations():
    frame = pl.DataFrame({"recv_ts_ns": [10], "right_recv_ts_ns": [11]})
    report = verify_causal_integrity(frame, "recv_ts_ns", "right_recv_ts_ns")
    assert report["n_violations"] == 1


def test_live_reference_join_uses_ts_seconds_grid():
    left = pl.DataFrame({"recv_ts_ns": [2_500_000_000]})
    live = pl.DataFrame({"ts_seconds": [1, 2], "btc_usd_price": [100.0, 101.0]})
    joined = asof_join_live_reference(left, live)
    assert joined["btc_usd_price"].to_list() == [101.0]
