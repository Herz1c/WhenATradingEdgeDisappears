import polars as pl

from model_factory.evaluation.walkforward import AnchoredWalkForwardEvaluator


def _df() -> pl.DataFrame:
    day_ns = 86_400_000_000_000
    return pl.DataFrame({"recv_ts_ns": [i * day_ns for i in range(6)], "x": list(range(6)), "y": [0, 1, 0, 1, 0, 1]})


def test_no_future_data_in_any_fold():
    folds = AnchoredWalkForwardEvaluator("recv_ts_ns", purge_days=0, min_train_days=3).split(_df())
    for train, val in folds:
        assert train["recv_ts_ns"].max() < val["recv_ts_ns"].min()


def test_purge_gap_correctly_applied():
    folds = AnchoredWalkForwardEvaluator("recv_ts_ns", purge_days=1, min_train_days=3).split(_df())
    for train, val in folds:
        gap_days = (val["recv_ts_ns"].min() - train["recv_ts_ns"].max()) / 86_400_000_000_000
        assert gap_days >= 2


def test_fold_count_correct_for_date_range():
    folds = AnchoredWalkForwardEvaluator("recv_ts_ns", purge_days=0, min_train_days=3).split(_df())
    assert len(folds) == 3


def test_anchored_train_starts_same_date():
    folds = AnchoredWalkForwardEvaluator("recv_ts_ns", purge_days=0, min_train_days=3).split(_df())
    assert {fold[0]["recv_ts_ns"].min() for fold in folds} == {0}

