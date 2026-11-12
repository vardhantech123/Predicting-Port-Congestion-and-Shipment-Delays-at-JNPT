import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import clean, load_raw
from src.features import TurnaroundBaseline, add_label, state_features


@pytest.fixture(scope="module")
def df():
    return clean(load_raw())[0]


def test_cleaning_removes_bad_rows(df):
    assert (df.teu > 0).all()
    assert df.tat.between(C.MIN_TAT_H, C.MAX_TAT_H).all()
    assert (df.handling_h > 0).all()


def test_labels_rates_reasonable(df):
    base = TurnaroundBaseline().fit(df)
    d = add_label(df, base)
    assert 0.15 < d.y_long.mean() < 0.35 and 0.05 < d.y_excess.mean() < 0.35


def test_state_features_no_future_leakage():
    t0 = pd.Timestamp("2025-01-01 12:00")
    calls = pd.DataFrame({
        "arr": [t0 - pd.Timedelta(hours=5), t0, t0 + pd.Timedelta(hours=1)],
        "dep": [t0 + pd.Timedelta(hours=3), t0 + pd.Timedelta(hours=9), t0 + pd.Timedelta(hours=20)],
        "term": ["A", "A", "A"], "excess": [99.0, 99.0, 99.0]})
    f = state_features(t0, "A", calls)
    assert f["port_occ"] == 1            # only the earlier, still-berthed call
    assert f["port_arr24"] == 1          # same-time/future arrivals excluded
    assert np.isnan(f["term_hist"])      # nothing had sailed yet -> no hist leak


def test_model_smoke():
    from src.predict import load_bundle, score_call
    b = load_bundle()
    hi = score_call(b, "2025-12-10 14:00", "BMCT", "Large", 300, 8000, 4500, 2200, 2300)
    lo = score_call(b, "2025-12-10 14:00", "NSICT", "Small", 150, 1500, 600, 300, 300)
    assert hi["long_stay"]["pred"]["prob_delay"] > lo["long_stay"]["pred"]["prob_delay"]
    assert 0.5 <= hi["long_stay"]["pred"]["confidence"] <= 1
