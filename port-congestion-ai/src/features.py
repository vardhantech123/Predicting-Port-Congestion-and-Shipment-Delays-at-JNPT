"""Label definition and leak-free feature engineering.

Principle: every feature for a call is computed ONLY from information available at
the instant the vessel arrives at berth (strictly earlier arrivals, departures that
had already happened, and the vessel's own planned particulars).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from .config import (DELAY_THRESHOLD_H, HISTORY_WINDOW_H, ARRIVAL_WINDOW_H,
                     LONG_STAY_QUANTILE)


# --------------------------------------------------------------------------
# 1. Workload-expected turnaround (the baseline that defines "excess")
# --------------------------------------------------------------------------
class TurnaroundBaseline:
    """Robust (Huber) regression of TAT on workload: expected hours for a call."""

    def _X(self, df):
        lt = np.log1p(df["teu"].to_numpy(float))
        loa = df["loa"].fillna(self.loa_med).to_numpy(float) / 100
        return np.c_[lt, lt ** 2, loa]

    def fit(self, df):
        self.loa_med = float(df["loa"].median())
        self.model = HuberRegressor(max_iter=500).fit(self._X(df), df["tat"])
        self.long_stay_h = float(df["tat"].quantile(LONG_STAY_QUANTILE))
        return self

    def predict(self, df):
        return self.model.predict(self._X(df))


def add_label(df, baseline):
    df = df.copy()
    df["exp_tat"] = baseline.predict(df)
    df["excess"] = df["tat"] - df["exp_tat"]
    df["y_excess"] = (df["excess"] > DELAY_THRESHOLD_H).astype(int)
    df["y_long"] = (df["tat"] > baseline.long_stay_h).astype(int)
    return df


# --------------------------------------------------------------------------
# 2. Port-state features at arrival time t
# --------------------------------------------------------------------------
def state_features(t, term, calls):
    """Port/terminal state at time t from the history table `calls`.

    `calls` needs columns: arr, dep, term, excess. Only rows with arr < t are used;
    `dep`/`excess` of a row are used only if it had already sailed by t.
    """
    past = calls[calls["arr"] < t]
    at_berth = past[past["dep"] > t]                      # still alongside at t
    recent_arr = past[past["arr"] >= t - pd.Timedelta(hours=ARRIVAL_WINDOW_H)]
    sailed = past[(past["dep"] <= t) &
                  (past["dep"] >= t - pd.Timedelta(hours=HISTORY_WINDOW_H))]
    sailed_t = sailed[sailed["term"] == term]
    return {
        "port_occ": int(len(at_berth)),
        "term_occ": int((at_berth["term"] == term).sum()),
        "port_arr24": int(len(recent_arr)),
        "term_arr24": int((recent_arr["term"] == term).sum()),
        "port_hist": float(sailed["excess"].mean()) if len(sailed) >= 3 else np.nan,
        "term_hist": float(sailed_t["excess"].mean()) if len(sailed_t) >= 2 else np.nan,
    }


def static_features(arr, teu, imp, exp, loa, teu_cap, term, size):
    imp_share = np.nan
    if pd.notna(imp) and pd.notna(exp) and (imp + exp) > 0:
        imp_share = float(imp) / float(imp + exp)
    hour, dow = arr.hour, arr.dayofweek
    return {
        "log_teu": float(np.log1p(teu)), "loa": loa, "teu_cap": teu_cap,
        "weekend": int(dow >= 5), "night": int(hour >= 22 or hour < 6),
        "imp_share": imp_share, "term": term, "size": size,
    }


def build_feature_table(df):
    """Feature matrix for every call in `df` (needs a label/`excess` column)."""
    rows = []
    for r in df.itertuples(index=False):
        f = static_features(r.arr, r.teu, r.imp, r.exp, r.loa, r.teu_cap, r.term, r.size)
        f.update(state_features(r.arr, r.term, df))
        rows.append(f)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows).drop(
        columns=["term", "size", "loa", "teu_cap"])], axis=1)
