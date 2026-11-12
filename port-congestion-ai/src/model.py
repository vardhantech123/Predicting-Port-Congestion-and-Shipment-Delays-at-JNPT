"""Risk model: regularised logistic regression + bootstrap uncertainty."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (ALL_FEATURES, CAT_FEATURES, NUM_FEATURES, N_BOOTSTRAP,
                     RANDOM_STATE, TARGETS, MAX_FLAG_RATE)


def make_pipeline_(estimator=None):
    est = estimator if estimator is not None else LogisticRegression(C=0.3, max_iter=1000)
    ct = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    return make_pipeline(ct, est)


def oof_threshold(train, target, fracs=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Pick the probability cut-off maximising F1 on expanding-window,
    out-of-fold predictions inside the training period (no test leakage)."""
    n, preds, ys = len(train), [], []
    for f in fracs:
        a, b = int(n * f), int(n * min(f + 0.1, 1.0))
        if b <= a:
            continue
        m = make_pipeline_().fit(train.iloc[:a][ALL_FEATURES], train.iloc[:a][target])
        preds.append(m.predict_proba(train.iloc[a:b][ALL_FEATURES])[:, 1])
        ys.append(train.iloc[a:b][target].to_numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    # max-F1 cut-off, constrained so that at most MAX_FLAG_RATE of calls are flagged
    grid = [t for t in np.linspace(0.05, 0.9, 86) if (p >= t).mean() <= MAX_FLAG_RATE]
    return float(grid[int(np.argmax([f1_score(y, p >= t) for t in grid]))])


class CongestionModel:
    """Probability of delay + bootstrap-based confidence + exact explanations."""

    def __init__(self, target="long_stay"):
        assert target in TARGETS
        self.target = target

    def fit(self, train, baseline):
        col = TARGETS[self.target]
        self.baseline = baseline
        self.base_rate = float(train[col].mean())
        self.threshold = oof_threshold(train, col)
        X, y = train[ALL_FEATURES], train[col].to_numpy()
        self.pipe = make_pipeline_().fit(X, y)
        rng = np.random.default_rng(RANDOM_STATE)
        self.boot = []
        while len(self.boot) < N_BOOTSTRAP:
            idx = rng.integers(0, len(train), len(train))
            if y[idx].min() == y[idx].max():
                continue
            self.boot.append(make_pipeline_().fit(X.iloc[idx], y[idx]))
        # reference (training-mean) design row for contribution baselines
        Z = self.pipe[0].transform(X)
        self.ref = np.asarray(Z.mean(axis=0)).ravel()
        self.feature_names = self.pipe[0].get_feature_names_out()
        return self

    # -------------------------------------------------------------- predict
    def predict(self, X):
        """Return DataFrame: prob, p10, p90, flag, confidence, risk_tier."""
        X = X[ALL_FEATURES]
        p = self.pipe.predict_proba(X)[:, 1]
        B = np.column_stack([m.predict_proba(X)[:, 1] for m in self.boot])
        flag = p >= self.threshold
        agree = np.where(flag[:, None], B >= self.threshold, B < self.threshold).mean(1)
        tier = np.where(p >= self.threshold, "HIGH",
                        np.where(p >= self.base_rate, "ELEVATED", "LOW"))
        return pd.DataFrame({
            "prob_delay": p, "p10": np.percentile(B, 10, axis=1),
            "p90": np.percentile(B, 90, axis=1), "flag_delay": flag.astype(int),
            "confidence": agree, "risk_tier": tier}, index=X.index)

    # -------------------------------------------------------------- explain
    def contributions(self, X):
        """Per-feature log-odds contribution vs the average training call.
        Exact for a linear model: coef * (z - z_ref). One-hot columns are summed
        back to their parent feature."""
        Z = np.asarray(self.pipe[0].transform(X[ALL_FEATURES]))
        coef = self.pipe[-1].coef_.ravel()
        C = (Z - self.ref) * coef
        out = {}
        for j, name in enumerate(self.feature_names):
            parent = name.split("__", 1)[1]
            if name.startswith("cat__"):
                parent = parent.rsplit("_", 1)[0] if parent.split("_")[0] not in CAT_FEATURES \
                    else parent.split("_", 1)[0]
            out.setdefault(parent, np.zeros(len(X)))
            out[parent] = out[parent] + C[:, j]
        return pd.DataFrame(out, index=X.index)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
