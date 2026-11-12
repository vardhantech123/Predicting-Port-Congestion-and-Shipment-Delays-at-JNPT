"""End-to-end training + evaluation.   Run:  python -m src.train"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

from . import config as C
from .data import clean, load_raw
from .features import TurnaroundBaseline, add_label, build_feature_table
from .model import CongestionModel, make_pipeline_
from . import plots

warnings.filterwarnings("ignore")
HIST_COLS = ["arr", "dep", "term", "excess"]


def prepare(df, fit_rows):
    """Fit the workload baseline on `fit_rows` only, then label + featurise all."""
    base = TurnaroundBaseline().fit(df.iloc[:fit_rows])
    return base, build_feature_table(add_label(df, base))


_CACHE = {}


def prepare_cached(df, fit_rows):
    key = (len(df), fit_rows)
    if key not in _CACHE:
        _CACHE[key] = prepare(df, fit_rows)
    return _CACHE[key]


def boot_ci(y, p, fn, n=1000, seed=0):
    rng, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(fn(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def calibration_bins(y, p, k=4):
    q = pd.qcut(p, k, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin", observed=True)
    return [{"mean_pred": float(a.p.mean()), "observed": float(a.y.mean()), "n": int(len(a))}
            for _, a in g]


def expanding_cv(df, target, estimator_fn, feats_num=None, cat=None,
                 fracs=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Pooled out-of-time AUC / PR-AUC; baseline + labels re-fit inside each fold."""
    import src.model as M
    old = (M.NUM_FEATURES, M.CAT_FEATURES)
    M.NUM_FEATURES = feats_num if feats_num is not None else C.NUM_FEATURES
    M.CAT_FEATURES = cat if cat is not None else C.CAT_FEATURES
    P, Y, n = [], [], len(df)
    try:
        for f in fracs:
            a, b = int(n * f), int(n * min(f + 0.1, 1))
            _, ft = prepare_cached(df, a)
            cols = M.NUM_FEATURES + M.CAT_FEATURES
            m = M.make_pipeline_(estimator_fn()).fit(ft.iloc[:a][cols], ft.iloc[:a][target])
            P.append(m.predict_proba(ft.iloc[a:b][cols])[:, 1])
            Y.append(ft.iloc[a:b][target].to_numpy())
    finally:
        M.NUM_FEATURES, M.CAT_FEATURES = old
    P, Y = np.concatenate(P), np.concatenate(Y)
    return {"auc": float(roc_auc_score(Y, P)), "pr_auc": float(average_precision_score(Y, P)),
            "base_rate": float(Y.mean()), "n": int(len(Y))}


def permutation_importance(model, te, target, repeats=30, seed=0):
    rng = np.random.default_rng(seed)
    y = te[target].to_numpy()
    base = roc_auc_score(y, model.pipe.predict_proba(te[C.ALL_FEATURES])[:, 1])
    out = {}
    for f in C.ALL_FEATURES:
        d = []
        for _ in range(repeats):
            X = te[C.ALL_FEATURES].copy()
            X[f] = rng.permutation(X[f].to_numpy())
            d.append(base - roc_auc_score(y, model.pipe.predict_proba(X)[:, 1]))
        out[f] = float(np.mean(d))
    return out


def evaluate_target(df, ft, base, name, n_train):
    col = C.TARGETS[name]
    tr, te = ft.iloc[:n_train], ft.iloc[n_train:]
    model = CongestionModel(name).fit(tr, base)
    pred = model.predict(te)
    y, p = te[col].to_numpy(), pred["prob_delay"].to_numpy()
    flag = pred["flag_delay"].to_numpy().astype(bool)
    tp = int((flag & (y == 1)).sum())
    res = {
        "target": name, "n_train": int(len(tr)), "n_test": int(len(te)),
        "train_rate": float(tr[col].mean()), "test_rate": float(y.mean()),
        "threshold": model.threshold,
        "auc": float(roc_auc_score(y, p)), "auc_ci95": boot_ci(y, p, roc_auc_score),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_prior": float(brier_score_loss(y, np.full(len(y), tr[col].mean()))),
        "flagged_share": float(flag.mean()),
        "precision": float(tp / max(flag.sum(), 1)), "recall": float(tp / max(y.sum(), 1)),
        "lift": float((tp / max(flag.sum(), 1)) / max(y.mean(), 1e-9)),
        "calibration": calibration_bins(y, p),
    }
    # does the confidence score mean anything?  accuracy of the flag decision by confidence
    acc = (flag.astype(int) == y)
    hi = pred["confidence"].to_numpy() >= 0.9
    res["confidence_check"] = {
        "high_conf_share": float(hi.mean()),
        "accuracy_high": float(acc[hi].mean()) if hi.any() else None,
        "accuracy_low": float(acc[~hi].mean()) if (~hi).any() else None}
    res["perm_importance"] = permutation_importance(model, te, col)

    # time-ordered CV for the model and its competitors / ablations
    lr = lambda: LogisticRegression(C=0.3, max_iter=1000)
    state = ["port_occ", "term_occ", "port_arr24", "term_arr24", "term_hist", "port_hist"]
    res["cv"] = {
        "logistic (chosen)": expanding_cv(df, col, lr),
        "gradient boosting": expanding_cv(df, col, lambda: GradientBoostingClassifier(
            n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.7, random_state=0)),
        "random forest": expanding_cv(df, col, lambda: RandomForestClassifier(
            300, max_depth=5, min_samples_leaf=8, random_state=0)),
        "terminal only": expanding_cv(df, col, lr, [], ["term"]),
        "planned volume only": expanding_cv(df, col, lr, ["log_teu"], []),
        "port-state features only": expanding_cv(df, col, lr, state, ["term"]),
        "without planned volume": expanding_cv(
            df, col, lr, [f for f in C.NUM_FEATURES if f != "log_teu"], C.CAT_FEATURES),
    }
    return model, pred.assign(y_true=y, arr=te["arr"].values, term=te["term"].values,
                              tat=te["tat"].values, excess=te["excess"].values), res


def main():
    C.OUT_DIR.mkdir(exist_ok=True); C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_PATH.parent.mkdir(exist_ok=True)
    df, quality = clean(load_raw())
    n_train = int(len(df) * C.TRAIN_FRAC)
    base, ft = prepare(df, n_train)
    split_date = str(ft.iloc[n_train]["arr"])
    metrics = {"data_quality": quality, "split": {"train_until": split_date, "n_train": n_train},
               "long_stay_threshold_h": base.long_stay_h,
               "excess_threshold_h": C.DELAY_THRESHOLD_H}
    preds, models = {}, {}
    for name in C.TARGETS:
        models[name], preds[name], metrics[name] = evaluate_target(df, ft, base, name, n_train)
        print(f"[{name}] AUC {metrics[name]['auc']:.3f} "
              f"CI {np.round(metrics[name]['auc_ci95'], 3)}  PR-AUC {metrics[name]['pr_auc']:.3f} "
              f"precision {metrics[name]['precision']:.2f} recall {metrics[name]['recall']:.2f}")
    # descriptive stats for the report
    metrics["eda"] = {
        "tat_mean": float(df.tat.mean()), "tat_p75": float(df.tat.quantile(.75)),
        "corr_tat_teu": float(df[["tat", "teu"]].corr().iloc[0, 1]),
        "excess_by_terminal": ft.groupby("term")["excess"].mean().round(2).to_dict(),
        "long_rate_by_terminal": ft.groupby("term")["y_long"].mean().round(3).to_dict()}
    json.dump(metrics, open(C.OUT_DIR / "metrics.json", "w"), indent=2)
    for name, p in preds.items():
        p.to_csv(C.OUT_DIR / f"test_predictions_{name}.csv", index=False)
    ft.to_csv(C.OUT_DIR / "features_labelled.csv", index=False)
    plots.make_all(df, ft, base, models, preds, metrics)

    # ---- deployment fit on ALL history (baseline, labels, features refit) ----
    base_all, ft_all = prepare(df, len(df))
    bundle = {"baseline": base_all,
              "models": {k: CongestionModel(k).fit(ft_all, base_all) for k in C.TARGETS},
              "history": ft_all[HIST_COLS].copy(),
              "terminals": sorted(df.term.unique()), "sizes": sorted(df["size"].unique()),
              "stats": {"loa_med": float(df.loa.median()), "teu_cap_med": float(df.teu_cap.median())}}
    import joblib
    joblib.dump(bundle, C.MODEL_PATH)
    print("saved", C.MODEL_PATH)


if __name__ == "__main__":
    main()
