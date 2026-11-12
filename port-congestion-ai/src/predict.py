"""Scoring API + CLI.

Python:
    from src.predict import load_bundle, score_call
CLI (batch):
    python -m src.predict --csv calls.csv --out scored.csv
Required CSV columns: arrival (datetime), term, size, loa, teu_cap, planned_teu
Optional: import_teu, export_teu
"""
import argparse

import numpy as np
import pandas as pd

from . import config as C
from .explain import explain_row, narrative
from .features import state_features, static_features
import joblib


CAVEATS = {"excess": "[Low-skill model (AUC~0.6): indicative only, do not act on it alone.]"}


def load_bundle(path=C.MODEL_PATH):
    return joblib.load(path)


def score_call(bundle, arrival, term, size, loa, teu_cap, planned_teu,
               import_teu=np.nan, export_teu=np.nan, state_override=None):
    """Score one planned vessel call. `state_override` replaces any of the six
    port-state features (for what-if analysis)."""
    arrival = pd.Timestamp(arrival)
    f = static_features(arrival, planned_teu, import_teu, export_teu, loa, teu_cap, term, size)
    f.update(state_features(arrival, term, bundle["history"]))
    if state_override:
        f.update(state_override)
    X = pd.DataFrame([f])
    exp_tat = float(bundle["baseline"].predict(pd.DataFrame(
        {"teu": [planned_teu], "loa": [loa]}))[0])
    out = {"features": f, "expected_tat_h": exp_tat,
           "long_stay_threshold_h": bundle["baseline"].long_stay_h}
    for name, model in bundle["models"].items():
        pred = model.predict(X).iloc[0]
        ups, downs = explain_row(model.contributions(X).iloc[0], X.iloc[0])
        out[name] = {"pred": pred.to_dict(), "drivers_up": ups, "drivers_down": downs,
                     "summary": narrative(pred, ups, downs, CAVEATS.get(name, ""))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", required=True); ap.add_argument("--out", default="scored.csv")
    a = ap.parse_args()
    b = load_bundle()
    rows = []
    for r in pd.read_csv(a.csv, parse_dates=["arrival"]).itertuples():
        s = score_call(b, r.arrival, r.term, r.size, r.loa, r.teu_cap, r.planned_teu,
                       getattr(r, "import_teu", np.nan), getattr(r, "export_teu", np.nan))
        row = {"arrival": r.arrival, "term": r.term, "expected_tat_h": round(s["expected_tat_h"], 1)}
        for k in C.TARGETS:
            p = s[k]["pred"]
            row.update({f"{k}_prob": round(p["prob_delay"], 3), f"{k}_tier": p["risk_tier"],
                        f"{k}_confidence": round(p["confidence"], 2),
                        f"{k}_explanation": s[k]["summary"]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"scored {len(rows)} calls -> {a.out}")


if __name__ == "__main__":
    main()
