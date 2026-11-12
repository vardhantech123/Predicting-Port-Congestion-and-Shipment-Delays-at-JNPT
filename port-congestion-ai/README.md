# Predicting Port Congestion & Shipment Delays at JNPT

Aiqon AI Internship take-home. Given a planned vessel call, the system outputs
**(1) probability of delay, (2) a confidence score, (3) the key factors behind the prediction**
with suggested operational actions. Full write-up: [`docs/Technical_Report.pdf`](docs/Technical_Report.pdf).

## What "delay / congestion" means (ground truth)
The data is real JNPT berth-call data (464 calls, 4 terminals, Sep-Dec 2025). Arrival time is the
*berthing* time, so there is no anchorage-queue measurement. Turnaround (TAT = arrival -> sailing) is
strongly workload-driven, so two targets are modelled:

| Target | Definition |
|---|---|
| **Long-stay risk** (primary) | TAT > 75th percentile of the training period (~24 h): overruns the normal berth window |
| **Excess-delay risk** (diagnostic) | TAT > workload-expected TAT + 4 h (expected TAT = robust regression on TEU & length) |

## Headline results (time-ordered hold-out, last 30% of period)
| | AUC | Notes |
|---|---|---|
| Long-stay | **0.89** (95% CI 0.82-0.94) | flags 39% of calls, catches 82% of long stays, 58% precision |
| Excess-delay | 0.58 (CI 0.47-0.68) | not reliably predictable from available fields - reported as low-skill |

Honest takeaways (see report): planned cargo volume explains almost all long-stay predictability;
berth-occupancy / arrival-density features add little in this dataset; weather, queue and crane data are absent.

## Quick start
```bash
pip install -r requirements.txt
python -m src.train                 # clean, label, featurise, evaluate, plot, save model (~40 s)
python -m pytest -q tests           # 4 tests incl. a no-future-leakage check
streamlit run app.py                # interactive demo
python -m src.predict --csv data/example_calls.csv --out outputs/example_scored.csv   # batch scoring
python scripts/make_report.py       # rebuild the PDF report from outputs/metrics.json
```
Python snippet:
```python
from src.predict import load_bundle, score_call
b = load_bundle()
r = score_call(b, "2025-12-10 14:00", term="BMCT", size="Large", loa=300, teu_cap=8000,
               planned_teu=4500, import_teu=2200, export_teu=2300)
print(r["long_stay"]["summary"])
```

## Repo layout
```
data/Port_data.xlsx        original dataset            data/example_calls.csv  batch-scoring example
src/config.py              label definitions, thresholds, feature lists
src/data.py                loading + cleaning (every removal counted)
src/features.py            workload baseline, labels, leak-free port-state features
src/model.py               logistic model, bootstrap confidence, exact contributions
src/explain.py             plain-language drivers + suggested actions
src/train.py               evaluation (time split, expanding-window CV, ablations) + final fit
src/predict.py             scoring API + CLI          src/plots.py   report figures
app.py                     Streamlit demo             scripts/make_report.py   PDF builder
outputs/                   metrics.json, test predictions, figures
docs/                      Technical_Report.pdf, DEMO_SCRIPT.md
```

## Method in brief
* **Features (known at berth arrival only):** planned TEU, vessel length/capacity/size, import share; vessels
  at berth and arrivals in last 24 h (port + terminal); mean excess-TAT of calls that sailed in the last 72 h;
  weekend/night; terminal.
* **Model:** L2 logistic regression. It matched or beat gradient boosting and random forest under time-ordered CV and
  gives exact per-feature log-odds contributions.
* **Confidence:** 200 bootstrap re-fits; share agreeing with the final HIGH/not-HIGH decision, plus an 80% probability range.
* **Evaluation:** chronological split (no random shuffling), pooled expanding-window CV, bootstrap CIs, calibration,
  ablations, permutation importance.

## Assumptions & limitations
Planned TEU is known at arrival (stow plan); single port and ~10 weeks, so seasonality and cross-port variability are
not learnable here; occupancy counts only calls present in the table; thresholds (75th pct, +4 h) are configurable
choices; explanations are associations, not causal claims. 2 corrupt rows (operation end before start), 4 zero-cargo
and 5 out-of-range turnarounds are removed (counts in `outputs/metrics.json`).
