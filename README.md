# Project Guide: Predicting Port Congestion and Shipment Delays at JNPT

A plain-language explanation of the concept, the data, the method, the results and how to use the system.
When a container ship reaches a port, it occupies a berth while cranes unload and load containers. If the call takes
longer than planned, the next ship waits, trucks and trains queue, and delivery dates slip. That chain reaction is what
people call **port congestion**.

This project builds a small decision-support system. For a **planned vessel call** it answers three questions:

1. **How likely is it to be delayed?** A probability.
2. **How sure is the system?** A confidence score and a probability range.
3. **Why?** The factors pushing the risk up or down, with a suggested action for each.

The goal is not only to predict but to make predictions **interpretable and actionable**, so a port planner can act
before the delay happens.

---

## 2. The data

* **Source:** `data/Port_data.xlsx`, real berth-call records from JNPT (Nhava Sheva, India).
* **Size:** 464 vessel calls, 28 Sep to 17 Dec 2025, four terminals (BMCT, NSICT, NSFT, NSIGT).
* **Per call:** vessel name and flag, TEU capacity, size class, length and beam, berth arrival time, sailing time,
  operation start and end times, import / export / total TEU handled.
* **TEU** = twenty-foot equivalent unit, the standard container measure. A 4,000 TEU exchange means roughly 4,000
  twenty-foot containers moved on and off the ship.

### Two important properties of the data

1. **"Arrival" means berthing time.** There is no record of how long a ship waited at anchorage, so waiting time cannot
   be measured directly.
2. **Turnaround time (TAT) depends mainly on workload.** TAT is the time from berth arrival to sailing. It correlates
   about 0.59 with TEU handled. A ship moving 4,000 TEU naturally stays longer than one moving 500 TEU.

These two facts shaped how delay is defined below.

### Cleaning (every removal is counted in `outputs/metrics.json`)

| Issue | Rows |
|---|---|
| Operation end earlier than operation start (negative handling hours, corrupt timestamps) | 2 |
| Zero cargo handled | 4 |
| Implausible turnaround (< 6 h or > 45 h) | 5 |
| **Rows kept** | **453** of 464 |

Other handling: turnaround is recomputed from timestamps instead of trusting the raw column; missing vessel capacity
(64 rows) and length (5 rows) are filled with the median, using training data only.

---

## 3. What counts as "delay"? (the ground truth)

Real data has no ready-made "delayed" label, so one must be defined. A single "TAT is long" label would mostly rediscover
"big exchanges take long", which is not congestion. So two complementary targets are used:

| Target | Definition | What it means for operations |
|---|---|---|
| **Long-stay risk** (primary) | TAT above the 75th percentile of the training period (about 24.2 h) | The call overran the normal berth window and blocks the berth for the next vessel |
| **Excess-delay risk** (diagnostic) | TAT more than 4 h above what the workload predicts | The call was slower than its cargo volume justifies: possible queueing, equipment, labour or yard inefficiency |

**Expected TAT** for the second target comes from a robust (Huber) regression of TAT on log TEU, log TEU squared and
vessel length, fitted on the training period only.

**Worked example.** A call moving 2,000 TEU has an expected TAT of about 21 h. If it actually takes 27 h, its excess is
+6 h, so it is "excess-delayed". It is also a "long stay" because 27 h is above 24.2 h. A call moving 5,000 TEU that takes
26 h is a long stay (it blocks the berth) but not excess-delayed (the workload explains it).

The thresholds (75th percentile and +4 h) are choices, stated openly and configurable in `src/config.py`.

---

## 4. Features: what the model looks at

All features use only information available **at the moment the vessel berths**. Nothing from the future is used, and a
unit test checks this.

| Group | Features | Why it should matter |
|---|---|---|
| Workload / vessel | planned TEU (log), vessel length, TEU capacity, size class, import share | More cargo moves means more crane time; cargo mix affects yard handling |
| Berth pressure | vessels already at berth (port-wide and same terminal); arrivals in the last 24 h (port and terminal) | Arrival density versus berth capacity, the classic congestion signal |
| Historical delay | average excess-TAT of calls that sailed in the last 72 h (terminal and port) | Persistence: slow terminals tend to stay slow |
| Temporal / terminal | weekend, night arrival, terminal ID | Labour rosters and terminal-specific productivity |

**Not available in this dataset, so not modelled:** weather, anchorage queue length, crane and berth counts, scheduled
ETA. The design could accept them directly once available.

---

## 5. The model, confidence score and explanations

### Model
An L2-regularised **logistic regression**, one per target (median imputation, standardisation, one-hot terminals).
With about 450 calls, a simple model is the right size. It matched or beat gradient boosting and random forest under
time-ordered validation, and it allows exact explanations.

### Probability
The model's output is the probability that the call is a long stay (or excess-delayed).

### Risk tiers
* **HIGH**: probability at or above a cut-off chosen on time-ordered out-of-fold training predictions (best F1), capped so
  that at most 35% of calls are flagged (a planner's capacity assumption). Cut-offs: long-stay 0.33, excess-delay 0.24.
* **ELEVATED**: above the base rate but below the cut-off.
* **LOW**: below the base rate.

### Confidence score
The model is re-fitted 200 times on resampled training data (bootstrap). **Confidence** is the share of those re-fits that
agree with the final HIGH / not-HIGH decision (between 50% and 100%). An **80% probability range** is reported alongside.
A low confidence or wide range means: treat as "watch" and re-score when the ETA firms up.

### Explanations
For a linear model, each feature's contribution to the log-odds, compared with an average call, is exactly
`coefficient x (value - average)`. The system lists the strongest risk-raising and risk-lowering drivers in plain text,
with the actual values and a suggested action.

Example: *"HIGH delay risk: 78% probability (80% range 71%-86%), model confidence 100%. Main risk drivers: planned cargo
volume (4,500 TEU); vessel TEU capacity (8,000 TEU)."*

These are associations for decision support, **not causal claims**.

---

## 6. How the system is evaluated

* **Time-ordered split only.** Training uses the first 70% of the period (to 22 Nov 2025, 317 calls); testing uses the last
  30% (136 calls). A random split would leak port state from the future. The expected-TAT baseline, the long-stay
  threshold and the HIGH cut-off are all fitted on training data only.
* **Pooled expanding-window cross-validation** (6 folds, everything re-fitted per fold), because the test set is small.
* **Bootstrap 95% confidence intervals** on the test AUC.
* **Calibration** (do predicted probabilities match observed rates?), **Brier score**, **permutation importance**, and
  **ablations** (what if features are removed?).
* **Metrics explained:** AUC = how well the model ranks delayed calls above normal ones (0.5 is chance, 1.0 is perfect).
  Precision = of the calls flagged, the share truly delayed. Recall = of the truly delayed calls, the share flagged.
  Lift = precision divided by the base rate.

---

## 7. Results

### Held-out test (last 30%)

| Target | AUC (95% CI) | Flagged | Precision / recall | Lift |
|---|---|---|---|---|
| **Long-stay** | **0.89** (0.82-0.94) | 39% | 0.58 / 0.82 | 2.1x |
| Excess-delay | 0.58 (0.47-0.68) | 27% | 0.19 / 0.26 | 1.0x |

### Key findings

1. **Long-stay risk is predictable.** Flagging 39% of calls catches 82% of long stays. The Brier score is 0.124 against
   0.202 for always predicting the base rate. The confidence score is meaningful: the 68% of test calls with confidence of
   90% or more were classified correctly 88% of the time, against 58% for the rest.
2. **Planned cargo volume dominates.** Volume alone reaches CV AUC 0.87, equal to the full model. Port-state features alone
   reach only 0.54. Removing volume drops the model to 0.69. Berth occupancy and arrival-density features add little in
   this data.
3. **Workload-adjusted congestion is not reliably predictable** from these fields. The excess-delay AUC confidence range
   includes 0.5 and the Brier score is no better than the base rate, so the system labels this output
   "low-skill: indicative only". The only consistent signals are terminal and vessel class (NSICT averages about +1.8 h
   versus expectation, NSFT about -1.2 h). This is read as a data limitation (no queue, weather or equipment data), not
   proof that such delays are random.

---

## 8. From prediction to operational decisions

| Signal | Decision it supports |
|---|---|
| Long-stay HIGH with confidence of 90% or more | Reserve a longer berth window; do not schedule a tight follow-on vessel behind it; alert rail, road and yard teams that release may fall beyond 24 h |
| Expected TAT versus the 24 h window | Quote a realistic sailing ETA to shipper and line; plan pilot and tug slots |
| Driver text: terminal busy or recent terminal delay | Stagger arrivals, or steer a flexible call to a less loaded terminal |
| Low confidence or a wide 80% range | Treat as "watch" and re-score when the ETA firms up |

---

## 9. How to run it

```bash
pip install -r requirements.txt
python -m src.train          # clean, label, featurise, evaluate, plot, save model (about 40 s)
python -m pytest -q tests    # 4 tests, including a no-future-leakage check
streamlit run app.py         # interactive demo
python -m src.predict --csv data/example_calls.csv --out outputs/example_scored.csv
python scripts/make_report.py
```

---
