"""Central configuration: paths, label definition and feature lists."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Port_data.xlsx"
MODEL_PATH = ROOT / "models" / "congestion_model.joblib"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# ---- Label definitions ------------------------------------------------------
# Two complementary targets (both defined from berth arrival -> sailing, "TAT"):
#  1. long_stay : TAT above the 75th percentile of the TRAINING period, i.e. the
#                 call overran the normal berth window (blocks the berth).
#  2. excess    : TAT exceeds the turnaround expected for the call's workload by
#                 more than DELAY_THRESHOLD_H hours (workload-adjusted congestion
#                 / inefficiency proxy).
LONG_STAY_QUANTILE = 0.75
DELAY_THRESHOLD_H = 4.0
TARGETS = {"long_stay": "y_long", "excess": "y_excess"}
MAX_FLAG_RATE = 0.35       # operating point: flag at most ~35% of calls as HIGH

# ---- Data cleaning bounds ---------------------------------------------------
MIN_TAT_H, MAX_TAT_H = 6.0, 45.0   # drop physically implausible turnarounds

# ---- Evaluation -------------------------------------------------------------
TRAIN_FRAC = 0.70          # chronological split: first 70% train, last 30% test
N_BOOTSTRAP = 200          # bootstrap models for uncertainty / confidence
HISTORY_WINDOW_H = 72      # look-back for recent-delay features
ARRIVAL_WINDOW_H = 24      # look-back for arrival-density features
RANDOM_STATE = 42

# ---- Features (all computable at the moment the vessel arrives) -------------
NUM_FEATURES = [
    "log_teu", "loa", "teu_cap",             # vessel / workload
    "port_occ", "term_occ",                  # berths occupied at arrival
    "port_arr24", "term_arr24",              # arrival density, last 24h
    "term_hist", "port_hist",                # recent excess-TAT (historical delay)
    "weekend", "night", "imp_share",         # temporal + cargo mix
]
CAT_FEATURES = ["term", "size"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

FEATURE_LABELS = {
    "log_teu": "Planned cargo volume (TEU)",
    "loa": "Vessel length (LOA)",
    "teu_cap": "Vessel TEU capacity",
    "port_occ": "Vessels already at berth (port-wide)",
    "term_occ": "Vessels already at berth (same terminal)",
    "port_arr24": "Port arrivals in last 24h",
    "term_arr24": "Terminal arrivals in last 24h",
    "term_hist": "Recent delay at this terminal (72h)",
    "port_hist": "Recent delay port-wide (72h)",
    "weekend": "Weekend arrival",
    "night": "Night arrival (22:00-06:00)",
    "imp_share": "Import share of cargo",
    "term": "Terminal",
    "size": "Vessel size class",
}
