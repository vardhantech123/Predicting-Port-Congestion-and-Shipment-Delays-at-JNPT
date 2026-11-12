"""Turn model contributions into plain-language, operational explanations."""
import numpy as np
import pandas as pd

from .config import FEATURE_LABELS

# What an operator can do about each driver (kept deliberately conservative:
# these are decision-support suggestions, not causal claims).
ACTIONS = {
    "term_occ": "Stagger berth windows / hold a berth buffer at this terminal.",
    "port_occ": "Consider re-allocating the call to a less busy terminal.",
    "term_arr24": "Smooth arrivals at this terminal: re-time ETA or pre-book the berth.",
    "port_arr24": "Port-wide arrival bunching: coordinate pilotage and tug slots.",
    "term_hist": "Terminal already running slow: pre-alert yard/crane planning and the shipper.",
    "port_hist": "Port-wide slowdown: add buffer to downstream (rail/road) bookings.",
    "log_teu": "Large exchange: pre-stage yard equipment and extra gangs.",
    "loa": "Long vessel: confirm berth length and tug requirement in advance.",
    "teu_cap": "Large-capacity vessel: plan extra crane intensity.",
    "weekend": "Weekend arrival: confirm labour and gate roster cover.",
    "night": "Night arrival: confirm night-shift crane and yard crew.",
    "imp_share": "Cargo mix skewed: pre-plan yard space for the dominant flow.",
    "term": "Terminal-specific tendency: review this terminal's recent productivity.",
    "size": "Vessel class effect: check berth/crane fit.",
}


def _value_text(key, row):
    v = row.get(key)
    if key in ("term", "size"):
        return str(v)
    if key == "log_teu":
        return f"{np.expm1(v):,.0f} TEU"
    if key in ("weekend", "night"):
        return "yes" if v else "no"
    if key == "imp_share":
        return "n/a" if pd.isna(v) else f"{v:.0%} import"
    if key in ("term_hist", "port_hist"):
        return "n/a" if pd.isna(v) else f"{v:+.1f} h vs expected"
    if key == "loa":
        return f"{v:.0f} m"
    if key == "teu_cap":
        return "n/a" if pd.isna(v) else f"{v:,.0f} TEU"
    return f"{v:g}"


def explain_row(contrib_row: pd.Series, feat_row: pd.Series, top=3):
    """Return list of dicts for the strongest risk-raising / risk-lowering drivers."""
    s = contrib_row.sort_values(ascending=False)
    up = [(k, v) for k, v in s.items() if v > 0.15][:top]
    down = [(k, v) for k, v in s.items() if v < -0.15][::-1][:2]
    mk = lambda k, v: {"feature": k, "label": FEATURE_LABELS.get(k, k),
                       "value": _value_text(k, feat_row), "log_odds": float(v),
                       "action": ACTIONS.get(k, "")}
    return [mk(k, v) for k, v in up], [mk(k, v) for k, v in down]


def narrative(pred_row, ups, downs, caveat=""):
    p, lo, hi = pred_row["prob_delay"], pred_row["p10"], pred_row["p90"]
    head = (f"{pred_row['risk_tier']} delay risk: {p:.0%} probability "
            f"(80% range {lo:.0%}-{hi:.0%}), model confidence {pred_row['confidence']:.0%}.")
    if ups:
        head += " Main risk drivers: " + "; ".join(
            f"{u['label'].lower()} ({u['value']})" for u in ups) + "."
    if downs:
        head += " Mitigating: " + "; ".join(
            f"{d['label'].lower()} ({d['value']})" for d in downs) + "."
    return head + (" " + caveat if caveat else "")
