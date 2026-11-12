"""Interactive decision-support demo.   Run:  streamlit run app.py"""
import pandas as pd

try:
    import streamlit as st  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: install Streamlit with `pip install streamlit`") from exc

from src import config as C
from src.predict import load_bundle, score_call

st.set_page_config(page_title="Port Delay Risk - JNPT", layout="wide")


@st.cache_resource
def bundle():
    return load_bundle()


try:
    b = bundle()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` first."); st.stop()

st.title("Port congestion & delay risk - JNPT")
st.caption("Scores a planned vessel call using only information known at berth arrival.")

with st.sidebar:
    st.header("Planned vessel call")
    d = st.date_input("Arrival date", pd.Timestamp("2025-12-10"))
    t = st.time_input("Arrival time", pd.Timestamp("2025-12-10 14:00").time())
    term = st.selectbox("Terminal", b["terminals"])
    size = st.selectbox("Vessel size class", b["sizes"], index=len(b["sizes"]) - 1)
    loa = st.slider("Length overall (m)", 100, 400, 260)
    cap = st.slider("Vessel capacity (TEU)", 500, 20000, 5000, 100)
    teu = st.slider("Planned TEU to exchange", 100, 9000, 1800, 50)
    imp = st.slider("of which import (TEU)", 0, teu, teu // 2, 50)
    st.header("Port state at arrival")
    st.caption("Auto-derived from the historical call table. Tick to override (what-if).")
    manual = st.checkbox("Override port state")

arrival = pd.Timestamp.combine(d, t)
auto = score_call(b, arrival, term, size, loa, cap, teu, imp, teu - imp)["features"]
override = None
if manual:
    with st.sidebar:
        override = {
            "port_occ": st.slider("Vessels at berth (port)", 0, 12, int(auto["port_occ"])),
            "term_occ": st.slider("Vessels at berth (terminal)", 0, 6, int(auto["term_occ"])),
            "port_arr24": st.slider("Port arrivals, last 24h", 0, 20, int(auto["port_arr24"])),
            "term_arr24": st.slider("Terminal arrivals, last 24h", 0, 10, int(auto["term_arr24"])),
        }
res = score_call(b, arrival, term, size, loa, cap, teu, imp, teu - imp, state_override=override)

c1, c2, c3 = st.columns(3)
ls = res["long_stay"]["pred"]
c1.metric("Long-stay risk", f"{ls['prob_delay']:.0%}", ls["risk_tier"])
c2.metric("Model confidence", f"{ls['confidence']:.0%}",
          f"80% range {ls['p10']:.0%}-{ls['p90']:.0%}", delta_color="off")
c3.metric("Expected turnaround", f"{res['expected_tat_h']:.1f} h",
          f"long-stay if > {res['long_stay_threshold_h']:.0f} h", delta_color="off")

st.subheader("Why")
st.write(res["long_stay"]["summary"])
cols = st.columns(2)
cols[0].markdown("**Raising risk**")
for u in res["long_stay"]["drivers_up"]:
    cols[0].markdown(f"- **{u['label']}** ({u['value']}) - _{u['action']}_")
cols[1].markdown("**Lowering risk**")
for u in res["long_stay"]["drivers_down"]:
    cols[1].markdown(f"- {u['label']} ({u['value']})")

with st.expander("Excess-delay (workload-adjusted congestion) - low-skill model"):
    st.write(res["excess"]["summary"])
with st.expander("Port state used"):
    st.json({k: v for k, v in res["features"].items()})
