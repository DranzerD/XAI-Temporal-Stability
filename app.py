"""XDrift results explorer.   Run:  streamlit run app.py

Reads the small result CSVs committed to the repo (results_v2/, results_v3/); no data download needed.
"""
import os

import altair as alt
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="XDrift results", layout="wide")


@st.cache_data
def load(path):
    p = os.path.join(HERE, path)
    return pd.read_csv(p) if os.path.exists(p) else None


v2 = load("results_v2/windows_v2.csv")
v3 = load("results_v3/panel_v3.csv")

st.title("XDrift — does explanation drift tell you when a credit model is decaying?")
st.caption("LendingClub 2007–2018 · XGBoost + TreeSHAP · pre-declared analysis plans · negative and mixed results are reported as such")

tab1, tab2, tab3 = st.tabs(["Explanation drift vs noise floor (v2)", "Frozen-model monitoring (v3)", "Read this first"])

with tab1:
    if v2 is None:
        st.warning("results_v2/windows_v2.csv not found.")
    else:
        v2["train_end"] = pd.to_datetime(v2["train_end"])
        lo, hi = v2["train_end"].min().date(), v2["train_end"].max().date()
        a, b = st.slider("Window end date range", lo, hi, (lo, hi))
        d = v2[(v2["train_end"].dt.date >= a) & (v2["train_end"].dt.date <= b)].dropna(subset=["xsi_abs_consec"])  # first window has no predecessor
        band = alt.Chart(d).mark_area(opacity=0.3, color="#3182bd").encode(
            x=alt.X("train_end:T", title="Window end"), y=alt.Y("xsi_abs_self_min:Q", title="XSI (|SHAP| profile)", scale=alt.Scale(zero=False)),
            y2="xsi_abs_self_max:Q")
        line = alt.Chart(d).mark_line(point=True, color="#222").encode(x="train_end:T", y="xsi_abs_consec:Q")
        st.altair_chart((band + line).properties(height=300, title="Consecutive-window XSI (black) vs. retraining noise floor (blue band)"),
                        width="stretch")
        vix = alt.Chart(d).mark_line(color="#e6850e").encode(x="train_end:T", y=alt.Y("vix_train_mean:Q", title="Mean VIX (training window)"))
        st.altair_chart(vix.properties(height=160), width="stretch")
        below = (d["xsi_abs_consec"] < d["xsi_abs_self_min"]).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Windows below the noise band", f"{below:.0%}")
        c2.metric("Mean XSI (consecutive)", f"{d['xsi_abs_consec'].mean():.3f}")
        c3.metric("Mean noise floor", f"{d['xsi_abs_self'].mean():.3f}")

with tab2:
    if v3 is None:
        st.info("results_v3/panel_v3.csv not found — run `python xdrift_v3.py all` to generate it.")
    else:
        v3["origin"] = pd.to_datetime(v3["origin"])
        origins = sorted(v3["origin"].dt.strftime("%Y-%m").unique())
        sel = st.selectbox("Frozen model (training window ends at…)", origins, index=min(6, len(origins) - 1))
        d = v3[v3["origin"].dt.strftime("%Y-%m") == sel].sort_values("k")
        monitors = {"AUC decay vs. reference": "decay", "Explanation excess drift": "xsi_abs_excess", "Score PSI": "score_psi",
                    "Feature PSI (non-macro)": "feat_psi_nomacro", "Score KS": "score_ks"}
        pick = st.multiselect("Monitors", list(monitors), default=["AUC decay vs. reference", "Explanation excess drift", "Score PSI"])
        long = pd.concat([d[["k", monitors[m]]].rename(columns={monitors[m]: "value"}).assign(monitor=m) for m in pick])
        if len(long):
            long["z"] = long.groupby("monitor")["value"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
            st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("k:O", title="Quarters since go-live"), y=alt.Y("z:Q", title="Standardised within this model"),
                color="monitor:N").properties(height=320), width="stretch")
        st.dataframe(d[["k", "auc", "decay", "xsi_abs_excess", "score_psi", "feat_psi_nomacro", "score_ks"]].round(4), hide_index=True)

with tab3:
    st.markdown("""
**What this is.** A study of whether SHAP-based *explanation stability* (XSI) is useful for monitoring a deployed credit model.

**What it found (see `RESULTS.md` and `RESULTS_V3.md` in the repo):**
- Explanations drift beyond what retraining noise alone explains (v2).
- The pre-declared hypotheses about market stress and early warning were **not** confirmed at conventional significance (v2).
- v3 compares explanation drift with standard monitors (score PSI, feature PSI) on frozen models; see its results file for the verdict.

**Caveats.** One dataset (LendingClub), few independent models, right-censored late-period labels, and many uncorrected secondary p-values.
Treat anything not in the primary Holm-corrected family as exploratory.
""")
