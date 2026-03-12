import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="4PL Fitting",
    page_icon="◈",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* Title */
.title-block {
    border-left: 4px solid #e94560;
    padding: 8px 0 8px 18px;
    margin-bottom: 28px;
}
.title-block h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    color: #e94560;
    margin: 0;
    letter-spacing: 2px;
}
.title-block p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8892a4;
    margin: 4px 0 0 0;
    letter-spacing: 1px;
}

/* Section headers */
.section-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: #8892a4;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #21262d;
}

/* Param cards */
.param-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 18px;
}
.param-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px 14px;
}
.param-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #8892a4;
    letter-spacing: 1px;
}
.param-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    color: #53d8fb;
    font-weight: 600;
}

/* Result highlight */
.result-box {
    background: #0d2137;
    border: 1px solid #53d8fb;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 14px 0;
    font-family: 'IBM Plex Mono', monospace;
}
.result-box .od-label { color: #8892a4; font-size: 0.75rem; }
.result-box .od-val   { color: #e6edf3; font-size: 1rem; }
.result-box .arrow    { color: #e94560; font-size: 1.1rem; margin: 0 8px; }
.result-box .conc-val { color: #4caf84; font-size: 1.2rem; font-weight: 600; }

/* Status pills */
.pill-success {
    display: inline-block;
    background: #0f2d1f;
    color: #4caf84;
    border: 1px solid #4caf84;
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
}
.pill-warn {
    display: inline-block;
    background: #2d1f00;
    color: #f0a500;
    border: 1px solid #f0a500;
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
}

/* Streamlit widget overrides */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background-color: #0d1117 !important;
    border: 1px solid #21262d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 6px !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: #e94560 !important;
    box-shadow: 0 0 0 1px #e94560 !important;
}
label { color: #8892a4 !important; font-size: 0.78rem !important; letter-spacing: 0.5px; }

/* Buttons */
div[data-testid="stButton"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
    transition: all 0.15s ease !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid #21262d;
    border-radius: 8px;
    overflow: hidden;
}

/* Divider */
hr { border-color: #21262d !important; margin: 20px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Math (unchanged) ───────────────────────────────────────────────────────────
def four_param_logistic(x, A, B, C, D):
    return D + (A - D) / (1 + (x / C)**B)

def inverse_four_param_logistic(OD, A, B, C, D):
    return C * np.abs((A - OD) / (OD - D)) ** (1 / B)

def fit_model(concentration, OD):
    params, _ = opt.curve_fit(four_param_logistic, concentration, OD)
    return params  # A, B, C, D

# ── Plot ───────────────────────────────────────────────────────────────────────
def make_figure(A, B, C, D, OD, concentration, OD_sample=None, conc_sample=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    x_vals = np.linspace(np.min(concentration), np.max(concentration), 500)
    y_vals = four_param_logistic(x_vals, A, B, C, D)

    ax.plot(y_vals, x_vals, color="#53d8fb", linewidth=2, label="Fitted 4PL Curve", zorder=2)
    ax.scatter(OD, concentration, color="#e94560", s=65, zorder=3,
               label="Standard Points", edgecolors="#ffffff", linewidths=0.5)

    if OD_sample is not None and conc_sample is not None:
        ax.scatter([OD_sample], [conc_sample], color="#4caf84", s=100, zorder=4,
                   marker="D", label=f"Sample  ({OD_sample:.3f} → {conc_sample:.2f})",
                   edgecolors="#ffffff", linewidths=0.7)
        ax.axhline(conc_sample, color="#4caf84", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.axvline(OD_sample,   color="#4caf84", linewidth=0.8, linestyle="--", alpha=0.4)

    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.tick_params(colors="#8892a4", labelsize=8)
    ax.xaxis.label.set_color("#8892a4")
    ax.yaxis.label.set_color("#8892a4")
    ax.set_xlabel("OD", fontsize=9, fontfamily="monospace")
    ax.set_ylabel("Concentration", fontsize=9, fontfamily="monospace")
    ax.set_title("4PL Model Fitting", color="#e6edf3", fontsize=11,
                 fontfamily="monospace", pad=12)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#21262d", alpha=0.9)
    legend = ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#21262d",
                       labelcolor="#e6edf3", loc="best")
    fig.tight_layout(pad=2)
    return fig

# ── Session state defaults ─────────────────────────────────────────────────────
for key, val in {
    "model_ready": False,
    "A": None, "B": None, "C": None, "D": None,
    "concentration": None, "OD": None,
    "results": [],           # list of (od, conc) tuples
    "last_od": None,
    "last_conc": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>◈ 4PL MODEL FITTING</h1>
  <p>Four-Parameter Logistic Regression · Standard Curve Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ──────────────────────────────────────────────────────────
left, right = st.columns([1, 1.9], gap="large")

with left:
    # ── Standard curve inputs
    st.markdown('<div class="section-head">Standard Curve</div>', unsafe_allow_html=True)

    conc_raw = st.text_input(
        "Concentration values (comma-separated)",
        placeholder="e.g. 0, 5, 10, 20, 40, 80",
        key="conc_input"
    )
    od_raw = st.text_input(
        "OD values (comma-separated)",
        placeholder="e.g. 0.05, 0.12, 0.25, 0.48, 0.79, 1.1",
        key="od_input"
    )

    fit_clicked = st.button("▶  FIT MODEL", type="primary", use_container_width=True)

    if fit_clicked:
        try:
            conc = np.array([float(v.strip()) for v in conc_raw.split(",")])
            od   = np.array([float(v.strip()) for v in od_raw.split(",")])
            if len(conc) != len(od):
                st.error("Concentration and OD arrays must be the same length.")
            else:
                A, B, C, D = fit_model(conc, od)
                st.session_state.update({
                    "model_ready": True,
                    "A": A, "B": B, "C": C, "D": D,
                    "concentration": conc,
                    "OD": od,
                    "last_od": None,
                    "last_conc": None,
                })
                st.markdown('<span class="pill-success">✓ Model fitted</span>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

    # ── Model parameters display
    if st.session_state.model_ready:
        st.markdown("---")
        st.markdown('<div class="section-head">Model Parameters</div>', unsafe_allow_html=True)
        A, B, C, D = st.session_state.A, st.session_state.B, st.session_state.C, st.session_state.D
        st.markdown(f"""
        <div class="param-grid">
            <div class="param-card"><div class="label">A — Bottom asymptote</div><div class="value">{A:.5f}</div></div>
            <div class="param-card"><div class="label">B — Hill slope</div><div class="value">{B:.5f}</div></div>
            <div class="param-card"><div class="label">C — EC50 / inflection</div><div class="value">{C:.5f}</div></div>
            <div class="param-card"><div class="label">D — Top asymptote</div><div class="value">{D:.5f}</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Sample calculation
    st.markdown('<div class="section-head">Sample Calculation</div>', unsafe_allow_html=True)

    sample_od = st.number_input(
        "Sample OD value",
        min_value=0.0, step=0.001, format="%.4f",
        disabled=not st.session_state.model_ready,
        key="sample_od"
    )

    calc_clicked = st.button("⊕  CALCULATE CONCENTRATION",
                             use_container_width=True,
                             disabled=not st.session_state.model_ready)

    if calc_clicked:
        try:
            A, B, C, D = st.session_state.A, st.session_state.B, st.session_state.C, st.session_state.D
            conc_val = inverse_four_param_logistic(sample_od, A, B, C, D)
            st.session_state.last_od   = sample_od
            st.session_state.last_conc = conc_val
            st.session_state.results.append({"OD": round(sample_od, 4), "Concentration": round(conc_val, 4)})
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.last_conc is not None:
        st.markdown(f"""
        <div class="result-box">
            <div class="od-label">RESULT</div>
            <span class="od-val">OD {st.session_state.last_od:.4f}</span>
            <span class="arrow">→</span>
            <span class="conc-val">{st.session_state.last_conc:.4f}</span>
            <span style="color:#8892a4; font-size:0.75rem;"> conc</span>
        </div>
        """, unsafe_allow_html=True)

with right:
    # ── Graph
    st.markdown('<div class="section-head">Curve</div>', unsafe_allow_html=True)

    if st.session_state.model_ready:
        fig = make_figure(
            st.session_state.A, st.session_state.B,
            st.session_state.C, st.session_state.D,
            st.session_state.OD, st.session_state.concentration,
            st.session_state.last_od, st.session_state.last_conc
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.markdown("""
        <div style="background:#0d1117; border:1px dashed #21262d; border-radius:8px;
                    height:320px; display:flex; align-items:center; justify-content:center;">
            <span style="color:#8892a4; font-family:'IBM Plex Mono',monospace; font-size:0.85rem;">
                Fit a model to see the curve
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Results table
    if st.session_state.results:
        st.markdown("---")
        st.markdown('<div class="section-head">Results History</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("✕  Clear", use_container_width=True):
                st.session_state.results = []
                st.session_state.last_od = None
                st.session_state.last_conc = None
                st.rerun()

        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇  Export CSV", csv, "4pl_results.csv", "text/csv",
                           use_container_width=True)
