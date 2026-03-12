import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
import pandas as pd
import streamlit_analytics2 as streamlit_analytics

streamlit_analytics.start_tracking()
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ELISA 4PL Fitting",
    page_icon="favicon.png",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Title */
.title-block {
    border-left: 3px solid var(--text-color, currentColor);
    padding: 6px 0 6px 16px;
    margin-bottom: 28px;
    opacity: 0.95;
}
.title-block-h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    margin: 0;
    letter-spacing: 1px;
    font-weight: 500;
}
.title-block p {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    opacity: 0.5;
    margin: 4px 0 0 0;
    letter-spacing: 0.5px;
}

/* Section headers */
.section-head {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 3px;
    opacity: 0.4;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid rgba(128,128,128,0.2);
}

/* Param cards — theme-aware */
.param-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 18px;
}
.param-card {
    background: rgba(128,128,128,0.07);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 6px;
    padding: 10px 14px;
}
.param-card .label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    opacity: 0.45;
    letter-spacing: 0.5px;
}
.param-card .value {
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
}

/* Result highlight — theme-aware */
.result-box {
    background: rgba(128,128,128,0.07);
    border: 1px solid rgba(128,128,128,0.15);
    border-left: 3px solid rgba(128,128,128,0.5);
    border-radius: 6px;
    padding: 14px 18px;
    margin: 14px 0;
    font-family: 'DM Mono', monospace;
}
.result-box .od-label { font-size: 0.7rem; opacity: 0.45; }
.result-box .od-val   { font-size: 1rem; }
.result-box .arrow    { opacity: 0.4; font-size: 1rem; margin: 0 8px; }
.result-box .conc-val { font-size: 1.2rem; font-weight: 600; }

/* Status pills */
.pill-success {
    display: inline-block;
    background: rgba(45,122,85,0.1);
    color: #2d7a55;
    border: 1px solid rgba(45,122,85,0.3);
    border-radius: 4px;
    padding: 3px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.5px;
}
.pill-warn {
    display: inline-block;
    background: rgba(160,96,0,0.1);
    color: #a06000;
    border: 1px solid rgba(160,96,0,0.3);
    border-radius: 4px;
    padding: 3px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
}

/* Inputs — font only, let Streamlit handle colors */
input, textarea {
    font-family: 'DM Mono', monospace !important;
    border-radius: 5px !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    font-family: 'DM Mono', monospace !important;
    border-radius: 5px !important;
}
label { font-size: 0.75rem !important; letter-spacing: 0.3px; }

/* Buttons */
div[data-testid="stButton"] button {
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    border-radius: 5px !important;
    transition: all 0.15s ease !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 6px;
    overflow: hidden;
}

/* Divider */
hr { border-color: rgba(128,128,128,0.2) !important; margin: 20px 0 !important; }
</style>

<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-V6593BYP4X"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-V6593BYP4X');
</script>

""", unsafe_allow_html=True)

# ── Math ──────────────────────────────────────────────────────────────────────
def four_param_logistic(x, A, B, C, D):
    return D + (A - D) / (1 + (x / C)**B)

def inverse_four_param_logistic(OD, A, B, C, D):
    # Check sign to avoid silent abs() masking bad inputs
    numerator = (A - OD) / (OD - D)
    if numerator <= 0:
        raise ValueError(
            f"OD value {OD:.4f} is outside the range of the standard curve "
            f"(valid range: {min(A,D):.4f} – {max(A,D):.4f}). "
            "Result would be extrapolated and unreliable."
        )
    return C * (numerator ** (1 / B))

def fit_model(concentration, OD):
    # Bounds: A,D unconstrained; B > 0 (positive slope); C > 0 (EC50 must be positive)
    lower = [-np.inf, 1e-6,  1e-9, -np.inf]
    upper = [ np.inf, np.inf, np.inf,  np.inf]
    params, covariance = opt.curve_fit(
        four_param_logistic, concentration, OD,
        p0=[min(OD), 1.0, np.median(concentration), max(OD)],
        bounds=(lower, upper),
        maxfev=10000
    )
    return params, covariance

def compute_r2(concentration, OD, A, B, C, D):
    predicted = four_param_logistic(concentration, A, B, C, D)
    ss_res = np.sum((OD - predicted) ** 2)
    ss_tot = np.sum((OD - np.mean(OD)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

def check_duplicates(concentration):
    seen = {}
    for c in concentration:
        seen[c] = seen.get(c, 0) + 1
    return [c for c, count in seen.items() if count > 1]

# ── Plot ───────────────────────────────────────────────────────────────────────
def make_figure(A, B, C, D, OD, concentration, OD_sample=None, conc_sample=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#f9f9f7")
    ax.set_facecolor("#ffffff")

    x_vals = np.linspace(np.min(concentration), np.max(concentration), 500)
    y_vals = four_param_logistic(x_vals, A, B, C, D)

    ax.plot(y_vals, x_vals, color="#1a1a1a", linewidth=2, label="Fitted 4PL Curve", zorder=2)
    ax.scatter(OD, concentration, color="#e03e3e", s=65, zorder=3,
               label="Standard Points", edgecolors="#fff", linewidths=0.5)

    if OD_sample is not None and conc_sample is not None:
        ax.scatter([OD_sample], [conc_sample], color="#2e55e2", s=100, zorder=4,
                   marker="D", label=f"Sample  ({OD_sample:.3f} → {conc_sample:.2f})",
                   edgecolors="#fff", linewidths=0.7)
        ax.axhline(conc_sample, color="#2d7a55", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.axvline(OD_sample,   color="#2d7a55", linewidth=0.8, linestyle="--", alpha=0.4)

    for spine in ax.spines.values():
        spine.set_edgecolor("#e8e8e4")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")
    ax.set_xlabel("OD", fontsize=9, fontfamily="monospace")
    ax.set_ylabel("Concentration", fontsize=9, fontfamily="monospace")
    ax.set_title("4PL Model Fitting", color="#1a1a1a", fontsize=11,
                 fontfamily="monospace", pad=12)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#e8e8e4", alpha=0.9)
    legend = ax.legend(fontsize=8, facecolor="#fff", edgecolor="#e8e8e4",
                       labelcolor="#1a1a1a", loc="best")
    fig.tight_layout(pad=2)
    return fig

# ── Session state defaults ─────────────────────────────────────────────────────
for key, val in {
    "model_ready": False,
    "A": None, "B": None, "C": None, "D": None,
    "concentration": None, "OD": None,
    "r2": None,
    "results": [],
    "last_od": None,
    "last_conc": None,
    "last_extrapolated": False,
    "input_mode": "bulk",
    "conc_list": [],
    "od_list": [],
    "new_conc_val": "",
    "fit_count": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <div class="title-block-h1">4PL MODEL FITTING</div>
  <p>ELISA Standard Curve Analysis · Four-Parameter Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ──────────────────────────────────────────────────────────
left, right = st.columns([1, 1.9], gap="large")

with left:
    # ── Standard curve inputs
    st.markdown('<div class="section-head">Standard Curve</div>', unsafe_allow_html=True)

    # Mode toggle
    mode = st.radio(
        "Input mode",
        ["Bulk (comma-separated)", "One by one"],
        horizontal=True,
        key="input_mode_radio",
        label_visibility="collapsed"
    )
    st.session_state.input_mode = "bulk" if mode == "Bulk (comma-separated)" else "onebyone"

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    conc_final = None
    od_final   = None

    # ── BULK MODE
    if st.session_state.input_mode == "bulk":
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
        if conc_raw and od_raw:
            try:
                conc_final = [float(v.strip()) for v in conc_raw.split(",")]
                od_final   = [float(v.strip()) for v in od_raw.split(",")]
            except Exception:
                pass

    # ── ONE-BY-ONE MODE
    else:
        # Start with one pair if empty
        if not st.session_state.conc_list:
            st.session_state.conc_list = [None]
            st.session_state.od_list   = [None]

        to_remove = None
        for i in range(len(st.session_state.conc_list)):
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
                f"color:#aaa;letter-spacing:2px;margin-bottom:4px;margin-top:{'0' if i==0 else '14px'}'"
                f">POINT {i+1}</div>",
                unsafe_allow_html=True
            )
            c_col, od_col, x_col = st.columns([2, 2, 0.5])
            with c_col:
                c_val = st.text_input(
                    "Concentration", placeholder="e.g. 10",
                    key=f"conc_row_{i}",
                    value="" if st.session_state.conc_list[i] is None else str(st.session_state.conc_list[i]),
                )
                if c_val.strip():
                    try:
                        st.session_state.conc_list[i] = float(c_val.strip())
                    except Exception:
                        pass
            with od_col:
                o_val = st.text_input(
                    "OD", placeholder="e.g. 0.48",
                    key=f"od_row_{i}",
                    value="" if st.session_state.od_list[i] is None else str(st.session_state.od_list[i]),
                )
                if o_val.strip():
                    try:
                        st.session_state.od_list[i] = float(o_val.strip())
                    except Exception:
                        pass
            with x_col:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if len(st.session_state.conc_list) > 1:
                    if st.button("✕", key=f"remove_{i}"):
                        to_remove = i

        if to_remove is not None:
            st.session_state.conc_list.pop(to_remove)
            st.session_state.od_list.pop(to_remove)
            st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("＋  Add another point", use_container_width=True):
            st.session_state.conc_list.append(None)
            st.session_state.od_list.append(None)
            st.rerun()

        # Build final arrays only if all filled
        all_filled = (
            all(v is not None for v in st.session_state.conc_list) and
            all(v is not None for v in st.session_state.od_list) and
            len(st.session_state.conc_list) >= 2
        )
        if all_filled:
            conc_final = st.session_state.conc_list
            od_final   = st.session_state.od_list

        if st.button("✕  Reset all", use_container_width=True):
            st.session_state.conc_list = [None]
            st.session_state.od_list   = [None]
            st.rerun()

    # ── Fit button (shared)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    fit_clicked = st.button("▶  FIT MODEL", type="primary", use_container_width=True)

    if fit_clicked:
        if not conc_final or not od_final:
            st.error("Fill in all values before fitting.")
        else:
            try:
                conc = np.array(conc_final, dtype=float)
                od   = np.array(od_final,   dtype=float)
                if len(conc) != len(od):
                    st.error("Concentration and OD arrays must be the same length.")
                elif len(conc) < 4:
                    st.error("Need at least 4 data points to fit a 4PL model.")
                elif np.any(conc < 0):
                    st.error("Concentration values must be non-negative.")
                else:
                    # Warn about duplicates but still allow fitting
                    dupes = check_duplicates(conc.tolist())
                    if dupes:
                        st.warning(f"Duplicate concentration values detected: {dupes}. This may affect fit quality.")
                    (A, B, C, D), cov = fit_model(conc, od)
                    r2 = compute_r2(conc, od, A, B, C, D)
                    st.session_state.update({
                        "model_ready": True,
                        "A": A, "B": B, "C": C, "D": D,
                        "r2": r2,
                        "concentration": conc,
                        "OD": od,
                        "last_od": None,
                        "last_conc": None,
                        "last_extrapolated": False,
                        "fit_count": st.session_state.fit_count + 1,
                    })
                    st.markdown('<span class="pill-success">✓ Model fitted</span>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Model parameters display
    if st.session_state.model_ready:
        st.markdown("---")
        st.markdown('<div class="section-head">Model Parameters</div>', unsafe_allow_html=True)
        A, B, C, D = st.session_state.A, st.session_state.B, st.session_state.C, st.session_state.D
        r2 = st.session_state.r2
        r2_color = "#2d7a55" if r2 >= 0.99 else "#a06000" if r2 >= 0.95 else "#c0392b"
        r2_label = "excellent" if r2 >= 0.99 else "acceptable" if r2 >= 0.95 else "poor — check data"
        st.markdown(f"""
        <div class="param-grid">
            <div class="param-card"><div class="label">A — Bottom asymptote</div><div class="value">{A:.5f}</div></div>
            <div class="param-card"><div class="label">B — Hill slope</div><div class="value">{B:.5f}</div></div>
            <div class="param-card"><div class="label">C — EC50 / inflection</div><div class="value">{C:.5f}</div></div>
            <div class="param-card"><div class="label">D — Top asymptote</div><div class="value">{D:.5f}</div></div>
        </div>
        <div class="param-card" style="margin-bottom:10px">
            <div class="label">R² — Goodness of fit</div>
            <div class="value" style="color:{r2_color}">{r2:.6f} <span style="font-size:0.65rem;color:{r2_color}">({r2_label})</span></div>
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
            od_min = float(np.min(st.session_state.OD))
            od_max = float(np.max(st.session_state.OD))
            extrapolated = sample_od < od_min or sample_od > od_max
            st.session_state.last_od          = sample_od
            st.session_state.last_conc        = conc_val
            st.session_state.last_extrapolated = extrapolated
            flag = " ⚠ extrapolated" if extrapolated else ""
            st.session_state.results.append({
                "Model Fit #": st.session_state.fit_count,
                "OD": round(sample_od, 4),
                "Concentration": round(conc_val, 4),
                "Note": "extrapolated" if extrapolated else ""
            })
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.last_conc is not None:
        extrap = st.session_state.get("last_extrapolated", False)
        border_color = "#e8a020" if extrap else "#1a1a1a"
        extra_note = '<div style="color:#a06000;font-size:0.68rem;margin-top:6px">⚠ OD is outside standard curve range — treat with caution</div>' if extrap else ""
        st.markdown(f"""
        <div class="result-box" style="border-left-color:{border_color}">
            <div class="od-label">RESULT</div>
            <span class="od-val">OD {st.session_state.last_od:.4f}</span>
            <span class="arrow">→</span>
            <span class="conc-val">{st.session_state.last_conc:.4f}</span>
            <span style="color:#aaa; font-size:0.75rem;"> conc</span>
            {extra_note}
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
        <div style="background:#fff; border:1px dashed #ddddd8; border-radius:8px;
                    height:320px; display:flex; align-items:center; justify-content:center;">
            <span style="color:#bbb; font-family:'DM Mono',monospace; font-size:0.85rem;">
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

st.markdown("""
<div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#ccc;
            text-align:center; padding: 24px 0 8px 0;">
    built by Omnia Abouhaikal &nbsp;·&nbsp; @oniaz
</div>
""", unsafe_allow_html=True)

streamlit_analytics.stop_tracking()