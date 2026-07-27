import io
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

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

<img src="https://hits.sh/elisa-4pl.streamlit.app.svg" style="display:none"/>

""", unsafe_allow_html=True)

# ── Math ──────────────────────────────────────────────────────────────────────
def four_param_logistic(x, A, B, C, D):
    return D + (A - D) / (1 + (x / C)**B)

def inverse_four_param_logistic(OD, A, B, C, D):
    denom = OD - D
    if denom == 0:
        return np.nan  # OD sits exactly on the top asymptote — can't invert
    numerator = (A - OD) / denom
    if numerator < 0:
        return np.nan  # would require a fractional power of a negative number
    return C * (numerator ** (1 / B))

def calculate_sample(sample_od, A, B, C, D, zero_od, has_zero, od_min, od_max):
    """
    Run one sample OD through the fitted curve. Returns a dict with the
    zero-corrected OD, the back-calculated concentration (or None if it
    couldn't be determined), and below-LOD/extrapolated flags.
    """
    od_corrected = sample_od - zero_od if has_zero else sample_od

    # If corrected OD ≤ 0 and we have a zero standard, the sample is at or
    # below the zero standard — concentration is 0, no need to invert the curve.
    if has_zero and od_corrected <= 0:
        # Only a true zero if the raw OD is essentially the zero standard itself.
        # A small negative corrected OD is just noise around the blank — below LOD.
        if abs(od_corrected) < 1e-4:
            conc_val, extrapolated, below_lod = 0.0, False, False
        else:
            conc_val, extrapolated, below_lod = None, False, True
    elif od_corrected < od_min:
        # Corrected OD is positive but below the lowest standard on the curve
        conc_val, extrapolated, below_lod = None, False, True
    else:
        conc_val = inverse_four_param_logistic(od_corrected, A, B, C, D)
        extrapolated = od_corrected > od_max
        below_lod = False
        # OD above top asymptote makes numerator negative → nan
        if np.isnan(conc_val):
            conc_val, extrapolated = None, True

    return {"od_corrected": od_corrected, "conc_val": conc_val,
            "extrapolated": extrapolated, "below_lod": below_lod}

def parse_batch_od(text):
    """Parse pasted OD values (comma- and/or newline-separated) into floats."""
    tokens = [t.strip() for t in text.replace("\n", ",").split(",")]
    tokens = [t for t in tokens if t]
    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError:
            raise ValueError(f"Couldn't parse '{t}' as a number.")
    return values

def fit_model(concentration, OD):
    # A = bottom asymptote (≤ 0 after zero subtraction), D = top asymptote (≥ 0)
    # B > 0 (positive slope), C > 0 (EC50 must be positive)
    lower = [-np.inf, 1e-6,  1e-9,    0.0]
    upper = [   0.0, np.inf, np.inf, np.inf]
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

def compute_param_stderr(covariance):
    """
    Standard error for each of A, B, C, D from the diagonal of the covariance
    matrix returned by curve_fit. A None entry means the fit's covariance
    wasn't finite for that parameter — i.e. curve_fit couldn't pin it down
    at all, which is a stronger warning sign than a merely large SE.
    """
    diag = np.diag(covariance)
    stderrs = []
    for d in diag:
        if not np.isfinite(d) or d < 0:
            stderrs.append(None)
        else:
            stderrs.append(float(np.sqrt(d)))
    return stderrs

def param_uncertainty_flag(kind, value, stderr, scale, threshold=0.5):
    """
    Classify a parameter's uncertainty as 'unknown' (SE couldn't be estimated),
    'high' (poorly constrained), or 'ok'.

    kind='ratio' (B, the dimensionless Hill slope): judged relative to its own
    value, since B is never legitimately near zero.

    kind='scale' (A, D — OD-scale; C — concentration-scale): judged relative to
    the data's own OD or concentration range instead of the parameter's value.
    This matters because A (the bottom asymptote) is *supposed* to land near
    zero once a zero standard has been subtracted — dividing its small, healthy
    standard error by its near-zero value would produce a huge, misleading
    ratio even for a well-constrained fit.
    """
    if stderr is None:
        return "unknown"
    if kind == "ratio":
        if value == 0:
            return "high" if stderr > 1e-6 else "ok"
        return "high" if abs(stderr / value) > threshold else "ok"
    # kind == "scale"
    if scale is None or scale <= 0:
        return "unknown"
    return "high" if stderr > threshold * scale else "ok"

def check_duplicates(concentration):
    seen = {}
    for c in concentration:
        seen[c] = seen.get(c, 0) + 1
    return [c for c, count in seen.items() if count > 1]

# ── File import/export for standard curve data ────────────────────────────────
def parse_standard_csv(uploaded_file):
    """
    Parse an uploaded standard-curve CSV into (concentration_list, od_list, error).
    Expects two columns named 'concentration' and 'od' (case-insensitive). Falls
    back to a plain two-column, no-header file if those names aren't found.
    """
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, None, f"Couldn't read the file as a CSV: {e}"

    cols_lower = [str(c).strip().lower() for c in df.columns]

    if "concentration" in cols_lower and "od" in cols_lower:
        conc_col = df.columns[cols_lower.index("concentration")]
        od_col   = df.columns[cols_lower.index("od")]
    elif df.shape[1] == 2:
        # No 'concentration'/'od' header found — check whether the "header" row
        # is actually numeric, meaning the file has no header row at all.
        try:
            float(df.columns[0])
            float(df.columns[1])
        except (ValueError, TypeError):
            return None, None, (
                "Found 2 columns but no 'concentration'/'od' header. "
                "Use a CSV with a header row, e.g.:\nconcentration,od\n0,0.05\n10,0.18"
            )
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=None, names=["concentration", "od"])
        conc_col, od_col = "concentration", "od"
    else:
        return None, None, f"Expected exactly 2 columns (concentration, od) but found {df.shape[1]}."

    try:
        conc_vals = [float(v) for v in df[conc_col]]
        od_vals   = [float(v) for v in df[od_col]]
    except (ValueError, TypeError) as e:
        return None, None, f"Non-numeric value found in the file: {e}"

    if len(conc_vals) < 2:
        return None, None, "File needs at least 2 data rows."
    if len(conc_vals) != len(od_vals):
        return None, None, "Concentration and OD columns have different lengths."

    return conc_vals, od_vals, None

def build_standard_csv(concentration, od):
    """Serialize entered standard-curve points to CSV bytes for export."""
    df = pd.DataFrame({"concentration": concentration, "od": od})
    return df.to_csv(index=False).encode()

# ── Plot ───────────────────────────────────────────────────────────────────────
def make_figure(A, B, C, D, OD, concentration, sample_points=None, units=""):
    """
    Plot per manual: X-axis = concentration, Y-axis = OD (corrected if applicable).
    OD and concentration arrays passed in are already corrected (zero-subtracted if applicable).

    sample_points: optional list of {"od": float, "conc": float} dicts to plot as
    sample markers. A single point gets dashed guide lines to the axes (the classic
    "read the concentration off the curve" view); multiple points are shown as a
    cluster of diamonds without guide lines to avoid clutter.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#f9f9f7")
    ax.set_facecolor("#ffffff")

    # Curve: x = concentration, y = OD
    x_vals = np.linspace(np.min(concentration), np.max(concentration), 500)
    y_vals = four_param_logistic(x_vals, A, B, C, D)

    ax.plot(x_vals, y_vals, color="#1a1a1a", linewidth=2, label="Fitted 4PL Curve", zorder=2)
    ax.scatter(concentration, OD, color="#e03e3e", s=65, zorder=3,
               label="Standard Points", edgecolors="#fff", linewidths=0.5)

    sample_points = sample_points or []
    if len(sample_points) == 1:
        sp = sample_points[0]
        ax.scatter([sp["conc"]], [sp["od"]], color="#2e55e2", s=100, zorder=4,
                   marker="D", label=f"Sample  (OD {sp['od']:.3f} → {sp['conc']:.2f})",
                   edgecolors="#fff", linewidths=0.7)
        # Dashed lines: horizontal from Y-axis to curve, then vertical down to X-axis
        ax.axhline(sp["od"],   color="#2d7a55", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.axvline(sp["conc"], color="#2d7a55", linewidth=0.8, linestyle="--", alpha=0.4)
    elif len(sample_points) > 1:
        xs = [sp["conc"] for sp in sample_points]
        ys = [sp["od"] for sp in sample_points]
        ax.scatter(xs, ys, color="#2e55e2", s=80, zorder=4, marker="D",
                   label=f"Samples ({len(sample_points)})",
                   edgecolors="#fff", linewidths=0.6)

    for spine in ax.spines.values():
        spine.set_edgecolor("#e8e8e4")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")
    x_label = f"Concentration ({units})" if units else "Concentration"
    ax.set_xlabel(x_label, fontsize=9, fontfamily="monospace")
    ax.set_ylabel("OD (450 nm)", fontsize=9, fontfamily="monospace")
    ax.set_title("4PL Standard Curve", color="#1a1a1a", fontsize=11,
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
    "param_stderr": [None, None, None, None],
    "results": [],
    "last_od": None,
    "last_od_raw": None,
    "last_conc": None,
    "last_extrapolated": False,
    "last_below_lod": False,
    "last_batch_points": [],
    "zero_od": 0.0,
    "has_zero_standard": False,
    "input_mode": "bulk",
    "conc_list": [],
    "od_list": [],
    "row_ids": [],
    "next_row_id": 1,
    "new_conc_val": "",
    "fit_count": 0,
    "confirm_reset_points": False,
    "confirm_clear_results": False,
    "units": "",
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

    UNIT_OPTIONS = [
        "(none)", "pg/mL", "ng/mL", "µg/mL", "mg/mL",
        "mIU/mL", "IU/mL", "U/mL", "ng/dL", "pg/dL",
        "nmol/L", "pmol/L", "nM", "µM", "Custom…"
    ]
    unit_choice = st.selectbox(
        "Concentration units (optional)",
        UNIT_OPTIONS,
        key="unit_choice",
        help="Purely cosmetic — labels the axis, results, and exports. Doesn't affect the fit or any calculations."
    )
    if unit_choice == "Custom…":
        st.session_state.units = st.text_input(
            "Custom unit", placeholder="e.g. copies/mL", key="custom_unit_input"
        ).strip()
    elif unit_choice == "(none)":
        st.session_state.units = ""
    else:
        st.session_state.units = unit_choice

    # Mode toggle
    mode = st.radio(
        "Input mode",
        ["Bulk (comma-separated)", "One by one", "Import file"],
        horizontal=True,
        key="input_mode_radio",
        label_visibility="collapsed"
    )
    if mode == "Bulk (comma-separated)":
        st.session_state.input_mode = "bulk"
    elif mode == "One by one":
        st.session_state.input_mode = "onebyone"
    else:
        st.session_state.input_mode = "import"

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    conc_final = None
    od_final   = None
    bulk_parse_error = None

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
            except ValueError as e:
                bulk_parse_error = f"Couldn't parse concentration values — {e}"
                conc_final = None
            try:
                od_final = [float(v.strip()) for v in od_raw.split(",")]
            except ValueError as e:
                msg = f"Couldn't parse OD values — {e}"
                bulk_parse_error = f"{bulk_parse_error} {msg}" if bulk_parse_error else msg
                od_final = None
            if bulk_parse_error:
                st.error(bulk_parse_error)

    # ── ONE-BY-ONE MODE
    elif st.session_state.input_mode == "onebyone":
        # Start with one pair if empty
        if not st.session_state.conc_list:
            st.session_state.conc_list = [None]
            st.session_state.od_list   = [None]
            st.session_state.row_ids   = [st.session_state.next_row_id]
            st.session_state.next_row_id += 1
        # Safety net: keep row_ids in sync with conc_list length even if they
        # ever drift (e.g. from an older session state shape).
        while len(st.session_state.row_ids) < len(st.session_state.conc_list):
            st.session_state.row_ids.append(st.session_state.next_row_id)
            st.session_state.next_row_id += 1

        to_remove = None
        for i in range(len(st.session_state.conc_list)):
            rid = st.session_state.row_ids[i]
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
                    key=f"conc_row_{rid}",
                    value="" if st.session_state.conc_list[i] is None else str(st.session_state.conc_list[i]),
                )
                if c_val.strip():
                    try:
                        st.session_state.conc_list[i] = float(c_val.strip())
                    except Exception:
                        pass
                else:
                    st.session_state.conc_list[i] = None
            with od_col:
                o_val = st.text_input(
                    "OD", placeholder="e.g. 0.48",
                    key=f"od_row_{rid}",
                    value="" if st.session_state.od_list[i] is None else str(st.session_state.od_list[i]),
                )
                if o_val.strip():
                    try:
                        st.session_state.od_list[i] = float(o_val.strip())
                    except Exception:
                        pass
                else:
                    st.session_state.od_list[i] = None
            with x_col:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if len(st.session_state.conc_list) > 1:
                    if st.button("✕", key=f"remove_{rid}"):
                        to_remove = i

        if to_remove is not None:
            st.session_state.conc_list.pop(to_remove)
            st.session_state.od_list.pop(to_remove)
            st.session_state.row_ids.pop(to_remove)
            st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("＋  Add another point", use_container_width=True):
            st.session_state.conc_list.append(None)
            st.session_state.od_list.append(None)
            st.session_state.row_ids.append(st.session_state.next_row_id)
            st.session_state.next_row_id += 1
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

        if st.session_state.confirm_reset_points:
            st.warning("This clears every point you've entered. Are you sure?")
            yes_col, no_col = st.columns(2)
            with yes_col:
                if st.button("✕  Yes, reset", use_container_width=True, type="primary"):
                    st.session_state.conc_list = [None]
                    st.session_state.od_list   = [None]
                    st.session_state.row_ids   = [st.session_state.next_row_id]
                    st.session_state.next_row_id += 1
                    st.session_state.confirm_reset_points = False
                    st.rerun()
            with no_col:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_reset_points = False
                    st.rerun()
        else:
            if st.button("✕  Reset all", use_container_width=True):
                st.session_state.confirm_reset_points = True
                st.rerun()

    # ── IMPORT MODE
    elif st.session_state.input_mode == "import":
        uploaded_csv = st.file_uploader(
            "Standard curve CSV",
            type=["csv"],
            key="standard_csv_upload",
            help=(
                "CSV with two columns, **concentration** and **od**, one standard "
                "point per row. Example:\n\n"
                "concentration,od\n0,0.05\n10,0.18\n20,0.35\n40,0.62\n80,0.95\n\n"
                "A header row is preferred, but a plain two-column file without "
                "one also works. This is the same format produced by "
                "'Export data as CSV' below."
            ),
        )
        if uploaded_csv is not None:
            conc_parsed, od_parsed, parse_err = parse_standard_csv(uploaded_csv)
            if parse_err:
                st.error(parse_err)
            else:
                conc_final = conc_parsed
                od_final   = od_parsed
                st.success(f"Loaded {len(conc_final)} standard points from file.")
                st.dataframe(
                    pd.DataFrame({"concentration": conc_final, "od": od_final}),
                    use_container_width=True, hide_index=True
                )

    # ── Export current standard curve data (shared across all input modes)
    if conc_final and od_final and len(conc_final) == len(od_final) and len(conc_final) >= 2:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.download_button(
            "⬇  Export data as CSV",
            build_standard_csv(conc_final, od_final),
            "standard_curve_data.csv",
            "text/csv",
            use_container_width=True,
            help="Save these concentration/OD points so you can re-import them later instead of retyping."
        )

    # ── Fit button (shared)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    fit_clicked = st.button("▶  FIT MODEL", type="primary", use_container_width=True)

    if fit_clicked:
        if not conc_final or not od_final:
            if st.session_state.input_mode == "bulk" and bulk_parse_error:
                st.error("Fix the input errors above before fitting.")
            else:
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

                    # Zero-standard subtraction: only when a 0-concentration standard is present
                    zero_mask = conc == 0
                    if zero_mask.any():
                        zero_od = float(np.mean(od[zero_mask]))
                        od_corrected = od - zero_od
                        st.info(f"Zero standard detected (mean OD = {zero_od:.4f}). "
                                f"Subtracting from all OD values before fitting.")
                    else:
                        zero_od = 0.0
                        od_corrected = od  # no subtraction

                    with st.spinner("Fitting 4PL model…"):
                        (A, B, C, D), cov = fit_model(conc, od_corrected)
                    r2 = compute_r2(conc, od_corrected, A, B, C, D)
                    param_stderr = compute_param_stderr(cov)

                    had_prior_result = st.session_state.model_ready and (
                        st.session_state.last_od is not None or st.session_state.last_batch_points
                    )

                    st.session_state.update({
                        "model_ready": True,
                        "A": A, "B": B, "C": C, "D": D,
                        "r2": r2,
                        "param_stderr": param_stderr,
                        "concentration": conc,
                        "OD": od_corrected,
                        "zero_od": zero_od,
                        "has_zero_standard": bool(zero_mask.any()),
                        "last_od": None,
                        "last_conc": None,
                        "last_extrapolated": False,
                        "last_batch_points": [],
                        "fit_count": st.session_state.fit_count + 1,
                    })
                    st.markdown('<span class="pill-success">✓ Model fitted</span>', unsafe_allow_html=True)
                    if had_prior_result:
                        st.info("Re-fitting cleared the previously displayed sample result — recalculate it against the new curve if you still need it.")
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

        stderrs = st.session_state.get("param_stderr") or [None, None, None, None]
        param_names = ["A — Bottom asymptote", "B — Hill slope", "C — EC50 / inflection", "D — Top asymptote"]
        param_values = [A, B, C, D]
        od_range   = float(np.max(st.session_state.OD) - np.min(st.session_state.OD))
        conc_range = float(np.max(st.session_state.concentration) - np.min(st.session_state.concentration))
        param_kinds  = ["scale", "ratio", "scale", "scale"]
        param_scales = [od_range, None, conc_range, od_range]
        flags = [
            param_uncertainty_flag(kind, v, se, scale)
            for kind, v, se, scale in zip(param_kinds, param_values, stderrs, param_scales)
        ]

        cards_html = ""
        flagged_names = []
        for name, value, se, flag in zip(param_names, param_values, stderrs, flags):
            if se is None:
                se_html = '<span style="color:#c0392b;font-size:0.65rem;"> ± could not be estimated</span>'
            else:
                se_color = "#c0392b" if flag == "high" else "#999"
                se_html = f'<span style="color:{se_color};font-size:0.65rem;"> ± {se:.5f}</span>'
            border = "border-left:3px solid #c0392b;" if flag in ("high", "unknown") else ""
            cards_html += (
                f'<div class="param-card" style="{border}">'
                f'<div class="label">{name}</div>'
                f'<div class="value">{value:.5f}{se_html}</div>'
                f'</div>'
            )
            if flag in ("high", "unknown"):
                flagged_names.append(name.split(" — ")[0])

        st.markdown(f"""
        <div class="param-grid">
            {cards_html}
        </div>
        <div class="param-card" style="margin-bottom:10px">
            <div class="label">R² — Goodness of fit</div>
            <div class="value" style="color:{r2_color}">{r2:.6f} <span style="font-size:0.65rem;color:{r2_color}">({r2_label})</span></div>
        </div>
        """, unsafe_allow_html=True)

        if flagged_names:
            st.warning(
                f"Parameter(s) {', '.join(flagged_names)} are poorly constrained by this data "
                f"(large or non-estimable standard error) even though R² may look fine — R² measures "
                f"how well the curve fits the standard points, not how reliable each parameter is. "
                f"Consider adding more standards, especially near the low/high ends of the curve."
            )

    st.markdown("---")

    # ── Sample calculation
    st.markdown('<div class="section-head">Sample Calculation</div>', unsafe_allow_html=True)

    if not st.session_state.model_ready:
        st.caption("Fit a model above to enable sample calculation.")

    calc_mode = st.radio(
        "Sample calculation mode",
        ["Single", "Batch (paste multiple)"],
        horizontal=True,
        key="calc_mode",
        label_visibility="collapsed",
        disabled=not st.session_state.model_ready,
    )

    if calc_mode == "Single":
        sample_od = st.number_input(
            "Sample OD value",
            min_value=0.0, step=0.001, format="%.4f",
            value=None, placeholder="e.g. 0.4800",
            disabled=not st.session_state.model_ready,
            key="sample_od"
        )

        calc_clicked = st.button("⊕  CALCULATE CONCENTRATION",
                                 use_container_width=True,
                                 disabled=not st.session_state.model_ready)

        if calc_clicked:
            if sample_od is None:
                st.error("Enter a sample OD value before calculating.")
            else:
                try:
                    A, B, C, D = st.session_state.A, st.session_state.B, st.session_state.C, st.session_state.D
                    zero_od = st.session_state.get("zero_od", 0.0)
                    has_zero = st.session_state.get("has_zero_standard", False)
                    od_min = 0.0 if has_zero else float(np.min(st.session_state.OD))
                    od_max = float(np.max(st.session_state.OD))

                    r = calculate_sample(sample_od, A, B, C, D, zero_od, has_zero, od_min, od_max)
                    od_corrected, conc_val = r["od_corrected"], r["conc_val"]
                    extrapolated, below_lod = r["extrapolated"], r["below_lod"]

                    st.session_state.last_od           = od_corrected
                    st.session_state.last_od_raw       = sample_od
                    st.session_state.last_conc         = conc_val
                    st.session_state.last_extrapolated = extrapolated
                    st.session_state.last_below_lod    = below_lod
                    st.session_state.last_batch_points = []  # a fresh single calc supersedes any prior batch
                    st.session_state.results.append({
                        "Model Fit #": st.session_state.fit_count,
                        "Raw OD": round(sample_od, 4),
                        "Corrected OD": round(od_corrected, 4) if has_zero else "—",
                        "Concentration": "below LOD" if below_lod else ("> curve max (extrapolated)" if extrapolated and conc_val is None else round(conc_val, 4)),
                        "Note": "extrapolated" if extrapolated else ("below LOD" if below_lod else ""),
                        "_od_corrected": od_corrected,
                        "_conc_value": conc_val,
                    })
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error: {e}")

    else:  # Batch mode
        batch_text = st.text_area(
            "Sample OD values",
            placeholder="e.g. 0.42, 0.55, 0.61\nor one per line, pasted straight from a spreadsheet column",
            height=120,
            disabled=not st.session_state.model_ready,
            key="batch_od_input",
            help="Comma-separated and/or one-per-line — both work, and you can mix them."
        )

        calc_all_clicked = st.button("⊕  CALCULATE ALL",
                                     use_container_width=True,
                                     disabled=not st.session_state.model_ready)

        if calc_all_clicked:
            if not batch_text or not batch_text.strip():
                st.error("Paste one or more OD values before calculating.")
            else:
                try:
                    values = parse_batch_od(batch_text)
                    if not values:
                        st.error("No OD values found.")
                    else:
                        A, B, C, D = st.session_state.A, st.session_state.B, st.session_state.C, st.session_state.D
                        zero_od = st.session_state.get("zero_od", 0.0)
                        has_zero = st.session_state.get("has_zero_standard", False)
                        od_min = 0.0 if has_zero else float(np.min(st.session_state.OD))
                        od_max = float(np.max(st.session_state.OD))

                        n_ok = n_below = n_extrap = 0
                        batch_points = []
                        for v in values:
                            r = calculate_sample(v, A, B, C, D, zero_od, has_zero, od_min, od_max)
                            od_corrected, conc_val = r["od_corrected"], r["conc_val"]
                            extrapolated, below_lod = r["extrapolated"], r["below_lod"]
                            st.session_state.results.append({
                                "Model Fit #": st.session_state.fit_count,
                                "Raw OD": round(v, 4),
                                "Corrected OD": round(od_corrected, 4) if has_zero else "—",
                                "Concentration": "below LOD" if below_lod else ("> curve max (extrapolated)" if extrapolated and conc_val is None else round(conc_val, 4)),
                                "Note": "extrapolated" if extrapolated else ("below LOD" if below_lod else ""),
                                "_od_corrected": od_corrected,
                                "_conc_value": conc_val,
                            })
                            if below_lod:
                                n_below += 1
                            elif extrapolated:
                                n_extrap += 1
                            else:
                                n_ok += 1
                            if conc_val is not None:
                                batch_points.append({"od": od_corrected, "conc": conc_val})

                        # A single "latest result" box doesn't make sense for a batch —
                        # clear it and instead show the whole batch as the "Latest result"
                        # cluster on the curve, so nothing extra needs to be toggled.
                        st.session_state.last_od = None
                        st.session_state.last_conc = None
                        st.session_state.last_batch_points = batch_points

                        summary = f"✓ {len(values)} sample(s) calculated — {n_ok} in range"
                        if n_below:
                            summary += f", {n_below} below LOD"
                        if n_extrap:
                            summary += f", {n_extrap} extrapolated"
                        summary += ". Plotted on the curve below."
                        st.success(summary)
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.last_od is not None:
        extrap     = st.session_state.get("last_extrapolated", False)
        below_lod  = st.session_state.get("last_below_lod", False)
        has_zero   = st.session_state.get("has_zero_standard", False)
        zero_od    = st.session_state.get("zero_od", 0.0)
        raw_od     = st.session_state.get("last_od_raw", st.session_state.last_od)
        corr_od    = st.session_state.last_od
        conc_val   = st.session_state.last_conc

        border_color = "#e8a020" if extrap else "#c0392b" if below_lod else "#1a1a1a"

        if below_lod:
            extra_note = '<div style="color:#c0392b;font-size:0.68rem;margin-top:6px">⚠ OD is below the lowest standard — concentration is below the limit of detection for this curve</div>'
        elif extrap:
            extra_note = '<div style="color:#a06000;font-size:0.68rem;margin-top:6px">⚠ OD is outside standard curve range — treat with caution</div>'
        else:
            extra_note = ""

        if has_zero:
            od_display = (
                f'<span class="od-val">OD {raw_od:.4f}</span>'
                f'<span style="color:#aaa;font-size:0.75rem;"> − {zero_od:.4f} (zero std) = </span>'
                f'<span class="od-val">{corr_od:.4f}</span>'
            )
        else:
            od_display = f'<span class="od-val">OD {raw_od:.4f}</span>'

        unit_suffix = st.session_state.units if st.session_state.units else "conc"
        conc_display = (
            '<span class="conc-val" style="color:#c0392b">below LOD</span>'
            '<span style="color:#aaa; font-size:0.75rem;"> (below lowest standard)</span>'
        ) if below_lod else (
            '<span class="conc-val" style="color:#a06000">&gt; curve max</span>'
            '<span style="color:#aaa; font-size:0.75rem;"> (cannot extrapolate)</span>'
        ) if extrap and conc_val is None else (
            f'<span class="conc-val">{conc_val:.4f}</span>'
            f'<span style="color:#aaa; font-size:0.75rem;"> {unit_suffix}</span>'
        )

        st.markdown(f"""
        <div class="result-box" style="border-left-color:{border_color}">
            <div class="od-label">RESULT</div>
            {od_display}
            <span class="arrow">→</span>
            {conc_display}
            {extra_note}
        </div>
        """, unsafe_allow_html=True)

with right:
    # ── Graph
    st.markdown('<div class="section-head">Curve</div>', unsafe_allow_html=True)

    plot_mode = st.radio(
        "Points to plot on curve",
        ["Latest result", "All results", "Selected rows"],
        horizontal=True,
        key="plot_mode",
        label_visibility="collapsed",
        help=(
            "Latest result — only the most recently calculated sample.\n\n"
            "All results — every calculated sample from the current fit.\n\n"
            "Selected rows — pick specific rows in Results History below."
        ),
    )
    curve_placeholder = st.empty()

    # ── Results table (rendered here so its selection is available for the
    # placeholder above, filled in further down)
    selected_rows = []
    if st.session_state.results:
        st.markdown("---")
        st.markdown('<div class="section-head">Results History</div>', unsafe_allow_html=True)

        if st.session_state.confirm_clear_results:
            warn_col, yes_col, no_col = st.columns([3, 1, 1])
            with warn_col:
                st.warning("This clears all results. Are you sure?")
            with yes_col:
                if st.button("Yes, clear", use_container_width=True, type="primary"):
                    st.session_state.results = []
                    st.session_state.last_od = None
                    st.session_state.last_conc = None
                    st.session_state.last_batch_points = []
                    st.session_state.confirm_clear_results = False
                    st.rerun()
            with no_col:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear_results = False
                    st.rerun()
        else:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("✕  Clear all", use_container_width=True):
                    st.session_state.confirm_clear_results = True
                    st.rerun()

        df_full = pd.DataFrame(st.session_state.results)
        df_display = df_full.drop(columns=["_od_corrected", "_conc_value"], errors="ignore")
        if st.session_state.units:
            df_display = df_display.rename(columns={"Concentration": f"Concentration ({st.session_state.units})"})
        selection = st.dataframe(
            df_display, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="multi-row", key="results_table"
        )
        selected_rows = selection.selection.rows if selection and selection.selection else []
        if selected_rows:
            if st.button(f"✕  Delete {len(selected_rows)} selected row(s)", use_container_width=True):
                st.session_state.results = [
                    r for i, r in enumerate(st.session_state.results) if i not in selected_rows
                ]
                st.session_state.last_batch_points = []
                st.rerun()

        csv = df_display.to_csv(index=False).encode()
        st.download_button("⬇  Export CSV", csv, "4pl_results.csv", "text/csv",
                           use_container_width=True)

    # ── Fill in the curve now that we know the current table selection
    with curve_placeholder.container():
        if st.session_state.model_ready:
            current_fit = st.session_state.fit_count
            sample_points = []
            skipped_other_fit = 0
            skipped_no_value = 0

            if plot_mode == "Latest result":
                if st.session_state.last_batch_points:
                    sample_points = st.session_state.last_batch_points
                elif st.session_state.last_od is not None and st.session_state.last_conc is not None:
                    sample_points = [{"od": st.session_state.last_od, "conc": st.session_state.last_conc}]
            else:
                if plot_mode == "All results":
                    candidates = st.session_state.results
                else:  # Selected rows
                    candidates = [st.session_state.results[i] for i in selected_rows]

                for r in candidates:
                    if r["Model Fit #"] != current_fit:
                        skipped_other_fit += 1
                    elif r.get("_conc_value") is None:
                        skipped_no_value += 1
                    else:
                        sample_points.append({"od": r["_od_corrected"], "conc": r["_conc_value"]})

                if plot_mode == "Selected rows" and not selected_rows:
                    st.caption("Select one or more rows in Results History below to plot them here.")

            fig = make_figure(
                st.session_state.A, st.session_state.B,
                st.session_state.C, st.session_state.D,
                st.session_state.OD, st.session_state.concentration,
                sample_points, units=st.session_state.units
            )
            st.pyplot(fig, use_container_width=True)

            if skipped_other_fit:
                st.caption(f"{skipped_other_fit} point(s) skipped — calculated against a previous curve fit.")
            if skipped_no_value:
                st.caption(f"{skipped_no_value} point(s) skipped — below LOD or above curve max, no concentration to plot.")

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", dpi=200, bbox_inches="tight")
            img_buf.seek(0)
            st.download_button(
                "⬇  Export curve image (PNG)",
                img_buf,
                f"standard_curve_fit_{st.session_state.fit_count}.png",
                "image/png",
                use_container_width=True,
                help="Download the chart above as a PNG image."
            )

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

st.markdown("""
<div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#ccc;
            text-align:center; padding: 24px 0 8px 0;">
    built by Omnia Abouhaikal &nbsp;·&nbsp; @oniaz
</div>
""", unsafe_allow_html=True)