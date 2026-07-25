# ELISA Standard Curve Analyzer - 4PL Model Fitting
![App Icon](favicon.png)

A clean, minimal web app for fitting **Four-Parameter Logistic (4PL)** curves to ELISA standard curve data and calculating sample concentrations from OD readings.

Built with Python + Streamlit.

---

## What it does

In ELISA, you run a series of standards with known concentrations and measure their optical density (OD). This app:

1. Takes your standard concentration and OD values
2. Fits a 4PL curve to the data
3. Lets you enter unknown sample OD values and back-calculates their concentrations
4. Flags extrapolated results (OD outside standard curve range)
5. Shows R² so you can assess fit quality
6. Exports results to CSV

---

## The 4PL Model

The Four-Parameter Logistic equation used is:

```
y = D + (A - D) / (1 + (x / C)^B)
```

| Parameter | Description |
|-----------|-------------|
| A | Bottom asymptote (minimum response) |
| B | Hill slope (steepness of the curve) |
| C | EC50 / inflection point |
| D | Top asymptote (maximum response) |

---

## Getting started

### Requirements

```bash
pip install streamlit numpy scipy matplotlib pandas
```

### Run locally

```bash
streamlit run app.py
```
---

## How to use

### Entering standard curve data

You have three input modes:

**Bulk** — paste all values comma-separated:
```
Concentration: 0, 10, 20, 40, 80, 160
OD:            0.05, 0.18, 0.35, 0.62, 0.95, 1.28
```

**One by one** — add each concentration/OD pair individually using the `+ Add another point` button. Good for entering values directly from your plate reader as you go.

**Import file** — upload a CSV with `concentration` and `od` columns to skip retyping data you've entered before. See [Import/Export](#importexport-standard-curve-data) below.

### Import/Export standard curve data

Once you've typed in a standard curve (via Bulk or One by one), click **⬇ Export data as CSV** to save it. That file can later be re-loaded via the **Import file** mode, so you don't have to retype the same standards every session.

Expected file shape — two columns, `concentration` and `od`, one standard point per row:

```
concentration,od
0,0.05
10,0.18
20,0.35
40,0.62
80,0.95
160,1.28
```

A header row is preferred, but a plain two-column file without one is also accepted. Hover the **?** next to the file upload field in the app for a reminder of this format.

Note: this only imports/exports the standard curve *inputs* (concentration/OD pairs) — not fitted parameters or the sample results history. Use **⬇ Export CSV** in Results History to export calculated sample results.

### Fitting the model

Click **▶ FIT MODEL**. The app will:
- Validate your inputs (minimum 4 points, no negative concentrations)
- Warn you about duplicate concentration values
- Display the fitted curve and parameters A, B, C, D
- Show R² with a color-coded quality indicator

| R² | Quality |
|----|---------|
| ≥ 0.99 | Excellent ✓ |
| ≥ 0.95 | Acceptable |
| < 0.95 | Poor — check your data |

### Calculating sample concentrations

Enter your sample's OD value and click **⊕ CALCULATE CONCENTRATION**.

- Results within the standard curve range are shown normally
- Results outside the range are flagged with a ⚠ extrapolation warning — treat these with caution

All results are logged in the **Results History** table with the fit number they belong to, so you can tell which results came from which standard curve if you re-fit mid-session.

### Exporting

Click **⬇ Export CSV** to download all calculated results.

Click **⬇ Export curve image (PNG)** above the plot to save the fitted curve chart itself.

---

## Project structure

```
.
├── app.py            # Main Streamlit app
├── requirements.txt  # Python dependencies
├── favicon.png       # App favicon
└── README.md
```

---

## Built by

Omnia Abouhaikal · [@oniaz](https://github.com/oniaz)