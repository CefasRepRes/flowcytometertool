# qc_plots.py — R-parity QC plots for the watcher
# Dependencies: pandas, numpy, matplotlib, seaborn
# Thread-safe: uses Agg backend, no GUI required

import os
import math
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # critical for background threads
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110

# ---------- IO helpers ----------

def load_instrument_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[qc] Failed reading instrument CSV: {e}")
        return None

def load_predictions_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _safe_savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_fws_scatter(preds_df, file_prefix, out_dir, max_points_cap=60000):
    """
    Per-file scatter of Forward Scatter Left vs Right with adaptive thinning.
    Expected columns in predictions CSV:
      - 'Forward_Scatter_Left_total'
      - 'Forward_Scatter_Right_total'

    Thinning strategy (probabilistic / Poisson thinning):
      - Compute target_points = base + growth * log10(N / scale + 1)
      - Keep each point with probability p = min(1, target_points / N)
      - Cap the target at max_points_cap to bound runtime and file size.

    This preserves the overall shape without biasing dense regions (vs top-N).
    """
    if preds_df is None:
        return
    req = ["Forward_Scatter_Left_total", "Forward_Scatter_Right_total"]
    if not all(c in preds_df.columns for c in req):
        return

    df = preds_df[req].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n == 0:
        return

    # --- Adaptive target curve (sub-linear growth with log) ---
    # Tunables:
    base = 8000          # show ~all up to ~8k
    growth = 22000       # additional points gained via log growth
    scale = 3000         # controls where the log starts to take off
    target = base + growth * np.log10(n / scale + 1.0)
    target = int(min(max_points_cap, max(base, target)))

    # --- Probabilistic thinning (density-agnostic) ---
    if target < n:
        p = target / n
        # Reproducible RNG per file (stable plot when re-run)
        seed = abs(hash(file_prefix)) % (2**32)
        rng = np.random.default_rng(seed)
        keep_mask = rng.random(n) < p
        df = df.loc[keep_mask]
    # else: n is already small; keep all

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(
        df["Forward_Scatter_Left_total"].values,
        df["Forward_Scatter_Right_total"].values,
        s=4, alpha=0.7, linewidths=0, color="#4C72B0"
    )
    ax.set_xlabel("Forward Scatter (Left) — total")
    ax.set_ylabel("Forward Scatter (Right) — total")
    ax.set_title(f"FWS L vs R (thinned): {file_prefix}  •  n={n:,}  →  shown={len(df):,}")

    # Tidy axes (robust to outliers)
    try:
        xlims = np.nanpercentile(df["Forward_Scatter_Left_total"], [1, 99])
        ylims = np.nanpercentile(df["Forward_Scatter_Right_total"], [1, 99])
        pad_x = 0.08 * (xlims[1] - xlims[0] + 1e-9)
        pad_y = 0.08 * (ylims[1] - ylims[0] + 1e-9)
        ax.set_xlim(xlims[0] - pad_x, xlims[1] + pad_x)
        ax.set_ylim(ylims[0] - pad_y, ylims[1] + pad_y)
    except Exception:
        pass

    # Optional: y=x reference (quick symmetry check)
    try:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", alpha=0.7)
    except Exception:
        pass

    _safe_savefig(os.path.join(out_dir, f"{file_prefix}_FWS_L_vs_R.png"))

def _pick_timestamp(instrument_df, instrument_csv_path=None):
    """
    Try common timestamp fields; fallback to file mtime to keep time-series usable.
    """
    candidates = [
        "start",
        "measurementResults_start",
        "instrument_measurementResults_start",
        "measurementResults.dateTime",
        "measurementResults_dateTime",
        "dateTime",
        "instrument_dateTime",
        "instrument.dateTime",
    ]
    for col in candidates:
        if col in instrument_df.columns:
            val = instrument_df[col].iloc[0]
            ts = pd.to_datetime(val, errors="coerce", utc=True)
            if pd.notna(ts):
                return ts
            # epoch seconds?
            try:
                if pd.notna(val) and str(val).strip() != "":
                    f = float(val)
                    if f > 10_000:
                        return pd.to_datetime(f, unit="s", utc=True)
            except Exception:
                pass

    if instrument_csv_path and os.path.exists(instrument_csv_path):
        mtime = os.path.getmtime(instrument_csv_path)
        return pd.to_datetime(mtime, unit="s", utc=True)
    return pd.NaT

# ---------- Sensors & limits we track (names map to instrument columns) ----------

SENSOR_MAP = {
    # name_in_qc_table : [candidate column names in instrument.csv]
    "absolutePressure": ["measurementResults_absolutePressure", "absolutePressure"],
    "differentialPressure": ["measurementResults_differentialPressure", "differentialPressure"],
    "sheathTemperature": ["measurementResults_sheathTemperature", "sheathTemperature"],
    "systemTemperature": ["measurementResults_systemTemperature", "systemTemperature"],
    "laserBeamWidth": ["laserBeamWidth", "measurementSettings_CytoSettings_LaserBeamWidth"],
    "sampleCoreSpeed": ["sampleCoreSpeed", "measurementSettings_CytoSettings_SampleCorespeed", "measurementSettings_CytoSettings_SampleCoreSpeed"],
    "particleRateSensor": ["measurementResults_particleRate", "particleRate"],  # if present
}

# Limits map: we’ll try to discover from instrument.csv SensorLimits_* columns
# e.g., SensorLimits_SystemTemp_minValue / SensorLimits_SystemTemp_maxValue
SENSOR_LIMIT_CANDIDATES = {
    "sheathTemperature": [("measurementSettings_CytoSettings_SensorLimits_SheathTemp_minValue",
                           "measurementSettings_CytoSettings_SensorLimits_SheathTemp_maxValue")],
    "systemTemperature": [("measurementSettings_CytoSettings_SensorLimits_SystemTemp_minValue",
                           "measurementSettings_CytoSettings_SensorLimits_SystemTemp_maxValue")],
    "absolutePressure": [("measurementSettings_CytoSettings_SensorLimits_PressureAbs_minValue",
                          "measurementSettings_CytoSettings_SensorLimits_PressureAbs_maxValue")],
    "differentialPressure": [("measurementSettings_CytoSettings_SensorLimits_PressureDiff_minValue",
                               "measurementSettings_CytoSettings_SensorLimits_PressureDiff_maxValue")],
}

def _get_first_present(df, names):
    for n in names:
        if n in df.columns:
            try:
                v = df[n].iloc[0]
                return float(v)
            except Exception:
                try:
                    return pd.to_numeric(df[n].iloc[0], errors="coerce")
                except Exception:
                    return np.nan
    return np.nan

def _get_limits_from_instrument(df):
    limits = {}
    for sensor, pair_list in SENSOR_LIMIT_CANDIDATES.items():
        for (min_col, max_col) in pair_list:
            if min_col in df.columns or max_col in df.columns:
                vmin = pd.to_numeric(df.get(min_col, pd.Series([np.nan])).iloc[0], errors="coerce")
                vmax = pd.to_numeric(df.get(max_col, pd.Series([np.nan])).iloc[0], errors="coerce")
                limits[sensor] = (float(vmin) if pd.notna(vmin) else None,
                                  float(vmax) if pd.notna(vmax) else None)
                break
    return limits

# ---------- QC measurement row ----------

def build_measurement_row(instrument_df, predictions_df, instrument_csv_path=None):
    row = {}

    # 1) Timestamp
    row["start"] = _pick_timestamp(instrument_df, instrument_csv_path)

    # 2) Trigger level
    if "triggerLevel" in instrument_df.columns:
        row["triggerLevel"] = instrument_df["triggerLevel"].iloc[0]
    else:
        guess = [c for c in instrument_df.columns if "trigger" in c.lower()]
        row["triggerLevel"] = instrument_df[guess[0]].iloc[0] if guess else "unknown"

    # 3) Volumes
    row["pumpedVolume"]   = _get_first_present(instrument_df, ["pumpedVolume", "measurementResults_pumped_volume"])
    row["analysedVolume"] = _get_first_present(instrument_df, ["analysedVolume", "measurementResults_analysed_volume"])
    row["halfPumpedVolume"] = row["pumpedVolume"]/2 if pd.notna(row["pumpedVolume"]) else np.nan

    # 4) Duration
    row["duration"] = _get_first_present(instrument_df, ["duration", "measurementResults_maximum_measurement_time_s"])

    # 5) Particles (from predictions)
    if predictions_df is not None:
        particleCount = len(predictions_df)
        row["particleCount"] = particleCount
        row["particleConcentration"] = (
            particleCount / row["pumpedVolume"] if pd.notna(row["pumpedVolume"]) and row["pumpedVolume"] > 0 else np.nan
        )
        row["particleRate"] = (
            particleCount / row["duration"] if pd.notna(row["duration"]) and row["duration"] > 0 else np.nan
        )
    else:
        row["particleCount"] = np.nan
        row["particleConcentration"] = np.nan
        row["particleRate"] = np.nan

    # 6) Key sensors we want to trend (store a single number per file)
    for key, candidates in SENSOR_MAP.items():
        row[key] = _get_first_present(instrument_df, candidates)

    return row

# ---------- Rolling stores ----------

def append_to_csv_rowwise(row, csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)
    return df

def append_class_counts(preds_df, file_id, when_ts, out_csv):
    """
    Maintain a long-format class counts table over time for stacked composition plots.
    Columns: start, file_id, class_label, count, proportion
    """
    if preds_df is None or "predicted_label" not in preds_df.columns:
        return pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(
            columns=["start", "file_id", "class_label", "count", "proportion"]
        )

    counts = preds_df["predicted_label"].value_counts().rename_axis("class_label").reset_index(name="count")
    total = counts["count"].sum()
    counts["proportion"] = counts["count"] / total if total > 0 else 0.0
    counts["file_id"] = file_id
    counts["start"] = when_ts

    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        df = pd.concat([prev, counts], ignore_index=True)
    else:
        df = counts
    df.to_csv(out_csv, index=False)
    return df

# ---------- Small utilities for plotting ----------

def _prepare_time_df(df):
    work = df.copy()
    if "start" in work.columns:
        work["start"] = pd.to_datetime(work["start"], errors="coerce", utc=True)
        work = work.sort_values("start")
        no_time = work["start"].isna().all()
        if no_time:
            work = work.reset_index(drop=True)
            work["__x__"] = work.index
        return work, no_time
    work = work.reset_index(drop=True)
    work["__x__"] = work.index
    return work, True

def _axes_date_or_seq(ax, no_time):
    ax.set_xlabel("Sequence" if no_time else "Date")

# ---------- Core plots (Volumes & Particle metrics) ----------

def plot_volumes(df, out_path):
    work, no_time = _prepare_time_df(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = "__x__" if no_time else "start"

    plotted = False
    for col, label in [("pumpedVolume", "Pumped"),
                       ("analysedVolume", "Analysed"),
                       ("halfPumpedVolume", "Half pumped")]:
        if col in work.columns and work[col].notna().any():
            ax.plot(work[x], work[col], label=label)
            plotted = True
    if not plotted:
        print('Data on volumes missing!')        
        plt.close(); return

    _axes_date_or_seq(ax, no_time)
    ax.set_ylabel("Volume (µL)")
    ax.set_title("Volumes over time")
    ax.legend()
    _safe_savefig(out_path)

def plot_particles(df, out_path):
    work, no_time = _prepare_time_df(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = "__x__" if no_time else "start"

    plotted = False
    for col, label in [("particleCount", "Count"),
                       ("particleConcentration", "Abundance (n/µL)"),
                       ("particleRate", "Rate (n/s)")]:
        if col in work.columns and work[col].notna().any():
            ax.plot(work[x], work[col], label=label)
            plotted = True
    if not plotted:
        print('Data on particle counts missing!')
        plt.close(); return

    _axes_date_or_seq(ax, no_time)
    ax.set_ylabel("Value")
    ax.set_title("Particle metrics over time")
    ax.legend()
    _safe_savefig(out_path)

# ---------- Per-file plots (pie + bar) ----------

def plot_class_pie_and_bar(predictions_df, file_prefix, out_dir):
    if predictions_df is None or "predicted_label" not in predictions_df.columns:
        return

    counts = predictions_df["predicted_label"].value_counts()
    if counts.empty:
        return

    # Pie
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%.1f%%")
    ax.set_title(f"Class proportions: {file_prefix}")
    _safe_savefig(os.path.join(out_dir, f"{file_prefix}_classProps.png"))

    # Bar with counts
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = counts.sort_values(ascending=False)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4C72B0")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.set_title(f"Class counts: {file_prefix}")
    plt.xticks(rotation=30, ha="right")
    _safe_savefig(os.path.join(out_dir, f"{file_prefix}_classCounts.png"))

# ---------- Class composition over time ----------

def plot_class_stacked_over_time(class_counts_df, out_path):
    if class_counts_df is None or class_counts_df.empty:
        return
    df = class_counts_df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    df = df.dropna(subset=["start"])
    if df.empty:
        return

    # pivot to wide, rows=time, cols=class, values=proportion
    wide = df.pivot_table(index="start", columns="class_label", values="proportion", aggfunc="mean").fillna(0.0)
    wide = wide.sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = wide.index
    y = wide.values.T  # classes x time
    labels = list(wide.columns)
    ax.stackplot(x, y, labels=labels, alpha=0.9)
    ax.set_title("Class composition over time (stacked)")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _safe_savefig(out_path)

# ---------- Daily medians ----------

def plot_daily_medians(qc_df, out_path):
    if qc_df is None or qc_df.empty:
        return
    work = qc_df.copy()
    work["start"] = pd.to_datetime(work["start"], errors="coerce", utc=True)
    work = work.dropna(subset=["start"])
    if work.empty:
        return

    work["date"] = work["start"].dt.date
    metrics = ["pumpedVolume", "analysedVolume", "particleCount", "particleConcentration", "particleRate"]
    avail = [m for m in metrics if m in work.columns]
    if not avail:
        return

    daily = work.groupby("date")[avail].median().reset_index()
    fig, axes = plt.subplots(nrows=math.ceil(len(avail)/2), ncols=2, figsize=(12, 6), squeeze=False)
    axes = axes.flatten()

    for i, m in enumerate(avail):
        ax = axes[i]
        ax.plot(daily["date"], daily[m], marker="o")
        ax.set_title(f"Daily median: {m}")
        ax.set_xlabel("Date"); ax.set_ylabel(m.replace("particle", "particle ").title())
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    _safe_savefig(out_path)

# ---------- Sensors panel with QC bands ----------

def plot_sensors_panel(qc_df, sensor_limits, out_path):
    if qc_df is None or qc_df.empty:
        return
    work, no_time = _prepare_time_df(qc_df)
    x = "__x__" if no_time else "start"

    sensors = [s for s in SENSOR_MAP.keys() if s in work.columns and work[s].notna().any()]
    if not sensors:
        print('Data from sensors missing!')
        return

    n = len(sensors)
    cols = 2
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, rows*3)), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for i, s in enumerate(sensors):
        ax = axes[i]
        ax.plot(work[x], work[s], color="#4C72B0")
        vmin, vmax = sensor_limits.get(s, (None, None)) if sensor_limits else (None, None)
        if (vmin is not None) or (vmax is not None):
            lo = vmin if vmin is not None else np.nanmin(work[s].values)
            hi = vmax if vmax is not None else np.nanmax(work[s].values)
            ax.axhspan(lo, hi, color="#C7E9C0", alpha=0.35, zorder=0)
            # Highlight outliers
            mask_low = pd.notna(work[s]) & (work[s] < lo)
            mask_hi  = pd.notna(work[s]) & (work[s] > hi)
            ax.scatter(work[x][mask_low], work[s][mask_low], color="#D62728", s=14, label="Below limit")
            ax.scatter(work[x][mask_hi],  work[s][mask_hi],  color="#D62728", s=14, label="Above limit")
        ax.set_title(s)
        if i % cols == 0:
            ax.set_ylabel("Value")
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    axes[max(0, min(i, len(axes)-1))].set_xlabel("Sequence" if no_time else "Date")
    fig.suptitle("Sensor diagnostics (with QC bands)", y=1.02)
    plt.tight_layout()
    _safe_savefig(out_path)

# ---------- Batch overview: grid of pies (last N files) ----------

def plot_batch_pies_grid(class_counts_df, out_path, last_n=12):
    if class_counts_df is None or class_counts_df.empty:
        return
    df = class_counts_df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    df = df.dropna(subset=["start"])
    if df.empty:
        return

    # last N files by timestamp
    files = (df.groupby(["file_id"])["start"]
             .max().sort_values(ascending=False).head(last_n).index.tolist())
    sub = df[df["file_id"].isin(files)].copy()
    order = (sub.groupby("file_id")["start"].max()
             .sort_values(ascending=True).index.tolist())
    sub["file_id"] = pd.Categorical(sub["file_id"], categories=order, ordered=True)

    n = len(order)
    cols = 4
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.atleast_1d(axes).flatten()

    for i, fid in enumerate(order):
        ax = axes[i]
        dd = sub[sub["file_id"] == fid]
        if dd.empty:
            ax.axis("off"); continue
        counts = dd.set_index("class_label")["count"].sort_values(ascending=False)
        ax.pie(counts.values, labels=counts.index, autopct="%.0f")
        ax.set_title(str(fid), fontsize=9)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Batch overview: class pies (last N files)", y=1.02)
    _safe_savefig(out_path)

# ---------- Main entry ----------

def update_after_file(instrument_csv, predictions_csv, plots_dir):
    inst = load_instrument_csv(instrument_csv)
    preds = load_predictions_csv(predictions_csv)

    if inst is None:
        print("[qc] No instrument.csv loaded; skipping QC plots.")
        return

    # --- Build rows and append to rolling stores ---
    row = build_measurement_row(inst, preds, instrument_csv_path=instrument_csv)

    qc_table_path = os.path.join(plots_dir, "qc_measurements.csv")
    qc_df = append_to_csv_rowwise(row, qc_table_path)

    file_prefix = os.path.basename(instrument_csv).replace("_instrument.csv", "")
    class_counts_path = os.path.join(plots_dir, "qc_class_counts.csv")
    class_df = append_class_counts(preds, file_prefix, row["start"], class_counts_path)

    # --- Limits for sensors (from current instrument file, if available) ---
    sensor_limits = _get_limits_from_instrument(inst)

    # --- Ensure subfolders ---
    os.makedirs(plots_dir, exist_ok=True)
    diag_dir = os.path.join(plots_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    # --- Generate plots (R-parity set) ---
    # Core time-series
    plot_volumes(qc_df, os.path.join(plots_dir, "Volumes.png"))
    plot_particles(qc_df, os.path.join(plots_dir, "particlePlots.png"))

    # Per-file pies and bars
    plot_class_pie_and_bar(preds, file_prefix, plots_dir)

    # Per-file FWS-L vs FWS-R scatter (w/ adaptive thinning)
    plot_fws_scatter(preds, file_prefix, plots_dir)
    
    # Class composition over time + batch grid
    plot_class_stacked_over_time(class_df, os.path.join(plots_dir, "classComposition_over_time.png"))
    plot_batch_pies_grid(class_df, os.path.join(plots_dir, "batch_pies_grid.png"), last_n=12)

    # Daily medians
    plot_daily_medians(qc_df, os.path.join(plots_dir, "daily_medians.png"))
    
    # Sensors panel with QC bands & outlier dots
    plot_sensors_panel(qc_df, sensor_limits, os.path.join(diag_dir, "sensors_panel.png"))

    print("[qc] QC plots updated.")
