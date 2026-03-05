# qc_plots.py
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# Utility: Safe load of instrument.csv and predictions.csv
# -------------------------------------------------------------
def load_instrument_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"[qc] Failed reading instrument CSV: {e}")
        return None

def load_predictions_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


# --- Add this helper above build_measurement_row ---
from datetime import datetime
import os

def _pick_timestamp(instrument_df, instrument_csv_path=None):
    """
    Find a timestamp column in instrument_df and return a pandas.Timestamp.
    Fallback to file mtime if none found.
    """
    # Candidate columns we often see after JSON flattening
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
            # Try direct pandas parse first
            ts = pd.to_datetime(val, errors="coerce", utc=True)
            if pd.notna(ts):
                return ts

            # If it's numeric epoch seconds (rare), convert
            try:
                if pd.notna(val) and str(val).strip() != "":
                    as_float = float(val)
                    if as_float > 10_000:  # naive guard against tiny values
                        return pd.to_datetime(as_float, unit="s", utc=True)
            except Exception:
                pass

    # Fallback: file mtime (still gives chronological plots)
    if instrument_csv_path and os.path.exists(instrument_csv_path):
        mtime = os.path.getmtime(instrument_csv_path)
        return pd.to_datetime(mtime, unit="s", utc=True)

    return pd.NaT  

def build_measurement_row(instrument_df, predictions_df, instrument_csv_path=None):
    row = {}

    # 1) Timestamp (robust detection)
    row["start"] = _pick_timestamp(instrument_df, instrument_csv_path)

    # 2) Trigger level (best-effort)
    # prefer normalized names, else search any column that contains 'trigger' (case-insensitive)
    if "triggerLevel" in instrument_df.columns:
        row["triggerLevel"] = instrument_df["triggerLevel"].iloc[0]
    else:
        guess = [c for c in instrument_df.columns if "trigger" in c.lower()]
        row["triggerLevel"] = instrument_df[guess[0]].iloc[0] if guess else "unknown"

    # 3) Volumes (try unprefixed first, then measurementResults_* variants)
    def _get_num(*names):
        for n in names:
            if n in instrument_df.columns:
                try:
                    return float(instrument_df[n].iloc[0])
                except Exception:
                    return np.nan
        return np.nan

    row["pumpedVolume"]   = _get_num("pumpedVolume",   "measurementResults_pumped_volume")
    row["analysedVolume"] = _get_num("analysedVolume", "measurementResults_analysed_volume")
    row["halfPumpedVolume"] = row["pumpedVolume"] / 2 if pd.notna(row["pumpedVolume"]) else np.nan

    # 4) Duration (seconds)
    row["duration"] = _get_num("duration", "measurementResults_maximum_measurement_time_s")

    # 5) Particles (from predictions_df if available)
    if predictions_df is not None:
        particleCount = len(predictions_df)
        row["particleCount"] = particleCount
        row["particleConcentration"] = (
            (particleCount / row["pumpedVolume"])
            if pd.notna(row["pumpedVolume"]) and row["pumpedVolume"] > 0 else np.nan
        )
        row["particleRate"] = (
            (particleCount / row["duration"])
            if pd.notna(row["duration"]) and row["duration"] > 0 else np.nan
        )
    else:
        row["particleCount"] = np.nan
        row["particleConcentration"] = np.nan
        row["particleRate"] = np.nan

    return row


# -------------------------------------------------------------
# Append row to a rolling qc_measurements.csv file
# -------------------------------------------------------------
def append_to_qc_table(row, qc_table_path):
    if os.path.exists(qc_table_path):
        df = pd.read_csv(qc_table_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(qc_table_path, index=False)
    return df


def _prepare_time_df(df):
    work = df.copy()
    if "start" in work.columns:
        work["start"] = pd.to_datetime(work["start"], errors="coerce", utc=True)
        work = work.sort_values("start")
        # Drop rows with missing timestamps for the time-series
        no_time = work["start"].isna().all()
        if no_time:
            # fabricate an index-based x-axis for a fallback view
            work = work.reset_index(drop=True)
            work["__x__"] = work.index
        return work, no_time
    # no start column at all -> index fallback
    work = work.reset_index(drop=True)
    work["__x__"] = work.index
    return work, True

def plot_volumes(df, out_path):
    work, no_time = _prepare_time_df(df)

    plt.figure(figsize=(10, 5))
    x = "__x__" if no_time else "start"
    any_plotted = False
    for col in ["pumpedVolume", "analysedVolume", "halfPumpedVolume"]:
        if col in work.columns and work[col].notna().any():
            plt.plot(work[x], work[col], label=col); any_plotted = True

    if not any_plotted:
        plt.close(); return

    plt.xlabel("Sequence" if no_time else "Date")
    plt.ylabel("Volume (µL)")
    plt.title("Pumped / Analysed / Half-pumped volumes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def plot_particles(df, out_path):
    work, no_time = _prepare_time_df(df)

    plt.figure(figsize=(10, 6))
    x = "__x__" if no_time else "start"
    plotted = False
    for ycol, label in [
        ("particleCount", "Count"),
        ("particleConcentration", "Abundance (n/µL)"),
        ("particleRate", "Rate (n/s)"),
    ]:
        if ycol in work.columns and work[ycol].notna().any():
            plt.plot(work[x], work[ycol], label=label); plotted = True

    if not plotted:
        plt.close(); return

    plt.xlabel("Sequence" if no_time else "Date")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Particle metrics through time")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

# -------------------------------------------------------------
# Plot 3: Per-file class proportion pie chart
# -------------------------------------------------------------
def plot_class_pies(predictions_df, file_prefix, out_dir):
    if predictions_df is None:
        return
    if "predicted_label" not in predictions_df.columns:
        return

    counts = predictions_df["predicted_label"].value_counts()
    labels = counts.index
    values = counts.values

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct="%.1f%%")
    plt.title(f"Class proportions: {file_prefix}")
    plt.savefig(os.path.join(out_dir, f"{file_prefix}_classProps.png"))
    plt.close()


# -------------------------------------------------------------
# Plot 4: Other sensor measurements
# -------------------------------------------------------------
def plot_other_measurements(df, out_dir):
    ignore = {
        "start", "triggerLevel",
        "pumpedVolume", "analysedVolume", "halfPumpedVolume",
        "particleCount", "particleConcentration", "particleRate",
        "duration"
    }

    sensor_cols = [c for c in df.columns if c not in ignore and df[c].dtype in [int, float, np.float64, np.int64]]
    for col in sensor_cols:
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(df["start"], df[col])
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.title(col)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{col}.png"))
            plt.close()
        except Exception:
            pass


# -------------------------------------------------------------
# Main entry from watcher
# -------------------------------------------------------------
def update_after_file(instrument_csv, predictions_csv, plots_dir):
    inst = load_instrument_csv(instrument_csv)
    preds = load_predictions_csv(predictions_csv)

    if inst is None:
        print("[qc] No instrument.csv loaded; skipping QC plots.")
        return

    row = build_measurement_row(inst, preds, instrument_csv_path=instrument_csv)

    # Rolling QC database
    qc_table_path = os.path.join(plots_dir, "qc_measurements.csv")
    df = append_to_qc_table(row, qc_table_path)

    # Make plots directory if needed
    vol_dir = plots_dir
    other_dir = os.path.join(plots_dir, "other_measurements")
    os.makedirs(other_dir, exist_ok=True)

    # Generate plots
    plot_volumes(df, os.path.join(vol_dir, "Volumes.png"))
    plot_particles(df, os.path.join(vol_dir, "particlePlots.png"))

    # Per-file pies
    file_prefix = os.path.basename(instrument_csv).replace("_instrument.csv", "")
    plot_class_pies(preds, file_prefix, plots_dir)

    # Other sensors
    plot_other_measurements(df, other_dir)

    print("[qc] QC plots updated.")