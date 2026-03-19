# qc_plots.py — Full QC module (plots only, no reporting or metadata logic)
# Compatible with metadata_extraction.py and reporting_core.py
# Saves PNG plots next to the output JSON (directory pattern B)


from reporting_core import write_report_packet_flat, apply_sampling_protocol_mutations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from metadata_extraction import extract_metadata
from reporting_core import write_report_packet_flat
import os
from flowcytosender import send_to_dashboard
import json

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def _ensure_dir(p: str | Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_savefig(path: str | Path):
    path = Path(path)
    _ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _load_csv(path: str | Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ------------------------------------------------------------
# Rolling stores (for live QC only; safe no-ops if missing)
# ------------------------------------------------------------

def _append_row_rolling(row_dict: dict, csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([row_dict])
    df.to_csv(csv_path, index=False)
    return df


def _append_class_counts(preds_df: pd.DataFrame | None,
                         file_id: str,
                         when_ts,
                         out_csv: str | Path) -> pd.DataFrame:
    """Maintain long-format class counts over time.
    Columns: start, file_id, class_label, count, proportion
    """
    out_csv = Path(out_csv)
    if preds_df is None or "predicted_label" not in preds_df.columns:
        return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(
            columns=["start", "file_id", "class_label", "count", "proportion"]
        )
    counts = preds_df["predicted_label"].value_counts().rename_axis("class_label").reset_index(name="count")
    total = counts["count"].sum()
    counts["proportion"] = counts["count"] / total if total > 0 else 0.0
    counts["file_id"] = file_id
    counts["start"] = when_ts
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        df = pd.concat([prev, counts], ignore_index=True)
    else:
        df = counts
    df.to_csv(out_csv, index=False)
    return df


# ------------------------------------------------------------
# Sensors / limits helpers (optional shaded bands)
# ------------------------------------------------------------

SENSOR_MAP = {
    "absolutePressure": ["measurementResults_pressureAbsolute", "pressureAbsolute"],
    "differentialPressure": ["measurementResults_pressureDifferential", "pressureDifferential"],
    "sheathTemperature": ["measurementResults_sheathTemperature", "sheathTemperature"],
    "systemTemperature": ["measurementResults_systemTemperature", "systemTemperature"],
    "laserBeamWidth": ["laserBeamWidth", "measurementSettings_CytoSettings_LaserBeamWidth"],
    "sampleCoreSpeed": ["sampleCoreSpeed", "measurementSettings_CytoSettings_SampleCoreSpeed"],
    "laserTemperature": ["measurementResults_laserTemperature", "laserTemperature"],
    "PMTtemperature": ["measurementResults_PMTtemperature", "PMTtemperature"],
}

LIMIT_CANDIDATES = {
    "sheathTemperature": [
        ("measurementSettings_CytoSettings_SensorLimits_SheathTemp_minValue",
         "measurementSettings_CytoSettings_SensorLimits_SheathTemp_maxValue")
    ],
    "systemTemperature": [
        ("measurementSettings_CytoSettings_SensorLimits_SystemTemp_minValue",
         "measurementSettings_CytoSettings_SensorLimits_SystemTemp_maxValue")
    ],
    "absolutePressure": [
        ("measurementSettings_CytoSettings_SensorLimits_PressureAbs_minValue",
         "measurementSettings_CytoSettings_SensorLimits_PressureAbs_maxValue")
    ],
    "differentialPressure": [
        ("measurementSettings_CytoSettings_SensorLimits_PressureDiff_minValue",
         "measurementSettings_CytoSettings_SensorLimits_PressureDiff_maxValue")
    ],
    "laserTemperature": [
        ("measurementSettings_CytoSettings_SensorLimits_LaserTemp_minValue",
         "measurementSettings_CytoSettings_SensorLimits_LaserTemp_maxValue")
    ],
    "PMTtemperature": [
        ("measurementSettings_CytoSettings_SensorLimits_PMTTemp_minValue",
         "measurementSettings_CytoSettings_SensorLimits_PMTTemp_maxValue")
    ],
}



# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def _prepare_time_df(df: pd.DataFrame):
    work = df.copy()
    if "start" in work.columns:
        work["start"] = pd.to_datetime(work["start"], errors="coerce", utc=True)
        work = work.sort_values("start")
        no_time = work["start"].isna().all()
        if no_time:
            work = work.reset_index(drop=True)
            work["__x__"] = work.index
            return work, True
        work = work.reset_index(drop=True)
        return work, False
    # fallback sequential x
    work = work.reset_index(drop=True)
    work["__x__"] = work.index
    return work, True


def _axis_label_time(ax, no_time: bool):
    ax.set_xlabel("Sequence" if no_time else "Date")


# ------------------------------------------------------------
# Individual plots
# ------------------------------------------------------------

def plot_thinned_fws_scatter(preds_df: pd.DataFrame | None, out_path: str | Path,
                             left_col="Forward_Scatter_Left_total",
                             right_col="Forward_Scatter_Right_total",
                             max_points_cap: int = 60000):
    """Thinned scatter of FWS L vs R. Only thinned plot is saved (per user request)."""
    if preds_df is None or not all(c in preds_df.columns for c in [left_col, right_col]):
        return
    df = preds_df[[left_col, right_col]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n == 0:
        return
    # Adaptive thinning curve (log growth)
    base = 8000
    growth = 22000
    scale = 3000
    target = int(min(max_points_cap, max(base, base + growth * np.log10(n / scale + 1.0))))
    if target < n:
        p = target / n
        # reproducible thinning
        rng = np.random.default_rng(abs(hash((left_col, right_col))) % (2**32))
        keep = rng.random(n) < p
        df = df.loc[keep]
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(df[left_col].values, df[right_col].values,
               s=4, alpha=0.7, linewidths=0, color="#4C72B0")
    ax.set_xlabel("Forward Scatter (Left) — total")
    ax.set_ylabel("Forward Scatter (Right) — total")
    # Robust axis limits
    try:
        xlims = np.nanpercentile(df[left_col], [1, 99])
        ylims = np.nanpercentile(df[right_col], [1, 99])
        pad_x = 0.08 * (xlims[1] - xlims[0] + 1e-9)
        pad_y = 0.08 * (ylims[1] - ylims[0] + 1e-9)
        ax.set_xlim(xlims[0] - pad_x, xlims[1] + pad_x)
        ax.set_ylim(ylims[0] - pad_y, ylims[1] + pad_y)
    except Exception:
        pass
    # Reference y=x line
    try:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="#888", lw=1, ls="--", alpha=0.7)
    except Exception:
        pass
    ax.set_title("FWS L vs R (thinned)")
    _safe_savefig(out_path)


def plot_volumes_over_time(qc_df: pd.DataFrame | None, out_path: str | Path):
    if qc_df is None or qc_df.empty:
        return
    work, no_time = _prepare_time_df(qc_df)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = "__x__" if no_time else "start"
    plotted = False
    for col, label in [("pumpedVolume", "Pumped"), ("analysedVolume", "Analysed")]:
        if col in work.columns and work[col].notna().any():
            ax.plot(work[x], work[col], label=label)
            plotted = True
    if not plotted:
        plt.close()
        return
    _axis_label_time(ax, no_time)
    ax.set_ylabel("Volume (µL)")
    ax.set_title("Volumes over time")
    ax.legend()
    _safe_savefig(out_path)


def plot_particle_metrics_over_time(qc_df: pd.DataFrame | None, out_path: str | Path):
    if qc_df is None or qc_df.empty:
        return
    work, no_time = _prepare_time_df(qc_df)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = "__x__" if no_time else "start"
    plotted = False
    for col, label in [
        ("particleCount", "Count"),
        ("particleConcentration", "Abundance (n/µL)"),
        ("particleRate", "Rate (n/s)")
    ]:
        if col in work.columns and work[col].notna().any():
            ax.plot(work[x], work[col], label=label)
            plotted = True
    if not plotted:
        plt.close()
        return
    _axis_label_time(ax, no_time)
    ax.set_ylabel("Value")
    ax.set_title("Particle metrics over time")
    ax.legend()
    _safe_savefig(out_path)


def plot_daily_medians(qc_df: pd.DataFrame | None, out_path: str | Path):
    if qc_df is None or qc_df.empty:
        return
    work = qc_df.copy()
    work["start"] = pd.to_datetime(work.get("start"), errors="coerce", utc=True)
    work = work.dropna(subset=["start"]) if "start" in work.columns else work
    if work.empty:
        return
    work["date"] = work["start"].dt.date if "start" in work.columns else None
    metrics = ["pumpedVolume", "analysedVolume", "particleCount", "particleConcentration", "particleRate"]
    avail = [m for m in metrics if m in work.columns]
    if not avail:
        return
    daily = work.groupby("date")[avail].median(numeric_only=True).reset_index()
    n = len(avail)
    rows = math.ceil(n / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6), squeeze=False)
    axes = axes.flatten()
    for i, m in enumerate(avail):
        ax = axes[i]
        ax.plot(daily["date"], daily[m], marker="o")
        ax.set_title(f"Daily median: {m}")
        ax.set_xlabel("Date")
        ax.set_ylabel(m)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    _safe_savefig(out_path)


def plot_sensors_panel(qc_df: pd.DataFrame | None, sensor_limits: dict, out_path: str | Path):
    if qc_df is None or qc_df.empty:
        return
    work, no_time = _prepare_time_df(qc_df)
    x = "__x__" if no_time else "start"
    sensors = [s for s in SENSOR_MAP.keys() if s in work.columns and work[s].notna().any()]
    if not sensors:
        return
    n = len(sensors)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, rows * 3)), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for i, s in enumerate(sensors):
        ax = axes[i]
        ax.plot(work[x], work[s], color="#4C72B0")
        vmin, vmax = sensor_limits.get(s, (None, None)) if sensor_limits else (None, None)
        if vmin is not None or vmax is not None:
            lo = vmin if vmin is not None else np.nanmin(work[s].values)
            hi = vmax if vmax is not None else np.nanmax(work[s].values)
            ax.axhspan(lo, hi, color="#C7E9C0", alpha=0.35, zorder=0)
            # highlight outliers
            lo_mask = pd.notna(work[s]) & (work[s] < lo)
            hi_mask = pd.notna(work[s]) & (work[s] > hi)
            ax.scatter(work[x][lo_mask], work[s][lo_mask], color="#D62728", s=12)
            ax.scatter(work[x][hi_mask], work[s][hi_mask], color="#D62728", s=12)
        ax.set_title(s)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    axes[max(0, min(i, len(axes) - 1))].set_xlabel("Sequence" if no_time else "Date")
    fig.suptitle("Sensor diagnostics (with QC bands)", y=1.02)
    plt.tight_layout()
    _safe_savefig(out_path)


def plot_class_pie_and_bar(preds_df: pd.DataFrame | None,
                           pie_out: str | Path,
                           bar_out: str | Path):
    if preds_df is None or "predicted_label" not in preds_df.columns:
        return
    counts = preds_df["predicted_label"].value_counts()
    if counts.empty:
        return
    # Pie
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%.1f%%")
    ax.set_title("Class proportions")
    _safe_savefig(pie_out)
    # Bar
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = counts.sort_values(ascending=False)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4C72B0")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.set_title("Class counts")
    plt.xticks(rotation=30, ha="right")
    _safe_savefig(bar_out)


def plot_class_stacked_over_time(class_counts_df: pd.DataFrame | None, out_path: str | Path):
    if class_counts_df is None or class_counts_df.empty:
        return
    df = class_counts_df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    df = df.dropna(subset=["start"])
    if df.empty:
        return
    wide = df.pivot_table(index="start", columns="class_label", values="proportion", aggfunc="mean").fillna(0.0)
    wide = wide.sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = wide.index
    y = wide.values.T
    labels = list(wide.columns)
    ax.stackplot(x, y, labels=labels, alpha=0.9)
    ax.set_title("Class composition over time (stacked)")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _safe_savefig(out_path)


def plot_batch_pies_grid(class_counts_df: pd.DataFrame | None, out_path: str | Path, last_n: int = 12):
    if class_counts_df is None or class_counts_df.empty:
        return
    df = class_counts_df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    df = df.dropna(subset=["start"])
    if df.empty:
        return
    # last N files by latest timestamp
    files = (
        df.groupby(["file_id"])['start'].max().sort_values(ascending=False).head(last_n).index.tolist()
    )
    sub = df[df["file_id"].isin(files)].copy()
    order = (
        sub.groupby("file_id")["start"].max().sort_values(ascending=True).index.tolist()
    )
    sub["file_id"] = pd.Categorical(sub["file_id"], categories=order, ordered=True)
    n = len(order)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).flatten()
    for i, fid in enumerate(order):
        ax = axes[i]
        dd = sub[sub["file_id"] == fid]
        if dd.empty:
            ax.axis("off"); continue
        counts = dd.set_index("class_label")["count"].sort_values(ascending=False)
        ax.pie(counts.values, labels=counts.index, autopct="%.0f")
        ax.set_title(str(fid), fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Batch overview: class pies (last N files)", y=1.02)
    _safe_savefig(out_path)


def plot_saturation(preds_df: pd.DataFrame | None, out_path: str | Path, threshold: float = 0.8,
                    channels=(
                        ("Fl_Yellow_average", "Fl_Yellow_maximum", "Yellow"),
                        ("Fl_Orange_average", "Fl_Orange_maximum", "Orange"),
                        ("Fl_Red_average", "Fl_Red_maximum", "Red"),
                    ), max_points_cap: int = 60000):
    """Plot % of max (avg/max) for each fluorescence channel in a stacked figure.
    Uses adaptive thinning per channel. Saves a single combined PNG.
    """
    if preds_df is None:
        return
    per_channel = []
    for avg_col, max_col, label in channels:
        if avg_col in preds_df.columns and max_col in preds_df.columns:
            df = preds_df[[avg_col, max_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) == 0:
                continue
            ratio = pd.to_numeric(df[avg_col], errors="coerce") / pd.to_numeric(df[max_col], errors="coerce").replace(0, np.nan)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if ratio.empty:
                continue
            per_channel.append((label, ratio.values))
    if not per_channel:
        return
    rows = len(per_channel)
    fig, axes = plt.subplots(rows, 1, figsize=(7, 3.5 * rows), squeeze=False)
    axes = axes.flatten()
    for ax, (label, vals) in zip(axes, per_channel):
        N = len(vals)
        # thinning
        base, growth, scale = 8000, 22000, 3000
        target = int(min(max_points_cap, max(base, base + growth * np.log10(N / scale + 1.0))))
        if target < N:
            p = target / N
            rng = np.random.default_rng(abs(hash((label, "sat"))) % (2**32))
            keep = rng.random(N) < p
            vals = vals[keep]
        x = np.arange(len(vals))
        ax.scatter(x, vals, s=4, alpha=0.7, color="#4C72B0")
        ax.axhline(threshold, color="red", lw=1.2, ls="--", alpha=0.8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Average / Maximum")
        ax.set_title(f"{label} (% of max): n={N:,} → shown={len(vals):,}")
    axes[-1].set_xlabel("Particle index (thinned)")
    fig.suptitle("% of Max Fluorescence Signal", y=1.02)
    _safe_savefig(out_path)


# ------------------------------------------------------------
# Health summary (lightweight, traffic lights)
# ------------------------------------------------------------

def _median_ratio(preds_df: pd.DataFrame | None, num_col: str, den_col: str) -> float | None:
    if preds_df is None or num_col not in preds_df.columns or den_col not in preds_df.columns:
        return None
    tmp = preds_df[[num_col, den_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return None
    r = pd.to_numeric(tmp[num_col], errors="coerce") / pd.to_numeric(tmp[den_col], errors="coerce").replace(0, np.nan)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return None
    return float(np.median(r))


def _health_flags(preds_df: pd.DataFrame | None,
                  meta_row: dict,
                  sensor_limits: dict) -> tuple[dict, bool, list]:
    """Return (health_dict, overall_ok, failed_list)."""
    health, fails = {}, []
    # 1) analysedVolume
    analysed = meta_row.get("analysedVolume")
    ok = analysed is not None and analysed > 3000  # threshold per user discussion
    health["analysedVolume"] = {"value": analysed, "ok": ok, "rule": "> 3000 µL"}
    if not ok: fails.append("Analysed volume")
    # 2) total events
    total_events = meta_row.get("particleCount")
    ok = total_events is not None and total_events >= 5000
    health["totalEvents"] = {"value": total_events, "ok": ok, "rule": "≥ 5000"}
    if not ok: fails.append("Total events")
    # 3) particle rate
    rate = meta_row.get("particleRate")
    lo, hi = 5, 5000
    ok = rate is not None and (lo <= rate <= hi)
    health["particlesPerSec"] = {"value": rate, "ok": ok, "rule": f"{lo}–{hi} 1/s"}
    if not ok: fails.append("Particles/sec")
    # 4) FWS ratio
    fws_ratio = _median_ratio(preds_df, "Forward_Scatter_Right_total", "Forward_Scatter_Left_total")
    ok = fws_ratio is not None and (0.75 <= fws_ratio <= 1.25)
    health["FWSR_over_FWSL"] = {"value": fws_ratio, "ok": ok, "rule": "0.75–1.25 (median)"}
    if not ok: fails.append("FWSR/FWSL ratio")
    # 5) Temperatures: laser / sheath / PMT / system
    for metric, fallback in [
        ("laserTemperature", (10.0, 45.0)),
        ("sheathTemperature", (5.0, 35.0)),
        ("PMTtemperature", (5.0, 55.0)),
        ("systemTemperature", (5.0, 55.0)),
    ]:
        val = meta_row.get(metric)
        vmin, vmax = None, None
        if sensor_limits and metric in sensor_limits:
            vmin, vmax = sensor_limits.get(metric, (None, None))
        if vmin is None or vmax is None:
            vmin, vmax = fallback
        ok = True
        if val is not None and vmin is not None and val < vmin:
            ok = False
        if val is not None and vmax is not None and val > vmax:
            ok = False
        health[metric] = {"value": val, "ok": ok, "rule": f"{vmin}–{vmax} °C"}
        if not ok:
            label = metric.replace("Temperature", " Temperature").replace("PMT", "PMT ")
            fails.append(label.strip())
    # 6) flat signal fraction (from metadata_extraction if provided there)
    flat_frac = meta_row.get("flatSignalFraction")
    if flat_frac is not None:
        ok = flat_frac <= 0.10
        health["flatSignalFrac"] = {"value": flat_frac, "ok": ok, "rule": "≤ 10% with avg/max ≥ 0.8"}
        if not ok: fails.append("% flat signal")
    overall_ok = len(fails) == 0
    return health, overall_ok, fails


def _append_health_row(start_ts, file_id, health: dict, overall_ok: bool, csv_path: str | Path) -> pd.DataFrame:
    """Append a flattened health row into qc_health.csv as a rolling store."""
    csv_path = Path(csv_path)
    flat = {
        "start": start_ts,
        "file_id": file_id,
        "overall_ok": bool(overall_ok),
    }
    for k, d in (health or {}).items():
        flat[f"{k}_value"] = d.get("value")
        flat[f"{k}_ok"] = bool(d.get("ok"))
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([flat])], ignore_index=True)
    else:
        df = pd.DataFrame([flat])
    df.to_csv(csv_path, index=False)
    return df


def plot_health_summary(health: dict, overall_ok: bool, failures: list, out_path: str | Path):
    if not health:
        return
    items = [
        ("Analysed volume", "analysedVolume", "µL"),
        ("Total events", "totalEvents", ""),
        ("Particles/sec", "particlesPerSec", "1/s"),
        ("FWSR/FWSL", "FWSR_over_FWSL", ""),
        ("Laser temp", "laserTemperature", "°C"),
        ("Sheath temp", "sheathTemperature", "°C"),
        ("PMT temp", "PMTtemperature", "°C"),
        ("System temp", "systemTemperature", "°C"),
        ("% flat signal", "flatSignalFrac", ""),
    ]
    rows = [(label, key, unit) for (label, key, unit) in items if key in health]
    fig, ax = plt.subplots(figsize=(9.5, 0.7 + 0.45 * len(rows)))
    ax.axis("off")
    y0, dy = 1.0, 0.08
    for i, (label, key, unit) in enumerate(rows):
        rec_y = y0 - (i + 1) * dy
        ok = bool(health[key]["ok"])
        color = "#4CAF50" if ok else "#E74C3C"
        ax.add_patch(plt.Rectangle((0.01, rec_y - 0.035), 0.02, 0.06, color=color, transform=ax.transAxes))
        ax.text(0.05, rec_y, label, fontsize=10, va="center", transform=ax.transAxes)
        v = health[key]["value"]
        v_str = "—"
        if v is not None:
            if key == "flatSignalFrac":
                v_str = f"{100 * v:.1f}%"
            elif isinstance(v, float):
                v_str = f"{v:,.3g} {unit}".strip()
            else:
                v_str = f"{v} {unit}".strip()
        rule = health[key]["rule"]
        ax.text(0.42, rec_y, v_str, fontsize=10, va="center", transform=ax.transAxes)
        ax.text(0.64, rec_y, rule, fontsize=9, va="center", color="#555", transform=ax.transAxes)
    header = f"HEALTH: {'GREEN' if overall_ok else 'RED'}"
    subtitle = "" if overall_ok else ("Failed: " + ", ".join(failures[:5]) + ("…" if len(failures) > 5 else ""))
    ax.text(0.01, y0, header, fontsize=12, weight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.01, y0 - 0.08, subtitle, fontsize=9, color="#E74C3C", transform=ax.transAxes)
    _safe_savefig(out_path)


# ------------------------------------------------------------
# Orchestrator: end-to-end for a single file
# ------------------------------------------------------------

def extract_sensor_limits_from_json(instrument):
    limits = {}
    settings = instrument.get("measurementSettings", {}).get("CytoSettings", {})
    sensor_limits = settings.get("SensorLimits", {})

    mapping = {
        "sheathTemperature": ("SheathTemp_minValue", "SheathTemp_maxValue"),
        "systemTemperature": ("SystemTemp_minValue", "SystemTemp_maxValue"),
        "absolutePressure": ("PressureAbs_minValue", "PressureAbs_maxValue"),
        "differentialPressure": ("PressureDiff_minValue", "PressureDiff_maxValue"),
        "laserTemperature": ("LaserTemp_minValue", "LaserTemp_maxValue"),
        "PMTtemperature": ("PMTTemp_minValue", "PMTTemp_maxValue"),
    }

    for key, (mn, mx) in mapping.items():
        vmin = sensor_limits.get(mn)
        vmax = sensor_limits.get(mx)
        if vmin is not None or vmax is not None:
            limits[key] = (
                float(vmin) if vmin is not None else None,
                float(vmax) if vmax is not None else None,
            )

    return limits

def update_after_file(
    cyz_json_path,
    predictions_csv,
    out_dir,
    grablist_path="grablist.txt",
) -> dict:
    """
    Full QC pipeline for a single processed file:
      - Load CSVs
      - Extract metadata (stateless)
      - Maintain rolling stores (qc_measurements.csv, qc_class_counts.csv, qc_health.csv)
      - Generate ALL QC plots (PNG) next to the output JSON (pattern B)
      - Build and write the flat reporting packet (report_packet.json)

    Returns: the reporting packet dict (also written to disk)
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    with open(cyz_json_path) as f:
        full_js = json.load(f)

    instrument = full_js.get("instrument", {})
    
    
    

    # ------------------------------------------------------------
    # ENFORCE MODEL–FILE COMPATIBILITY: serial number + PMT string
    # ------------------------------------------------------------
    # Load the model's train settings (from the pinned model folder)
    model_settings_path = os.path.join(
        os.path.expanduser("~/Documents/flowcytometertool/selectedvalidappliedmodel"),
        "modeltrainsettings.json"
    )
    if not os.path.exists(model_settings_path):
        raise RuntimeError(
            f"Expected modeltrainsettings.json at {model_settings_path}, "
            "but it was not found."
        )
    with open(model_settings_path, "r") as mf:
        model_settings = json.load(mf)
        
        
        
    expected_serial = model_settings["instrument"]["serialNumber"]
    expected_pmt    = model_settings["instrument"]["measurementSettings"]["CytoSettings"]["PMTlevels_str"]
    # Extract actual values from the live JSON
    actual_serial = instrument.get("serialNumber")
    actual_pmt = (
        instrument.get("measurementSettings", {})
                  .get("CytoSettings", {})
                  .get("PMTlevels_str")
    )

    # Compare – raise immediately if mismatch
    if expected_serial is not None and actual_serial != expected_serial:
        print(f"Modelsettings mismatch: stopping watcher."
            f"Serial number mismatch.\n"
            f" Model expects: {expected_serial}\n"
            f" Incoming file: {actual_serial}")
        raise SystemExit(        )
    if expected_pmt is not None and actual_pmt != expected_pmt:
        print(f"Modelsettings mismatch: stopping watcher."
            "PMT settings mismatch.\n"
            f" Model expects: {expected_pmt}\n"
            f" Incoming file: {actual_pmt}")
        raise SystemExit(        )
        
    print(f" Model expects: {expected_serial}\n")
    print(f" Incoming file: {actual_serial}")
    print(f" Model expects: {expected_pmt}\n")
    print(f" Incoming file: {actual_pmt}")
    
    measurementResults = instrument.get("measurementResults", {})
    CytoSettings = instrument.get("measurementSettings", {}).get("CytoSettings", {})

    sensor_limits = extract_sensor_limits_from_json(instrument)


    preds_df = _load_csv(predictions_csv) if predictions_csv else None

    file_prefix = Path(cyz_json_path).stem.replace(".cyz", "")
    json_path = cyz_json_path    
    
    # 1) metadata (stateless)
    meta = extract_metadata(cyz_json_path=json_path, predictions_df=preds_df)
    
    qc_meas_csv = out_dir / "qc_measurements.csv"
    qc_class_csv = out_dir / "qc_class_counts.csv"



    

    # Build a measurement row (consistent with meta + a few extras)
    meas_row = {
        "start": meta.get("start"),
        "file_id": file_prefix,
        "triggerLevel": meta.get("triggerLevel"),
        "pumpedVolume": meta.get("pumpedVolume"),
        "analysedVolume": meta.get("analysedVolume"),
        "particleCount": meta.get("particleCount"),
        "particleRate": meta.get("particleRate"),
        "particleConcentration": meta.get("particleConcentration"),
        # Sensors
        "absolutePressure": meta.get("absolutePressure"),
        "differentialPressure": meta.get("differentialPressure"),
        "sheathTemperature": meta.get("sheathTemperature"),
        "systemTemperature": meta.get("systemTemperature"),
        "laserTemperature": meta.get("laserTemperature"),
        "PMTtemperature": meta.get("PMTtemperature"),
        "sampleCoreSpeed": meta.get("sampleCoreSpeed"),
        "laserBeamWidth": meta.get("laserBeamWidth"),
    }
    qc_df = _append_row_rolling(meas_row, qc_meas_csv)
    class_df = _append_class_counts(preds_df, file_prefix, meas_row["start"], qc_class_csv)


    # 4) Plots — save PNGs next to the output JSON (pattern B)
    paths = {
        "scatter": out_dir / "scatter.png",
        "volumes": out_dir / "volumes.png",
        "particle_metrics": out_dir / "particle_metrics.png",
        "daily_medians": out_dir / "daily_medians.png",
        "sensors": out_dir / "sensors.png",
        "saturation": out_dir / "saturation.png",
        "class_pie": out_dir / "class_pie.png",
        "class_counts": out_dir / "class_counts.png",
        "class_composition_over_time": out_dir / "class_composition_over_time.png",
        "batch_pies_grid": out_dir / "batch_pies_grid.png",
        "health_summary": out_dir / "health_summary.png",
    }

    # Core plots
    plot_thinned_fws_scatter(preds_df, paths["scatter"])  # thinned only
    plot_volumes_over_time(qc_df, paths["volumes"])  # pumped & analysed
    plot_particle_metrics_over_time(qc_df, paths["particle_metrics"])  # rate, conc, count
    plot_daily_medians(qc_df, paths["daily_medians"])  # daily medians panel
    plot_sensors_panel(qc_df, sensor_limits, paths["sensors"])  # sensors panel
    plot_saturation(preds_df, paths["saturation"])  # %max combined figure
    plot_class_pie_and_bar(preds_df, paths["class_pie"], paths["class_counts"])  # pies + bars
    plot_class_stacked_over_time(class_df, paths["class_composition_over_time"])  # stacked area
    plot_batch_pies_grid(class_df, paths["batch_pies_grid"], last_n=12)

    # 4b) Health flags + rolling health + health summary plot
    health_csv = out_dir / "qc_health.csv"
    health, overall_ok, failures = _health_flags(preds_df, meta, sensor_limits)
    _append_health_row(meta.get("start"), file_prefix, health, overall_ok, health_csv)
    plot_health_summary(health, overall_ok, failures, paths["health_summary"])

    # 5) Reporting packet (flat) next to plots
    packet_json = out_dir / "report_packet.json"
    plot_paths_rel = {k: str(Path(v).name) for k, v in paths.items() if Path(v).exists()}

    model_settings_dir = Path(model_settings_path).parent
    model_version = str(list(model_settings_dir.glob("*.pkl"))[0])

    packet = write_report_packet_flat(
        metadata=meta,
        modelsettings=model_settings,
        predictions_df=preds_df,
        grablist_path=str(grablist_path) if grablist_path else None,
        json_path=json_path,
        plot_paths=plot_paths_rel,
        output_path=str(packet_json),
        modelversion = model_version
    )
    

    apply_sampling_protocol_mutations(packet)

    with open(packet_json, "w") as f:
        json.dump(packet, f, indent=2)
    
    
    print(packet)
    send_to_dashboard(packet)
    
    
    
    
    
    
    return packet
