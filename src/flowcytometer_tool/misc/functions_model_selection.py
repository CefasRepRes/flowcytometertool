# functions file written whilst writing flow_cytometer_tool.py
import time
import traceback
import requests
import subprocess
import os
import json
import pandas as pd
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import tkinter as tk
import csv
from flowcytometer_tool.tabs.download_train.listmode import extract
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import shutil
from tkinter import simpledialog, ttk
from azure.storage.blob import ContainerClient, BlobServiceClient
import joblib
import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import zipfile
import re
from urllib.parse import urlparse
import argparse
import platform
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tkinter import filedialog
import tempfile
import sys
import time
import webbrowser
from collections import Counter
import flowcytometer_tool.tabs.continuous_sample_analyser.qc_plots as qc_plots
import json
from flowcytometer_tool.tabs.blob_tools.auth import get_credential
import hashlib
import glob
import xml.etree.ElementTree as ET
from flowcytometer_tool.misc.xmlfunctions import (
    parse_gate_setlist,
    assign_classes_from_gates,
    convert_selected_cyzs_to_listmode,
    build_consensual_dataset_from_cyz_xmls,
    build_consensual_dataset_from_selected_cyzs_and_xmls,
    load_file
)
from flowcytometer_tool.misc.normalise_training_person_name import _normalise_training_person_name
from flowcytometer_tool.misc.person_to_weight_from_expertise_levels import _person_to_weight_from_expertise_levels
from flowcytometer_tool.misc.compute_consensual_labels_and_sample_weights import _compute_consensual_labels_and_sample_weights
from flowcytometer_tool.misc.bead_training import (
    prepare_training_dataframe_with_optional_bead_calibration,
    beadcalibrated_model_path,
    update_modeltrainsettings_bead_flag,
)
from flowcytometer_tool.tabs.continuous_sample_analyser.json_safe import json_safe
 
from flowcytometer_tool.tabs.continuous_sample_analyser.bead_calibration import (
    load_latest_beadscalibration_record,
    apply_saved_bead_calibration_to_dataframe,
)

from flowcytometer_tool.tabs.continuous_sample_analyser.protocols import detect_sampling_protocol
import yaml
__all__ = [
    "append_fwscalibration_record",
    "init_label_change_log",
    "record_label_merge",
    "record_label_delete",
    "load_app_config",
    "save_app_config",
    "active_model_dir",
    "resolve_active_model_path",
    "list_available_model_versions",
    "download_model_version",
    "set_active_model",
    "set_active_uncalibrated_model",
    "set_active_beadcalibrated_model",
    "resolve_active_raw_model_path",
    "resolve_active_beadcalibrated_model_path",
    "compute_fws_binned_calibration_from_df",
]


if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")


# === Active Model Configuration & Selection (new) ============================
from pathlib import Path
import shutil

# Optional YAML (falls back to JSON-like dump if PyYAML not present)
try:
    import yaml
except Exception:
    yaml = None

from flowcytometer_tool.tabs.blob_tools.storage_clients import _split_blob_url, get_container_client, get_blob_client  # existing helpers
from flowcytometer_tool.config.runtime import get_runtime_config

expertise_matrix_path = os.path.join(base_path, "..", "matrices", "expertise_matrix.csv")

# Locations
_RUNTIME_CONFIG = get_runtime_config()
_TOOL_DIR = _RUNTIME_CONFIG.paths.tool_dir
_SELECTED_UNCALIBRATED_MODEL_DIR = _RUNTIME_CONFIG.paths.selected_uncalibrated_model_dir
_SELECTED_BEADCALIBRATED_MODEL_DIR = _RUNTIME_CONFIG.paths.selected_beadcalibrated_model_dir
# Compatibility alias for older callers: the generic selected model slot now means raw/uncalibrated.
_SELECTED_MODEL_DIR = _SELECTED_UNCALIBRATED_MODEL_DIR
_CONFIG_PATH = _RUNTIME_CONFIG.paths.config_path

# Default trained-models container (adjust if you use a different one)
_DEFAULT_TRAINED_MODELS_CONTAINER = _RUNTIME_CONFIG.options.default_trained_models_container_url

import json, datetime, os
from pathlib import Path

import math

def _fit_slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope for y = slope*x with intercept forced to 0."""
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)

def _r2_for_origin_fit(x: np.ndarray, y: np.ndarray, slope: float) -> float:
    """R² for y ~ slope*x (intercept 0). Uses standard 1 - SSE/SST."""
    if not np.isfinite(slope):
        return float("nan")
    yhat = slope * x
    sse = float(np.sum((y - yhat) ** 2))
    ybar = float(np.mean(y))
    sst = float(np.sum((y - ybar) ** 2))
    if sst <= 0:
        return float("nan")
    return float(1.0 - sse / sst)

def _robust_trim_mask(x: np.ndarray, y: np.ndarray, lo=0.01, hi=0.99) -> np.ndarray:
    """
    Trim extremes based on y quantiles (diameter) and x quantiles (FWS),
    returning a boolean mask. Keeps central [lo,hi] of both distributions.
    """
    if len(x) < 10:
        return np.ones(len(x), dtype=bool)

    x_lo, x_hi = np.quantile(x, [lo, hi])
    y_lo, y_hi = np.quantile(y, [lo, hi])
    return (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)

def compute_fws_binned_calibration_from_df(
    df: pd.DataFrame,
    *,
    fws_col: str = "FWS_total",
    diam_um_col: str = "img_equiv_diameter_um",
    bin_edges_um=None,            # e.g. np.arange(0, 21, 1)
    min_per_bin: int = 8,
    diagnostic_png_path: str | None = None,
    max_scatter_points: int = 200_000,
    errorbar: str = "sd",         # "sd" or "se"
) -> dict | None:
    """
    Binned calibration with WITHIN-BIN robust outlier removal (log-MAD).
    """

    print("[FWSCAL BINNED] Starting binned FWS calibration")

    if df is None or df.empty:
        print("[FWSCAL BINNED] ❌ Input df is None or empty")
        return None

    if fws_col not in df.columns or diam_um_col not in df.columns:
        print("[FWSCAL BINNED] ❌ Missing required columns")
        return None

    if bin_edges_um is None:
        bin_edges_um = np.arange(0.0, 21.0, 1.0)
        print("[FWSCAL BINNED] Using default bin_edges_um:", bin_edges_um)

    print(f"[FWSCAL BINNED] min_per_bin = {min_per_bin}")
    print(f"[FWSCAL BINNED] errorbar mode = {errorbar}")

    d = pd.to_numeric(df[diam_um_col], errors="coerce")
    f = pd.to_numeric(df[fws_col], errors="coerce")

    ok = np.isfinite(d) & np.isfinite(f) & (d >= 0) & (f > 0)
    n_ok = int(ok.sum())

    print(f"[FWSCAL BINNED] Valid points after basic filtering: {n_ok}")

    if n_ok < min_per_bin:
        print("[FWSCAL BINNED] ❌ Not enough valid points overall")
        return None

    d = d[ok]
    f = f[ok]

    print(
        "[FWSCAL BINNED] Diameter range used: "
        f"{float(d.min()):.3f} – {float(d.max()):.3f} µm"
    )

    # --------------------------------------------------
    # Bin assignment
    # --------------------------------------------------
    bins = pd.cut(d, bins=bin_edges_um, right=False, include_lowest=True)
    print("[FWSCAL BINNED] Binning complete")

    out_bins = []
    total_bins_seen = 0
    total_bins_kept = 0
    total_outliers_removed = 0

    fws_mad_k = 4

    for interval, idx in bins.groupby(bins).groups.items():
        if interval is pd.NA:
            continue

        total_bins_seen += 1
        vals = f.loc[idx]
        n_raw = len(vals)

        if n_raw < min_per_bin:
            print(
                f"[FWSCAL BINNED] Bin {interval} skipped "
                f"(n={n_raw} < {min_per_bin})"
            )
            continue

        # -----------------------------
        # WITHIN-BIN outlier detection
        # -----------------------------
        vals_used = vals.copy()

        if n_raw >= max(min_per_bin, 10):
            logv = np.log(vals_used)
            med = np.nanmedian(logv)
            mad = np.nanmedian(np.abs(logv - med))

            if mad > 0 and np.isfinite(mad):
                z = np.abs(logv - med) / mad
                mask = z <= fws_mad_k
                n_out = int((~mask).sum())

                if n_out > 0:
                    print(
                        f"[FWSCAL BINNED] Bin {interval}: "
                        f"removed {n_out}/{n_raw} outliers (log-MAD, k={fws_mad_k})"
                    )

                vals_used = vals_used[mask]
                total_outliers_removed += n_out

        n_used = len(vals_used)

        if n_used < min_per_bin:
            print(
                f"[FWSCAL BINNED] Bin {interval} discarded after outlier removal "
                f"(n_used={n_used} < {min_per_bin})"
            )
            continue

        mu = float(vals_used.mean())
        sd = float(vals_used.std(ddof=1))

        print(
            f"[FWSCAL BINNED] Bin {interval}: "
            f"n={n_used}/{n_raw}, mean={mu:.4g}, sd={sd:.4g}"
        )

        out_bins.append({
            "bin_lo_um": float(interval.left),
            "bin_hi_um": float(interval.right),
            "fws_mean": mu,
            "fws_sd": sd,
            "n": int(n_used),
        })

        total_bins_kept += 1

    print(
        f"[FWSCAL BINNED] Bins kept: {total_bins_kept} / {total_bins_seen}"
    )
    print(
        f"[FWSCAL BINNED] Total within-bin FWS outliers removed: "
        f"{total_outliers_removed}"
    )

    if not out_bins:
        print("[FWSCAL BINNED] ❌ No bins survived filtering")
        return None

    out_bins.sort(key=lambda b: (b["bin_lo_um"], b["bin_hi_um"]))

    calib = {
        "fit_model": "binned_fws_by_size",
        "binning": {
            "bin_edges_um": [float(x) for x in bin_edges_um],
            "min_per_bin": int(min_per_bin),
            "errorbar": errorbar,
            "within_bin_outlier_method": "log_mad",
            "within_bin_mad_k": fws_mad_k,
        },
        "bins": out_bins,
        "n_raw": int(len(df)),
        "n_used": int(sum(b["n"] for b in out_bins)),
        "diam_um_median": float(np.median(d)),
        "diam_um_p10": float(np.quantile(d, 0.10)),
        "diam_um_p90": float(np.quantile(d, 0.90)),
        "outliers_removed_total": int(total_outliers_removed),
    }

    print("[FWSCAL BINNED] Calibration dict assembled")

    # --------------------------------------------------
    # Diagnostic plot
    # --------------------------------------------------
    if diagnostic_png_path:
        print(f"[FWSCAL BINNED] Writing diagnostic plot → {diagnostic_png_path}")

        d_plot = d.to_numpy()
        f_plot = f.to_numpy()

        if len(d_plot) > max_scatter_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(d_plot), size=max_scatter_points, replace=False)
            d_plot = d_plot[idx]
            f_plot = f_plot[idx]

        centers = np.array(
            [(b["bin_lo_um"] + b["bin_hi_um"]) / 2 for b in out_bins]
        )
        means = np.array([b["fws_mean"] for b in out_bins])
        sds = np.array([b["fws_sd"] for b in out_bins])
        ns = np.array([b["n"] for b in out_bins])

        yerr = sds / np.sqrt(ns) if errorbar.lower() == "se" else sds
        err_label = "SE" if errorbar.lower() == "se" else "SD"

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(d_plot, f_plot, s=6, alpha=0.08, color="0.35",
                   label="raw (filtered)")

        ax.errorbar(
            centers, means, yerr=yerr,
            fmt="o-", color="crimson", ecolor="crimson",
            capsize=3,
            label=f"bin mean ± {err_label} (n≥{min_per_bin})"
        )

        # Y-axis cap (visual only)
        y_hi = np.nanpercentile(f_plot, 98)
        if np.isfinite(y_hi) and y_hi > 0:
            ax.set_ylim(0, y_hi)

        ax.set_xlabel("Equivalent diameter (µm)")
        ax.set_ylabel("FWS_total")
        ax.set_title("FWS calibration (binned, within-bin outlier removal)")
        ax.grid(True, alpha=0.25)
        ax.legend()

        plt.tight_layout()
        plt.savefig(diagnostic_png_path, dpi=150)
        plt.close(fig)

        print("[FWSCAL BINNED] Diagnostic plot written")

    print("[FWSCAL BINNED] ✅ Calibration complete")
    return calib


def append_fwscalibration_record(record: dict, out_path: str | Path) -> None:
    """
    Append one calibration record to JSONL on disk (memory efficient).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record=json_safe(record)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def init_label_change_log(session_dir):
    """Create an empty log for a single combine/labelling session."""
    log_path = Path(session_dir) / "label_changes.json"
    record={
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "actions": []
        }
    record=json_safe(record)
    with open(log_path, "w") as f:
        json.dump(record, f, indent=2)
    return str(log_path)


def record_label_merge(log_path, original_labels, new_label):
    """Append a merge event."""
    with open(log_path, "r") as f:
        log = json.load(f)
    log["actions"].append({
        "type": "merge",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "from": list(original_labels),
        "to": new_label
    })
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def record_label_delete(log_path, deleted_labels):
    """Append a delete event."""
    with open(log_path, "r") as f:
        log = json.load(f)
    log["actions"].append({
        "type": "delete",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "labels": list(deleted_labels)
    })
    log=json_safe(log)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

def _ensure_dirs():
    _TOOL_DIR.mkdir(parents=True, exist_ok=True)
    _SELECTED_UNCALIBRATED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _SELECTED_BEADCALIBRATED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
def load_app_config() -> dict:
    """
    Load flowcytometertoolconfig.yaml.

    Returns a configuration dictionary with all required defaults applied.
    Never raises for configuration load failures; falls back to defaults.
    """
    print("\n" + "=" * 80)
    print("[load_app_config] Starting configuration load.")
    print(f"[load_app_config] Config path: {_CONFIG_PATH}")
    print("=" * 80)

    _ensure_dirs()

    defaults = {
        "active_model": None,
        "active_uncalibrated_model": None,
        "active_beadcalibrated_model": None,
        "trained_models_container_url": _DEFAULT_TRAINED_MODELS_CONTAINER,
    }

    print("[load_app_config] Default configuration:")
    for k, v in defaults.items():
        print(f"    {k}: {v!r}")

    if not _CONFIG_PATH.exists():
        print(
            "[load_app_config] Configuration file does not exist."
        )
        print(
            "[load_app_config] Returning default configuration."
        )

        return defaults.copy()

    print(
        "[load_app_config] Configuration file exists."
    )

    try:
        print(
            f"[load_app_config] YAML library available: {yaml is not None}"
        )

        if yaml:
            print(
                "[load_app_config] Loading configuration using "
                "yaml.safe_load()."
            )

            with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)

            print(
                "[load_app_config] Raw YAML load result:"
            )
            print(repr(cfg))

            cfg = cfg or {}

        else:
            print(
                "[load_app_config] PyYAML unavailable. "
                "Using fallback parser."
            )

            import json

            raw_text = _CONFIG_PATH.read_text(
                encoding="utf-8"
            )

            print(
                f"[load_app_config] Read {len(raw_text)} characters "
                "from config file."
            )

            transformed = (
                raw_text
                .replace("'", '"')
                .replace("\n", ",\n")
            )

            cfg = json.loads(transformed)

            print(
                "[load_app_config] Raw fallback parse result:"
            )
            print(repr(cfg))

    except Exception as e:
        print(f"Failed to load config: {e}")
        traceback.print_exc()
        cfg = {}

    print(
        "[load_app_config] Configuration before defaults:"
    )
    print(repr(cfg))

    for key, value in defaults.items():
        if key not in cfg:
            print(
                f"[load_app_config] Missing key '{key}'. "
                f"Applying default value: {value!r}"
            )
        else:
            print(
                f"[load_app_config] Found key '{key}' "
                f"with value: {cfg[key]!r}"
            )

        cfg.setdefault(key, value)

    print(
        "[load_app_config] Final configuration:"
    )

    for key, value in cfg.items():
        print(f"    {key}: {value!r}")

    print(
        "[load_app_config] Configuration load complete."
    )
    print("=" * 80 + "\n")

    return cfg


def save_app_config(cfg: dict) -> None:
    _ensure_dirs()
    if yaml:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    else:
        # minimal fallback writer
        txt = []
        for k, v in cfg.items():
            txt.append(f"{k}: {v!r}")
        _CONFIG_PATH.write_text("\n".join(txt), encoding="utf-8")

def active_model_dir(calibrated: bool = False) -> str:
    """Return the active model folder.

    calibrated=False -> selectedbeaduncalibratedmodel
    calibrated=True  -> selectedbeadcalibratedmodel
    """
    _ensure_dirs()
    return str(_SELECTED_BEADCALIBRATED_MODEL_DIR if calibrated else _SELECTED_UNCALIBRATED_MODEL_DIR)

def _primary_pkl_in_dir(folder: Path, label: str) -> str:
    """Return the single primary classifier .pkl from an active model folder."""
    _ensure_dirs()
    pkls = [p for p in folder.glob("*.pkl") if not str(p).endswith("probabilistic.pkl")]
    if len(pkls) == 1:
        return str(pkls[0])
    if len(pkls) == 0:
        raise FileNotFoundError(
            f"No active {label} model found in {folder}. Use the relevant 'Download & Set Active' control."
        )
    raise RuntimeError(
        f"Multiple primary .pkl files found in {folder}. Please clear the folder and set a single active {label} model."
    )

def resolve_active_model_path() -> str:
    """Compatibility wrapper: resolve the active raw/uncalibrated model path."""
    return resolve_active_raw_model_path()

def list_available_model_versions(container_url: str | None = None) -> list[str]:
    """
    Return sorted list of version prefixes by scanning '<version>/...' in the container.
    """
    url = container_url or load_app_config().get("trained_models_container_url") or _DEFAULT_TRAINED_MODELS_CONTAINER
    account_url, container_name, _ = _split_blob_url(url)
    cc = get_container_client(account_url, container_name, anonymous=False)
    versions = set()
    for b in cc.list_blobs():
        # expect 'YYYYMMDD_HHMMSS/filename'
        parts = b.name.split("/", 1)
        if parts and parts[0]:
            versions.add(parts[0])
    return sorted(versions)

def _clear_selected_folder():
    _clear_folder(_SELECTED_UNCALIBRATED_MODEL_DIR)

def download_model_version(version: str, container_url: str | None = None) -> list[str]:
    """
    Download all artifacts under '<version>/' into selectedbeaduncalibratedmodel.
    Returns local file paths.
    """
    return _download_model_to_dir(version, _SELECTED_UNCALIBRATED_MODEL_DIR, container_url=container_url)

def set_active_model(version: str, container_url: str | None = None) -> str:
    """Compatibility wrapper: set the active raw/uncalibrated model."""
    return set_active_uncalibrated_model(version, container_url=container_url)



def _clear_folder(folder: Path) -> None:
    """Remove all contents of a folder without deleting the folder itself."""
    _ensure_dirs()
    for p in folder.glob("*"):
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception:
            pass


def _download_model_to_dir(version: str, dest_dir: Path, container_url: str | None = None) -> list[str]:
    """Download all artifacts under '<version>/' into dest_dir. Returns local file paths."""
    if not version or "/" in version or "\\" in version:
        raise ValueError("Invalid version prefix.")
    url = container_url or load_app_config().get("trained_models_container_url") or _DEFAULT_TRAINED_MODELS_CONTAINER
    account_url, container_name, _ = _split_blob_url(url)
    cc = get_container_client(account_url, container_name, anonymous=False)
    _clear_folder(dest_dir)
    local_paths = []
    prefix = f"{version}/"
    for b in cc.list_blobs(name_starts_with=prefix):
        blob_rel = b.name[len(prefix):]
        if not blob_rel:
            continue
        local_path = dest_dir / blob_rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as fh:
            cc.download_blob(b.name).readinto(fh)
        local_paths.append(str(local_path))
    if not local_paths:
        raise FileNotFoundError(f"No blobs found under '{prefix}' in trained models container.")
    return local_paths


def _primary_pkl_from_paths(paths: list[str]) -> str:
    """Return the main classifier .pkl (excluding probabilistic calibration files)."""
    primary = [p for p in paths if p.endswith(".pkl") and not p.endswith("probabilistic.pkl")]
    if not primary:
        raise FileNotFoundError("Downloaded artifacts don't include a primary '*.pkl' model.")
    return primary[0]


def set_active_uncalibrated_model(version: str, container_url: str | None = None) -> str:
    """
    Download a model version into selectedbeaduncalibratedmodel and record it in config
    as active_uncalibrated_model.  Returns the primary .pkl path.
    """
    paths = _download_model_to_dir(version, _SELECTED_UNCALIBRATED_MODEL_DIR, container_url=container_url)
    primary = _primary_pkl_from_paths(paths)
    cfg = load_app_config()
    cfg["active_uncalibrated_model"] = {
        "version": version,
        "local_dir": str(_SELECTED_UNCALIBRATED_MODEL_DIR),
        "primary_model": primary,
        "container_url": container_url or cfg.get("trained_models_container_url"),
    }
    save_app_config(cfg)
    return primary


def set_active_beadcalibrated_model(version: str, container_url: str | None = None) -> str:
    """
    Download a model version into selectedbeadcalibratedmodel and record it in config
    as active_beadcalibrated_model.  Returns the primary .pkl path.
    """
    paths = _download_model_to_dir(version, _SELECTED_BEADCALIBRATED_MODEL_DIR, container_url=container_url)
    primary = _primary_pkl_from_paths(paths)
    cfg = load_app_config()
    cfg["active_beadcalibrated_model"] = {
        "version": version,
        "local_dir": str(_SELECTED_BEADCALIBRATED_MODEL_DIR),
        "primary_model": primary,
        "container_url": container_url or cfg.get("trained_models_container_url"),
    }
    save_app_config(cfg)
    return primary


def resolve_active_raw_model_path() -> str:
    """Return the path to the active raw/uncalibrated model."""
    _ensure_dirs()
    cfg = load_app_config()

    for key in ("active_uncalibrated_model", "active_model"):
        slot = cfg.get(key)
        if slot and isinstance(slot, dict):
            primary = slot.get("primary_model")
            if primary:
                primary_path = Path(primary)
                try:
                    is_current_slot = (
                        primary_path == _SELECTED_UNCALIBRATED_MODEL_DIR
                        or _SELECTED_UNCALIBRATED_MODEL_DIR in primary_path.parents
                    )
                except Exception:
                    is_current_slot = False
                if is_current_slot and primary_path.is_file():
                    return str(primary_path)

    return _primary_pkl_in_dir(_SELECTED_UNCALIBRATED_MODEL_DIR, "raw/uncalibrated")


def resolve_active_beadcalibrated_model_path() -> str | None:
    """
    Resolve and return the path to the currently active bead-calibrated model.
    """
    print("[resolve_active_beadcalibrated_model_path] Starting resolution.")

    try:
        _ensure_dirs()
        print(
            "[resolve_active_beadcalibrated_model_path] "
            "Verified application directories."
        )
    except Exception as e:
        print(
            "[resolve_active_beadcalibrated_model_path] "
            f"Directory validation failed: {e}"
        )
        raise

    cfg = load_app_config()

    print(
        "[resolve_active_beadcalibrated_model_path] "
        f"Configuration loaded. Keys: {list(cfg.keys())}"
    )

    slot = cfg.get("active_beadcalibrated_model")

    print(
        "[resolve_active_beadcalibrated_model_path] "
        f"active_beadcalibrated_model = {slot!r}"
    )

    if not slot:
        print(
            "[resolve_active_beadcalibrated_model_path] "
            "No active_beadcalibrated_model entry found in configuration."
        )
        return None

    if not isinstance(slot, dict):
        print(
            "[resolve_active_beadcalibrated_model_path] "
            f"Expected dict but found {type(slot).__name__}. "
            "Configuration appears invalid."
        )
        return None

    primary = slot.get("primary_model")

    print(
        "[resolve_active_beadcalibrated_model_path] "
        f"Configured primary_model = {primary!r}"
    )

    if primary:
        exists = Path(primary).is_file()

        print(
            "[resolve_active_beadcalibrated_model_path] "
            f"Primary path exists = {exists}"
        )

        if exists:
            resolved = str(Path(primary))
            print(
                "[resolve_active_beadcalibrated_model_path] "
                f"Using configured primary model: {resolved}"
            )
            return resolved

        print(
            "[resolve_active_beadcalibrated_model_path] "
            "Configured primary model path is stale or missing."
        )
    else:
        print(
            "[resolve_active_beadcalibrated_model_path] "
            "No primary_model value present in configuration."
        )

    print(
        "[resolve_active_beadcalibrated_model_path] "
        f"Scanning fallback directory: {_SELECTED_BEADCALIBRATED_MODEL_DIR}"
    )

    try:
        all_pkls = list(_SELECTED_BEADCALIBRATED_MODEL_DIR.glob("*.pkl"))

        print(
            "[resolve_active_beadcalibrated_model_path] "
            f"Found {len(all_pkls)} .pkl file(s)."
        )

        for p in all_pkls:
            print(
                "[resolve_active_beadcalibrated_model_path] "
                f"Candidate file: {p}"
            )

        pkls = [
            p
            for p in all_pkls
            if not str(p).endswith("probabilistic.pkl")
        ]

        print(
            "[resolve_active_beadcalibrated_model_path] "
            f"{len(pkls)} candidate model(s) remain after excluding "
            "probabilistic models."
        )

    except Exception as e:
        print(
            "[resolve_active_beadcalibrated_model_path] "
            f"Directory scan failed: {e}"
        )
        return None

    if pkls:
        selected = str(pkls[0])

        print(
            "[resolve_active_beadcalibrated_model_path] "
            f"Selected fallback model: {selected}"
        )

        return selected

    print(
        "[resolve_active_beadcalibrated_model_path] "
        "No valid bead-calibrated models could be resolved."
    )

    return None
