# functions file written whilst writing flow_cytometer_tool.py
import requests
import subprocess
import os
import json
import pandas as pd
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import tkinter as tk
import csv
from listmode import extract
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
import qc_plots
import json
from auth import get_credential
import hashlib
import glob

__all__ = ["BlobServiceClient","choose_zone_folders","build_consensual_dataset","platform","run_backend_only","argparse","summarize_predictions","download_blobs", "convert_cyz_to_json", "compile_cyz2json_from_release",
    "compile_r_requirements", "flatten_dict", "dict_to_csv", "clear_temp_folder", "download_file",
    "load_file", "to_listmode", "apply_r", "select_output_dir", "load_json", "select_particles",
    "get_pulses", "display_image", "update_navigation_buttons", "save_metadata", "plot3d",
    "os", "shutil", "tk", "messagebox", "filedialog", "simpledialog", "ttk", "ContainerClient",
    "urlparse", "pd", "np", "joblib", "datetime", "json", "csv", "plt", "FigureCanvasTkAgg",
    "Line2D", "PolygonSelector", "Path", 
    "subprocess", "zipfile", "extract", "re","test_model","train_classifier","combine_csv_files","convert_json_to_listmode",
    "FileHandler","log_message","Observer","FileSystemEventHandler","filedialog","init_label_change_log","record_label_delete","record_label_merge",
    "sample_rows","upload_to_blob", "mix_blob_files","list_blobs","extract_processed_url","apply_python_model","delete_file","combine_csvs","train_model","test_classifier","expertise_matrix_path"]


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

from storage_clients import _split_blob_url, get_container_client, get_blob_client  # existing helpers

expertise_matrix_path = os.path.join(base_path, "..\matrices\expertise_matrix.csv")

# Locations
_TOOL_DIR = Path.home() / "Documents" / "flowcytometertool"
_SELECTED_MODEL_DIR = _TOOL_DIR / "selectedvalidappliedmodel"
_CONFIG_PATH = _TOOL_DIR / "flowcytometertoolconfig.yaml"

# Default trained-models container (adjust if you use a different one)
_DEFAULT_TRAINED_MODELS_CONTAINER = "https://citprodflowcytosa.blob.core.windows.net/trainedmodels"

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
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def init_label_change_log(session_dir):
    """Create an empty log for a single combine/labelling session."""
    log_path = Path(session_dir) / "label_changes.json"
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "actions": []
        }, f, indent=2)
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
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

def _ensure_dirs():
    _TOOL_DIR.mkdir(parents=True, exist_ok=True)
    _SELECTED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_app_config() -> dict:
    """Load flowcytometertoolconfig.yaml; return defaults if absent/invalid."""
    _ensure_dirs()
    if not _CONFIG_PATH.exists():
        return {
            "active_model": None,
            "trained_models_container_url": _DEFAULT_TRAINED_MODELS_CONTAINER,
        }
    try:
        if yaml:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        else:
            # naive fallback parser (expects simple key: value YAML)
            import json
            cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8")
                             .replace("'", '"').replace("\n", ",\n"))
    except Exception:
        cfg = {}
    # sensible defaults
    cfg.setdefault("active_model", None)
    cfg.setdefault("trained_models_container_url", _DEFAULT_TRAINED_MODELS_CONTAINER)
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

def active_model_dir() -> str:
    """Return the absolute path to the folder where the active model must reside."""
    _ensure_dirs()
    return str(_SELECTED_MODEL_DIR)

def resolve_active_model_path() -> str:
    """
    Find the pinned model file (*.pkl) in selectedvalidappliedmodel.
    We intentionally exclude probabilistic calibrations from the primary classifier.
    """
    _ensure_dirs()
    pkls = [p for p in _SELECTED_MODEL_DIR.glob("*.pkl") if not str(p).endswith("probabilistic.pkl")]
    if len(pkls) == 1:
        return str(pkls[0])
    if len(pkls) == 0:
        raise FileNotFoundError(
            "No active model found. Use the 'Download & Set Active' control to choose a model "
            "and place it in selectedvalidappliedmodel."
        )
    raise RuntimeError(
        "Multiple model files found in selectedvalidappliedmodel. "
        "Please clear the folder and set a single active model."
    )

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
    _ensure_dirs()
    for p in _SELECTED_MODEL_DIR.glob("*"):
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception:
            pass

def download_model_version(version: str, container_url: str | None = None) -> list[str]:
    """
    Download all artifacts under '<version>/' from the trained models container
    into selectedvalidappliedmodel (clearing it first). Returns local file paths.
    """
    if not version or "/" in version or "\\" in version:
        raise ValueError("Invalid version prefix.")
    url = container_url or load_app_config().get("trained_models_container_url") or _DEFAULT_TRAINED_MODELS_CONTAINER
    account_url, container_name, _ = _split_blob_url(url)
    cc = get_container_client(account_url, container_name, anonymous=False)

    # Clear then fetch
    _clear_selected_folder()
    local_paths = []
    prefix = f"{version}/"
    for b in cc.list_blobs(name_starts_with=prefix):
        blob_rel = b.name[len(prefix):]
        if not blob_rel:  # safety
            continue
        local_path = _SELECTED_MODEL_DIR / blob_rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as fh:
            cc.download_blob(b.name).readinto(fh)
        local_paths.append(str(local_path))
    if not local_paths:
        raise FileNotFoundError(f"No blobs found under '{prefix}' in trained models container.")
    return local_paths

def set_active_model(version: str, container_url: str | None = None) -> str:
    """
    High-level: download version -> write config -> return the primary model path.
    """
    paths = download_model_version(version, container_url=container_url)
    # pick the main classifier pkl
    primary = [p for p in paths if p.endswith(".pkl") and not p.endswith("probabilistic.pkl")]
    if not primary:
        raise FileNotFoundError("Downloaded artifacts don't include a primary '*.pkl' model.")
    # persist config
    cfg = load_app_config()
    cfg["active_model"] = {
        "version": version,
        "local_dir": str(_SELECTED_MODEL_DIR),
        "primary_model": primary[0],
        "container_url": container_url or cfg["trained_models_container_url"],
    }
    save_app_config(cfg)
    return primary[0]
# === end new block ===========================================================


def stratified_subsample(df, target_column, max_per_class=1000):
    return df.groupby(target_column, group_keys=False).apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42)).reset_index(drop=True)


def compile_cyz2json(clone_dir, path_entry):
    """Clone and compile the cyz2json tool."""
    if os.path.exists(clone_dir):
        messagebox.showinfo("Info", "cyz2json already exists in " + clone_dir)
        return

    try:
        subprocess.run(["git", "clone", "https://github.com/OBAMANEXT/cyz2json.git", clone_dir], check=True)
        subprocess.run(["dotnet", "build", "-o", "bin"], cwd=clone_dir, check=True)
        path_entry.delete(0, tk.END)
        path_entry.insert(0, os.path.join(clone_dir, "bin", "Cyz2Json.dll"))
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Compilation Error", f"Failed to compile cyz2json: {e}. Have you installed the requirement DotNet version 8.0? See https://github.com/OBAMANEXT/cyz2json")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def train_model(df, plots_dir, model_path, rootdir, nogui=False, self = None, calibration_enabled = False, max_per_class = 100000):
    
    # --- PRE-FLIGHT AUTH: trigger AAD prompt pre training ---
    cred = get_credential()
    # Request a token for Azure Storage so the browser pops up immediately
    # (scope must be 'https://storage.azure.com/.default')
    try:
        cred.get_token("https://storage.azure.com/.default")
    except Exception as e:
        # If the prompt is cancelled, fail fast so the user tries again
        if not nogui:
            from tkinter import messagebox
            messagebox.showerror("Sign-in required", f"Azure sign-in is required to train & publish models.\n\n{e}")
        raise
    
    # Keep your existing timestamped naming (this string is your 'version')
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # When invoked from the UI, override model_path to include the timestamped version
    if self is not None and hasattr(self, "tool_dir"):
        self.model_path = os.path.join(self.tool_dir, f'models/final_model_{ts}.pkl')
        model_path = self.model_path

    try:
        if df is None:
            if nogui:
                print("Error: No data to train on.")
            else:
                from tkinter import messagebox
                messagebox.showerror("Error", "No data to train on.")
            return

        # --- train exactly as before ---
        train_classifier(df, plots_dir, model_path, max_per_class, calibration_enabled)


        # Infer version from filename 'final_model_<VERSION>.pkl'
        m = re.search(r"final_model_(\d{8}_\d{6})\.pkl$", os.path.basename(model_path))
        version = m.group(1) if m else ts  # fallback to timestamp if pattern changes

        # Resolve sibling artifact file paths you already write in train_classifier(...)
        model_dir = os.path.dirname(model_path)
        cv_results_csv       = os.path.join(model_dir, "cv_results"      + os.path.basename(model_path) + ".csv")
        learning_curve_csv   = os.path.join(model_dir, "learning_curve"  + os.path.basename(model_path) + ".csv")
        perm_importance_csv  = os.path.join(model_dir, "permutation_importance_" + os.path.basename(model_path) + ".csv")
        confusion_png        = os.path.join(model_dir, "confusionmatrix_" + os.path.basename(model_path) + '.png')
        modeltrainsettingsjson = os.path.join(model_dir, "modeltrainsettings.json")
        modeltrainenv = os.path.join(rootdir, "environment.yml")
        modelcalibrated = model_path+'.probabilistic.pkl'

        artifacts = [model_path, cv_results_csv, learning_curve_csv, perm_importance_csv, confusion_png, modeltrainsettingsjson, modelcalibrated, expertise_matrix_path, modeltrainenv, self.label_change_log_path]


        # Only attempt upload when called from the app (so we can read the account/container URL)
        if self is not None and hasattr(self, "url_entry_blob"):
            container_url = 'https://citprodflowcytosa.blob.core.windows.net/trainedmodels'  # e.g., https://<acct>.blob.core.windows.net/<container>
            # Push into: container 'models' / folder '<version>/...'
            _push_model_artifacts_to_models_container(version=version, file_paths=artifacts, container_url=container_url)

            msg = f"Pushed {len([p for p in artifacts if p and os.path.exists(p)])} artifacts to blob: models/{version}/"
            if nogui:
                print(msg)
            else:
                from tkinter import messagebox
                messagebox.showinfo("Model Upload", msg)
                
        if nogui:
            print("Model training completed successfully.")
        else:
            from tkinter import messagebox
            messagebox.showinfo("Training Complete", "Model training completed successfully.")
        notebook = ttk.Notebook(self.root)

    except Exception as e:
        if nogui:
            print(f"Training Error: Failed to train model: {e}")
        else:
            from tkinter import messagebox
            messagebox.showerror("Training Error", f"Failed to train model: {e}")



def test_classifier(df, model_path, nogui=False):
    try:
        if not os.path.exists(model_path):
            msg = "Trained model not found. Please train the model first."
            if nogui:
                print(f"Model Error: {msg}")
            else:
                from tkinter import messagebox
                messagebox.showerror("Model Error", msg)
            return df, None
        if df is None:
            msg = "No dataset loaded. Please load or combine CSVs first."
            if nogui:
                print(f"Data Error: {msg}")
            else:
                from tkinter import messagebox
                messagebox.showerror("Data Error", msg)
            return df, None
        df, summary = test_model(df, model_path)
        if nogui:
            print("Prediction Summary:\n", summary)
        else:
            messagebox.showinfo("Prediction Summary", f"Predictions made successfully.\n\n{summary}")
        return df, summary
    except Exception as e:
        if nogui:
            print(f"Test Error: Failed to test classifier: {e}")
        else:
            from tkinter import messagebox
            messagebox.showerror("Test Error", f"Failed to test classifier: {e}")
        return df, None

        
def combine_csvs(dataseturl, root_path, output_path, expertise_matrix_path, max_per_class_entry = None, nogui=False, prompt_merge_fn = None, premerge_plot_fn = None, delete_labels_fn=None):
    if nogui:
        zonechoices = "FAKEBALTIC"#PELTIC  # Not ideal - hard coded so if the underlying dataset changes, the github actions workflow will break
    else:
        zonechoices = choose_zone_folders(output_path)
        
    modeltrainsettings_out=Path(output_path) / f"../models/modeltrainsettings.json"
    dataseturl=dataseturl
    
    try:
        # Where to save:
        # - modeltrainsettings.json -> inside the zone folder by default
        # - thin metadata packets -> under ~/Documents/flowcytometertool/zone_metadata/<zone>/
        packets_dir = os.path.join(os.path.expanduser("~"), "Documents", "flowcytometertool", "zone_metadata")
        collect_zone_metadata_and_assert(
            dataseturl=dataseturl.get().strip(),
            repo_root = root_path,
            base_path=output_path,
            zonechoice=zonechoices,
            grablist_path="grablist.txt",              # graceful no-op if missing
            modeltrainsettings_out=modeltrainsettings_out,               
            packets_out_dir=packets_dir,
            nogui=nogui,
        )
    except ValueError:
        # Mismatch of serialNumber / PMTlevels_str blocks the combine
        return None


    try:
        expertise_matrix = pd.read_csv(expertise_matrix_path, index_col=0)
        expertise_levels = expertise_matrix.loc[zonechoices].to_dict()
        expertise_levels = {
            'expert': [k for k, v in expertise_levels.items() if v == 3],
            'advanced': [k for k, v in expertise_levels.items() if v == 2],
            'non_expert': [k for k, v in expertise_levels.items() if v == 1]
        }

        print("Zone choices:", zonechoices)
        print("expertise_levels:", expertise_levels)
        combined_df = build_consensual_dataset(output_path, expertise_levels, zonechoices, prompt_merge_fn, premerge_plot_fn, delete_labels_fn)
        #print("set(list(combined_df['source_label']))")
        #print(set(list(combined_df['source_label'])))
        #print("set(list(combined_df['consensus_label']))")
        #print(set(list(combined_df['consensus_label'])))
        #combined_df['source_label'] = [
        #    re.sub(r'[^a-zA-Z]', '', item).lower() for item in combined_df['source_label']
        #]
        #combined_df.loc[combined_df['source_label'] == 'nophyto', 'source_label'] = 'nophytoplankton'
        #print('Cleaned group names to something consistent')
        #print("Cleaned source labels:", list(set(combined_df['source_label'])))
        print("Now dropping columns: ['consensus_label','person','index','id','sample_weight']")
        combined_df = combined_df.drop(columns=['person','id'])
                
        # --------------------------------------------------------------
        # TRAINING DATASET SUMMARY → append to modeltrainsettings.json
        # --------------------------------------------------------------
        try:
            # Load existing modeltrainsettings
            mts_path = modeltrainsettings_out
            with open(mts_path, "r") as f:
                mts = json.load(f)
            summary = {}
            summary["total_particles"] = int(len(combined_df))
            cls_counts = combined_df["source_label"].value_counts().to_dict()
            summary["counts_per_class"] = {str(k): int(v) for k, v in cls_counts.items()}
            if "person" in combined_df.columns:
                person_counts = combined_df["person"].value_counts().to_dict()
                summary["counts_per_person"] = {str(k): int(v) for k, v in person_counts.items()}
            if "sample_weight" in combined_df.columns:
                summary["sample_weight"] = {
                    "min": float(combined_df["sample_weight"].min()),
                    "max": float(combined_df["sample_weight"].max()),
                    "median": float(combined_df["sample_weight"].median())
                }
            mts["training_dataset_summary"] = summary
            cleaning = {}
            cleaning["post_merge_nn_cleaning_ran"] = "False"
            cleaning["max_per_class_entry"] = max_per_class_entry           
            mts["cleaning"] = cleaning

            with open(mts_path, "w") as f:
                json.dump(mts, f, indent=2)
                
            print("Added training_dataset_summary to modeltrainsettings.json")
        except Exception as e:
            print(f"[WARN] Could not update modeltrainsettings.json: {e}")        
        
        # Set NN cleaning flag
        



        if combined_df is not None and not combined_df.empty:
            if nogui:
                print("CSV files combined successfully.")
            else:
                messagebox.showinfo("Success", "CSV files combined successfully.")
            return combined_df
        else:
            if nogui:
                print("No CSV files found to combine.")
            else:
                messagebox.showwarning("No CSVs", "No CSV files found to combine.")
            return None
    except Exception as e:
        if nogui:
            print(f"Combine Error: Failed to combine CSVs: {e}")
        else:
            messagebox.showerror("Combine Error", f"Failed to combine CSVs: {e}")
        return None


def nn_homogenize_df(
    df,
    *,
    label_col="source_label",
    feature_cols=("FWS_total", "Fl Red_total", "Fl Orange_total"),
    keep_unconsidered="keep",        # "keep" | "drop"
    downsample_n=None,
    random_state=42,
    max_iters=100,
    # --- NEW: isolation guards ---
    enforce_density=True,
    k_neighbors=10,
    min_same_neighbors=1,
    prune_tiny_components=True,
    min_component_size=3,
    eps_factor=1.5,                  # ε = eps_factor * median(2nd-NN distance within class)
):
    """
    Iteratively remove particles whose nearest neighbour is a different class (both removed),
    until stable. Then apply isolation guards to catch tiny odd cliques (pairs, triplets).

    Isolation guards (in robustly normalised space):
      1) k-NN same-class density: require at least `min_same_neighbors` within k neighbors.
      2) ε-graph tiny-component pruning: per class, drop components with size < min_component_size,
         with ε chosen adaptively from the class' 2nd-NN distance median.

    Returns a DataFrame preserving all original columns for surviving rows (and optionally
    non-evaluable rows if keep_unconsidered="keep").
    """
    import numpy as np
    import pandas as pd

    # ---------- helpers ----------
    def _normalise_for_nn(df_eval, feature_cols):
        """Robust per-axis scaling: (x - median)/MAD; fallback to 95% IPR if MAD=0."""
        X = df_eval[list(feature_cols)].to_numpy(float)
        Xn = np.zeros_like(X)
        for i, col in enumerate(feature_cols):
            v = X[:, i]
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            if mad > 0:
                Xn[:, i] = (v - med) / mad
            else:
                lo, hi = np.percentile(v, [2.5, 97.5])
                rng = hi - lo if hi > lo else 1.0
                Xn[:, i] = (v - med) / rng
        return Xn

    def _nn_indices(points):
        """Nearest neighbour indices (excluding self), preferring cKDTree/sklearn."""
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            _, idx = tree.query(points, k=2, workers=-1)
            return idx[:, 1]
        except Exception:
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=-1)
                nn.fit(points)
                _, idx = nn.kneighbors(points, n_neighbors=2, return_distance=True)
                return idx[:, 1]
            except Exception:
                m = points.shape[0]
                if m > 30000:
                    raise RuntimeError(
                        "No fast NN backend (scipy/sklearn) and dataset is large. "
                        "Install scipy or use downsample_n."
                    )
                idx_nn = np.empty(m, dtype=int)
                for i in range(m):
                    d2 = ((points - points[i]) ** 2).sum(axis=1)
                    d2[i] = np.inf
                    idx_nn[i] = int(np.argmin(d2))
                return idx_nn

    def _knn_indices(points, k):
        """Return k-NN indices (excluding self) for density voting."""
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            d, idx = tree.query(points, k=min(k + 1, len(points)), workers=-1)
            return idx[:, 1:], d[:, 1:]
        except Exception:
            try:
                from sklearn.neighbors import NearestNeighbors
                k_eff = min(k + 1, len(points))
                nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1)
                nn.fit(points)
                d, idx = nn.kneighbors(points, n_neighbors=k_eff, return_distance=True)
                return idx[:, 1:], d[:, 1:]
            except Exception:
                # Brute-force fallback (small n)
                m = points.shape[0]
                k_eff = min(k + 1, m)
                D = np.zeros((m, m), dtype=float)
                for i in range(m):
                    D[i] = ((points - points[i]) ** 2).sum(axis=1)
                    D[i, i] = np.inf
                idx = np.argsort(D, axis=1)[:, :k_eff]
                # distances:
                rows = np.arange(m)[:, None]
                d = np.sqrt(D[rows, idx])
                return idx[:, 1:], d[:, 1:]

    def _prune_components_classwise(Xn, labels, target_class, min_size, eps_factor):
        """Return a boolean mask of points to KEEP within the specified class."""
        import numpy as np
        cls_mask = (labels == target_class)
        idx_cls = np.where(cls_mask)[0]
        if len(idx_cls) == 0:
            return np.ones(len(labels), dtype=bool)
        Xc = Xn[idx_cls]

        # Skip if trivially small
        if len(Xc) < min_size:
            keep_local = np.zeros(len(Xc), dtype=bool)  # drop all tiny blobs
            keep = np.ones(len(labels), dtype=bool)
            keep[idx_cls] = keep_local
            return keep

        # Adaptive ε from class’ 2nd-NN median
        knn_idx, knn_dist = _knn_indices(Xc, k=2)
        d2 = knn_dist[:, 1] if knn_dist.shape[1] >= 2 else knn_dist[:, -1]
        eps = float(np.median(d2)) * eps_factor
        if not np.isfinite(eps) or eps <= 0:
            # fallback: overall scale
            eps = float(np.median(knn_dist)) if np.isfinite(np.median(knn_dist)) else 1.0

        # Build graph via ε-neighbourhood
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(Xc)
            # all undirected edges under eps
            pairs = list(tree.query_pairs(r=eps))
        except Exception:
            # fallback: brute-force edges
            pairs = []
            for i in range(len(Xc)):
                for j in range(i + 1, len(Xc)):
                    if np.linalg.norm(Xc[i] - Xc[j]) <= eps:
                        pairs.append((i, j))

        # Connected components
        adj = [[] for _ in range(len(Xc))]
        for i, j in pairs:
            adj[i].append(j)
            adj[j].append(i)

        visited = np.zeros(len(Xc), dtype=bool)
        keep_local = np.ones(len(Xc), dtype=bool)
        for s in range(len(Xc)):
            if visited[s]:
                continue
            # BFS
            comp = []
            stack = [s]
            visited[s] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            # If component size < min_size -> drop all nodes in this component
            if len(comp) < min_size:
                keep_local[comp] = False

        keep = np.ones(len(labels), dtype=bool)
        keep[idx_cls] = keep_local
        return keep

    # ---------- main ----------
    if len(feature_cols) != 3:
        raise ValueError("feature_cols must be a 3-tuple (x, y, z).")
    fx, fy, fz = feature_cols

    needed = [fx, fy, fz, label_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    eval_mask = df[needed].notna().all(axis=1)
    df_eval = df.loc[eval_mask].copy()
    if len(df_eval) == 0:
        return df.copy() if keep_unconsidered == "keep" else df.iloc[0:0].copy()

    # Optional downsampling (affects BOTH NN and guards)
    if downsample_n is not None and len(df_eval) > downsample_n:
        df_eval = df_eval.sample(n=downsample_n, random_state=random_state)

    # Robustly normalised coordinates (used only for neighbour logic)
    Xn = _normalise_for_nn(df_eval, (fx, fy, fz))
    y = df_eval[label_col].astype(str).to_numpy()

    # --- Stage 1: Cross-class NN elimination (synchronous) ---
    keep = np.ones(len(df_eval), dtype=bool)
    iters = 0
    while iters < max_iters:
        iters += 1
        idx_active = np.where(keep)[0]
        if len(idx_active) <= 1:
            break
        Xa = Xn[idx_active]
        ya = y[idx_active]
        nn = _nn_indices(Xa)
        conflict = ya != ya[nn]
        if not np.any(conflict):
            break
        to_remove = conflict.copy()
        to_remove[nn[conflict]] = True
        keep[idx_active[to_remove]] = False

    survivors_local = np.where(keep)[0]
    Xn_surv = Xn[survivors_local]
    y_surv = y[survivors_local]
    survivors_index = df_eval.iloc[survivors_local].index

    # --- Stage 2: Isolation guards ---
    # 2.1 k-NN density: require at least `min_same_neighbors` same-class within k
    if enforce_density and len(survivors_local) > 0 and min_same_neighbors > 0:
        knn_idx, _ = _knn_indices(Xn_surv, k=k_neighbors)
        same_counts = np.zeros(len(survivors_local), dtype=int)
        for i in range(len(survivors_local)):
            neigh = knn_idx[i]
            same_counts[i] = int(np.sum(y_surv[neigh] == y_surv[i]))
        keep_density = same_counts >= min_same_neighbors
        survivors_index = survivors_index[keep_density]
        Xn_surv = Xn_surv[keep_density]
        y_surv = y_surv[keep_density]

    # 2.2 ε-graph tiny-component pruning (per class)
    if prune_tiny_components and len(survivors_index) > 0 and min_component_size > 1:
        keep_cc = np.ones(len(survivors_index), dtype=bool)
        classes = np.unique(y_surv)
        for cls in classes:
            class_keep = _prune_components_classwise(
                Xn_surv, y_surv, cls, min_component_size, eps_factor
            )
            keep_cc &= class_keep
        survivors_index = survivors_index[keep_cc]

    # --- Build output df ---
    if keep_unconsidered == "keep":
        df_out = pd.concat([df.loc[~eval_mask], df.loc[survivors_index]], axis=0).sort_index(kind="mergesort")
    else:
        df_out = df.loc[survivors_index].copy()
    return df_out
    


def sample_rows(df, sample_rate=0.001):
    return df.sample(frac=sample_rate)




class FileHandler(FileSystemEventHandler):
    def __init__(self, cyz2json_path, output_folder, model_path):
        self.cyz2json_path = cyz2json_path
        self.output_folder = output_folder
        self.model_path = model_path

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".cyz"):
            self.process_file(event.src_path)
    
    def on_moved(self, event):
        if event.is_directory:
            return
        # event.dest_path is the new filename after rename
        if event.dest_path.lower().endswith(".cyz"):
            self.process_file(event.dest_path)            

    def process_file(self, file_path):
        try:
            log_message(f"Processing file: {file_path}")
            base_filename = os.path.basename(file_path)
            json_file = os.path.join(self.output_folder, base_filename.replace(".cyz", ".json"))
            listmode_file = os.path.join(self.output_folder, base_filename.replace(".cyz", ".csv"))
            predictions_file = os.path.join(self.output_folder, "predictions.csv")
            imagedir = os.path.join(self.output_folder, "images/")

            if not wait_for_file_release(file_path):
                log_message(f"Timeout: File still locked after waiting: {file_path}")
                return

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    load_file(self.cyz2json_path, file_path, json_file)
                    break
                except Exception as e:
                    log_message(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(4)
                    else:
                        raise

            log_message(f"Success: Cyz2json applied {file_path}")
#            to_listmode(json_file, listmode_file,'')
            to_listmode(json_file, listmode_file,imagedir, True, diagnosticR2pngpath = os.path.join(self.output_folder, "calibrationcurve.png"))
            log_message(f"Success: Listmode applied {file_path}")
            apply_python_model(listmode_file, predictions_file, self.model_path)
            log_message(f"Success: Predictions made for {file_path}")

            predictions_df = pd.read_csv(predictions_file)
            prediction_counts = predictions_df['predicted_label'].value_counts().reset_index()
            prediction_counts.columns = ['class', 'count']
            prediction_counts_path = predictions_file + "_counts.csv"
            prediction_counts.to_csv(prediction_counts_path, index=False)            
            predictions_csv = predictions_file
            qc_plots.update_after_file(json_file,predictions_csv, self.output_folder)                        
            log_message(f"Success: counted {file_path}")

            data = pd.read_csv(predictions_file)
            data['category'] = data['predicted_label']
            unique_categories = data['category'].unique()
            preset_colors = {
                'rednano': 'red',
                'orapicoprok': 'orange',
                'micro': 'blue',
                'beads': 'green',
                'oranano': 'purple',
                'noise': 'gray',
                'C_undetermined': 'black',
                'redpico': 'pink'
            }
            color_map = {
                category: preset_colors.get(
                    category,
                    f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
                ) for category in unique_categories
            }
            data['color'] = data['category'].map(color_map)
            x_99 = np.percentile(data["Fl_Yellow_total"], 99.5)
            y_99 = np.percentile(data["Fl_Red_total"], 99.5)
            z_99 = np.percentile(data["Fl_Orange_total"], 99.5)
            scatter = go.Scatter3d(
                x=data["Fl_Yellow_total"],
                y=data["Fl_Red_total"],
                z=data["Fl_Orange_total"],
                mode='markers',
                marker=dict(size=5, color=data['color'], showscale=False),
                text=data['category'],
                name='Data Points'
            )
            camera = dict(
                eye=dict(x=-1.5, y=-1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
            fig = go.Figure(data=[scatter])
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[0, x_99], title="Fl_Yellow_total"),
                    yaxis=dict(range=[0, y_99], title="Fl_Red_total"),
                    zaxis=dict(range=[0, z_99], title="Fl_Orange_total"),
                    camera=camera
                ),
                title='3D Data Points'
            )
            plot3d_prediction_path = predictions_file + "_3d.html"
            pio.write_html(fig, file=plot3d_prediction_path, auto_open=False)
            log_message("Plot saved as '3D_Plot.html'.")
            delete_file(listmode_file)
            delete_file(json_file)
            delete_file(imagedir)
#            delete_file(plot3d_prediction_path)
#            delete_file(predictions_file)
#            delete_file(prediction_counts_path)
        except Exception as e:
            log_message(f"Error: An error occurred processing {file_path}: {e}")


def extract_processed_url(line):
    prefix = "Success: counted "
    if line.startswith(prefix):
        return line[len(prefix):].strip()
    return None

# Function to log messages to both terminal and a log file
def log_message(message, log_file="process_log.txt"):
    print(message)
    with open(log_file, "a") as file:
        file.write(message + "\n")



def download_file(url, tool_dir, filename):
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
        downloaded_file = os.path.join(tool_dir, filename)
        with open(downloaded_file, 'wb') as file:
            file.write(response.content)
        return downloaded_file
    except requests.RequestException as e:
        log_message(f"Download Error: Failed to download file: {e}")
        return None


def wait_for_file_release(file_path, timeout=30, interval=1):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with open(file_path, 'rb'):
                return True
        except IOError:
            time.sleep(interval)
    return False

def load_file(cyz2json_path, downloaded_file, json_file):
    try:
        subprocess.run(["dotnet", cyz2json_path, downloaded_file, "--output", json_file, "--metadatagreedy"], check=True)
    except subprocess.CalledProcessError as e:
        log_message(f"Processing Error: Failed to process file: {e}")


def to_listmode(json_file, listmode_file, imagedir='', segment_largest_object=False, diagnosticR2pngpath = "../../diagnosticR2.png"):
    try:
        data = json.load(open(json_file, encoding="utf-8-sig"))

        # ---- Detect image-protocol (IIFCheck) ----
        iif_check = (
            data.get("instrument", {})
                .get("measurementSettings", {})
                .get("CytoSettings", {})
                .get("IIFCheck", False)
        )
        # normalise bool-ish values
        if isinstance(iif_check, str):
            iif_check = iif_check.strip().lower() in ("true", "t", "1", "yes", "y")
        else:
            iif_check = bool(iif_check)

        # ---- Pull background for segmentation (your existing path) ----
        bg = None
        try:
            bg = (
                data.get("instrument", {})
                    .get("measurementSettings", {})
                    .get("CytoSettings", {})
                    .get("CytoSettings", {})
                    .get("iif", {})
                    .get("Background", {})
                    .get("Data", False)
            )
        except Exception:
            bg = None

        # ---- Pull image scale for µm conversion (from iif block) ----
        image_scale_um_per_px = None
        try:
            image_scale_um_per_px = (
                data.get("instrument", {})
                    .get("measurementSettings", {})
                    .get("CytoSettings", {})
                    .get("CytoSettings", {})
                    .get("iif", {})
                    .get("ImageScaleMuPerPixelP", None)
            )
            if image_scale_um_per_px is not None:
                image_scale_um_per_px = float(image_scale_um_per_px)
        except Exception:
            image_scale_um_per_px = None

        # ---- Build listmode rows (your existing extract) ----
        lines = extract(
            particles=data["particles"],
            dateandtime=data["instrument"]["measurementResults"]["start"],
            images=data.get("images", []),
            save_images_to=imagedir,
            segment_largest_object=segment_largest_object,
            background=bg,
            image_scale_um_per_px=image_scale_um_per_px,
        )

        df = pd.DataFrame(lines)
        df.to_csv(listmode_file, index=False)

        # ------------------------------------------------------------------
        # FWScalibration persistence (ONLY for image-protocol runs with images)
        # ------------------------------------------------------------------
        ENABLE_FWS_CALIBRATIONS = True  # << feature toggle

        if ENABLE_FWS_CALIBRATIONS and iif_check:
            print('Storing image calibration')
            # Need both columns present and enough non-null pairs

            calib = compute_fws_binned_calibration_from_df(
                df,
                fws_col="FWS_total",
                diam_um_col="img_equiv_diameter_um",
                min_per_bin=8,
                diagnostic_png_path=diagnosticR2pngpath,
                )            

            if calib is not None:
                # identify file_id and timestamp
                file_id = Path(json_file).stem.replace(".cyz", "")
                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

                rec = {
                    "time_calculated": ts,
                    "file_id": file_id,
                    "protocol": "imageprotocol",
                    "image_scale_um_per_px": image_scale_um_per_px,
                    **calib,
                }

                # choose a stable location under the tool dir
                out_dir = Path.home() / "Documents" / "flowcytometertool" / "FWScalibrations"
                out_path = out_dir / "FWScalibrations.jsonl"
                append_fwscalibration_record(rec, out_path)

    except subprocess.CalledProcessError as e:
        log_message(f"Processing Error: Failed to process file: {e}")
        


def apply_python_model(listmode_file, predictions_file, model_path):
    """
    Apply the ACTIVE (pinned) model only.
    'model_path' is ignored to prevent accidental defaults.
    """
    from custom_functions_for_python import loadClassifier
    import pandas as pd

    # Always resolve the pinned model directory
    model_dir = os.path.dirname(resolve_active_model_path())

    try:
        model, classes, features = loadClassifier(model_dir)
        df = pd.read_csv(listmode_file)
        df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
        df = df.dropna()

        print("Your model expects these columns:", features)
        print("Your data file has these columns:", df.columns.tolist())

        try:
            df = df[features]
        except Exception:
            print("Getting a not-in-index error? Columns in this data file don't match the model's training features.")

        print("Predicting ...")
        predictions = model.predict(df[features])
        proba_predict = pd.DataFrame(model.predict_proba(df[features]))
        predicted_data = df
        predicted_data['predicted_label'] = predictions
        proba_predict = proba_predict.set_axis(classes, axis=1)
        full_predicted = pd.concat([predicted_data, proba_predict], axis=1)
        full_predicted.to_csv(predictions_file)
        log_message(f"Prediction Success: Predictions saved to {predictions_file}")
    except Exception as e:
        log_message(f"Prediction Error: Failed to apply Python model: {e}")




def delete_file(path):
    try:
        if not os.path.exists(path):
            log_message(f"Path not found: {path}")
            return

        if os.path.isfile(path):
            os.remove(path)
            log_message(f"Deleted file: {path}")

        elif os.path.isdir(path):
            shutil.rmtree(path)
            log_message(f"Deleted directory and contents: {path}")

        else:
            log_message(f"Unknown path type, not deleted: {path}")

    except Exception as e:
        log_message(f"Error deleting path {path}: {e}")

def convert_json_to_listmode(output_path):
    for root, _, files in os.walk(output_path):
        for file in files:
            if file.lower().endswith(".json"):
                json_file = os.path.join(root, file)
                listmode_file = os.path.splitext(json_file)[0] + ".csv"
                try:
                    with open(json_file, encoding="utf-8-sig") as f:
                        data = json.load(f)
                    lines = extract(
                        particles=data["particles"],
                        dateandtime=data["instrument"]["measurementResults"]["start"],
                        images='',
                        save_images_to=''
                    )
                    df = pd.DataFrame(lines)
                    df.to_csv(listmode_file, index=False)
                    print(f"Converted: {json_file} → {listmode_file}")
                except Exception as e:
                    print(f"Error processing file: {json_file}")
                    print(f"Exception: {e}")

def combine_csv_files(output_path):
    variation_pattern = re.compile(r'_(\w+)\.cyz\.csv$')
    all_data = []
    for root, _, files in os.walk(output_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                match = variation_pattern.search(file)
                if match:
                    label = match.group(1)
                    df['source_label'] = label
                    all_data.append(df)
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.columns = combined_df.columns.str.replace(r'\s+', '_', regex=True)
        combined_df = combined_df.dropna()
        return combined_df
    else:
        return None



def choose_zone_folders(output_path):
    folders = [name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))]
    zonechoice = simpledialog.askstring("Zone Choice", f"Choose a zone from: {', '.join(folders)}")
    return zonechoice


def compute_consensual_labels_and_sample_weights(
    data: pd.DataFrame,
    *,
    # identity columns
    filename_col: str = "filename",
    id_col: str = "id",
    # labels & weights
    label_col: str = "source_label",
    weight_col: str = "weight",
    # Adoption policy
    apply_only_on_unanimous: bool = False,   # keep originals unless 100% agreement
    # Safety/diagnostics during development
    assert_preserve_order: bool = True,
    assert_no_row_count_change: bool = True,
) -> pd.DataFrame:
    """
    Compute consensus per *physical particle* identified by (filename + id),
    preserving the original row order and length.

    Returns the same dataframe plus:
        - 'consensus_label'  (weighted mode within each (file,id) group)
        - 'sample_weight'    (max_label_weight / total_weight in group, ∈ [0,1])

    If `apply_only_on_unanimous=True`, `label_col` is overwritten only where
    `sample_weight == 1.0`; otherwise labels are not changed.
    """

    if id_col not in data.columns:
        raise KeyError(f"Column '{id_col}' not found.")
    if label_col not in data.columns:
        raise KeyError(f"Column '{label_col}' not found.")
    if weight_col not in data.columns:
        raise KeyError(f"Column '{weight_col}' not found.")

    # We require a file-level identity so that IDs from different files don't collide.
    filename_candidates = [filename_col, "source_file"]
    real_filename_col = next((c for c in filename_candidates if c in data.columns), None)
    if real_filename_col is None:
        raise KeyError(
            f"No filename column found. Expected one of: {filename_candidates}. "
            f"Ensure the combine step adds a file identifier column."
        )

    df = data.copy()
    original_index = df.index.copy()
    original_len = len(df)

    # Normalize text columns used in grouping
    df[real_filename_col] = df[real_filename_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()

    # Build unique identity per physical particle
    df["_uid"] = df[real_filename_col] + "_" + df[id_col].astype(str).str.strip()

    # --- Fast path: if each (_uid) appears once and all weights==1, stamp outputs
    counts = df["_uid"].value_counts(dropna=False)
    is_singleton = (counts.max() == 1)
    if is_singleton and df[weight_col].fillna(1.0).eq(1.0).all():
        df["consensus_label"] = df[label_col]
        df["sample_weight"] = 1.0
        if apply_only_on_unanimous:
            # identical outcome; nothing to change
            pass
        if assert_preserve_order:
            assert df.index.equals(original_index), "Index changed unexpectedly (fast-path)."
        if assert_no_row_count_change:
            assert len(df) == original_len, "Row count changed unexpectedly (fast-path)."
        df.drop(columns=["_uid"], inplace=True)
        return df

    # --- General path: compute weighted mode within each (_uid)
    from collections import Counter

    records = []
    for uid, grp in df.groupby("_uid", sort=False):
        labels = grp[label_col].tolist()
        weights = grp[weight_col].astype(float).fillna(0.0).tolist()

        acc = Counter()
        total_w = 0.0
        for l, w in zip(labels, weights):
            acc[l] += w
            total_w += w

        if not acc:
            # fallback: no weights—keep first label, zero share
            consensus_label = labels[0] if labels else ""
            share = 0.0
        else:
            max_w = max(acc.values())
            # deterministic tie-break on label lexicographic order
            winners = sorted([k for k, v in acc.items() if v == max_w])
            consensus_label = winners[0]
            share = (max_w / total_w) if total_w > 0 else 0.0

        records.append((uid, consensus_label, share))

    cdf = pd.DataFrame(records, columns=["_uid", "consensus_label", "sample_weight"]).set_index("_uid")

    # LEFT-JOIN back to preserve order and cardinality
    df = df.join(cdf, on="_uid", how="left")

    # Optionally adopt consensus where (and only where) unanimous
    if apply_only_on_unanimous:
        unanimous = df["sample_weight"].ge(0.999999999)
        df.loc[unanimous, label_col] = df.loc[unanimous, "consensus_label"]

    # Safety checks
    if assert_preserve_order:
        assert df.index.equals(original_index), "Index changed during consensus join."
    if assert_no_row_count_change:
        assert len(df) == original_len, "Row count changed during consensus join."

    df.drop(columns=["_uid"], inplace=True)
    return df


def build_consensual_dataset(base_path, expertise_levels, zonechoice, prompt_merge_fn = None, premerge_plot_fn=None, delete_labels_fn=None):
    """
    Build a consensual dataset from flow cytometry CSV files.
    
    Parameters:
    - base_path: str, the base directory containing subfolders for each person.
    - expertise_levels: dict, a dictionary with expertise levels as keys and lists of people as values.
    
    Returns:
    - pd.DataFrame, the combined DataFrame with consensus labels and sample weights.
    """

    variation_pattern = re.compile(r'_(\w+)\.cyz\.csv$')
    all_data = []
    expertise_weights = {'expert': 3, 'advanced': 2, 'non_expert': 1}
    print(os.path.join(base_path,zonechoice))
    # Traverse the directory structure
    for root, _, files in os.walk(os.path.join(base_path,zonechoice)):
        for file in files:
            print(file)
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                base = os.path.basename(file_path)
                m = re.match(r"(.+?)_\w+\.cyz\.csv$", base, flags=re.IGNORECASE)
                cyz_base = m.group(1) if m else os.path.splitext(base)[0]
                try:
                    df = pd.read_csv(file_path)
                    if "filename" not in df.columns:
                        df["filename"] = cyz_base                    
                except:
                    print(f"Skipping empty or malformed file: {file_path}")
                    continue
                match = variation_pattern.search(file)
                if match:
                    label = match.group(1)
                    person = os.path.basename(os.path.dirname(root))
                    df['source_label'] = label
                    df['person'] = person
                    all_data.append(df)
    
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    try:
        if premerge_plot_fn is not None:
            premerge_plot_fn(combined_df)
        else:
            # default behavior if no callback was provided:
            # save next to the working directory as a one-off
            default_out = os.path.join(os.path.expanduser("~"), "Documents",
                                       "flowcytometertool", "Training plots",
                                       "premerge_3d_fluorescence.html")
            os.makedirs(os.path.dirname(default_out), exist_ok=True)
            plot_3d_fluorescence_premerge(
                combined_df, label_col="source_label", out_html=default_out
            )
    except Exception as e:
        print(f"[warn] pre-merge 3D plot not created: {e}")

    if delete_labels_fn is not None:
        delete_labels_fn(combined_df)

    if prompt_merge_fn is not None:
        prompt_merge_fn(combined_df)
    
    combined_df.columns = combined_df.columns.str.replace(r'\s+', '_', regex=True)
    combined_df = combined_df.dropna()
    print(combined_df)
    
    # Flatten the expertise_levels into a person-to-weight mapping
    person_to_weight = {
        person: expertise_weights[level]
        for level, people in expertise_levels.items()
        for person in people
    }

    # Assign person weights
    combined_df['weight'] = combined_df['person'].map(person_to_weight).fillna(1)

    # Compute consensus label per particls - this was indeed where the labels were shuffled , because it was not respecting particle ID 1 from file 1 is not the same as particle ID 1 from file 2
    print(combined_df)
    combined_df = compute_consensual_labels_and_sample_weights(combined_df)
    combined_df['source_label'] = combined_df['consensus_label']
    print(combined_df)
    return combined_df


def plot_cv_results(cv_results, plots_dir):
    plotlist = []
    best_results = cv_results[cv_results['iter'] == cv_results['iter'].max()].groupby(
        ['param_classifier', 'outer_splits']
    ).apply(lambda x: x.loc[x['mean_test_score'].idxmax()])
    for outer in cv_results['outer_splits'].unique():
        outer_data = cv_results[cv_results['outer_splits'] == outer]
        outer_score = round(outer_data['outer_split_test_score'].unique()[0], 3)
        best_params = best_results[best_results['outer_splits'] == outer][[
            'param_classifier',
            'param_classifier__learning_rate',
            'param_classifier__max_depth',
            'param_classifier__max_features',
            'param_classifier__C',
            'param_classifier__l1_ratio',
            'param_classifier__max_samples'
        ]].values[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.lineplot(data=outer_data, x='iter', y='mean_test_score', hue='param_classifier', marker='o', ax=ax)
        # Compute and plot delta MCC
        for classifier in outer_data['param_classifier'].unique():
            clf_data = outer_data[outer_data['param_classifier'] == classifier].sort_values('iter')
            clf_data['delta_mcc'] = clf_data['mean_test_score'].diff()
            sns.lineplot(data=clf_data, x='iter', y='delta_mcc', label=f"{classifier} ΔMCC", linestyle='--', ax=ax)
        ax.set_title(f"Outer split {outer}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MCC and ΔMCC")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Classifier")
        fig.text(0.5, -0.1, f"Best Classifier (used in outer CV) : {best_params}\nOuter CV test score : {outer_score}",
                 wrap=True, horizontalalignment='center', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'cv_results_outer_{outer}.png'))
        plt.close(fig)
        plotlist.append(fig)
    return plotlist



def plot_classifier_props(cv_results):
    plotlist = []
    best_results = cv_results[cv_results['iter'] == cv_results['iter'].max()].groupby(['param_classifier', 'outer_splits']).apply(lambda x: x.loc[x['mean_test_score'].idxmax()])

    for outer in cv_results['outer_splits'].unique():
        outer_score = round(cv_results[cv_results['outer_splits'] == outer]['outer_split_test_score'].unique()[0], 3)
        best_params = best_results[best_results['outer_splits'] == outer][['param_classifier','param_classifier__learning_rate','param_classifier__max_depth','param_classifier__max_features','param_classifier__C','param_classifier__l1_ratio','param_classifier__max_samples']].values[0]
        
        plt.figure(figsize=(12, 8))
        sns.histplot(data=cv_results[cv_results['outer_splits'] == outer], x='iter', hue='param_classifier', multiple='stack')
        plt.title(f"Outer split {outer}")
        plt.xlabel("Iteration")
        plt.ylabel("Proportion of candidates")
        plt.xticks(rotation=45)
        plt.legend(title="Classifier")
        plt.figtext(0.5, -0.1, f"Best Classifier (used in outer CV) : {best_params}\nOuter CV test score : {outer_score}", wrap=True, horizontalalignment='center', fontsize=10)
        plt.tight_layout()
        plotlist.append(plt)
        plt.show()

    return plotlist


def plot_all_hyperpars_combi_and_classifiers_scores(cv_results, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    def plot_all_hyperpars_combi(cv_results, classifier_name, hyperparameters):
        def plot_hyperpar_combi(cv_results, classifier_name, x_axis, y_axis):
            filtered_results = cv_results.copy()
            if x_axis == "degree" or y_axis == "degree":
                filtered_results = filtered_results[filtered_results['param_classifier__kernel'] == "poly"]
            filtered_results = filtered_results[filtered_results['param_classifier'] == classifier_name]
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = sns.scatterplot(
                data=filtered_results,
                x=x_axis,
                y=y_axis,
                hue='mean_test_score',
                palette='viridis',
                size='mean_test_score',
                sizes=(20, 200),
                ax=ax
            )
            if x_axis in ["C", "gamma", "learning_rate"]:
                ax.set_xscale('log')
            if y_axis in ["C", "gamma", "learning_rate"]:
                ax.set_yscale('log')
            ax.set_xlabel(x_axis.replace("_", " "))
            ax.set_ylabel(y_axis.replace("_", " "))
            ax.set_title(f"{classifier_name} - {x_axis} vs {y_axis}")
            ax.legend(title="Mean MCC")
            fig.tight_layout()
            return fig
        grid = [(x, y) for x in hyperparameters for y in hyperparameters if x != y]
        plot_list = [plot_hyperpar_combi(cv_results, classifier_name, x, y) for x, y in grid]
        return plot_list
    logreg_hyperpars = ["param_classifier__C", "param_classifier__l1_ratio"]
    rf_hyperpars = ["param_classifier__max_features", "param_classifier__max_samples"]
    hgb_hyperpars = ["param_classifier__max_depth", "param_classifier__max_features", "param_classifier__learning_rate"]
    classifiers_hyperpars = {
        "LogisticRegression": logreg_hyperpars,
        "RandomForestClassifier": rf_hyperpars,
        "HistGradientBoostingClassifier": hgb_hyperpars
    }
    for classifier_name, hyperparameters in classifiers_hyperpars.items():
        print(f"Plotting hyperparameter combinations for {classifier_name}")
        plot_list = plot_all_hyperpars_combi(cv_results, classifier_name, hyperparameters)
        for i, fig in enumerate(plot_list):
            fig.savefig(os.path.join(plots_dir, f'{classifier_name}_plot_{i+1}.png'))
            plt.close(fig)



def train_classifier(df, plots_dir, model_path, max_per_class, calibration_enabled = False):
    from custom_functions_for_python import buildSupervisedClassifier, loadClassifier
    df = stratified_subsample(df, target_column="source_label", max_per_class=max_per_class)
    df["group"] = df.index # This means no grouping. i.e. it does not matter which file the particle label came from.
    cleaned_df = df[[col for col in df.columns if col not in ["filename","consensus_label","datetime", "user_id", "location"]]] #cleaned_df = df[["source_label","group","weight","Fl_Yellow_total",  "Fl_Red_total",  "Fl_Orange_total"]]
    print('cleaned_df.columns')
    print(cleaned_df.columns)
    # Detect if running from PyInstaller bundle
    is_frozen = getattr(sys, 'frozen', False)
    # Detect if running on Linux
    is_linux = platform.system().lower() == "linux"
    # Set cores to 1 if on Linux (to avoid joblib memory leak from actions workflow) or frozen executable which similarly does not seem to work parallelised
    cores = 1 if is_frozen or is_linux else os.cpu_count()
    print('cores:')
    print(cores)

    
    # Split the data
    train_df, test_df = train_test_split(cleaned_df, test_size=0.2, stratify=cleaned_df["source_label"], random_state=42)
    
    # Train on training set
    buildSupervisedClassifier(
        training_set=train_df,
        target_name="source_label",
        group_name="group",
        weight_name="weight",
        select_K=5,
        cores=cores,
        n_sizes=4,
        filename_cvResults=os.path.join(os.path.dirname(model_path),"cv_results" + os.path.basename(model_path) + ".csv"),
        filename_learningCurve=os.path.join(os.path.dirname(model_path),"learning_curve" + os.path.basename(model_path) + ".csv"),
        filename_finalFittedModel=model_path,
        filename_finalCalibratedModel=model_path+".probabilistic.pkl",
        filename_importance = os.path.join(os.path.dirname(model_path), "permutation_importance_" + os.path.basename(model_path) + ".csv"),
        validation_set = test_df,
        plots_dir = plots_dir,
        calibration_enabled = calibration_enabled
    )

    # Evaluate on test set
    model, classes, features = loadClassifier(os.path.dirname(model_path))
    test_df_filtered=test_df[features]
    predictions = model.predict(test_df_filtered)
    proba_predict = pd.DataFrame(model.predict_proba(test_df_filtered)) # compute class prediction probabilities and store in data frame
    predicted_data = test_df
    # Add prediction to original test table
    predicted_data['predicted_label'] = predictions 
    # Make the column names of this data frame the class names (instead of numbers)
    proba_predict = proba_predict.set_axis(classes, axis=1)
    # Bind both data frames by column
    full_predicted = pd.concat([predicted_data, proba_predict], axis=1)
    # Save final predicted table
    #full_predicted.to_csv(predict_name)        
    print("Test Set Evaluation:\n", classification_report(test_df["source_label"], predictions))

    # Confusion Matrix
    cm = confusion_matrix(test_df["source_label"], predictions)
    print("Confusion Matrix:\n", cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(plots_dir, f'confusionmatrix_{os.path.basename(model_path)}.png'))
    plt.show()
    
    
    cv_results = pd.read_csv(os.path.join(os.path.dirname(model_path),"cv_results" + os.path.basename(model_path) + ".csv"))

    try:
        plot_cv_results(cv_results,plots_dir)
        plot_classifier_props(cv_results)
        plot_all_hyperpars_combi_and_classifiers_scores(cv_results,plots_dir)
    except Exception as e:
        print(f"Could not plot CV results: {e}")

def test_model(df, model_path):
    from custom_functions_for_python import loadClassifier
    model_dir = active_model_dir()
    model, classes, features = loadClassifier(model_dir)
    df=df[features]
    predictions = model.predict(df[features])
    proba_predict = pd.DataFrame(model.predict_proba(df[features])) # compute class prediction probabilities and store in data frame
    predicted_data = df
    # Add prediction to original test table
    predicted_data['predicted_label'] = predictions 
    # Make the column names of this data frame the class names (instead of numbers)
    proba_predict = proba_predict.set_axis(classes, axis=1)
    # Bind both data frames by column
    full_predicted = pd.concat([predicted_data, proba_predict], axis=1)
    # Save final predicted table
    #full_predicted.to_csv(predict_name) 
    df['predicted_label'] = predictions
    summary = df['predicted_label'].value_counts().to_string()
    return df, summary


# functions.py (new/updated parts)
import os
from urllib.parse import urlparse
from storage_clients import _split_blob_url, get_container_client, get_blob_client

def download_blobs(blob_url: str, download_path: str, sas_token=None):
    """
    New behavior: ignores sas_token; uses Entra ID (InteractiveBrowserCredential).
    Preserves 'public exampledata' anonymous access.
    """
    account_url, container_name, prefix = _split_blob_url(blob_url)
    anonymous = ("public/exampledata/" in blob_url)  # keep your current public behavior [1](https://cefas-my.sharepoint.com/personal/joseph_ribeiro_cefas_gov_uk/Documents/Microsoft%20Copilot%20Chat%20Files/functions.py)
    cc = get_container_client(account_url, container_name, anonymous=anonymous)
    os.makedirs(download_path, exist_ok=True)
    for blob in cc.list_blobs(name_starts_with=prefix):
        local_path = os.path.join(download_path, os.path.relpath(blob.name, prefix or "."))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fh:
            cc.download_blob(blob.name).readinto(fh)

def list_blobs(container_url: str, sas_token):
    """
    New behavior: ignores sas_token; uses Entra ID.
    """
    account_url, container_name, _ = _split_blob_url(container_url)
    cc = get_container_client(account_url, container_name, anonymous=False)
    return [b.name for b in cc.list_blobs()]

def upload_to_blob(file_path: str, sas_token, container: str, output_blob_folder: str):
    """
    New behavior: ignores sas_token; uses Entra ID.
    'container' param is the *source container URL* (e.g., https://.../<container>).
    We upload into 'output_blob_folder' within the same storage account.
    """
    account_url, src_container, _ = _split_blob_url(container)
    # Destination container name is output_blob_folder; blob name is the basename
    bc = get_blob_client(account_url, output_blob_folder, os.path.basename(file_path), anonymous=False)
    with open(file_path, "rb") as data:
        bc.upload_blob(data, overwrite=True)

def mix_blob_files(container: str, sas_token, output_blob_folder: str, sample_rate=0.005):
    """
    New behavior: ignores sas_token; uses Entra ID.
    Same logic as before for reading CSVs and concatenating a sample.
    """
    import pandas as pd, tempfile
    account_url, container_name, _ = _split_blob_url(container)
    cc = get_container_client(account_url, container_name, anonymous=False)

    all_sampled = pd.DataFrame()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        out_csv = temp_file.name
        for blob in cc.list_blobs():
            if blob.name.endswith("_predictions.csv"):
                df = pd.read_csv(cc.get_blob_client(blob.name).download_blob())
                sampled = df.sample(frac=sample_rate)
                all_sampled = pd.concat([all_sampled, sampled], ignore_index=True)
        if not all_sampled.empty:
            all_sampled.to_csv(out_csv, index=False)
            # Upload mixed file
            upload_to_blob(out_csv, sas_token=None, container=container, output_blob_folder=output_blob_folder)

def convert_cyz_to_json(input_dir, output_dir, dll_path):
    import os, subprocess
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".cyz"):
                print(file)
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, input_dir)
                rel_dir = os.path.dirname(rel_path)
                dst_dir = os.path.join(output_dir, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst_file = os.path.join(dst_dir, file + ".json")
                subprocess.run(["dotnet", dll_path, full_path, "--output", dst_file, "--metadatagreedy"], check=True)


def compile_cyz2json_from_release(cyz2json_dir, path_entry):
    if os.path.exists(cyz2json_dir):
        print("Info: cyz2json already exists in " + cyz2json_dir)
        return
    try:
        os.makedirs(cyz2json_dir, exist_ok=True)
        zip_path = os.path.join(os.path.dirname(cyz2json_dir), "cyz2json.zip")
        # Detect OS and choose appropriate release
        system = platform.system().lower()
        if system == "windows":
            zip_url = "https://github.com/OBAMANEXT/cyz2json/releases/download/0.0.14/cyz2json-windows-latest.zip"
        elif system == "linux":
            zip_url = "https://github.com/OBAMANEXT/cyz2json/releases/download/0.0.14/cyz2json-ubuntu-latest.zip"
        else:
            raise RuntimeError(f"Unsupported OS: {system}")
        print(f"Downloading cyz2json for {system}...")
        subprocess.run(["curl", "-L", "-o", zip_path, zip_url], check=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cyz2json_dir)
        if path_entry:
            path_entry.delete(0, tk.END)
            path_entry.insert(0, os.path.join(cyz2json_dir, "bin", "Cyz2Json.dll"))
    except subprocess.CalledProcessError as e:
        print(f"Compilation Error: Failed to download cyz2json: {e}.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, sub_item in enumerate(v):
                if isinstance(sub_item, dict):
                    items.extend(flatten_dict(sub_item, f"{new_key}_{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}_{i}", sub_item))
        else:
            items.append((new_key, v))
    return dict(items)

def dict_to_csv(data, output_file):
    flattened_data = [flatten_dict(item) for item in data] if isinstance(data, list) else [flatten_dict(data)]    
    header = set()
    for item in flattened_data:
        header.update(item.keys())
    header = sorted(header)
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for item in flattened_data:
            writer.writerow(item)
    print(f"Data saved to {output_file}")


def clear_temp_folder(tool_dir):
    """Clear the temporary directory."""
    for filename in os.listdir(tool_dir):
        file_path = os.path.join(tool_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")


def compile_r_requirements(r_dir, rpath_entry):
    """Get R requirements"""
#    if os.path.exists(r_dir):
#        messagebox.showinfo("Info", "r installation already exists in " + r_dir)
#        return
    try:
        subprocess.run(["curl", "https://cran.r-project.org/bin/windows/base/old/4.3.3/R-4.3.3-win.exe", "--output", r_dir+"/R-4.3.3-win.exe"], check=True)
        subprocess.run([r_dir+"/R-4.3.3-win.exe", "/DIR="+r_dir], cwd=r_dir, check=True)
        subprocess.run([r_dir+"/bin/Rscript.exe", "./install_rpackages.R"], check=True)
        rpath_entry.delete(0, tk.END)
        rpath_entry.insert(0, os.path.join(r_dir, "bin", "Rscript.exe"))
        messagebox.showinfo("Download Success", f"R downloaded and libraries installed")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Compilation Error", f"Failed to compile r: {e}.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

def apply_r(listmode_file, predictions_file, rpath_entry):
    """Convert .cyz file to .json using cyz2json tool."""
    try:
        print(rpath_entry)
        print(listmode_file)
        print(predictions_file)
        subprocess.run([rpath_entry, "rf_predict.R", "final_rf_model.rds", listmode_file, predictions_file], check=True)
        messagebox.showinfo("Success", f"R applied successfully")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Processing Error", f"Failed to process file: {e}. Is R installed here?")

def select_output_dir(app):
    """Open a dialog to select the output directory."""
    app.output_dir = filedialog.askdirectory()
    if app.output_dir:
        messagebox.showinfo("Output Directory Selected", f"Output files will be saved in: {app.output_dir}")

def load_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def select_particles(json_data, particle_ids):
    particles = [p for p in json_data['particles'] if p['particleId'] in particle_ids]
    return particles if particles else None

def get_pulses(particles):
    pulses = {p['particleId']: p.get('pulseShapes') for p in particles}
    return pulses



def display_image(self,root,current_image_index, output_dir, image_label, tif_files, metadata, confidence_entry, species_entry):
    """Display the image and update metadata entry fields."""
    image_file = tif_files[current_image_index]
    image_path = os.path.join(output_dir, image_file)
    
    img = Image.open(image_path)
    img = img.resize((400, 400), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)

    if self.image_label is None:
        self.image_label = tk.Label(self.root, image=img_tk)
        self.image_label.image = img_tk
        self.image_label.pack(pady=10)
    else:
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    # Load saved metadata if it exists
    metadata = self.metadata.get(image_file, {"confidence": "", "species": ""})
    self.confidence_entry.delete(0, tk.END)
    self.confidence_entry.insert(0, metadata["confidence"])
    self.species_entry.delete(0, tk.END)
    self.species_entry.insert(0, metadata["species"])



def update_navigation_buttons(prev_button, next_button, current_image_index, total_images):
    """Update the state of navigation buttons based on the current image index."""
    prev_button.config(state=tk.NORMAL if current_image_index > 0 else tk.DISABLED)
    next_button.config(state=tk.NORMAL if current_image_index < total_images - 1 else tk.DISABLED)


def save_metadata(current_image_index, tif_files, metadata, confidence_entry, species_entry, output_dir):
    """Save metadata to a CSV file."""
    image_file = tif_files[current_image_index]
    confidence = confidence_entry.get()
    species = species_entry.get()
    metadata[image_file] = {"confidence": confidence, "species": species}

    metadata_file_path = os.path.join(output_dir, "label_data.csv")
    with open(metadata_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image File", "confidence", "Suspected Species"])
        for image, data in metadata.items():
            writer.writerow([image, data["confidence"], data["species"]])





def plot_3d_fluorescence_premerge(df, label_col, out_html):
    """
    Create a 3D fluorescence scatter of the raw (pre-merge) training data.
    Colors and shapes by `label_col` to inform merging decisions.
    Adds legend entries per class for quick filtering.
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import webbrowser

    # Accept either spaced, underscored, or dotted column names
    candidate_names = [
        ("FWS_total", "Fl Red_total", "Fl Orange_total"),
        ("FWS_total", "Fl_Red_total", "Fl_Orange_total"),
        ("FWS_total", "Fl Red_total", "Fl.Orange_total"),
        ("FWS_total", "Fl.Red_total", "Fl.Orange_total"),
    ]
    triplet = None
    for cand in candidate_names:
        if all(c in df.columns for c in cand):
            triplet = cand
            break
    if triplet is None:
        raise ValueError(
            "Could not find fluorescence columns. "
            "Looked for variants of Yellow/Red/Orange *_total."
        )
    fx, fy, fz = triplet

    # Keep required columns; drop rows with missing values
    work = df[[fx, fy, fz, label_col]].dropna().copy()

    # Downsample for responsiveness (tweak if needed)
    max_points = 120_000
    if len(work) > max_points:
        work = work.sample(n=max_points, random_state=42)

    # Axis clipping at 99.5th percentile (like your overlap tool)
    x99 = np.percentile(work[fx], 99.5)
    y99 = np.percentile(work[fy], 99.5)
    z99 = np.percentile(work[fz], 99.5)

    # Deterministic color palette (simple HUSL-like wheel)
    classes = sorted(work[label_col].astype(str).unique())
    def husl_palette(n):
        return [f"hsl({int(360*i/n)}, 65%, 50%)" for i in range(n)]
    palette = husl_palette(len(classes))
    color_map = dict(zip(classes, palette))
    work["_color"] = work[label_col].astype(str).map(color_map)

    # 🔷 Symbol cycling (broad set; friendly to 3D scatter)
    base_symbols = ['circle', 'circle-open', 'cross', 'diamond',
            'diamond-open', 'square', 'square-open', 'x']
            
    # Repeat/cycle to cover all classes
    sym_list = (base_symbols * ((len(classes) // len(base_symbols)) + 1))[:len(classes)]
    symbol_map = dict(zip(classes, sym_list))
    work["_symbol"] = work[label_col].astype(str).map(symbol_map)

    # One big data trace (fast) with per-point colors & symbols
    scatter = go.Scatter3d(
        x=work[fx],
        y=work[fy],
        z=work[fz],
        mode="markers",
        marker=dict(
            size=3,
            color=work["_color"],
            symbol=work["_symbol"],
            opacity=0.65,
            line=dict(width=0.3, color="rgba(20,20,20,0.4)")
        ),
        text=work[label_col].astype(str),
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"{fx}: %{{x:.2f}}<br>"
            f"{fy}: %{{y:.2f}}<br>"
            f"{fz}: %{{z:.2f}}<br>"
            "<extra></extra>"
        ),
        name="Raw training points",
        showlegend=False  # legend handled by tiny class traces below
    )

    # Legend entries: one tiny invisible-in-scene trace per class
    legend_traces = []
    for cls in classes:
        legend_traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(
                    size=6,
                    color=color_map[cls],
                    symbol=symbol_map[cls],
                    line=dict(width=1, color="rgba(20,20,20,0.6)")
                ),
                name=str(cls),
                showlegend=True
            )
        )

    fig = go.Figure(data=[scatter] + legend_traces)
    fig.update_layout(
        title=f"3D Fluorescence (pre-merge) — colored by {label_col}",
        height=800,
        scene=dict(
            xaxis=dict(range=[0, x99], title=fx),
            yaxis=dict(range=[0, y99], title=fy),
            zaxis=dict(range=[0, z99], title=fz),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.5))
        ),
        legend=dict(
            title="Classes",
            itemsizing="trace",
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.6)"
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    # Save + auto-open
    pio.write_html(fig, file=out_html, auto_open=False)
    webbrowser.open("file://" + os.path.abspath(out_html))
    return out_html


def FWS_size_plot_3d_fluorescence_premerge(
    df,
    label_col,
    out_html,
    *,
    size_log=True,          # log-scale SWS sizes for better spread
    size_min=2,             # minimum marker size (px)
    size_max=10,            # maximum marker size (px)
    size_clip_pct=99.0      # clip SWS at this percentile before scaling
):
    """
    Create a 3D fluorescence scatter of the raw (pre-merge) training data.
    Colors and shapes by `label_col` to inform merging decisions.
    Marker size is driven by a fourth variable: total SWS.
    Adds legend entries per class for quick filtering.
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import webbrowser

    # --- Accept either spaced, underscored, or dotted column names for fluorescence ---
    fl_candidate_triplets = [
        ("Fl Yellow_total", "Fl Red_total", "Fl Orange_total"),
        ("Fl_Yellow_total", "Fl_Red_total", "Fl_Orange_total"),
        ("Fl Yellow_total", "Fl Red_total", "Fl.Orange_total"),
        ("Fl.Yellow_total", "Fl.Red_total", "Fl.Orange_total"),
    ]
    triplet = None
    for cand in fl_candidate_triplets:
        if all(c in df.columns for c in cand):
            triplet = cand
            break
    if triplet is None:
        raise ValueError(
            "Could not find fluorescence columns. "
            "Looked for variants of Yellow/Red/Orange *_total."
        )
    fx, fy, fz = triplet

    # --- Fourth variable (SWS) for marker size: try common variants ---
    sws_candidates = [ "FWS_total"    ]
    size_col = None
    for c in sws_candidates:
        if c in df.columns:
            size_col = c
            break
    if size_col is None:
        raise ValueError(
            "Could not find SWS/side-scatter column. "
            "Tried: " + ", ".join(sws_candidates)
        )

    # --- Keep required columns; drop rows with missing values ---
    work = df[[fx, fy, fz, size_col, label_col]].dropna().copy()

    # --- Downsample for responsiveness (tweak if needed) ---
    max_points = 120_000
    if len(work) > max_points:
        work = work.sample(n=max_points, random_state=42)

    # --- Axis clipping at 99.5th percentile (like your overlap tool) ---
    x99 = np.percentile(work[fx], 99.5)
    y99 = np.percentile(work[fy], 99.5)
    z99 = np.percentile(work[fz], 99.5)

    # --- Compute marker sizes from SWS ---
    sws_vals = work[size_col].astype(float).to_numpy()
    upper = np.percentile(sws_vals, size_clip_pct)  # robust upper-clip
    sws_clipped = np.minimum(sws_vals, upper)

    if size_log:
        sws_trans = np.log10(1.0 + np.maximum(sws_clipped, 0.0))
    else:
        sws_trans = np.maximum(sws_clipped, 0.0)

    vmin = float(np.min(sws_trans))
    vmax = float(np.max(sws_trans))
    if vmax > vmin:
        sizes = size_min + (sws_trans - vmin) * (size_max - size_min) / (vmax - vmin)
    else:
        sizes = np.full_like(sws_trans, (size_min + size_max) / 2.0)

    # --- Deterministic color palette (simple HUSL-like wheel) ---
    classes = sorted(work[label_col].astype(str).unique())

    def husl_palette(n):
        return [f"hsl({int(360*i/n)}, 65%, 50%)" for i in range(n)]

    palette = husl_palette(len(classes))
    color_map = dict(zip(classes, palette))
    work["_color"] = work[label_col].astype(str).map(color_map)

    # --- Symbol cycling ---
    base_symbols = [
        "circle", "circle-open", "cross", "diamond",
        "diamond-open", "square", "square-open", "x"
    ]
    sym_list = (base_symbols * ((len(classes) // len(base_symbols)) + 1))[:len(classes)]
    symbol_map = dict(zip(classes, sym_list))
    work["_symbol"] = work[label_col].astype(str).map(symbol_map)

    # --- Main data trace ---
    scatter = go.Scatter3d(
        x=work[fx],
        y=work[fy],
        z=work[fz],
        mode="markers",
        marker=dict(
            size=sizes,                  # <- size from SWS
            color=work["_color"],
            symbol=work["_symbol"],
            opacity=0.65,
            line=dict(width=0.3, color="rgba(20,20,20,0.4)")
        ),
        text=work[label_col].astype(str),
        # IMPORTANT: escape Plotly placeholders in f-strings using DOUBLE BRACES
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"{fx}: %{{x:.2f}}<br>"
            f"{fy}: %{{y:.2f}}<br>"
            f"{fz}: %{{z:.2f}}<br>"
            f"{size_col} (scaled): %{{marker.size:.2f}} px<br>"
            "<extra></extra>"
        ),
        name="Raw training points",
        showlegend=False
    )

    # --- Legend entries: class-only traces ---
    legend_traces = []
    for cls in classes:
        legend_traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(
                    size=6,
                    color=color_map[cls],
                    symbol=symbol_map[cls],
                    line=dict(width=1, color="rgba(20,20,20,0.6)")
                ),
                name=str(cls),
                showlegend=True
            )
        )

    title_suffix = (
        f" — colored by {label_col}, size by {size_col}"
        + (" (log scaled)" if size_log else "")
    )

    fig = go.Figure(data=[scatter] + legend_traces)
    fig.update_layout(
        title=f"3D Fluorescence (pre-merge){title_suffix}",
        height=800,
        scene=dict(
            xaxis=dict(range=[0, x99], title=fx),
            yaxis=dict(range=[0, y99], title=fy),
            zaxis=dict(range=[0, z99], title=fz),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.5))
        ),
        legend=dict(
            title="Classes",
            itemsizing="trace",
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.6)"
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    pio.write_html(fig, file=out_html, auto_open=False)
    webbrowser.open("file://" + os.path.abspath(out_html))
    return out_html


def plot3d(predictions_file):
    data = pd.read_csv(predictions_file)
    data['category'] = data['predicted_label']
    unique_categories = data['category'].unique()

    preset_colors = {
        'rednano': 'red',
        'orapicoprok': 'orange',
        'micro': 'blue',
        'beads': 'green',
        'oranano': 'purple',
        'noise': 'gray',
        'C_undetermined': 'black',
        'redpico': 'pink'
    }

    color_map = {
        category: preset_colors.get(
            category,
            f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
        ) for category in unique_categories
    }
    data['color'] = data['category'].map(color_map)

    x_99 = np.percentile(data['Fl.Yellow_total'], 99.5)
    y_99 = np.percentile(data['Fl.Red_total'], 99.5)
    z_99 = np.percentile(data['Fl.Orange_total'], 99.5)

    scatter = go.Scatter3d(
        x=data['Fl.Yellow_total'],
        y=data['Fl.Red_total'],
        z=data['Fl.Orange_total'],
        mode='markers',
        marker=dict(size=5, color=data['color'], showscale=False),
        text=data['category'],
        name='Data Points'
    )

    camera = dict(
        eye=dict(x=-1.5, y=-1.5, z=1.5),  
        center=dict(x=0, y=0, z=0),        
        up=dict(x=0, y=0, z=1)            
    )
    fig = go.Figure(data=[scatter])

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, x_99], title='Fl.Yellow_total'),
            yaxis=dict(range=[0, y_99], title='FL.Red_total'),
            zaxis=dict(range=[0, z_99], title='FL.Orange_total'),
            camera=camera
        ),
        title='3D Data Points'
    )
    pio.write_html(fig, file=predictions_file+"_3d.html", auto_open=True)

    print("Plot saved as '3D_Plot.html'.")

def summarize_predictions(df, pumped_volume):
    """Generate a summary of labelled and predicted data counts."""
    summary = []
    labels = df['label'].dropna().unique() if 'label' in df.columns else []
    preds = df['predicted_label'].dropna().unique() if 'predicted_label' in df.columns else []
    all_classes = set(labels).union(preds)
    for cls in all_classes:
        label_count = (df['label'] == cls).sum() / pumped_volume if 'label' in df.columns else 0
        pred_count = (df['predicted_label'] == cls).sum() / pumped_volume if 'predicted_label' in df.columns else 0
        percent = (pred_count / label_count * 100) if label_count else 0
        summary.append((cls, label_count, pred_count, f"{percent:.2f}%"))
    return summary
    
def upload_to_blob_path(file_path: str, container_url: str, blob_path: str):
    """
    Upload a local file to the *same container as container_url* at the given blob_path.
    Example:
      container_url = 'https://<acct>.blob.core.windows.net/trainedmodels'
      blob_path     = '<version>/final_model_<version>.pkl'
    """
    from storage_clients import _split_blob_url, get_blob_client
    account_url, container_name, _ = _split_blob_url(container_url)
    bc = get_blob_client(account_url, container_name, blob_path, anonymous=False)
    with open(file_path, "rb") as data:
        bc.upload_blob(data, overwrite=True, timeout=900, max_concurrency=4)


def _push_model_artifacts_to_models_container(version: str, file_paths: list[str], container_url: str):
    """
    Upload each path in file_paths into the 'models' container under '<version>/'.
    """
    for p in file_paths:
        if p and os.path.exists(p):
            blob_path = f"{version}/{os.path.basename(p)}"
            upload_to_blob_path(p, container_url=container_url, blob_path=blob_path)

# --- Safety sweep for Combine CSVs: zone metadata collection & validation ---
from pathlib import Path
from typing import Dict, Any, List, Tuple

def _load_grablist_file(grablist_path: str | None) -> List[str]:
    """
    Read a grablist file (one dotted JSON path per line), ignoring empty and comment lines.
    If not found / None -> return empty list (graceful no-op).
    """
    items: List[str] = []
    if not grablist_path:
        return items
    p = Path(grablist_path)
    if not p.exists():
        return items
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                items.append(s)
    return items

def _extract_dotted(js: Dict[str, Any], dotted: str):
    """Follow a dotted path 'a.b.c' in a nested dict; return None if any segment missing."""
    cur = js
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def _iter_zone_jsons(base_path: str, zonechoice: str) -> List[Path]:
    root = Path(base_path) / zonechoice
    return [p for p in root.rglob("*.json") if p.is_file()]

def collect_zone_metadata_and_assert(
    dataseturl: str,
    repo_root,
    base_path: str,
    zonechoice: str,
    *,
    grablist_path: str | None = "grablist.txt",
    modeltrainsettings_out: str | None = None,
    packets_out_dir: str | None = None,
    nogui: bool = False,
) -> Dict[str, Any]:
    """
    Sweep all JSONs in the selected zone, assert critical sameness, and
    return a dict suitable for writing to modeltrainsettings.json.

    If instrument.serialNumber or ...CytoSettings.PMTlevels_str vary -> raises ValueError
    with exact message required by the user.
    """
    import json
    from metadata_extraction import extract_metadata  # re-use your extractor
    # ---------- discover files ----------
    json_files = _iter_zone_jsons(base_path, zonechoice)
    if not json_files:
        # Nothing to validate; return empty but benign structure
        return {
            "zonechoice": zonechoice,
            "files_checked": 0,
            "instrument": {"serialNumber": None, "PMTlevels_str": None},
            "grablist": {"consistent": {}, "inconsistent": []},
            "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }

    # ---------- collect critical props ----------
    serials, pmt_levels = set(), set()
    # Grablist handling (graceful if file missing)
    gl_items = _load_grablist_file(grablist_path)
    per_key_values: Dict[str, List[Any]] = {k: [] for k in gl_items}

    # Optionally write thin per-file packets
    packets_dir = Path(packets_out_dir) if packets_out_dir else None
    if packets_dir:
        packets_dir.mkdir(parents=True, exist_ok=True)

    for jp in json_files:
        with jp.open("r", encoding="utf-8-sig") as f:
            js = json.load(f)
        inst = js.get("instrument", {})
        # Critical sameness
        serials.add(inst.get("serialNumber"))
        pmt = (
            inst.get("measurementSettings", {})
               .get("CytoSettings", {})
               .get("PMTlevels_str")
        )
        pmt_levels.add(pmt)
        # Grablist capture
        for k in gl_items:
            per_key_values[k].append(_extract_dotted(js, k))
        # Optional: metadata packet per file (for auditing / provenance)
        if packets_dir:
            meta = extract_metadata(cyz_json_path=str(jp))
            outp = packets_dir / (jp.stem + "_meta.json")
            with outp.open("w", encoding="utf-8") as fo:
                json.dump(meta, fo, indent=2, default=str)

    # ---------- assert sameness on critical props ----------
    # NB: if any missing/None slipped in, sets will have {None, ...} -> treated as inconsistent
    if len(serials) != 1 or len(pmt_levels) != 1:
        msg = "all samples must share the same instrument.serialNumber and instrument.measurementSettings.CytoSettings.PMTlevels_str"
        try:
            if not nogui:
                from tkinter import messagebox
                messagebox.showerror("Combine blocked", msg)
        finally:
            raise ValueError(msg)

    serial = next(iter(serials))
    pmt_str = next(iter(pmt_levels))

    # ---------- compute grablist consistency ----------
    consistent: Dict[str, Any] = {}
    inconsistent: List[str] = []
    for k, vals in per_key_values.items():
        # treat all-equal including None as "consistent" only if *not* all None
        uniq = set(vals)
        if len(uniq) == 1 and list(uniq)[0] is not None:
            consistent[k] = vals[0]
        else:
            inconsistent.append(k)

    settings = {
        "dataset URL": dataseturl,
        "zonechoice": zonechoice,
        "files_checked": len(json_files),
        "instrument": {
            "serialNumber": serial,
            "measurementSettings": {
                "CytoSettings": {"PMTlevels_str": pmt_str}
            },
        },
        "grablist": {
            "consistent": consistent,           # these are written into settings for retention
            "inconsistent": sorted(inconsistent)  # flags only; does not stop training
        },
        "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }

    out_path = Path(modeltrainsettings_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # Git repo SHA since we are in active development
    # -----------------------------------------
    try:
        print(repo_root)
        print(os.getcwd())
        
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()        
    except Exception as e:
        print(e)
        head_sha = "unknown"

    # -----------------------------------------
    # Collect labelled CYZ file names + MD5
    # -----------------------------------------
    labelled_cyz_info = []

    for cyz_path in glob.glob(os.path.join(base_path,zonechoice, "**", "*.cyz"), recursive=True):
        labelled_cyz_info.append({
            "filename": os.path.basename(cyz_path),
            "md5": md5_of_file(cyz_path)
        })
    
    settings["labelled_cyz_files"] = labelled_cyz_info
    settings["git_sha"] = head_sha

    with out_path.open("w", encoding="utf-8") as fo:
        json.dump(settings, fo, indent=2)    
    

    return settings
    
    
def run_backend_only():
    print("🔧 Running in no-GUI mode...")

    # Setup paths
    tool_dir = os.path.expanduser("~/Documents/flowcytometertool/")
    download_path = os.path.join("exampledata/")
    output_path = os.path.join("extraction/")
    cyz2json_dir = os.path.join(tool_dir, "cyz2json")
    model_dir = os.path.join(tool_dir, "models")
    plots_dir = os.path.join(tool_dir, "plots")
    model_path = os.path.join(model_dir, f'final_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    try:
        # 1. Download Files
        blob_url = "https://citprodflowcytosa.blob.core.windows.net/public/exampledata/"
        print("⬇️ Downloading files...")
        download_blobs(blob_url, download_path)

        # 2. Download cyz2json
        print("📦 Installing requirements...")
        compile_cyz2json_from_release(cyz2json_dir, None)

        # 3. Cyz2json
        print("🔄 Converting CYZ to JSON...")
        convert_cyz_to_json(download_path, output_path, os.path.join(cyz2json_dir, "Cyz2Json.dll"))

        # 4. To listmode
        print("📄 Converting JSON to listmode...")
        convert_json_to_listmode(output_path)

        # 5. Combine CSVs
        print("📊 Combining CSV files...")
        df = combine_csvs(output_path, expertise_matrix_path, nogui=True, premerge_plot_fn= False)
        if df is None:
            print("⚠️ No CSV files found.")
            return

        # 6. Train Model
        print("🤖 Training model...")
        
        train_model(df, plots_dir, model_path, nogui=True, self = None, max_per_class = 1000)
    
        # 7. Predict Test Set using updated function
        print("🧪 Predicting test set...")
        from custom_functions_for_python import predictTestSet

        predict_name = os.path.join(tool_dir, "test_predictions.csv")
        cm_filename = os.path.join(tool_dir, "confusion_matrix.csv")
        report_filename = os.path.join(tool_dir, "classification_report.csv")
        text_file_path = os.path.join(tool_dir, "prediction_log.txt")

        with open(text_file_path, "w") as text_file:
            predictTestSet(
                self=None,
                model_path=model_path,
                predict_name=predict_name,
                data=df,
                target_name="source_label",
                weight_name="weight",
                cm_filename=cm_filename,
                report_filename=report_filename,
                text_file=text_file
            )
        print("✅ Test set predictions completed and saved.")

    except Exception as e:
        print(f"❌ Error during headless execution: {e}")
        raise