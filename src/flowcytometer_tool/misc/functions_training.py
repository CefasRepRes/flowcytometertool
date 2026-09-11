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
    update_modeltrainsettings,
    update_modeltrainsettings_bead_flag,
)
from flowcytometer_tool.tabs.continuous_sample_analyser.json_safe import json_safe
 
from flowcytometer_tool.tabs.continuous_sample_analyser.bead_calibration import (
    load_latest_beadscalibration_record,
    apply_saved_bead_calibration_to_dataframe,
)

from flowcytometer_tool.tabs.continuous_sample_analyser.protocols import detect_sampling_protocol
import yaml

from flowcytometer_tool.misc.functions_model_selection import (
    resolve_active_raw_model_path,
)
from flowcytometer_tool.misc.functions_runtime import (
    build_consensual_dataset,
    choose_zone_folders,
)

__all__ = [
    "stratified_subsample",
    "compile_cyz2json",
    "md5_of_file",
    "train_model",
    "test_classifier",
    "combine_csvs",
    "nn_homogenize_df",
    "sample_rows",
    "train_classifier",
    "test_model",
    "collect_zone_metadata_and_assert",
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
_SELECTED_MODEL_DIR = _SELECTED_UNCALIBRATED_MODEL_DIR
_CONFIG_PATH = _RUNTIME_CONFIG.paths.config_path

# Default trained-models container (adjust if you use a different one)
_DEFAULT_TRAINED_MODELS_CONTAINER = _RUNTIME_CONFIG.options.default_trained_models_container_url

import json, datetime, os
from pathlib import Path

import math

def stratified_subsample(df, target_column, max_per_class=1000):

    sampled = []

    for _, grp in df.groupby(target_column):
        sampled.append(
            grp.sample(
                n=min(len(grp), max_per_class),
                random_state=42
            )
        )

    return pd.concat(sampled, ignore_index=True)


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


def train_model(
    df,
    plots_dir,
    model_path,
    rootdir,
    nogui=False,
    self=None,
    calibration_enabled=False,
    max_per_class=1000,
    bead_samples=None,
):
    """
    Train and publish a model.

    Important behaviour:
    - calibration_enabled controls probabilistic calibration of the trained classifier.
    - bead_samples control bead calibration of the training dataframe.
    - If valid bead_samples are supplied, the dataframe passed to train_classifier is
      replaced with the bead-calibrated, feature-limited dataframe returned by
      prepare_training_dataframe_with_optional_bead_calibration.
    - If bead_samples are supplied but no valid bead calibration can be applied,
      training stops rather than silently training on the raw 135/141-column dataframe.
    """

    # --- PRE-FLIGHT AUTH: trigger AAD prompt pre training ---
    cred = get_credential()

    # Request a token for Azure Storage so the browser pops up immediately.
    try:
        cred.get_token("https://storage.azure.com/.default")
    except Exception as e:
        # If the prompt is cancelled, fail fast so the user tries again.
        if not nogui:
            from tkinter import messagebox
            messagebox.showerror(
                "Sign-in required",
                f"Azure sign-in is required to train & publish models.\n\n{e}",
            )
        raise

    # Keep your existing timestamped naming. This string is your model version.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # When invoked from the UI, override model_path to include the timestamped version.
    if self is not None and hasattr(self, "tool_dir"):
        self.model_path = os.path.join(self.tool_dir, f"models/final_model_{ts}.pkl")
        model_path = self.model_path

    try:
        if df is None or df.empty:
            if nogui:
                print("Error: No data to train on.")
            else:
                from tkinter import messagebox
                messagebox.showerror("Error", "No data to train on.")
            return

        print("START OF train_model")
        print(f"bead_samples supplied={bead_samples is not None}")
        print(f"incoming df shape={df.shape}")
        print("incoming df columns:")
        print(list(df.columns))
        probabilistic_calibration_enabled = bool(calibration_enabled)
        print(f"probabilistic_calibration_enabled={probabilistic_calibration_enabled}")

        # Start from a copy so we do not mutate self.df unexpectedly.
        train_df = df.copy()
        used_bead_calibration = False
        bead_meta = {
            "bead_sample_count": 0,
            "calibrated_columns": [],
        }

        # Bead calibration is independent of probabilistic calibration.
        if bead_samples:
            train_df, used_bead_calibration, bead_meta = prepare_training_dataframe_with_optional_bead_calibration(
                train_df,
                bead_samples=bead_samples,
                allow_spoof_bead_calibration=True, # Dangerous... remember to turn this off again if you set it to True! Ideally this should be a GUI checkbox.
                diagnostic_dir=plots_dir,
                beads_output_root=getattr(self, "tool_dir", None) if self is not None else None,
            )

            print("AFTER bead calibration/preparation")
            print(f"used_bead_calibration={used_bead_calibration}")
            print(f"bead_sample_count={bead_meta.get('bead_sample_count', 0)}")
            print("calibrated_columns:")
            print(bead_meta.get("calibrated_columns", []))
            print(f"prepared train_df shape={train_df.shape}")
            print("prepared train_df columns:")
            print(list(train_df.columns))

            # Fail closed. Do not silently train on raw wide data when bead samples exist.
            if not used_bead_calibration:
                msg = (
                    "Bead samples were supplied, but no "
                    "valid bead calibration was applied. Stopping to avoid training "
                    "on the raw/unfiltered dataframe."
                )
                if nogui:
                    raise RuntimeError(msg)
                else:
                    from tkinter import messagebox
                    messagebox.showerror("Bead calibration failed", msg)
                    return

            # Save as bead-calibrated model.
            model_path = beadcalibrated_model_path(model_path)
            if self is not None:
                self.model_path = model_path

        # Extra defensive check.
        # If bead calibration was used, these columns should not survive.
        if used_bead_calibration:
            forbidden_prefixes = (
                "Curvature_",
                "Sidewards_Scatter_",
                "Forward_Scatter_Left_",
                "Forward_Scatter_Right_",
            )
            forbidden_cols = [
                c for c in train_df.columns
                if str(c).startswith(forbidden_prefixes)
            ]

            if forbidden_cols:
                msg = (
                    "Bead-calibrated training dataframe still contains forbidden "
                    f"raw/non-model columns: {forbidden_cols[:30]}"
                )
                if len(forbidden_cols) > 30:
                    msg += f" ... plus {len(forbidden_cols) - 30} more"
                raise RuntimeError(msg)

        # --- train exactly as before, but on train_df, not df ---
        train_classifier(
            train_df,
            plots_dir,
            model_path,
            max_per_class,
            calibration_enabled=probabilistic_calibration_enabled,
        )

        # Infer version from filename 'final_model_<VERSION>.pkl'
        m = re.search(
            r"final_model_(\d{8}_\d{6})(?:_beadcalibrated)?\.pkl$",
            os.path.basename(model_path),
        )
        version = m.group(1) if m else ts

        # Resolve sibling artifact file paths written by train_classifier(...)
        model_dir = os.path.dirname(model_path)
        model_basename = os.path.basename(model_path)

        cv_results_csv = os.path.join(
            model_dir,
            "cv_results" + model_basename + ".csv",
        )
        learning_curve_csv = os.path.join(
            model_dir,
            "learning_curve" + model_basename + ".csv",
        )
        perm_importance_csv = os.path.join(
            model_dir,
            "permutation_importance_" + model_basename + ".csv",
        )
        confusion_png = os.path.join(
            model_dir,
            "confusionmatrix_" + model_basename + ".png",
        )
        modeltrainsettingsjson = os.path.join(model_dir, "modeltrainsettings.json")
        modeltrainenv = os.path.join(rootdir, "environment.yml")
        modelcalibrated = model_path + ".probabilistic.pkl"

        update_modeltrainsettings_bead_flag(
            modeltrainsettingsjson,
            used_bead_calibration=used_bead_calibration,
            bead_sample_count=bead_meta.get("bead_sample_count", 0),
        )
        
        try:
            update_modeltrainsettings(
                modeltrainsettingsjson,
                {
                    "bead_calibration_mode": bead_meta.get("calibration_mode", "none"),
                    "used_spoof_bead_calibration": bool(
                        bead_meta.get("used_spoof_bead_calibration", False)
                    ),
                    "calibrated_columns": bead_meta.get("calibrated_columns", []),
                    "bead_calibration_status": bead_meta.get("calibration_status", "unknown"),
                    "beads_calibration_plot": bead_meta.get("beads_calibration_plot"),
                    "beads_calibration_record": bead_meta.get("beads_calibration_record"),
                },
            )

        except Exception as e:
            print(f"[warn] could not write detailed bead calibration metadata: {e}")        

        label_change_log_path = getattr(self, "label_change_log_path", None)

        artifacts = [
            model_path,
            cv_results_csv,
            learning_curve_csv,
            perm_importance_csv,
            confusion_png,
            modeltrainsettingsjson,
            modelcalibrated,
            expertise_matrix_path,
            modeltrainenv,
            label_change_log_path,
        ]

        # Only attempt upload when called from the app.
        if self is not None and hasattr(self, "url_entry_blob"):
            container_url = _DEFAULT_TRAINED_MODELS_CONTAINER

            from flowcytometer_tool.misc.functions_blob_and_utils import _push_model_artifacts_to_models_container

            _push_model_artifacts_to_models_container(
                version=version,
                file_paths=artifacts,
                container_url=container_url,
            )

            uploaded_count = len([p for p in artifacts if p and os.path.exists(p)])
            msg = f"Pushed {uploaded_count} artifacts to blob: models/{version}/"

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

        return model_path

    except Exception as e:
        if nogui:
            print(f"Training Error: Failed to train model: {e}")
        else:
            from tkinter import messagebox
            messagebox.showerror("Training Error", f"Failed to train model: {e}")
        raise


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
    print('... running combine_csvs')
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
            grablist_path=os.path.join("flowcytometer_tool", "config", "grablist.txt"),              # graceful no-op if missing
            modeltrainsettings_out=modeltrainsettings_out,               
            packets_out_dir=packets_dir,
            nogui=nogui,
        )
    except ValueError:
        print('Mismatch of serialNumber / PMTlevels_str - blocking the csv combine')
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
            
            mts=json_safe(mts)

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



def train_classifier(df, plots_dir, model_path, max_per_class, calibration_enabled = False):
    from flowcytometer_tool.tabs.download_train.custom_functions_for_python import buildSupervisedClassifier, loadClassifier
    
    print("START OF train_classifier")
    print(df.columns.tolist())    
    df = stratified_subsample(df, target_column="source_label", max_per_class=max_per_class)
    print("AFTER stratified_subsample")
    print(df.columns.tolist())    
    df["group"] = df.index # This means no grouping. i.e. it does not matter which file the particle label came from.
    cleaned_df = df[[col for col in df.columns if col not in ["filename","consensus_label","datetime", "user_id", "location",'terminal_classes', 'n_terminal_classes', 'filename', 'person', 'consensus_label', 'sample_weight']]] #cleaned_df = df[["source_label","group","weight","Fl_Yellow_total",  "Fl_Red_total",  "Fl_Orange_total"]]
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

    #try: # one of these plots is broken
    #    plot_cv_results(cv_results,plots_dir) 
    #    plot_classifier_props(cv_results) # Likely one of these functions that is broken
    #    plot_all_hyperpars_combi_and_classifiers_scores(cv_results,plots_dir) # Likely one of these functions that is broken
    #except Exception as e:
    #    print(f"Could not plot CV results: {e}")

def test_model(df, model_path):
    from flowcytometer_tool.tabs.download_train.custom_functions_for_python import loadClassifier
    model_dir = os.path.dirname(resolve_active_raw_model_path())
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
    grablist_path: str | None = os.path.join("flowcytometer_tool", "config", "grablist.txt"),
    modeltrainsettings_out: str | None = None,
    packets_out_dir: str | None = None,
    nogui: bool = False,
) -> Dict[str, Any]:
    """
    Sweep all JSONs in the selected zone, assert critical sameness, and
    return a dict suitable for writing to modeltrainsettings.json.

    Very verbose diagnostic version.

    This function deliberately prints a lot of information so that failures in
    modeltrainsettings.json population can be traced. In particular, it reports:

      - the resolved base path and zone path
      - how many JSON files were discovered by _iter_zone_jsons()
      - which JSON files are being inspected
      - whether instrument.serialNumber exists in each JSON
      - whether instrument.measurementSettings.CytoSettings.PMTlevels_str exists
      - what grablist file was loaded
      - how git SHA resolution was attempted
      - what repo_root and current working directory were used
      - how many labelled CYZ files were found
      - exactly where modeltrainsettings.json is written

    If instrument.serialNumber or
    instrument.measurementSettings.CytoSettings.PMTlevels_str vary across files,
    raises ValueError with the exact message required by the user.
    """

    import json
    import os
    import glob
    import shutil
    import subprocess
    import datetime
    import traceback
    from pathlib import Path
    from typing import Dict, Any, List

    from flowcytometer_tool.tabs.continuous_sample_analyser.metadata_extraction import (
        extract_metadata,
    )

    def _vprint(message: str) -> None:
        """
        Local verbose logger for this function.

        Kept deliberately simple because this is a diagnostic path and should
        work even before the wider app logging system is configured.
        """
        print(f"[collect_zone_metadata_and_assert] {message}")

    def _repr_path(value) -> str:
        """
        Return a safe printable representation of a path-like value.
        """
        try:
            return str(Path(value).resolve())
        except Exception:
            return str(value)

    def _safe_json_get_instrument_fields(js: Dict[str, Any], jp: Path) -> tuple[Any, Any]:
        """
        Extract the two critical instrument fields with verbose diagnostics.

        Returns:
            (serial_number, pmt_levels_str)
        """
        inst = js.get("instrument", None)

        if inst is None:
            _vprint(f"WARNING: file has no top-level 'instrument' key: {jp}")
            return None, None

        if not isinstance(inst, dict):
            _vprint(
                f"WARNING: top-level 'instrument' is not a dict in {jp}; "
                f"type={type(inst).__name__!r}, value={inst!r}"
            )
            return None, None

        serial = inst.get("serialNumber", None)

        if serial is None:
            _vprint(f"WARNING: instrument.serialNumber is missing or None in {jp}")
        else:
            _vprint(f"instrument.serialNumber in {jp.name}: {serial!r}")

        measurement_settings = inst.get("measurementSettings", None)
        if measurement_settings is None:
            _vprint(
                f"WARNING: instrument.measurementSettings is missing or None in {jp}"
            )
            return serial, None

        if not isinstance(measurement_settings, dict):
            _vprint(
                f"WARNING: instrument.measurementSettings is not a dict in {jp}; "
                f"type={type(measurement_settings).__name__!r}, "
                f"value={measurement_settings!r}"
            )
            return serial, None

        cyto_settings = measurement_settings.get("CytoSettings", None)
        if cyto_settings is None:
            _vprint(
                f"WARNING: instrument.measurementSettings.CytoSettings "
                f"is missing or None in {jp}"
            )
            return serial, None

        if not isinstance(cyto_settings, dict):
            _vprint(
                f"WARNING: instrument.measurementSettings.CytoSettings is not a dict "
                f"in {jp}; type={type(cyto_settings).__name__!r}, "
                f"value={cyto_settings!r}"
            )
            return serial, None

        pmt = cyto_settings.get("PMTlevels_str", None)

        if pmt is None:
            _vprint(
                f"WARNING: instrument.measurementSettings.CytoSettings.PMTlevels_str "
                f"is missing or None in {jp}"
            )
        else:
            _vprint(
                f"instrument.measurementSettings.CytoSettings.PMTlevels_str "
                f"in {jp.name}: {pmt!r}"
            )

        return serial, pmt

    def _resolve_git_sha_verbose(repo_root_value) -> tuple[str, Dict[str, Any]]:
        """
        Resolve the current git SHA with detailed diagnostics.

        The previous implementation ran git rev-parse HEAD without specifying
        cwd, so it depended on os.getcwd(). That can easily fail if training is
        launched from a different working directory.

        This version tries repo_root first, then the current working directory
        as a fallback. It records every attempt in git_debug.
        """
        git_debug: Dict[str, Any] = {
            "repo_root_input": str(repo_root_value),
            "repo_root_resolved": None,
            "current_working_directory": os.getcwd(),
            "git_executable": shutil.which("git"),
            "attempts": [],
        }

        candidate_dirs: List[tuple[str, Path | None]] = []

        try:
            if repo_root_value is not None:
                repo_root_path = Path(repo_root_value).expanduser().resolve()
                git_debug["repo_root_resolved"] = str(repo_root_path)
                candidate_dirs.append(("repo_root", repo_root_path))
            else:
                _vprint("WARNING: repo_root is None; cannot try git from repo_root")
        except Exception as e:
            git_debug["repo_root_resolve_error"] = repr(e)
            _vprint(f"WARNING: could not resolve repo_root={repo_root_value!r}: {e!r}")

        try:
            candidate_dirs.append(("current_working_directory", Path(os.getcwd()).resolve()))
        except Exception as e:
            git_debug["cwd_resolve_error"] = repr(e)
            candidate_dirs.append(("current_working_directory", None))

        if git_debug["git_executable"] is None:
            _vprint("WARNING: git executable was not found on PATH")
        else:
            _vprint(f"git executable found at: {git_debug['git_executable']}")

        for label, candidate_dir in candidate_dirs:
            attempt: Dict[str, Any] = {
                "label": label,
                "cwd": str(candidate_dir) if candidate_dir is not None else None,
                "success": False,
                "stdout": None,
                "stderr": None,
                "error": None,
            }

            if candidate_dir is None:
                attempt["error"] = "candidate_dir was None"
                git_debug["attempts"].append(attempt)
                continue

            _vprint(f"Attempting git SHA lookup using {label}: {candidate_dir}")

            try:
                completed = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(candidate_dir),
                    capture_output=True,
                    text=True,
                    check=False,
                )

                attempt["returncode"] = completed.returncode
                attempt["stdout"] = completed.stdout.strip()
                attempt["stderr"] = completed.stderr.strip()

                if completed.returncode == 0 and completed.stdout.strip():
                    sha = completed.stdout.strip()
                    attempt["success"] = True
                    git_debug["attempts"].append(attempt)
                    _vprint(f"Resolved git SHA from {label}: {sha}")
                    return sha, git_debug

                _vprint(
                    f"git rev-parse failed using {label}; "
                    f"returncode={completed.returncode}, "
                    f"stderr={completed.stderr.strip()!r}"
                )

            except Exception as e:
                attempt["error"] = repr(e)
                attempt["traceback"] = traceback.format_exc()
                _vprint(
                    f"Exception while resolving git SHA using {label}: {e!r}"
                )

            git_debug["attempts"].append(attempt)

        _vprint("WARNING: could not resolve git SHA; using 'unknown'")
        return "unknown", git_debug

    _vprint("Starting metadata collection")
    _vprint(f"dataseturl: {dataseturl!r}")
    _vprint(f"repo_root input: {repo_root!r}")
    _vprint(f"repo_root resolved, if possible: {_repr_path(repo_root)}")
    _vprint(f"base_path input: {base_path!r}")
    _vprint(f"base_path resolved, if possible: {_repr_path(base_path)}")
    _vprint(f"zonechoice: {zonechoice!r}")
    _vprint(f"grablist_path input: {grablist_path!r}")
    _vprint(f"modeltrainsettings_out input: {modeltrainsettings_out!r}")
    _vprint(f"packets_out_dir input: {packets_out_dir!r}")
    _vprint(f"nogui: {nogui!r}")
    _vprint(f"os.getcwd(): {os.getcwd()}")

    zone_path = Path(base_path) / zonechoice
    _vprint(f"Expected selected zone path: {zone_path}")
    _vprint(f"Selected zone path exists: {zone_path.exists()}")
    _vprint(f"Selected zone path is dir: {zone_path.is_dir()}")

    if grablist_path is not None:
        try:
            gp = Path(grablist_path)
            _vprint(f"grablist_path resolved, if possible: {_repr_path(gp)}")
            _vprint(f"grablist_path exists: {gp.exists()}")
            _vprint(f"grablist_path is file: {gp.is_file()}")
        except Exception as e:
            _vprint(f"WARNING: could not inspect grablist_path: {e!r}")
    else:
        _vprint("grablist_path is None; grablist handling will use an empty/default list")

    # ---------- discover files ----------
    _vprint("Calling _iter_zone_jsons(base_path, zonechoice)")
    json_files = _iter_zone_jsons(base_path, zonechoice)

    try:
        json_files = list(json_files)
    except TypeError:
        _vprint(
            "WARNING: _iter_zone_jsons did not return an iterable that can be "
            "converted to a list. Re-raising."
        )
        raise

    _vprint(f"Number of JSON files discovered: {len(json_files)}")

    for idx, jp in enumerate(json_files, start=1):
        _vprint(f"JSON file {idx}/{len(json_files)}: {jp}")

    if not json_files:
        _vprint(
            "No JSON files were discovered. Returning benign empty settings. "
            "This means instrument info cannot be populated because no source "
            "JSONs were found for this base_path and zonechoice."
        )

        settings = {
            "dataset URL": dataseturl,
            "zonechoice": zonechoice,
            "files_checked": 0,
            "instrument": {
                "serialNumber": None,
                "PMTlevels_str": None,
            },
            "grablist": {
                "consistent": {},
                "inconsistent": [],
            },
            "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }

        head_sha, git_debug = _resolve_git_sha_verbose(repo_root)
        settings["git_sha"] = head_sha
        settings["git_sha_debug"] = git_debug
        settings["labelled_cyz_files"] = []

        settings = json_safe(settings)

        if modeltrainsettings_out:
            out_path = Path(modeltrainsettings_out)
            _vprint(f"Writing empty settings to: {out_path}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fo:
                json.dump(settings, fo, indent=2)
        else:
            _vprint(
                "modeltrainsettings_out is None or empty; not writing "
                "modeltrainsettings.json"
            )

        return settings

    # ---------- collect critical props ----------
    serials = set()
    pmt_levels = set()

    serials_by_file: Dict[str, Any] = {}
    pmt_levels_by_file: Dict[str, Any] = {}

    # Grablist handling, graceful if file missing
    _vprint("Loading grablist file")
    gl_items = _load_grablist_file(grablist_path)
    _vprint(f"Number of grablist items loaded: {len(gl_items)}")

    if gl_items:
        for idx, item in enumerate(gl_items, start=1):
            _vprint(f"grablist item {idx}/{len(gl_items)}: {item!r}")
    else:
        _vprint(
            "Grablist is empty. This may be expected if the file is missing "
            "or intentionally blank."
        )

    per_key_values: Dict[str, List[Any]] = {k: [] for k in gl_items}

    # Optionally write thin per-file packets
    packets_dir = Path(packets_out_dir) if packets_out_dir else None
    if packets_dir:
        _vprint(f"Metadata packets enabled. Output dir: {packets_dir}")
        packets_dir.mkdir(parents=True, exist_ok=True)
        _vprint(f"Metadata packets dir exists after mkdir: {packets_dir.exists()}")
    else:
        _vprint("Metadata packets disabled because packets_out_dir is None or empty")

    for file_index, jp in enumerate(json_files, start=1):
        jp = Path(jp)
        _vprint("")
        _vprint(f"Processing JSON {file_index}/{len(json_files)}: {jp}")
        _vprint(f"JSON exists: {jp.exists()}")
        _vprint(f"JSON is file: {jp.is_file()}")

        try:
            file_size = jp.stat().st_size
            _vprint(f"JSON size, bytes: {file_size}")
        except Exception as e:
            _vprint(f"WARNING: could not stat JSON file {jp}: {e!r}")

        try:
            with jp.open("r", encoding="utf-8-sig") as f:
                js = json.load(f)
            _vprint(f"Loaded JSON successfully: {jp}")
        except Exception as e:
            _vprint(f"ERROR: failed to load JSON file {jp}: {e!r}")
            raise

        if not isinstance(js, dict):
            _vprint(
                f"ERROR: JSON root is not a dict in {jp}; "
                f"type={type(js).__name__!r}"
            )
            raise ValueError(f"JSON root is not a dict: {jp}")

        _vprint(f"Top-level JSON keys in {jp.name}: {sorted(js.keys())}")

        serial, pmt = _safe_json_get_instrument_fields(js, jp)

        serials.add(serial)
        pmt_levels.add(pmt)

        serials_by_file[str(jp)] = serial
        pmt_levels_by_file[str(jp)] = pmt

        _vprint(f"Current unique serial values: {serials!r}")
        _vprint(f"Current unique PMTlevels_str values: {pmt_levels!r}")

        # Grablist capture
        if gl_items:
            _vprint(f"Extracting {len(gl_items)} grablist values from {jp.name}")

        for k in gl_items:
            try:
                value = _extract_dotted(js, k)
                per_key_values[k].append(value)
                _vprint(f"grablist value for {k!r} in {jp.name}: {value!r}")
            except Exception as e:
                _vprint(
                    f"WARNING: failed to extract grablist key {k!r} "
                    f"from {jp}: {e!r}"
                )
                per_key_values[k].append(None)

        # Optional: metadata packet per file, for auditing / provenance
        if packets_dir:
            _vprint(f"Creating metadata packet for: {jp}")

            try:
                meta = extract_metadata(cyz_json_path=str(jp))
                _vprint(f"extract_metadata succeeded for: {jp}")
            except Exception as e:
                _vprint(f"ERROR: extract_metadata failed for {jp}: {e!r}")
                raise

            try:
                meta = json_safe(meta)
            except Exception as e:
                _vprint(f"ERROR: json_safe(meta) failed for {jp}: {e!r}")
                raise

            outp = packets_dir / (jp.stem + "_meta.json")
            _vprint(f"Writing metadata packet to: {outp}")

            try:
                with outp.open("w", encoding="utf-8") as fo:
                    json.dump(meta, fo, indent=2, default=str)
                _vprint(f"Wrote metadata packet successfully: {outp}")
            except Exception as e:
                _vprint(f"ERROR: failed to write metadata packet {outp}: {e!r}")
                raise

    # ---------- assert sameness on critical props ----------
    _vprint("")
    _vprint("Completed JSON sweep")
    _vprint(f"Final unique instrument.serialNumber values: {serials!r}")
    _vprint(
        "Final unique "
        "instrument.measurementSettings.CytoSettings.PMTlevels_str values: "
        f"{pmt_levels!r}"
    )

    if len(serials) != 1 or len(pmt_levels) != 1:
        _vprint("ERROR: critical instrument sameness check failed")
        _vprint(f"serials_by_file: {serials_by_file!r}")
        _vprint(f"pmt_levels_by_file: {pmt_levels_by_file!r}")

        msg = (
            "all samples must share the same instrument.serialNumber and "
            "instrument.measurementSettings.CytoSettings.PMTlevels_str"
        )

        try:
            if not nogui:
                from tkinter import messagebox
                messagebox.showerror("Combine blocked", msg)
        finally:
            raise ValueError(msg)

    serial = next(iter(serials))
    pmt_str = next(iter(pmt_levels))

    _vprint(f"Critical sameness check passed")
    _vprint(f"Selected serialNumber for settings: {serial!r}")
    _vprint(f"Selected PMTlevels_str for settings: {pmt_str!r}")

    # ---------- compute grablist consistency ----------
    _vprint("")
    _vprint("Computing grablist consistency")

    consistent: Dict[str, Any] = {}
    inconsistent: List[str] = []

    for k, vals in per_key_values.items():
        try:
            uniq = set(vals)
        except TypeError:
            # Some values may be unhashable, for example dicts/lists.
            # Fall back to JSON string representation for consistency checking.
            uniq = set(json.dumps(json_safe(v), sort_keys=True, default=str) for v in vals)

        _vprint(f"Grablist key {k!r}: values={vals!r}")
        _vprint(f"Grablist key {k!r}: unique_count={len(uniq)}")

        # Treat all-equal including None as consistent only if not all None.
        if len(uniq) == 1 and vals and vals[0] is not None:
            consistent[k] = vals[0]
            _vprint(f"Grablist key {k!r} is consistent: {vals[0]!r}")
        else:
            inconsistent.append(k)
            _vprint(f"Grablist key {k!r} is inconsistent or all None")

    _vprint(f"Number of consistent grablist keys: {len(consistent)}")
    _vprint(f"Number of inconsistent grablist keys: {len(inconsistent)}")

    settings = {
        "dataset URL": dataseturl,
        "zonechoice": zonechoice,
        "files_checked": len(json_files),
        "instrument": {
            "serialNumber": serial,
            "measurementSettings": {
                "CytoSettings": {
                    "PMTlevels_str": pmt_str,
                },
            },
        },
        "grablist": {
            "consistent": consistent,
            "inconsistent": sorted(inconsistent),
        },
        "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }

    # -----------------------------------------
    # Git repo SHA since we are in active development
    # -----------------------------------------
    _vprint("")
    _vprint("Resolving git SHA")

    head_sha, git_debug = _resolve_git_sha_verbose(repo_root)

    settings["git_sha"] = head_sha
    settings["git_sha_debug"] = git_debug

    _vprint(f"git_sha to be written into settings: {head_sha!r}")

    # -----------------------------------------
    # Collect labelled CYZ file names + MD5
    # -----------------------------------------
    _vprint("")
    _vprint("Collecting labelled CYZ file names and MD5 hashes")

    labelled_cyz_info = []
    cyz_glob_pattern = os.path.join(base_path, zonechoice, "**", "*.cyz")

    _vprint(f"CYZ glob pattern: {cyz_glob_pattern!r}")

    cyz_paths = glob.glob(cyz_glob_pattern, recursive=True)
    _vprint(f"Number of CYZ files discovered: {len(cyz_paths)}")

    for cyz_index, cyz_path in enumerate(cyz_paths, start=1):
        _vprint(f"Processing CYZ {cyz_index}/{len(cyz_paths)}: {cyz_path}")

        try:
            cyz_md5 = md5_of_file(cyz_path)
            _vprint(f"MD5 for {cyz_path}: {cyz_md5}")
        except Exception as e:
            _vprint(f"ERROR: failed to calculate MD5 for {cyz_path}: {e!r}")
            raise

        labelled_cyz_info.append(
            {
                "filename": os.path.basename(cyz_path),
                "path": cyz_path,
                "md5": cyz_md5,
            }
        )

    settings["labelled_cyz_files"] = labelled_cyz_info

    # Extra diagnostics to help explain missing population issues later.
    settings["metadata_collection_debug"] = {
        "base_path_input": str(base_path),
        "base_path_resolved": _repr_path(base_path),
        "zonechoice": zonechoice,
        "zone_path": str(zone_path),
        "zone_path_exists": zone_path.exists(),
        "zone_path_is_dir": zone_path.is_dir(),
        "json_files": [str(p) for p in json_files],
        "serials_by_file": serials_by_file,
        "pmt_levels_by_file": pmt_levels_by_file,
        "grablist_path": str(grablist_path) if grablist_path is not None else None,
        "grablist_items": gl_items,
        "cyz_glob_pattern": cyz_glob_pattern,
        "cyz_files": cyz_paths,
        "current_working_directory": os.getcwd(),
    }

    _vprint("")
    _vprint("Converting settings through json_safe")

    try:
        settings = json_safe(settings)
    except Exception as e:
        _vprint(f"ERROR: json_safe(settings) failed: {e!r}")
        raise

    # -----------------------------------------
    # Write modeltrainsettings.json
    # -----------------------------------------
    _vprint("")
    _vprint("Writing model training settings")

    if not modeltrainsettings_out:
        _vprint(
            "modeltrainsettings_out is None or empty. Settings will be returned "
            "but no modeltrainsettings.json file will be written."
        )
        return settings

    out_path = Path(modeltrainsettings_out)
    _vprint(f"modeltrainsettings_out path: {out_path}")
    _vprint(f"modeltrainsettings_out parent: {out_path.parent}")

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _vprint(f"Ensured output parent directory exists: {out_path.parent}")
    except Exception as e:
        _vprint(
            f"ERROR: failed to create modeltrainsettings output parent "
            f"{out_path.parent}: {e!r}"
        )
        raise

    try:
        with out_path.open("w", encoding="utf-8") as fo:
            json.dump(settings, fo, indent=2)
        _vprint(f"Wrote modeltrainsettings.json successfully: {out_path}")
    except Exception as e:
        _vprint(f"ERROR: failed to write modeltrainsettings.json {out_path}: {e!r}")
        raise

    _vprint("Finished metadata collection successfully")

    return settings