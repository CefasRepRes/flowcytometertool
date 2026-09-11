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
from flowcytometer_tool.config.runtime import get_runtime_config
from flowcytometer_tool.core.file_processing import (
    build_processed_file_paths,
    build_protocol_packet,
    detect_protocol_from_json,
    normalise_dataframe_columns,
    wait_for_file_release as wait_for_file_release_utility,
)
from flowcytometer_tool.core.inference_orchestration import (
    apply_bead_calibration_for_model_features,
    latest_bead_calibration_status,
    load_classifier_from_exact_path,
    run_prediction,
)

# Cross-module dependencies introduced by splitting the original functions.py.
# These used to be available from the single shared module namespace.
from flowcytometer_tool.misc.functions_model_selection import (
    append_fwscalibration_record,
    compute_fws_binned_calibration_from_df,
    resolve_active_beadcalibrated_model_path,
    resolve_active_model_path,
    resolve_active_raw_model_path,
)
from flowcytometer_tool.misc.functions_visualisation import (
    plot_3d_fluorescence_premerge,
)

__all__ = [
    "FileHandler",
    "extract_processed_url",
    "log_message",
    "download_file",
    "wait_for_file_release",
    "to_listmode",
    "apply_python_model",
    "delete_file",
    "combine_csv_files",
    "choose_zone_folders",
    "build_consensual_dataset",
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
            file_paths = build_processed_file_paths(file_path, self.output_folder)
            if not wait_for_file_release(file_paths.source_cyz_path):
                log_message(f"Timeout: File still locked after waiting: {file_path}")
                return

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    load_file(self.cyz2json_path, file_paths.source_cyz_path, file_paths.json_path)
                    break
                except Exception as e:
                    log_message(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(4)
                    else:
                        raise

            log_message(f"Success: Cyz2json applied {file_path}")
            to_listmode(
                file_paths.json_path,
                file_paths.listmode_csv_path,
                file_paths.image_dir_path,
                True,
                diagnosticR2pngpath=os.path.join(self.output_folder, "calibrationcurve.png"),
            )
            log_message(f"Success: Listmode applied {file_path}")

            # Detect sampling protocol before deciding how to proceed
            try:
                detected_protocol, protocol_packet = detect_protocol_from_json(
                    file_paths.json_path,
                    detect_sampling_protocol,
                )
            except Exception as e:
                detected_protocol = "unknownprotocol"
                protocol_packet = build_protocol_packet({})
                log_message(f"Warning: could not detect sampling protocol for {file_path}: {e}")

            if detected_protocol == "beadsprotocol":
                # Bead sample: compute and persist calibration, then skip classification
                try:
                    from flowcytometer_tool.tabs.continuous_sample_analyser.bead_calibration import (
                        run_protocol_postprocessing,
                    )
                    file_id = os.path.splitext(file_paths.base_filename)[0]
                    df_beads = pd.read_csv(file_paths.listmode_csv_path)
                    df_beads = normalise_dataframe_columns(df_beads)
                    run_protocol_postprocessing(
                        protocol=detected_protocol,
                        packet=protocol_packet,
                        dataframe=df_beads,
                        file_id=file_id,
                        diagnostic_dir=self.output_folder,
                    )
                    log_message(f"Success: bead calibration computed and persisted for {file_path}")
                except Exception as e:
                    log_message(f"Error: bead calibration failed for {file_path}: {e}")
                delete_file(file_paths.listmode_csv_path)
                delete_file(file_paths.json_path)
                delete_file(file_paths.image_dir_path)
                log_message(f"Success: counted {file_path}")
                return

            # Normal sample: classify using the appropriate model
            apply_python_model(file_paths.listmode_csv_path, file_paths.predictions_csv_path, self.model_path)
            log_message(f"Success: Predictions made for {file_path}")

            predictions_df = pd.read_csv(file_paths.predictions_csv_path)
            prediction_counts = predictions_df['predicted_label'].value_counts().reset_index()
            prediction_counts.columns = ['class', 'count']
            prediction_counts_path = file_paths.prediction_counts_csv_path
            prediction_counts.to_csv(prediction_counts_path, index=False)
            predictions_csv = file_paths.predictions_csv_path
            qc_plots.update_after_file(file_paths.json_path, predictions_csv, self.output_folder)
            log_message(f"Success: counted {file_path}")

            data = pd.read_csv(file_paths.predictions_csv_path)
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
            plot3d_prediction_path = file_paths.prediction_plot_3d_html_path
            pio.write_html(fig, file=plot3d_prediction_path, auto_open=False)
            log_message("Plot saved as '3D_Plot.html'.")
            delete_file(file_paths.listmode_csv_path)
            delete_file(file_paths.json_path)
            delete_file(file_paths.image_dir_path)
#            delete_file(plot3d_prediction_path)
#            delete_file(file_paths.predictions_csv_path)
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
    return wait_for_file_release_utility(file_path, timeout=timeout, interval=interval)


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
        print('extracting lines = extract(            particles=data["particles"],            dateandtime=data["instrument"]...etc')
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

                out_path = _RUNTIME_CONFIG.paths.fws_calibration_store_path
                append_fwscalibration_record(rec, out_path)

    except subprocess.CalledProcessError as e:
        log_message(f"Processing Error: Failed to process file: {e}")
       

def _build_prediction_calibration_audit(record):
    ts = None
    age_seconds = None
    source_file = None
    qc_passed = None

    if isinstance(record, dict):
        ts = record.get("time_calculated")
        source_file = record.get("file_id")
        if "qc_pass" in record:
            qc_passed = bool(record.get("qc_pass"))
        if ts:
            try:
                parsed = pd.to_datetime(ts, utc=True, errors="coerce")
                if not pd.isna(parsed):
                    age_seconds = max(
                        0.0,
                        float(
                            (datetime.datetime.now(datetime.timezone.utc) - parsed.to_pydatetime()).total_seconds()
                        ),
                    )
            except Exception:
                age_seconds = None

    return {
        "bead_calibration_used": False,
        "model_mode_used": "raw_uncalibrated_model",
        "calibration_timestamp": ts,
        "calibration_age_seconds": age_seconds,
        "calibration_source_file": source_file,
        "calibration_qc_passed": qc_passed,
    }


def apply_python_model(listmode_file, predictions_file, model_path):
    """
    Apply the appropriate active model to a listmode CSV.

    Decision logic, evaluated at call time:
      1. If a bead calibration younger than 1 year exists and an active
         bead-calibrated model is configured:
           - Load the exact active_beadcalibrated_model path.
           - Apply the saved bead calibration in the feature shape expected
             by that exact model.
      2. Otherwise:
           - Load the exact active_uncalibrated_model path.
           - If no active_uncalibrated_model is configured, fall back to the

    The model_path argument is retained for API compatibility but is not used.
    """
    import warnings

    if model_path is not None:
        warnings.warn(
            "The model_path argument to apply_python_model() is deprecated and will be "
            "removed in a future release. The active model is resolved from config.",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        df = pd.read_csv(listmode_file)
        df = normalise_dataframe_columns(df)
        df = df.dropna()

        # ------------------------------------------------------------------
        # Decide model path first.
        # ------------------------------------------------------------------
        beadcal_model_path = None
        bead_cal_record = None
        use_bead_calibrated = False
        prediction_calibration_audit = _build_prediction_calibration_audit(None)

        try:
            beadcal_model_path = resolve_active_beadcalibrated_model_path()
            print('beadcal_model_path:')
            print(beadcal_model_path)
        except Exception as e:
            print(f"Could not resolve active bead-calibrated model: {e}")
            beadcal_model_path = None

        if beadcal_model_path is not None:
            bead_cal_ok, bead_cal_record, bead_cal_reason = (
                latest_bead_calibration_status(
                    load_latest_beadscalibration_record,
                    max_age_seconds=_RUNTIME_CONFIG.options.bead_calibration_max_age_seconds,
                )
            )
            print(f"Bead calibration check: {bead_cal_reason}")
            prediction_calibration_audit = _build_prediction_calibration_audit(bead_cal_record)

            if bead_cal_ok:
                use_bead_calibrated = True
        else:
            print("No active bead-calibrated model is configured.")
            prediction_calibration_audit = _build_prediction_calibration_audit(load_latest_beadscalibration_record())

        # ------------------------------------------------------------------
        # Load exact model and prepare dataframe for that exact model.
        # ------------------------------------------------------------------
        if use_bead_calibrated:
            try:
                print("Inference path: bead-calibrated model")
                model, classes, features = load_classifier_from_exact_path(
                    beadcal_model_path,
                    "bead-calibrated",
                )

                print("Your bead-calibrated model expects these columns:", features)
                print("Your raw data file has these columns:", df.columns.tolist())

                df_for_prediction = apply_bead_calibration_for_model_features(
                    df,
                    bead_cal_record,
                    features,
                    apply_saved_bead_calibration_to_dataframe_fn=apply_saved_bead_calibration_to_dataframe,
                )

            except Exception as e:
                print(
                    f"Bead-calibrated inference setup failed ({e}); "
                    "falling back to uncalibrated (raw) model."
                )
                use_bead_calibrated = False

        if not use_bead_calibrated:
            print("Inference path: uncalibrated (raw) model")
            try:
                raw_model_path = resolve_active_raw_model_path()
            except Exception as raw_error:
                print(f"Could not resolve active uncalibrated model: {raw_error}")
                raise
            model, classes, features = load_classifier_from_exact_path(
                raw_model_path,
                "uncalibrated/raw",
            )
            df_for_prediction = df
        if use_bead_calibrated:
            prediction_calibration_audit["model_mode_used"] = "bead_calibrated_model"
        prediction_calibration_audit["bead_calibration_used"] = bool(use_bead_calibrated)

        full_predicted, rows_before = run_prediction(model, classes, features, df_for_prediction)
        for key, value in prediction_calibration_audit.items():
            full_predicted[key] = value
        full_predicted.to_csv(predictions_file, index=False)

        print(
            f"Prediction rows written: {len(full_predicted)} "
            f"of {rows_before} feature-selected rows."
        )
        print("Prediction CSV includes raw + calibrated columns for downstream QC.")
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
    combined_df = _compute_consensual_labels_and_sample_weights(combined_df)
    combined_df['source_label'] = combined_df['consensus_label']
    print(combined_df)
    return combined_df
