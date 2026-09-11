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
    "download_blobs",
    "list_blobs",
    "upload_to_blob",
    "mix_blob_files",
    "convert_cyz_to_json",
    "compile_cyz2json_from_release",
    "flatten_dict",
    "dict_to_csv",
    "clear_temp_folder",
    "compile_r_requirements",
    "apply_r",
    "select_output_dir",
    "load_json",
    "select_particles",
    "get_pulses",
    "display_image",
    "update_navigation_buttons",
    "save_metadata",
    "summarize_predictions",
    "upload_to_blob_path",
    "run_backend_only",
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

# functions.py (new/updated parts)
import os
from urllib.parse import urlparse
from flowcytometer_tool.tabs.blob_tools.storage_clients import _split_blob_url, get_container_client, get_blob_client

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
    Upload a local file to the same container as container_url at the given blob_path.
    Extremely verbose version for debugging.
    """

    print("=" * 80)
    print("UPLOAD_TO_BLOB_PATH START")
    print(f"Local file path : {file_path}")
    print(f"Container URL   : {container_url}")
    print(f"Target blob path: {blob_path}")
    print("=" * 80)

    try:
        from flowcytometer_tool.tabs.blob_tools.storage_clients import (
            _split_blob_url,
            get_blob_client,
        )

        print("Checking file exists...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        file_size = os.path.getsize(file_path)
        print(f"File found.")
        print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

        print("Parsing container URL...")
        account_url, container_name, existing_blob = _split_blob_url(container_url)

        print(f"Account URL      : {account_url}")
        print(f"Container name   : {container_name}")
        print(f"Existing blob    : {existing_blob}")

        print("Creating BlobClient...")
        bc = get_blob_client(
            account_url,
            container_name,
            blob_path,
            anonymous=False,
        )

        print("BlobClient created successfully.")
        print(f"Target blob name: {blob_path}")

        start_time = time.time()

        print("Opening local file...")
        with open(file_path, "rb") as data:
            print("Beginning upload...")
            print(f"Blob URL: {bc.url}")
            print(f"File size: {os.path.getsize(file_path):,}")
            start = time.time()

            bc.upload_blob(
                data,
                overwrite=True,
            )

            print(f"Upload took {time.time()-start:.1f}s")

        elapsed = time.time() - start_time

        print("UPLOAD SUCCEEDED")
        print(f"Uploaded file : {os.path.basename(file_path)}")
        print(f"Blob path     : {blob_path}")
        print(f"Elapsed time  : {elapsed:.2f} seconds")

    except Exception as e:
        print("UPLOAD FAILED")
        print(f"File      : {file_path}")
        print(f"Blob path : {blob_path}")
        print(f"Exception : {type(e).__name__}")
        print(f"Message   : {e}")
        print(traceback.format_exc())
        raise

    finally:
        print("UPLOAD_TO_BLOB_PATH END")
        print("=" * 80)


def _push_model_artifacts_to_models_container(
    version: str,
    file_paths: list[str],
    container_url: str,
):
    """
    Upload model artefacts under:
        <version>/<filename>

    Extremely verbose version.
    """

    print("\n" + "#" * 100)
    print("PUSH MODEL ARTEFACTS START")
    print(f"Version       : {version}")
    print(f"Container URL : {container_url}")
    print(f"Number of files supplied: {len(file_paths)}")
    print("#" * 100)

    uploaded = 0
    skipped = 0
    failed = 0

    for idx, p in enumerate(file_paths, start=1):

        print("\n" + "-" * 80)
        print(f"Processing artifact {idx}/{len(file_paths)}")
        print(f"Path: {p}")

        if not p:
            print("SKIPPED: Empty path provided.")
            skipped += 1
            continue

        if not os.path.exists(p):
            print("SKIPPED: File does not exist.")
            print(f"Missing file: {p}")
            skipped += 1
            continue

        blob_path = f"{version}/{os.path.basename(p)}"

        print(f"Source file : {p}")
        print(f"Blob target : {blob_path}")
        print(f"File size   : {os.path.getsize(p):,} bytes")

        try:
            upload_to_blob_path(
                p,
                container_url=container_url,
                blob_path=blob_path,
            )

            uploaded += 1

            print("RESULT: SUCCESS")

        except Exception as e:
            failed += 1

            print("RESULT: FAILED")
            print(f"Exception: {e}")
            print(traceback.format_exc())

    print("\n" + "#" * 100)
    print("PUSH MODEL ARTEFACTS COMPLETE")
    print(f"Total supplied : {len(file_paths)}")
    print(f"Uploaded       : {uploaded}")
    print(f"Skipped        : {skipped}")
    print(f"Failed         : {failed}")
    print("#" * 100)
    

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
    
    
def run_backend_only():
    print("🔧 Running in no-GUI mode...")
    from flowcytometer_tool.misc.functions_training import combine_csvs, train_model

    # Setup paths
    tool_dir = str(_TOOL_DIR)
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
        from flowcytometer_tool.tabs.download_train.custom_functions_for_python import predictTestSet

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
