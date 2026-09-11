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
    "plot_cv_results",
    "plot_classifier_props",
    "plot_all_hyperpars_combi_and_classifiers_scores",
    "plot_3d_fluorescence_premerge",
    "FWS_size_plot_3d_fluorescence_premerge",
    "plot3d",
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

def plot_cv_results(cv_results, plots_dir):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(plots_dir, exist_ok=True)

    if cv_results is None:
        print("Could not plot CV results: cv_results is None")
        return []

    if not isinstance(cv_results, pd.DataFrame):
        cv_results = pd.DataFrame(cv_results)

    if cv_results.empty:
        print("Could not plot CV results: cv_results is empty")
        return []

    # Nested CV output should already contain outer_splits.
    # Plain GridSearchCV/HalvingRandomSearchCV cv_results_ will not.
    if "outer_splits" not in cv_results.columns:
        print(
            "CV results do not contain 'outer_splits'. "
            "Treating results as a single CV run for plotting."
        )
        cv_results = cv_results.copy()
        cv_results["outer_splits"] = 1

    if "outer_split_test_score" not in cv_results.columns:
        cv_results = cv_results.copy()
        cv_results["outer_split_test_score"] = float("nan")

    if "iter" not in cv_results.columns:
        cv_results = cv_results.copy()
        cv_results["iter"] = 0

    if "mean_test_score" not in cv_results.columns:
        print(
            "Could not plot CV results: missing 'mean_test_score'. "
            f"Available columns: {list(cv_results.columns)}"
        )
        return []

    if "param_classifier" not in cv_results.columns:
        cv_results = cv_results.copy()
        cv_results["param_classifier"] = "classifier"

    plotlist = []

    try:
        best_results = (
            cv_results[cv_results["iter"] == cv_results["iter"].max()]
            .groupby(["param_classifier", "outer_splits"], dropna=False)
            .apply(lambda x: x.loc[x["mean_test_score"].idxmax()])
        )
    except Exception as e:
        print(f"Could not compute best CV results for plotting: {e}")
        best_results = None

    for outer in cv_results["outer_splits"].dropna().unique():
        outer_data = cv_results[cv_results["outer_splits"] == outer].copy()

        if outer_data.empty:
            continue

        try:
            outer_score_values = outer_data["outer_split_test_score"].dropna().unique()
            outer_score = round(float(outer_score_values[0]), 3) if len(outer_score_values) else float("nan")
        except Exception:
            outer_score = float("nan")

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.lineplot(
            data=outer_data,
            x="iter",
            y="mean_test_score",
            hue="param_classifier",
            marker="o",
            ax=ax,
        )

        for classifier in outer_data["param_classifier"].dropna().unique():
            clf_data = (
                outer_data[outer_data["param_classifier"] == classifier]
                .sort_values("iter")
                .copy()
            )

            if len(clf_data) > 1:
                clf_data["delta_mcc"] = clf_data["mean_test_score"].diff()
                sns.lineplot(
                    data=clf_data,
                    x="iter",
                    y="delta_mcc",
                    label=f"{classifier} delta MCC",
                    linestyle="--",
                    ax=ax,
                )

        ax.set_title(f"CV results, outer split {outer}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MCC and delta MCC")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Classifier")

        fig.text(
            0.5,
            -0.08,
            f"Outer CV test score: {outer_score}",
            wrap=True,
            horizontalalignment="center",
            fontsize=10,
        )

        fig.tight_layout()

        plot_path = os.path.join(plots_dir, f"cv_results_outer_{outer}.png")
        fig.savefig(plot_path, bbox_inches="tight")
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
