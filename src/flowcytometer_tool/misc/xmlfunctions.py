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
import matplotlib.path as mpath
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
from flowcytometer_tool.misc.convert_json_to_listmode import convert_json_to_listmode
from flowcytometer_tool.misc.normalise_training_person_name import _normalise_training_person_name
from flowcytometer_tool.misc.person_to_weight_from_expertise_levels import _person_to_weight_from_expertise_levels
from flowcytometer_tool.misc.compute_consensual_labels_and_sample_weights import _compute_consensual_labels_and_sample_weights

__all__ = [
    "parse_gate_setlist",
    "assign_classes_from_gates",
    "convert_selected_cyzs_to_listmode",
    "build_consensual_dataset_from_cyz_xmls",
    "build_consensual_dataset_from_selected_cyzs_and_xmls",
    "_cyz_training_stem",
    "load_file"
]

# ============================================================
# XML gate parsing and class assignment
# (adapted from misc/generic_gate_demo.py)
# ============================================================


def _cyz_training_stem(path):
    """Normalise a CYZ/listmode basename so repeated labels for the same file share a filename key."""
    base = os.path.basename(str(path))
    lower = base.lower()
    for suffix in (".cyz.json.csv", ".json.csv", ".cyz.csv", ".csv", ".json", ".cyz"):
        if lower.endswith(suffix):
            return base[:-len(suffix)]
    return os.path.splitext(base)[0]


def _gate_xml_local_name(el_or_tag):
    """Return an XML tag name without any namespace prefix."""
    tag = el_or_tag.tag if hasattr(el_or_tag, "tag") else str(el_or_tag)
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _gate_xml_get_attr_case_insensitive(el, name, default=None):
    """Read an XML attribute without caring about exact attribute casing."""
    if el is None:
        return default
    if name in el.attrib:
        return el.attrib.get(name, default)
    lname = name.lower()
    for k, v in el.attrib.items():
        if k.lower() == lname:
            return v
    return default


def _gate_axis_name_from_info(axis_info):
    """Return the best available display/resolution name from an axis info dict."""
    if not axis_info:
        return None
    for key in ("name", "channel_name", "parameter"):
        val = axis_info.get(key)
        if val not in (None, ""):
            return val
    return None


def _gate_get_axis_info(el):
    """
    Extract XML gate axis metadata.

    CytoClus/XML axes can be either simple axes, e.g. ``FLRed`` or ``SWS``,
    or calculated ratio axes made from numerator/denominator child elements,
    e.g. ``FLRed / FLYellow``.  This parser preserves enough metadata for
    later gate application to resolve the underlying DataFrame columns and
    calculate the ratio before applying the gate.
    """
    if el is None:
        return None

    info = {
        "name": _gate_xml_get_attr_case_insensitive(el, "Name"),
        "is_log": str(_gate_xml_get_attr_case_insensitive(el, "IsLog", "False")).lower() == "true",
        "type": _gate_xml_get_attr_case_insensitive(el, "Type"),
        "channel_type": _gate_xml_get_attr_case_insensitive(el, "ChannelType"),
        "channel_name": _gate_xml_get_attr_case_insensitive(el, "ChannelName"),
        "parameter": _gate_xml_get_attr_case_insensitive(el, "Parameter"),
    }

    def ratio_part_info(part_el):
        # Some XML writes AxisNumerator/AxisDenominator as the axis element
        # itself; other exports use it as a wrapper around an Axis-like child.
        part_info = _gate_get_axis_info(part_el)
        if _gate_axis_name_from_info(part_info) is not None:
            return part_info
        for nested in list(part_el):
            nested_info = _gate_get_axis_info(nested)
            if _gate_axis_name_from_info(nested_info) is not None:
                return nested_info
        return part_info

    numerator = None
    denominator = None
    for child in list(el):
        local = _gate_xml_local_name(child).lower()
        compact = re.sub(r"[^a-z0-9]+", "", local)
        if compact in {"axisnumerator", "numerator"} or compact.endswith("numerator"):
            numerator = ratio_part_info(child)
        elif compact in {"axisdenominator", "denominator"} or compact.endswith("denominator"):
            denominator = ratio_part_info(child)

    if numerator is not None:
        info["numerator"] = numerator
    if denominator is not None:
        info["denominator"] = denominator
    return info


def _gate_parse_single(gate_el):
    """Parse one gate XML element into a dict while preserving axis metadata."""
    tag = _gate_xml_local_name(gate_el)

    if tag == "RangeGate":
        axis_info = _gate_get_axis_info(gate_el.find("Axis"))
        return {
            "type": "range",
            "axis": _gate_axis_name_from_info(axis_info),
            "axis_info": axis_info,
            "min": float(gate_el.attrib["RangeMin"]),
            "max": float(gate_el.attrib["RangeMax"]),
        }

    if tag == "RectangleGate":
        x_info = _gate_get_axis_info(gate_el.find("XAxis"))
        y_info = _gate_get_axis_info(gate_el.find("YAxis"))
        x = float(gate_el.attrib["X"])
        y = float(gate_el.attrib["Y"])
        w = float(gate_el.attrib["Width"])
        h = float(gate_el.attrib["Height"])
        return {
            "type": "rectangle",
            "x_axis": _gate_axis_name_from_info(x_info),
            "y_axis": _gate_axis_name_from_info(y_info),
            "x_axis_info": x_info,
            "y_axis_info": y_info,
            "x_min": x,
            "x_max": x + w,
            "y_min": y,
            "y_max": y + h,
        }

    if tag == "PolygonGate":
        x_info = _gate_get_axis_info(gate_el.find("XAxis"))
        y_info = _gate_get_axis_info(gate_el.find("YAxis"))
        points = []
        path_el = gate_el.find("Path")
        if path_el is not None:
            for pnt in path_el.findall("Point"):
                points.append((float(pnt.attrib["X"]), float(pnt.attrib["Y"])))
        return {
            "type": "polygon",
            "x_axis": _gate_axis_name_from_info(x_info),
            "y_axis": _gate_axis_name_from_info(y_info),
            "x_axis_info": x_info,
            "y_axis_info": y_info,
            "points": points,
        }

    raise ValueError(f"Unsupported gate type: {tag}")

def parse_gate_setlist(xml_path):
    """
    Parse a gates XML file (SetList format) into a dict of gate sets.

    Returns
    -------
    sets : dict  {list_id (int) -> set_dict}
        Each set_dict has at minimum: kind, id, name.
        kind is one of: 'default', 'gate_based', 'or', 'combined'.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sets = {}

    for child in root:
        if "ListID" not in child.attrib:
            continue
        list_id = int(child.attrib["ListID"])
        name = child.attrib.get("Name", f"Set {list_id}")
        tag = child.tag

        if tag == "DefaultSet":
            sets[list_id] = {"kind": "default", "id": list_id, "name": name}

        elif tag == "GateBasedSet":
            gates = []
            gc = child.find("GateCollection")
            if gc is not None:
                for gate_el in list(gc):
                    gates.append(_gate_parse_single(gate_el))
            sets[list_id] = {"kind": "gate_based", "id": list_id, "name": name, "gates": gates}

        elif tag == "OrSet":
            ids = []
            setlist_el = child.find("SetList")
            if setlist_el is not None:
                for id_el in setlist_el.findall("ListID"):
                    ids.append(int(id_el.text))
            auto = str(child.findtext("AutoSet", "False")).strip().lower() == "true"
            sets[list_id] = {"kind": "or", "id": list_id, "name": name, "members": ids, "auto": auto}

        elif tag == "CombinedSet":
            sets[list_id] = {
                "kind": "combined",
                "id": list_id,
                "name": name,
                "set1": int(child.findtext("Set1ListID")),
                "set2": int(child.findtext("Set2ListID")),
                "combination": child.findtext("CombinationType"),
            }

    return sets



class GateAxisResolutionError(KeyError):
    """Raised when an XML gate axis cannot be matched to a DataFrame column."""


def _gate_axis_canonical(axis_name):
    """
    Canonicalise CytoClus/XML axis names and listmode DataFrame column names
    so harmless spelling, unit, case, and separator differences can be matched.
    """
    if axis_name is None:
        return ""

    text = str(axis_name).strip().lower()
    text = re.sub(r"\[[^\]]*\]", " ", text)          # remove units such as [mV]
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    metric_aliases = {
        "average": "average", "avg": "average", "mean": "average",
        "total": "total", "sum": "total",
        "maximum": "maximum", "max": "maximum",
        "minimum": "minimum", "min": "minimum",
    }

    metric = None
    remaining = []
    for tok in tokens:
        if tok in metric_aliases and metric is None:
            metric = metric_aliases[tok]
        else:
            remaining.append(tok)

    channel = " ".join(remaining)
    channel_compact = re.sub(r"[^a-z0-9]+", "", channel)

    # Channel aliases seen in CytoClus XML versus exported listmode CSVs.
    channel_aliases = {
        "sidewardscatter": "sidewardsscatter",
        "sidewardsscatter": "sidewardsscatter",
        "sws": "sidewardsscatter",
        "forwardscatter": "fws",
        "fwscatter": "fws",
        "fws": "fws",
        "flyellow": "flyellow",
        "florange": "florange",
        "flred": "flred",
    }
    channel_key = channel_aliases.get(channel_compact, channel_compact)
    return f"{channel_key}|{metric}" if metric else channel_key


def _resolve_gate_axis_name(df, xml_axis_name, gate_type):
    """
    Resolve an XML gate axis name to the actual DataFrame column name.

    Raises GateAxisResolutionError with enough context to fix either the XML or
    the listmode export if no safe match is found.
    """
    if xml_axis_name in df.columns:
        return xml_axis_name

    attempted = _gate_axis_canonical(xml_axis_name)
    canonical_to_columns = {}
    for col in df.columns:
        canonical_to_columns.setdefault(_gate_axis_canonical(col), []).append(col)

    matches = canonical_to_columns.get(attempted, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise GateAxisResolutionError(
            "Ambiguous gate axis resolution. "
            f"XML axis: {xml_axis_name!r}; gate type: {gate_type}; "
            f"attempted canonical form: {attempted!r}; matching columns: {matches[:10]!r}"
        )

    # If the XML gives only the channel name, e.g. FLRed, but the listmode
    # export contains a single metric-qualified column, e.g. FLRed_total,
    # resolve to that unique column. If several metrics exist, keep the
    # existing ambiguous-resolution behaviour by raising a clear error below.
    if "|" not in attempted:
        metric_matches = []
        prefix = attempted + "|"
        for col in df.columns:
            if _gate_axis_canonical(col).startswith(prefix):
                metric_matches.append(col)
        if len(metric_matches) == 1:
            return metric_matches[0]
        if len(metric_matches) > 1:
            raise GateAxisResolutionError(
                "Ambiguous gate axis resolution. "
                f"XML axis: {xml_axis_name!r}; gate type: {gate_type}; "
                f"attempted canonical form: {attempted!r}; matching columns: {metric_matches[:10]!r}"
            )

    sample_cols = list(df.columns[:30])
    raise GateAxisResolutionError(
        "Could not resolve XML gate axis to a DataFrame column. "
        f"XML axis: {xml_axis_name!r}; gate type: {gate_type}; "
        f"attempted canonical form: {attempted!r}; "
        f"available DataFrame columns sample: {sample_cols!r}"
    )

def _gate_safe_ratio_column_name(numerator_col, denominator_col):
    """Create a deterministic safe column name for a calculated XML ratio axis."""
    def clean(part):
        part = str(part).strip()
        part = re.sub(r"\s+", "_", part)
        part = re.sub(r"[^0-9A-Za-z_]+", "_", part)
        part = re.sub(r"_+", "_", part).strip("_")
        return part or "axis"
    return f"{clean(numerator_col)}_over_{clean(denominator_col)}"


def _gate_lower_first_camel(value):
    """Return CytoClus-style Parameter text as a listmode suffix candidate."""
    value = "" if value is None else str(value).strip()
    if not value:
        return ""
    compact = re.sub(r"\s+", "", value)
    return compact[:1].lower() + compact[1:]


def _gate_channel_parameter_column_candidates(axis_info):
    """
    Build DataFrame column-name candidates from XML ChannelName + Parameter.

    This deliberately prefers explicit XML metadata over the display/name field.
    Examples:
      ChannelName='FWS', Parameter='Length'         -> FWS_length
      ChannelName='FWS', Parameter='NumberOfCells'  -> FWS_numberOfCells
      ChannelName='Fl Red', Parameter='Total'       -> Fl Red_total
    """
    if not axis_info:
        return []

    channel = axis_info.get("channel_name")
    parameter = axis_info.get("parameter")
    if channel in (None, "") or parameter in (None, ""):
        return []

    channel = str(channel).strip()
    parameter = str(parameter).strip()
    if not channel or not parameter:
        return []

    suffixes = []
    lower_first = _gate_lower_first_camel(parameter)
    if lower_first:
        suffixes.append(lower_first)

    lower_all = re.sub(r"\s+", "", parameter).lower()
    if lower_all and lower_all not in suffixes:
        suffixes.append(lower_all)

    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", re.sub(r"\s+", "", parameter)).lower()
    if snake and snake not in suffixes:
        suffixes.append(snake)

    return [f"{channel}_{suffix}" for suffix in suffixes]


def _gate_resolve_axis_from_channel_parameter(df, axis_info, gate_type):
    """Resolve an XML axis using ChannelName + Parameter metadata, if possible."""
    candidates = _gate_channel_parameter_column_candidates(axis_info)
    if not candidates:
        return None

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # Still metadata-first: try the derived names through the existing
    # canonical/alias layer before falling back to XML Name.
    for candidate in candidates:
        try:
            return _resolve_gate_axis_name(df, candidate, gate_type)
        except GateAxisResolutionError:
            pass
    return None


def _gate_resolve_axis_part(df, axis_info, gate_type, part_name):
    """Resolve the numerator or denominator part of a calculated ratio axis."""
    metadata_col = _gate_resolve_axis_from_channel_parameter(df, axis_info, gate_type)
    if metadata_col is not None:
        return metadata_col

    xml_name = _gate_axis_name_from_info(axis_info)
    try:
        return _resolve_gate_axis_name(df, xml_name, gate_type)
    except GateAxisResolutionError as exc:
        available = list(df.columns)
        candidates = _gate_channel_parameter_column_candidates(axis_info)
        raise GateAxisResolutionError(
            "Could not resolve calculated XML ratio gate axis component. "
            f"Gate type: {gate_type}; missing part: {part_name}; "
            f"XML axis name: {xml_name!r}; "
            f"ChannelName: {(axis_info or {}).get('channel_name')!r}; "
            f"Parameter: {(axis_info or {}).get('parameter')!r}; "
            f"metadata-derived candidates: {candidates!r}; "
            f"available DataFrame columns: {available!r}"
        ) from exc


def _gate_evaluate_axis(df, axis_info, gate_type):
    """
    Resolve and evaluate a gate axis against a DataFrame.

    Simple XML axes are resolved using the existing alias/canonicalisation
    logic.  XML ratio axes with numerator/denominator children are calculated
    as ``DataFrame[numerator] / DataFrame[denominator]``.  The calculated ratio
    is added to ``df`` as a stable column and then used exactly like a normal
    gate axis.  Divide-by-zero and non-finite denominators produce ``np.nan``
    rather than raising warnings or crashing gate assignment.
    """
    if axis_info is None:
        raise GateAxisResolutionError(
            f"Gate axis metadata is missing for gate type: {gate_type}"
        )

    numerator_info = axis_info.get("numerator")
    denominator_info = axis_info.get("denominator")
    if numerator_info is not None or denominator_info is not None:
        if numerator_info is None or denominator_info is None:
            missing = "numerator" if numerator_info is None else "denominator"
            raise GateAxisResolutionError(
                "Incomplete calculated XML ratio gate axis. "
                f"Gate type: {gate_type}; missing part: {missing}; "
                f"axis name: {_gate_axis_name_from_info(axis_info)!r}; "
                f"available DataFrame columns: {list(df.columns)!r}"
            )

        numerator_col = _gate_resolve_axis_part(df, numerator_info, gate_type, "numerator")
        denominator_col = _gate_resolve_axis_part(df, denominator_info, gate_type, "denominator")
        ratio_col = _gate_safe_ratio_column_name(numerator_col, denominator_col)

        if ratio_col not in df.columns:
            numerator = df[numerator_col].to_numpy(dtype=float)
            denominator = df[denominator_col].to_numpy(dtype=float)
            ratio = np.full(len(df), np.nan, dtype=float)
            valid = np.isfinite(denominator) & (denominator != 0)
            np.divide(numerator, denominator, out=ratio, where=valid)
            ratio[~np.isfinite(ratio)] = np.nan
            df[ratio_col] = ratio

        return ratio_col, df[ratio_col].to_numpy(dtype=float)

    axis_name = _gate_axis_name_from_info(axis_info)
    axis_col = _resolve_gate_axis_name(df, axis_name, gate_type)
    return axis_col, df[axis_col].to_numpy(dtype=float)


def _gate_apply_single(df, gate):
    """Apply one parsed gate to df; returns a boolean numpy array."""
    if gate["type"] == "range":
        _, x = _gate_evaluate_axis(df, gate.get("axis_info") or {"name": gate.get("axis")}, "RangeGate")
        return (x >= gate["min"]) & (x <= gate["max"])

    if gate["type"] == "rectangle":
        _, x = _gate_evaluate_axis(df, gate.get("x_axis_info") or {"name": gate.get("x_axis")}, "RectangleGate")
        _, y = _gate_evaluate_axis(df, gate.get("y_axis_info") or {"name": gate.get("y_axis")}, "RectangleGate")
        return (x >= gate["x_min"]) & (x <= gate["x_max"]) & (y >= gate["y_min"]) & (y <= gate["y_max"])

    if gate["type"] == "polygon":
        _, x = _gate_evaluate_axis(df, gate.get("x_axis_info") or {"name": gate.get("x_axis")}, "PolygonGate")
        _, y = _gate_evaluate_axis(df, gate.get("y_axis_info") or {"name": gate.get("y_axis")}, "PolygonGate")
        points = gate["points"]
        if len(points) < 3:
            return np.zeros(len(df), dtype=bool)
        xy = np.column_stack([x, y])
        return mpath.Path(points).contains_points(xy)

    raise ValueError(f"Unsupported gate type: {gate['type']}")

def _gate_evaluate_sets(df, sets):
    """
    Evaluate all gate sets against df.

    Returns
    -------
    masks : dict  {list_id -> bool numpy array of length len(df)}
    """
    masks = {}

    def eval_one(sid):
        if sid in masks:
            return masks[sid]
        if sid not in sets:
            raise KeyError(f"Gate set ListID {sid} referenced but not present in XML")
        s = sets[sid]
        kind = s["kind"]

        if kind == "default":
            mask = np.ones(len(df), dtype=bool)
        elif kind == "gate_based":
            gates = s["gates"]
            if len(gates) == 0:
                mask = np.zeros(len(df), dtype=bool)
            else:
                mask = np.ones(len(df), dtype=bool)
                for gate in gates:
                    mask &= _gate_apply_single(df, gate)
        elif kind == "or":
            mask = np.zeros(len(df), dtype=bool)
            for member_id in s["members"]:
                mask |= eval_one(member_id)
        elif kind == "combined":
            m1 = eval_one(s["set1"])
            m2 = eval_one(s["set2"])
            comb = s["combination"]
            if comb == "set1AND2":
                mask = m1 & m2
            elif comb == "set1OR2":
                mask = m1 | m2
            elif comb == "set1NOT2":
                mask = m1 & ~m2
            elif comb == "set2NOT1":
                mask = m2 & ~m1
            else:
                raise ValueError(f"Unsupported CombinationType: {comb}")
        else:
            raise ValueError(f"Unsupported gate kind: {kind}")

        masks[sid] = mask
        return mask

    for sid in sorted(sets):
        eval_one(sid)
    return masks


def assign_classes_from_gates(df, xml_path):
    """
    Classify particles in *df* using the gate definitions in *xml_path*.

    Terminal sets (sets that are not used as inputs to OrSet/CombinedSet,
    excluding the default set) define the class labels.  When a particle
    matches multiple terminal sets, the one with the highest ListID wins
    (``primary_class``).  Unmatched particles receive the label
    ``"Unclassified"``.

    The returned DataFrame is a copy of *df* with an added ``source_label``
    column (``= primary_class``).

    Parameters
    ----------
    df : pd.DataFrame
        Listmode data.  Column names must match the ``Name`` attributes of
        the gate axes in the XML.
    xml_path : str or Path
        Path to the gates XML file.

    Returns
    -------
    pd.DataFrame
    """
    sets = parse_gate_setlist(xml_path)
    out = df.copy()
    # Evaluate gates on the returned DataFrame copy so calculated XML ratio
    # columns are retained for diagnostics and downstream training data.
    masks = _gate_evaluate_sets(out, sets)

    # Identify terminal set IDs: not the default (0) and not used as
    # input to another set.
    referenced = set()
    for s in sets.values():
        if s["kind"] == "or":
            referenced.update(s["members"])
        elif s["kind"] == "combined":
            referenced.add(s["set1"])
            referenced.add(s["set2"])

    terminal_ids = [
        sid for sid in sorted(sets)
        if sid != 0 and sid not in referenced
    ]

    primary = np.array(["Unclassified"] * len(out), dtype=object)
    terminal_classes = [[] for _ in range(len(out))]
    for sid in sorted(terminal_ids):
        name = sets[sid]["name"]
        mask = masks[sid]
        for idx in np.where(mask)[0]:
            terminal_classes[idx].append(name)

    for sid in sorted(terminal_ids, reverse=True):
        name = sets[sid]["name"]
        mask = masks[sid] & (primary == "Unclassified")
        primary[mask] = name

    out["terminal_classes"] = [";".join(classes) for classes in terminal_classes]
    out["n_terminal_classes"] = [len(classes) for classes in terminal_classes]
    out["source_label"] = primary
    return out


def build_consensual_dataset_from_cyz_xmls(
    listmode_csv_paths,
    person_xml_paths,
    expertise_levels,
    *,
    prompt_merge_fn=None,
    premerge_plot_fn=None,
    delete_labels_fn=None,
):
    """
    Build a consensus training dataset from selected CYZ/listmode CSV files plus
    one XML gate file per person.

    This supports the newer training-data layout:
      - the same selected .cyz/listmode files can be labelled by multiple people;
      - each person supplies their own XML gates for those same files;
      - particles are matched across people by (filename, id), then consensus labels
        and sample weights are computed using the expertise matrix.
    """
    listmode_csv_paths = [str(p) for p in (listmode_csv_paths or []) if str(p).lower().endswith('.csv')]
    person_xml_paths = {
        _normalise_training_person_name(person): str(xml_path)
        for person, xml_path in (person_xml_paths or {}).items()
        if person and xml_path and os.path.isfile(str(xml_path))
    }

    if not listmode_csv_paths:
        raise ValueError("No listmode CSV files were supplied for XML-based training.")
    if not person_xml_paths:
        raise ValueError("No per-person XML files were supplied for XML-based training.")

    all_data = []
    for person, xml_path in person_xml_paths.items():
        for csv_path in listmode_csv_paths:
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    print(f"Skipping empty listmode CSV: {csv_path}")
                    continue
                labelled = assign_classes_from_gates(df, xml_path)
                if labelled is None or labelled.empty:
                    print(f"Skipping {csv_path} for {person}: XML produced no labelled rows")
                    continue
                labelled = labelled.copy()
                labelled["filename"] = _cyz_training_stem(csv_path)
                labelled["person"] = person
                all_data.append(labelled)
            except GateAxisResolutionError:
                raise
            except Exception as e:
                print(f"Skipping {csv_path} for {person}: {e}")

    if not all_data:
        return None

    combined_df = pd.concat(all_data, ignore_index=True)

    try:
        if premerge_plot_fn is not None:
            premerge_plot_fn(combined_df)
        else:
            default_out = os.path.join(
                os.path.expanduser("~"), "Documents", "flowcytometertool",
                "Training plots", "premerge_3d_fluorescence.html"
            )
            os.makedirs(os.path.dirname(default_out), exist_ok=True)
            plot_3d_fluorescence_premerge(combined_df, label_col="source_label", out_html=default_out)
    except Exception as e:
        print(f"[warn] pre-merge 3D plot not created: {e}")

    if delete_labels_fn is not None:
        delete_labels_fn(combined_df)
    if prompt_merge_fn is not None:
        prompt_merge_fn(combined_df)

    combined_df.columns = combined_df.columns.str.replace(r'\s+', '_', regex=True)
    combined_df = combined_df.dropna()

    person_to_weight = _person_to_weight_from_expertise_levels(expertise_levels)
    combined_df["weight"] = combined_df["person"].map(person_to_weight).fillna(1)

    combined_df = _compute_consensual_labels_and_sample_weights(combined_df)
    combined_df["source_label"] = combined_df["consensus_label"]
    return combined_df



def convert_selected_cyzs_to_listmode(cyz_paths, output_path, cyz2json_path):
    """
    Convert an explicit selection of .cyz files to listmode CSVs under
    output_path/selected_cyz_xml_training/listmode, preserving one CSV per CYZ.
    """
    cyz_paths = [str(p) for p in (cyz_paths or []) if str(p).lower().endswith('.cyz')]
    if not cyz_paths:
        raise ValueError("No .cyz files were selected.")
    if not cyz2json_path or not os.path.isfile(str(cyz2json_path)):
        raise FileNotFoundError(f"Cyz2Json.dll was not found: {cyz2json_path}")

    work_dir = os.path.join(output_path, "selected_cyz_xml_training")
    json_dir = os.path.join(work_dir, "json")
    listmode_dir = os.path.join(work_dir, "listmode")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(listmode_dir, exist_ok=True)

    listmode_paths = []
    for cyz_path in cyz_paths:
        stem = _cyz_training_stem(cyz_path)
        json_path = os.path.join(json_dir, stem + ".cyz.json")
        csv_path = os.path.join(json_dir, stem + ".cyz.csv")
        load_file(cyz2json_path, cyz_path, json_path)
        # move or write the produced CSV to csv_path if needed
        listmode_paths.append(csv_path)
    convert_json_to_listmode(json_dir)
        
    return listmode_paths
    
    
def load_file(cyz2json_path, downloaded_file, json_file):
    try:
        subprocess.run(["dotnet", cyz2json_path, downloaded_file, "--output", json_file, "--metadatagreedy"], check=True)
    except subprocess.CalledProcessError as e:
        log_message(f"Processing Error: Failed to process file: {e}")


def build_consensual_dataset_from_selected_cyzs_and_xmls(
    cyz_paths,
    output_path,
    cyz2json_path,
    person_xml_paths,
    expertise_levels,
    *,
    prompt_merge_fn=None,
    premerge_plot_fn=None,
    delete_labels_fn=None,
):
    """
    Full helper for the GUI: selected .cyz files -> JSON/listmode -> per-person
    XML gate classification -> consensus dataset.
    """
    listmode_paths = convert_selected_cyzs_to_listmode(cyz_paths, output_path, cyz2json_path)
    return build_consensual_dataset_from_cyz_xmls(
        listmode_paths,
        person_xml_paths,
        expertise_levels,
        prompt_merge_fn=prompt_merge_fn,
        premerge_plot_fn=premerge_plot_fn,
        delete_labels_fn=delete_labels_fn,
    )
