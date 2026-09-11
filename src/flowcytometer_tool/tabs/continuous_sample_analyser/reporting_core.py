# reporting_core.py
"""
Core reporting logic: converts metadata + grablist + summary signals
into the flattened packet expected by the dashboard.

This module has no QC logic and no plotting.
It is safe for use in both training and live pipelines.
"""

import json
from pathlib import Path
import pandas as pd
from flowcytometer_tool.tabs.continuous_sample_analyser.protocols import apply_sampling_protocol_mutations as _apply_sampling_protocol_mutations_registry
from flowcytometer_tool.tabs.continuous_sample_analyser.json_safe import json_safe

def apply_sampling_protocol_mutations(packet: dict) -> dict:
    return _apply_sampling_protocol_mutations_registry(packet)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _safe_number(x):
    """Convert to float if possible, else return as-is or None."""
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            if x.strip() == "":
                return None
            return float(x)
    except Exception:
        return None
    return x


PREDICTION_CALIBRATION_PACKET_FIELDS = (
    "bead_calibration_used",
    "model_mode_used",
    "calibration_timestamp",
    "calibration_age_seconds",
    "calibration_source_file",
    "calibration_qc_passed",
)


def extract_prediction_calibration_packet_updates(predictions_df):
    if predictions_df is None or predictions_df.empty:
        return {}

    first = predictions_df.iloc[0]
    out = {}
    for field in PREDICTION_CALIBRATION_PACKET_FIELDS:
        if field not in predictions_df.columns:
            continue
        value = first[field]
        if pd.isna(value):
            out[field] = None
            continue
        if field in {"bead_calibration_used", "calibration_qc_passed"} and isinstance(value, str):
            out[field] = value.strip().lower() in ("true", "1", "yes", "y", "t")
            continue
        out[field] = value
    return out


# ------------------------------------------------------------
# GRABLIST HANDLING
# ------------------------------------------------------------

def _load_grablist(grablist_path):
    """
    Reads grablist file, returns list of JSON paths to extract.
    Format: one dotted JSON key per line.
    Example: instrument.measurementResults.start
    """
    grablist_path = Path(grablist_path)
    items = []
    with open(grablist_path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                items.append(s)
    return items


def _extract_grablist_fields(js, grablist_items):
    """
    For each dotted path in grablist_items, pull the value from
    the JSON object (nested dict). Missing → None.
    """
    out = {}
    for dotted in grablist_items:
        current = js
        parts = dotted.split(".")
        for p in parts:
            if isinstance(current, dict) and p in current:
                current = current[p]
            else:
                current = None
                break
        out[dotted] = current
    return out

def write_report_packet_flat(
    metadata,
    modelsettings,
    predictions_df=None,
    grablist_path=None,
    json_path=None,
    plot_paths=None,
    output_path=None,
    modelversion=None
):
    """
    Build a reporting packet with:
      - metadata from metadata_extraction.extract_metadata
      - selected JSON fields from grablist
      - dashboard header fields
      - optional plot paths
      - optional predictions-based summary flags

    Returns the packet dict. If output_path is provided, writes JSON there.

    This version is deliberately defensive: missing JSON sections such as
    'instrument' no longer raise KeyError and instead produce sensible fallback
    values.
    """
    print("building report packet")

    from datetime import datetime, timezone, timedelta
    from pathlib import Path
    import os
    import json
    import pandas as pd

    def _nested_get(obj, path, default=None):
        """
        Safely walk a nested dict using a list/tuple of keys.
        Example:
            _nested_get(full_js, ["instrument", "measurementResults", "start"])
        """
        cur = obj
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur

    def _as_dict(value):
        return value if isinstance(value, dict) else {}

    def _safe_iso(value, default=None):
        """
        Convert datetime-like values to ISO strings where possible.
        Return default if value is missing or cannot be converted.
        """
        if value is None:
            return default
        try:
            if hasattr(value, "isoformat"):
                return value.isoformat()
            dt = pd.to_datetime(value, errors="coerce")
            if pd.isna(dt):
                return default
            return dt.to_pydatetime().isoformat()
        except Exception:
            return default

    packet = {}

    # -----------------------------
    # 1. Core metadata
    # -----------------------------
    if isinstance(metadata, dict):
        packet.update(metadata)
    else:
        metadata = {}

    # -----------------------------
    # 2. GRABLIST FIELDS + dashboard header source JSON
    # -----------------------------
    full_js = {}
    if json_path:
        try:
            with open(json_path, "r", encoding="utf-8-sig") as f:
                loaded = json.load(f)
            full_js = loaded if isinstance(loaded, dict) else {}
        except Exception as e:
            print(f"[write_report_packet_flat] warning: could not load JSON {json_path}: {e}")
            full_js = {}

    print('instrument = _as_dict(full_js.get("instrument"))')
    instrument = _as_dict(full_js.get("instrument"))
    measurement_settings = _as_dict(instrument.get("measurementSettings"))
    cyto_settings = _as_dict(measurement_settings.get("CytoSettings"))
    measurement_results = _as_dict(instrument.get("measurementResults"))

    # Robust protocol flags
    iifcheck = cyto_settings.get("IIFCheck", None)
    legacy_beads_flag = measurement_settings.get("beads_measurement_2", None)
    beads_flag = cyto_settings.get("IsBeadsMeasurement", legacy_beads_flag)

    packet["instrument.measurementSettings.CytoSettings.IIFCheck"] = iifcheck
    packet["instrument.measurementSettings.CytoSettings.IsBeadsMeasurement"] = beads_flag
    packet["instrument.measurementSettings.beads_measurement_2"] = legacy_beads_flag

    # Extract user grablist fields, safely
    grablist_items = []
    if grablist_path:
        try:
            grablist_items = _load_grablist(grablist_path)
        except Exception as e:
            print(f"[write_report_packet_flat] warning: could not load grablist {grablist_path}: {e}")
            grablist_items = []

    try:
        grabbed = _extract_grablist_fields(full_js, grablist_items)
        for k, v in grabbed.items():
            packet[k] = v
    except Exception as e:
        print(f"[write_report_packet_flat] warning: grablist extraction failed: {e}")

    # -----------------------------
    # 3. DASHBOARD HEADER FIELDS
    # -----------------------------
    packet["version"] = "0.0.3"

    serial = instrument.get("serialNumber") or metadata.get("system_serial_no") or "unknown"
    packet["system_serial_no"] = serial

    now = datetime.now(timezone.utc).isoformat()
    packet["timestamp"] = now

    # Measurement start and end
    start_raw = measurement_results.get("start", None)
    duration_raw = measurement_results.get("duration", None)

    time_start_dt = None
    if start_raw is not None:
        try:
            parsed_start = pd.to_datetime(start_raw, errors="coerce")
            if not pd.isna(parsed_start):
                time_start_dt = parsed_start.to_pydatetime()
        except Exception:
            time_start_dt = None

    if time_start_dt is None:
        meta_start = metadata.get("start")
        try:
            parsed_meta_start = pd.to_datetime(meta_start, errors="coerce")
            if not pd.isna(parsed_meta_start):
                time_start_dt = parsed_meta_start.to_pydatetime()
        except Exception:
            time_start_dt = None

    if time_start_dt is not None:
        packet["time_start"] = time_start_dt.isoformat()
    else:
        packet["time_start"] = now

    time_end_dt = None
    if time_start_dt is not None and duration_raw is not None:
        try:
            duration_seconds = float(duration_raw)
            time_end_dt = time_start_dt + timedelta(seconds=duration_seconds)
        except Exception:
            time_end_dt = None

    if time_end_dt is not None:
        packet["time_end"] = time_end_dt.isoformat()
    else:
        packet["time_end"] = now

    # Latitude/longitude remain fixed unless metadata already provides them
    packet["latitude"] = metadata.get("latitude", 0)
    packet["longitude"] = metadata.get("longitude", 0)

    # Survey name from SaveTextbox, with fallback
    survey_raw = cyto_settings.get("SaveTextbox", None)
    if survey_raw:
        survey = os.path.basename(str(survey_raw).replace("\\", "/"))
        packet["survey"] = survey or "not specified"
    else:
        packet["survey"] = metadata.get("survey", "not specified")

    packet["model_file"] = modelversion

    # Optional plot paths
    if plot_paths:
        try:
            if isinstance(plot_paths, dict):
                for k, v in plot_paths.items():
                    packet[f"plot_{k}"] = str(v)
            elif isinstance(plot_paths, (list, tuple)):
                for i, v in enumerate(plot_paths):
                    packet[f"plot_{i + 1}"] = str(v)
            else:
                packet["plot_paths"] = str(plot_paths)
        except Exception as e:
            print(f"[write_report_packet_flat] warning: plot path handling failed: {e}")

    # -----------------------------
    # 4. Predictions summary
    # -----------------------------
    if predictions_df is not None and len(predictions_df) > 0:
        packet["n_predictions"] = int(len(predictions_df))
    else:
        packet["n_predictions"] = 0

    if predictions_df is not None and hasattr(predictions_df, "columns") and "predicted_label" in predictions_df.columns:
        vals = predictions_df["predicted_label"].value_counts().to_dict()
        for label, count in vals.items():
            safe_label = str(label).strip()
            safe_label = safe_label.replace(" ", "_")
            safe_name = f"{safe_label}_Count"
            packet[safe_name] = int(count)

    # -----------------------------
    # 5. Final flattening rules
    # -----------------------------
    if "start" in packet and hasattr(packet["start"], "isoformat"):
        packet["start"] = packet["start"].isoformat()

    # Ensure simple numeric strings are converted where sensible
    for k, v in list(packet.items()):
        if isinstance(v, (int, float)) or v is None:
            continue
        if isinstance(v, str):
            converted = _safe_number(v)
            packet[k] = converted if converted is not None else v

    # Make everything JSON-safe before returning or writing
    packet = json_safe(packet)

    # -----------------------------
    # 6. Write output
    # -----------------------------
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(packet, f, indent=2)

    return packet