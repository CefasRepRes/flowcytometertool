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

def apply_sampling_protocol_mutations(packet: dict) -> dict:
    """
    Mutate the report packet in-place (and return it) by:
      1) Determining samplingprotocol from two instrument settings:
         - instrument.measurementSettings.CytoSettings.SamplePompSpeed
         - instrument.measurementSettings.CytoSettings.TriggerLevel1e
      2) Adding packet["samplingprotocol"] ∈ {"nanoprotocol","picoprotocol","unknownprotocol"}
      3) Renaming any keys ending with "_Count" to include protocol suffix:
           "<label>_Count" -> "<label>_Count_<samplingprotocol>"
         (only if they don't already have a protocol suffix)
    """
    def _to_float(v):
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str) and v.strip() != "":
                return float(v)
        except Exception:
            return None
        return None

    def _in_range(x, lo, hi):
        return (x is not None) and (lo <= x <= hi)

    # --- pull values from packet (these exist if your grablist includes them) ---
    pump_key = "instrument.measurementSettings.CytoSettings.SamplePompSpeed"
    trig_key = "instrument.measurementSettings.CytoSettings.TriggerLevel1e"

    pump = _to_float(packet.get(pump_key))
    trig = _to_float(packet.get(trig_key))

    # --- classify each signal ---
    pump_is_nano = _in_range(pump, 8.5, 9.5)
    pump_is_pico = _in_range(pump, 3.5, 4.5)

    trig_is_nano = _in_range(trig, 2.9, 5.1)
    trig_is_pico = _in_range(trig, 1.8, 2.5)

    # --- rule: BOTH conditions must be met for a protocol ---
    if pump_is_nano and trig_is_nano:
        protocol = "nanoprotocol"
    elif pump_is_pico and trig_is_pico:
        protocol = "picoprotocol"
    else:
        protocol = "unknownprotocol"

    if packet.get("instrument.measurementSettings.CytoSettings.IIFCheck"):
        protocol = "imageprotocol"
    
    packet["samplingprotocol"] = protocol

    # --- rename *_Count keys to include protocol suffix ---
    # e.g. "redpico_Count" -> "redpico_Count_picoprotocol"
    # Do this on a snapshot of keys to avoid runtime mutation issues.
    keys = list(packet.keys())
    for k in keys:
        if not k.endswith("_Count"):
            continue

        # Skip if it already has a protocol suffix
        if k.endswith("_Count_nanoprotocol") or k.endswith("_Count_picoprotocol") or k.endswith("_Count_unknownprotocol"):
            continue

        new_key = f"{k}_{protocol}"

        # Avoid clobbering if it already exists for some reason
        if new_key in packet:
            continue

        packet[new_key] = packet.pop(k)

    return packet

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


# ------------------------------------------------------------
# MAIN: WRITE REPORT PACKET (FLAT)
# ------------------------------------------------------------
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
      - metadata (from metadata_extraction.extract_metadata)
      - selected JSON fields from grablist
      - dashboard header fields (version, serial, timestamps, survey, lat/lon)
      - optional plot paths
      - optional predictions-based summary flags

    Returns the packet dict. If output_path provided, writes JSON there.
    """

    from datetime import datetime, timezone
    import os

    packet = {}

    # -----------------------------
    # 1. Core metadata
    # -----------------------------
    packet.update(metadata)

    # -----------------------------
    # 2. GRABLIST FIELDS + dashboard header source JSON
    # -----------------------------
    grablist_items = _load_grablist(grablist_path)

    with open(json_path, "r") as f:
        full_js = json.load(f)


    try:
        iifcheck = (
            full_js.get("instrument", {})
                  .get("measurementSettings", {})
                  .get("CytoSettings", {})
                  .get("IIFCheck", None)
        )
    except Exception:
        iifcheck = None
    packet["instrument.measurementSettings.CytoSettings.IIFCheck"] = iifcheck   

    # Extract user grablist fields (flattened into packet)
    grabbed = _extract_grablist_fields(full_js, grablist_items)
    for k, v in grabbed.items():
        packet[k] = v

    # -----------------------------
    # 3. DASHBOARD HEADER FIELDS
    # -----------------------------
    # version
    packet["version"] = "0.0.3"

    # system_serial_no  (real one from JSON)
    serial = None
    try:
        serial = full_js["instrument"]["serialNumber"]
    except Exception:
        serial = "unknown"

    packet["system_serial_no"] = serial

    # current timestamp / packet build time
    now = datetime.now(timezone.utc).isoformat()
    packet["timestamp"] = now
    packet["time_end"] = now
    
    
    from datetime import datetime, timedelta

    # Extract real start time from JSON
    start_str = full_js["instrument"]["measurementResults"]["start"]
    #time_start_dt = datetime.fromisoformat(start_str)
    time_start_dt = pd.to_datetime(start_str).to_pydatetime()
            #time_start_dt = metadata["start"]

    # Extract duration (seconds)
    dur = full_js["instrument"]["measurementResults"]["duration"]
    # Compute time_end = start + duration
    time_end_dt = time_start_dt + timedelta(seconds=dur)

    # Store in packet as ISO strings
    packet["time_start"] = time_start_dt.isoformat()
    packet["time_end"]   = time_end_dt.isoformat()    
    
    

    # measurement start time (REAL one from JSON)
    start_js = None
    try:
        start_js = full_js["instrument"]["measurementResults"]["start"]
    except Exception:
        start_js = None

    # if JSON start present → use it
    # else fallback to metadata['start']
    if start_js:
        packet["time_start"] = start_js
    else:
        if "start" in metadata and metadata["start"] is not None:
            packet["time_start"] = metadata["start"].isoformat()
        else:
            packet["time_start"] = now

    # lat/lon remain fixed at 0
    packet["latitude"] = 0
    packet["longitude"] = 0

    # survey name extracted from SaveTextbox
    # e.g. "C:\\Users\\Cyto\\Documents\\My CytoSense\\Datafiles\\mNCEA_june2023"
    survey_raw = None
    try:
        survey_raw = full_js["instrument"]["measurementSettings"]["CytoSettings"]["SaveTextbox"]
    except Exception:
        survey_raw = None

    if survey_raw:
        # extract final folder name
        survey = os.path.basename(survey_raw.replace("\\", "/"))
        packet["survey"] = survey
    else:
        packet["survey"] = "not specified"

    packet["model_file"] = modelversion

    # -----------------------------
    # 4. Predictions summary
    # -----------------------------
    if predictions_df is not None and len(predictions_df) > 0:
        packet["n_predictions"] = int(len(predictions_df))
    else:
        packet["n_predictions"] = 0
        
    if predictions_df is not None and "predicted_label" in predictions_df.columns:
        vals = predictions_df["predicted_label"].value_counts().to_dict()
        for label, count in vals.items():
            safe_name = f"{label}_Count"
            packet[safe_name] = int(count)
            
    # -----------------------------
    # 5. Final flattening rules
    # -----------------------------
    # Convert datetimes to ISO format
    if "start" in packet and hasattr(packet["start"], "isoformat"):
        packet["start"] = packet["start"].isoformat()

    # Ensure numeric fields are JSON‑serialisable
    for k, v in list(packet.items()):
        if isinstance(v, (int, float)) or v is None:
            continue
        if isinstance(v, str):
            packet[k] = _safe_number(v) if _safe_number(v) is not None else v

    # -----------------------------
    # 6. Write output
    # -----------------------------
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(packet, f, indent=2)

    return packet