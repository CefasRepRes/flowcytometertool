"""
reporting.py — centralised JSON report‑packet builder
Used by: qc_plots.py, watcher, blob‑processor.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------
# 1. Load grablist.txt
# ------------------------------------------------------------

def load_grablist(grablist_path: str):
    paths = []
    with open(grablist_path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    return paths


# ------------------------------------------------------------
# 2. Path parsing and JSON navigation
# ------------------------------------------------------------

import re
_PATH_TOKEN_RE = re.compile(r"""
 (?P<key>[^.\[\]]+)     # key
|\[(?P<idx>\d+)\]       # [3]
|\[\*\]                 # [*]
""", re.VERBOSE)

def iter_tokens(path: str):
    """Yield ('key', name) or ('idx', int) or ('wild', None)"""
    pos = 0
    while pos < len(path):
        m = _PATH_TOKEN_RE.match(path, pos)
        if not m:
            raise ValueError(f"Bad path syntax near: {path[pos:]} in '{path}'")
        if m.group("key"):
            yield ("key", m.group("key"))
        elif m.group("idx"):
            yield ("idx", int(m.group("idx")))
        else:
            yield ("wild", None)
        pos = m.end()


def walk_json(root, path: str, *, join_all=False, sep=";"):
    """
    Reusable version of qc_plots._get_by_path.
    Returns first match unless join_all=True.
    """

    def step(node, tokens):
        if not tokens:
            return [node]
        ttype, tval = tokens[0]
        rest = tokens[1:]

        if ttype == "key":
            if isinstance(node, dict) and tval in node:
                return step(node[tval], rest)
            return []
        elif ttype == "idx":
            if isinstance(node, list) and 0 <= tval < len(node):
                return step(node[tval], rest)
            return []
        else:  # wildcard
            if not isinstance(node, list):
                return []
            out = []
            for item in node:
                out.extend(step(item, rest))
            return out

    try:
        tokens = list(iter_tokens(path))
    except ValueError:
        return None

    matches = step(root, tokens)
    if not matches:
        return None

    if not join_all:
        return matches[0]

    def as_text(x):
        if x is None: return ""
        if isinstance(x, (bool, int, float, str)): return str(x)
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    return sep.join(as_text(m) for m in matches)


# ------------------------------------------------------------
# 3. Extract grablist fields
# ------------------------------------------------------------

def extract_grablist_fields(inst_json: dict, grablist_path: str, join_wildcards=False):
    out = {}
    paths = load_grablist(grablist_path)
    for p in paths:
        if p == "filename":
            v = inst_json.get("filename")
        else:
            v = walk_json(inst_json, p, join_all=join_wildcards)
        out[p] = to_scalar(v)
    return out


# ------------------------------------------------------------
# 4. Utility for JSON-serialisable scalars
# ------------------------------------------------------------

def to_scalar(x):
    """Return JSON‑safe primitive. NaN/Inf → None."""
    try:
        if x is None:
            return None
        if isinstance(x, (bool, int, float, str)):
            if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                return None
            return x
        if isinstance(x, (np.floating,)):
            v = float(x)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
        if isinstance(x, dict):
            return {k: to_scalar(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_scalar(v) for v in x]
    except Exception:
        return str(x)
    return str(x)


# ------------------------------------------------------------
# 5. Build the main packet
# ------------------------------------------------------------

def build_packet(file_prefix, qc_row: dict, preds_df, inst_json=None,
                 grablist_path=None):
    """
    Returns a flat dashboard-compatible dict.
    qc_row = from build_measurement_row()
    preds_df = predictions dataframe
    inst_json = cyz2json JSON dict (optional)
    """

    packet = {
        "version": "0.0.3",
        "system_serial_no": "flowcytometer01",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "file_id": file_prefix,
    }

    # --- Time fields ---
    packet["time_start"] = to_scalar(qc_row.get("start"))
    packet["time_end"] = packet["timestamp"]

    # --- Core metrics copied directly from qc_row ---
    for key in [
        "pumpedVolume", "analysedVolume", "particleCount",
        "particleConcentration", "particleRate", "externalPumpTime",
        "duration", "triggerLevel"
    ]:
        packet[key] = to_scalar(qc_row.get(key))

    # --- Sensors (if present) ---
    sensor_keys = [
        "absolutePressure", "differentialPressure",
        "sheathTemperature", "systemTemperature",
        "laserTemperature", "PMTtemperature",
        "sampleCoreSpeed", "laserBeamWidth"
    ]
    for k in sensor_keys:
        if k in qc_row:
            packet[k] = to_scalar(qc_row.get(k))

    # --- Class counts ---
    if preds_df is not None and "predicted_label" in preds_df.columns:
        counts = preds_df["predicted_label"].value_counts().to_dict()
        for label, count in counts.items():
            packet[f"{label}_Count"] = int(count)

    # --- Grablist metadata (if inst_json provided) ---
    if inst_json is not None and grablist_path is not None:
        packet["grablist"] = extract_grablist_fields(
            inst_json, grablist_path, join_wildcards=False
        )

    return packet


# ------------------------------------------------------------
# 6. Write packet to disk
# ------------------------------------------------------------

def write_packet_json(packet, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2)
    return out_path