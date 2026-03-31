
# metadata_extraction.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# -------------------------------------------
# Utility helpers
# -------------------------------------------

def first_present_numeric(df: pd.DataFrame, candidates: list[str]) -> float | None:
    """
    Return the first present column's numeric value (row 0), coercing with NaN safety.
    """
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            v = pd.to_numeric(df[col].iloc[0], errors="coerce")
            if pd.notna(v):
                return float(v)
    return None
    
def _safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def first_present(df, candidates):
    for col in candidates:
        if col in df and df[col].notna().any():
            val = df[col].iloc[0]
            return _safe_float(val) if not isinstance(val, str) else val
    return None

def pick_timestamp(df):
    candidates = [
        "start", "measurementResults_start", "instrument_measurementResults_start",
        "measurementResults.dateTime", "measurementResults_dateTime",
        "dateTime", "instrument_dateTime",
    ]
    for col in candidates:
        if col in df and df[col].notna().any():
            ts = df[col].iloc[0]
            try:
                return pd.to_datetime(ts)
            except Exception:
                pass
    return None

# -------------------------------------------
# CSV + JSON parsing
# -------------------------------------------

def parse_instrument_json(path):
    with open(path, "r") as f:
        js = json.load(f)
    inst = js.get("instrument", {})

    flat = {}
    
    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}_{k}" if prefix else k, v)
        else:
            flat[prefix] = obj
    
    walk("", inst)

    df = pd.DataFrame([flat])
    return df

# -------------------------------------------
# Derived metrics
# -------------------------------------------

def compute_particle_rate(meta):
    cnt = meta.get("particleCount")
    ext = meta.get("externalPumpTime")
    dur = meta.get("duration")
    if cnt is None:
        return None
    if ext and ext > 0:
        return cnt / ext
    if dur and dur > 0:
        return cnt / dur
    return None

def compute_particle_concentration(meta):
    cnt = meta.get("particleCount")
    pumped = meta.get("pumpedVolume")
    if cnt is None or pumped is None or pumped <= 0:
        return None
    return cnt / pumped

# -------------------------------------------
# Optional saturation fraction
# -------------------------------------------

def compute_flat_signal_fraction(preds_df):
    if preds_df is None or len(preds_df) == 0:
        return None
    frac_list = []
    for channel in ["Fl_Yellow", "Fl_Orange", "Fl_Red"]:
        avg = preds_df.get(f"{channel}_average")
        mxx = preds_df.get(f"{channel}_maximum")
        if avg is None or mxx is None:
            continue
        ratio = avg / mxx.replace(0, np.nan)
        frac = (ratio >= 0.8).mean()
        if not np.isnan(frac):
            frac_list.append(frac)
    return float(np.mean(frac_list)) if frac_list else None

# -------------------------------------------
# Main entry point
# -------------------------------------------

def extract_metadata(instrument_csv_path=None, cyz_json_path=None, predictions_df=None):
    if cyz_json_path:
        df = parse_instrument_json(cyz_json_path)
        json_path = cyz_json_path
    else:
        raise ValueError("reporting requires a cyz_as_json path — cannot use instrument CSV")
    
    meta = {}

    meta["start"] = pick_timestamp(df)
    meta["triggerLevel"] = first_present(df, ["triggerLevel", "measurementSettings_CytoSettings_triggerLevel"])
    meta["pumpedVolume"] = first_present(df, ["pumpedVolume", "measurementResults_pumpedVolume"])
    meta["analysedVolume"] = first_present(df, ["analysedVolume", "measurementResults_analysedVolume"])

    pv = meta["pumpedVolume"]
    meta["halfPumpedVolume"] = pv / 2 if pv else None

    meta["duration"] = first_present(df, ["duration", "measurementResults_duration"])
    meta["externalPumpTime"] = first_present(df, ["externalPumpTime", "measurementResults_externalPumpTime"])
    meta["particleCount"] = first_present(df, ["measurementResults_particleCount", "particleCount"])

    meta["particleRate"] = compute_particle_rate(meta)
    meta["particleConcentration"] = compute_particle_concentration(meta)

    # --- Sensors ---
    meta["absolutePressure"]     = first_present_numeric(df, ["measurementResults_pressureAbsolute", "pressureAbsolute", "absolutePressure"])
    meta["differentialPressure"] = first_present_numeric(df, ["measurementResults_pressureDifferential", "pressureDifferential", "differentialPressure"])
    meta["sheathTemperature"]    = first_present_numeric(df, ["measurementResults_sheathTemperature", "sheathTemperature"])
    meta["systemTemperature"]    = first_present_numeric(df, ["measurementResults_systemTemperature", "systemTemperature"])
    meta["laserTemperature"]     = first_present_numeric(df, ["measurementResults_laserTemperature", "laserTemperature"])
    meta["PMTtemperature"]       = first_present_numeric(df, ["measurementResults_PMTtemperature", "PMTtemperature"])
    meta["laserBeamWidth"]       = first_present_numeric(df, ["laserBeamWidth", "measurementSettings_CytoSettings_LaserBeamWidth"])

    # Sample core speed appears under several spellings in exports
    meta["sampleCoreSpeed"]      = first_present_numeric(
        df,
        [
            "sampleCoreSpeed",
            "measurementSettings_CytoSettings_SampleCoreSpeed",
            "measurementSettings_CytoSettings_SampleCorespeed",
        ],
    )

    meta["flatSignalFraction"] = compute_flat_signal_fraction(predictions_df)

    meta["instrument_json_path"] = json_path
    return meta
