from __future__ import annotations

import re

import numpy as np
import pandas as pd

CHANNEL_TOKEN_MAP: dict[str, tuple[str, ...]] = {
    "orange": ("orange", "ora", "mepe"),
    "red": ("red", "flr", "rws", "meptr"),
    "yellow": ("yellow", "fly", "yb", "meapc"),
}

CALIBRATABLE_FLUORESCENCE_SUFFIXES = {
    "total",
    "maximum",
    "average",
}

NON_CALIBRATABLE_FLUORESCENCE_SUFFIXES = {
    "length",
    "inertia",
    "centreofgravity",
    "centerofgravity",
    "fillfactor",
    "asymmetry",
    "numberofcells",
    "samplelength",
    "timeofarrival",
    "first",
    "last",
    "swscov",
    "variablelength",
}


def normalise_feature_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def infer_beads_channel_from_column(column_name: str) -> str | None:
    norm = normalise_feature_name(column_name)
    for channel, tokens in CHANNEL_TOKEN_MAP.items():
        if any(tok in norm for tok in tokens):
            return channel
    return None


def feature_suffix(column_name: str) -> str:
    return normalise_feature_name(str(column_name).split("_")[-1])


def is_calibratable_fluorescence_column(column_name: str) -> bool:
    if infer_beads_channel_from_column(column_name) is None:
        return False

    suffix = feature_suffix(column_name)
    if suffix in NON_CALIBRATABLE_FLUORESCENCE_SUFFIXES:
        return False
    return suffix in CALIBRATABLE_FLUORESCENCE_SUFFIXES


def find_column_case_insensitive(dataframe: pd.DataFrame, wanted: str) -> str | None:
    wanted_norm = normalise_feature_name(wanted)
    for col in dataframe.columns:
        if normalise_feature_name(col) == wanted_norm:
            return col
    return None


def apply_log10_calibration(values: pd.Series, slope: float, intercept: float) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr > 0)

    if np.any(mask):
        out[mask] = np.power(10.0, (float(slope) * np.log10(arr[mask])) + float(intercept))
    return pd.Series(out, index=values.index)
