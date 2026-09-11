from __future__ import annotations

import json
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from flowcytometer_tool.config.runtime import get_runtime_config
from flowcytometer_tool.tabs.continuous_sample_analyser.json_safe import json_safe

from flowcytometer_tool.tabs.continuous_sample_analyser.bead_calibration import (
    compute_beads_calibration_from_df,
    append_beadscalibration_record,
)
from flowcytometer_tool.tabs.continuous_sample_analyser.protocols import (
    detect_sampling_protocol,
    is_bead_sample,
)
from flowcytometer_tool.core.feature_columns import (
    apply_log10_calibration as _apply_log10_calibration,
    infer_beads_channel_from_column as _infer_beads_channel_from_column,
    is_calibratable_fluorescence_column as _is_calibratable_fluorescence_column,
    normalise_feature_name as _normalise_feature_name,
)


def _is_fws_related_column(column_name: str) -> bool:
    return "fws" in _normalise_feature_name(column_name)


def _calibrated_feature_column_name(column_name: str) -> str:
    """
    Return the explicit calibrated feature name for a raw instrument feature.

    Example:
        Fl_Red_total -> Fl_Red_total_calibrated

    If a column is already marked as calibrated, leave it unchanged.
    """
    col = str(column_name).strip()
    if col.endswith("_calibrated"):
        return col
    return f"{col}_calibrated"

def _extract_bead_sample(sample) -> tuple[dict | None, pd.DataFrame | None]:
    packet = None
    dataframe = None
    if isinstance(sample, dict):
        packet = sample.get("packet")
        dataframe = sample.get("dataframe", sample.get("df"))
    elif isinstance(sample, (tuple, list)) and len(sample) == 2:
        packet, dataframe = sample
    return packet, dataframe if isinstance(dataframe, pd.DataFrame) else None

def _clean_calibrated_training_dataframe(
    df: pd.DataFrame,
    *,
    target_col: str = "source_label",
) -> tuple[pd.DataFrame, dict]:
    """
    Remove rows with NaN/inf in actual model feature columns only.

    Metadata columns are not treated as features and are not converted to numeric.
    This avoids dropping all rows because columns like filename/person/datetime
    cannot be converted to numbers.
    """

    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    metadata_cols = {
        target_col,
        "weight",
        "sample_weight",
        "filename",
        "datetime",
        "person",
        "consensus_label",
        "terminal_classes",
        "n_terminal_classes",
        "id",
        "user_id",
        "location",
        "group",
    }

    metadata_cols = {c for c in metadata_cols if c in cleaned.columns}

    feature_cols = [
        c for c in cleaned.columns
        if c not in metadata_cols
    ]

    # Convert only model feature columns to numeric.
    for col in feature_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    nan_counts = cleaned[feature_cols].isna().sum()
    bad_feature_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

    bad_row_mask = cleaned[feature_cols].isna().any(axis=1)

    meta = {
        "rows_before_nan_cleaning": int(len(cleaned)),
        "rows_with_nan_or_inf_in_features": int(bad_row_mask.sum()),
        "rows_after_nan_cleaning": int((~bad_row_mask).sum()),
        "metadata_columns_ignored_for_nan_cleaning": sorted(metadata_cols),
        "feature_columns_checked_for_nan_cleaning": feature_cols,
        "bad_feature_nan_counts": {
            str(k): int(v)
            for k, v in bad_feature_cols.to_dict().items()
        },
    }

    if bad_feature_cols.empty:
        print("[calibrated training clean] no NaN/inf feature values found")
        return cleaned, meta

    print("[calibrated training clean] NaN/inf detected in actual feature columns")
    print("[calibrated training clean] metadata columns ignored:")
    print(sorted(metadata_cols))
    print("[calibrated training clean] bad feature columns and counts:")
    print(bad_feature_cols.to_string())
    print(
        "[calibrated training clean] dropping "
        f"{int(bad_row_mask.sum())} rows from {len(cleaned)}"
    )

    cleaned = cleaned.loc[~bad_row_mask].copy()

    return cleaned, meta


   
def prepare_training_dataframe_with_optional_bead_calibration(
    training_df: pd.DataFrame,
    bead_samples: list | None = None,
    *,
    allow_spoof_bead_calibration: bool = False,
    diagnostic_dir: str | Path | None = None,
    beads_output_root: str | Path | None = None,
) -> tuple[pd.DataFrame, bool, dict]:
    """
    Prepare training dataframe using bead calibration.

    Real mode:
        Uses the supplied bead sample and detects bead peaks.

    Spoof fallback mode:
        If real calibration fails and allow_spoof_bead_calibration=True,
        retries compute_beads_calibration_from_df(..., spoof_bead_sample=True).

    IMPORTANT:
        Spoof calibration is synthetic/debug calibration. It is useful for testing
        the training pipeline and feature filtering, but it is not a real calibration
        derived from the bead sample.
    """

    meta = {
        "bead_sample_count": 0,
        "valid_bead_frame_count": 0,
        "bead_frame_shapes": [],
        "calibration_status": "not_started",
        "used_spoof_bead_calibration": False,
        "calibration_mode": "none",
        "calibration_channels": [],
        "channel_coeffs": {},
        "calibrated_columns": [],
        "calibrated_column_map": {},
        "raw_columns_replaced_by_calibrated": [],
        "kept_columns": [],
        "dropped_columns": [],
    }

    if training_df is None:
        meta["calibration_status"] = "training_df_is_none"
        print("[bead calibration] training_df is None")
        return training_df, False, meta

    if not bead_samples:
        meta["calibration_status"] = "no_bead_samples"
        print("[bead calibration] no bead_samples supplied")
        return training_df, False, meta

    meta["bead_sample_count"] = len(bead_samples)

    valid_bead_frames: list[pd.DataFrame] = []

    for i, sample in enumerate(bead_samples):
        packet, bead_df = _extract_bead_sample(sample)

        print(f"[bead calibration] checking bead sample {i}")
        print(f"[bead calibration] packet={packet}")

        if bead_df is None:
            print(f"[bead calibration] sample {i} skipped: bead dataframe is None")
            continue

        if bead_df.empty:
            print(f"[bead calibration] sample {i} skipped: bead dataframe is empty")
            continue

        if not isinstance(packet, dict):
            print(f"[bead calibration] sample {i} skipped: packet is not a dict")
            continue

        protocol = detect_sampling_protocol(packet)
        print(f"[bead calibration] sample {i} detected protocol={protocol}")

        if not is_bead_sample(packet, protocol=protocol):
            print(f"[bead calibration] sample {i} skipped: not beadsprotocol")
            continue

        bead_df_prepared = bead_df.copy()
        bead_df_prepared.columns = (
            bead_df_prepared.columns
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
        )

        valid_bead_frames.append(bead_df_prepared)
        meta["bead_frame_shapes"].append(tuple(bead_df_prepared.shape))

    meta["valid_bead_frame_count"] = len(valid_bead_frames)

    if not valid_bead_frames:
        meta["calibration_status"] = "no_valid_bead_frames"
        print("[bead calibration] no valid bead frames after protocol filtering")
        return training_df, False, meta

    bead_reference_df = pd.concat(valid_bead_frames, ignore_index=True)

    print("[bead calibration] bead_reference_df shape:")
    print(bead_reference_df.shape)
    print("[bead calibration] bead_reference_df columns:")
    print(list(bead_reference_df.columns))


    diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir else Path(".")
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    real_diag_path = diagnostic_dir / "training_beads_calibration_diagnostic_real.png"
    spoof_diag_path = diagnostic_dir / "training_beads_calibration_diagnostic_spoofed.png"


    # 1. Try real calibration first.
    calibration = compute_beads_calibration_from_df(
        bead_reference_df,
        spoof_bead_sample=False,
        diagnostic_png_path=real_diag_path,
        verbose=True,
    )

    if calibration is not None:
        meta["calibration_mode"] = "real_bead_sample"
        meta["used_spoof_bead_calibration"] = False
        meta["beads_calibration_plot"] = str(real_diag_path)
        print("[bead calibration] real bead calibration succeeded")
        print(f"[bead calibration] diagnostic plot written: {real_diag_path}")
        
        
        try:
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if beads_output_root is None:
                beads_output_root = get_runtime_config().paths.tool_dir

            out_dir = Path(beads_output_root) / "BeadsCalibrations"
            out_path = out_dir / "BeadsCalibrations.jsonl"

            rec = {
                "time_calculated": ts,
                "file_id": "training_bead_sample",
                "protocol": "beadsprotocol",
                **calibration,
                "diagnostic_plot": meta.get("beads_calibration_plot"),
                "training_context": True,
                "calibration_mode": meta.get("calibration_mode"),
                "used_spoof_bead_calibration": bool(meta.get("used_spoof_bead_calibration", False)),
            }

            append_beadscalibration_record(rec, out_path)
            meta["beads_calibration_record"] = str(out_path)

            print(f"[bead calibration] calibration JSONL record appended: {out_path}")

        except Exception as e:
            print(f"[bead calibration] warning: could not append calibration JSONL record: {e}")        

    # 2. Optional spoof fallback.
    elif allow_spoof_bead_calibration:
        print(
            "[bead calibration] real bead calibration failed; "
            "retrying with spoof_bead_sample=True"
        )

        calibration = compute_beads_calibration_from_df(
            bead_reference_df,
            spoof_bead_sample=True,
            diagnostic_png_path=spoof_diag_path,
            verbose=True,
        )

        if calibration is not None:
            meta["calibration_mode"] = "spoofed_synthetic_bead_sample"
            meta["used_spoof_bead_calibration"] = True
            meta["beads_calibration_plot"] = str(spoof_diag_path)
            print(
                "[bead calibration] WARNING: using spoofed synthetic bead calibration. "
                "This is suitable for pipeline testing, not real scientific calibration."
            )
            print(f"[bead calibration] diagnostic plot written: {spoof_diag_path}")
            
                        
            try:
                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

                if beads_output_root is None:
                    beads_output_root = get_runtime_config().paths.tool_dir

                out_dir = Path(beads_output_root) / "BeadsCalibrations"
                out_path = out_dir / "BeadsCalibrations.jsonl"

                rec = {
                    "time_calculated": ts,
                    "file_id": "training_bead_sample",
                    "protocol": "beadsprotocol",
                    **calibration,
                    "diagnostic_plot": meta.get("beads_calibration_plot"),
                    "training_context": True,
                    "calibration_mode": meta.get("calibration_mode"),
                    "used_spoof_bead_calibration": bool(meta.get("used_spoof_bead_calibration", False)),
                }

                append_beadscalibration_record(rec, out_path)
                meta["beads_calibration_record"] = str(out_path)

                print(f"[bead calibration] calibration JSONL record appended: {out_path}")

            except Exception as e:
                print(f"[bead calibration] warning: could not append calibration JSONL record: {e}")
    
    
    if calibration is None:
        meta["calibration_status"] = "calibration_failed_real_and_spoof"
        print("[bead calibration] calibration failed")
        return training_df, False, meta

    channels = calibration.get("channels") or {}
    meta["calibration_channels"] = list(channels.keys())

    if not channels:
        meta["calibration_status"] = "calibration_has_no_channels"
        print("[bead calibration] calibration object has no channels")
        return training_df, False, meta

    channel_coeffs: dict[str, tuple[float, float]] = {}

    for channel, rec in channels.items():
        coeffs = rec.get("calibration_curve_coefficients", {}) if isinstance(rec, dict) else {}
        slope = coeffs.get("slope")
        intercept = coeffs.get("intercept")

        print(f"[bead calibration] channel={channel}, slope={slope}, intercept={intercept}")

        if slope is None or intercept is None:
            continue

        try:
            slope_f = float(slope)
            intercept_f = float(intercept)
        except Exception:
            continue

        if np.isfinite(slope_f) and np.isfinite(intercept_f):
            channel_coeffs[str(channel).lower()] = (slope_f, intercept_f)

    meta["channel_coeffs"] = {
        k: {"slope": v[0], "intercept": v[1]}
        for k, v in channel_coeffs.items()
    }

    if not channel_coeffs:
        meta["calibration_status"] = "no_finite_channel_coefficients"
        print("[bead calibration] no finite slope/intercept coefficients found")
        return training_df, False, meta

    prepared = training_df.copy()
    prepared.columns = (
        prepared.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    print("[bead calibration] incoming training_df shape:")
    print(prepared.shape)
    print("[bead calibration] incoming training_df columns:")
    print(list(prepared.columns))

    calibrated_columns: list[str] = []
    calibrated_column_map: dict[str, str] = {}
    raw_columns_replaced_by_calibrated: list[str] = []

    for col in list(prepared.columns):
        if not _is_calibratable_fluorescence_column(col):
            continue

        channel = _infer_beads_channel_from_column(col)
        if channel is None:
            continue

        channel = str(channel).lower()

        if channel not in channel_coeffs:
            continue

        slope, intercept = channel_coeffs[channel]
        calibrated_col = _calibrated_feature_column_name(col)

        prepared[calibrated_col] = _apply_log10_calibration(
            prepared[col],
            slope=slope,
            intercept=intercept,
        )

        calibrated_columns.append(calibrated_col)
        calibrated_column_map[col] = calibrated_col
        raw_columns_replaced_by_calibrated.append(col)

    meta["calibrated_columns"] = calibrated_columns
    meta["calibrated_column_map"] = calibrated_column_map
    meta["raw_columns_replaced_by_calibrated"] = raw_columns_replaced_by_calibrated

    if not calibrated_columns:
        meta["calibration_status"] = "no_training_columns_matched_calibration_channels"
        print("[bead calibration] coefficients were found, but no training columns were calibrated")
        print("[bead calibration] available channel_coeffs:")
        print(channel_coeffs)
        return training_df, False, meta

    always_keep = {
        "source_label",
        "weight",
        "filename",
        "consensus_label",
        "datetime",
        "user_id",
        "location",
        "terminal_classes",
        "n_terminal_classes",
        "person",
        "sample_weight",
        "id",
    }

    keep_cols: list[str] = []
    calibrated_set = set(calibrated_columns)
    raw_calibrated_source_set = set(raw_columns_replaced_by_calibrated)

    for col in prepared.columns:
        if col in always_keep:
            keep_cols.append(col)
            continue

        # Option A:
        # Keep the explicitly calibrated predictor.
        if col in calibrated_set:
            keep_cols.append(col)
            continue

        # Drop the raw fluorescence predictor when a calibrated version exists.
        # Example:
        #   drop Fl_Red_total
        #   keep Fl_Red_total_calibrated
        if col in raw_calibrated_source_set:
            continue

        # Keep FWS-related columns as before.
        if _is_fws_related_column(col):
            keep_cols.append(col)
            continue

    dropped_cols = [c for c in prepared.columns if c not in keep_cols]

    prepared = prepared.loc[:, keep_cols].copy()

    meta["kept_columns"] = keep_cols
    meta["dropped_columns"] = dropped_cols

    prepared, nan_clean_meta = _clean_calibrated_training_dataframe(
        prepared,
        target_col="source_label",
    )

    meta["nan_cleaning"] = nan_clean_meta

    if prepared.empty:
        meta["calibration_status"] = "all_rows_removed_by_nan_cleaning"
        print("[bead calibration] ERROR: all rows were removed during NaN cleaning")
        return training_df, False, meta

    meta["calibration_status"] = "success"

    print("[bead calibration] SUCCESS")
    print(f"[bead calibration] calibration_mode={meta.get('calibration_mode')}")
    print(f"[bead calibration] used_spoof_bead_calibration={meta.get('used_spoof_bead_calibration')}")
    print(f"[bead calibration] calibrated columns created: {calibrated_columns}")
    print(f"[bead calibration] calibrated column map: {calibrated_column_map}")
    print(
        "[bead calibration] raw fluorescence columns dropped from model features: "
        f"{raw_columns_replaced_by_calibrated}"
    )
    print(f"[bead calibration] kept column count: {len(keep_cols)}")
    print(f"[bead calibration] dropped column count: {len(dropped_cols)}")
    print("[bead calibration] NaN cleaning summary:")
    print(nan_clean_meta)
    print("[bead calibration] final prepared shape:")
    print(prepared.shape)
    print("[bead calibration] final kept columns:")
    print(list(prepared.columns))

    return prepared, True, meta



def beadcalibrated_model_path(model_path: str) -> str:
    if model_path.endswith("_beadcalibrated.pkl"):
        return model_path
    if model_path.endswith(".pkl"):
        return model_path[:-4] + "_beadcalibrated.pkl"
    return model_path + "_beadcalibrated"


def update_modeltrainsettings(
    modeltrainsettings_path: str,
    training_updates: dict,
) -> dict:
    mts_path = Path(modeltrainsettings_path)
    mts = {}
    if mts_path.exists():
        with open(mts_path, "r", encoding="utf-8") as f:
            mts = json.load(f)
    training = mts.get("training", {})
    training.update(training_updates)
    mts["training"] = training
    mts_path.parent.mkdir(parents=True, exist_ok=True)
    mts = json_safe(mts)
    with open(mts_path, "w", encoding="utf-8") as f:
        json.dump(mts, f, indent=2)
    return mts


def update_modeltrainsettings_bead_flag(modeltrainsettings_path: str, used_bead_calibration: bool, bead_sample_count: int = 0) -> None:
    update_modeltrainsettings(
        modeltrainsettings_path,
        {
            "bead_calibrated": bool(used_bead_calibration),
            "bead_sample_count": int(bead_sample_count),
        },
    )
