from __future__ import annotations

import datetime as _dt
import os

import joblib
import numpy as np
import pandas as pd

from flowcytometer_tool.core.feature_columns import (
    apply_log10_calibration,
    find_column_case_insensitive,
    infer_beads_channel_from_column,
)

_CALIBRATED_SUFFIX = "_calibrated"


def load_classifier_from_exact_path(exact_model_path: str | None, label: str):
    if exact_model_path is None:
        raise FileNotFoundError(f"No {label} model path is configured.")
    if not os.path.isfile(exact_model_path):
        raise FileNotFoundError(f"{label} model file does not exist: {exact_model_path}")

    print(f"loading exact {label} model:")
    print(exact_model_path)

    fitted_model = joblib.load(exact_model_path)
    classes_ = fitted_model["classifier"].classes_
    features_ = fitted_model["selector"].feature_names_in_
    return fitted_model, classes_, features_


def extract_bead_coefficients(bead_record: dict | None) -> dict[str, tuple[float, float]]:
    coeffs: dict[str, tuple[float, float]] = {}
    channels = (bead_record or {}).get("channels") or {}

    for channel, channel_record in channels.items():
        if not isinstance(channel_record, dict):
            continue

        calibration_coeffs = (
            channel_record.get("calibration_curve_coefficients")
            or channel_record.get("calibration_coefficients")
            or {}
        )

        try:
            slope = float(calibration_coeffs.get("slope"))
            intercept = float(calibration_coeffs.get("intercept"))
        except Exception:
            continue

        if np.isfinite(slope) and np.isfinite(intercept):
            coeffs[str(channel).lower()] = (slope, intercept)
    return coeffs


def latest_bead_calibration_status(load_latest_record_fn, *, max_age_seconds: int) -> tuple[bool, dict | None, str]:
    max_age_seconds = int(max_age_seconds)
    try:
        record = load_latest_record_fn()
        if record is None:
            return False, None, "no saved bead calibration record found"

        time_str = record.get("time_calculated")
        if not time_str:
            return False, record, "saved bead calibration has no time_calculated value"

        cal_time = pd.to_datetime(time_str, utc=True)
        age = _dt.datetime.now(_dt.timezone.utc) - cal_time.to_pydatetime()

        if age.total_seconds() >= max_age_seconds:
            return False, record, f"saved bead calibration is older than max age ({max_age_seconds} seconds)"

        return True, record, f"saved bead calibration is younger than max age ({max_age_seconds} seconds)"
    except Exception as e:
        return False, None, f"could not check bead calibration age: {e}"


def apply_bead_calibration_for_model_features(
    raw_df: pd.DataFrame,
    bead_record: dict | None,
    model_features,
    *,
    apply_saved_bead_calibration_to_dataframe_fn,
) -> pd.DataFrame:
    df_out = raw_df.copy()
    df_out.columns = df_out.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)

    raw_source = raw_df.copy()
    raw_source.columns = raw_source.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)

    coeffs = extract_bead_coefficients(bead_record)
    if not coeffs:
        raise ValueError("Saved bead calibration record contains no finite coefficients.")

    model_features = [str(f) for f in model_features]
    explicit_calibrated_features = [f for f in model_features if f.endswith(_CALIBRATED_SUFFIX)]
    created_columns = []

    if explicit_calibrated_features:
        for feature in explicit_calibrated_features:
            raw_feature_name = feature[: -len(_CALIBRATED_SUFFIX)]
            raw_col = find_column_case_insensitive(raw_source, raw_feature_name)
            if raw_col is None:
                continue

            channel = infer_beads_channel_from_column(raw_col)
            if channel is None or channel not in coeffs:
                continue

            slope, intercept = coeffs[channel]
            df_out[feature] = apply_log10_calibration(raw_source[raw_col], slope=slope, intercept=intercept)
            created_columns.append(feature)

        if not created_columns:
            raise ValueError(
                "The bead-calibrated model expects *_calibrated features, "
                "but no matching raw columns could be calibrated."
            )

        print("Applied saved bead calibration to explicit calibrated model features:")
        print(created_columns)
        return df_out

    df_out = apply_saved_bead_calibration_to_dataframe_fn(df_out, bead_record)
    print(
        "Applied saved bead calibration using in-place runtime helper "
        "(model does not expect *_calibrated columns)."
    )
    return df_out


def run_prediction(model, classes, features, df_for_prediction: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    features = [str(f) for f in features]
    print("Your model expects these columns:", features)
    print("Your data file has these columns:", df_for_prediction.columns.tolist())

    missing_features = [f for f in features if f not in df_for_prediction.columns]
    if missing_features:
        raise KeyError(
            "Columns in this data file do not match the model's training features. "
            f"Missing columns: {missing_features}"
        )

    X = df_for_prediction.loc[:, features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    rows_before = len(X)
    good_rows = ~X.isna().any(axis=1)
    if not good_rows.all():
        dropped = int((~good_rows).sum())
        print(
            f"Dropping {dropped} rows with NaN/inf in model feature columns "
            f"after calibration/feature selection."
        )
        X = X.loc[good_rows].copy()

    if X.empty:
        raise ValueError("No rows remain for prediction after feature selection and NaN/inf cleaning.")

    print("Predicting ...")
    predictions = model.predict(X)
    proba_predict = pd.DataFrame(model.predict_proba(X), index=X.index).set_axis(classes, axis=1)

    predicted_data = df_for_prediction.loc[X.index].copy()
    if "predicted_label" in predicted_data.columns:
        predicted_data = predicted_data.drop(columns=["predicted_label"])

    predicted_data["predicted_label"] = predictions
    existing_probability_cols = [c for c in proba_predict.columns if c in predicted_data.columns]
    if existing_probability_cols:
        predicted_data = predicted_data.drop(columns=existing_probability_cols, errors="ignore")

    full_predicted = pd.concat([predicted_data, proba_predict], axis=1)
    return full_predicted, rows_before
