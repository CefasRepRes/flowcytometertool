from __future__ import annotations

import datetime
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flowcytometer_tool.config.runtime import get_runtime_config
from flowcytometer_tool.core.feature_columns import (
    apply_log10_calibration as _apply_runtime_log10_calibration,
    infer_beads_channel_from_column as _infer_runtime_beads_channel_from_column,
    is_calibratable_fluorescence_column as _is_runtime_calibratable_fluorescence_column,
    normalise_feature_name as _normalise_runtime_feature_name,
)
from flowcytometer_tool.tabs.continuous_sample_analyser.json_safe import json_safe

def _find_beads_channel_column(df: pd.DataFrame, channel_name: str) -> str | None:
    channel_tokens = {
        "orange": ("orange", "ora", "mepe"),
        "red": ("red", "rws", "flr", "meptr"),
        "yellow": ("yellow", "yb", "fly", "meapc"),
    }[channel_name]

    priority_suffixes = ("_total", "_maximum", "_average")
    cols = list(df.columns)
    ranked: list[tuple[int, str]] = []
    for col in cols:
        c = col.lower()
        if not any(tok in c for tok in channel_tokens):
            continue
        score = 0
        for i, suff in enumerate(priority_suffixes):
            if c.endswith(suff):
                score = 100 - i
                break
        ranked.append((score, col))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def _top_modes_gmm(values: np.ndarray) -> np.ndarray:
    from sklearn.mixture import GaussianMixture

    vals = values[np.isfinite(values)]
    vals = vals[vals > 0]
    if len(vals) < 16:
        raise ValueError("Not enough positive values for beads calibration")

    log_vals = np.log2(vals).reshape(-1, 1)
    gmm = GaussianMixture(n_components=BEADS_COMPONENT_COUNT, random_state=0, covariance_type="full")
    gmm.fit(log_vals)

    means = gmm.means_.reshape(-1)
    weights = gmm.weights_.reshape(-1)
    order = np.argsort(weights)[::-1][:BEADS_COMPONENT_COUNT]
    selected = np.sort(means[order])
    return np.power(2.0, selected)


def _fit_log_calibration(detected_peaks: np.ndarray, expected_intensities: np.ndarray) -> dict:
    x = np.log2(detected_peaks)
    y = np.log2(expected_intensities)
    coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    y_hat = slope * x + intercept
    sse = float(np.sum((y - y_hat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2}



BEADS_CHANNEL_CONFIG: dict[str, dict] = {
    "yellow": {
        "expected_intensities": np.array([125000,	4500000,	31200000], dtype=float),
        "fluorophore_unit": "Nile Red (ERF) in 574/26 band",
    },
    "orange": {
        "expected_intensities": np.array([40900,	1520000,	11700000], dtype=float),
        "fluorophore_unit": "Nile Red (ERF) in 590/40 band",
    },
    "red": {
        "expected_intensities": np.array([13700,	466000, 	5910000], dtype=float),
        "fluorophore_unit": "Nile Red (ERF) in 695/40 band",
    },
}



BEADS_COMPONENT_COUNT: int = min(
    len(cfg["expected_intensities"]) for cfg in BEADS_CHANNEL_CONFIG.values()
)


def compute_beads_calibration_from_df(
    df: pd.DataFrame,
    *,
    diagnostic_png_path: str | Path | None = None,
    verbose: bool = True,
    spoof_bead_sample: bool = True,
) -> dict | None:
    """
    Verbose bead-ladder calibration function.

    Peak detection is still rank-matched: the configured bead populations are
    sorted dimmest-to-brightest and paired with the expected bead ladder values:

        [125000, 4500000, 31200000]

    The fitted calibration is now absolute in the configured channel unit:

        orange -> MEPE
        red    -> MEPTR
        yellow -> MEAPC

    The model fitted for each channel is:

        log10(expected_fluorophore_value)
            = slope * log10(detected_peak_intensity) + intercept

    Fixed columns used:
        orange -> Fl_Orange_total
        red    -> Fl_Red_total
        yellow -> Fl_Yellow_total

    If spoof_bead_sample=True, the function replaces the fluorescence total
    columns in a local copy of df with a synthetic bead sample containing:
        - the configured number of fluorescent bead populations
        - log-spaced peak centres
        - slight overlap between neighbouring populations
        - broad low-level background/noise events

    This is for debugging peak detection and diagnostic plotting only.
    It does not modify the caller's original dataframe.

    Returns None if:
        - dataframe is missing;
        - dataframe is empty;
        - any required _total column is missing;
        - any required column has too few positive finite values;
        - peak detection fails;
        - the fit fails.
    """

    def dbg(msg: str) -> None:
        if verbose:
            print(f"[beads calibration DEBUG] {msg}")

    dbg("Starting compute_beads_calibration_from_df()")

    if df is None:
        dbg("FAILED: df is None")
        return None

    if df.empty:
        dbg("FAILED: df is empty")
        return None

    dbg(f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
    dbg("DataFrame columns:")
    for i, col in enumerate(df.columns):
        dbg(f"  [{i}] {repr(col)}")

    def _get_required_column_case_insensitive(
        dataframe: pd.DataFrame,
        required_column: str,
    ) -> str | None:
        """
        Return the actual dataframe column matching required_column,
        case-insensitively.

        Deliberately no fallback. No _maximum. No _average. No fuzzy tokens.
        """
        def _normalise(column_name: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(column_name).lower())

        required_lower = _normalise(required_column)
        matches = [
            col for col in dataframe.columns
            if _normalise(col) == required_lower
        ]

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            dbg(
                f"WARNING: multiple case-insensitive matches for {required_column}: "
                f"{matches}. Using first match: {matches[0]}"
            )
            return matches[0]

        return None

    channels = {
        "orange": {
            "required_column": "Fl_Orange_total",
            "instrument_channel": "Fl_Orange_total",
        },
        "red": {
            "required_column": "Fl_Red_total",
            "instrument_channel": "Fl_Red_total",
        },
        "yellow": {
            "required_column": "Fl_Yellow_total",
            "instrument_channel": "Fl_Yellow_total",
        },
    }

    if spoof_bead_sample:
        dbg("")
        dbg("DEBUG MODE ENABLED: spoof_bead_sample=True")
        dbg("Replacing fluorescence total columns with synthetic bead-ladder data")
        dbg("Original dataframe will not be modified; operating on a local copy")

        rng = np.random.default_rng(42)
        df = df.copy()

        def _make_synthetic_bead_ladder(
            n_events: int,
            *,
            channel: str,
        ) -> np.ndarray:
            """
            Create synthetic fluorescence totals resembling a bead acquisition.

            The output is a 1D array with:
                - exactly the configured number of bead populations from
                  BEADS_CHANNEL_CONFIG;
                - slight overlap;
                - background/noise events;
                - occasional dim and bright outliers.

            Values are in arbitrary instrument-intensity units.
            """
            if n_events <= 0:
                return np.array([], dtype=float)

            channel_scale = {
                "orange": 1.00,
                "red": 1.35,
                "yellow": 0.75,
            }[channel]

            expected_peaks = np.asarray(
                BEADS_CHANNEL_CONFIG[channel]["expected_intensities"],
                dtype=float,
            )
            if expected_peaks.size != BEADS_COMPONENT_COUNT:
                raise ValueError(
                    f"{channel}: expected bead ladder size {expected_peaks.size} does not "
                    f"match BEADS_COMPONENT_COUNT={BEADS_COMPONENT_COUNT}"
                )

            peak_centres = channel_scale * expected_peaks
            peak_centres = np.maximum(peak_centres, 1.0)

            n_background = max(1, int(n_events * 0.18))
            n_outliers = max(1, int(n_events * 0.02))
            n_peak_events_total = max(1, n_events - n_background - n_outliers)

            peak_weights = np.full(BEADS_COMPONENT_COUNT, 1.0 / BEADS_COMPONENT_COUNT, dtype=float)
            peak_counts = rng.multinomial(n_peak_events_total, peak_weights)

            values: list[np.ndarray] = []

            background = 10.0 ** rng.uniform(
                np.log10(channel_scale * 30.0),
                np.log10(channel_scale * 1.8e4),
                size=n_background,
            )
            values.append(background)

            n_dim = n_outliers // 2
            n_bright = n_outliers - n_dim

            if n_dim > 0:
                dim_noise = 10.0 ** rng.uniform(
                    np.log10(channel_scale * 5.0),
                    np.log10(channel_scale * 40.0),
                    size=n_dim,
                )
                values.append(dim_noise)

            if n_bright > 0:
                bright_noise = 10.0 ** rng.uniform(
                    np.log10(channel_scale * 1.5e4),
                    np.log10(channel_scale * 4.0e4),
                    size=n_bright,
                )
                values.append(bright_noise)

            for centre, count in zip(peak_centres, peak_counts):
                if count <= 0:
                    continue
                peak = rng.lognormal(
                    mean=np.log(centre),
                    sigma=0.17,
                    size=count,
                )
                values.append(peak)

            synthetic = np.concatenate(values).astype(float)

            if len(synthetic) > n_events:
                synthetic = rng.choice(synthetic, size=n_events, replace=False)
            elif len(synthetic) < n_events:
                extra = rng.choice(
                    synthetic,
                    size=n_events - len(synthetic),
                    replace=True,
                )
                synthetic = np.concatenate([synthetic, extra])

            rng.shuffle(synthetic)
            return synthetic

        for channel, cfg in channels.items():
            required_column = cfg["required_column"]
            source_column = _get_required_column_case_insensitive(
                df,
                required_column,
            )

            if source_column is None:
                dbg(
                    f"FAILED: spoof_bead_sample=True but required column "
                    f"{required_column!r} was not found."
                )
                dbg("Available dataframe columns were:")
                for i, col in enumerate(df.columns):
                    dbg(f"  [{i}] {repr(col)}")
                return None

            synthetic_vals = _make_synthetic_bead_ladder(
                n_events=len(df),
                channel=channel,
            )
            df[source_column] = synthetic_vals

            dbg(
                f"{channel}: replaced {source_column!r} with synthetic bead sample; "
                f"n={len(synthetic_vals)}, "
                f"min={float(np.min(synthetic_vals)):.6g}, "
                f"median={float(np.median(synthetic_vals)):.6g}, "
                f"max={float(np.max(synthetic_vals)):.6g}"
            )

        dbg("Synthetic bead-ladder spoofing complete")
        dbg("")

    bead_population_ranks = np.arange(1, BEADS_COMPONENT_COUNT + 1, dtype=float)
    dbg(f"BEADS_COMPONENT_COUNT: {BEADS_COMPONENT_COUNT}")
    dbg(f"Bead population ranks used only for order-matching: {bead_population_ranks.tolist()}")

    channel_results: dict[str, dict[str, Any]] = {}

    def _detect_bead_peaks_log10(values: np.ndarray, channel: str) -> np.ndarray:
        """
        Detect bead population centres using a smoothed log10 histogram.

        This is deliberately mode-based rather than GMM-based.
        Returns detected peak positions back in original instrument units.
        """
        vals = values[np.isfinite(values)]
        vals = vals[vals > 0]

        dbg(
            f"{channel}: positive finite values entering histogram peak detector: "
            f"{len(vals)}"
        )

        if len(vals) < 16:
            raise ValueError(
                f"{channel}: not enough positive values for bead population detection "
                f"({len(vals)} found, need at least 16)"
            )

        log_vals = np.log10(vals)
        log_min = float(np.min(log_vals))
        log_max = float(np.max(log_vals))

        dbg(
            f"{channel}: log10 value range entering histogram peak detector: "
            f"{log_min:.6g} to {log_max:.6g}"
        )

        if not np.isfinite(log_min) or not np.isfinite(log_max) or log_max <= log_min:
            raise ValueError(f"{channel}: invalid log10 value range for peak detection")

        n_bins = int(np.clip(np.sqrt(len(log_vals)) * 3, 96, 384))
        hist, edges = np.histogram(
            log_vals,
            bins=n_bins,
            range=(log_min, log_max),
        )
        centres = 0.5 * (edges[:-1] + edges[1:])

        dbg(f"{channel}: histogram bins used: {n_bins}")
        dbg(f"{channel}: raw histogram max count: {int(np.max(hist))}")

        sigma_bins = 2.0
        kernel_radius = int(max(3, round(sigma_bins * 4)))
        kx = np.arange(-kernel_radius, kernel_radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (kx / sigma_bins) ** 2)
        kernel = kernel / np.sum(kernel)
        smooth = np.convolve(hist.astype(float), kernel, mode="same")

        dbg(f"{channel}: smoothed histogram max count: {float(np.max(smooth)):.6g}")

        if np.max(smooth) <= 0:
            raise ValueError(f"{channel}: smoothed histogram is empty")

        candidate_idx = np.where(
            (smooth[1:-1] > smooth[:-2])
            & (smooth[1:-1] >= smooth[2:])
        )[0] + 1

        dbg(f"{channel}: initial local maxima candidates: {len(candidate_idx)}")

        if len(candidate_idx) == 0:
            raise ValueError(f"{channel}: no local maxima found in smoothed histogram")

        min_height = 0.02 * float(np.max(smooth))
        candidate_idx = np.array(
            [idx for idx in candidate_idx if smooth[idx] >= min_height],
            dtype=int,
        )

        dbg(
            f"{channel}: candidates after height threshold "
            f"({min_height:.6g}): {len(candidate_idx)}"
        )

        if len(candidate_idx) == 0:
            raise ValueError(f"{channel}: no peaks survived height threshold")

        # Rank by smoothed histogram height, strongest first.
        candidate_idx = candidate_idx[np.argsort(smooth[candidate_idx])[::-1]]

        # Prevent selecting small shoulder peaks too close to already-selected modes.
        min_distance_log10 = 0.11
        selected: list[int] = []

        for idx in candidate_idx:
            c = centres[idx]
            too_close = any(
                abs(c - centres[chosen]) < min_distance_log10
                for chosen in selected
            )
            if too_close:
                continue
            selected.append(int(idx))
            if len(selected) == BEADS_COMPONENT_COUNT:
                break

        dbg(f"{channel}: selected peaks after spacing filter: {len(selected)}")

        if len(selected) < BEADS_COMPONENT_COUNT:
            dbg(
                f"{channel}: fewer than {BEADS_COMPONENT_COUNT} peaks found; "
                "relaxing spacing threshold"
            )

            selected = []
            min_distance_log10_relaxed = 0.06

            for idx in candidate_idx:
                c = centres[idx]
                too_close = any(
                    abs(c - centres[chosen]) < min_distance_log10_relaxed
                    for chosen in selected
                )
                if too_close:
                    continue
                selected.append(int(idx))
                if len(selected) == BEADS_COMPONENT_COUNT:
                    break

            dbg(f"{channel}: selected peaks after relaxed spacing filter: {len(selected)}")

        if len(selected) < BEADS_COMPONENT_COUNT:
            dbg(
                f"{channel}: WARNING: peak spacing filters found only {len(selected)} "
                f"peaks; falling back to strongest {BEADS_COMPONENT_COUNT} local maxima"
            )
            selected = [int(idx) for idx in candidate_idx[:BEADS_COMPONENT_COUNT]]

        if len(selected) < BEADS_COMPONENT_COUNT:
            raise ValueError(
                f"{channel}: could not identify {BEADS_COMPONENT_COUNT} histogram modes; "
                f"found {len(selected)}"
            )

        selected = np.array(selected[:BEADS_COMPONENT_COUNT], dtype=int)

        # Sort dimmest to brightest. This is the rank-matching step.
        selected = selected[np.argsort(centres[selected])]
        selected_log10 = centres[selected]
        detected_peaks = 10.0 ** selected_log10

        dbg(
            f"{channel}: selected histogram peak centres log10: "
            f"{[float(x) for x in selected_log10]}"
        )
        dbg(
            f"{channel}: detected peak positions in instrument units: "
            f"{[float(x) for x in detected_peaks]}"
        )
        dbg(
            f"{channel}: selected peak smoothed counts: "
            f"{[float(smooth[i]) for i in selected]}"
        )

        return detected_peaks

    for channel, cfg in channels.items():
        dbg("")
        dbg(f"Processing channel: {channel}")

        required_column = cfg["required_column"]
        dbg(f"{channel}: required fixed column: {required_column}")

        source_column = _get_required_column_case_insensitive(df, required_column)

        if source_column is None:
            dbg(f"FAILED: required column {required_column!r} was not found.")
            dbg("Available dataframe columns were:")
            for i, col in enumerate(df.columns):
                dbg(f"  [{i}] {repr(col)}")
            return None

        dbg(f"{channel}: found source column: {source_column!r}")

        expected_fluorophore_values = np.asarray(
            BEADS_CHANNEL_CONFIG[channel]["expected_intensities"],
            dtype=float,
        )
        fluorophore_unit = str(BEADS_CHANNEL_CONFIG[channel]["fluorophore_unit"])

        if len(expected_fluorophore_values) != BEADS_COMPONENT_COUNT:
            dbg(
                f"FAILED: {channel} expected_intensities has "
                f"{len(expected_fluorophore_values)} values; expected {BEADS_COMPONENT_COUNT}"
            )
            return None

        dbg(
            f"{channel}: expected fluorophore ladder values ({fluorophore_unit}): "
            f"{[float(x) for x in expected_fluorophore_values]}"
        )

        raw_series = pd.to_numeric(df[source_column], errors="coerce")
        vals = raw_series.to_numpy(dtype=float)

        vals_finite = vals[np.isfinite(vals)]
        vals_positive = vals_finite[vals_finite > 0]

        n_total = int(len(vals))
        n_finite = int(len(vals_finite))
        n_positive = int(len(vals_positive))
        n_nan_or_non_numeric = int(n_total - n_finite)
        n_zero_or_negative = int(n_finite - n_positive)

        dbg(f"{channel}: total raw values: {n_total}")
        dbg(f"{channel}: finite numeric values: {n_finite}")
        dbg(f"{channel}: NaN/non-numeric values after conversion: {n_nan_or_non_numeric}")
        dbg(f"{channel}: zero or negative finite values removed: {n_zero_or_negative}")
        dbg(f"{channel}: positive finite values used: {n_positive}")

        if n_positive > 0:
            dbg(f"{channel}: min positive value: {float(np.min(vals_positive)):.6g}")
            dbg(f"{channel}: median positive value: {float(np.median(vals_positive)):.6g}")
            dbg(f"{channel}: max positive value: {float(np.max(vals_positive)):.6g}")

        if n_positive < 16:
            dbg(
                f"FAILED: {channel} / {source_column!r} has too few positive finite "
                f"values for histogram peak detection. Found {n_positive}, need at least 16."
            )
            return None

        try:
            detected_peaks = _detect_bead_peaks_log10(vals_positive, channel)
        except Exception as e:
            dbg(
                f"FAILED: histogram peak detection failed for {channel}: "
                f"{type(e).__name__}: {e}"
            )
            return None

        detected_peaks = np.asarray(detected_peaks, dtype=float)
        dbg(f"{channel}: number of detected peaks: {len(detected_peaks)}")

        if len(detected_peaks) != BEADS_COMPONENT_COUNT:
            dbg(
                f"FAILED: expected {BEADS_COMPONENT_COUNT} detected peaks for {channel}, "
                f"got {len(detected_peaks)}"
            )
            return None

        # IMPORTANT:
        # The rank is used only to pair dimmest-to-brightest detected peaks with
        # dimmest-to-brightest expected fluorophore values. The fitted y-axis is
        # the absolute configured fluorophore ladder, not the rank.
        valid_pair_mask = (
            np.isfinite(detected_peaks)
            & np.isfinite(expected_fluorophore_values)
            & (detected_peaks > 0)
            & (expected_fluorophore_values > 0)
        )

        n_valid_pairs = int(np.count_nonzero(valid_pair_mask))
        dbg(f"{channel}: valid detected-peak/fluorophore-value pairs: {n_valid_pairs}")

        if n_valid_pairs < 2:
            dbg(f"FAILED: fewer than 2 valid peak/fluorophore pairs for {channel}")
            return None

        detected_used = detected_peaks[valid_pair_mask]
        ranks_used = bead_population_ranks[valid_pair_mask]
        fluorophore_used = expected_fluorophore_values[valid_pair_mask]

        x_log10 = np.log10(detected_used)
        y_log10 = np.log10(fluorophore_used)

        dbg(f"{channel}: detected peaks used: {[float(x) for x in detected_used]}")
        dbg(f"{channel}: bead ranks used for matching only: {[float(y) for y in ranks_used]}")
        dbg(
            f"{channel}: expected fluorophore values used ({fluorophore_unit}): "
            f"{[float(y) for y in fluorophore_used]}"
        )
        dbg(f"{channel}: log10 detected peaks: {[float(x) for x in x_log10]}")
        dbg(f"{channel}: log10 expected fluorophore values: {[float(y) for y in y_log10]}")

        try:
            coeffs = np.polyfit(x_log10, y_log10, deg=1)
        except Exception as e:
            dbg(f"FAILED: polyfit failed for {channel}: {type(e).__name__}: {e}")
            return None

        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        fitted_log10_fluorophore = slope * x_log10 + intercept
        residuals_log10 = y_log10 - fitted_log10_fluorophore

        sse = float(np.sum(residuals_log10 ** 2))
        sst = float(np.sum((y_log10 - np.mean(y_log10)) ** 2))
        r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
        rmse_log10 = float(np.sqrt(np.mean(residuals_log10 ** 2)))

        dbg(f"{channel}: fit slope: {slope:.8g}")
        dbg(f"{channel}: fit intercept: {intercept:.8g}")
        dbg(f"{channel}: fit R2: {r2:.8g}")
        dbg(f"{channel}: fit RMSE log10: {rmse_log10:.8g}")
        dbg(f"{channel}: residuals log10: {[float(r) for r in residuals_log10]}")

        bead_population_pairs = []

        for i, (
            detected,
            rank,
            expected_fluor,
            x10,
            y10,
            fitted_y10,
            resid10,
        ) in enumerate(
            zip(
                detected_used,
                ranks_used,
                fluorophore_used,
                x_log10,
                y_log10,
                fitted_log10_fluorophore,
                residuals_log10,
            ),
            start=1,
        ):
            bead_population_pairs.append(
                {
                    "bead_population_index": int(i),
                    "detected_peak_intensity": float(detected),
                    "bead_population_rank": float(rank),
                    "expected_fluorophore_value": float(expected_fluor),
                    "fluorophore_unit": fluorophore_unit,
                    "log10_detected_peak_intensity": float(x10),
                    "log10_expected_fluorophore_value": float(y10),
                    "log10_fitted_fluorophore_value": float(fitted_y10),
                    "log10_residual": float(resid10),
                    "fitted_fluorophore_value": float(10.0 ** fitted_y10),
                }
            )

        channel_results[channel] = {
            "analysis_type": "absolute_fluorophore_calibration_after_rank_matching",
            "peak_detection_method": "smoothed_log10_histogram_local_maxima",
            "rank_matching_method": (
                "Detected bead peaks are sorted dimmest-to-brightest and paired "
                "with expected_fluorophore_values in the same order."
            ),
            "instrument_channel": cfg["instrument_channel"],
            "source_column": source_column,
            "required_column": required_column,
            "fluorophore_unit": fluorophore_unit,
            "expected_intensity_positions": [float(x) for x in expected_fluorophore_values],
            "expected_fluorophore_values": [float(x) for x in expected_fluorophore_values],
            "detected_peak_positions": [float(x) for x in detected_peaks],
            "bead_population_ranks": [float(x) for x in bead_population_ranks],
            "bead_population_pairs_used": bead_population_pairs,
            "n_raw_values_total": n_total,
            "n_raw_values_finite": n_finite,
            "n_raw_values_nan_or_non_numeric": n_nan_or_non_numeric,
            "n_raw_values_zero_or_negative": n_zero_or_negative,
            "n_raw_values_positive_used_for_peak_detection": n_positive,
            "raw_positive_value_min": float(np.min(vals_positive)),
            "raw_positive_value_max": float(np.max(vals_positive)),
            "raw_positive_value_median": float(np.median(vals_positive)),
            "calibration_coefficients": {
                "model": (
                    f"log10({fluorophore_unit}) = "
                    "slope * log10(detected_peak_intensity) + intercept"
                ),
                "log_base": 10,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "rmse_log10": rmse_log10,
            },
            "calibration_curve_coefficients": {
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
            },
            # Backwards-compatible alias, in case downstream code still expects
            # this key from the previous QC version.
            "ladder_alignment_coefficients": {
                "model": (
                    f"log10({fluorophore_unit}) = "
                    "slope * log10(detected_peak_intensity) + intercept"
                ),
                "log_base": 10,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "rmse_log10": rmse_log10,
            },
        }

    qc_pass = all(
        math.isfinite(v["calibration_coefficients"]["r2"])
        and v["calibration_coefficients"]["r2"] >= 0.90
        for v in channel_results.values()
    )

    dbg("")
    dbg(f"Overall QC pass: {qc_pass}")

    if diagnostic_png_path:
        dbg(f"Creating diagnostic plot at: {diagnostic_png_path}")

        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator, NullFormatter

        outp = Path(diagnostic_png_path)
        outp.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 8),
            constrained_layout=True,
        )

        rng = np.random.default_rng(0)

        for ax, channel in zip(axes, channels.keys()):
            rec = channel_results[channel]
            source_column = rec["source_column"]
            co = rec["calibration_coefficients"]
            fluorophore_unit = rec["fluorophore_unit"]
            expected_values = np.asarray(rec["expected_fluorophore_values"], dtype=float)

            raw_vals = pd.to_numeric(df[source_column], errors="coerce").to_numpy(dtype=float)
            raw_vals = raw_vals[np.isfinite(raw_vals)]
            raw_vals = raw_vals[raw_vals > 0]

            pairs = rec["bead_population_pairs_used"]

            detected = np.array(
                [p["detected_peak_intensity"] for p in pairs],
                dtype=float,
            )
            fluor_values = np.array(
                [p["expected_fluorophore_value"] for p in pairs],
                dtype=float,
            )
            fitted_fluor_values = np.array(
                [p["fitted_fluorophore_value"] for p in pairs],
                dtype=float,
            )

            x_min = min(np.min(raw_vals), np.min(detected)) if len(raw_vals) else np.min(detected)
            x_max = max(np.max(raw_vals), np.max(detected)) if len(raw_vals) else np.max(detected)
            x_lower = 10.0 ** (np.log10(x_min) - 0.08)
            x_upper = 10.0 ** (np.log10(x_max) + 0.08)

            y_min = 10.0 ** (np.log10(np.min(expected_values)) - 0.10)
            y_max = 10.0 ** (np.log10(np.max(expected_values)) + 0.10)

            if len(raw_vals) > 0:
                raw_y_jitter = 10.0 ** rng.uniform(
                    np.log10(y_min),
                    np.log10(y_max),
                    size=len(raw_vals),
                )
                ax.scatter(
                    raw_vals,
                    raw_y_jitter,
                    color="0.45",
                    alpha=0.10,
                    s=7,
                    linewidths=0,
                    zorder=1,
                    label="Raw positive events, jittered on y",
                )

            ax.scatter(
                detected,
                fluor_values,
                color="tab:blue",
                edgecolor="white",
                linewidth=0.5,
                s=60,
                zorder=4,
                label=f"Rank-matched bead modes / expected {fluorophore_unit}",
            )

            for i, (xv, yv) in enumerate(zip(detected, fluor_values), start=1):
                ax.annotate(
                    f"{i}: {yv:g}",
                    xy=(xv, yv),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="tab:blue",
                    zorder=6,
                )

            x_line = np.logspace(
                np.log10(np.min(detected)) - 0.03,
                np.log10(np.max(detected)) + 0.03,
                200,
                base=10.0,
            )
            y_line = 10.0 ** (
                co["slope"] * np.log10(x_line) + co["intercept"]
            )

            ax.plot(
                x_line,
                y_line,
                color="tab:red",
                linewidth=2,
                zorder=3,
                label=f"Log10 calibration fit to {fluorophore_unit}",
            )

            ax.scatter(
                detected,
                fitted_fluor_values,
                color="tab:red",
                marker="x",
                s=60,
                zorder=5,
                label=f"Fitted {fluorophore_unit} at detected peaks",
            )

            ax.set_xscale("log", base=10)
            ax.set_yscale("log", base=10)
            ax.set_xlim(x_lower, x_upper)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(expected_values)
            ax.set_yticklabels([f"{v:g}" for v in expected_values])
            ax.xaxis.set_major_locator(LogLocator(base=10.0))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())

            ax.set_xlabel(f"Detected instrument intensity: {source_column}")
            ax.set_ylabel(f"Expected fluorescence equivalent ({fluorophore_unit})")

            spoof_note = " synthetic spoofed data" if spoof_bead_sample else ""
            ax.set_title(
                f"{channel} bead calibration{spoof_note}\n"
                f"{source_column}; R2={co['r2']:.3f}"
            )

            ax.grid(True, which="both", alpha=0.25)

            table_rows = [
                [
                    str(p["bead_population_index"]),
                    f"{p['detected_peak_intensity']:.3g}",
                    f"{p['expected_fluorophore_value']:.3g}",
                    f"{p['log10_residual']:.3g}",
                ]
                for p in pairs
            ]

            table = ax.table(
                cellText=table_rows,
                colLabels=[
                    "Peak",
                    "Detected",
                    fluorophore_unit,
                    "log10 resid.",
                ],
                loc="bottom",
                cellLoc="center",
                bbox=[0.0, -0.62, 1.0, 0.42],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)

            if len(raw_vals) > 0:
                # Marginal histogram below the table.
                hist_ax = ax.inset_axes([0.0, -0.98, 1.0, 0.22])
                hist_bins = np.logspace(
                    np.log10(np.min(raw_vals)),
                    np.log10(np.max(raw_vals)),
                    90,
                    base=10.0,
                )
                hist_ax.hist(
                    raw_vals,
                    bins=hist_bins,
                    color="0.25",
                    alpha=0.45,
                    edgecolor="none",
                )
                for xv in detected:
                    hist_ax.axvline(
                        xv,
                        color="tab:blue",
                        alpha=0.80,
                        linewidth=1.2,
                    )
                hist_ax.set_xscale("log", base=10)
                hist_ax.set_xlim(x_lower, x_upper)
                hist_ax.set_yticks([])
                hist_ax.tick_params(axis="x", labelsize=7)
                hist_ax.set_xlabel(
                    f"Histogram of {source_column}",
                    fontsize=8,
                )
                hist_ax.set_ylabel(
                    "count",
                    fontsize=7,
                    rotation=0,
                    labelpad=14,
                    va="center",
                )
                hist_ax.grid(True, which="both", axis="x", alpha=0.15)
                for spine in ("top", "right", "left"):
                    hist_ax.spines[spine].set_visible(False)

            ax.legend(fontsize=8, loc="best")

        fig.suptitle(
            (
                "Bead calibration: raw events, histogram-detected peaks, rank matching, "
                "and absolute fluorescence-equivalent fit\n"
                "Detected peaks are sorted by brightness, then rank-match assigned to "
                "yellow → [125000, 4500000, 31200000]"
                "orange → [40900, 1520000, 11700000]"
                "red → [13700, 466000, 5910000]"
                + ("; fluorescence totals have been synthetically spoofed" if spoof_bead_sample else "")
            ),
            fontsize=14,
        )

        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)

        dbg(f"Diagnostic plot saved to: {outp}")

    result = {
        "fit_model": "beads_absolute_fluorophore_calibration_after_rank_matching",
        "analysis_type": "absolute_fluorophore_calibration_after_rank_matching",
        "peak_detection_method": "smoothed_log10_histogram_local_maxima",
        "rank_matching_method": (
                "Detected peaks are sorted by brightness, then rank-match assigned to "
                "yellow → [125000, 4500000, 31200000]"
                "orange → [40900, 1520000, 11700000]"
                "red → [13700, 466000, 5910000]"
                ),
        "spoof_bead_sample": bool(spoof_bead_sample),
        "required_columns": {
            "orange": "Fl_Orange_total",
            "red": "Fl_Red_total",
            "yellow": "Fl_Yellow_total",
        },
        "channels": channel_results,
        "qc_pass": bool(qc_pass),
    }

    dbg("Finished compute_beads_calibration_from_df() successfully")
    return result

def append_beadscalibration_record(record: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record=json_safe(record)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def beads_calibration_store_path(beads_output_root: str | Path | None = None) -> Path:
    """
    Stable runtime location for bead calibrations produced by the continuous analyser.
    """
    runtime_cfg = get_runtime_config()
    store_path = runtime_cfg.paths.beads_calibration_store_path
    if beads_output_root is None:
        return store_path
    root = Path(beads_output_root)
    try:
        relative_store = store_path.relative_to(runtime_cfg.paths.tool_dir)
    except ValueError:
        relative_store = Path("BeadsCalibrations") / store_path.name
    return root / relative_store


def load_latest_beadscalibration_record(beads_output_root: str | Path | None = None) -> dict | None:
    """
    Return the newest usable bead calibration record from BeadsCalibrations.jsonl.
    Corrupt/blank lines are ignored so one bad append does not break the watcher.
    """
    path = beads_calibration_store_path(beads_output_root)
    if not path.exists():
        return None
    latest: dict | None = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("protocol") != "beadsprotocol":
                continue
            if not isinstance(rec.get("channels"), dict):
                continue
            latest = rec
    return latest


def latest_beadscalibration_summary(beads_output_root: str | Path | None = None) -> dict:
    """
    Small metadata object for downstream packets/logging: enough to say which
    file produced the current runtime calibration and when it became available.
    """
    path = beads_calibration_store_path(beads_output_root)
    rec = load_latest_beadscalibration_record(beads_output_root)
    if rec is None:
        return {
            "latest_calibration_available": False,
            "latest_calibration_record_path": str(path),
        }
    return {
        "latest_calibration_available": True,
        "latest_calibration_file_id": rec.get("file_id"),
        "latest_calibration_time_calculated": rec.get("time_calculated"),
        "latest_calibration_record_path": str(path),
        "latest_calibration_diagnostic_plot": rec.get("diagnostic_plot"),
        "latest_calibration_qc": "pass" if rec.get("qc_pass") else "fail",
    }


def apply_latest_beads_calibration_to_dataframe(
    df: pd.DataFrame,
    *,
    beads_output_root: str | Path | None = None,
    required_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the latest continuous-runtime bead calibration to a dataframe.

    The function is deliberately non-fatal: if no runtime calibration exists, or
    if the dataframe has no matching fluorescence columns, the original dataframe
    is returned with metadata explaining what happened.
    """
    meta = latest_beadscalibration_summary(beads_output_root)
    if df is None or df.empty:
        meta["runtime_calibration_applied"] = False
        meta["runtime_calibration_status"] = "empty_dataframe"
        return df, meta

    rec = load_latest_beadscalibration_record(beads_output_root)
    if rec is None:
        meta["runtime_calibration_applied"] = False
        meta["runtime_calibration_status"] = "no_latest_calibration"
        return df, meta

    channels = rec.get("channels") or {}
    coeffs: dict[str, tuple[float, float]] = {}
    for channel, crec in channels.items():
        if not isinstance(crec, dict):
            continue
        c = crec.get("calibration_curve_coefficients") or crec.get("calibration_coefficients") or {}
        try:
            slope = float(c.get("slope"))
            intercept = float(c.get("intercept"))
        except Exception:
            continue
        if np.isfinite(slope) and np.isfinite(intercept):
            coeffs[str(channel).lower()] = (slope, intercept)

    if not coeffs:
        meta["runtime_calibration_applied"] = False
        meta["runtime_calibration_status"] = "no_finite_coefficients"
        return df, meta

    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    allowed = set(required_columns) if required_columns is not None else None
    calibrated_columns: list[str] = []

    for col in list(out.columns):
        if allowed is not None and col not in allowed:
            continue
        if not _is_runtime_calibratable_fluorescence_column(col):
            continue
        channel = _infer_runtime_beads_channel_from_column(col)
        if channel not in coeffs:
            continue
        slope, intercept = coeffs[channel]
        out[col] = _apply_runtime_log10_calibration(out[col], slope, intercept)
        calibrated_columns.append(col)

    meta["runtime_calibration_applied"] = bool(calibrated_columns)
    meta["runtime_calibration_status"] = "success" if calibrated_columns else "no_matching_columns"
    meta["runtime_calibrated_columns"] = calibrated_columns
    return out, meta


# Channels for which apply_saved_bead_calibration_to_dataframe creates
# calibrated predictor columns and retains raw copies.
_CALIBRATION_CHANNELS = ("orange", "yellow", "red")

# Column names that are calibrated; raw copies will be stored as <col>_raw.
_CALIBRATED_TOTAL_COLUMNS = (
    "Fl_Orange_total",
    "Fl_Yellow_total",
    "Fl_Red_total",
)


def apply_saved_bead_calibration_to_dataframe(
    df: pd.DataFrame,
    record: dict,
) -> pd.DataFrame:
    """
    Apply a specific saved bead calibration record to a listmode dataframe using
    the same log10 calibration transform used at training time.

    For each of the relevant orange / yellow / red fluorescence total columns:
      - Raw values are preserved in a ``<column>_raw`` copy (e.g.
        ``Fl_Orange_total_raw``).
      - The column itself is replaced with the log10-calibrated values:
        ``log10(calibrated) = slope * log10(raw) + intercept``

    Any column whose channel is absent from the calibration record (i.e. no
    finite slope/intercept) is left unchanged and no ``_raw`` copy is created.

    Returns a modified copy of the dataframe; the original is not modified.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    channels = record.get("channels") or {}
    coeffs: dict[str, tuple[float, float]] = {}
    for channel, crec in channels.items():
        if not isinstance(crec, dict):
            continue
        c = (
            crec.get("calibration_curve_coefficients")
            or crec.get("calibration_coefficients")
            or {}
        )
        try:
            slope = float(c.get("slope"))
            intercept = float(c.get("intercept"))
        except Exception:
            continue
        if np.isfinite(slope) and np.isfinite(intercept):
            coeffs[str(channel).lower()] = (slope, intercept)

    if not coeffs:
        return df.copy()

    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)

    for col in _CALIBRATED_TOTAL_COLUMNS:
        # Case-insensitive column lookup
        norm_col = _normalise_runtime_feature_name(col)
        matched = next(
            (c for c in out.columns if _normalise_runtime_feature_name(c) == norm_col),
            None,
        )
        if matched is None:
            continue

        channel = _infer_runtime_beads_channel_from_column(matched)
        if channel not in coeffs:
            continue

        slope, intercept = coeffs[channel]

        # Preserve raw values
        raw_col = f"{matched}_raw"
        out[raw_col] = out[matched].copy()

        # Apply log10 calibration transform in-place
        out[matched] = _apply_runtime_log10_calibration(out[matched], slope, intercept)

    return out


def run_protocol_postprocessing(
    *,
    protocol: str,
    packet: dict,
    dataframe: pd.DataFrame | None,
    file_id: str | None = None,
    diagnostic_dir: str | Path | None = None,
    beads_output_root: str | Path | None = None,
) -> dict:
    """
    Protocol-specific runtime post-processing for the continuous analyser.

    For beadsprotocol files this computes the real bead calibration before any
    classification step, appends it to BeadsCalibrations/BeadsCalibrations.jsonl,
    writes the diagnostic PNG into the continuous analyser plots folder, and
    returns metadata identifying the latest runtime calibration.
    """
    updates: dict[str, Any] = {}
    if protocol != "beadsprotocol":
        return updates

    diag_dir = Path(diagnostic_dir) if diagnostic_dir else Path(".")
    diag_dir.mkdir(parents=True, exist_ok=True)
    safe_file_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(file_id or "unknown_file"))
    diag_path = diag_dir / f"beads_calibration_diagnostic_{safe_file_id}.png"

    beads_cal = compute_beads_calibration_from_df(
        dataframe,
        diagnostic_png_path=diag_path,
        verbose=True,
        spoof_bead_sample=False,
    )
    if beads_cal is None:
        updates["bead_calibration_created"] = False
        updates["bead_calibration_qc_passed"] = False
        updates["bead_calibration_diagnostic_plot_path"] = str(diag_path)
        updates["beads_calibration_qc"] = "fail"
        updates["latest_beads_calibration"] = latest_beadscalibration_summary(beads_output_root)
        return updates

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if beads_output_root is None:
        beads_output_root = get_runtime_config().paths.tool_dir
    out_path = beads_calibration_store_path(beads_output_root)

    rec = {
        "time_calculated": ts,
        "file_id": file_id or "unknown_file",
        "protocol": "beadsprotocol",
        **beads_cal,
        "diagnostic_plot": str(diag_path),
        "runtime_context": "continuous_sample_analyser",
        "latest_calibration_available": True,
        "latest_calibration_file_id": file_id or "unknown_file",
        "latest_calibration_time_calculated": ts,
        "latest_calibration_record_path": str(out_path),
        "source_system_serial_no": packet.get("system_serial_no") or packet.get("serialNumber"),
        "source_start": packet.get("start"),
    }
    append_beadscalibration_record(rec, out_path)

    latest = latest_beadscalibration_summary(beads_output_root)
    updates["bead_calibration_created"] = True
    updates["bead_calibration_qc_passed"] = bool(beads_cal.get("qc_pass"))
    updates["bead_calibration_diagnostic_plot_path"] = str(diag_path)
    updates["beads_calibration"] = beads_cal
    updates["beads_calibration_qc"] = "pass" if beads_cal["qc_pass"] else "fail"
    updates["beads_calibration_plot"] = str(diag_path.name)
    updates["beads_calibration_record_path"] = str(out_path)
    updates["latest_beads_calibration"] = latest
    updates.update(latest)
    return updates
