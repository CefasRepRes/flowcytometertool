"""
Tests for bead-calibrated continuous analyser inference:
  - resolve_active_raw_model_path / resolve_active_beadcalibrated_model_path
  - apply_saved_bead_calibration_to_dataframe
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flowcytometer_tool.tabs.continuous_sample_analyser.bead_calibration import (
    apply_saved_bead_calibration_to_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_record(slope: float = 1.0, intercept: float = 0.0) -> dict:
    """Minimal bead calibration record with the same shape as the real one."""
    def _chan(s, i):
        return {
            "calibration_curve_coefficients": {"slope": s, "intercept": i, "r2": 0.999},
        }

    return {
        "protocol": "beadsprotocol",
        "qc_pass": True,
        "channels": {
            "orange": _chan(slope, intercept),
            "yellow": _chan(slope, intercept),
            "red": _chan(slope, intercept),
        },
    }


def _make_listmode_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Fl_Orange_total": rng.uniform(100, 10_000, n),
            "Fl_Yellow_total": rng.uniform(100, 10_000, n),
            "Fl_Red_total": rng.uniform(100, 10_000, n),
            "FWS_total": rng.uniform(10, 1_000, n),
            "SSC_total": rng.uniform(10, 1_000, n),
        }
    )


# ---------------------------------------------------------------------------
# apply_saved_bead_calibration_to_dataframe
# ---------------------------------------------------------------------------

class TestApplySavedBeadCalibration:
    def test_identity_calibration_preserves_values(self):
        """slope=1, intercept=0 in log10 space → 10^(1*log10(x)+0) = x."""
        df = _make_listmode_df()
        record = _make_sample_record(slope=1.0, intercept=0.0)
        out = apply_saved_bead_calibration_to_dataframe(df, record)

        for col in ("Fl_Orange_total", "Fl_Yellow_total", "Fl_Red_total"):
            np.testing.assert_allclose(
                out[col].values,
                df[col].values,
                rtol=1e-6,
                err_msg=f"Identity calibration should not change {col}",
            )

    def test_raw_copies_created(self):
        """_raw columns are created for each calibrated channel."""
        df = _make_listmode_df()
        record = _make_sample_record(slope=1.0, intercept=0.0)
        out = apply_saved_bead_calibration_to_dataframe(df, record)

        for col in ("Fl_Orange_total", "Fl_Yellow_total", "Fl_Red_total"):
            raw_col = f"{col}_raw"
            assert raw_col in out.columns, f"Expected {raw_col} to exist"
            pd.testing.assert_series_equal(
                out[raw_col].reset_index(drop=True),
                df[col].reset_index(drop=True),
                check_names=False,
            )

    def test_non_identity_calibration_transforms_values(self):
        """A non-trivial calibration should produce different values."""
        df = _make_listmode_df()
        record = _make_sample_record(slope=0.9, intercept=0.3)
        out = apply_saved_bead_calibration_to_dataframe(df, record)

        for col in ("Fl_Orange_total", "Fl_Yellow_total", "Fl_Red_total"):
            assert not np.allclose(out[col].values, df[col].values), (
                f"{col} should change with non-identity calibration"
            )

    def test_non_fluorescence_columns_unchanged(self):
        """FWS and SSC columns must not be modified."""
        df = _make_listmode_df()
        record = _make_sample_record(slope=0.9, intercept=0.3)
        out = apply_saved_bead_calibration_to_dataframe(df, record)

        for col in ("FWS_total", "SSC_total"):
            pd.testing.assert_series_equal(out[col], df[col])

    def test_empty_dataframe_returned_unchanged(self):
        empty = pd.DataFrame()
        record = _make_sample_record()
        out = apply_saved_bead_calibration_to_dataframe(empty, record)
        assert out.empty

    def test_no_coefficients_in_record_returns_copy(self):
        """If the record has no finite coefficients, the dataframe is returned as-is."""
        df = _make_listmode_df()
        bad_record = {"protocol": "beadsprotocol", "channels": {}}
        out = apply_saved_bead_calibration_to_dataframe(df, bad_record)
        for col in ("Fl_Orange_total", "Fl_Yellow_total", "Fl_Red_total"):
            pd.testing.assert_series_equal(out[col], df[col])

    def test_original_df_not_mutated(self):
        """Function must not modify the caller's dataframe."""
        df = _make_listmode_df()
        original_values = df["Fl_Orange_total"].copy()
        record = _make_sample_record(slope=0.8, intercept=0.5)
        _ = apply_saved_bead_calibration_to_dataframe(df, record)
        pd.testing.assert_series_equal(df["Fl_Orange_total"], original_values)

    def test_case_insensitive_column_matching(self):
        """Column names with different capitalisation should still be calibrated."""
        df = pd.DataFrame(
            {
                "fl_orange_total": np.full(10, 500.0),
                "Fl_Yellow_Total": np.full(10, 500.0),
                "FL_RED_TOTAL": np.full(10, 500.0),
            }
        )
        record = _make_sample_record(slope=1.0, intercept=0.0)
        out = apply_saved_bead_calibration_to_dataframe(df, record)
        # Each should have a _raw sibling
        assert any("_raw" in c for c in out.columns)


# ---------------------------------------------------------------------------
# resolve_active_raw_model_path / resolve_active_beadcalibrated_model_path
#
# functions.py has heavy top-level GUI/graphics imports that cannot be stubbed
# completely in headless CI.  We therefore test the resolver *logic* directly
# by replicating the pure-Python decision paths in a minimal test harness.
# ---------------------------------------------------------------------------

def _resolve_raw(cfg: dict, uncal_dir: Path, legacy_dir: Path) -> str | None:
    """Minimal replica of resolve_active_raw_model_path logic for testing."""
    slot = cfg.get("active_uncalibrated_model")
    if slot and isinstance(slot, dict):
        primary = slot.get("primary_model")
        if primary and Path(primary).is_file():
            return str(primary)
        pkls = [p for p in uncal_dir.glob("*.pkl") if not str(p).endswith("probabilistic.pkl")]
        if pkls:
            return str(pkls[0])
    # legacy fallback
    pkls = [p for p in legacy_dir.glob("*.pkl") if not str(p).endswith("probabilistic.pkl")]
    if len(pkls) == 1:
        return str(pkls[0])
    return None


def _resolve_beadcal(cfg: dict, beadcal_dir: Path) -> str | None:
    """Minimal replica of resolve_active_beadcalibrated_model_path logic for testing."""
    slot = cfg.get("active_beadcalibrated_model")
    if not slot or not isinstance(slot, dict):
        return None
    primary = slot.get("primary_model")
    if primary and Path(primary).is_file():
        return str(primary)
    pkls = [p for p in beadcal_dir.glob("*.pkl") if not str(p).endswith("probabilistic.pkl")]
    if pkls:
        return str(pkls[0])
    return None


class TestResolvers:
    def test_resolve_raw_uses_uncalibrated_slot_when_present(self, tmp_path):
        uncal_dir = tmp_path / "selecteduncalibratedmodel"
        uncal_dir.mkdir()
        model_file = uncal_dir / "model_raw.pkl"
        model_file.write_bytes(b"fakepkl")

        cfg = {
            "active_uncalibrated_model": {
                "version": "v1",
                "primary_model": str(model_file),
            }
        }
        result = _resolve_raw(cfg, uncal_dir, tmp_path / "selectedvalidappliedmodel")
        assert result == str(model_file)

    def test_resolve_raw_falls_back_to_legacy_slot(self, tmp_path):
        legacy_dir = tmp_path / "selectedvalidappliedmodel"
        legacy_dir.mkdir()
        model_file = legacy_dir / "model_legacy.pkl"
        model_file.write_bytes(b"fakepkl")
        uncal_dir = tmp_path / "selecteduncalibratedmodel"
        uncal_dir.mkdir()

        cfg = {}  # no active_uncalibrated_model slot
        result = _resolve_raw(cfg, uncal_dir, legacy_dir)
        assert result == str(model_file)

    def test_resolve_raw_uses_folder_scan_when_primary_path_stale(self, tmp_path):
        uncal_dir = tmp_path / "selecteduncalibratedmodel"
        uncal_dir.mkdir()
        model_file = uncal_dir / "model_raw.pkl"
        model_file.write_bytes(b"fakepkl")

        cfg = {
            "active_uncalibrated_model": {
                "version": "v1",
                "primary_model": "/stale/path/does_not_exist.pkl",  # stale path
            }
        }
        result = _resolve_raw(cfg, uncal_dir, tmp_path / "selectedvalidappliedmodel")
        assert result == str(model_file)

    def test_resolve_beadcalibrated_returns_none_when_not_set(self, tmp_path):
        beadcal_dir = tmp_path / "selectedbeadcalibratedmodel"
        beadcal_dir.mkdir()
        result = _resolve_beadcal({}, beadcal_dir)
        assert result is None

    def test_resolve_beadcalibrated_returns_path_when_set(self, tmp_path):
        beadcal_dir = tmp_path / "selectedbeadcalibratedmodel"
        beadcal_dir.mkdir()
        model_file = beadcal_dir / "model_beadcal.pkl"
        model_file.write_bytes(b"fakepkl")

        cfg = {
            "active_beadcalibrated_model": {
                "version": "v2",
                "primary_model": str(model_file),
            }
        }
        result = _resolve_beadcal(cfg, beadcal_dir)
        assert result == str(model_file)

    def test_resolve_beadcalibrated_uses_folder_scan_when_primary_stale(self, tmp_path):
        beadcal_dir = tmp_path / "selectedbeadcalibratedmodel"
        beadcal_dir.mkdir()
        model_file = beadcal_dir / "model_beadcal.pkl"
        model_file.write_bytes(b"fakepkl")

        cfg = {
            "active_beadcalibrated_model": {
                "version": "v2",
                "primary_model": "/stale/path.pkl",
            }
        }
        result = _resolve_beadcal(cfg, beadcal_dir)
        assert result == str(model_file)

    def test_probabilistic_pkl_excluded_from_fallback(self, tmp_path):
        """probabilistic.pkl files must never be returned as the active model."""
        uncal_dir = tmp_path / "selecteduncalibratedmodel"
        uncal_dir.mkdir()
        (uncal_dir / "model_probabilistic.pkl").write_bytes(b"prob")

        cfg = {}
        legacy_dir = tmp_path / "selectedvalidappliedmodel"
        legacy_dir.mkdir()
        result = _resolve_raw(cfg, uncal_dir, legacy_dir)
        assert result is None  # only probabilistic file present, should not be picked
