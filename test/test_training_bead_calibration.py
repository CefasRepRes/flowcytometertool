import json

import numpy as np
import pandas as pd

from flowcytometer_tool.misc.bead_training import (
    beadcalibrated_model_path,
    prepare_training_dataframe_with_optional_bead_calibration,
    update_modeltrainsettings,
    update_modeltrainsettings_bead_flag,
)


def _make_bead_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    expected = np.array([2, 8, 32], dtype=float)

    def _channel(scale: float) -> np.ndarray:
        vals = []
        for x in expected:
            vals.extend(scale * x * rng.lognormal(mean=0.0, sigma=0.03, size=80))
        return np.array(vals, dtype=float)

    return pd.DataFrame(
        {
            "Fl_Orange_total": _channel(1.0),
            "Fl_Red_total": _channel(1.2),
            "Fl_Yellow_total": _channel(0.9),
        }
    )


def _beads_packet() -> dict:
    return {
        "instrument.measurementSettings.CytoSettings.IIFCheck": True,
        "instrument.measurementSettings.beads_measurement_2": True,
    }


def test_training_without_bead_samples_is_unchanged():
    training_df = pd.DataFrame(
        {
            "source_label": ["A", "B"],
            "weight": [1.0, 1.0],
            "FL_Orange_total": [100.0, 200.0],
            "FWS_total": [10.0, 20.0],
            "SSC_total": [50.0, 60.0],
        }
    )

    out_df, used, meta = prepare_training_dataframe_with_optional_bead_calibration(training_df, bead_samples=None)
    assert used is False
    assert meta["bead_sample_count"] == 0
    pd.testing.assert_frame_equal(out_df, training_df)


def test_update_modeltrainsettings_merges_training_metadata(tmp_path):
    settings_path = tmp_path / "nested" / "modeltrainsettings.json"
    settings_path.parent.mkdir()
    settings_path.write_text(
        json.dumps({"instrument": {"serialNumber": "serial"}, "training": {"existing": True}}),
        encoding="utf-8",
    )

    updated = update_modeltrainsettings(settings_path, {"bead_calibrated": True})

    assert updated["instrument"] == {"serialNumber": "serial"}
    assert updated["training"] == {"existing": True, "bead_calibrated": True}
    assert json.loads(settings_path.read_text(encoding="utf-8")) == updated


def test_training_with_single_bead_sample_calibrates_and_filters_columns():
    training_df = pd.DataFrame(
        {
            "source_label": ["A", "B", "A"],
            "weight": [1.0, 1.0, 1.0],
            "FL_Orange_total": [100.0, 150.0, 210.0],
            "FL_Orange_max": [120.0, 160.0, 220.0],
            "FL_red_mean": [70.0, 80.0, 90.0],
            "FL_yellow_total": [60.0, 75.0, 95.0],
            "FWS_total": [10.0, 20.0, 30.0],
            "SSC_total": [9.0, 8.0, 7.0],
        }
    )
    bead_samples = [{"packet": _beads_packet(), "dataframe": _make_bead_df(seed=7)}]

    out_df, used, meta = prepare_training_dataframe_with_optional_bead_calibration(training_df, bead_samples=bead_samples)

    assert used is True
    assert meta["bead_sample_count"] == 1
    assert "SSC_total" not in out_df.columns
    assert "FWS_total" in out_df.columns
    expected_calibrated = {
        "FL_Orange_total": "FL_Orange_total_calibrated",
        "FL_yellow_total": "FL_yellow_total_calibrated",
    }
    for raw_col, calibrated_col in expected_calibrated.items():
        assert calibrated_col in out_df.columns
        assert raw_col not in out_df.columns
        assert not np.allclose(out_df[calibrated_col].to_numpy(), training_df[raw_col].to_numpy())


def test_training_with_multiple_bead_samples_uses_all_valid_beads():
    training_df = pd.DataFrame(
        {
            "source_label": ["A", "B", "A"],
            "weight": [1.0, 1.0, 1.0],
            "FL_Orange_total": [100.0, 150.0, 210.0],
            "FL_red_total": [70.0, 80.0, 90.0],
            "FL_yellow_total": [60.0, 75.0, 95.0],
            "FWS_height": [10.0, 20.0, 30.0],
            "Unrelated_numeric": [3.0, 4.0, 5.0],
        }
    )
    bead_samples = [
        {"packet": _beads_packet(), "dataframe": _make_bead_df(seed=11)},
        {"packet": _beads_packet(), "dataframe": _make_bead_df(seed=12)},
    ]

    out_df, used, meta = prepare_training_dataframe_with_optional_bead_calibration(training_df, bead_samples=bead_samples)
    assert used is True
    assert meta["bead_sample_count"] == 2
    assert "FWS_height" in out_df.columns
    assert "Unrelated_numeric" not in out_df.columns


def test_beadcalibrated_filename_suffix_and_settings_flag(tmp_path):
    model_path = str(tmp_path / "final_model_20260101_010203.pkl")
    calibrated_path = beadcalibrated_model_path(model_path)
    assert calibrated_path.endswith("_beadcalibrated.pkl")
    assert beadcalibrated_model_path(calibrated_path) == calibrated_path

    settings_path = tmp_path / "modeltrainsettings.json"
    update_modeltrainsettings_bead_flag(str(settings_path), used_bead_calibration=True, bead_sample_count=2)
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    assert data["training"]["bead_calibrated"] is True
    assert data["training"]["bead_sample_count"] == 2
