from pathlib import Path

import numpy as np
import pandas as pd

from bead_calibration import BEADS_CHANNEL_CONFIG, compute_beads_calibration_from_df
from protocols import apply_sampling_protocol_mutations, detect_sampling_protocol, is_bead_sample
from reporting_core import write_report_packet_flat


def test_detect_sampling_protocol_existing_and_new_rules():
    nano_packet = {
        "instrument.measurementSettings.CytoSettings.SamplePompSpeed": 9.0,
        "instrument.measurementSettings.CytoSettings.TriggerLevel1e": 4.0,
    }
    pico_packet = {
        "instrument.measurementSettings.CytoSettings.SamplePompSpeed": 4.0,
        "instrument.measurementSettings.CytoSettings.TriggerLevel1e": 2.0,
    }
    image_packet = {
        "instrument.measurementSettings.CytoSettings.SamplePompSpeed": 9.0,
        "instrument.measurementSettings.CytoSettings.TriggerLevel1e": 4.0,
        "instrument.measurementSettings.CytoSettings.IIFCheck": True,
    }
    beads_packet = {
        "instrument.measurementSettings.CytoSettings.SamplePompSpeed": 9.0,
        "instrument.measurementSettings.CytoSettings.TriggerLevel1e": 4.0,
        "instrument.measurementSettings.CytoSettings.IIFCheck": True,
        "instrument.measurementSettings.beads_measurement_2": True,
    }

    assert detect_sampling_protocol(nano_packet) == "nanoprotocol"
    assert detect_sampling_protocol(pico_packet) == "picoprotocol"
    assert detect_sampling_protocol(image_packet) == "imageprotocol"
    assert detect_sampling_protocol(beads_packet) == "beadsprotocol"
    assert is_bead_sample(beads_packet) is True
    assert is_bead_sample(nano_packet) is False


def test_apply_sampling_protocol_mutations_renames_count_keys():
    packet = {
        "instrument.measurementSettings.CytoSettings.SamplePompSpeed": 4.0,
        "instrument.measurementSettings.CytoSettings.TriggerLevel1e": 2.1,
        "Red_Count": 11,
        "Orange_Count": 7,
    }
    out = apply_sampling_protocol_mutations(packet)
    assert out["samplingprotocol"] == "picoprotocol"
    assert "Red_Count_picoprotocol" in out
    assert "Orange_Count_picoprotocol" in out
    assert "Red_Count" not in out


def test_compute_beads_calibration_from_df_outputs_expected_artifact(tmp_path):
    rng = np.random.default_rng(7)
    expected = np.array([2, 8, 32], dtype=float)

    def _channel(scale: float) -> np.ndarray:
        vals = []
        for x in expected:
            vals.extend(scale * x * rng.lognormal(mean=0.0, sigma=0.03, size=120))
        return np.array(vals, dtype=float)

    df = pd.DataFrame(
        {
            "FL Orange_total": _channel(1.0),
            "FL Red_total": _channel(1.3),
            "FL Yellow_total": _channel(0.8),
        }
    )

    diag = tmp_path / "beads_diag.png"
    artifact = compute_beads_calibration_from_df(df, diagnostic_png_path=diag)

    assert artifact is not None
    assert set(artifact["channels"].keys()) == {"orange", "red", "yellow"}
    assert "qc_pass" in artifact
    assert diag.exists()

    expected_units = {
        channel: BEADS_CHANNEL_CONFIG[channel]["fluorophore_unit"]
        for channel in ("orange", "red", "yellow")
    }
    for channel in ("orange", "red", "yellow"):
        ch = artifact["channels"][channel]
        assert len(ch["detected_peak_positions"]) == 3
        assert len(ch["expected_intensity_positions"]) == 3
        assert ch["fluorophore_unit"] == expected_units[channel]
        coeffs = ch["calibration_curve_coefficients"]
        assert np.isfinite(coeffs["slope"])
        assert np.isfinite(coeffs["intercept"])
        assert np.isfinite(coeffs["r2"])


def test_write_report_packet_flat_exposes_beads_flag_for_detection(tmp_path):
    src_json = tmp_path / "sample.json"
    grablist = tmp_path / "grablist.txt"
    out = tmp_path / "packet.json"
    grablist.write_text("instrument.measurementSettings.CytoSettings.SamplePompSpeed\n")

    src_json.write_text(
        """
{
  "instrument": {
    "serialNumber": "SERIAL-1",
    "measurementResults": {"start": "2025-01-01T00:00:00Z", "duration": 10},
    "measurementSettings": {
      "beads_measurement_2": true,
      "CytoSettings": {
        "IIFCheck": false,
        "SaveTextbox": "C:/tmp/survey",
        "SamplePompSpeed": 9.0,
        "TriggerLevel1e": 4.0
      }
    }
  }
}
"""
    )

    packet = write_report_packet_flat(
        metadata={"instrument_json_path": str(src_json)},
        modelsettings={},
        predictions_df=None,
        grablist_path=str(grablist),
        json_path=str(src_json),
        output_path=str(out),
        modelversion="model.pkl",
    )

    apply_sampling_protocol_mutations(packet)
    assert packet["instrument.measurementSettings.beads_measurement_2"] is True
    assert packet["samplingprotocol"] == "beadsprotocol"
