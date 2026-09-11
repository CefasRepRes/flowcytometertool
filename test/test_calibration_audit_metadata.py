from pathlib import Path

import numpy as np
import pandas as pd

from flowcytometer_tool.tabs.continuous_sample_analyser.bead_calibration import run_protocol_postprocessing
from flowcytometer_tool.tabs.continuous_sample_analyser.reporting_core import (
    extract_prediction_calibration_packet_updates,
)


def test_extract_prediction_calibration_packet_updates_reads_expected_fields():
    preds = pd.DataFrame(
        {
            "predicted_label": ["nano"],
            "bead_calibration_used": ["True"],
            "model_mode_used": ["bead_calibrated_model"],
            "calibration_timestamp": ["2026-01-01T00:00:00Z"],
            "calibration_age_seconds": [123.5],
            "calibration_source_file": ["beads_001"],
            "calibration_qc_passed": ["false"],
        }
    )

    updates = extract_prediction_calibration_packet_updates(preds)
    assert updates == {
        "bead_calibration_used": True,
        "model_mode_used": "bead_calibrated_model",
        "calibration_timestamp": "2026-01-01T00:00:00Z",
        "calibration_age_seconds": 123.5,
        "calibration_source_file": "beads_001",
        "calibration_qc_passed": False,
    }


def test_run_protocol_postprocessing_reports_bead_calibration_audit_fields(tmp_path):
    rng = np.random.default_rng(11)
    expected = np.array([2, 8, 32], dtype=float)

    def _channel(scale: float) -> np.ndarray:
        vals = []
        for x in expected:
            vals.extend(scale * x * rng.lognormal(mean=0.0, sigma=0.03, size=120))
        return np.array(vals, dtype=float)

    bead_df = pd.DataFrame(
        {
            "Fl_Orange_total": _channel(1.0),
            "Fl_Red_total": _channel(1.2),
            "Fl_Yellow_total": _channel(0.9),
        }
    )

    updates = run_protocol_postprocessing(
        protocol="beadsprotocol",
        packet={"system_serial_no": "SERIAL-1"},
        dataframe=bead_df,
        file_id="beads_file_1",
        diagnostic_dir=tmp_path,
        beads_output_root=tmp_path,
    )

    assert updates["bead_calibration_created"] is True
    assert isinstance(updates["bead_calibration_qc_passed"], bool)
    diag_path = Path(updates["bead_calibration_diagnostic_plot_path"])
    assert diag_path.exists()
    assert updates["beads_calibration_qc"] in {"pass", "fail"}
