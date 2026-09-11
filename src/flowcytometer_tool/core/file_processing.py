from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessedFilePaths:
    source_cyz_path: str
    base_filename: str
    json_path: str
    listmode_csv_path: str
    predictions_csv_path: str
    prediction_counts_csv_path: str
    prediction_plot_3d_html_path: str
    image_dir_path: str


def build_processed_file_paths(file_path: str, output_folder: str) -> ProcessedFilePaths:
    src = Path(file_path)
    out = Path(output_folder)
    stem = src.stem
    base = src.name
    return ProcessedFilePaths(
        source_cyz_path=str(src),
        base_filename=base,
        json_path=str(out / f"{stem}.json"),
        listmode_csv_path=str(out / f"{stem}.csv"),
        predictions_csv_path=str(out / "predictions.csv"),
        # Keep legacy naming for downstream compatibility with existing watcher outputs
        # and dashboard-side file lookups that currently read predictions.csv_* artifacts.
        prediction_counts_csv_path=str(out / "predictions.csv_counts.csv"),
        prediction_plot_3d_html_path=str(out / "predictions.csv_3d.html"),
        image_dir_path=f"{out / 'images'}{os.sep}",
    )


def wait_for_file_release(file_path: str, timeout: int = 30, interval: int = 1) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with open(file_path, "rb"):
                return True
        except IOError:
            time.sleep(interval)
    return False


def normalise_dataframe_columns(dataframe):
    out = dataframe.copy()
    out.columns = out.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    return out


def build_protocol_packet(measurement_settings: dict[str, Any] | None) -> dict[str, Any]:
    settings = measurement_settings if isinstance(measurement_settings, dict) else {}
    cyto_settings = settings.get("CytoSettings", {}) if isinstance(settings.get("CytoSettings"), dict) else {}

    return {
        "instrument.measurementSettings.CytoSettings.IIFCheck": cyto_settings.get("IIFCheck"),
        "instrument.measurementSettings.CytoSettings.IsBeadsMeasurement": cyto_settings.get(
            "IsBeadsMeasurement",
            settings.get("beads_measurement_2"),
        ),
        "instrument.measurementSettings.beads_measurement_2": settings.get("beads_measurement_2"),
    }


def detect_protocol_from_json(json_path: str, detect_sampling_protocol_fn) -> tuple[str, dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8-sig") as jf:
        full_js = json.load(jf)

    instrument = full_js.get("instrument", {}) if isinstance(full_js, dict) else {}
    measurement_settings = instrument.get("measurementSettings", {}) if isinstance(instrument, dict) else {}
    packet = build_protocol_packet(measurement_settings)
    return detect_sampling_protocol_fn(packet), packet
