from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    tool_dir: Path
    selected_uncalibrated_model_dir: Path
    selected_beadcalibrated_model_dir: Path
    config_path: Path
    fws_calibration_store_path: Path
    beads_calibration_store_path: Path

    @property
    def selected_model_dir(self) -> Path:
        # Historical alias retained for compatibility with existing callers.
        return self.selected_uncalibrated_model_dir


@dataclass(frozen=True)
class RuntimeOptions:
    default_trained_models_container_url: str
    bead_calibration_max_age_seconds: int


@dataclass(frozen=True)
class RuntimeConfig:
    paths: RuntimePaths
    options: RuntimeOptions


def get_runtime_config(tool_dir: str | Path | None = None) -> RuntimeConfig:
    base_tool_dir = Path(tool_dir) if tool_dir is not None else (Path.home() / "Documents" / "flowcytometertool")

    paths = RuntimePaths(
        tool_dir=base_tool_dir,
        selected_uncalibrated_model_dir=base_tool_dir / "selectedbeaduncalibratedmodel",
        selected_beadcalibrated_model_dir=base_tool_dir / "selectedbeadcalibratedmodel",
        config_path=base_tool_dir / "flowcytometertoolconfig.yaml",
        fws_calibration_store_path=base_tool_dir / "FWScalibrations" / "FWScalibrations.jsonl",
        beads_calibration_store_path=base_tool_dir / "BeadsCalibrations" / "BeadsCalibrations.jsonl",
    )

    options = RuntimeOptions(
        default_trained_models_container_url="https://citprodflowcytosa.blob.core.windows.net/trainedmodels",
        bead_calibration_max_age_seconds=365 * 24 * 3600,
    )

    return RuntimeConfig(paths=paths, options=options)
