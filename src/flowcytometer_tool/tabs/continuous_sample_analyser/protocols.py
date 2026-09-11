from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_SUFFIXES = (
    "_nanoprotocol",
    "_picoprotocol",
    "_unknownprotocol",
    "_imageprotocol",
    "_beadsprotocol",
)


def _to_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip() != "":
            return float(v)
    except Exception:
        return None
    return None


def _normalise_boolish(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "t", "1", "yes", "y")
    return bool(v)


@dataclass(frozen=True)
class ProtocolHandler:
    name: str
    priority: int

    def matches(self, packet: dict) -> bool:
        raise NotImplementedError


class ImageProtocolHandler(ProtocolHandler):
    def matches(self, packet: dict) -> bool:
        return _normalise_boolish(packet.get("instrument.measurementSettings.CytoSettings.IIFCheck"))


class BeadsProtocolHandler(ProtocolHandler):
    def matches(self, packet: dict) -> bool:
        return _normalise_boolish(
            packet.get("instrument.measurementSettings.CytoSettings.IsBeadsMeasurement")
            or packet.get("instrument.measurementSettings.beads_measurement_2")
        )


class NanoProtocolHandler(ProtocolHandler):
    def matches(self, packet: dict) -> bool:
        pump = _to_float(packet.get("instrument.measurementSettings.CytoSettings.SamplePompSpeed"))
        trig = _to_float(packet.get("instrument.measurementSettings.CytoSettings.TriggerLevel1e"))
        return (
            (pump is not None and 8.5 <= pump <= 9.5)
            and (trig is not None and 2.9 <= trig <= 5.1)
        )


class PicoProtocolHandler(ProtocolHandler):
    def matches(self, packet: dict) -> bool:
        pump = _to_float(packet.get("instrument.measurementSettings.CytoSettings.SamplePompSpeed"))
        trig = _to_float(packet.get("instrument.measurementSettings.CytoSettings.TriggerLevel1e"))
        return (
            (pump is not None and 3.5 <= pump <= 4.5)
            and (trig is not None and 1.8 <= trig <= 2.5)
        )


_REGISTRY: list[ProtocolHandler] = [
    BeadsProtocolHandler(name="beadsprotocol", priority=100),
    ImageProtocolHandler(name="imageprotocol", priority=90),
    NanoProtocolHandler(name="nanoprotocol", priority=10),
    PicoProtocolHandler(name="picoprotocol", priority=5),
]


def detect_sampling_protocol(packet: dict) -> str:
    for handler in sorted(_REGISTRY, key=lambda h: h.priority, reverse=True):
        if handler.matches(packet):
            return handler.name
    return "unknownprotocol"


def is_bead_sample(packet: dict, *, protocol: str | None = None) -> bool:
    """Return whether a packet is identified as a bead measurement."""
    return (protocol or detect_sampling_protocol(packet)) == "beadsprotocol"


def apply_sampling_protocol_mutations(packet: dict) -> dict:
    protocol = detect_sampling_protocol(packet)
    packet["samplingprotocol"] = protocol

    for key in list(packet):
        if not key.endswith("_Count"):
            continue
        if any(key.endswith(f"_Count{sfx}") for sfx in PROTOCOL_SUFFIXES):
            continue

        new_key = f"{key}_{protocol}"
        if new_key not in packet:
            packet[new_key] = packet.pop(key)
    return packet
