# test_gates.py
"""Tests for XML gate parsing and gate-based class assignment."""

import textwrap
import io
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from functions import (
    parse_gate_setlist,
    assign_classes_from_gates,
    combine_csvs_with_gates,
)


# ---------------------------------------------------------------------------
# Helpers – minimal XML builders
# ---------------------------------------------------------------------------

def _write_xml(tmp_path, content: str) -> Path:
    p = tmp_path / "gates.xml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


SIMPLE_RECTANGLE_XML = """\
    <?xml version="1.0" encoding="utf-8"?>
    <SetList>
      <DefaultSet ListID="0" Name="Default (all)" />
      <GateBasedSet ListID="1" Name="ClassA">
        <GateCollection>
          <RectangleGate X="0.0" Y="0.0" Width="10.0" Height="10.0">
            <XAxis Name="x" />
            <YAxis Name="y" />
          </RectangleGate>
        </GateCollection>
      </GateBasedSet>
      <GateBasedSet ListID="2" Name="ClassB">
        <GateCollection>
          <RectangleGate X="20.0" Y="20.0" Width="10.0" Height="10.0">
            <XAxis Name="x" />
            <YAxis Name="y" />
          </RectangleGate>
        </GateCollection>
      </GateBasedSet>
    </SetList>
"""

RANGE_GATE_XML = """\
    <?xml version="1.0" encoding="utf-8"?>
    <SetList>
      <DefaultSet ListID="0" Name="Default (all)" />
      <GateBasedSet ListID="1" Name="InRange">
        <GateCollection>
          <RangeGate RangeMin="5.0" RangeMax="15.0">
            <Axis Name="z" />
          </RangeGate>
        </GateCollection>
      </GateBasedSet>
    </SetList>
"""

POLYGON_GATE_XML = """\
    <?xml version="1.0" encoding="utf-8"?>
    <SetList>
      <DefaultSet ListID="0" Name="Default (all)" />
      <GateBasedSet ListID="1" Name="PolyClass">
        <GateCollection>
          <PolygonGate>
            <XAxis Name="px" />
            <YAxis Name="py" />
            <Path>
              <Point X="0.0" Y="0.0" />
              <Point X="10.0" Y="0.0" />
              <Point X="10.0" Y="10.0" />
              <Point X="0.0" Y="10.0" />
            </Path>
          </PolygonGate>
        </GateCollection>
      </GateBasedSet>
    </SetList>
"""

OR_SET_XML = """\
    <?xml version="1.0" encoding="utf-8"?>
    <SetList>
      <DefaultSet ListID="0" Name="Default (all)" />
      <GateBasedSet ListID="1" Name="ClassA">
        <GateCollection>
          <RectangleGate X="0.0" Y="0.0" Width="5.0" Height="5.0">
            <XAxis Name="x" />
            <YAxis Name="y" />
          </RectangleGate>
        </GateCollection>
      </GateBasedSet>
      <GateBasedSet ListID="2" Name="ClassB">
        <GateCollection>
          <RectangleGate X="10.0" Y="10.0" Width="5.0" Height="5.0">
            <XAxis Name="x" />
            <YAxis Name="y" />
          </RectangleGate>
        </GateCollection>
      </GateBasedSet>
      <OrSet ListID="3" Name="AorB">
        <SetList>
          <ListID>1</ListID>
          <ListID>2</ListID>
        </SetList>
        <AutoSet>False</AutoSet>
      </OrSet>
    </SetList>
"""


# ---------------------------------------------------------------------------
# parse_gate_setlist
# ---------------------------------------------------------------------------

class TestParseGateSetlist:
    def test_default_set(self, tmp_path):
        xml = _write_xml(tmp_path, SIMPLE_RECTANGLE_XML)
        sets = parse_gate_setlist(xml)
        assert 0 in sets
        assert sets[0]["kind"] == "default"

    def test_gate_based_sets_parsed(self, tmp_path):
        xml = _write_xml(tmp_path, SIMPLE_RECTANGLE_XML)
        sets = parse_gate_setlist(xml)
        assert 1 in sets and 2 in sets
        assert sets[1]["kind"] == "gate_based"
        assert sets[1]["name"] == "ClassA"
        assert sets[2]["name"] == "ClassB"

    def test_rectangle_gate_dimensions(self, tmp_path):
        xml = _write_xml(tmp_path, SIMPLE_RECTANGLE_XML)
        sets = parse_gate_setlist(xml)
        gate = sets[1]["gates"][0]
        assert gate["type"] == "rectangle"
        assert gate["x_min"] == pytest.approx(0.0)
        assert gate["x_max"] == pytest.approx(10.0)
        assert gate["y_min"] == pytest.approx(0.0)
        assert gate["y_max"] == pytest.approx(10.0)
        assert gate["x_axis"] == "x"
        assert gate["y_axis"] == "y"

    def test_range_gate(self, tmp_path):
        xml = _write_xml(tmp_path, RANGE_GATE_XML)
        sets = parse_gate_setlist(xml)
        gate = sets[1]["gates"][0]
        assert gate["type"] == "range"
        assert gate["axis"] == "z"
        assert gate["min"] == pytest.approx(5.0)
        assert gate["max"] == pytest.approx(15.0)

    def test_polygon_gate_points(self, tmp_path):
        xml = _write_xml(tmp_path, POLYGON_GATE_XML)
        sets = parse_gate_setlist(xml)
        gate = sets[1]["gates"][0]
        assert gate["type"] == "polygon"
        assert len(gate["points"]) == 4
        assert gate["x_axis"] == "px"
        assert gate["y_axis"] == "py"

    def test_or_set_members(self, tmp_path):
        xml = _write_xml(tmp_path, OR_SET_XML)
        sets = parse_gate_setlist(xml)
        assert 3 in sets
        assert sets[3]["kind"] == "or"
        assert set(sets[3]["members"]) == {1, 2}


# ---------------------------------------------------------------------------
# assign_classes_from_gates
# ---------------------------------------------------------------------------

class TestAssignClassesFromGates:
    def test_particles_in_classA(self, tmp_path):
        xml = _write_xml(tmp_path, SIMPLE_RECTANGLE_XML)
        df = pd.DataFrame({"x": [5.0, 25.0], "y": [5.0, 25.0]})
        result = assign_classes_from_gates(df, xml)
        assert result["source_label"].tolist() == ["ClassA", "ClassB"]

    def test_unclassified_particle(self, tmp_path):
        xml = _write_xml(tmp_path, SIMPLE_RECTANGLE_XML)
        df = pd.DataFrame({"x": [50.0], "y": [50.0]})
        result = assign_classes_from_gates(df, xml)
        assert result["source_label"].iloc[0] == "Unclassified"

    def test_range_gate_classification(self, tmp_path):
        xml = _write_xml(tmp_path, RANGE_GATE_XML)
        df = pd.DataFrame({"z": [10.0, 3.0, 15.0, 20.0]})
        result = assign_classes_from_gates(df, xml)
        expected = ["InRange", "Unclassified", "InRange", "Unclassified"]
        assert result["source_label"].tolist() == expected

    def test_polygon_gate_classification(self, tmp_path):
        xml = _write_xml(tmp_path, POLYGON_GATE_XML)
        df = pd.DataFrame({"px": [5.0, 15.0], "py": [5.0, 15.0]})
        result = assign_classes_from_gates(df, xml)
        assert result["source_label"].iloc[0] == "PolyClass"
        assert result["source_label"].iloc[1] == "Unclassified"

    def test_original_columns_preserved(self, tmp_path):
        xml = _write_xml(tmp_path, SIMPLE_RECTANGLE_XML)
        df = pd.DataFrame({"x": [5.0], "y": [5.0], "extra": [99]})
        result = assign_classes_from_gates(df, xml)
        assert "extra" in result.columns
        assert result["extra"].iloc[0] == 99

    def test_missing_column_raises(self, tmp_path):
        xml = _write_xml(tmp_path, SIMPLE_RECTANGLE_XML)
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})  # no 'x' or 'y'
        with pytest.raises(KeyError):
            assign_classes_from_gates(df, xml)

    def test_or_set_not_terminal(self, tmp_path):
        """OrSet members (1, 2) are not terminal; only the OrSet itself (3) is terminal."""
        xml = _write_xml(tmp_path, OR_SET_XML)
        df = pd.DataFrame({"x": [2.5, 12.5, 50.0], "y": [2.5, 12.5, 50.0]})
        result = assign_classes_from_gates(df, xml)
        # OrSet 3 ("AorB") covers both class-A and class-B regions
        assert result["source_label"].iloc[0] == "AorB"
        assert result["source_label"].iloc[1] == "AorB"
        assert result["source_label"].iloc[2] == "Unclassified"

