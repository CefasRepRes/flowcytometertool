# test_consensus.py
import pandas as pd
import pytest
import sys
from pathlib import Path

# add repo/src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowcytometertool.functions import compute_consensual_labels_and_sample_weights


def assert_eq_df_ignore_row_order(actual: pd.DataFrame, expected: pd.DataFrame, sort_by):
    """
    Helper to compare two dataframes ignoring row order but requiring
    exact equality on values (within float tolerance).
    """
    actual_sorted = actual.sort_values(sort_by).reset_index(drop=True)
    expected_sorted = expected.sort_values(sort_by).reset_index(drop=True)

    # Align column order for robustness
    actual_sorted = actual_sorted[expected_sorted.columns]
    pd.testing.assert_frame_equal(actual_sorted, expected_sorted, check_dtype=False, atol=1e-12, rtol=1e-12)


def test_basic_consensus_and_sample_weight_excluding_unassigned():
    """
    - Groups by 'id'
    - Computes weighted mode per id
    - Computes sample_weight = consensus_weight / total_weight
    - Excludes ids whose consensus label is "Unassigned Particles"
    - Merges consensus results back to original rows for the kept ids only
    """
    data = pd.DataFrame({
        "id": [1, 1, 1, 2, 2, 3, 3],
        "source_label": [
            "Diatom", "Diatom", "Coccolith",                 # id=1 -> Diatom wins
            "Unassigned Particles", "Cyanobacteria",         # id=2 -> depends on weights (set below)
            "Unassigned Particles", "Unassigned Particles"   # id=3 -> Unassigned wins, gets excluded
        ],
        "weight": [0.7, 0.2, 0.3, 0.6, 0.5, 1.0, 2.0]
    })

    # For id=1: Diatom weight = 0.7 + 0.2 = 0.9, Coccolith = 0.3; total = 1.2
    # sample_weight = 0.9 / 1.2 = 0.75
    # For id=2: Unassigned = 0.6, Cyanobacteria = 0.5 => Unassigned wins => id=2 excluded
    # For id=3: Unassigned wins => id=3 excluded

    result = compute_consensual_labels_and_sample_weights(data)

    # Expect only id=1 rows retained, merged back to the original rows for id=1
    expected = pd.DataFrame({
        "id": [1, 1, 1],
        "source_label": ["Diatom", "Diatom", "Coccolith"],
        "weight": [0.7, 0.2, 0.3],
        "consensus_label": ["Diatom", "Diatom", "Diatom"],
        "sample_weight": [0.75, 0.75, 0.75],
    })

    assert_eq_df_ignore_row_order(result, expected, sort_by=["id", "source_label", "weight"])


def test_weighted_tie_breaks_by_higher_total_weight():
    """
    Ensures the consensus is determined by the highest summed weight, not count.
    """
    data = pd.DataFrame({
        "id": [100, 100, 100, 100],
        "source_label": ["A", "A", "B", "B"],
        "weight": [0.3, 0.3, 0.7, 0.1],  # A total = 0.6, B total = 0.8 -> B should win
    })

    result = compute_consensual_labels_and_sample_weights(data)

    # total = 1.4, consensus_weight (B) = 0.8 -> sample_weight = 0.8 / 1.4
    expected_sw = 0.8 / 1.4
    assert set(result["consensus_label"].unique()) == {"B"}
    assert pytest.approx(result["sample_weight"].iloc[0], rel=1e-12) == expected_sw
    # Should be merged onto all rows with id=100:
    assert len(result) == 4


def test_multiple_ids_mixed_outcomes():
    data = pd.DataFrame({
        "id": [1, 1, 2, 2, 2, 3, 3, 4],
        "source_label": [
            "A", "B",          # id=1 => A=1.0, B=2.0 -> B wins
            "X", "X", "Y",     # id=2 => X=1.8, Y=0.2 -> X wins
            "Unassigned Particles", "Z",  # id=3 => Unassigned=0.5, Z=0.6 -> Z wins (kept)
            "Unassigned Particles"        # id=4 => only Unassigned -> excluded
        ],
        "weight": [
            1.0, 2.0,           # id=1
            1.0, 0.8, 0.2,      # id=2
            0.5, 0.6,           # id=3
            1.0                 # id=4
        ],
    })

    result = compute_consensual_labels_and_sample_weights(data)

    # Keep ids 1,2,3. Exclude id=4.
    assert set(result["id"].unique()) == {1, 2, 3}

    # Validate consensus per id
    by_id = result.groupby("id").agg(
        consensus=("consensus_label", lambda s: s.iloc[0]),
        sw=("sample_weight", lambda s: s.iloc[0]),
        total_rows=("consensus_label", "count")
    )

    # id=1: B wins (2.0 / 3.0)
    assert by_id.loc[1, "consensus"] == "B"
    assert pytest.approx(by_id.loc[1, "sw"], rel=1e-12) == 2.0 / 3.0
    assert by_id.loc[1, "total_rows"] == 2  # there were 2 original rows for id=1

    # id=2: X wins (1.8 / 2.0)
    assert by_id.loc[2, "consensus"] == "X"
    assert pytest.approx(by_id.loc[2, "sw"], rel=1e-12) == 1.8 / 2.0
    assert by_id.loc[2, "total_rows"] == 3

    # id=3: Z wins (0.6 / 1.1)
    assert by_id.loc[3, "consensus"] == "Z"
    assert pytest.approx(by_id.loc[3, "sw"], rel=1e-12) == 0.6 / 1.1
    assert by_id.loc[3, "total_rows"] == 2


def test_all_unassigned_leads_to_empty_merge():
    data = pd.DataFrame({
        "id": [9, 9, 10],
        "source_label": ["Unassigned Particles", "Unassigned Particles", "Unassigned Particles"],
        "weight": [0.4, 0.6, 1.0],
    })
    result = compute_consensual_labels_and_sample_weights(data)
    # No ids retained ⇒ inner merge produces empty result
    assert result.empty


def test_zero_total_weight_raises_zero_division_error():
    """
    With all weights zero, total_weight=0 would cause division by zero per current implementation.
    This test documents (and protects) the current behavior.
    If you later change the function to handle this case gracefully, update the test accordingly.
    """
    data = pd.DataFrame({
        "id": [1, 1],
        "source_label": ["A", "B"],
        "weight": [0.0, 0.0],
    })
    with pytest.raises(ZeroDivisionError):
        compute_consensual_labels_and_sample_weights(data)