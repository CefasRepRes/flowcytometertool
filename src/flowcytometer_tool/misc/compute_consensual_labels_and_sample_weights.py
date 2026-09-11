import pandas as pd
from collections import Counter

def _compute_consensual_labels_and_sample_weights(
    data: pd.DataFrame,
    *,
    # identity columns
    filename_col: str = "filename",
    id_col: str = "id",
    # labels & weights
    label_col: str = "source_label",
    weight_col: str = "weight",
    # Adoption policy
    apply_only_on_unanimous: bool = False,   # keep originals unless 100% agreement
    # Safety/diagnostics during development
    assert_preserve_order: bool = True,
    assert_no_row_count_change: bool = True,
) -> pd.DataFrame:
    """
    Compute consensus per *physical particle* identified by (filename + id),
    preserving the original row order and length.

    Returns the same dataframe plus:
        - 'consensus_label'  (weighted mode within each (file,id) group)
        - 'sample_weight'    (max_label_weight / total_weight in group, ∈ [0,1])

    If `apply_only_on_unanimous=True`, `label_col` is overwritten only where
    `sample_weight == 1.0`; otherwise labels are not changed.
    """

    if id_col not in data.columns:
        raise KeyError(f"Column '{id_col}' not found.")
    if label_col not in data.columns:
        raise KeyError(f"Column '{label_col}' not found.")
    if weight_col not in data.columns:
        raise KeyError(f"Column '{weight_col}' not found.")

    # We require a file-level identity so that IDs from different files don't collide.
    filename_candidates = [filename_col, "source_file"]
    real_filename_col = next((c for c in filename_candidates if c in data.columns), None)
    if real_filename_col is None:
        raise KeyError(
            f"No filename column found. Expected one of: {filename_candidates}. "
            f"Ensure the combine step adds a file identifier column."
        )

    df = data.copy()
    original_index = df.index.copy()
    original_len = len(df)

    # Normalize text columns used in grouping
    df[real_filename_col] = df[real_filename_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()

    # Build unique identity per physical particle
    df["_uid"] = df[real_filename_col] + "_" + df[id_col].astype(str).str.strip()

    # --- Fast path: if each (_uid) appears once and all weights==1, stamp outputs
    counts = df["_uid"].value_counts(dropna=False)
    is_singleton = (counts.max() == 1)
    if is_singleton and df[weight_col].fillna(1.0).eq(1.0).all():
        df["consensus_label"] = df[label_col]
        df["sample_weight"] = 1.0
        if apply_only_on_unanimous:
            # identical outcome; nothing to change
            pass
        if assert_preserve_order:
            assert df.index.equals(original_index), "Index changed unexpectedly (fast-path)."
        if assert_no_row_count_change:
            assert len(df) == original_len, "Row count changed unexpectedly (fast-path)."
        df.drop(columns=["_uid"], inplace=True)
        return df

    # --- General path: compute weighted mode within each (_uid)

    records = []
    for uid, grp in df.groupby("_uid", sort=False):
        labels = grp[label_col].tolist()
        weights = grp[weight_col].astype(float).fillna(0.0).tolist()

        acc = Counter()
        total_w = 0.0
        for l, w in zip(labels, weights):
            acc[l] += w
            total_w += w

        if not acc:
            # fallback: no weights—keep first label, zero share
            consensus_label = labels[0] if labels else ""
            share = 0.0
        else:
            max_w = max(acc.values())
            # deterministic tie-break on label lexicographic order
            winners = sorted([k for k, v in acc.items() if v == max_w])
            consensus_label = winners[0]
            share = (max_w / total_w) if total_w > 0 else 0.0

        records.append((uid, consensus_label, share))

    cdf = pd.DataFrame(records, columns=["_uid", "consensus_label", "sample_weight"]).set_index("_uid")

    # LEFT-JOIN back to preserve order and cardinality
    df = df.join(cdf, on="_uid", how="left")

    # Optionally adopt consensus where (and only where) unanimous
    if apply_only_on_unanimous:
        unanimous = df["sample_weight"].ge(0.999999999)
        df.loc[unanimous, label_col] = df.loc[unanimous, "consensus_label"]

    # Safety checks
    if assert_preserve_order:
        assert df.index.equals(original_index), "Index changed during consensus join."
    if assert_no_row_count_change:
        assert len(df) == original_len, "Row count changed during consensus join."

    df.drop(columns=["_uid"], inplace=True)
    return df
