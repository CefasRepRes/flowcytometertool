# -*- coding: utf-8 -*-
"""
Workflow to aggregate counts-only predictions and compare against Rutten Excel exports,
with a toggle to ingest either from Azure Blob (existing) or from a downloaded TAR
that contains a 'blob_tool_outputs' folder.

This version auto-appends a SAS token to the TAR URL if the URL has no query string.
"""

# -----------------------------
# Standard libs
# -----------------------------
import io
import os
import sys
import tarfile
import argparse
from urllib.parse import urlparse
from pathlib import Path

# -----------------------------
# Third-party
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px

# Azure SDK (only used for blob path)
from azure.storage.blob import BlobServiceClient

# -----------------------------
# SAS token helper
# -----------------------------
def get_sas_token(file_path):
    """Read a SAS token from a local file and return it as a trimmed string (without leading '?')."""
    with open(file_path, 'r') as file:
        tok = file.read().strip()
    # strip leading '?' if present
    if tok.startswith('?'):
        tok = tok[1:]
    return tok

# -----------------------------
# Config
# -----------------------------
def default_paths():
    user_home = Path.home()
    downloads = user_home / "Downloads"
    return {
        "sas_token_path": r"C:/Users/JR13/Documents/authenticationkeys/flowcytosaSAS.txt",
        "account_url": "https://citprodflowcytosa.blob.core.windows.net",
        "container_name": "mnceacyzfilesforthomasrutten",
        "output_master_table_csv": str(downloads / "master_table_counts_only_ordered.csv"),
        "output_timeseries_html": str(downloads / "counts_timeseries_ordered.html"),
        "rutten_thames_xlsx": str(downloads / "Thames_merge_VAL_clusSumTotal.xlsx"),
        "rutten_mersey_xlsx": str(downloads / "Mersey_merge_VAL_clusSumTotal.xlsx"),
    }

# -----------------------------
# Domain constants
# -----------------------------
class_columns = [
    'RedPico', 'Orapicoprok', 'Other', 'RedNano', 'nophytoplankton', 'noiseum',
    'no_phytoplankton', 'YB_um_beads', 'Plant_detritus', 'OraNano_crypto',
    'RWS_um_beads', 'Beads_rest', 'RedMicro', 'Bubbles', 'OraNano', 'C_undetermined'
]

TO_COLUMNS = [
    "Counts_Beads_rest", "Counts_Bubbles", "Counts_No_Phytopl", "Counts_No_Phyto",
    "Counts_No_Phyto_large", "Counts_OraNano", "Counts_OraNano_Crypt",
    "Counts_OraPicoProk", "Counts_Plant_detritus", "Counts_RWS_3um_beads",
    "Counts_RedMicro", "Counts_RedNano", "Counts_RedPico", "Counts_YB_1um_beads",
    "Counts_noise_gt1um", "Counts_noise_st1um", "Counts_not_recognized",
]

SOURCE_TO_TARGET = {
    "no_phyto": "Counts_No_Phyto",
    "no_phyto_large": "Counts_No_Phyto_large",
    "redpico": "Counts_RedPico",
    "rednano": "Counts_RedNano",
    "orapicoprok": "Counts_OraPicoProk",
    "1um_beads": "Counts_YB_1um_beads",
    "oranano_crypt": "Counts_OraNano_Crypt",
    "rws_3um_beads": "Counts_RWS_3um_beads",
    "redmicro": "Counts_RedMicro",
    "oranano": "Counts_OraNano",
    "plankt_detritus": "Counts_Plant_detritus",
    "beads_rest": "Counts_Beads_rest",
    # extras appearing in pipeline:
    "bubbles": "Counts_Bubbles",
    "noiseum": "Counts_noise_gt1um",
    "nophytoplankton": "Counts_No_Phytopl",
    "c_undetermined": "Counts_not_recognized",
}

def _normalize_key(s: str) -> str:
    return str(s).strip().lower().replace("__", "_").replace(" ", "_")

norm_map = { _normalize_key(k): v for k, v in SOURCE_TO_TARGET.items() }

# -----------------------------
# FNAME_LOOKUP (shortened placeholder here; in your runtime replace with full mapping as needed)
# For this downloadable script we include the keys you showed; extend as needed.
# -----------------------------
FNAME_LOOKUP = {
    # (Keep your full mapping here – omitted for brevity in this generated file)
}

DEST_TXT = 'destinations.txt'

def load_destination_order(txt_path: str) -> list:
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines()]
    labels = [ln for ln in lines if ln]
    if labels and labels[0].lower() == 'fname':
        labels = labels[1:]
    return labels

# =============================
# Ingestion helpers
# =============================

def _canonicalize_counts_dataframe(df_raw: pd.DataFrame, raw_name: str) -> pd.DataFrame:
    df_t = df_raw.T.reset_index()
    df_t.columns = df_t.iloc[0]
    df_t = df_t[1:]

    counts_out = pd.DataFrame({col: np.nan for col in TO_COLUMNS}, index=df_t.index)

    for col in df_t.columns:
        ncol = _normalize_key(col)
        if ncol in {"yb_um_beads", "yb_1um_beads", "1_um_beads"}:
            ncol = "1um_beads"
        if ncol in {"rws_um_beads", "rws3um_beads", "rws_3_um_beads"}:
            ncol = "rws_3um_beads"
        target = norm_map.get(ncol)
        if target is None:
            continue
        counts_out[target] = pd.to_numeric(df_t[col], errors="coerce")

    non_count_cols = [c for c in df_t.columns if _normalize_key(c) not in norm_map]
    df_t = pd.concat([df_t[non_count_cols], counts_out], axis=1)

    df_t['filename'] = raw_name
    from_lookup = FNAME_LOOKUP.get(raw_name, None)
    df_t['sample_id'] = from_lookup if from_lookup is not None else raw_name

    for col in class_columns:
        if col not in df_t.columns:
            df_t[col] = np.nan
        df_t[col] = pd.to_numeric(df_t[col], errors='coerce')

    time_candidates = ['start_time', 'Start', 'start', 'timestamp', 'time']
    parsed_time = None
    for cand in time_candidates:
        if cand in df_t.columns:
            parsed_time = pd.to_datetime(df_t[cand], errors='coerce', utc=True)
            break
    df_t['start_time'] = parsed_time if parsed_time is not None else pd.NaT

    return df_t


def ingest_from_blob(account_url: str, container_name: str, sas_token: str) -> pd.DataFrame:
    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container_client = blob_service_client.get_container_client(container_name)

    master_table = pd.DataFrame()
    for blob in container_client.list_blobs():
        if not blob.name.endswith("counts.csv"):
            continue
        blob_client = container_client.get_blob_client(blob)
        csv_content = blob_client.download_blob().readall()
        df_raw = pd.read_csv(io.BytesIO(csv_content))
        raw_name = blob.name.split('/')[-1]
        df_t = _canonicalize_counts_dataframe(df_raw, raw_name)
        master_table = pd.concat([master_table, df_t], ignore_index=True, sort=False)

    return master_table


def _extract_tar(tar_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if str(tar_path).endswith(".tar.gz") else "r:"
    with tarfile.open(tar_path, mode) as tf:
        tf.extractall(path=extract_dir)
    return extract_dir


def _ensure_signed_url(base_url: str, sas_token_path: str) -> str:
    """Append SAS token to base_url if it has no query string."""
    parsed = urlparse(base_url)
    if parsed.query and len(parsed.query.strip()) > 0:
        return base_url  # already signed
    sas = get_sas_token(sas_token_path)
    sep = '&' if '?' in base_url else '?'
    return f"{base_url}{sep}{sas}"


def ingest_from_tar_url(tar_url: str, downloads_dir: Path, sas_token_path: str) -> pd.DataFrame:
    import urllib.request

    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Auto-append SAS if missing
    signed_url = _ensure_signed_url(tar_url, sas_token_path)

    tar_name = os.path.basename(tar_url.split('?')[0]) or "archive.tar"
    tar_path = downloads_dir / tar_name
    extract_root = downloads_dir / tar_name.replace(".tar.gz", "").replace(".tar", "")

    if not tar_path.exists():
        print(f"Downloading TAR from {signed_url} -> {tar_path}")
        urllib.request.urlretrieve(signed_url, tar_path)
    else:
        print(f"Using existing TAR at {tar_path}")

    print(f"Extracting {tar_path} -> {extract_root}")
    _extract_tar(tar_path, extract_root)

    # Find blob_tool_outputs
    candidates = [extract_root / "blob_tool_outputs"]
    if extract_root.exists():
        for p in extract_root.iterdir():
            if p.is_dir():
                candidates.append(p / "blob_tool_outputs")

    blob_outputs = None
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            blob_outputs = cand
            break
    if blob_outputs is None:
        raise FileNotFoundError("'blob_tool_outputs' not found after extracting TAR.")

    master_table = pd.DataFrame()
    count_files = list(blob_outputs.rglob("*counts.csv"))
    for csv_path in count_files:
        df_raw = pd.read_csv(csv_path)
        raw_name = csv_path.name
        df_t = _canonicalize_counts_dataframe(df_raw, raw_name)
        master_table = pd.concat([master_table, df_t], ignore_index=True, sort=False)

    print(f"Ingested {len(count_files)} 'counts.csv' files from TAR.")
    return master_table


# =============================
# Main pipeline
# =============================

def main():
    cfg = default_paths()

    parser = argparse.ArgumentParser(
        description="Aggregate counts-only predictions and compare against Rutten exports, with source toggle (auto-appends SAS for TAR URLs)."
    )
    parser.add_argument(
        "--source",
        choices=["blob", "tar-url"],
        default="blob",
        help="Where to ingest counts from: 'blob' (Azure) or 'tar-url' (download & unpack first).",
    )
    parser.add_argument(
        "--tar-url",
        default="https://citprodflowcytosa.blob.core.windows.net/mnceacyzfilesforthomasrutten/model_trained_on_nn_cleaned_94_pct_but_may_not_be_used.tar",
        help="TAR URL that contains a 'blob_tool_outputs' folder (signed or unsigned)."
    )
    parser.add_argument(
        "--downloads-dir",
        default=str(Path.home() / "Downloads"),
        help="Where to save the TAR and extracted contents (default: your Downloads folder)."
    )

    # Outputs & creds
    parser.add_argument("--out-csv", default=cfg["output_master_table_csv"]) 
    parser.add_argument("--out-ts-html", default=cfg["output_timeseries_html"]) 
    parser.add_argument("--rutten-thames", default=cfg["rutten_thames_xlsx"]) 
    parser.add_argument("--rutten-mersey", default=cfg["rutten_mersey_xlsx"]) 
    parser.add_argument("--sas-token-path", default=cfg["sas_token_path"], help="Path to file containing SAS token (will be appended to TAR URL if missing)") 
    parser.add_argument("--account-url", default=cfg["account_url"]) 
    parser.add_argument("--container-name", default=cfg["container_name"]) 
    args = parser.parse_args()

    # Ingest
    if args.source == "blob":
        sas_token = get_sas_token(args.sas_token_path)
        master_table = ingest_from_blob(args.account_url, args.container_name, sas_token)
    else:
        downloads_dir = Path(args.downloads_dir)
        master_table = ingest_from_tar_url(args.tar_url, downloads_dir, args.sas_token_path)

    if master_table.empty:
        print("No rows ingested. Exiting.")
        sys.exit(0)

    # Ordering & tidy
    DEST_ORDER = load_destination_order(DEST_TXT)

    if not master_table.empty:
        present = [lab for lab in DEST_ORDER if lab in set(master_table['sample_id'])]
        extras = [lab for lab in master_table['sample_id'].drop_duplicates().tolist() if lab not in DEST_ORDER]
        ordered_categories = present + extras
        master_table['sample_id'] = pd.Categorical(master_table['sample_id'], categories=ordered_categories, ordered=True)
        master_table['order_index'] = master_table['sample_id'].cat.codes

        master_table['start_time'] = pd.to_datetime(master_table['start_time'], errors='coerce', utc=True)
        master_table = master_table.sort_values(by=['order_index', 'start_time'], kind='mergesort')

        cols_to_drop = [
            "Other", "RedNano", "nophytoplankton", "noiseum", "no_phytoplankton",
            "YB_um_beads", "Plant_detritus", "OraNano_crypto", "RWS_um_beads",
            "Beads_rest", "RedMicro", "Bubbles", "OraNano", "C_undetermined",
            "start_time", "order_index"
        ]
        cols_safe_to_drop = []
        for col in cols_to_drop:
            if col not in master_table.columns:
                continue
            numeric_series = pd.to_numeric(master_table[col], errors="coerce")
            if numeric_series.notna().sum() == 0 or numeric_series.sum() == 0:
                cols_safe_to_drop.append(col)
        master_table = master_table.drop(columns=cols_safe_to_drop, errors="ignore")

    # Save master table
    master_table.to_csv(args.out_csv, index=False)
    print(f"Saved counts-only master table to: {args.out_csv}")

    # Time series plot
    ts = master_table.copy()
    ts['start_time'] = pd.to_datetime(ts['start_time'], errors='coerce', utc=True)
    ts = ts.dropna(subset=['start_time'])
    if not ts.empty:
        fig = px.scatter(
            ts, x='start_time', y=class_columns,
            labels={'value': 'Counts', 'start_time': 'Time'},
            title='Counts per Class Over Time',
            category_orders={'sample_id': ordered_categories} if 'ordered_categories' in locals() else None
        )
        fig.update_layout(
            xaxis_title='Time', yaxis_title='Counts', legend_title='Class',
            xaxis=dict(tickangle=45), template='plotly_white'
        )
        fig.write_html(args.out_ts_html)
        try:
            fig.show()
        except Exception:
            pass
        print(f"Saved timeseries scatter to: {args.out_ts_html}")
    else:
        print("No valid timestamps found — timeseries plot skipped.")

    # Rutten Excel joins
    def load_rutten_file(path):
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        if "fname" not in df.columns:
            raise ValueError(f"'fname' column not found in {path}")
        return df

    try:
        rutten_1 = load_rutten_file(args.rutten_thames)
        rutten_2 = load_rutten_file(args.rutten_mersey)
        ruttenxlsx = pd.concat([rutten_1, rutten_2], ignore_index=True)
        print("Rutten combined dataframe shape:", ruttenxlsx.shape)

        merge_key_master = "sample_id"
        merge_key_rutten = "fname"
        ruttenxlsx_renamed = ruttenxlsx.rename(columns={merge_key_rutten: merge_key_master})

        joined = master_table.merge(ruttenxlsx_renamed, on=merge_key_master, suffixes=("_master", "_rutten"))
        print("Matched rows:", joined.shape)

        count_cols_master = [c for c in master_table.columns if c.startswith("Counts_")]
        count_cols_rutten = [c for c in ruttenxlsx.columns if c.startswith("Counts_")]
        shared = sorted(set(count_cols_master).intersection(set(count_cols_rutten)))
        print("Shared count columns:", shared)

        if len(shared) > 0:
            import math
            n = len(shared)
            ncols, nrows = 4, math.ceil(n / 4)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols+1, 3.5*nrows+1), squeeze=False)
            axes = axes.flatten()

            for ax, col in zip(axes, shared):
                mcol, rcol = col + "_master", col + "_rutten"
                if mcol not in joined or rcol not in joined:
                    ax.set_visible(False); continue
                ax.scatter(joined[mcol], joined[rcol], alpha=0.7, s=18, edgecolor="none")
                ax.set_xlabel(f"{col} (master)"); ax.set_ylabel(f"{col} (Rutten)"); ax.set_title(col, fontsize=10)

                x_max = joined[mcol].max(skipna=True)
                y_max = joined[rcol].max(skipna=True)
                try:
                    candidates = [v for v in [x_max, y_max] if pd.notna(v)]
                    local_max = float(max(candidates)) if candidates else None
                except Exception:
                    local_max = None
                if local_max is not None and local_max > 0:
                    ax.plot([0, local_max], [0, local_max], "r--", linewidth=1)
                    pad = local_max * 1.05
                    ax.set_xlim(0, pad); ax.set_ylim(0, pad)
                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

            for k in range(len(shared), len(axes)):
                axes[k].set_visible(False)

            out_png = str(Path(args.out_csv).with_name("counts_comparison_subplots.png"))
            out_pdf = str(Path(args.out_csv).with_name("counts_comparison_subplots.pdf"))
            fig.suptitle("Master vs Rutten: Counts Comparison", fontsize=14, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(out_png, dpi=200)
            fig.savefig(out_pdf)
            plt.close(fig)
            print(f"Saved combined subplot figure to: {out_png} and {out_pdf}")
        else:
            print("No shared 'Counts_' columns found between master and Rutten—nothing to plot.")
    except FileNotFoundError as e:
        print(f"Rutten Excel files not found: {e}. Skipping comparison plots.")

if __name__ == "__main__":
    main()
