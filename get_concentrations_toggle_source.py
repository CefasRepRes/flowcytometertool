# -*- coding: utf-8 -*-
"""
Workflow to aggregate counts-only predictions and compare against Rutten Excel exports,
with a toggle to ingest either from Azure Blob (existing) or from a downloaded TAR
that contains a 'blob_tool_outputs' folder.

Adapted from: get_concentrations_from_job_2026_to_match_trutten.py
"""

# -----------------------------
# Standard libs
# -----------------------------
import io
import os
import sys
import tarfile
import argparse
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
# SAS token helper (unchanged)
# -----------------------------
def get_sas_token(file_path):
    """Read a SAS token from a local file and return it as a trimmed string."""
    with open(file_path, 'r') as file:
        return file.read().strip()

# -----------------------------
# Config (moved into a function so we can reuse)
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
# Your existing constants (from original)
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
# FNAME_LOOKUP (verbatim from original)
# -----------------------------
FNAME_LOOKUP = {
 '12-08-2025_LV01_nano 2025-08-15 07h44.cyz_predictions.csv_counts.csv': '12-08-2025_LV01_nano 2025-08-15 07h44',
 '12-08-2025_LV01_pico 2025-08-15 07h55.cyz_predictions.csv_counts.csv': '12-08-2025_LV01_pico 2025-08-15 07h55',
 '12-08-2025_LV06_nano 2025-08-15 08h12.cyz_predictions.csv_counts.csv': '12-08-2025_LV06_nano 2025-08-15 08h12',
 '12-08-2025_LV06_pico 2025-08-15 08h06.cyz_predictions.csv_counts.csv': '12-08-2025_LV06_pico 2025-08-15 08h06',
 '12-08-2025_LV07_nano 2025-08-15 08h36.cyz_predictions.csv_counts.csv': '12-08-2025_LV07_nano 2025-08-15 08h36',
 '12-08-2025_LV07_pico 2025-08-15 08h31.cyz_predictions.csv_counts.csv': '12-08-2025_LV07_pico 2025-08-15 08h31',
 '12-08-2025_LV08_nano 2025-08-15 09h24.cyz_predictions.csv_counts.csv': '12-08-2025_LV08_nano 2025-08-15 09h24',
 '12-08-2025_LV08_pico 2025-08-15 09h18.cyz_predictions.csv_counts.csv': '12-08-2025_LV08_pico 2025-08-15 09h18',
 '12-08-2025_LV10_nano 2025-08-15 09h44.cyz_predictions.csv_counts.csv': '12-08-2025_LV10_nano 2025-08-15 09h44',
 '12-08-2025_LV10_pico 2025-08-15 09h39.cyz_predictions.csv_counts.csv': '12-08-2025_LV10_pico 2025-08-15 09h39',
 '12-08-2025_LV16_nano 2025-08-15 10h07.cyz_predictions.csv_counts.csv': '12-08-2025_LV16_nano 2025-08-15 10h07',
 '12-08-2025_LV16_pico 2025-08-15 10h02.cyz_predictions.csv_counts.csv': '12-08-2025_LV16_pico 2025-08-15 10h02',
 '12-08-2025_LV20_nano 2025-08-15 10h33.cyz_predictions.csv_counts.csv': '12-08-2025_LV20_nano 2025-08-15 10h33',
 '12-08-2025_LV20_pico 2025-08-15 10h23.cyz_predictions.csv_counts.csv': '12-08-2025_LV20_pico 2025-08-15 10h23',
 '12-08-2025_LV22_nano 2025-08-15 10h49.cyz_predictions.csv_counts.csv': '12-08-2025_LV22_nano 2025-08-15 10h49',
 '12-08-2025_LV22_pico 2025-08-15 11h00.cyz_predictions.csv_counts.csv': '12-08-2025_LV22_pico 2025-08-15 11h00',
 '12-08-2025_LV23_nano 2025-08-15 11h08.cyz_predictions.csv_counts.csv': '12-08-2025_LV23_nano 2025-08-15 11h08',
 '12-08-2025_LV23_pico 2025-08-15 11h18.cyz_predictions.csv_counts.csv': '12-08-2025_LV23_pico 2025-08-15 11h18',
 '12-08-2025_LVSB_nano 2025-08-15 11h27.cyz_predictions.csv_counts.csv': '12-08-2025_LVSB_nano 2025-08-15 11h27',
 '12-08-2025_LVSB_pico 2025-08-15 11h37.cyz_predictions.csv_counts.csv': '12-08-2025_LVSB_pico 2025-08-15 11h37',
 '16-08-2025_MA1_nano 2025-08-19 11h09.cyz_predictions.csv_counts.csv': '16-08-2025_MA1_nano 2025-08-19 11h09',
 '16-08-2025_MA1_pico 2025-08-19 11h20.cyz_predictions.csv_counts.csv': '16-08-2025_MA1_pico 2025-08-19 11h20',
 '16-08-2025_MA2_nano 2025-08-19 09h55.cyz_predictions.csv_counts.csv': '16-08-2025_MA2_nano 2025-08-19 09h55',
 '16-08-2025_MA2_pico 2025-08-19 09h50.cyz_predictions.csv_counts.csv': '16-08-2025_MA2_pico 2025-08-19 09h50',
 '16-08-2025_MA3_nano 2025-08-19 08h45.cyz_predictions.csv_counts.csv': '16-08-2025_MA3_nano 2025-08-19 08h45',
 '16-08-2025_MA3_pico 2025-08-19 08h39.cyz_predictions.csv_counts.csv': '16-08-2025_MA3_pico 2025-08-19 08h39',
 '16-08-2025_MA4_nano 2025-08-19 08h11.cyz_predictions.csv_counts.csv': '16-08-2025_MA4_nano 2025-08-19 08h11',
 '16-08-2025_MA4_pico 2025-08-19 08h05.cyz_predictions.csv_counts.csv': '16-08-2025_MA4_pico 2025-08-19 08h05',
 '16-08-2025_NrGravesend_nano 2025-08-19 07h45.cyz_predictions.csv_counts.csv': '16-08-2025_NrGravesend_nano 2025-08-19 07h45',
 '16-08-2025_NrGravesend_pico 2025-08-19 07h39.cyz_predictions.csv_counts.csv': '16-08-2025_NrGravesend_pico 2025-08-19 07h39',
 'Mersey_LV017c_3 2024-12-11 11h35.cyz_predictions.csv_counts.csv': 'Mersey_LV017c_3 2024-12-11 11h35',
 'Mersey_LV017c_9 2024-12-11 11h25.cyz_predictions.csv_counts.csv': 'Mersey_LV017c_9 2024-12-11 11h25',
 'Mersey_LV01_3 2023-06-16 13h22.cyz_predictions.csv_counts.csv': 'Mersey_LV01_3 2023-06-16 13h22',
 'Mersey_LV01_3 2024-07-18 12h09.cyz_predictions.csv_counts.csv': 'Mersey_LV01_3 2024-07-18 12h09',
 'Mersey_LV01_3.42 2024-02-21 12h42.cyz_predictions.csv_counts.csv': 'Mersey_LV01_3.42 2024-02-21 12h42',
 'Mersey_LV01_9 2023-06-16 13h08.cyz_predictions.csv_counts.csv': 'Mersey_LV01_9 2023-06-16 13h08',
 'Mersey_LV01_9 2024-02-21 12h27.cyz_predictions.csv_counts.csv': 'Mersey_LV01_9 2024-02-21 12h27',
 'Mersey_LV01_9 2024-07-18 12h23.cyz_predictions.csv_counts.csv': 'Mersey_LV01_9 2024-07-18 12h23',
 'Mersey_LV068b_3 2024-12-11 12h04.cyz_predictions.csv_counts.csv': 'Mersey_LV068b_3 2024-12-11 12h04',
 'Mersey_LV068b_9 2024-12-11 11h53.cyz_predictions.csv_counts.csv': 'Mersey_LV068b_9 2024-12-11 11h53',
 'Mersey_LV06_3 2023-06-16 13h47.cyz_predictions.csv_counts.csv': 'Mersey_LV06_3 2023-06-16 13h47',
 'Mersey_LV06_3 2024-07-18 12h36.cyz_predictions.csv_counts.csv': 'Mersey_LV06_3 2024-07-18 12h36',
 'Mersey_LV06_3.42 2024-02-21 09h12.cyz_predictions.csv_counts.csv': 'Mersey_LV06_3.42 2024-02-21 09h12',
 'Mersey_LV06_9 2023-06-16 13h33.cyz_predictions.csv_counts.csv': 'Mersey_LV06_9 2023-06-16 13h33',
 'Mersey_LV06_9 2024-02-21 08h58.cyz_predictions.csv_counts.csv': 'Mersey_LV06_9 2024-02-21 08h58',
 'Mersey_LV06_9 2024-07-18 12h50.cyz_predictions.csv_counts.csv': 'Mersey_LV06_9 2024-07-18 12h50',
 'Mersey_LV079c_3 2024-12-11 12h28.cyz_predictions.csv_counts.csv': 'Mersey_LV079c_3 2024-12-11 12h28',
 'Mersey_LV079c_9 2024-12-11 12h15.cyz_predictions.csv_counts.csv': 'Mersey_LV079c_9 2024-12-11 12h15',
 'Mersey_LV07_3 2023-06-16 14h14.cyz_predictions.csv_counts.csv': 'Mersey_LV07_3 2023-06-16 14h14',
 'Mersey_LV07_3 2024-07-18 13h08.cyz_predictions.csv_counts.csv': 'Mersey_LV07_3 2024-07-18 13h08',
 'Mersey_LV07_3.42 2024-02-21 13h13.cyz_predictions.csv_counts.csv': 'Mersey_LV07_3.42 2024-02-21 13h13',
 'Mersey_LV07_9 2024-02-21 12h59.cyz_predictions.csv_counts.csv': 'Mersey_LV07_9 2024-02-21 12h59',
 'Mersey_LV07_9 2024-07-18 13h22.cyz_predictions.csv_counts.csv': 'Mersey_LV07_9 2024-07-18 13h22',
 'Mersey_LV0810a_3 2024-12-11 12h49.cyz_predictions.csv_counts.csv': 'Mersey_LV0810a_3 2024-12-11 12h49',
 'Mersey_LV0810a_9 2024-12-11 12h36.cyz_predictions.csv_counts.csv': 'Mersey_LV0810a_9 2024-12-11 12h36',
 'Mersey_LV08_3 2023-06-16 14h42.cyz_predictions.csv_counts.csv': 'Mersey_LV08_3 2023-06-16 14h42',
 'Mersey_LV08_3 2024-07-18 13h34.cyz_predictions.csv_counts.csv': 'Mersey_LV08_3 2024-07-18 13h34',
 'Mersey_LV08_3.42 2024-02-21 13h44.cyz_predictions.csv_counts.csv': 'Mersey_LV08_3.42 2024-02-21 13h44',
 'Mersey_LV08_9 2024-02-21 13h30.cyz_predictions.csv_counts.csv': 'Mersey_LV08_9 2024-02-21 13h30',
 'Mersey_LV08_9 2024-07-18 13h48.cyz_predictions.csv_counts.csv': 'Mersey_LV08_9 2024-07-18 13h48',
 'Mersey_LV106c_3 2024-12-11 13h10.cyz_predictions.csv_counts.csv': 'Mersey_LV106c_3 2024-12-11 13h10',
 'Mersey_LV106c_9 2024-12-11 12h57.cyz_predictions.csv_counts.csv': 'Mersey_LV106c_9 2024-12-11 12h57',
 'Mersey_LV10_3 2023-06-16 12h51.cyz_predictions.csv_counts.csv': 'Mersey_LV10_3 2023-06-16 12h51',
 'Mersey_LV10_3 2024-07-18 11h34.cyz_predictions.csv_counts.csv': 'Mersey_LV10_3 2024-07-18 11h34',
 'Mersey_LV10_3.42 2024-02-21 11h58.cyz_predictions.csv_counts.csv': 'Mersey_LV10_3.42 2024-02-21 11h58',
 'Mersey_LV10_9 2024-02-21 11h44.cyz_predictions.csv_counts.csv': 'Mersey_LV10_9 2024-02-21 11h44',
 'Mersey_LV10_9 2024-07-18 11h48.cyz_predictions.csv_counts.csv': 'Mersey_LV10_9 2024-07-18 11h48',
 'Mersey_LV162C_15 2024-12-12 07h37.cyz_predictions.csv_counts.csv': 'Mersey_LV162C_15 2024-12-12 07h37',
 'Mersey_LV162c_3 2024-12-11 13h38.cyz_predictions.csv_counts.csv': 'Mersey_LV162c_3 2024-12-11 13h38',
 'Mersey_LV162c_9 2024-12-11 13h25.cyz_predictions.csv_counts.csv': 'Mersey_LV162c_9 2024-12-11 13h25',
 'Mersey_LV16_3 2024-07-18 09h32.cyz_predictions.csv_counts.csv': 'Mersey_LV16_3 2024-07-18 09h32',
 'Mersey_LV16_3.42 2024-02-21 10h00.cyz_predictions.csv_counts.csv': 'Mersey_LV16_3.42 2024-02-21 10h00',
 'Mersey_LV16_9 2024-02-21 09h47.cyz_predictions.csv_counts.csv': 'Mersey_LV16_9 2024-02-21 09h47',
 'Mersey_LV16_9 2024-07-18 09h46.cyz_predictions.csv_counts.csv': 'Mersey_LV16_9 2024-07-18 09h46',
 'Mersey_LV201a_3 2024-12-11 13h58.cyz_predictions.csv_counts.csv': 'Mersey_LV201a_3 2024-12-11 13h58',
 'Mersey_LV201a_9 2024-12-11 13h45.cyz_predictions.csv_counts.csv': 'Mersey_LV201a_9 2024-12-11 13h45',
 'Mersey_LV20_3 2024-07-18 09h02.cyz_predictions.csv_counts.csv': 'Mersey_LV20_3 2024-07-18 09h02',
 'Mersey_LV20_3.42 2024-02-21 09h38.cyz_predictions.csv_counts.csv': 'Mersey_LV20_3.42 2024-02-21 09h38',
 'Mersey_LV20_9 2024-02-21 09h24.cyz_predictions.csv_counts.csv': 'Mersey_LV20_9 2024-02-21 09h24',
 'Mersey_LV20_9 2024-07-18 09h16.cyz_predictions.csv_counts.csv': 'Mersey_LV20_9 2024-07-18 09h16',
 'Mersey_LV225b_3 2024-12-11 14h18.cyz_predictions.csv_counts.csv': 'Mersey_LV225b_3 2024-12-11 14h18',
 'Mersey_LV225b_9 2024-12-11 14h05.cyz_predictions.csv_counts.csv': 'Mersey_LV225b_9 2024-12-11 14h05',
 'Mersey_LV22_3 2024-07-18 11h06.cyz_predictions.csv_counts.csv': 'Mersey_LV22_3 2024-07-18 11h06',
 'Mersey_LV22_3.42 2024-02-21 11h29.cyz_predictions.csv_counts.csv': 'Mersey_LV22_3.42 2024-02-21 11h29',
 'Mersey_LV22_9 2024-02-21 11h16.cyz_predictions.csv_counts.csv': 'Mersey_LV22_9 2024-02-21 11h16',
 'Mersey_LV22_9 2024-07-18 11h20.cyz_predictions.csv_counts.csv': 'Mersey_LV22_9 2024-07-18 11h20',
 'Mersey_LV233b_3 2024-12-11 14h48.cyz_predictions.csv_counts.csv': 'Mersey_LV233b_3 2024-12-11 14h48',
 'Mersey_LV233b_9 2024-12-11 14h35.cyz_predictions.csv_counts.csv': 'Mersey_LV233b_9 2024-12-11 14h35',
 'Mersey_LV23_3 2024-07-18 10h08.cyz_predictions.csv_counts.csv': 'Mersey_LV23_3 2024-07-18 10h08',
 'Mersey_LV23_3.42 2024-02-21 10h40.cyz_predictions.csv_counts.csv': 'Mersey_LV23_3.42 2024-02-21 10h40',
 'Mersey_LV23_9 2024-02-21 10h26.cyz_predictions.csv_counts.csv': 'Mersey_LV23_9 2024-02-21 10h26',
 'Mersey_LV23_9 2024-07-18 10h22.cyz_predictions.csv_counts.csv': 'Mersey_LV23_9 2024-07-18 10h22',
 'Mersey_LVSB4b_3 2024-12-11 15h08.cyz_predictions.csv_counts.csv': 'Mersey_LVSB4b_3 2024-12-11 15h08',
 'Mersey_LVSB4b_9 2024-12-11 14h55.cyz_predictions.csv_counts.csv': 'Mersey_LVSB4b_9 2024-12-11 14h55',
 'Mersey_LVSB_3 2023-06-16 11h40.cyz_predictions.csv_counts.csv': 'Mersey_LVSB_3 2023-06-16 11h40',
 'Mersey_LVSB_3 2024-07-18 10h36.cyz_predictions.csv_counts.csv': 'Mersey_LVSB_3 2024-07-18 10h36',
 'Mersey_LVSB_3.42 2024-02-21 11h05.cyz_predictions.csv_counts.csv': 'Mersey_LVSB_3.42 2024-02-21 11h05',
 'Mersey_LVSB_9 2023-06-16 11h26.cyz_predictions.csv_counts.csv': 'Mersey_LVSB_9 2023-06-16 11h26',
 'Mersey_LVSB_9 2024-02-21 10h51.cyz_predictions.csv_counts.csv': 'Mersey_LVSB_9 2024-02-21 10h51',
 'Mersey_LVSB_9 2024-07-18 10h50.cyz_predictions.csv_counts.csv': 'Mersey_LVSB_9 2024-07-18 10h50',
 'Thames_EWarp_3 2023-06-19 08h51.cyz_predictions.csv_counts.csv': 'Thames_EWarp_3 2023-06-19 08h51',
 'Thames_EWarp_3.42 2024-02-12 10h26.cyz_predictions.csv_counts.csv': 'Thames_EWarp_3.42 2024-02-12 10h26',
 'Thames_EWarp_3.57 2023-07-18 09h13.cyz_predictions.csv_counts.csv': 'Thames_EWarp_3.57 2023-07-18 09h13',
 'Thames_EWarp_9 2023-06-19 08h37.cyz_predictions.csv_counts.csv': 'Thames_EWarp_9 2023-06-19 08h37',
 'Thames_EWarp_9 2023-07-18 08h59.cyz_predictions.csv_counts.csv': 'Thames_EWarp_9 2023-07-18 08h59',
 'Thames_EWarp_9 2024-02-12 10h20.cyz_predictions.csv_counts.csv': 'Thames_EWarp_9 2024-02-12 10h20',
 'Thames_EastOfWarp_3.42 2024-01-08 08h50.cyz_predictions.csv_counts.csv': 'Thames_EastOfWarp_3.42 2024-01-08 08h50',
 'Thames_EastOfWarp_9 2024-01-08 08h43.cyz_predictions.csv_counts.csv': 'Thames_EastOfWarp_9 2024-01-08 08h43',
 'Thames_EastofWarp_3 2024-07-15 09h40.cyz_predictions.csv_counts.csv': 'Thames_EastofWarp_3 2024-07-15 09h40',
 'Thames_EastofWarp_9 2024-07-15 09h54.cyz_predictions.csv_counts.csv': 'Thames_EastofWarp_9 2024-07-15 09h54',
 'Thames_Gravesend_3 2023-06-19 13h51.cyz_predictions.csv_counts.csv': 'Thames_Gravesend_3 2023-06-19 13h51',
 'Thames_Gravesend_3 2024-07-15 12h15.cyz_predictions.csv_counts.csv': 'Thames_Gravesend_3 2024-07-15 12h15',
 'Thames_Gravesend_3.42 2024-01-08 10h45.cyz_predictions.csv_counts.csv': 'Thames_Gravesend_3.42 2024-01-08 10h45',
 'Thames_Gravesend_3.57 2023-07-18 13h47.cyz_predictions.csv_counts.csv': 'Thames_Gravesend_3.57 2023-07-18 13h47',
 'Thames_Gravesend_9 2023-06-19 13h37.cyz_predictions.csv_counts.csv': 'Thames_Gravesend_9 2023-06-19 13h37',
 'Thames_Gravesend_9 2023-07-18 13h32.cyz_predictions.csv_counts.csv': 'Thames_Gravesend_9 2023-07-18 13h32',
 'Thames_Gravesend_9 2024-01-08 10h38.cyz_predictions.csv_counts.csv': 'Thames_Gravesend_9 2024-01-08 10h38',
 'Thames_Gravesend_9 2024-07-15 12h29.cyz_predictions.csv_counts.csv': 'Thames_Gravesend_9 2024-07-15 12h29',
 'Thames_MA1_3 2023-06-19 09h22.cyz_predictions.csv_counts.csv': 'Thames_MA1_3 2023-06-19 09h22',
 'Thames_MA1_3 2024-07-15 12h50.cyz_predictions.csv_counts.csv': 'Thames_MA1_3 2024-07-15 12h50',
 'Thames_MA1_3.42 2024-01-08 09h29.cyz_predictions.csv_counts.csv': 'Thames_MA1_3.42 2024-01-08 09h29',
 'Thames_MA1_3.57 2023-07-18 08h36.cyz_predictions.csv_counts.csv': 'Thames_MA1_3.57 2023-07-18 08h36',
 'Thames_MA1_9 2023-06-19 09h08.cyz_predictions.csv_counts.csv': 'Thames_MA1_9 2023-06-19 09h08',
 'Thames_MA1_9 2023-07-18 08h22.cyz_predictions.csv_counts.csv': 'Thames_MA1_9 2023-07-18 08h22',
 'Thames_MA1_9 2024-01-08 09h22.cyz_predictions.csv_counts.csv': 'Thames_MA1_9 2024-01-08 09h22',
 'Thames_MA1_9 2024-07-15 13h04.cyz_predictions.csv_counts.csv': 'Thames_MA1_9 2024-07-15 13h04',
 'Thames_MA2_3 2023-06-19 11h25.cyz_predictions.csv_counts.csv': 'Thames_MA2_3 2023-06-19 11h25',
 'Thames_MA2_3 2024-07-15 09h06.cyz_predictions.csv_counts.csv': 'Thames_MA2_3 2024-07-15 09h06',
 'Thames_MA2_3.42 2024-01-08 10h07.cyz_predictions.csv_counts.csv': 'Thames_MA2_3.42 2024-01-08 10h07',
 'Thames_MA2_3.57 2023-07-18 12h16.cyz_predictions.csv_counts.csv': 'Thames_MA2_3.57 2023-07-18 12h16',
 'Thames_MA2_9 2023-06-19 11h11.cyz_predictions.csv_counts.csv': 'Thames_MA2_9 2023-06-19 11h11',
 'Thames_MA2_9 2023-07-18 12h01.cyz_predictions.csv_counts.csv': 'Thames_MA2_9 2023-07-18 12h01',
 'Thames_MA2_9 2024-01-08 10h00.cyz_predictions.csv_counts.csv': 'Thames_MA2_9 2024-01-08 10h00',
 'Thames_MA2_9 2024-07-15 09h20.cyz_predictions.csv_counts.csv': 'Thames_MA2_9 2024-07-15 09h20',
 'Thames_MA3_3 2023-06-19 11h58.cyz_predictions.csv_counts.csv': 'Thames_MA3_3 2023-06-19 11h58',
 'Thames_MA3_3 2024-07-15 10h47.cyz_predictions.csv_counts.csv': 'Thames_MA3_3 2024-07-15 10h47',
 'Thames_MA3_3.42 2024-01-08 10h26.cyz_predictions.csv_counts.csv': 'Thames_MA3_3.42 2024-01-08 10h26',
 'Thames_MA3_9 2023-06-19 11h44.cyz_predictions.csv_counts.csv': 'Thames_MA3_9 2023-06-19 11h44',
 'Thames_MA3_9 2023-07-18 10h22.cyz_predictions.csv_counts.csv': 'Thames_MA3_9 2023-07-18 10h22',
 'Thames_MA3_9 2024-01-08 10h19.cyz_predictions.csv_counts.csv': 'Thames_MA3_9 2024-01-08 10h19',
 'Thames_MA3_9 2024-07-15 11h02.cyz_predictions.csv_counts.csv': 'Thames_MA3_9 2024-07-15 11h02',
 'Thames_MA4_3 2023-06-19 13h25.cyz_predictions.csv_counts.csv': 'Thames_MA4_3 2023-06-19 13h25',
 'Thames_MA4_3 2024-07-15 11h18.cyz_predictions.csv_counts.csv': 'Thames_MA4_3 2024-07-15 11h18',
 'Thames_MA4_3.42 2024-01-08 11h04.cyz_predictions.csv_counts.csv': 'Thames_MA4_3.42 2024-01-08 11h04',
 'Thames_MA4_3.57 2023-07-18 12h53.cyz_predictions.csv_counts.csv': 'Thames_MA4_3.57 2023-07-18 12h53',
 'Thames_MA4_9 2023-06-19 13h10.cyz_predictions.csv_counts.csv': 'Thames_MA4_9 2023-06-19 13h10',
 'Thames_MA4_9 2023-07-18 12h39.cyz_predictions.csv_counts.csv': 'Thames_MA4_9 2023-07-18 12h39',
 'Thames_MA4_9 2024-01-08 10h57.cyz_predictions.csv_counts.csv': 'Thames_MA4_9 2024-01-08 10h57',
 'Thames_MA4_9 2024-07-15 11h32.cyz_predictions.csv_counts.csv': 'Thames_MA4_9 2024-07-15 11h32',
 'Thames_WarpSB_3 2023-06-19 10h35.cyz_predictions.csv_counts.csv': 'Thames_WarpSB_3 2023-06-19 10h35',
 'Thames_WarpSB_3 2024-07-15 10h18.cyz_predictions.csv_counts.csv': 'Thames_WarpSB_3 2024-07-15 10h18',
 'Thames_WarpSB_3.42 2024-01-08 09h11.cyz_predictions.csv_counts.csv': 'Thames_WarpSB_3.42 2024-01-08 09h11',
 'Thames_WarpSB_3.42 2024-02-12 11h02.cyz_predictions.csv_counts.csv': 'Thames_WarpSB_3.42 2024-02-12 11h02',
 'Thames_WarpSB_9 2023-06-19 10h21.cyz_predictions.csv_counts.csv': 'Thames_WarpSB_9 2023-06-19 10h21',
 'Thames_WarpSB_9 2023-07-18 11h16.cyz_predictions.csv_counts.csv': 'Thames_WarpSB_9 2023-07-18 11h16',
 'Thames_WarpSB_9 2024-01-08 09h03.cyz_predictions.csv_counts.csv': 'Thames_WarpSB_9 2024-01-08 09h03',
 'Thames_WarpSB_9 2024-02-12 10h56.cyz_predictions.csv_counts.csv': 'Thames_WarpSB_9 2024-02-12 10h56',
 'Thames_WarpSB_9 2024-07-15 10h32.cyz_predictions.csv_counts.csv': 'Thames_WarpSB_9 2024-07-15 10h32',
 'Thames_flush_3.57 2023-07-18 15h10.cyz_predictions.csv_counts.csv': 'Thames_flush_3.57 2023-07-18 15h10',
 'Thames_flush_9 2023-07-18 14h44.cyz_predictions.csv_counts.csv': 'Thames_flush_9 2023-07-18 14h44',
 'nano_Thamea_NRG 2024-08-12 09h53.cyz_predictions.csv_counts.csv': 'nano_Thamea_NRG 2024-08-12 09h53',
 'pico_Thames_NRG 2024-08-12 10h04.cyz_predictions.csv_counts.csv': 'pico_Thames_NRG 2024-08-12 10h04',
 'pico_Thames_warp 2024-08-12 11h25.cyz_predictions.csv_counts.csv': 'pico_Thames_warp 2024-08-12 11h25'
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
# Ingestion helpers (NEW)
# =============================

def _canonicalize_counts_dataframe(df_raw: pd.DataFrame, raw_name: str) -> pd.DataFrame:
    # Transpose into row-wise record
    df_t = df_raw.T.reset_index()
    df_t.columns = df_t.iloc[0]
    df_t = df_t[1:]

    # Build canonical counts frame
    counts_out = pd.DataFrame({col: np.nan for col in TO_COLUMNS}, index=df_t.index)

    for col in df_t.columns:
        ncol = _normalize_key(col)
        # harmonize near-variants
        if ncol in {"yb_um_beads", "yb_1um_beads", "1_um_beads"}:
            ncol = "1um_beads"
        if ncol in {"rws_um_beads", "rws3um_beads", "rws_3_um_beads"}:
            ncol = "rws_3um_beads"
        target = norm_map.get(ncol)
        if target is None:
            continue
        counts_out[target] = pd.to_numeric(df_t[col], errors="coerce")

    # Keep metadata / non-count columns
    non_count_cols = [c for c in df_t.columns if _normalize_key(c) not in norm_map]
    df_t = pd.concat([df_t[non_count_cols], counts_out], axis=1)

    # add filename + sample_id
    df_t['filename'] = raw_name
    from_lookup = FNAME_LOOKUP.get(raw_name, None)
    df_t['sample_id'] = from_lookup if from_lookup is not None else raw_name

    # ensure class columns exist & numeric
    for col in class_columns:
        if col not in df_t.columns:
            df_t[col] = np.nan
        df_t[col] = pd.to_numeric(df_t[col], errors='coerce')

    # parse a timestamp if available
    time_candidates = ['start_time', 'Start', 'start', 'timestamp', 'time']
    parsed_time = None
    for cand in time_candidates:
        if cand in df_t.columns:
            parsed_time = pd.to_datetime(df_t[cand], errors='coerce', utc=True)
            break
    df_t['start_time'] = parsed_time if parsed_time is not None else pd.NaT

    return df_t


def ingest_from_blob(account_url: str, container_name: str, sas_token: str) -> pd.DataFrame:
    """Existing path: list blobs, pull *counts.csv, build master_table."""
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
    """Extract tar(.gz) into extract_dir and return the root path."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if str(tar_path).endswith(".tar.gz") else "r:"
    with tarfile.open(tar_path, mode) as tf:
        tf.extractall(path=extract_dir)
    return extract_dir


def ingest_from_tar_url(tar_url: str, downloads_dir: Path) -> pd.DataFrame:
    """
    NEW: Download a .tar (.tar or .tar.gz) into Downloads, extract,
    then read CSVs under **blob_tool_outputs/** exactly like the blob path.
    """
    import urllib.request  # local-only download, no extra deps

    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Local file targets
    tar_name = os.path.basename(tar_url.split('?')[0]) or "archive.tar"
    tar_path = downloads_dir / tar_name
    extract_root = downloads_dir / tar_name.replace(".tar.gz", "").replace(".tar", "")  # folder name
    blob_outputs = None

    # Download (skip if already present)
    if not tar_path.exists():
        print(f"Downloading TAR from {tar_url} -> {tar_path}")
        urllib.request.urlretrieve(tar_url, tar_path)
    else:
        print(f"Using existing TAR at {tar_path}")

    # Extract
    print(f"Extracting {tar_path} -> {extract_root}")
    _extract_tar(tar_path, extract_root)

    # Find blob_tool_outputs folder
    candidates = [
        extract_root / "blob_tool_outputs",
    ]
    if extract_root.exists():
        for p in extract_root.iterdir():
            if p.is_dir():
                candidates.append(p / "blob_tool_outputs")

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            blob_outputs = cand
            break

    if blob_outputs is None:
        raise FileNotFoundError(
            f"'blob_tool_outputs' not found under {extract_root}. "
            "Ensure the TAR contains that folder."
        )

    # Ingest all *counts.csv from blob_tool_outputs
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
        description="Aggregate counts-only predictions and compare against Rutten exports, with source toggle."
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
        help="TAR URL that contains a 'blob_tool_outputs' folder (used when --source=tar-url)."
    )
    parser.add_argument(
        "--downloads-dir",
        default=str(Path.home() / "Downloads"),
        help="Where to save the TAR and extracted contents (default: your Downloads folder)."
    )

    # Preserve your file outputs as overridable
    parser.add_argument("--out-csv", default=cfg["output_master_table_csv"]) 
    parser.add_argument("--out-ts-html", default=cfg["output_timeseries_html"]) 
    parser.add_argument("--rutten-thames", default=cfg["rutten_thames_xlsx"]) 
    parser.add_argument("--rutten-mersey", default=cfg["rutten_mersey_xlsx"]) 
    parser.add_argument("--sas-token-path", default=cfg["sas_token_path"]) 
    parser.add_argument("--account-url", default=cfg["account_url"]) 
    parser.add_argument("--container-name", default=cfg["container_name"]) 
    args = parser.parse_args()

    # -----------------------------
    # Ingest
    # -----------------------------
    if args.source == "blob":
        sas_token = get_sas_token(args.sas_token_path)
        master_table = ingest_from_blob(args.account_url, args.container_name, sas_token)
    else:
        downloads_dir = Path(args.downloads_dir)
        master_table = ingest_from_tar_url(args.tar_url, downloads_dir)

    if master_table.empty:
        print("No rows ingested. Exiting.")
        sys.exit(0)

    # -----------------------------
    # Apply explicit destination order to sample_id, then tidy (unchanged logic)
    # -----------------------------
    DEST_ORDER = load_destination_order(DEST_TXT)

    if not master_table.empty:
        present = [lab for lab in DEST_ORDER if lab in set(master_table['sample_id'])]
        extras = [lab for lab in master_table['sample_id'].drop_duplicates().tolist() if lab not in DEST_ORDER]
        ordered_categories = present + extras
        master_table['sample_id'] = pd.Categorical(master_table['sample_id'], categories=ordered_categories, ordered=True)
        master_table['order_index'] = master_table['sample_id'].cat.codes

        master_table['start_time'] = pd.to_datetime(master_table['start_time'], errors='coerce', utc=True)
        master_table = master_table.sort_values(by=['order_index', 'start_time'], kind='mergesort')

        # Drop empty columns from known list (same as original)
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

    # -----------------------------
    # Save master table
    # -----------------------------
    master_table.to_csv(args.out_csv, index=False)
    print(f"Saved counts-only master table to: {args.out_csv}")

    # -----------------------------
    # Time series plot
    # -----------------------------
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
        try: fig.show()
        except Exception: pass
        print(f"Saved timeseries scatter to: {args.out_ts_html}")
    else:
        print("No valid timestamps found — timeseries plot skipped.")

    # -----------------------------
    # Rutten Excel joins (unchanged)
    # -----------------------------
    def load_rutten_file(path):
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        if "fname" not in df.columns:
            raise ValueError(f"'fname' column not found in {path}")
        return df

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

if __name__ == "__main__":
    main()
