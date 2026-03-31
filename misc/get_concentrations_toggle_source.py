# Gathers up predictions csv files from the blob store associated with a comparison exercise requested by Veronique Creach
# Cytoclus has a process for constructing a label database from a few xml files. Here we compare the export from that, provided as two excel files, against the random forest model that Lucinda developed.
# There was a lot of confusion when this model was trained - test score was only 75%
# I suspect cytoclus has a method for handling that. Here we "throw the labels at it" without cleaning them, but mapping them across to best match Thomas's names. 
# cytoclus's excel files were shared through a sharepoint link:
# https://cefas.sharepoint.com/sites/FlowCytometryWorkingGroup/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FFlowCytometryWorkingGroup%2FShared%20Documents%2FData%2Fthomas%5Freport%5F2026&viewid=449d5d77%2D3e15%2D4447%2D8bd2%2D737908016f10&p=true&ct=1769708971342&or=Teams%2DHL&LOF=1
# The dataset cyzs:
# https://citprodflowcytosa.blob.core.windows.net/mnceacyzfilesforthomasrutten/
# The random forest model:
# https://citprodflowcytosa.blob.core.windows.net/mnceacyzfilesforthomasrutten/model_not_nn_cleaned_used_for_blob_inference/
# The predictions:
# https://citprodflowcytosa.blob.core.windows.net/mnceacyzfilesforthomasrutten/blob_tool_outputs/

# This script also facilitates loading from  a local folder on disk – toggle via SOURCE_MODE below. It points to a model which I "cleaned" using nearest neighbour elimination to achieve a better (94%) test score, however it did not improve the fit vs the cytoclus method (I believe so-called 'hybrid') method
# Since this did not improve anything, the version presented to Veronique is the one held on the blob store "The predictions:" # https://citprodflowcytosa.blob.core.windows.net/mnceacyzfilesforthomasrutten/blob_tool_outputs/


# -*- coding: utf-8 -*-
"""get_concentrations_toggle_source.py

UPDATED VERSION (30 Mar 2026)

Purpose
-------
1) Gather model-derived per-sample class counts (Counts_*) from *_counts.csv files, sourced from:
   - Azure Blob Storage (default) OR
   - a local folder tree (toggle via CONC_SOURCE_MODE)

2) Build a wide master table (one row per sample) and save to CSV.

3) NEW: Also export a long-format (tidy) table with one row per (sample, class), including:
   - class label and count
   - percent of total counts in the sample
   - concentration [n/μl] using measurementResults_analysed_volume from the matching *_instrument.csv
   - per-class summary statistics (min/max/mean/sd/sum) for selected pulse metrics, computed from
     the matching *_predictions.csv.

4) Compare master counts against Cytoclus Excel summaries (scatter plots + per-class totals and % differences).

Notes
-----
Filename conventions assumed:
    <base>.cyz_predictions.csv_counts.csv  -> counts-only table
    <base>.cyz_predictions.csv            -> per-particle predictions table
    <base>.cyz_instrument.csv             -> instrument metadata (contains analysed volume)

Blob mode builds a basename->blobname index so related files can be fetched regardless of folder prefixes.
"""

import io
import os
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import functools
#import time

# -----------------------------
# Source toggle
# -----------------------------
# Set to 'blob' to read from Azure Blob Storage
# Set to 'folder' to read from a local directory of CSV files
SOURCE_MODE = "folder"#os.environ.get('CONC_SOURCE_MODE', 'blob').lower() 

# Local folder to use when SOURCE_MODE == 'folder'
LOCAL_COUNTS_DIR = r"C:\Users\JR13\Downloads\model_trained_on_nn_cleaned_94_pct_but_may_not_be_used\model_trained_on_nn_cleaned_94_pct_but_may_not_be_used\blob_tool_outputs"
LOCAL_SOURCE_ROOT = os.environ.get('CONC_LOCAL_ROOT', LOCAL_COUNTS_DIR)

# Azure Blob config (used only when SOURCE_MODE == 'blob')
ACCOUNT_URL = "https://citprodflowcytosa.blob.core.windows.net"
CONTAINER_NAME = "mnceacyzfilesforthomasrutten"
SAS_TOKEN_PATH = r"C:/Users/JR13/Documents/authenticationkeys/flowcytosaSAS.txt"

# -----------------------------
# Helpers
# -----------------------------

def get_sas_token(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read().strip()

# Columns that define the class set we expect (counts-only), to keep the schema stable.
class_columns = [
    'RedPico', 'Orapicoprok', 'Other', 'RedNano', 'nophytoplankton', 'noiseum',
    'no_phytoplankton', 'YB_um_beads', 'Plant_detritus', 'OraNano_crypto',
    'RWS_um_beads', 'Beads_rest', 'RedMicro', 'Bubbles', 'OraNano', 'C_undetermined'
]

# Output locations

# Output locations
output_master_table_csv = r"C:/Users/JR13/Downloads/master_table_counts_only_ordered.csv"
output_master_table_long_csv = r"C:/Users/JR13/Downloads/master_table_counts_long.csv"
output_timeseries_html = r"C:/Users/JR13/Downloads/counts_timeseries_ordered.html"  # unused at present

# Metrics to summarise from *_predictions.csv (per predicted class)
PRED_METRICS = [
    "FWS_length","FWS_total","FWS_maximum","FWS_average",
    "Sidewards_Scatter_length","Sidewards_Scatter_total","Sidewards_Scatter_maximum","Sidewards_Scatter_average",
    "Fl_Yellow_length","Fl_Yellow_total","Fl_Yellow_maximum","Fl_Yellow_average",
    "Fl_Orange_length","Fl_Orange_total","Fl_Orange_maximum","Fl_Orange_average",
    "Fl_Red_length","Fl_Red_total","Fl_Red_maximum","Fl_Red_average",
]

# -----------------------------
# Filename to Thomas filename
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

# ---------------------------------------------
# Map columns into thomas colnames

# ---------------------------------------------
# Map columns into Thomas colnames
# ---------------------------------------------
TO_COLUMNS = [
    "Counts_Beads_rest",
    "Counts_Bubbles",
    "Counts_No_Phytopl",
    "Counts_No_Phyto",
    "Counts_No_Phyto_large",
    "Counts_OraNano",
    "Counts_OraNano_Crypt",
    "Counts_OraPicoProk",
    "Counts_Plant_detritus",
    "Counts_RWS_3um_beads",
    "Counts_RedMicro",
    "Counts_RedNano",
    "Counts_RedPico",
    "Counts_YB_1um_beads",
    "Counts_noise_gt1um",
    "Counts_noise_st1um",
    "Counts_not_recognized",
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
    # Not in your “from” list, but these appear in your pipeline and fit the target schema:
    "bubbles": "Counts_Bubbles",
    "noiseum": "Counts_noise_gt1um",          # noise > 1 µm
    "nophytoplankton": "Counts_No_Phytopl",    # longer form mapped to "..._No_Phytopl"
    "c_undetermined": "Counts_not_recognized", # catch-all class
    # "YB_um_beads" / "RWS_um_beads" handled by the normalizer below.
}


def _normalize_key(s: str) -> str:
    return str(s).strip().lower().replace("__", "_").replace(" ", "_")

norm_map = {_normalize_key(k): v for k, v in SOURCE_TO_TARGET.items()}

# Optional explicit destination order
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

DEST_ORDER = load_destination_order(DEST_TXT)

# -----------------------------
# Source readers and indexing
# -----------------------------

def _basename(path: str) -> str:
    path = path.replace(r'\\', r'/')
    #path = path.replace(r'\', r'/')
    return path.rsplit('/', 1)[-1]


def build_folder_index(root: str) -> Dict[str, str]:
    """Map basename -> full path for all CSV files under a local root."""
    idx: Dict[str, str] = {}
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Local folder not found: {root}")
    for p in root_path.rglob('*.csv'):
        idx[p.name] = str(p)
    return idx


def get_blob_container_client():
    from azure.storage.blob import BlobServiceClient
    sas_token = get_sas_token(SAS_TOKEN_PATH)
    blob_service_client = BlobServiceClient(account_url=ACCOUNT_URL, credential=sas_token)
    return blob_service_client.get_container_client(CONTAINER_NAME)


def build_blob_index(container_client) -> Dict[str, str]:
    """Map basename -> full blob name for all blobs in the container."""
    idx: Dict[str, str] = {}
    for blob in container_client.list_blobs():
        idx[_basename(blob.name)] = blob.name
    return idx


def iter_counts_csv_from_folder(folder: str):
    """Yield (basename, bytes) for files in a local folder tree that end with 'counts.csv'."""
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Local folder not found: {folder}")
    for path in p.rglob('*counts.csv'):
        yield path.name, path.read_bytes()


def iter_counts_csv_from_blob(container_client):
    """Yield (basename, bytes) for blobs ending with 'counts.csv'."""
    for blob in container_client.list_blobs():
        if not blob.name.endswith('counts.csv'):
            continue
        blob_client = container_client.get_blob_client(blob)
        content = blob_client.download_blob().readall()
        yield _basename(blob.name), content


def _counts_to_predictions_basename(counts_basename: str) -> str:
    return counts_basename[:-len('_counts.csv')] if counts_basename.endswith('_counts.csv') else counts_basename


def _predictions_to_instrument_basename(pred_basename: str) -> str:
    return pred_basename.replace('_predictions.csv', '_instrument.csv')


def _find_class_col(pred_df: pd.DataFrame) -> Optional[str]:
    candidates = [
        'predicted_class','predicted_label','pred_label','prediction',
        'class','Class','label','Label','assignedClass','AssignedClass',
        'predicted','Predicted'
    ]
    for c in candidates:
        if c in pred_df.columns:
            return c
    for c in pred_df.columns:
        lc = str(c).lower()
        if 'class' in lc or 'label' in lc:
            return c
    return None


def _map_pred_label_to_counts_col(label: str) -> Optional[str]:
    ncol = _normalize_key(label)
    if ncol in {'yb_um_beads', 'yb_1um_beads', '1_um_beads'}:
        ncol = '1um_beads'
    if ncol in {'rws_um_beads', 'rws3um_beads', 'rws_3_um_beads'}:
        ncol = 'rws_3um_beads'
    return norm_map.get(ncol)


def compute_prediction_stats(pred_df: pd.DataFrame, predictions_basename: str) -> pd.DataFrame:
    class_col = _find_class_col(pred_df)
    if class_col is None:
        return pd.DataFrame(columns=['predictions_basename', 'Counts_col'])

    df = pred_df.copy()
    df['Counts_col'] = df[class_col].map(_map_pred_label_to_counts_col)
    df = df.dropna(subset=['Counts_col'])
    if df.empty:
        return pd.DataFrame(columns=['predictions_basename', 'Counts_col'])

    present_metrics = [m for m in PRED_METRICS if m in df.columns]
    if not present_metrics:
        return pd.DataFrame(columns=['predictions_basename', 'Counts_col'])

    for m in present_metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')

    grouped = df.groupby('Counts_col', dropna=True)

    out = pd.DataFrame({'Counts_col': list(grouped.groups.keys())})
    out['predictions_basename'] = predictions_basename

    for m in present_metrics:
        g = grouped[m]
        out[f'min_{m}'] = g.min().values
        out[f'max_{m}'] = g.max().values
        out[f'mean_{m}'] = g.mean().values
        out[f'sd_{m}'] = g.std(ddof=1).values
        out[f'sum_{m}'] = g.sum().values

    return out


# -----------------------------
# Load counts-only tables into master_table
# -----------------------------

master_table = pd.DataFrame()

blob_container_client = None
blob_index = None
folder_index = None

if SOURCE_MODE == 'folder':
    folder_index = build_folder_index(LOCAL_SOURCE_ROOT)
    iterator = iter_counts_csv_from_folder(LOCAL_SOURCE_ROOT)
elif SOURCE_MODE == 'blob':
    blob_container_client = get_blob_container_client()
    blob_index = build_blob_index(blob_container_client)
    iterator = iter_counts_csv_from_blob(blob_container_client)
else:
    raise ValueError("SOURCE_MODE must be 'folder' or 'blob'")

@functools.lru_cache(maxsize=4096)
def get_csv_bytes_by_basename(basename: str) -> Optional[bytes]:
    """Fetch bytes for a CSV by basename from the selected source."""
    if SOURCE_MODE == 'blob':
        if blob_container_client is None or blob_index is None:
            return None
        blob_name = blob_index.get(basename)
        if blob_name is None:
            return None
        bc = blob_container_client.get_blob_client(blob_name)
        return bc.download_blob().readall()
    else:
        if folder_index is None:
            return None
        path = folder_index.get(basename)
        if path is None:
            return None
        return Path(path).read_bytes()


for raw_name, csv_bytes in iterator:
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Transpose into a row-wise record (originals are often columnar per class)
    df_t = df.T.reset_index()
    df_t.columns = df_t.iloc[0]
    df_t = df_t[1:]

    # Start with a frame for all target columns (NaNs by default)
    counts_out = pd.DataFrame({col: np.nan for col in TO_COLUMNS}, index=df_t.index)

    for col in df_t.columns:
        ncol = _normalize_key(col)
        if ncol in {'yb_um_beads', 'yb_1um_beads', '1_um_beads'}:
            ncol = '1um_beads'
        if ncol in {'rws_um_beads', 'rws3um_beads', 'rws_3_um_beads'}:
            ncol = 'rws_3um_beads'

        target = norm_map.get(ncol)
        if target is None:
            continue
        counts_out[target] = pd.to_numeric(df_t[col], errors='coerce')

    non_count_cols = [c for c in df_t.columns if _normalize_key(c) not in norm_map]
    df_t = pd.concat([df_t[non_count_cols], counts_out], axis=1)

    df_t['filename'] = raw_name

    # Derive basenames for related files
    pred_basename = _counts_to_predictions_basename(raw_name)
    instr_basename = _predictions_to_instrument_basename(pred_basename)
    df_t['predictions_basename'] = pred_basename
    df_t['instrument_basename'] = instr_basename

    df_t['sample_id'] = df_t['filename'].map(FNAME_LOOKUP).fillna(df_t['filename'])

    # Ensure legacy class columns exist and are numeric
    for col in class_columns:
        if col not in df_t.columns:
            df_t[col] = np.nan
        df_t[col] = pd.to_numeric(df_t[col], errors='coerce')

    # Parse a timestamp if any of these columns are present
    time_candidates = ['start_time', 'Start', 'start', 'timestamp', 'time']
    parsed_time = None
    for cand in time_candidates:
        if cand in df_t.columns:
            parsed_time = pd.to_datetime(df_t[cand], errors='coerce', utc=True)
            break
    df_t['start_time'] = parsed_time if parsed_time is not None else pd.NaT

    master_table = pd.concat([master_table, df_t], ignore_index=True, sort=False)


# -----------------------------
# Apply explicit destination order to sample_id (and secondarily by time)
# -----------------------------

if not master_table.empty:
    present = [lab for lab in DEST_ORDER if lab in set(master_table['sample_id'])]
    extras = [lab for lab in master_table['sample_id'].drop_duplicates().tolist() if lab not in DEST_ORDER]
    ordered_categories = present + extras

    master_table['sample_id'] = pd.Categorical(master_table['sample_id'], categories=ordered_categories, ordered=True)
    master_table['order_index'] = master_table['sample_id'].cat.codes
    master_table = master_table.sort_values(by=['order_index', 'start_time'], kind='mergesort')

    # Drop legacy non-count columns if all missing/zero
    cols_to_drop = [
        'Other', 'RedNano', 'nophytoplankton', 'noiseum', 'no_phytoplankton',
        'YB_um_beads', 'Plant_detritus', 'OraNano_crypto', 'RWS_um_beads',
        'Beads_rest', 'RedMicro', 'Bubbles', 'OraNano', 'C_undetermined',
        'order_index'
    ]

    cols_safe_to_drop = []
    for col in cols_to_drop:
        if col not in master_table.columns:
            continue
        numeric_series = pd.to_numeric(master_table[col], errors='coerce')
        if numeric_series.notna().sum() == 0 or float(numeric_series.sum(skipna=True)) == 0.0:
            cols_safe_to_drop.append(col)

    if cols_safe_to_drop:
        master_table = master_table.drop(columns=cols_safe_to_drop)


# -----------------------------
# Save master table (wide)
# -----------------------------

master_table.to_csv(output_master_table_csv, index=False)
print(f"Saved counts-only master table to: {output_master_table_csv}")


# ==========================================================
# NEW: Long-format output with % totals, concentration, and per-class stats
# ==========================================================

def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


def _extract_volume(instr_df: pd.DataFrame) -> float:
    col = 'measurementResults_analysed_volume'
    if col not in instr_df.columns:
        return np.nan
    v = pd.to_numeric(instr_df[col], errors='coerce').dropna()
    return float(v.iloc[0]) if not v.empty else np.nan


volume_lookup: Dict[str, float] = {}
stats_frames: List[pd.DataFrame] = []

unique_pairs = master_table[['predictions_basename', 'instrument_basename']].drop_duplicates()

for pred_base, instr_base in unique_pairs.itertuples(index=False):
    # Volume
    instr_bytes = get_csv_bytes_by_basename(instr_base)
    if instr_bytes is not None:
        try:
            instr_df = _read_csv_bytes(instr_bytes)
            volume_lookup[pred_base] = _extract_volume(instr_df)
        except Exception:
            volume_lookup[pred_base] = np.nan
    else:
        volume_lookup[pred_base] = np.nan

    # Stats
    pred_bytes = get_csv_bytes_by_basename(pred_base)
    if pred_bytes is None:
        continue
    try:
        pred_df = _read_csv_bytes(pred_bytes)
        st = compute_prediction_stats(pred_df, predictions_basename=pred_base)
        if not st.empty:
            stats_frames.append(st)
    except Exception:
        continue

stats_df = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame(columns=['predictions_basename','Counts_col'])

count_cols = [c for c in master_table.columns if str(c).startswith('Counts_')]

id_vars = [c for c in ['filename','sample_id','start_time','predictions_basename','instrument_basename'] if c in master_table.columns]

long_df = master_table.melt(
    id_vars=id_vars,
    value_vars=count_cols,
    var_name='Counts_col',
    value_name='count'
)

long_df['class label'] = long_df['Counts_col'].str.replace('^Counts_', '', regex=True)
long_df['count'] = pd.to_numeric(long_df['count'], errors='coerce').fillna(0.0)

file_total = long_df.groupby('filename')['count'].transform('sum')
long_df['count_percent_of_total'] = np.where(file_total > 0, 100.0 * long_df['count'] / file_total, np.nan)

long_df['measurementResults_analysed_volume'] = long_df['predictions_basename'].map(volume_lookup)
vol = pd.to_numeric(long_df['measurementResults_analysed_volume'], errors='coerce')
long_df['concentration [n/μl]'] = np.where(vol > 0, long_df['count'] / vol, np.nan)

if not stats_df.empty:
    long_df = long_df.merge(stats_df, how='left', on=['predictions_basename','Counts_col'])


# ----------------------------------------------------------
# Filter and order long-format table
# ----------------------------------------------------------

# Keep only filenames containing 'pico' or 'nano' (case-insensitive)
#mask_pico = long_df['filename'].str.contains('pico', case=False, na=False)
#mask_nano = long_df['filename'].str.contains('nano', case=False, na=False)
#long_df = long_df[mask_pico | mask_nano].copy()

# Primary sort: filename (lexicographic)
#long_df = long_df.sort_values(by='filename', kind='mergesort')

# Secondary sort: pico before nano
# (False < True, so pico=False, nano=True works)
long_df['_is_nano'] = long_df['filename'].str.contains('nano', case=False, na=False)
long_df = long_df.sort_values(
    by=['_is_nano', 'filename'],
    kind='mergesort'
)

# Clean up helper column
#long_df = long_df.drop(columns=['_is_nano'])

long_df.to_csv(output_master_table_long_csv, index=False)
print(f"Saved long-format master table to: {output_master_table_long_csv}")

subsetlist = ["Thames_MA4_9 2023-06-19 12h38","Thames_MA2_9 2023-06-19 11h11","Thames_MA3_9 2024-09-17 08h26","Thames_MA4_3 2024-06-17 10h40","Thames_MA1_9 2024-01-08 09h22","Mersey_LV16_9 2023-06-16 10h02","Mersey_LV16_3 2024-07-18 09h32","Mersey_LV10_9 2024-07-18 11h48","Mersey_LV08_9 2023-06-16 14h28","Mersey_LV01_9 2024-02-21 12h27"]
long_df[ long_df["sample_id"].isin(subsetlist)].to_csv(    output_master_table_long_csv + "subset.csv",    index=False)


# ==========================================================
# Cytoclus comparison
# ==========================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_cytoclus_file(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    if 'fname' not in df.columns:
        raise ValueError(f"'fname' column not found in {path}")
    return df


cytoclus_1 = load_cytoclus_file('C:/Users/JR13/Downloads/Thames_merge_VAL_clusSumTotal.xlsx')
cytoclus_2 = load_cytoclus_file('C:/Users/JR13/Downloads/Mersey_merge_VAL_clusSumTotal.xlsx')

cytoclusxlsx = pd.concat([cytoclus_1, cytoclus_2], ignore_index=True)
print('cytoclus combined dataframe shape:', cytoclusxlsx.shape)

# If the master table uses 'filename' instead of 'sample_id', swap keys here.
# merge_key_master = "filename"
# merge_key_cytoclus = "fname"
merge_key_master = 'sample_id'
merge_key_cytoclus = 'fname'

# Align the cytoclus key to match the master's key name
cytoclusxlsx_renamed = cytoclusxlsx.rename(columns={merge_key_cytoclus: merge_key_master})

joined = master_table.merge(
    cytoclusxlsx_renamed,
    on=merge_key_master,
    suffixes=('_master', '_cytoclus')
)
print('Matched rows:', joined.shape)

count_cols_master = [c for c in master_table.columns if str(c).startswith('Counts_')]
count_cols_cytoclus = [c for c in cytoclusxlsx.columns if str(c).startswith('Counts_')]
shared = sorted(set(count_cols_master).intersection(set(count_cols_cytoclus)))
print('Shared count columns:', shared)

if len(shared) == 0:
    print("No shared 'Counts_' columns found between master and cytoclus—nothing to plot.")
else:
    n = len(shared)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4 * ncols + 1, 3.5 * nrows + 1),
                             squeeze=False)
    axes = axes.flatten()

    for ax, col in zip(axes, shared):
        mcol = col + '_master'
        rcol = col + '_cytoclus'
        if mcol not in joined.columns or rcol not in joined.columns:
            ax.set_visible(False)
            continue

        ax.scatter(joined[mcol], joined[rcol], alpha=0.7, s=18, edgecolor='none')
        ax.set_xlabel(f"{col} (master)")
        ax.set_ylabel(f"{col} (cytoclus)")
        ax.set_title(col, fontsize=10)

        x_max = joined[mcol].max(skipna=True)
        y_max = joined[rcol].max(skipna=True)
        candidates = [v for v in [x_max, y_max] if pd.notna(v)]
        local_max = float(max(candidates)) if candidates else None

        if local_max is not None and local_max > 0:
            ax.plot([0, local_max], [0, local_max], 'r--', linewidth=1)
            pad = local_max * 1.05
            ax.set_xlim(0, pad)
            ax.set_ylim(0, pad)

        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)

    for k in range(len(shared), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle('Master vs cytoclus: Counts Comparison', fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    combined_png = 'C:/Users/JR13/Downloads/counts_comparison_subplots.png'
    combined_pdf = 'C:/Users/JR13/Downloads/counts_comparison_subplots.pdf'
    fig.savefig(combined_png, dpi=200)
    fig.savefig(combined_pdf)
    plt.close(fig)

    print(f"Saved combined subplot figure to: {combined_png} and {combined_pdf}")


if len(shared) > 0 and not joined.empty:
    master_cols = [f"{c}_master" for c in shared if f"{c}_master" in joined.columns]
    cytoclus_cols = [f"{c}_cytoclus" for c in shared if f"{c}_cytoclus" in joined.columns]

    master_totals = joined[master_cols].sum(numeric_only=True)
    cytoclus_totals = joined[cytoclus_cols].sum(numeric_only=True)

    master_totals.index = [c.replace('_master', '') for c in master_totals.index]
    cytoclus_totals.index = [c.replace('_cytoclus', '') for c in cytoclus_totals.index]

    all_classes = sorted(set(master_totals.index) | set(cytoclus_totals.index))

    summary_df = pd.DataFrame({
        'class': all_classes,
        'total_master': [master_totals.get(cls, 0.0) for cls in all_classes],
        'total_cytoclus': [cytoclus_totals.get(cls, 0.0) for cls in all_classes],
    })

    summary_df['diff'] = summary_df['total_master'] - summary_df['total_cytoclus']

    eps = 1e-12
    denom = summary_df['total_cytoclus'].abs().clip(lower=eps)
    summary_df['pct_diff'] = 100.0 * summary_df['diff'] / denom
    summary_df['pct_diff_str'] = summary_df['pct_diff'].map(lambda x: f"{x:+.1f}%")

    summary_df = summary_df.sort_values(by='pct_diff', key=lambda s: s.abs(), ascending=False)

    out_csv = r"C:/Users/JR13/Downloads/counts_class_differences_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved per-class difference summary to: {out_csv}")

    preview_cols = ['class', 'total_master', 'total_cytoclus', 'diff', 'pct_diff_str']
    with pd.option_context('display.max_rows', None, 'display.float_format', '{:,.3f}'.format):
        print("=== Per-class totals and differences (master vs cytoclus) ===")
        print(summary_df[preview_cols])
else:
    print("No shared 'Counts_' columns or empty join; skipping per-class diff summary.")
