# functions file written whilst writing flow_cytometer_tool.py
import requests
import subprocess
import os
import json
import pandas as pd
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import tkinter as tk
import csv
from listmode import extract
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import shutil
from tkinter import simpledialog, ttk
from azure.storage.blob import ContainerClient, BlobServiceClient
import joblib
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from custom_functions_for_python import buildSupervisedClassifier, loadClassifier
import zipfile
import re
from urllib.parse import urlparse
import argparse
import platform
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tkinter import filedialog
import tempfile
import sys


__all__ = ["BlobServiceClient","choose_zone_folders","build_consensual_dataset","platform","run_backend_only","argparse","summarize_predictions","download_blobs", "convert_cyz_to_json", "compile_cyz2json_from_release",
    "compile_r_requirements", "flatten_dict", "dict_to_csv", "clear_temp_folder", "download_file",
    "load_file", "to_listmode", "apply_r", "select_output_dir", "load_json", "select_particles",
    "get_pulses", "display_image", "update_navigation_buttons", "save_metadata", "plot3d",
    "os", "shutil", "tk", "messagebox", "filedialog", "simpledialog", "ttk", "ContainerClient",
    "urlparse", "pd", "np", "joblib", "datetime", "json", "csv", "plt", "FigureCanvasTkAgg",
    "Line2D", "PolygonSelector", "Path", "buildSupervisedClassifier", "loadClassifier",
    "subprocess", "zipfile", "extract", "re","test_model","train_classifier","combine_csv_files","convert_json_to_listmode",
    "FileHandler","log_message","Observer","FileSystemEventHandler","filedialog",
    "sample_rows","upload_to_blob", "get_sas_token","mix_blob_files","list_blobs","extract_processed_url","apply_python_model","delete_file","combine_csvs","train_model","test_classifier","expertise_matrix_path"]


if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

expertise_matrix_path = os.path.join(base_path, "expertise_matrix.csv")


def train_model(df, plots_dir, model_path, nogui=False, self = None):
    try:
        if df is None:
            if nogui:
                print("Error: No data to train on.")
            else:
                from tkinter import messagebox
                messagebox.showerror("Error", "No data to train on.")
            return
        train_classifier(df, plots_dir, model_path)
        if nogui:
            print("Model training completed successfully.")
        else:
            from tkinter import messagebox
            messagebox.showinfo("Training Complete", "Model training completed successfully.")
            notebook = ttk.Notebook(self.root)
    except Exception as e:
        if nogui:
            print(f"Training Error: Failed to train model: {e}")
        else:
            from tkinter import messagebox
            messagebox.showerror("Training Error", f"Failed to train model: {e}")


def test_classifier(df, model_path, nogui=False):
    try:
        if not os.path.exists(model_path):
            msg = "Trained model not found. Please train the model first."
            if nogui:
                print(f"Model Error: {msg}")
            else:
                from tkinter import messagebox
                messagebox.showerror("Model Error", msg)
            return df, None
        if df is None:
            msg = "No dataset loaded. Please load or combine CSVs first."
            if nogui:
                print(f"Data Error: {msg}")
            else:
                from tkinter import messagebox
                messagebox.showerror("Data Error", msg)
            return df, None
        df, summary = test_model(df, model_path)
        if nogui:
            print("Prediction Summary:\n", summary)
        else:
            messagebox.showinfo("Prediction Summary", f"Predictions made successfully.\n\n{summary}")
        return df, summary
    except Exception as e:
        if nogui:
            print(f"Test Error: Failed to test classifier: {e}")
        else:
            from tkinter import messagebox
            messagebox.showerror("Test Error", f"Failed to test classifier: {e}")
        return df, None

def combine_csvs(output_path, expertise_matrix_path, nogui=False):
    if nogui:
        zonechoices = "PELTIC"  # Not ideal - hard coded so if the underlying dataset changes, the github actions workflow will break
    else:
        zonechoices = choose_zone_folders(output_path)

    try:
        expertise_matrix = pd.read_csv(expertise_matrix_path, index_col=0)
        expertise_levels = expertise_matrix.loc[zonechoices].to_dict()
        expertise_levels = {
            'expert': [k for k, v in expertise_levels.items() if v == 3],
            'advanced': [k for k, v in expertise_levels.items() if v == 2],
            'non_expert': [k for k, v in expertise_levels.items() if v == 1]
        }

        print("Zone choices:", zonechoices)
        print("expertise_levels:", expertise_levels)
        combined_df = build_consensual_dataset(output_path, expertise_levels, zonechoices)
        combined_df['source_label'] = [
            re.sub(r'[^a-zA-Z]', '', item).lower() for item in combined_df['source_label']
        ]
        combined_df.loc[combined_df['source_label'] == 'nophyto', 'source_label'] = 'nophytoplankton'
        print('Cleaned group names to something consistent')
        print("Cleaned source labels:", list(set(combined_df['source_label'])))
        print("Now dropping columns: ['consensus_label','person','index','id','sample_weight']")
        combined_df = combined_df.drop(columns=['consensus_label','person','index','id','sample_weight'])
        if combined_df is not None and not combined_df.empty:
            if nogui:
                print("CSV files combined successfully.")
            else:
                messagebox.showinfo("Success", "CSV files combined successfully.")
            return combined_df
        else:
            if nogui:
                print("No CSV files found to combine.")
            else:
                messagebox.showwarning("No CSVs", "No CSV files found to combine.")
            return None
    except Exception as e:
        if nogui:
            print(f"Combine Error: Failed to combine CSVs: {e}")
        else:
            messagebox.showerror("Combine Error", f"Failed to combine CSVs: {e}")
        return None


def sample_rows(df, sample_rate=0.001):
    return df.sample(frac=sample_rate)

def upload_to_blob(file_path, sas_token, container, output_blob_folder):
    try:
        full_url = f"{container}{sas_token}"
        blob_service_client = BlobServiceClient(account_url=full_url)
        blob_client = blob_service_client.get_blob_client(container=output_blob_folder, blob=os.path.basename(file_path))
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        log_message(f"Upload Success: File {file_path} uploaded to {container} container.")
    except Exception as e:
        log_message(f"Upload Error: Failed to upload file: {e}")


def mix_blob_files(container, sas_token, output_blob_folder, sample_rate=0.005):
    parts = container.split(r"/")
    container_url = '/'.join(parts[:-1]) + '/'
    blob_service_client = BlobServiceClient(account_url=f"{container_url}{sas_token}")
    container_client = blob_service_client.get_container_client(parts[-1])
    blob_list = container_client.list_blobs()
    all_sampled_df = pd.DataFrame()
    file_count = 0
    print("Mixing csv files, this will take a long time...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        output_file = temp_file.name
        i=0
        for blob in blob_list:
            if blob.name.endswith("_predictions.csv"):
                i=i+1
                print(blob.name)
                try:
                    blob_client = container_client.get_blob_client(blob)
                    download_stream = blob_client.download_blob()
                    df = pd.read_csv(download_stream)
                    sampled_df = sample_rows(df, sample_rate)
                    all_sampled_df = pd.concat([all_sampled_df, sampled_df], ignore_index=True)
                    file_count += 1
                    if file_count % 100 == 0:
                        all_sampled_df.to_csv(output_file, index=False)
                        upload_to_blob(output_file,  sas_token, container, output_blob_folder)
                except Exception as e:
                    print(f"Error processing {blob.name}: {e}")
        if not all_sampled_df.empty:
            all_sampled_df.to_csv(output_file, index=False)
            upload_to_blob(output_file,  sas_token, container,output_blob_folder)




class FileHandler(FileSystemEventHandler):
    def __init__(self, cyz2json_path, output_folder, model_path):
        self.cyz2json_path = cyz2json_path
        self.output_folder = output_folder
        self.model_path = model_path

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".cyz"):
            self.process_file(event.src_path)

    def process_file(self, file_path):
        try:
            log_message(f"Processing file: {file_path}")
            base_filename = os.path.basename(file_path)
            json_file = os.path.join(self.output_folder, base_filename.replace(".cyz", ".json"))
            listmode_file = os.path.join(self.output_folder, base_filename.replace(".cyz", ".csv"))
            predictions_file = os.path.join(self.output_folder, base_filename.replace(".cyz", "_predictions.csv"))

            load_file(self.cyz2json_path, file_path, json_file)
            log_message(f"Success: Cyz2json applied {file_path}")
            to_listmode(json_file, listmode_file)
            instrument_file = os.path.join(self.output_folder, f"{base_filename}_instrument.csv")
            os.rename(listmode_file + "instrument.csv", instrument_file)
            log_message(f"Success: Listmode applied {file_path}")
            apply_python_model(listmode_file, predictions_file, self.model_path)
            log_message(f"Success: Predictions made for {file_path}")

            predictions_df = pd.read_csv(predictions_file)
            prediction_counts = predictions_df['predicted_label'].value_counts().reset_index()
            prediction_counts.columns = ['class', 'count']
            prediction_counts_path = predictions_file + "_counts.csv"
            prediction_counts.to_csv(prediction_counts_path, index=False)
            log_message(f"Success: counted {file_path}")

            data = pd.read_csv(predictions_file)
            data['category'] = data['predicted_label']
            unique_categories = data['category'].unique()
            preset_colors = {
                'rednano': 'red',
                'orapicoprok': 'orange',
                'micro': 'blue',
                'beads': 'green',
                'oranano': 'purple',
                'noise': 'gray',
                'C_undetermined': 'black',
                'redpico': 'pink'
            }
            color_map = {
                category: preset_colors.get(
                    category,
                    f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
                ) for category in unique_categories
            }
            data['color'] = data['category'].map(color_map)
            x_99 = np.percentile(data["Fl_Yellow_total"], 99.5)
            y_99 = np.percentile(data["Fl_Red_total"], 99.5)
            z_99 = np.percentile(data["Fl_Orange_total"], 99.5)
            scatter = go.Scatter3d(
                x=data["Fl_Yellow_total"],
                y=data["Fl_Red_total"],
                z=data["Fl_Orange_total"],
                mode='markers',
                marker=dict(size=5, color=data['color'], showscale=False),
                text=data['category'],
                name='Data Points'
            )
            camera = dict(
                eye=dict(x=-1.5, y=-1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
            fig = go.Figure(data=[scatter])
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[0, x_99], title="Fl_Yellow_total"),
                    yaxis=dict(range=[0, y_99], title="Fl_Red_total"),
                    zaxis=dict(range=[0, z_99], title="Fl_Orange_total"),
                    camera=camera
                ),
                title='3D Data Points'
            )
            plot3d_prediction_path = predictions_file + "_3d.html"
            pio.write_html(fig, file=plot3d_prediction_path, auto_open=False)
            log_message("Plot saved as '3D_Plot.html'.")
            delete_file(listmode_file)
            delete_file(json_file)
#            delete_file(plot3d_prediction_path)
#            delete_file(instrument_file)
#            delete_file(predictions_file)
#            delete_file(prediction_counts_path)
        except Exception as e:
            log_message(f"Error: An error occurred processing {file_path}: {e}")


def extract_processed_url(line):
    prefix = "Success: counted "
    if line.startswith(prefix):
        return line[len(prefix):].strip()
    return None

# Function to log messages to both terminal and a log file
def log_message(message, log_file="process_log.txt"):
    print(message)
    with open(log_file, "a") as file:
        file.write(message + "\n")

def get_sas_token(file_path):
    with open(file_path, 'r') as file:
        sas_token = file.read().strip()
    return sas_token


def list_blobs(container_url, sas_token):
    try:    
        parts = container_url.split(r"/")
        container_url='/'.join(parts[:-1]) + '/'
        container = parts[-1]
        full_url = f"{container_url}{sas_token}"
        blob_service_client = BlobServiceClient(account_url=full_url)
        container_client = blob_service_client.get_container_client(container)
        blob_list = [blob.name for blob in container_client.list_blobs() if blob.name.endswith('.cyz')]
        return blob_list
    except Exception as e:
        log_message(f"Blob Listing Error: Failed to list blobs: {e}")
        return []

def download_file(url, tool_dir, filename):
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
        downloaded_file = os.path.join(tool_dir, filename)
        with open(downloaded_file, 'wb') as file:
            file.write(response.content)
        return downloaded_file
    except requests.RequestException as e:
        log_message(f"Download Error: Failed to download file: {e}")
        return None

def load_file(cyz2json_path, downloaded_file, json_file):
    try:
        subprocess.run(["dotnet", cyz2json_path, downloaded_file, "--output", json_file], check=True)
    except subprocess.CalledProcessError as e:
        log_message(f"Processing Error: Failed to process file: {e}")

def to_listmode(json_file, listmode_file):
    try:
        data = json.load(open(json_file, encoding="utf-8-sig"))
        lines = extract(particles=data["particles"], dateandtime=data["instrument"]["measurementResults"]["start"], images='', save_images_to='')
        df = pd.DataFrame(lines)
        df.to_csv(listmode_file, index=False)
        dict_to_csv(data['instrument'], listmode_file + 'instrument.csv')   
    except subprocess.CalledProcessError as e:
        log_message(f"Processing Error: Failed to process file: {e}")


def apply_python_model(listmode_file, predictions_file, model_path):
    try:
        model, classes, features = loadClassifier(model_path)
        df = pd.read_csv(listmode_file)
        df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
        df = df.dropna()
        # Ensure only required features are used
        print("Your model expects these columns:", features)
        print("Your data file has these columns:", df.columns.tolist())
        try:
            df = df[features]
        except:
            print("Getting a not in index error? That means columns in this data file do not match those the model was trained on. Is this file from a different flow cytometer?")
        print("Predicting ...")
        # Classify data, predict the labels and probabilities
        predictions = model.predict(df[features])
        proba_predict = pd.DataFrame(model.predict_proba(df[features])) # compute class prediction probabilities and store in data frame
        predicted_data = df
        # Add prediction to original test table
        predicted_data['predicted_label'] = predictions 
        # Make the column names of this data frame the class names (instead of numbers)
        proba_predict = proba_predict.set_axis(classes, axis=1)
        # Bind both data frames by column
        full_predicted = pd.concat([predicted_data, proba_predict], axis=1)
        # Save final predicted table
        full_predicted.to_csv(predictions_file)        
        log_message(f"Prediction Success: Predictions saved to {predictions_file}")
    except Exception as e:
        log_message(f"Prediction Error: Failed to apply Python model: {e}")





def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            log_message(f"Deleted file: {file_path}")
        else:
            log_message(f"File not found: {file_path}")
    except Exception as e:
        log_message(f"Error deleting file {file_path}: {e}") 


def convert_json_to_listmode(output_path):
    for root, _, files in os.walk(output_path):
        for file in files:
            if file.lower().endswith(".json") and not file.endswith("instrument.csv"):
                json_file = os.path.join(root, file)
                listmode_file = os.path.splitext(json_file)[0] + ".csv"
                try:
                    with open(json_file, encoding="utf-8-sig") as f:
                        data = json.load(f)
                    lines = extract(
                        particles=data["particles"],
                        dateandtime=data["instrument"]["measurementResults"]["start"],
                        images='',
                        save_images_to=''
                    )
                    df = pd.DataFrame(lines)
                    df.to_csv(listmode_file, index=False)
                    dict_to_csv(data['instrument'], listmode_file + 'instrument.csv')
                    print(f"Converted: {json_file} → {listmode_file}")
                except Exception as e:
                    print(f"Error processing file: {json_file}")
                    print(f"Exception: {e}")

def combine_csv_files(output_path):
    variation_pattern = re.compile(r'_(\w+)\.cyz\.csv$')
    all_data = []
    for root, _, files in os.walk(output_path):
        for file in files:
            if file.endswith(".csv") and not file.endswith("instrument.csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                match = variation_pattern.search(file)
                if match:
                    label = match.group(1)
                    df['source_label'] = label
                    all_data.append(df)
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.columns = combined_df.columns.str.replace(r'\s+', '_', regex=True)
        combined_df = combined_df.dropna()
        return combined_df
    else:
        return None



def choose_zone_folders(output_path):
    folders = [name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))]
    zonechoice = simpledialog.askstring("Zone Choice", f"Choose a zone from: {', '.join(folders)}")
    return zonechoice


def compute_consensual_labels_and_sample_weights(data):
    from collections import Counter

    def get_weighted_mode(labels, weights):
        counter = Counter()
        for label, weight in zip(labels, weights):
            counter[label] += weight
        most_common_label, most_common_weight = counter.most_common(1)[0]
        return most_common_label, most_common_weight

    # Group by 'id'
    grouped = data.groupby('id')

    # Initialize lists to store results
    consensus_labels = []
    sample_weights = []
    ids = []

    for name, group in grouped:
        labels = group['source_label']
        weights = group['weight']
        consensus_label, consensus_weight = get_weighted_mode(labels, weights)
        total_weight = weights.sum()
        sample_weight = consensus_weight / total_weight
        if consensus_label != "Unassigned Particles":
            ids.append(name)
            consensus_labels.append(consensus_label)
            sample_weights.append(sample_weight)

    # Create a DataFrame with consensus results
    consensus_df = pd.DataFrame({
        'id': ids,
        'consensus_label': consensus_labels,
        'sample_weight': sample_weights
    })

    # Merge back into the original data
    merged_df = data.merge(consensus_df, on='id', how='inner')

    return merged_df




def build_consensual_dataset(base_path, expertise_levels, zonechoice):
    """
    Build a consensual dataset from flow cytometry CSV files.
    
    Parameters:
    - base_path: str, the base directory containing subfolders for each person.
    - expertise_levels: dict, a dictionary with expertise levels as keys and lists of people as values.
    
    Returns:
    - pd.DataFrame, the combined DataFrame with consensus labels and sample weights.
    """

    variation_pattern = re.compile(r'_(\w+)\.cyz\.csv$')
    all_data = []
    expertise_weights = {'expert': 3, 'advanced': 2, 'non_expert': 1}
    print(os.path.join(base_path,zonechoice))
    # Traverse the directory structure
    for root, _, files in os.walk(os.path.join(base_path,zonechoice)):
        for file in files:
            print(file)
            if file.endswith(".csv") and not file.endswith("instrument.csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                except:
                    print(f"Skipping empty or malformed file: {file_path}")
                    continue
                match = variation_pattern.search(file)
                if match:
                    label = match.group(1)
                    person = os.path.basename(os.path.dirname(root))
                    df['source_label'] = label
                    df['person'] = person
                    all_data.append(df)
    
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.columns = combined_df.columns.str.replace(r'\s+', '_', regex=True)
    combined_df = combined_df.dropna()
    print(combined_df)
    
    # Flatten the expertise_levels into a person-to-weight mapping
    person_to_weight = {
        person: expertise_weights[level]
        for level, people in expertise_levels.items()
        for person in people
    }

    # Assign person weights
    combined_df['weight'] = combined_df['person'].map(person_to_weight).fillna(1)

    # Compute consensus label per particls
    combined_df = compute_consensual_labels_and_sample_weights(combined_df)
    combined_df['source_label'] = combined_df['consensus_label']
    print(combined_df)
    combined_df = combined_df.reset_index()
    print(combined_df)
    return combined_df


def plot_cv_results(cv_results, plots_dir):
    plotlist = []
    best_results = cv_results[cv_results['iter'] == cv_results['iter'].max()].groupby(
        ['param_classifier', 'outer_splits']
    ).apply(lambda x: x.loc[x['mean_test_score'].idxmax()])
    for outer in cv_results['outer_splits'].unique():
        outer_data = cv_results[cv_results['outer_splits'] == outer]
        outer_score = round(outer_data['outer_split_test_score'].unique()[0], 3)
        best_params = best_results[best_results['outer_splits'] == outer][[
            'param_classifier',
            'param_classifier__learning_rate',
            'param_classifier__max_depth',
            'param_classifier__max_features',
            'param_classifier__C',
            'param_classifier__l1_ratio',
            'param_classifier__max_samples'
        ]].values[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.lineplot(data=outer_data, x='iter', y='mean_test_score', hue='param_classifier', marker='o', ax=ax)
        # Compute and plot delta MCC
        for classifier in outer_data['param_classifier'].unique():
            clf_data = outer_data[outer_data['param_classifier'] == classifier].sort_values('iter')
            clf_data['delta_mcc'] = clf_data['mean_test_score'].diff()
            sns.lineplot(data=clf_data, x='iter', y='delta_mcc', label=f"{classifier} ΔMCC", linestyle='--', ax=ax)
        ax.set_title(f"Outer split {outer}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MCC and ΔMCC")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Classifier")
        fig.text(0.5, -0.1, f"Best Classifier (used in outer CV) : {best_params}\nOuter CV test score : {outer_score}",
                 wrap=True, horizontalalignment='center', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'cv_results_outer_{outer}.png'))
        plt.close(fig)
        plotlist.append(fig)
    return plotlist



def plot_classifier_props(cv_results):
    plotlist = []
    best_results = cv_results[cv_results['iter'] == cv_results['iter'].max()].groupby(['param_classifier', 'outer_splits']).apply(lambda x: x.loc[x['mean_test_score'].idxmax()])

    for outer in cv_results['outer_splits'].unique():
        outer_score = round(cv_results[cv_results['outer_splits'] == outer]['outer_split_test_score'].unique()[0], 3)
        best_params = best_results[best_results['outer_splits'] == outer][['param_classifier','param_classifier__learning_rate','param_classifier__max_depth','param_classifier__max_features','param_classifier__C','param_classifier__l1_ratio','param_classifier__max_samples']].values[0]
        
        plt.figure(figsize=(12, 8))
        sns.histplot(data=cv_results[cv_results['outer_splits'] == outer], x='iter', hue='param_classifier', multiple='stack')
        plt.title(f"Outer split {outer}")
        plt.xlabel("Iteration")
        plt.ylabel("Proportion of candidates")
        plt.xticks(rotation=45)
        plt.legend(title="Classifier")
        plt.figtext(0.5, -0.1, f"Best Classifier (used in outer CV) : {best_params}\nOuter CV test score : {outer_score}", wrap=True, horizontalalignment='center', fontsize=10)
        plt.tight_layout()
        plotlist.append(plt)
        plt.show()

    return plotlist


def plot_all_hyperpars_combi_and_classifiers_scores(cv_results, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    def plot_all_hyperpars_combi(cv_results, classifier_name, hyperparameters):
        def plot_hyperpar_combi(cv_results, classifier_name, x_axis, y_axis):
            filtered_results = cv_results.copy()
            if x_axis == "degree" or y_axis == "degree":
                filtered_results = filtered_results[filtered_results['param_classifier__kernel'] == "poly"]
            filtered_results = filtered_results[filtered_results['param_classifier'] == classifier_name]
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = sns.scatterplot(
                data=filtered_results,
                x=x_axis,
                y=y_axis,
                hue='mean_test_score',
                palette='viridis',
                size='mean_test_score',
                sizes=(20, 200),
                ax=ax
            )
            if x_axis in ["C", "gamma", "learning_rate"]:
                ax.set_xscale('log')
            if y_axis in ["C", "gamma", "learning_rate"]:
                ax.set_yscale('log')
            ax.set_xlabel(x_axis.replace("_", " "))
            ax.set_ylabel(y_axis.replace("_", " "))
            ax.set_title(f"{classifier_name} - {x_axis} vs {y_axis}")
            ax.legend(title="Mean MCC")
            fig.tight_layout()
            return fig
        grid = [(x, y) for x in hyperparameters for y in hyperparameters if x != y]
        plot_list = [plot_hyperpar_combi(cv_results, classifier_name, x, y) for x, y in grid]
        return plot_list
    logreg_hyperpars = ["param_classifier__C", "param_classifier__l1_ratio"]
    rf_hyperpars = ["param_classifier__max_features", "param_classifier__max_samples"]
    hgb_hyperpars = ["param_classifier__max_depth", "param_classifier__max_features", "param_classifier__learning_rate"]
    classifiers_hyperpars = {
        "LogisticRegression": logreg_hyperpars,
        "RandomForestClassifier": rf_hyperpars,
        "HistGradientBoostingClassifier": hgb_hyperpars
    }
    for classifier_name, hyperparameters in classifiers_hyperpars.items():
        print(f"Plotting hyperparameter combinations for {classifier_name}")
        plot_list = plot_all_hyperpars_combi(cv_results, classifier_name, hyperparameters)
        for i, fig in enumerate(plot_list):
            fig.savefig(os.path.join(plots_dir, f'{classifier_name}_plot_{i+1}.png'))
            plt.close(fig)


def train_classifier(df, plots_dir, model_path):
    df["group"] = df.index # This means no grouping. i.e. it does not matter which file the particle label came from.
    cleaned_df = df[[col for col in df.columns if col not in ["datetime", "user_id", "location"]]]
    # Detect if running from PyInstaller bundle
    is_frozen = getattr(sys, 'frozen', False)
    cores = 1 if is_frozen else os.cpu_count()
    
    # Split the data
    train_df, test_df = train_test_split(cleaned_df, test_size=0.2, stratify=cleaned_df["source_label"], random_state=42)
    
    # Train on training set
    buildSupervisedClassifier(
        training_set=train_df,
        target_name="source_label",
        group_name="group",
        weight_name="weight",
        select_K=5,
        cores=cores,
        n_sizes=4,
        filename_cvResults=os.path.join(os.path.dirname(model_path),"cv_results" + os.path.basename(model_path) + ".csv"),
        filename_learningCurve=os.path.join(os.path.dirname(model_path),"learning_curve" + os.path.basename(model_path) + ".csv"),
        filename_finalFittedModel=model_path,
        filename_finalCalibratedModel=os.path.join(os.path.dirname(model_path),'calibrated_' + os.path.basename(model_path)),
        filename_importance = os.path.join(os.path.dirname(model_path), "permutation_importance_" + os.path.basename(model_path) + ".csv"),
        validation_set = test_df,
        plots_dir = plots_dir        
    )

    # Evaluate on test set
    model, classes, features = loadClassifier(model_path)
    test_df_filtered=test_df[features]
    predictions = model.predict(test_df_filtered)
    proba_predict = pd.DataFrame(model.predict_proba(test_df_filtered)) # compute class prediction probabilities and store in data frame
    predicted_data = test_df
    # Add prediction to original test table
    predicted_data['predicted_label'] = predictions 
    # Make the column names of this data frame the class names (instead of numbers)
    proba_predict = proba_predict.set_axis(classes, axis=1)
    # Bind both data frames by column
    full_predicted = pd.concat([predicted_data, proba_predict], axis=1)
    # Save final predicted table
    #full_predicted.to_csv(predict_name)        
    print("Test Set Evaluation:\n", classification_report(test_df["source_label"], predictions))

    # Confusion Matrix
    cm = confusion_matrix(test_df["source_label"], predictions)
    print("Confusion Matrix:\n", cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(plots_dir+f'/confusionmatrix.png')
    plt.show()
    
    
    cv_results = pd.read_csv(os.path.join(os.path.dirname(model_path),"cv_results" + os.path.basename(model_path) + ".csv"))

    try:
        plot_cv_results(cv_results,plots_dir)
        plot_classifier_props(cv_results)
        plot_all_hyperpars_combi_and_classifiers_scores(cv_results,plots_dir)
    except Exception as e:
        print(f"Could not plot CV results: {e}")

def test_model(df, model_path):
    model, classes, features = loadClassifier(model_path)
    df=df[features]
    predictions = model.predict(df[features])
    proba_predict = pd.DataFrame(model.predict_proba(df[features])) # compute class prediction probabilities and store in data frame
    predicted_data = df
    # Add prediction to original test table
    predicted_data['predicted_label'] = predictions 
    # Make the column names of this data frame the class names (instead of numbers)
    proba_predict = proba_predict.set_axis(classes, axis=1)
    # Bind both data frames by column
    full_predicted = pd.concat([predicted_data, proba_predict], axis=1)
    # Save final predicted table
    #full_predicted.to_csv(predict_name) 
    df['predicted_label'] = predictions
    summary = df['predicted_label'].value_counts().to_string()
    return df, summary



def download_blobs(blob_url, download_path, sas_token = None):
    parsed_url = urlparse(blob_url)
    account_url = f"https://{parsed_url.netloc}"
    container_name = parsed_url.path.strip("/").split("/")[0]
    prefix = "/".join(parsed_url.path.strip("/").split("/")[1:])
    container_client = ContainerClient(account_url=account_url, container_name=container_name, credential=sas_token)
    blobs = container_client.list_blobs(name_starts_with=prefix)
    for blob in blobs:
        print(blob.name)
        blob_path = blob.name
        local_file_path = os.path.join(download_path, os.path.relpath(blob_path, prefix))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, "wb") as file:
            blob_data = container_client.download_blob(blob_path)
            file.write(blob_data.readall())


def convert_cyz_to_json(input_dir, output_dir, dll_path):
    import os, subprocess
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".cyz"):
                print(file)
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, input_dir)
                rel_dir = os.path.dirname(rel_path)
                dst_dir = os.path.join(output_dir, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst_file = os.path.join(dst_dir, file + ".json")
                subprocess.run(["dotnet", dll_path, full_path, "--output", dst_file], check=True)


def compile_cyz2json_from_release(cyz2json_dir, path_entry):
    if os.path.exists(cyz2json_dir):
        print("Info: cyz2json already exists in " + cyz2json_dir)
        return
    try:
        os.makedirs(cyz2json_dir, exist_ok=True)
        zip_path = os.path.join(os.path.dirname(cyz2json_dir), "cyz2json.zip")
        # Detect OS and choose appropriate release
        system = platform.system().lower()
        if system == "windows":
            zip_url = "https://github.com/OBAMANEXT/cyz2json/releases/download/v0.0.5/cyz2json-windows-latest.zip"
        elif system == "linux":
            zip_url = "https://github.com/OBAMANEXT/cyz2json/releases/download/v0.0.5/cyz2json-ubuntu-latest.zip"
        else:
            raise RuntimeError(f"Unsupported OS: {system}")
        print(f"Downloading cyz2json for {system}...")
        subprocess.run(["curl", "-L", "-o", zip_path, zip_url], check=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cyz2json_dir)
        if path_entry:
            path_entry.delete(0, tk.END)
            path_entry.insert(0, os.path.join(cyz2json_dir, "bin", "Cyz2Json.dll"))
    except subprocess.CalledProcessError as e:
        print(f"Compilation Error: Failed to download cyz2json: {e}.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, sub_item in enumerate(v):
                if isinstance(sub_item, dict):
                    items.extend(flatten_dict(sub_item, f"{new_key}_{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}_{i}", sub_item))
        else:
            items.append((new_key, v))
    return dict(items)

def dict_to_csv(data, output_file):
    flattened_data = [flatten_dict(item) for item in data] if isinstance(data, list) else [flatten_dict(data)]    
    header = set()
    for item in flattened_data:
        header.update(item.keys())
    header = sorted(header)
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for item in flattened_data:
            writer.writerow(item)
    print(f"Data saved to {output_file}")


def clear_temp_folder(tool_dir):
    """Clear the temporary directory."""
    for filename in os.listdir(tool_dir):
        file_path = os.path.join(tool_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")


def compile_r_requirements(r_dir, rpath_entry):
    """Get R requirements"""
#    if os.path.exists(r_dir):
#        messagebox.showinfo("Info", "r installation already exists in " + r_dir)
#        return
    try:
        subprocess.run(["curl", "https://cran.r-project.org/bin/windows/base/old/4.3.3/R-4.3.3-win.exe", "--output", r_dir+"/R-4.3.3-win.exe"], check=True)
        subprocess.run([r_dir+"/R-4.3.3-win.exe", "/DIR="+r_dir], cwd=r_dir, check=True)
        subprocess.run([r_dir+"/bin/Rscript.exe", "./install_rpackages.R"], check=True)
        rpath_entry.delete(0, tk.END)
        rpath_entry.insert(0, os.path.join(r_dir, "bin", "Rscript.exe"))
        messagebox.showinfo("Download Success", f"R downloaded and libraries installed")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Compilation Error", f"Failed to compile r: {e}.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

def apply_r(listmode_file, predictions_file, rpath_entry):
    """Convert .cyz file to .json using cyz2json tool."""
    try:
        print(rpath_entry)
        print(listmode_file)
        print(predictions_file)
        subprocess.run([rpath_entry, "rf_predict.R", "final_rf_model.rds", listmode_file, predictions_file], check=True)
        messagebox.showinfo("Success", f"R applied successfully")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Processing Error", f"Failed to process file: {e}. Is R installed here?")

def select_output_dir(app):
    """Open a dialog to select the output directory."""
    app.output_dir = filedialog.askdirectory()
    if app.output_dir:
        messagebox.showinfo("Output Directory Selected", f"Output files will be saved in: {app.output_dir}")

def load_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def select_particles(json_data, particle_ids):
    particles = [p for p in json_data['particles'] if p['particleId'] in particle_ids]
    return particles if particles else None

def get_pulses(particles):
    pulses = {p['particleId']: p.get('pulseShapes') for p in particles}
    return pulses



def display_image(self,root,current_image_index, output_dir, image_label, tif_files, metadata, confidence_entry, species_entry):
    """Display the image and update metadata entry fields."""
    image_file = tif_files[current_image_index]
    image_path = os.path.join(output_dir, image_file)
    
    img = Image.open(image_path)
    img = img.resize((400, 400), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)

    if self.image_label is None:
        self.image_label = tk.Label(self.root, image=img_tk)
        self.image_label.image = img_tk
        self.image_label.pack(pady=10)
    else:
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    # Load saved metadata if it exists
    metadata = self.metadata.get(image_file, {"confidence": "", "species": ""})
    self.confidence_entry.delete(0, tk.END)
    self.confidence_entry.insert(0, metadata["confidence"])
    self.species_entry.delete(0, tk.END)
    self.species_entry.insert(0, metadata["species"])



def update_navigation_buttons(prev_button, next_button, current_image_index, total_images):
    """Update the state of navigation buttons based on the current image index."""
    prev_button.config(state=tk.NORMAL if current_image_index > 0 else tk.DISABLED)
    next_button.config(state=tk.NORMAL if current_image_index < total_images - 1 else tk.DISABLED)


def save_metadata(current_image_index, tif_files, metadata, confidence_entry, species_entry, output_dir):
    """Save metadata to a CSV file."""
    image_file = tif_files[current_image_index]
    confidence = confidence_entry.get()
    species = species_entry.get()
    metadata[image_file] = {"confidence": confidence, "species": species}

    metadata_file_path = os.path.join(output_dir, "label_data.csv")
    with open(metadata_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image File", "confidence", "Suspected Species"])
        for image, data in metadata.items():
            writer.writerow([image, data["confidence"], data["species"]])


def plot3d(predictions_file):
    data = pd.read_csv(predictions_file)
    data['category'] = data['predicted_label']
    unique_categories = data['category'].unique()

    preset_colors = {
        'rednano': 'red',
        'orapicoprok': 'orange',
        'micro': 'blue',
        'beads': 'green',
        'oranano': 'purple',
        'noise': 'gray',
        'C_undetermined': 'black',
        'redpico': 'pink'
    }

    color_map = {
        category: preset_colors.get(
            category,
            f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
        ) for category in unique_categories
    }
    data['color'] = data['category'].map(color_map)

    x_99 = np.percentile(data['Fl.Yellow_total'], 99.5)
    y_99 = np.percentile(data['Fl.Red_total'], 99.5)
    z_99 = np.percentile(data['Fl.Orange_total'], 99.5)

    scatter = go.Scatter3d(
        x=data['Fl.Yellow_total'],
        y=data['Fl.Red_total'],
        z=data['Fl.Orange_total'],
        mode='markers',
        marker=dict(size=5, color=data['color'], showscale=False),
        text=data['category'],
        name='Data Points'
    )

    camera = dict(
        eye=dict(x=-1.5, y=-1.5, z=1.5),  
        center=dict(x=0, y=0, z=0),        
        up=dict(x=0, y=0, z=1)            
    )
    fig = go.Figure(data=[scatter])

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, x_99], title='Fl.Yellow_total'),
            yaxis=dict(range=[0, y_99], title='FL.Red_total'),
            zaxis=dict(range=[0, z_99], title='FL.Orange_total'),
            camera=camera
        ),
        title='3D Data Points'
    )
    pio.write_html(fig, file=predictions_file+"_3d.html", auto_open=True)

    print("Plot saved as '3D_Plot.html'.")

def summarize_predictions(df, pumped_volume):
    """Generate a summary of labelled and predicted data counts."""
    summary = []
    labels = df['label'].dropna().unique() if 'label' in df.columns else []
    preds = df['predicted_label'].dropna().unique() if 'predicted_label' in df.columns else []
    all_classes = set(labels).union(preds)
    for cls in all_classes:
        label_count = (df['label'] == cls).sum() / pumped_volume if 'label' in df.columns else 0
        pred_count = (df['predicted_label'] == cls).sum() / pumped_volume if 'predicted_label' in df.columns else 0
        percent = (pred_count / label_count * 100) if label_count else 0
        summary.append((cls, label_count, pred_count, f"{percent:.2f}%"))
    return summary
    
    
def run_backend_only():
    print("🔧 Running in no-GUI mode...")

    # Setup paths
    tool_dir = os.path.expanduser("~/Documents/flowcytometertool/")
    download_path = os.path.join("exampledata/")
    output_path = os.path.join("extraction/")
    cyz2json_dir = os.path.join(tool_dir, "cyz2json")
    model_dir = os.path.join(tool_dir, "models")
    plots_dir = os.path.join(tool_dir, "plots")
    model_path = os.path.join(model_dir, "final_model.pkl")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    try:
        # 1. Download Files
        blob_url = "https://citprodflowcytosa.blob.core.windows.net/public/exampledata/"
        print("⬇️ Downloading files...")
        download_blobs(blob_url, download_path)

        # 2. Download cyz2json
        print("📦 Installing requirements...")
        compile_cyz2json_from_release(cyz2json_dir, None)

        # 3. Cyz2json
        print("🔄 Converting CYZ to JSON...")
        convert_cyz_to_json(download_path, output_path, os.path.join(cyz2json_dir, "Cyz2Json.dll"))

        # 4. To listmode
        print("📄 Converting JSON to listmode...")
        convert_json_to_listmode(output_path)

        # 5. Combine CSVs
        print("📊 Combining CSV files...")
        df = combine_csvs(output_path, expertise_matrix_path, nogui=True)
        if df is None:
            print("⚠️ No CSV files found.")
            return

        # 6. Train Model
        print("🤖 Training model...")
        train_model(df, plots_dir, model_path, nogui=True, self = None)
    
        # 7. Predict Test Set using updated function
        print("🧪 Predicting test set...")
        from custom_functions_for_python import predictTestSet

        predict_name = os.path.join(tool_dir, "test_predictions.csv")
        cm_filename = os.path.join(tool_dir, "confusion_matrix.csv")
        report_filename = os.path.join(tool_dir, "classification_report.csv")
        text_file_path = os.path.join(tool_dir, "prediction_log.txt")

        with open(text_file_path, "w") as text_file:
            predictTestSet(
                self=None,
                model_path=model_path,
                predict_name=predict_name,
                data=df,
                target_name="source_label",
                weight_name="weight",
                cm_filename=cm_filename,
                report_filename=report_filename,
                text_file=text_file
            )
        print("✅ Test set predictions completed and saved.")

    except Exception as e:
        print(f"❌ Error during headless execution: {e}")
        raise
