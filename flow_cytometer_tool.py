import requests
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox
import os
import pandas as pd
import json
from listmode import extract
from tkinter import ttk
from azure.storage.blob import BlobServiceClient, BlobClient
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import csv
import re
import joblib
import tempfile
from custom_functions_for_python import buildSupervisedClassifier, loadClassifier
import functions
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from functions import *
import threading
import time
from PIL import Image, ImageTk
import platform
import urllib.request

#import multiprocessing

import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os


import pandas as pd
import numpy as np
import os


import pandas as pd
import numpy as np
import os

    
import pandas as pd
from sklearn.cluster import KMeans


import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---------------------------
# Spoof Calibration Function
# ---------------------------
def spoof_calibration(csv_path, output_path=None):
    df = pd.read_csv(csv_path)
    required_cols = ['Fl Red_total', 'Fl Orange_total', 'Fl Yellow_total']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV.")

    num_bands = 8
    band_edges = {}
    band_widths = {}
    for col in required_cols:
        col_min = df[col].min()
        col_max = np.percentile(df[col], 90)
        edges = np.linspace(col_min, col_max, num_bands + 1)
        band_edges[col] = edges
        band_widths[col] = edges[1] - edges[0]

    def get_band_index(value, edges):
        for i in range(len(edges) - 1):
            if edges[i] <= value <= edges[i + 1]:
                return i
        return None

    band_indices = pd.DataFrame(index=df.index)
    for col in required_cols:
        band_indices[col] = df[col].apply(lambda v: get_band_index(v, band_edges[col]))

    same_band_mask = (band_indices.notnull().all(axis=1)) & \
                     (band_indices[required_cols[0]] == band_indices[required_cols[1]]) & \
                     (band_indices[required_cols[1]] == band_indices[required_cols[2]])

    filtered_df = df[same_band_mask].copy()
    filtered_indices = band_indices[same_band_mask].copy()

    band_counts = filtered_indices[required_cols[0]].value_counts()
    if band_counts.empty:
        raise ValueError("No rows found after filtering for same band across all three dimensions.")

    target_count = band_counts.max()
    balanced_rows = []
    for band in range(num_bands):
        band_mask = filtered_indices[required_cols[0]] == band
        band_rows = filtered_df[band_mask].copy()
        count = len(band_rows)
        if count == 0:
            continue
        if count < target_count:
            new_rows = []
            last_row = None
            for i in range(target_count - count):
                if last_row is None:
                    base_row = band_rows.sample(n=1).iloc[0].copy()
                else:
                    base_row = last_row.copy()
                for col in required_cols:
                    width = band_widths[col]
                    jitter = np.random.uniform(-0.01 * width, 0.01 * width)
                    base_row[col] += jitter
                new_rows.append(base_row)
                last_row = base_row
            additional_rows = pd.DataFrame(new_rows)
            band_rows = pd.concat([band_rows, additional_rows], ignore_index=True)
        balanced_rows.append(band_rows)

    balanced_df = pd.concat(balanced_rows, ignore_index=True)
    balanced_df.to_csv(output_path, index=False)
    return output_path



def plot_calibration_curves_from_json(json_path):
    """
    Load calibration vertices from JSON and plot true size vs mean cluster response
    for all three axes on one plot, with error bars.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    vertices = data.get("calibration_vertices", [])
    if not vertices:
        raise ValueError("No calibration vertices found in JSON.")

    true_sizes = [v['true_size'] for v in vertices]
    axes = ['Fl Red_total', 'Fl Orange_total', 'Fl Yellow_total']
    colors = ['red', 'orange', 'gold']

    plt.figure(figsize=(8, 6))
    for axis, color in zip(axes, colors):
        means = [v['means'][axis] for v in vertices]
        stds = [v['stds'][axis] for v in vertices]
        plt.errorbar(true_sizes, means, yerr=stds, fmt='o-', capsize=5, color=color, label=axis)

    plt.xlabel('True Size (microns)')
    plt.ylabel('Mean Cluster Response')
    plt.title('Calibration Curves for All Axes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------
# Clustering Function
# ---------------------------
def cluster_colour_bands(csv_path, n_clusters=8):
    df = pd.read_csv(csv_path)
    required_cols = ['Fl Red_total', 'Fl Orange_total', 'Fl Yellow_total']
    X = df[required_cols].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return {'all_centers': kmeans.cluster_centers_, 'labels': kmeans.labels_, 'data': df}

# ---------------------------
# Plot Clusters Over Data
# ---------------------------
def plot_clusters_over_data(csv_path, cluster_centers):
    df = pd.read_csv(csv_path)
    pairs = [
        ('Fl Red_total', 'Fl Yellow_total', 'Red vs Yellow'),
        ('Fl Red_total', 'Fl Orange_total', 'Red vs Orange'),
        ('Fl Yellow_total', 'Fl Orange_total', 'Yellow vs Orange')
    ]
    for x, y, title in pairs:
        plt.figure(figsize=(7, 7))
        plt.scatter(df[x], df[y], s=10, alpha=0.3, color='gray', label='Data')
        idx_x = ['Fl Red_total', 'Fl Orange_total', 'Fl Yellow_total'].index(x)
        idx_y = ['Fl Red_total', 'Fl Orange_total', 'Fl Yellow_total'].index(y)
        plt.scatter(cluster_centers[:, idx_x], cluster_centers[:, idx_y],
                    s=120, c='red', marker='X', label='Cluster Centers')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'Cluster Centers over Data: {title}')
        plt.legend()
        plt.tight_layout()
        plt.show()

# ---------------------------
# Extract Calibration Vertices
# ---------------------------
def extract_calibration_vertices(csv_path, true_sizes):
    df = pd.read_csv(csv_path)
    required_cols = ['Fl Red_total', 'Fl Orange_total', 'Fl Yellow_total']
    X = df[required_cols].values
    kmeans = KMeans(n_clusters=8, random_state=42)
    labels = kmeans.fit_predict(X)

    cluster_means = []
    for cluster_id in range(8):
        cluster_data = df[labels == cluster_id]
        means = cluster_data[required_cols].mean().to_dict()
        stds = cluster_data[required_cols].std().to_dict()
        cluster_means.append((means['Fl Red_total'], cluster_id, means, stds))
    cluster_means.sort()

    vertices = []
    for i, (red_mean, cluster_id, means, stds) in enumerate(cluster_means):
        vertex = {
            "cluster_id": int(cluster_id),
            "true_size": float(true_sizes[i]),
            "means": {k: float(v) for k, v in means.items()},
            "stds": {k: float(v) for k, v in stds.items()}
        }
        vertices.append(vertex)
    return vertices



class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flow Cytometry Tools")
        self.root.geometry("1800x1000")
        self.tool_dir = os.path.expanduser("~/Documents/flowcytometertool/")
        self.download_path = os.path.join(os.path.expanduser("~/Documents/flowcytometertool"), 'downloadeddata/')
        self.output_path = os.path.join(os.path.expanduser("~/Documents/flowcytometertool"), 'downloadeddata/')
        os.makedirs(self.download_path, exist_ok=True)
        self.cyz2json_dir = os.path.join(self.tool_dir, "cyz2json")
        model_dir = os.path.join(self.tool_dir, "models")
        self.model_path = os.path.join(self.tool_dir, f'models/final_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        self.plots_dir = os.path.join(self.tool_dir, "Training plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.df = None
        self.pumped_volume = 1
        self.selector = None
        self.polygons = []
        self.predicted_labels = []
        self.current_polygon = None
        self.current_predicted_label = None
        self.dest_path = None
        self.create_widgets()
        self.path_entry = tk.Entry(self.tab_download, width=100)
        self.path_entry.insert(0, self.cyz2json_dir + "\\Cyz2Json.dll")
        self.cyz_file = os.path.join(self.tool_dir, "tempfile.cyz")
        self.json_file = os.path.join(self.tool_dir, "tempfile.json")
        self.listmode_file = os.path.join(self.tool_dir, "tempfile.csv")

    def process_calibration_file(self):
        try:
            cyz_path = self.calibration_file_entry.get().strip()
            if not cyz_path or not cyz_path.lower().endswith('.cyz'):
                messagebox.showerror("Error", "Please select a valid .cyz calibration file.")
                return

            # Paths
            json_path = cyz_path.replace('.cyz', '_calib.json')
            csv_path = cyz_path.replace('.cyz', '_calib.csv')
            json_export_path = cyz_path.replace('.cyz', '_calibration_vertices.json')

            # Step 1: Convert CYZ → JSON → CSV
            load_file(self.path_entry.get(), cyz_path, json_path)
            to_listmode(json_path, csv_path)


            if self.spoof_calibration_var.get():
                # User wants spoofing
                spoof_calibration(csv_path, csv_path)
                clustering_input_csv = csv_path
            else:
                # Use original CSV
                clustering_input_csv = csv_path

            # Step 3: Perform Clustering
            result = cluster_colour_bands(csv_path, n_clusters=8)
            cluster_centers = result['all_centers']

            # Step 4: Plot Data with Cluster Centers
            plot_clusters_over_data(csv_path, cluster_centers)

            # Step 5: Extract Calibration Vertices
            true_sizes = [float(x.strip()) for x in self.true_sizes_entry.get().split(',')]
            if len(true_sizes) != 8:
                messagebox.showerror("Input Error", "Please enter exactly 8 values for true sizes.")
                return
            vertices = extract_calibration_vertices(csv_path, true_sizes)
            self.calibration_vertices = {"calibration_vertices": vertices}

            # Step 6: Export Vertices to JSON
            with open(json_export_path, 'w') as f:
                json.dump(self.calibration_vertices, f, indent=2)
                
            plot_calibration_curves_from_json(json_export_path)
            self.calibration_processed = True
            self.update_start_labelling_button()
            self.calibrationfile_info = {
                "filename": cyz_path,
                "calibrationinstrumentmetadatacsv": csv_path + "instrument.csv",  # adjust if needed
                "calibration_vertices": vertices  # or self.calibration_vertices["calibration_vertices"]
            }
            messagebox.showinfo("Success", f"Calibration processing complete.\nVertices saved to:\n{json_export_path}")
        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to process calibration file:\n{e}")
            
            
    def build_plots_tab(self, notebook):
        self.tab_plots = ttk.Frame(notebook)
        notebook.add(self.tab_plots, text="Plots")
        frame = tk.Frame(self.tab_plots)
        frame.pack(fill='both', expand=True)
        self.plot_listbox = tk.Listbox(frame, width=50)
        self.plot_listbox.pack(side='left', fill='y', padx=10, pady=10)
        self.image_label = tk.Label(frame)
        self.image_label.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        def list_plot_files():
            return [f for f in os.listdir(self.plots_dir) if f.lower().endswith('.png')]

        def update_plot_list():
            current_files = list_plot_files()
            self.plot_listbox.delete(0, tk.END)
            for file in current_files:
                self.plot_listbox.insert(tk.END, file)

        def show_selected_image(event):
            selection = self.plot_listbox.curselection()
            if selection:
                filename = self.plot_listbox.get(selection[0])
                image_path = os.path.join(self.plots_dir, filename)
                try:
                    image = Image.open(image_path)
                    image.thumbnail((800, 800))  # Resize for display
                    self.tk_image = ImageTk.PhotoImage(image)
                    self.image_label.config(image=self.tk_image)
                except Exception as e:
                    messagebox.showerror("Image Error", f"Failed to load image:\n{e}")

        def watch_plots_folder():
            previous_files = set()
            while True:
                current_files = set(list_plot_files())
                if current_files != previous_files:
                    update_plot_list()
                    previous_files = current_files
                time.sleep(2)

        self.plot_listbox.bind("<<ListboxSelect>>", show_selected_image)
        threading.Thread(target=watch_plots_folder, daemon=True).start()


    def display_readme(self, parent_frame):
        try:
            if getattr(sys, 'frozen', False):
                # Running in a PyInstaller bundle
                base_path = sys._MEIPASS
            else:
                # Running in a normal Python environment
                base_path = os.path.abspath(".")
            readme_path = os.path.join(base_path, "README.md")
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
            text_widget = ScrolledText(parent_frame, wrap=tk.WORD)
            text_widget.insert(tk.END, readme_content)
            text_widget.configure(state='disabled')  # Make it read-only
            text_widget.pack(expand=True, fill='both')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load README.md:\n{e}")

    def select_local_watch_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.local_watch_folder_entry.delete(0, tk.END)
            self.local_watch_folder_entry.insert(0, folder)

    def select_local_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.local_output_folder_entry.delete(0, tk.END)
            self.local_output_folder_entry.insert(0, folder)

    def start_local_watching(self):
        watch_folder = self.local_watch_folder_entry.get().strip()
        output_folder = self.local_output_folder_entry.get().strip()
        cyz2json_path = self.local_path_entry.get().strip()
        if not watch_folder or not output_folder or not cyz2json_path:
            messagebox.showerror("Error", "Please fill in all paths.")
            return
        if hasattr(self, 'local_observer') and self.local_observer:
            self.local_observer.stop()
            self.local_observer.join()
        handler = FileHandler(cyz2json_path, output_folder, self.model_path)
        self.local_observer = Observer()
        self.local_observer.schedule(handler, watch_folder, recursive=False)
        self.local_observer.start()
        log_message(f"Started watching folder: {watch_folder}")

    def build_local_watcher_tab(self):
        tk.Label(self.tab_local_watcher, text="Path to cyz2json:").pack(pady=5)
        self.local_path_entry = tk.Entry(self.tab_local_watcher, width=100)
        self.local_path_entry.insert(0, os.path.join(self.cyz2json_dir, "Cyz2Json.dll"))
        self.local_path_entry.pack(pady=5)
        tk.Label(self.tab_local_watcher, text="Watch Folder:").pack(pady=5)
        self.local_watch_folder_entry = tk.Entry(self.tab_local_watcher, width=100)
        self.local_watch_folder_entry.pack(pady=5)
        tk.Button(self.tab_local_watcher, text="Select Watch Folder", command=self.select_local_watch_folder).pack(pady=5)
        tk.Label(self.tab_local_watcher, text="Output Folder:").pack(pady=5)
        self.local_output_folder_entry = tk.Entry(self.tab_local_watcher, width=100)
        self.local_output_folder_entry.pack(pady=5)
        tk.Button(self.tab_local_watcher, text="Select Output Folder", command=self.select_local_output_folder).pack(pady=5)
        tk.Button(self.tab_local_watcher, text="Start Watching", command=self.start_local_watching).pack(pady=10)

    def cyz2json(self):
        try:
            convert_cyz_to_json(self.download_path, self.output_path, self.cyz2json_dir + '/Cyz2Json.dll')
            messagebox.showinfo("Success", "CYZ to JSON conversion completed successfully.")
        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Conversion Error", f"Failed to convert CYZ to JSON:\n{e}")

    def install_all_requirements(self):
        self.root.update()
        if not os.path.exists(self.tool_dir):
            os.makedirs(self.tool_dir)
        self.cyz2json_dir = os.path.join(self.tool_dir, "cyz2json")
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, os.path.join(self.cyz2json_dir, "bin", "Cyz2Json.dll"))
        try:
            compile_cyz2json_from_release(self.cyz2json_dir, self.path_entry)
            messagebox.showinfo("Success", "Cyz2Json installed successfully.")
        except Exception as e:
            print(f"Installation Error: Failed to install requirements: {e}")
            messagebox.showerror("Installation Error", f"Failed to install requirements:\n{e}")
        # Only run this on Windows as linux should already have it installed
        if platform.system().lower() == "windows":
            try:
                subprocess.run(["winget", "install", "--id", "Microsoft.DotNet.SDK.8", "--source", "winget"], shell=True)
                messagebox.showinfo("Info", ".NET SDK installation started. Please follow any prompts that appear.")
                # Notify user that the app must be restarted
                should_exit = messagebox.askokcancel(
                    "Restart Required",
                    "To complete the installation, this application must now close after the .NET SDK installation.\n\n"
                    "Please reopen the app."
                )
                if should_exit:
                    self.root.destroy()
            except Exception as e:
                print(f"Failed to launch .NET SDK installation via winget: {e}")
                messagebox.showerror("Error", f"Failed to launch .NET SDK installation:\n{e}")



    def launch_sample_metadata_form(self, prefill=None):
        form = tk.Toplevel(self.root)
        form.title("Sample Metadata")
        form.geometry("500x700")
        form.grab_set()

        entries = {}

        def add_entry(label, key, default=""):
            tk.Label(form, text=label).pack()
            entry = tk.Entry(form, width=60)
            entry.insert(0, "" if default is None else default)
            entry.pack()
            entries[key] = entry

        # Use prefill or fallback to defaults
        prefill = prefill or {}
        add_entry("Location (latitude)", "latitude", prefill.get("latitude", ""))
        add_entry("Location (longitude)", "longitude", prefill.get("longitude", ""))
        add_entry("Location Approximate", "location_approximate", prefill.get("location_approximate", "UK Shelf Seas"))
        add_entry("Timestamp (ISO8601)", "timestamp", prefill.get("timestamp", ""))
        add_entry("Size (um)", "size_um", prefill.get("size_um", ""))
        add_entry("Source Vessel", "source_vessel", prefill.get("source_vessel", "RV Cefas Endeavour"))

        def submit():
            self.sample_metadata = {
                "location": {
                    "latitude": entries["latitude"].get().strip() or None,
                    "longitude": entries["longitude"].get().strip() or None
                },
                "location_approximate": entries["location_approximate"].get().strip() or None,
                "timestamp": entries["timestamp"].get().strip() or None,
                "size_um": int(entries["size_um"].get().strip()) if entries["size_um"].get().strip() else None,
                "source_vessel": entries["source_vessel"].get().strip() or None
            }
            form.destroy()
            self.start_image_labelling_workflow()

        tk.Button(form, text="Submit", command=submit).pack(pady=20)

    
    def build_individual_labelling_tab(self):
        tk.Button(self.tab_individual_labelling, text="Download cyz2json", command=self.install_all_requirements).pack(pady=10)

        # Calibration file selection
        tk.Label(self.tab_individual_labelling, text="Select Calibration File (.cyz):").pack(pady=5)
        self.calibration_file_entry = tk.Entry(self.tab_individual_labelling, width=80)
        self.calibration_file_entry.pack(pady=5)
        tk.Button(
            self.tab_individual_labelling,
            text="Browse Calibration File",
            command=lambda: self.select_cyz_file(self.calibration_file_entry)
        ).pack(pady=5)

        # Spoof calibration checkbox
        self.spoof_calibration_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.tab_individual_labelling,
            text="Spoof Calibration (If you don't have a multimodal rainbow-bead style calibration sample. Split any file into 8 non-overlapping bands)",
            variable=self.spoof_calibration_var
        ).pack(pady=5)

        tk.Label(self.tab_individual_labelling, text="Enter true micron sizes for each of the 8 clusters (comma-separated):").pack(pady=5)
        self.true_sizes_entry = tk.Entry(self.tab_individual_labelling, width=80)
        self.true_sizes_entry.insert(0, "0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0")  # Predefined sizes
        self.true_sizes_entry.pack(pady=5)

        tk.Button(
            self.tab_individual_labelling,
            text="Process Calibration File",
            command=self.process_calibration_file
        ).pack(pady=10)

        # Sample file selection
        tk.Label(self.tab_individual_labelling, text="Select Sample File (.cyz):").pack(pady=5)
        self.sample_file_entry = tk.Entry(self.tab_individual_labelling, width=80)
        self.sample_file_entry.pack(pady=5)
        tk.Button(
            self.tab_individual_labelling,
            text="Browse Sample File",
            command=lambda: self.select_cyz_file(self.sample_file_entry)
        ).pack(pady=5)        
        
        tk.Button(
            self.tab_individual_labelling,
            text="Process Sample for Individual Labelling",
            command=self.process_sample_for_individual_labelling
        ).pack(pady=10)
                
        # Add this inside build_individual_labelling_tab(self) after other buttons
        self.start_labelling_btn = tk.Button(
            self.tab_individual_labelling,
            text="Start Labelling Session",
            state=tk.DISABLED,  # Initially disabled
            command=self.start_labelling_session
        )
        self.start_labelling_btn.pack(pady=10)

        # Track processing states
        self.calibration_processed = False
        self.sample_processed = False



        
            
    def process_sample_for_individual_labelling(self):
        try:
            # Get sample file path from entry
            cyz_path = self.sample_file_entry.get().strip()
            if not cyz_path or not cyz_path.lower().endswith('.cyz'):
                messagebox.showerror("Error", "Please select a valid .cyz sample file.")
                return

            # Paths for output
            json_path = cyz_path.replace('.cyz', '_sample.json')
            csv_path = cyz_path.replace('.cyz', '_calib.csv')
            images_dir = cyz_path.replace('.cyz', '_images/')
            os.makedirs(images_dir, exist_ok=True)


            # Step 1: Convert CYZ to JSON
            load_file(self.path_entry.get(), cyz_path, json_path)

            # Step 2: Extract images using listmode_particleswithimagesonly
            import listmode_particleswithimagesonly
            data = json.load(open(json_path, encoding="utf-8-sig"))
            self.timestamp = data.get("instrument", {}).get("measurementResults", {}).get("start", "")

            # This will save images to images_dir and return particle info
            lines = listmode_particleswithimagesonly.extractimages(
                particles=data["particles"],
                dateandtime=data["instrument"]["measurementResults"]["start"],
                images=data["images"],
                save_images_to=images_dir
            )

            self.df_particles = pd.DataFrame(lines)


            # Step 3: Display images for labelling (you may want to launch your MetadataUI here)
            # Example: Launch a new window for labelling
            from metadata_ui import MetadataUI
            from metadata_handler import MetadataHandler

            metadata_handler = MetadataHandler(images_dir)
            tif_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
            # You can now use MetadataUI to display each image and collect labels
            # (You may need to adapt this to your app's navigation logic)

            self.sample_processed = True
            self.update_start_labelling_button()
            self.samplefile_info = {
                "filename": cyz_path,
                "imagesfolder": images_dir,
                "sampleinstrumentmetadatacsv": csv_path + "instrument.csv"  # adjust if needed
            }
            messagebox.showinfo("Success", f"Sample processed. Images saved to:\n{images_dir}")
        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to process sample file:\n{e}")


    def show_labelling_form(self):
        idx = self.current_image_index
        img_path = self.image_list[idx]
        prev_label = self.labels[idx] if self.labels[idx] else {}

        # --- Main window ---
        form = tk.Toplevel(self.root)
        form.title(f"Labelling Image {idx+1} of {len(self.image_list)}")
        form.geometry("1100x1100")
        form.grab_set()

        # --- Scrollable frame setup ---
        canvas = tk.Canvas(form, borderwidth=0, width=1180, height=880)
        vscroll = tk.Scrollbar(form, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        # --- Horizontal frame for image and plot ---
        image_plot_frame = tk.Frame(scroll_frame)
        image_plot_frame.pack(pady=10, fill='x')

        # --- Display Image (left) ---
        img = Image.open(img_path)
        img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        img_label = tk.Label(image_plot_frame, image=tk_img)
        img_label.image = tk_img
        img_label.pack(side='left', padx=10, pady=10)

        # --- Display Matplotlib Plot (right) ---
        plot_frame = tk.Frame(image_plot_frame)
        plot_frame.pack(side='right', padx=10, pady=10)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(
            self.df_particles['Fl Red_total'],
            self.df_particles['Fl Yellow_total'],
            c='gray', s=10, alpha=0.5, label='Particles'
        )
        ax.scatter(
            [self.df_particles.loc[idx, 'Fl Red_total']],
            [self.df_particles.loc[idx, 'Fl Yellow_total']],
            c='red', s=60, marker='x', label='Current'
        )
        ax.set_xlabel('Fl Red_total')
        ax.set_ylabel('Fl Yellow_total')
        ax.set_title('Red vs Yellow')
        ax.legend()

        mpl_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        mpl_canvas.get_tk_widget().pack()

        def on_click(event):
            if event.inaxes != ax:
                return
            x, y = event.xdata, event.ydata
            dists = ((self.df_particles['Fl Red_total'] - x)**2 +
                     (self.df_particles['Fl Yellow_total'] - y)**2)
            nearest_idx = dists.idxmin()
            self.current_image_index = nearest_idx
            form.destroy()
            self.show_labelling_form()
        mpl_canvas.mpl_connect('button_press_event', on_click)

        # --- Form fields below the image/plot row ---
        entries = {}

        def add_entry(label, key, default=""):
            tk.Label(scroll_frame, text=label).pack()
            val = prev_label.get(key, default)
            entry = tk.Entry(scroll_frame, width=60)
            entry.insert(0, val if val is not None else "")
            entry.pack()
            entries[key] = entry

        def add_dropdown(label, key, options, default=""):
            tk.Label(scroll_frame, text=label).pack()
            val = prev_label.get(key, default)
            var = tk.StringVar(value=val if val else options[0])
            dropdown = ttk.Combobox(scroll_frame, textvariable=var, values=options, state="readonly", width=57)
            dropdown.pack()
            entries[key] = dropdown

        # --- Main fields ---
        add_entry("Custom Note", "custom_note")
        add_dropdown("Certainty", "certainty", ["High", "Medium", "Low"])
        add_dropdown("Image Quality", "image_quality", ["High", "Medium", "Low"])
        add_dropdown("Class", "class", ["Organism", "Taxo_particle", "Non_taxo_particle"])

        # --- Taxonomy fields (grouped) ---
        tk.Label(scroll_frame, text="Taxonomy (leave blank if not applicable)", font=("Arial", 10, "bold")).pack(pady=5)
        taxonomy = prev_label.get("taxonomy", {}) if prev_label.get("taxonomy") else {}
        taxonomy_entries = {}
        for label, key in [("Aphia ID", "aphia_id"), ("Scientific Name", "scientific_name"), ("Taxonomic Rank", "taxonomic_rank")]:
            tk.Label(scroll_frame, text=label).pack()
            val = taxonomy.get(key, "")
            entry = tk.Entry(scroll_frame, width=60)
            entry.insert(0, val if val is not None else "")
            entry.pack()
            taxonomy_entries[key] = entry

        # --- Attributes fields (grouped) ---
        tk.Label(scroll_frame, text="Attributes", font=("Arial", 10, "bold")).pack(pady=5)
        attributes = prev_label.get("attributes", {}) if prev_label.get("attributes") else {}
        attributes_entries = {}
        for label, key in [("Life Stage", "life_stage"), ("Body Part", "body_part")]:
            tk.Label(scroll_frame, text=label).pack()
            val = attributes.get(key, "")
            entry = tk.Entry(scroll_frame, width=60)
            entry.insert(0, val if val is not None else "")
            entry.pack()
            attributes_entries[key] = entry

        # --- Navigation and Save ---
        nav_frame = tk.Frame(scroll_frame)
        nav_frame.pack(pady=20)
        tk.Button(nav_frame, text="Previous", command=lambda: save_and_prev()).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="Skip", command=lambda: skip()).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="Next", command=lambda: save_and_next()).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="Save Session", command=lambda: save_session()).pack(side=tk.RIGHT, padx=10)

        # --- Validation and navigation logic ---
        def validate_and_collect():
            for key in ["certainty", "image_quality", "class"]:
                if not entries[key].get():
                    messagebox.showerror("Validation Error", f"{key.replace('_', ' ').capitalize()} is required.")
                    return None
            taxonomy_dict = {k: v.get().strip() for k, v in taxonomy_entries.items()}
            if taxonomy_dict["aphia_id"]:
                try:
                    taxonomy_dict["aphia_id"] = int(taxonomy_dict["aphia_id"])
                except ValueError:
                    messagebox.showerror("Validation Error", "Aphia ID must be an integer or blank.")
                    return None
            else:
                taxonomy_dict = None if not any(taxonomy_entries[k].get().strip() for k in taxonomy_entries) else taxonomy_dict
            attributes_dict = {k: v.get().strip() or None for k, v in attributes_entries.items()}
            if not any(attributes_dict.values()):
                attributes_dict = None
            label = {
                "image_id": img_path,
                "custom_note": entries["custom_note"].get().strip(),
                "certainty": entries["certainty"].get(),
                "image_quality": entries["image_quality"].get(),
                "class": entries["class"].get(),
                "taxonomy": taxonomy_dict,
                "attributes": attributes_dict,
                "metadata": self.sample_metadata  # Use the sample-level metadata for every image
            }
            return label

        def save_and_next():
            label = validate_and_collect()
            if label is None:
                return
            self.labels[idx] = label
            form.destroy()
            if idx < len(self.image_list) - 1:
                self.current_image_index += 1
                self.show_labelling_form()
            else:
                messagebox.showinfo("Done", "You have labelled all images!")

        def save_and_prev():
            label = validate_and_collect()
            if label is None:
                return
            self.labels[idx] = label
            form.destroy()
            if idx > 0:
                self.current_image_index -= 1
                self.show_labelling_form()

        def skip():
            self.labels[idx] = {}  # Or mark as skipped if you wish
            form.destroy()
            if idx < len(self.image_list) - 1:
                self.current_image_index += 1
                self.show_labelling_form()

        def save_session():
            session_json = {
                "labeller": self.labeller_info,
                "samplefile": self.samplefile_info,
                "calibrationfile": self.calibrationfile_info,
                "sample_metadata": self.sample_metadata,
                "labels": self.labels
            }
            file_path = filedialog.asksaveasfilename(defaultextension=".json")
            if file_path:
                with open(file_path, "w") as f:
                    json.dump(session_json, f, indent=2)
                messagebox.showinfo("Saved", f"Session saved to {file_path}")




    # Add helper to enable button only when both processed
    def update_start_labelling_button(self):
        if self.calibration_processed and self.sample_processed:
            self.start_labelling_btn.config(state=tk.NORMAL)
        else:
            self.start_labelling_btn.config(state=tk.DISABLED)

    # Define start_labelling_session method
    def start_labelling_session(self):
        # This will launch the labeller metadata form and then image labelling workflow
        self.launch_labeller_metadata_form()
        
    def launch_labeller_metadata_form(self):
        # Create modal window
        form = tk.Toplevel(self.root)
        form.title("Labeller Metadata")
        form.geometry("600x1000")
        form.grab_set()  # Make modal

        # Dictionary to hold user inputs
        inputs = {}

        # Helper to create labeled entry
        def add_entry(label_text, key, default=""):
            tk.Label(form, text=label_text).pack(pady=3)
            entry = tk.Entry(form, width=50)
            entry.insert(0, default)
            entry.pack(pady=3)
            inputs[key] = entry

        # Helper to create dropdown
        def add_dropdown(label_text, key, options):
            tk.Label(form, text=label_text).pack(pady=3)
            var = tk.StringVar(value=options[0])
            dropdown = ttk.Combobox(form, textvariable=var, values=options, state="readonly", width=47)
            dropdown.pack(pady=3)
            inputs[key] = dropdown

        # Add fields
        add_entry("Name:", "name", "Joseph Ribeiro")
        add_entry("Institute:", "institute", "Cefas")
        add_entry("Email:", "email", "joseph.ribeiro@cefas.co.uk")
        add_dropdown("Confidence Level:", "confidence_level", ["Low", "Medium", "High"])
        add_dropdown("Familiar with labelling samples from this region?", "familiar_with_water", ["Yes", "Kinda", "No - first time"])
        add_entry("Years in labelling role:", "years_in_labelling_role", "0")
        add_dropdown("Do you label every week?", "do_you_label_every_week", ["Yes", "No"])
        add_dropdown("Do you label every month?", "do_you_label_every_month", ["Yes", "No"])
        add_dropdown("Do you label every year?", "do_you_label_every_year", ["Yes", "No"])
        add_dropdown("Will you be labelling taxonomy?", "will_you_be_labelling_taxonomy", ["Yes", "No"])
        add_dropdown("Intended taxonomy level:", "intended_taxonomy_level", ["species", "genus", "family"])
        add_entry("If not taxonomic, alternative strategy:", "if_not_taxonomic_state_alternative_labelling_strategy", "e.g. by functional group, living/nonliving, or leave empty")

        # Submit button
        def submit():
            # Validate email
            email = inputs["email"].get().strip()
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                messagebox.showerror("Validation Error", "Please enter a valid email address.")
                return

            # Validate years
            try:
                years = int(inputs["years_in_labelling_role"].get().strip())
                if years < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Validation Error", "Years in labelling role must be a non-negative integer.")
                return

            # Collect all values
            self.labeller_info = {key: widget.get().strip() for key, widget in inputs.items()}

            # Close form
            form.destroy()
            # Next step: start image labelling workflow
            prefill = {
                "latitude": None,
                "longitude": None,
                "location_approximate": "UK Shelf Seas",
                "timestamp": self.timestamp,
                "size_um": None,
                "source_vessel": "RV Cefas Endeavour"
            }
            self.launch_sample_metadata_form(prefill=prefill)
        tk.Button(form, text="Submit", command=submit).pack(pady=20)
            
    def start_image_labelling_workflow(self):
        # Prepare image list and state
        images_dir = self.sample_file_entry.get().replace('.cyz', '_images/')
        self.image_list = sorted(
            [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.tif')],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.current_image_index = 0
        self.labels = [{} for _ in self.image_list]  # Pre-fill with empty dicts

        # Launch the first image labelling form
        self.show_labelling_form()

    def select_cyz_file(self, entry_widget):
        file_path = filedialog.askopenfilename(filetypes=[("CYZ files", "*.cyz")])
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)


    def create_widgets(self):
        #self.redirect_stdout_to_gui() This seems to interfere with the model training functions
        tk.Label(self.root, text=f"Working Directory: {self.tool_dir}", fg="gray").pack(pady=(10, 0))
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both')
        self.tab_readme = ttk.Frame(notebook)
        notebook.add(self.tab_readme, text="README")
        self.display_readme(self.tab_readme)
        self.tab_download = ttk.Frame(notebook)
        self.tab_visualize = ttk.Frame(notebook)
        notebook.add(self.tab_download, text="Download & Train")
        self.build_plots_tab(notebook)         
        notebook.add(self.tab_visualize, text="Visualize & Label")
        self.build_download_tab()
        self.build_visualization_tab()
        self.build_blob_tools_tab()
        self.tab_local_watcher = ttk.Frame(notebook)
        notebook.add(self.tab_local_watcher, text="Local Watcher")
        self.build_local_watcher_tab()
        self.tab_individual_labelling = ttk.Frame(notebook)
        notebook.add(self.tab_individual_labelling, text="Individual Labelling")
        self.build_individual_labelling_tab()
        

    def redirect_stdout_to_gui(self):
        self.log_output = ScrolledText(self.root, height=10, state='disabled')
        self.log_output.pack(fill='both', padx=10, pady=5)

        class StdoutRedirector:
            def __init__(inner_self, widget):
                inner_self.widget = widget

            def write(inner_self, message):
                inner_self.widget.configure(state='normal')
                inner_self.widget.insert('end', message)
                inner_self.widget.configure(state='disabled')
                inner_self.widget.see('end')

            def flush(inner_self):
                pass

        sys.stdout = StdoutRedirector(self.log_output)
        sys.stderr = StdoutRedirector(self.log_output)

        
        

    def generate_mixfile(self):
        try:
            container = self.url_entry_blob.get().strip()
            print(container)
            sas_token_path = self.sas_token_entry.get().strip()
            sample_rate = float(self.sample_rate_entry.get().strip())
            sas_token = get_sas_token(sas_token_path)
            mix_blob_files(container, sas_token, self.output_blob_folder.get().strip(), sample_rate)
            messagebox.showinfo("Success", "Mixfile generated and uploaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate mixfile: {e}")

    def handle_combine_csvs(self):
        self.df = combine_csvs(self.output_path, expertise_matrix_path, nogui=False, prompt_merge_fn=self.prompt_class_grouping)

    def prompt_class_grouping(self,df):
        if df is None or 'source_label' not in df.columns:
            return

        import tkinter as tk
        from tkinter import simpledialog, messagebox

        while True:
            label_list = sorted(df['source_label'].dropna().unique())
            if len(label_list) <= 1:
                break
            top = tk.Toplevel(self.root)
            top.title("Merge Class Labels")

            listbox = tk.Listbox(top, selectmode=tk.MULTIPLE, width=50)
            for label in label_list:
                listbox.insert(tk.END, label)
            listbox.pack(padx=10, pady=10)

            def merge_selected():
                indices = listbox.curselection()
                if not indices:
                    top.destroy()
                    return
                selected_labels = [label_list[i] for i in indices]
                new_label = simpledialog.askstring("New Label", f"Merge {selected_labels} into:")
                if new_label:
                    df['source_label'] = df['source_label'].replace({lbl: new_label for lbl in selected_labels})
                    messagebox.showinfo("Merged", f"Merged {selected_labels} into {new_label}")
                top.destroy()

            merge_button = tk.Button(top, text="Merge Selected", command=merge_selected)
            merge_button.pack(pady=5)

            top.grab_set()
            self.root.wait_window(top)

            cont = messagebox.askyesno("Continue?", "Do you want to merge more class labels?")
            if not cont:
                break

        self.refresh_comboboxes()


    def handle_predict_test_set(self):
        if self.df is None:
            messagebox.showerror("Error", "No dataset loaded. Please load or combine CSVs first.")
            return
        try:
            predict_name = os.path.join(self.tool_dir, "test_predictions.csv")
            cm_filename = os.path.join(self.tool_dir, "confusion_matrix.csv")
            report_filename = os.path.join(self.tool_dir, "classification_report.csv")
            text_file = open(os.path.join(self.tool_dir, "prediction_log.txt"), "w")
            from custom_functions_for_python import predictTestSet
            predictTestSet(self,
                model_path=self.model_path,
                predict_name=predict_name,
                data=self.df,
                target_name="source_label",
                weight_name="weight",
                cm_filename=cm_filename,
                report_filename=report_filename,
                text_file=text_file
            )
            text_file.close()
            self.refresh_comboboxes()
            #self.update_plot()
            #self.update_summary_table()
            messagebox.showinfo("Success", "Test set predictions completed and saved.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict test set:\n{e}")


    def build_expertise_matrix_editor(self, parent_frame):
        tk.Label(parent_frame, text="Edit expertise levels assigned to your dataset (optional):").pack(pady=(20, 5))
        tree_frame = tk.Frame(parent_frame)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(tree_frame, show="headings")
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)

        try:
            df = pd.read_csv(expertise_matrix_path)
            self.expertise_df = df
            self.tree["columns"] = list(df.columns)

            for col in df.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100)

            for _, row in df.iterrows():
                self.tree.insert("", "end", values=list(row))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load expertise_matrix.csv:\n{e}")
            return

        self.tree.bind("<Double-1>", self.on_double_click)

        save_btn = tk.Button(parent_frame, text="Save Expertise Matrix", command=self.save_expertise_matrix)
        save_btn.pack(pady=10)


    def on_double_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        row_id = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        col_index = int(column[1:]) - 1

        x, y, width, height = self.tree.bbox(row_id, column)
        value = self.tree.set(row_id, column)

        entry = tk.Entry(self.tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, value)
        entry.focus()

        def on_focus_out(event):
            new_value = entry.get()
            if col_index > 0:  # Only validate numeric columns
                if new_value not in {"1", "2", "3"}:
                    messagebox.showerror("Invalid Input", "Please enter 1, 2, or 3.")
                    entry.destroy()
                    return
            self.tree.set(row_id, column, new_value)
            entry.destroy()

        entry.bind("<FocusOut>", on_focus_out)
        entry.bind("<Return>", lambda e: on_focus_out(e))

    def save_expertise_matrix(self):
        try:
            rows = []
            for item in self.tree.get_children():
                rows.append(self.tree.item(item)["values"])
            df = pd.DataFrame(rows, columns=self.expertise_df.columns)
            df.to_csv(expertise_matrix_path, index=False)
            messagebox.showinfo("Saved", "Expertise matrix saved successfully.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file:\n{e}")



    def build_download_tab(self):
        tk.Label(self.tab_download, text="Blob Directory URL:").pack(pady=5)
        self.url_entry = tk.Entry(self.tab_download, width=80)
        self.url_entry.insert(0, "https://citprodflowcytosa.blob.core.windows.net/public/exampledata/")
        #self.url_entry.insert(0, "https://citprodflowcytosa.blob.core.windows.net/labelledmultipleexperts3seas/external/") # This dataset depends on an SAS token having been passed in on the blob tools tab.
        self.url_entry.pack(pady=5)
        tk.Button(self.tab_download, text="Download Files", command=self.download_blob_directory).pack(pady=5)
        tk.Button(self.tab_download, text="Download cyz2json", command=self.install_all_requirements).pack(pady=5)
        tk.Button(self.tab_download, text="Cyz2json", command=self.cyz2json).pack(pady=5)
        tk.Button(self.tab_download, text="To listmode", command=self.to_listmode).pack(pady=5)
        tk.Button(self.tab_download, text="Combine CSVs", command=self.handle_combine_csvs).pack(pady=5)
        tk.Label(self.tab_download, text="Max samples per class:").pack(pady=5)
        self.max_per_class_entry = tk.Entry(self.tab_download, width=10)
        self.max_per_class_entry.insert(0, "100000")
        self.max_per_class_entry.pack(pady=5)
        self.calibration_var = tk.BooleanVar(value=False)  # default is OFF because of this pickling issue
        tk.Checkbutton(self.tab_download, text="Enable Calibration", variable=self.calibration_var).pack(pady=5)           
        tk.Button(self.tab_download, text="Train Model", command=lambda: train_model(self.df, self.plots_dir, self.model_path, calibration_enabled=self.calibration_var.get(), nogui=False, self=self, max_per_class = int(self.max_per_class_entry.get()))).pack(pady=5)
        tk.Button(self.tab_download, text="Predict Test Set", command=self.handle_predict_test_set).pack(pady=5)
        self.build_expertise_matrix_editor(self.tab_download)



    def build_visualization_tab(self):
        self.x_variable_combobox = ttk.Combobox(self.tab_visualize)
        self.y_variable_combobox = ttk.Combobox(self.tab_visualize)
        self.color_variable_combobox = ttk.Combobox(self.tab_visualize)
        self.x_variable_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.y_variable_combobox.grid(row=0, column=2, padx=5, pady=5)
        self.color_variable_combobox.grid(row=1, column=0, padx=5, pady=5)
        tk.Button(self.tab_visualize, text="Load CSV", command=self.load_csv).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(self.tab_visualize, text="Load JSON", command=self.load_json).grid(row=0, column=3, padx=5, pady=5)
        tk.Button(self.tab_visualize, text="Save CSV", command=self.save_csv).grid(row=0, column=4, padx=5, pady=5)
        self.pumped_volume_label = tk.Label(self.tab_visualize, text="")
        self.pumped_volume_label.grid(row=1, column=3, padx=5, pady=5)
        self.xmin_entry = tk.Entry(self.tab_visualize)
        self.xmax_entry = tk.Entry(self.tab_visualize)
        self.ymin_entry = tk.Entry(self.tab_visualize)
        self.ymax_entry = tk.Entry(self.tab_visualize)
        self.xmin_entry.grid(row=2, column=1)
        self.xmax_entry.grid(row=2, column=3)
        self.ymin_entry.grid(row=3, column=1)
        self.ymax_entry.grid(row=3, column=3)
        tk.Label(self.tab_visualize, text="X min").grid(row=2, column=0)
        tk.Label(self.tab_visualize, text="X max").grid(row=2, column=2)
        tk.Label(self.tab_visualize, text="Y min").grid(row=3, column=0)
        tk.Label(self.tab_visualize, text="Y max").grid(row=3, column=2)
        self.log_x_var = tk.BooleanVar()
        self.log_y_var = tk.BooleanVar()
        tk.Checkbutton(self.tab_visualize, text="Log X", variable=self.log_x_var).grid(row=4, column=0)
        tk.Checkbutton(self.tab_visualize, text="Log Y", variable=self.log_y_var).grid(row=4, column=1)
        tk.Button(self.tab_visualize, text="Update Plot", command=self.update_plot).grid(row=1, column=1, columnspan=2)
        self.summary_table = ttk.Treeview(self.tab_visualize, columns=("Instance", "Labelled", "Predicted", "%"), show='headings')
        for col in ("Instance", "Labelled", "Predicted", "%"):
            self.summary_table.heading(col, text=col)
        self.summary_table.grid(row=5, column=0, columnspan=6, padx=10, pady=10)


    def build_blob_tools_tab(self):
        self.tab_blob_tools = ttk.Frame(self.root)
        self.root.nametowidget(".!notebook").add(self.tab_blob_tools, text="Blob Tools")

        # SAS Token Input
        tk.Label(self.tab_blob_tools, text="SAS Token File Path:").pack(pady=5)
        self.sas_token_entry = tk.Entry(self.tab_blob_tools, width=100)
        self.sas_token_entry.insert(0, "C:/Users/JR13/Documents/authenticationkeys/flowcytosaSAS.txt")
        self.sas_token_entry.pack(pady=5)

        # Sample Rate Input
        tk.Label(self.tab_blob_tools, text="Sample Rate (e.g., 0.005):").pack(pady=5)
        self.sample_rate_entry = tk.Entry(self.tab_blob_tools, width=20)
        self.sample_rate_entry.insert(0, "0.005")
        self.sample_rate_entry.pack(pady=5)

        # Container URLs
        tk.Label(self.tab_blob_tools, text="Blob Container URL:").pack(pady=5)
        self.url_entry_blob = tk.Entry(self.tab_blob_tools, width=100)
        self.url_entry_blob.insert(0, "https://citprodflowcytosa.blob.core.windows.net/hdduploadnov2024")
        self.url_entry_blob.pack(pady=5)
        
        tk.Label(self.tab_blob_tools, text="Output Container Name:").pack(pady=5)
        self.output_blob_folder = tk.Entry(self.tab_blob_tools, width=100)
        self.output_blob_folder.insert(0, "blob_tool_outputs")  # default value
        self.output_blob_folder.pack(pady=5)

        # Buttons
        tk.Button(self.tab_blob_tools, text="Process all cyz files in blob store", command=self.process_all).pack(pady=10)
        tk.Button(self.tab_blob_tools, text="Generate Mixfile of prediction csvs", command=self.generate_mixfile).pack(pady=5)



    def build_process_blob_tab(self):
        tk.Label(self.tab_process_blob, text="SAS Token File Path:").pack(pady=5)
        self.sas_token_entry = tk.Entry(self.tab_process_blob, width=100)
        self.sas_token_entry.insert(0, "C:/Users/JR13/Documents/authenticationkeys/flowcytosaSAS.txt")
        self.sas_token_entry.pack(pady=5)

        tk.Label(self.tab_process_blob, text="Blob Container URL:").pack(pady=5)
        self.url_entry_blob = tk.Entry(self.tab_process_blob, width=100)
        self.url_entry_blob.insert(0, "https://citprodflowcytosa.blob.core.windows.net/hdduploadnov2024")
        self.url_entry_blob.pack(pady=5)

        tk.Button(self.tab_process_blob, text="Generate Mixfile", command=self.generate_mixfile).pack(pady=5)
        tk.Button(self.tab_process_blob, text="Process All", command=self.process_all).pack(pady=10)

    def download_blob_directory(self):
        try:
            try:
                sas_token_path = self.sas_token_entry.get().strip()
                sas_token = get_sas_token(sas_token_path)
            except Exception as e:
                messagebox.showerror("Token Error", f"Unable to load data access authentication token from the path given in 'blob tools' tab, you can ignore this error if you are using the Cefas public blob folder 'exampledata': {e}")            
                sas_token = ''
            blob_url = self.url_entry.get()
            if blob_url == "https://citprodflowcytosa.blob.core.windows.net/public/exampledata/":
                download_blobs(blob_url, self.download_path) # Pass no authentication for this folder, it is public data and actually by passing the key for another folder you get an error for this public folder
            else:                
                download_blobs(blob_url, self.download_path,sas_token)
            messagebox.showinfo("Success", "Files downloaded successfully.")
        except Exception as e:
            messagebox.showerror("Download Error", f"Failed to download files: {e}")

    def to_listmode(self):
        try:
            convert_json_to_listmode(self.output_path)
            messagebox.showinfo("Success", "Listmode extracted successfully.")
        except Exception as e:
            print(f"Processing Error: {e}")
            messagebox.showerror("Download Error", f"Failed to extract listmodes: {e}")

    def refresh_comboboxes(self):
        if self.df is not None:
            variables = list(self.df.columns)
            color_options = []
            if 'label' in self.df.columns:
                color_options.append('label')
            if 'predicted_label' in self.df.columns:
                color_options.append('predicted_label')
            if 'agreement' in self.df.columns:
                color_options.append('agreement')
            if not color_options:
                color_options = [col for col in self.df.columns if self.df[col].nunique() <= 50]
            self.x_variable_combobox['values'] = variables
            self.y_variable_combobox['values'] = variables
            self.color_variable_combobox['values'] = color_options
            if 'FWS_total' in variables and 'FWS_maximum' in variables:
                self.x_variable_combobox.set('FWS_total')
                self.y_variable_combobox.set('FWS_maximum')
            else:
                self.x_variable_combobox.set(variables[0])
                self.y_variable_combobox.set(variables[1] if len(variables) > 1 else variables[0])
            default_color = 'label' if 'label' in self.df.columns else (color_options[0] if color_options else variables[0])
            self.color_variable_combobox.set(default_color)


    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            if 'label' in self.df.columns and 'predicted_label' in self.df.columns:
                self.df['agreement'] = self.df['label'] == self.df['predicted_label']
            else:
                self.df['agreement'] = pd.NA
            if 'predicted_label' not in self.df.columns:
                self.df['predicted_label'] = pd.NA
            color_options = []
            if 'label' in self.df.columns:
                color_options.append('label')
            if 'predicted_label' in self.df.columns:
                color_options.append('predicted_label')
            if 'agreement' in self.df.columns:
                color_options.append('agreement')
            variables = self.df.columns.tolist()
            self.x_variable_combobox['values'] = variables
            self.y_variable_combobox['values'] = variables
            self.color_variable_combobox['values'] = color_options if color_options else [col for col in self.df.columns if self.df[col].nunique() <= 50]
            if 'FWS_total' in variables and 'FWS_maximum' in variables:
                self.x_variable_combobox.set('FWS_total')
                self.y_variable_combobox.set('FWS_maximum')
            else:
                self.x_variable_combobox.set(variables[0])
                self.y_variable_combobox.set(variables[1])
            self.color_variable_combobox.set('label' if 'label' in self.df.columns else (color_options[0] if color_options else variables[0]))  # Default color variable
            self.update_plot()
            self.update_summary_table()


    def load_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.pumped_volume = data.get("pumpedVolume", 1)
            self.pumped_volume_label.config(text=f"Pumped Volume: {self.pumped_volume}")
            self.update_summary_table()

    def update_plot(self):
        if self.df is None:
            return
        x = self.x_variable_combobox.get()
        y = self.y_variable_combobox.get()
        color = self.color_variable_combobox.get()
        fig, ax = plt.subplots()
        categories = self.df[color].dropna().unique()
        colors = plt.cm.get_cmap('tab10', len(categories))
        for i, cat in enumerate(categories):
            subset = self.df[self.df[color] == cat]
            ax.scatter(subset[x], subset[y], label=cat, alpha=0.5, s=10, color=colors(i))
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if self.log_x_var.get():
            ax.set_xscale('log')
        if self.log_y_var.get():
            ax.set_yscale('log')
        ax.legend()
        if hasattr(self, 'plot_area'):
            self.plot_area.get_tk_widget().destroy()
        self.plot_area = FigureCanvasTkAgg(fig, master=self.tab_visualize)
        self.plot_area.get_tk_widget().grid(row=6, column=0, columnspan=6)
        self.selector = PolygonSelector(ax, self.onselect)

    def onselect(self, verts):
        self.current_polygon = verts
        self.current_predicted_label = simpledialog.askstring("Prediction", "Enter prediction label:")
        self.commit_polygon()

    def commit_polygon(self):
        if self.df is None or self.current_polygon is None or self.current_predicted_label is None:
            return
        path = Path(self.current_polygon)
        x = self.x_variable_combobox.get()
        y = self.y_variable_combobox.get()
        mask = self.df.apply(lambda row: path.contains_point((row[x], row[y])), axis=1)
        self.df.loc[mask, 'predicted_label'] = self.current_predicted_label
        self.update_summary_table()
        self.update_plot()

    def update_summary_table(self):
        if self.df is None:
            return
        self.summary_table.delete(*self.summary_table.get_children())
        summary = summarize_predictions(self.df, self.pumped_volume)
        for row in summary:
            self.summary_table.insert("", "end", values=row)

    def save_csv(self):
        if self.df is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.df.to_csv(file_path, index=False)
                metadata = {
                    "polygons": self.polygons,
                    "predicted_labels": self.predicted_labels
                }
                with open(file_path.replace(".csv", "_metadata.json"), 'w') as f:
                    json.dump(metadata, f)

    def process_all(self):
        container = self.url_entry_blob.get().strip()#
        output_blob_folder = self.output_blob_folder.get().strip()
        print(container)
        container_url = self.url_entry_blob.get().strip()
        sas_token_path = self.sas_token_entry.get().strip()
        sas_token = get_sas_token(sas_token_path)
        blob_files = list_blobs(container_url, sas_token)
        processed_files = set()
        log_file_path = "process_log.txt"
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as log_file:
                for line in log_file:
                    processed_url = extract_processed_url(line)
                    if processed_url:
                        processed_files.add(processed_url)
        blob_files = [blob_file for blob_file in blob_files if f"{container_url}/{blob_file}" not in processed_files]
        for blob_file in blob_files:
            instrument_file = os.path.join(self.tool_dir, f"{os.path.basename(blob_file)}_instrument.csv")
            predictions_file = os.path.join(self.tool_dir, f"{os.path.basename(blob_file)}_predictions.csv")
            prediction_counts_path = predictions_file + "_counts.csv"
            plot3d_prediction_path = predictions_file + "_3d.html"
            url = f"{container_url}/{blob_file}{sas_token}"
            url_notoken = f"{container_url}/{blob_file}"
            for file_path in [plot3d_prediction_path,instrument_file,predictions_file,prediction_counts_path]:
                try:
                    delete_file(file_path)
                except Exception as e:
                    log_message(f"No file to delete: {e} (this is fine)")
            try:
                downloaded_file = download_file(url, self.tool_dir, self.cyz_file)
                log_message(f"Success: Blob downloaded for {url_notoken}")
                load_file(self.path_entry.get(), downloaded_file, self.json_file)
                log_message(f"Success: Cyz2json applied {url_notoken}")
                to_listmode(self.json_file, self.listmode_file)
                os.rename(self.listmode_file + "instrument.csv", instrument_file)
                log_message(f"Success: Listmode applied {url_notoken}")
                upload_to_blob(instrument_file,  sas_token,container,output_blob_folder)
                log_message(f"Success: Uploaded {url_notoken}")
                apply_python_model(self.listmode_file, predictions_file, self.model_path)
                log_message(f"Success: Blob downloaded and inferences made for {url_notoken}")
                upload_to_blob(predictions_file,  sas_token, container,output_blob_folder)
                log_message(f"Success: Uploaded {url_notoken}")
                predictions_df = pd.read_csv(predictions_file)
                prediction_counts = predictions_df['predicted_label'].value_counts().reset_index()
                prediction_counts.columns = ['class', 'count']
                prediction_counts.to_csv(prediction_counts_path, index=False)
                upload_to_blob(prediction_counts_path,  sas_token,container, output_blob_folder)
                log_message(f"Success: counted {url_notoken}")
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
                x_99 = np.percentile(data["Fl Yellow_total"], 99.5)
                y_99 = np.percentile(data["Fl Red_total"], 99.5)
                z_99 = np.percentile(data["Fl Orange_total"], 99.5)
                scatter = go.Scatter3d(
                    x=data["Fl Yellow_total"],
                    y=data["Fl Red_total"],
                    z=data["Fl Orange_total"],
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
                        xaxis=dict(range=[0, x_99], title="Fl Yellow_total"),
                        yaxis=dict(range=[0, y_99], title="Fl Red_total"),
                        zaxis=dict(range=[0, z_99], title="Fl Orange_total"),
                        camera=camera
                    ),
                    title='3D Data Points'
                )
                pio.write_html(fig, file=plot3d_prediction_path, auto_open=False)
                upload_to_blob(plot3d_prediction_path,  sas_token,container,output_blob_folder)
                log_message("Plot saved as '3D_Plot.html'.")
                delete_file(plot3d_prediction_path)
                delete_file(instrument_file)
                delete_file(predictions_file)
                delete_file(prediction_counts_path)
            except Exception as e:
                log_message(f"Error: An error occurred processing {url_notoken}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui", action="store_true", help="Run backend logic without launching the GUI")
    args = parser.parse_args()

    if args.nogui:
        run_backend_only()
    else:
        root = tk.Tk()
        app = UnifiedApp(root)
        root.mainloop()