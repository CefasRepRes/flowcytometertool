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
#import multiprocessing

class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flow Cytometry Tools")
        self.root.geometry("1800x1000")
        self.tool_dir = os.path.expanduser("~/Documents/flowcytometertool/")
        self.download_path = os.path.join(os.path.expanduser("~/Documents/flowcytometertool"), 'downloadeddata/')
        self.output_path = os.path.join(os.path.expanduser("~/Documents/flowcytometertool"), 'extraction/')
        os.makedirs(self.download_path, exist_ok=True)
        self.cyz2json_dir = os.path.join(self.tool_dir, "cyz2json")
        model_dir = os.path.join(self.tool_dir, "models")
        self.plots_dir = os.path.join(self.tool_dir, "Training plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "final_model.pkl")
        self.df = None
        self.pumped_volume = 1
        self.selector = None
        self.polygons = []
        self.predictions_datas = []
        self.current_polygon = None
        self.current_predictions_data = None
        self.dest_path = None
        self.create_widgets()
        self.path_entry = tk.Entry(self.tab_download, width=100)
        self.path_entry.insert(0, self.cyz2json_dir + "\\Cyz2Json.dll")
        self.cyz_file = os.path.join(self.tool_dir, "tempfile.cyz")
        self.json_file = os.path.join(self.tool_dir, "tempfile.json")
        self.listmode_file = os.path.join(self.tool_dir, "tempfile.csv")
        self.model_path = os.path.join(self.tool_dir, "models/final_model.pkl")

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
        self.df = combine_csvs(self.output_path, "expertise_matrix.csv", nogui=False)

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
            df = pd.read_csv("expertise_matrix.csv")
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
            df.to_csv("expertise_matrix.csv", index=False)
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
        tk.Button(self.tab_download, text="Train Model", command=lambda: train_model(self,self.df, self.plots_dir, self.model_path, nogui=False)).pack(pady=5)
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
            sas_token_path = self.sas_token_entry.get().strip()
            sas_token = get_sas_token(sas_token_path)
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
            if 'predictions_data' in self.df.columns:
                color_options.append('predictions_data')
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
            if 'label' in self.df.columns and 'predictions_data' in self.df.columns:
                self.df['agreement'] = self.df['label'] == self.df['predictions_data']
            else:
                self.df['agreement'] = pd.NA
            if 'predictions_data' not in self.df.columns:
                self.df['predictions_data'] = pd.NA
            color_options = []
            if 'label' in self.df.columns:
                color_options.append('label')
            if 'predictions_data' in self.df.columns:
                color_options.append('predictions_data')
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
        self.current_predictions_data = simpledialog.askstring("Prediction", "Enter prediction label:")
        self.commit_polygon()

    def commit_polygon(self):
        if self.df is None or self.current_polygon is None or self.current_predictions_data is None:
            return
        path = Path(self.current_polygon)
        x = self.x_variable_combobox.get()
        y = self.y_variable_combobox.get()
        mask = self.df.apply(lambda row: path.contains_point((row[x], row[y])), axis=1)
        self.df.loc[mask, 'predictions_data'] = self.current_predictions_data
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
                    "predictions_datas": self.predictions_datas
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
                prediction_counts = predictions_df['predictions_data'].value_counts().reset_index()
                prediction_counts.columns = ['class', 'count']
                prediction_counts.to_csv(prediction_counts_path, index=False)
                upload_to_blob(prediction_counts_path,  sas_token,container, output_blob_folder)
                log_message(f"Success: counted {url_notoken}")
                data = pd.read_csv(predictions_file)
                data['category'] = data['predictions_data']
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