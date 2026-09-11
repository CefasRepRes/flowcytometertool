import requests
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import os
import pandas as pd
import json
from flowcytometer_tool.tabs.download_train.listmode import extract
from tkinter import ttk
from azure.storage.blob import BlobServiceClient, BlobClient
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import csv
import re
import joblib
import tempfile
from flowcytometer_tool.misc.convert_json_to_listmode import convert_json_to_listmode
from flowcytometer_tool.tabs.download_train.custom_functions_for_python import buildSupervisedClassifier, loadClassifier
import flowcytometer_tool.misc.functions as functions
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from flowcytometer_tool.misc.functions import *
import threading
import time
from PIL import Image, ImageTk
import platform
import urllib.request
import flowcytometer_tool.tabs.continuous_sample_analyser.qc_plots as qc_plots
from flowcytometer_tool.tabs.continuous_sample_analyser.protocols import (
    detect_sampling_protocol,
    is_bead_sample,
)
import webbrowser
from flowcytometer_tool.tabs.continuous_sample_analyser.json_safe import json_safe
from flowcytometer_tool.config.runtime import get_runtime_config
import argparse
from watchdog.observers import Observer
#import multiprocessing

from flowcytometer_tool.misc.xmlfunctions import (
    build_consensual_dataset_from_selected_cyzs_and_xmls,
)

from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from flowcytometer_tool.misc.functions import (
    list_available_model_versions,
    set_active_model,
    set_active_uncalibrated_model,
    set_active_beadcalibrated_model,
    load_app_config,
    active_model_dir,
    resolve_active_model_path,
    resolve_active_raw_model_path,
    resolve_active_beadcalibrated_model_path,
)

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")
    
expertise_matrix_path = os.path.join(base_path, "..", "matrices", "expertise_matrix.csv")



class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flow Cytometry Tools")
        self.root.geometry("1800x1000")
        runtime_cfg = get_runtime_config()
        self.tool_dir = str(runtime_cfg.paths.tool_dir)
        self.download_path = os.path.join(self.tool_dir, 'downloadeddata/')
        self.output_path = os.path.join(self.tool_dir, 'downloadeddata/')
        os.makedirs(self.download_path, exist_ok=True)
        self.cyz2json_dir = os.path.join(self.tool_dir, "cyz2json")
        self.model_dir = os.path.join(self.tool_dir, "models") 
        model_dir = os.path.join(self.tool_dir, "models")
        self.modeltrainsettings_out = os.path.join(model_dir, "modeltrainsettings.json")        
        self.model_path = os.path.join(self.tool_dir, f'models/dummynamemodel.pkl')
        self.plots_dir = os.path.join(self.tool_dir, "Training plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.df = None
        self.bead_samples = None
        self.dest_path = None
        self.create_widgets()
        self.path_entry = tk.Entry(self.tab_download, width=100)
        self.path_entry.insert(0, self.cyz2json_dir + "\\Cyz2Json.dll")
        self.cyz_file = os.path.join(self.tool_dir, "tempfile.cyz")
        self.json_file = os.path.join(self.tool_dir, "tempfile.json")
        self.listmode_file = os.path.join(self.tool_dir, "tempfile.csv")
        self.selected_model_dir = str(runtime_cfg.paths.selected_uncalibrated_model_dir)
        os.makedirs(self.selected_model_dir, exist_ok=True)

    def handle_nn_cleaning(self):
        if self.df is None:
            messagebox.showerror("Error", "No dataset loaded. Combine CSVs first.")
            return
        try:
            from flowcytometer_tool.misc.functions import nn_homogenize_df, plot_3d_fluorescence_premerge

            out_html = os.path.join(self.plots_dir, "pre_nn_cleaning_3d.html")
            plot_3d_fluorescence_premerge(
                self.df,
                label_col="source_label",
                out_html=out_html
            )
            # Clean the df
            cleaned_df = nn_homogenize_df(
                self.df,
                label_col="source_label",
                feature_cols=("FWS_total", "Fl_Red_total", "Fl_Orange_total"),
                keep_unconsidered="keep",
                downsample_n=None
            )
            # Plot cleaned result
            self.df = cleaned_df
            out_html2 = os.path.join(self.plots_dir, "post_nn_cleaning_3d.html")
            plot_3d_fluorescence_premerge(
                self.df,
                label_col="source_label",
                out_html=out_html2
            )            
            functions.log_message(f"NN-cleaned 3D plot written: {out_html}")
            # Update stored df (optional)
            messagebox.showinfo("NN Cleaning Complete", f"Done! Cleaned df has {len(cleaned_df)} rows.")
            # update modeltrainsettings
            mts_path = self.modeltrainsettings_out
            with open(mts_path, "r") as f:
                mts = json.load(f)
            cleaning = {}
            cleaning["post_merge_nn_cleaning_ran"] = "True"
            cleaning["max_per_class_entry"] = self.max_per_class_entry.get()
            mts["cleaning"] = cleaning
            mts = json_safe(mts)
            with open(mts_path, "w") as f:
                json.dump(mts, f, indent=2)
                
            
            
        except Exception as e:
            messagebox.showerror("NN Cleaning Error", f"Failed during NN cleaning: {e}")

    def prompt_delete_labels(self, df):
        import tkinter as tk
        from tkinter import messagebox, filedialog, simpledialog

        labels = sorted(df['source_label'].dropna().unique())
        if not labels:
            return

        top = tk.Toplevel(self.root)
        top.title("Delete Label Groups")

        listbox = tk.Listbox(top, selectmode=tk.MULTIPLE, width=50, height=30)
        for label in labels:
            listbox.insert(tk.END, label)
        listbox.pack(padx=10, pady=10)

        def do_delete():
            indexes = listbox.curselection()
            if not indexes:
                top.destroy()
                return
            selected = [labels[i] for i in indexes]
            df.drop(df[df['source_label'].isin(selected)].index, inplace=True)
            record_label_delete(self.label_change_log_path, selected)
            messagebox.showinfo("Deleted", f"Removed: {selected}")
            top.destroy()

        tk.Button(top, text="Delete selected", command=do_delete).pack(pady=10)
        top.grab_set()
        self.root.wait_window(top)        

    def display_readme(self, parent_frame):
        try:
            if getattr(sys, 'frozen', False):
                # Running in a PyInstaller bundle
                base_path = sys._MEIPASS
            else:
                # Running in a normal Python environment
                base_path = os.path.abspath(".")
            readme_path = os.path.join(base_path, "..", "readme.md")
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
        self.output_folder = output_folder
        if hasattr(self, "_watcher_set_folder"):
            self._watcher_set_folder(output_folder)
        if hasattr(self, 'local_observer') and self.local_observer:
            self.local_observer.stop()
            self.local_observer.join()
        handler = FileHandler(cyz2json_path, output_folder, self.model_path)
        self.local_observer = Observer()
        self.local_observer.schedule(handler, watch_folder, recursive=True)
        self.local_observer.start()
        log_message(f"Started watching folder: {watch_folder}")

    def build_local_watcher_tab(self):
        tk.Label(self.tab_local_watcher, text="Path to cyz2json:").pack(pady=5)
        self.local_path_entry = tk.Entry(self.tab_local_watcher, width=100)
        self.local_path_entry.insert(0, os.path.join(self.cyz2json_dir, "Cyz2Json.dll"))
        self.local_path_entry.pack(pady=5)

        tk.Label(self.tab_local_watcher, text="Watch Folder (you may need to map as a network drive first):").pack(pady=5)
        self.local_watch_folder_entry = tk.Entry(self.tab_local_watcher, width=100)
        self.local_watch_folder_entry.pack(pady=5)
        self.local_watch_folder_entry.insert(0, r"A:/")
        tk.Button(self.tab_local_watcher, text="Select Watch Folder", command=self.select_local_watch_folder).pack(pady=5)

        tk.Label(self.tab_local_watcher, text="Output Folder:").pack(pady=5)
        self.local_output_folder_entry = tk.Entry(self.tab_local_watcher, width=100)
        self.local_output_folder_entry.insert(0, os.path.join(self.tool_dir, "results/"))
        self.local_output_folder_entry.pack(pady=5)
        tk.Button(self.tab_local_watcher, text="Select Output Folder", command=self.select_local_output_folder).pack(pady=5)

        tk.Button(self.tab_local_watcher, text="Start Watching", command=self.start_local_watching).pack(pady=10)

        # --- Active models section (raw + bead-calibrated) ---------------
        sep_w = ttk.Separator(self.tab_local_watcher, orient="horizontal")
        sep_w.pack(fill="x", pady=(8, 4))

        tk.Label(self.tab_local_watcher, text="Trained models container URL (for model download):").pack(pady=(0, 2))
        self.watcher_trained_models_container_entry = tk.Entry(self.tab_local_watcher, width=100)
        self.watcher_trained_models_container_entry.insert(0, load_app_config().get("trained_models_container_url") or "")
        self.watcher_trained_models_container_entry.pack(pady=(0, 4))

        # Raw / uncalibrated model row
        raw_row = tk.Frame(self.tab_local_watcher)
        raw_row.pack(pady=2, fill="x")
        tk.Button(raw_row, text="Refresh Versions", command=self._watcher_refresh_model_versions).pack(side="left")
        self.watcher_raw_model_version_cb = ttk.Combobox(raw_row, width=40)
        self.watcher_raw_model_version_cb.pack(side="left", padx=6)
        tk.Button(
            raw_row,
            text="Download & Set Active (Raw/Uncalibrated)",
            command=self._watcher_set_active_raw_model,
        ).pack(side="left", padx=(6, 0))

        self.watcher_raw_model_label = tk.Label(self.tab_local_watcher, text="", fg="gray")
        self.watcher_raw_model_label.pack(pady=(2, 0))

        # Bead-calibrated model row
        beadcal_row = tk.Frame(self.tab_local_watcher)
        beadcal_row.pack(pady=2, fill="x")
        self.watcher_beadcal_model_version_cb = ttk.Combobox(beadcal_row, width=40)
        self.watcher_beadcal_model_version_cb.pack(side="left", padx=6)
        tk.Button(
            beadcal_row,
            text="Download & Set Active (Bead-Calibrated)",
            command=self._watcher_set_active_beadcal_model,
        ).pack(side="left", padx=(6, 0))

        self.watcher_beadcal_model_label = tk.Label(self.tab_local_watcher, text="", fg="gray")
        self.watcher_beadcal_model_label.pack(pady=(2, 4))

        self._update_watcher_model_labels()

        sep_w2 = ttk.Separator(self.tab_local_watcher, orient="horizontal")
        sep_w2.pack(fill="x", pady=(4, 8))

        # ------------------------------------------------------------
        # Button: Open predictions.csv_3d.html in browser
        # ------------------------------------------------------------
        def open_predictions_3d_html():
            # Prefer the active output folder set by Start Watching, otherwise read the entry box
            folder = getattr(self, "output_folder", "") or self.local_output_folder_entry.get().strip()
            html_path = os.path.join(folder, "predictions.csv_3d.html")

            if not folder or not os.path.isdir(folder):
                messagebox.showerror("Open 3D plot", "Output folder is not set or does not exist.")
                return

            if not os.path.exists(html_path):
                messagebox.showerror(
                    "Open 3D plot",
                    f"Could not find:\n{html_path}\n\nRun a sample first so the HTML is generated."
                )
                return

            # Open in default browser (file:// URL)
            webbrowser.open("file://" + os.path.abspath(html_path))



        # ---------------------------------------------------------------------
        # NEW: Watcher plots panel (shows ALL PNGs in self.output_folder)
        # ---------------------------------------------------------------------
        watcher_frame = tk.LabelFrame(self.tab_local_watcher, text="Watcher plots (all .png in output folder)")
        watcher_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Canvas + scrollbars (vertical + horizontal for full-res images)
        self._watcher_canvas = tk.Canvas(watcher_frame, highlightthickness=0)
        self._watcher_vscroll = tk.Scrollbar(watcher_frame, orient="vertical", command=self._watcher_canvas.yview)
        self._watcher_hscroll = tk.Scrollbar(watcher_frame, orient="horizontal", command=self._watcher_canvas.xview)

        self._watcher_canvas.configure(
            yscrollcommand=self._watcher_vscroll.set,
            xscrollcommand=self._watcher_hscroll.set
        )

        self._watcher_vscroll.pack(side="right", fill="y")
        self._watcher_hscroll.pack(side="bottom", fill="x")
        self._watcher_canvas.pack(side="left", fill="both", expand=True)

        # Inner frame hosted inside the canvas
        self._watcher_inner = tk.Frame(self._watcher_canvas)
        self._watcher_window = self._watcher_canvas.create_window((0, 0), window=self._watcher_inner, anchor="nw")

        def _on_inner_config(_event=None):
            self._watcher_canvas.configure(scrollregion=self._watcher_canvas.bbox("all"))

        self._watcher_inner.bind("<Configure>", _on_inner_config)

        # Keep the inner frame width at least canvas width so headings don't clip;
        # images can still extend wider (horizontal scroll enabled).
        def _on_canvas_config(event):
            self._watcher_canvas.itemconfig(self._watcher_window, width=max(event.width, 10))

        self._watcher_canvas.bind("<Configure>", _on_canvas_config)

        # Optional: mouse wheel scrolling (Windows/macOS/Linux variants)
        def _on_mousewheel(event):
            # Windows / macOS
            if hasattr(event, "delta") and event.delta:
                self._watcher_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                # Linux (Button-4 / Button-5)
                if event.num == 4:
                    self._watcher_canvas.yview_scroll(-3, "units")
                elif event.num == 5:
                    self._watcher_canvas.yview_scroll(3, "units")

        self._watcher_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._watcher_canvas.bind_all("<Button-4>", _on_mousewheel)
        self._watcher_canvas.bind_all("<Button-5>", _on_mousewheel)

        # State: keep refs to PhotoImage to avoid GC
        self._watcher_photo_refs = []
        self._watcher_file_state = {}  # path -> (mtime, size)
        self.output_folder = getattr(self, "output_folder", "")

        tk.Button(
            self.tab_local_watcher,
            text="Open 3D predictions plot (predictions.csv_3d.html)",
            command=open_predictions_3d_html
        ).pack(pady=6)
        
        def _scan_pngs(folder):
            if not folder or not os.path.isdir(folder):
                return {}, []
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")]
            files = sorted(files)  # stable order
            st = {}
            for fp in files:
                try:
                    s = os.stat(fp)
                    st[fp] = (s.st_mtime, s.st_size)
                except Exception:
                    pass
            return st, files

        def _rebuild_view(files):
            # Clear existing widgets
            for w in list(self._watcher_inner.winfo_children()):
                w.destroy()
            self._watcher_photo_refs = []

            if not files:
                tk.Label(self._watcher_inner, text="No .png files found in output folder yet.").pack(
                    anchor="w", padx=8, pady=8
                )
                _on_inner_config()
                return

            for fp in files:
                name = os.path.basename(fp)

                tk.Label(self._watcher_inner, text=name, font=("Arial", 10, "bold")).pack(
                    anchor="w", padx=8, pady=(12, 2)
                )

                try:
                    img = Image.open(fp)
                    # FULL RESOLUTION: do NOT thumbnail/resize
                    photo = ImageTk.PhotoImage(img)
                    self._watcher_photo_refs.append(photo)

                    lbl = tk.Label(self._watcher_inner, image=photo)
                    lbl.pack(anchor="w", padx=8, pady=(0, 10))

                except Exception as e:
                    tk.Label(self._watcher_inner, text=f"[could not load {name}: {e}]", fg="red").pack(
                        anchor="w", padx=8, pady=(0, 10)
                    )

            _on_inner_config()

        def _refresh_loop():
            # Prefer self.output_folder if set by Start Watching, else read from entry
            folder = getattr(self, "output_folder", "") or self.local_output_folder_entry.get().strip()
            new_state, files = _scan_pngs(folder)

            # Update when files are added/removed OR overwritten/updated
            if new_state != self._watcher_file_state:
                self._watcher_file_state = new_state
                _rebuild_view(files)

            # Poll again
            self.root.after(1000, _refresh_loop)

        def _set_folder(folder):
            self.output_folder = folder
            # Force a rebuild next tick
            self._watcher_file_state = {}

        # Expose setter so start_local_watching can point this viewer correctly
        self._watcher_set_folder = _set_folder

        # Kick off the refresh loop
        self.root.after(200, _refresh_loop)
        
    
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
            messagebox.showinfo("Success", "Check terminal to verify that Cyz2Json was downloaded successfully.")
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



    def create_widgets(self):
        #self.redirect_stdout_to_gui() This seems to interfere with the model training functions
        tk.Label(self.root, text=f"Working Directory: {self.tool_dir}", fg="gray").pack(pady=(10, 0))
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both')

        self.tab_readme = ttk.Frame(notebook)
        notebook.add(self.tab_readme, text="README")
        self.display_readme(self.tab_readme)

        self.tab_local_watcher = ttk.Frame(notebook)
        notebook.add(self.tab_local_watcher, text="Continous sample analyser")
        self.build_local_watcher_tab()

        self.tab_download = ttk.Frame(notebook)
        notebook.add(self.tab_download, text="Download & Train")
        self.build_download_tab()

        self.build_blob_tools_tab()

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
            sample_rate = float(self.sample_rate_entry.get().strip())
            from flowcytometer_tool.misc.functions import mix_blob_files
            mix_blob_files(container, sas_token=None, output_blob_folder=self.output_blob_folder.get().strip(), sample_rate=sample_rate)
            messagebox.showinfo("Success", "Mixfile generated and uploaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate mixfile: {e}")        


    def _premerge_plot_callback(self, raw_df):
        out_html = os.path.join(self.plots_dir, "premerge_3d_fluorescence.html")

        try:
            from flowcytometer_tool.misc.functions import plot_3d_fluorescence_premerge
            plot_3d_fluorescence_premerge(
                raw_df, label_col="source_label", out_html=out_html
            )

            functions.log_message(f"Pre-merge 3D fluorescence plot written: {out_html}")

        except Exception as e:
            functions.log_message(f"[warn] could not write pre-merge 3D plot: {e}")

    def _get_training_zone_and_expertise_levels(self):
        """Ask which expertise-matrix row should be used for this training session."""
        try:
            expertise_matrix = pd.read_csv(expertise_matrix_path, index_col=0)
        except Exception as e:
            messagebox.showerror("Expertise matrix", f"Failed to load expertise_matrix.csv:\n{e}")
            return None, None

        zones = [str(z) for z in expertise_matrix.index.tolist()]
        if not zones:
            messagebox.showerror("Expertise matrix", "expertise_matrix.csv has no rows/zones.")
            return None, None

        zonechoice = simpledialog.askstring(
            "Training zone",
            "Choose the expertise-matrix row/zone for this training set:\n" + ", ".join(zones),
            initialvalue=zones[0],
            parent=self.root,
        )
        if not zonechoice:
            return None, None
        if zonechoice not in expertise_matrix.index:
            messagebox.showerror("Training zone", f"'{zonechoice}' is not a row in expertise_matrix.csv.")
            return None, None

        levels = expertise_matrix.loc[zonechoice].to_dict()
        expertise_levels = {
            "expert": [k for k, v in levels.items() if v == 3],
            "advanced": [k for k, v in levels.items() if v == 2],
            "non_expert": [k for k, v in levels.items() if v == 1],
        }
        return zonechoice, expertise_levels

    def handle_build_training_from_selected_cyz_and_xmls(self):
        """
        Gate-axis aliases are resolved inside functions.assign_classes_from_gates.
        New training-data path: select the .cyz files once, then supply one XML
        gate file per person listed in the selected expertise-matrix row.
        """
        try:
            cyz_paths = filedialog.askopenfilenames(
                title="Select the .cyz files to include in this training set",
                filetypes=[("CYZ files", "*.cyz"), ("All files", "*.*")],
                parent=self.root,
            )
            if not cyz_paths:
                messagebox.showwarning("No CYZ files", "No .cyz files were selected.")
                return

            zonechoice, expertise_levels = self._get_training_zone_and_expertise_levels()
            if not expertise_levels:
                return

            people = []
            for level in ("expert", "advanced", "non_expert"):
                people.extend(expertise_levels.get(level, []))
            people = [p for p in people if str(p).strip()]
            if not people:
                messagebox.showerror("Expertise matrix", f"No people with expertise level 1, 2 or 3 were found for {zonechoice}.")
                return

            messagebox.showinfo(
                "Per-person XML upload",
                "You will now be asked to select one XML gate file for each person in the expertise matrix.\n\n"
                "Cancel a person's file dialog to skip that person for this training build."
            )
            person_xml_paths = {}
            for person in people:
                xml_path = filedialog.askopenfilename(
                    title=f"Select XML gates for {person}",
                    filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
                    parent=self.root,
                )
                if xml_path:
                    person_xml_paths[person] = xml_path

            if not person_xml_paths:
                messagebox.showerror("No XML files", "No per-person XML files were selected.")
                return

            self.label_change_log_path = init_label_change_log(self.model_dir)

            cyz2json_path = self.path_entry.get().strip()
            self.df = build_consensual_dataset_from_selected_cyzs_and_xmls(
                cyz_paths=cyz_paths,
                output_path=self.output_path,
                cyz2json_path=cyz2json_path,
                person_xml_paths=person_xml_paths,
                expertise_levels=expertise_levels,
                prompt_merge_fn=self.prompt_class_grouping,
                premerge_plot_fn=self._premerge_plot_callback,
                delete_labels_fn=self.prompt_delete_labels,
            )
            self.bead_samples = self._build_training_bead_samples_from_selected_cyz_paths(cyz_paths)
            print(self.df.columns)
            print(self.bead_samples)

            if self.df is None or self.df.empty:
                messagebox.showwarning("No data", "No labelled particles were produced from the selected CYZ/XML files.")
                return

            try:
                self._write_selected_cyz_modeltrainsettings(
                    cyz_paths=cyz_paths,
                    person_xml_paths=person_xml_paths,
                    zonechoice=zonechoice,
                    expertise_levels=expertise_levels,
                    bead_samples=self.bead_samples,
                )
            except Exception as e:
                functions.log_message(
                    f"[warn] could not write rich selected-CYZ modeltrainsettings.json: "
                    f"{type(e).__name__}: {e}"
                )
            messagebox.showinfo(
                "Training data ready",
                f"Built XML-labelled consensus dataset with {len(self.df)} rows from {len(cyz_paths)} CYZ file(s) and {len(person_xml_paths)} XML file(s)."
            )
        except Exception as e:
            messagebox.showerror("XML training build failed", f"Failed to build training data from selected CYZ/XML files:\n{e}")




    def _safe_file_md5(self, path, chunk_size=1024 * 1024):
        """
        Return an MD5 checksum for a local file, or None if unavailable.
        Used only for provenance in modeltrainsettings.json.
        """
        import hashlib
        import os

        if not path or not os.path.isfile(str(path)):
            return None

        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()


    def _run_git_command_for_modelsettings(self, args, cwd=None):
        """
        Run a git command safely for provenance capture.
        Returns None on failure rather than breaking training.
        """
        import os
        import subprocess

        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=cwd or os.getcwd(),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                return None
            value = completed.stdout.strip()
            return value if value else None
        except Exception:
            return None


    def _collect_git_provenance_for_modelsettings(self):
        """
        Capture repository provenance for modeltrainsettings.json.

        This restores the important git SHA field that was lost when the
        selected-CYZ/per-person-XML path stopped using collect_zone_metadata_and_assert().
        """
        import os
        import sys
        import datetime

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if getattr(sys, "frozen", False):
            repo_root = os.path.abspath(".")

        sha = self._run_git_command_for_modelsettings(["rev-parse", "HEAD"], cwd=repo_root)
        short_sha = self._run_git_command_for_modelsettings(["rev-parse", "--short", "HEAD"], cwd=repo_root)
        branch = self._run_git_command_for_modelsettings(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        remote = self._run_git_command_for_modelsettings(["config", "--get", "remote.origin.url"], cwd=repo_root)
        status = self._run_git_command_for_modelsettings(["status", "--porcelain"], cwd=repo_root)

        return {
            "repo_root": repo_root,
            "git_sha": sha,
            "git_short_sha": short_sha,
            "git_branch": branch,
            "git_remote_origin": remote,
            "git_dirty": bool(status),
            "git_status_porcelain": status,
            "provenance_recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }


    def _extract_dotted_from_json_for_modelsettings(self, js, dotted):
        """
        Follow a dotted JSON path such as:
            instrument.measurementSettings.CytoSettings.PMTtemperature

        Returns None if any segment is missing.
        """
        cur = js
        for part in str(dotted).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur


    def _flatten_json_for_modelsettings(self, obj, prefix=""):
        """
        Flatten nested dict/list JSON into dotted-key form for provenance checking.
        Lists are indexed as key.0, key.1, etc.
        """
        out = {}

        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                out.update(self._flatten_json_for_modelsettings(v, key))
            return out

        if isinstance(obj, list):
            for i, v in enumerate(obj):
                key = f"{prefix}.{i}" if prefix else str(i)
                out.update(self._flatten_json_for_modelsettings(v, key))
            return out

        out[prefix] = obj
        return out


    def _load_training_grablist_for_modelsettings(self, grablist_path=None):
        """
        Load optional provenance grablist. This deliberately mirrors the old
        grablist behaviour but is safe when the file is absent.
        """
        import os

        if grablist_path is None:
            grablist_path = os.path.join("flowcytometer_tool", "config", "grablist.txt")

        items = []
        try:
            if not os.path.isfile(grablist_path):
                return items
            with open(grablist_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        items.append(s)
        except Exception:
            return []

        return items


    def _select_key_training_json_fields_for_modelsettings(self, full_js, grablist_items=None):
        """
        Extract the high-value fields that should travel with the trained model.

        This captures:
          - full measurementSettings
          - full CytoSettings
          - PMT-related fields wherever they occur
          - sensor limits
          - common acquisition/result fields
          - optional grablist fields
        """
        instrument = full_js.get("instrument", {}) if isinstance(full_js, dict) else {}
        measurement_settings = instrument.get("measurementSettings", {}) if isinstance(instrument, dict) else {}
        measurement_results = instrument.get("measurementResults", {}) if isinstance(instrument, dict) else {}
        cyto_settings = measurement_settings.get("CytoSettings", {}) if isinstance(measurement_settings, dict) else {}

        flattened = self._flatten_json_for_modelsettings(full_js)

        pmt_fields = {
            k: v
            for k, v in flattened.items()
            if "pmt" in str(k).lower()
        }

        sensor_limit_fields = {
            k: v
            for k, v in flattened.items()
            if "sensorlimit" in str(k).lower()
            or "sensorlimits" in str(k).lower()
            or "minvalue" in str(k).lower()
            or "maxvalue" in str(k).lower()
        }

        grablist = {}
        for dotted in grablist_items or []:
            grablist[dotted] = self._extract_dotted_from_json_for_modelsettings(full_js, dotted)

        common_result_fields = {}
        for key in (
            "start",
            "duration",
            "particleCount",
            "pumpedVolume",
            "analysedVolume",
            "externalPumpTime",
            "pressureAbsolute",
            "pressureDifferential",
            "sheathTemperature",
            "systemTemperature",
            "laserTemperature",
            "PMTtemperature",
        ):
            if isinstance(measurement_results, dict) and key in measurement_results:
                common_result_fields[f"instrument.measurementResults.{key}"] = measurement_results.get(key)

        common_setting_fields = {}
        for key in (
            "IIFCheck",
            "IsBeadsMeasurement",
            "LaserBeamWidth",
            "SampleCoreSpeed",
            "PMTtemperature",
            "SensorLimits",
        ):
            if isinstance(cyto_settings, dict) and key in cyto_settings:
                common_setting_fields[f"instrument.measurementSettings.CytoSettings.{key}"] = cyto_settings.get(key)

        if isinstance(measurement_settings, dict) and "beads_measurement_2" in measurement_settings:
            common_setting_fields["instrument.measurementSettings.beads_measurement_2"] = measurement_settings.get("beads_measurement_2")

        return {
            "measurementSettings": measurement_settings,
            "CytoSettings": cyto_settings,
            "measurementResults_selected": common_result_fields,
            "measurementSettings_selected": common_setting_fields,
            "pmt_fields": pmt_fields,
            "sensor_limit_fields": sensor_limit_fields,
            "grablist_fields": grablist,
        }


    def _summarise_consistency_for_modelsettings(self, per_file_records):
        """
        Summarise which selected JSON metadata fields are identical across all
        selected CYZs and which vary between files.
        """
        values_by_key = {}

        for rec in per_file_records or []:
            selected = rec.get("selected_metadata", {})
            flat = self._flatten_json_for_modelsettings(selected)
            for key, value in flat.items():
                values_by_key.setdefault(key, []).append(value)

        consistent = {}
        variable = {}

        for key, values in values_by_key.items():
            comparable = [json.dumps(v, sort_keys=True, default=str) for v in values]
            unique = sorted(set(comparable))
            if len(unique) == 1:
                consistent[key] = values[0] if values else None
            else:
                variable[key] = {
                    "unique_count": len(unique),
                    "values_by_file": [
                        {
                            "source_file": rec.get("source_file"),
                            "value": self._flatten_json_for_modelsettings(
                                rec.get("selected_metadata", {})
                            ).get(key),
                        }
                        for rec in per_file_records
                    ],
                }

        return {
            "consistent_fields": consistent,
            "variable_fields": variable,
            "consistent_field_count": len(consistent),
            "variable_field_count": len(variable),
        }


    def _resolve_selected_cyz_training_json_paths(self, cyz_paths):
        """
        Resolve the JSON/listmode paths produced by convert_selected_cyzs_to_listmode()
        for the selected CYZ/XML training route.
        """
        import os

        json_dir = os.path.join(self.output_path, "selected_cyz_xml_training", "json")
        records = []

        for cyz_path in (cyz_paths or []):
            cyz_path = str(cyz_path)
            stem = os.path.splitext(os.path.basename(cyz_path))[0]
            json_path = os.path.join(json_dir, f"{stem}.cyz.json")
            listmode_csv_path = os.path.join(json_dir, f"{stem}.cyz.csv")

            records.append(
                {
                    "cyz_path": cyz_path,
                    "cyz_basename": os.path.basename(cyz_path),
                    "json_path": json_path,
                    "json_basename": os.path.basename(json_path),
                    "listmode_csv_path": listmode_csv_path,
                    "listmode_csv_basename": os.path.basename(listmode_csv_path),
                    "json_exists": os.path.isfile(json_path),
                    "listmode_csv_exists": os.path.isfile(listmode_csv_path),
                }
            )

        return records


    def _collect_selected_cyz_modeltrainsettings_metadata(
        self,
        cyz_paths,
        person_xml_paths=None,
        zonechoice=None,
        expertise_levels=None,
        grablist_path=None,
    ):
        """
        Build the rich modeltrainsettings metadata for the new selected-CYZ plus
        per-person-XML training route.

        This is the replacement for the old collect_zone_metadata_and_assert()
        role in the new workflow. It does not assert zone sameness. Instead it
        records both consistent and variable metadata for the selected files.
        """
        import os
        import json
        import datetime

        grablist_items = self._load_training_grablist_for_modelsettings(grablist_path)
        path_records = self._resolve_selected_cyz_training_json_paths(cyz_paths)

        per_file_records = []
        warnings = []

        for path_rec in path_records:
            cyz_path = path_rec["cyz_path"]
            json_path = path_rec["json_path"]
            listmode_csv_path = path_rec["listmode_csv_path"]

            if not os.path.isfile(json_path):
                warnings.append(f"Missing converted JSON for selected CYZ: {cyz_path} -> {json_path}")
                continue

            try:
                with open(json_path, "r", encoding="utf-8-sig") as f:
                    full_js = json.load(f)

                selected_metadata = self._select_key_training_json_fields_for_modelsettings(
                    full_js,
                    grablist_items=grablist_items,
                )

                per_file_records.append(
                    {
                        "source_file": cyz_path,
                        "source_basename": os.path.basename(cyz_path),
                        "source_md5": self._safe_file_md5(cyz_path),
                        "converted_json_path": json_path,
                        "converted_json_basename": os.path.basename(json_path),
                        "converted_json_md5": self._safe_file_md5(json_path),
                        "listmode_csv_path": listmode_csv_path,
                        "listmode_csv_basename": os.path.basename(listmode_csv_path),
                        "listmode_csv_md5": self._safe_file_md5(listmode_csv_path),
                        "listmode_csv_exists": os.path.isfile(listmode_csv_path),
                        "selected_metadata": selected_metadata,
                    }
                )

            except Exception as e:
                warnings.append(f"Could not read metadata from {json_path}: {type(e).__name__}: {e}")

        xml_records = {}
        for person, xml_path in (person_xml_paths or {}).items():
            xml_records[str(person)] = {
                "xml_path": str(xml_path),
                "xml_basename": os.path.basename(str(xml_path)),
                "xml_md5": self._safe_file_md5(xml_path),
            }

        consistency = self._summarise_consistency_for_modelsettings(per_file_records)

        return {
            "training_data_input": {
                "mode": "selected_cyz_per_person_xml",
                "expertise_matrix_zone": zonechoice,
                "cyz_files": [os.path.basename(str(p)) for p in (cyz_paths or [])],
                "cyz_file_count": len(cyz_paths or []),
                "xml_files_by_person": {
                    str(p): os.path.basename(str(x))
                    for p, x in (person_xml_paths or {}).items()
                },
                "xml_file_count": len(person_xml_paths or {}),
                "expertise_levels": expertise_levels,
            },
            "source_file_provenance": {
                "selected_cyz_files": path_records,
                "per_person_xml_files": xml_records,
                "per_file_metadata": per_file_records,
                "warnings": warnings,
            },
            "training_json_metadata": {
                "grablist_path": grablist_path or os.path.join("flowcytometer_tool", "config", "grablist.txt"),
                "grablist_items": grablist_items,
                "per_file_metadata_count": len(per_file_records),
                "metadata_consistency_summary": consistency,
            },
            "software_provenance": self._collect_git_provenance_for_modelsettings(),
            "modeltrainsettings_updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }


    def _write_selected_cyz_modeltrainsettings(
        self,
        *,
        cyz_paths,
        person_xml_paths,
        zonechoice,
        expertise_levels,
        bead_samples,
    ):
        """
        Write or update modeltrainsettings.json for the selected-CYZ per-person XML
        training route.

        This restores git SHA, PMT fields, full measurement settings, sensor limits,
        grablist fields, and per-file source provenance.
        """
        import os
        import json
        import datetime

        mts_path = self.modeltrainsettings_out

        if os.path.exists(mts_path):
            with open(mts_path, "r", encoding="utf-8") as f:
                mts = json.load(f)
        else:
            mts = {}

        rich_meta = self._collect_selected_cyz_modeltrainsettings_metadata(
            cyz_paths=cyz_paths,
            person_xml_paths=person_xml_paths,
            zonechoice=zonechoice,
            expertise_levels=expertise_levels,
        )
        
        # Backwards-compatible top-level instrument block for existing watcher/loader code:

        per_file_metadata = (
            rich_meta
            .get("source_file_provenance", {})
            .get("per_file_metadata", [])
        )

        selected_metadata = (
            per_file_metadata[0].get("selected_metadata", {})
            if per_file_metadata else {}
        )

        measurement_settings = selected_metadata.get("measurementSettings", {}) or {}
        cyto_settings = (
            selected_metadata.get("CytoSettings", {})
            or measurement_settings.get("CytoSettings", {})
            or {}
        )

        # Find serial number from any captured serial field.
        flat_selected = {}

        def _flatten_for_legacy(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else str(k)
                    _flatten_for_legacy(v, key)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    key = f"{prefix}.{i}" if prefix else str(i)
                    _flatten_for_legacy(v, key)
            else:
                flat_selected[prefix] = obj

        _flatten_for_legacy(selected_metadata)

        serial_number = next(
            (
                v for k, v in flat_selected.items()
                if "serialnumber" in str(k).lower() and v not in (None, "")
            ),
            None,
        )

        # Ensure the old expected path exists.
        rich_meta["instrument"] = {
            "serialNumber": serial_number,
            "measurementSettings": {
                **measurement_settings,
                "CytoSettings": {
                    **cyto_settings,
                    "PMTlevels_str": (
                        cyto_settings.get("PMTlevels_str")
                        or selected_metadata.get("pmt_fields", {}).get(
                            "instrument.measurementSettings.CytoSettings.PMTlevels_str"
                        )
                        or next(
                            (
                                v for k, v in flat_selected.items()
                                if "pmtlevels_str" in str(k).lower() and v not in (None, "")
                            ),
                            None,
                        )
                    ),
                },
            },
        }        

        mts.update(rich_meta)

        mts["training_data_input"]["bead_cyz_files"] = [
            os.path.basename(sample.get("packet", {}).get("source_file", ""))
            for sample in (bead_samples or [])
            if isinstance(sample, dict)
            and sample.get("packet", {}).get("source_file")
        ]

        mts["training_dataset_summary"] = {
            "total_particles": int(len(self.df)) if self.df is not None else 0,
            "counts_per_class": {
                str(k): int(v)
                for k, v in self.df["source_label"].value_counts().to_dict().items()
            } if self.df is not None and "source_label" in self.df.columns else {},
            "columns": list(self.df.columns) if self.df is not None else [],
        }

        mts["cleaning"] = {
            "post_merge_nn_cleaning_ran": "False",
            "max_per_class_entry": self.max_per_class_entry.get()
            if hasattr(self, "max_per_class_entry")
            else None,
        }

        mts.setdefault("training", {})
        mts["training"]["created_by_route"] = "selected_cyz_per_person_xml"
        mts["training"]["modeltrainsettings_writer"] = "_write_selected_cyz_modeltrainsettings"
        mts["training"]["modeltrainsettings_writer_utc"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        mts = json_safe(mts)

        os.makedirs(os.path.dirname(mts_path), exist_ok=True)
        with open(mts_path, "w", encoding="utf-8") as f:
            json.dump(mts, f, indent=2, ensure_ascii=False)

        return mts


    def _build_training_bead_samples_from_selected_cyz_paths(self, cyz_paths):
        """
        Build train_model-compatible bead samples from selected CYZ files.

        Replacement version:
          - still returns list[{"packet": dict, "dataframe": DataFrame}]
          - still filters bead files using detect_sampling_protocol()
          - restores richer packet metadata from the converted JSON
          - stores per-file provenance for later modeltrainsettings writing
        """
        import os
        import json
        import pandas as pd

        print("_build_training_bead_samples_from_selected_cyz_paths")

        bead_samples = []
        self._selected_cyz_training_path_records = self._resolve_selected_cyz_training_json_paths(cyz_paths)

        for rec in self._selected_cyz_training_path_records:
            cyz_path = rec["cyz_path"]
            json_path = rec["json_path"]
            listmode_csv_path = rec["listmode_csv_path"]

            if not os.path.isfile(json_path):
                functions.log_message(
                    f"[warn] bead sample skipped; converted JSON missing for {cyz_path}: {json_path}"
                )
                continue

            if not os.path.isfile(listmode_csv_path):
                functions.log_message(
                    f"[warn] bead sample skipped; listmode CSV missing for {cyz_path}: {listmode_csv_path}"
                )
                continue

            try:
                with open(json_path, "r", encoding="utf-8-sig") as f:
                    full_js = json.load(f)

                instrument = full_js.get("instrument", {}) if isinstance(full_js, dict) else {}
                measurement_settings = instrument.get("measurementSettings", {}) if isinstance(instrument, dict) else {}
                measurement_results = instrument.get("measurementResults", {}) if isinstance(instrument, dict) else {}
                cyto_settings = measurement_settings.get("CytoSettings", {}) if isinstance(measurement_settings, dict) else {}

                selected_metadata = self._select_key_training_json_fields_for_modelsettings(full_js)

                packet = {
                    "source_file": str(cyz_path),
                    "source_basename": os.path.basename(str(cyz_path)),
                    "converted_json_path": str(json_path),
                    "converted_json_basename": os.path.basename(str(json_path)),
                    "listmode_csv_path": str(listmode_csv_path),
                    "listmode_csv_basename": os.path.basename(str(listmode_csv_path)),

                    "instrument.measurementSettings.CytoSettings.IIFCheck": cyto_settings.get("IIFCheck"),
                    "instrument.measurementSettings.CytoSettings.IsBeadsMeasurement": cyto_settings.get(
                        "IsBeadsMeasurement",
                        measurement_settings.get("beads_measurement_2"),
                    ),
                    "instrument.measurementSettings.beads_measurement_2": measurement_settings.get("beads_measurement_2"),

                    "instrument.measurementSettings": measurement_settings,
                    "instrument.measurementSettings.CytoSettings": cyto_settings,
                    "instrument.measurementResults": measurement_results,
                    "selected_metadata_for_modeltrainsettings": selected_metadata,
                    "source_md5": self._safe_file_md5(cyz_path),
                    "converted_json_md5": self._safe_file_md5(json_path),
                    "listmode_csv_md5": self._safe_file_md5(listmode_csv_path),
                }

                detected_protocol = detect_sampling_protocol(packet)
                packet["detected_sampling_protocol"] = detected_protocol

                if not is_bead_sample(packet, protocol=detected_protocol):
                    continue

                bead_df = pd.read_csv(listmode_csv_path)
                if bead_df.empty:
                    functions.log_message(f"[warn] bead sample skipped; empty listmode CSV: {listmode_csv_path}")
                    continue

                bead_samples.append({"packet": packet, "dataframe": bead_df})

            except Exception as e:
                functions.log_message(
                    f"[warn] bead sample skipped for {cyz_path}: {type(e).__name__}: {e}"
                )

        return bead_samples or None


    def handle_train_model(self):
        """
        Trigger model training, optionally with bead calibration samples.

        bead_samples contract:
          - list of dicts: {"packet": packet_dict, "dataframe" (or "df"): pandas.DataFrame}
          - or list of tuples: (packet_dict, pandas.DataFrame)
        Packet dict must identify bead protocol for calibration to apply.
        """
        train_model(
            self.df,
            self.plots_dir,
            self.model_path,
            '../',
            calibration_enabled=self.calibration_var.get(),
            nogui=False,
            self=self,
            max_per_class=int(self.max_per_class_entry.get()),
            bead_samples=self.bead_samples,
        )

    def handle_combine_csvs(self):
        self.bead_samples = None
        self.label_change_log_path = init_label_change_log(self.model_dir)
        # Version with NN cleaning in the 3 most important feature axes
        def _nn_cleaned_premerge_plot_callback(raw_df):
            out_html = os.path.join(self.plots_dir, "premerge_3d_fluorescence.html")
            try:
                # 1) Clean with NN homogenization (processing-only)
                from flowcytometer_tool.misc.functions import nn_homogenize_df
                cleaned_df = nn_homogenize_df(
                    raw_df,
                    label_col="source_label",
                    feature_cols=("FWS_total", "Fl Red_total", "Fl Orange_total"),
                    keep_unconsidered="keep",   # or "drop" if you prefer strict survivors
                    downsample_n=None           # set an int if you want faster previews
                )
                from flowcytometer_tool.misc.functions import plot_3d_fluorescence_premerge
                plot_3d_fluorescence_premerge(
                    cleaned_df,
                    label_col="source_label",
                    out_html=out_html
                )
                functions.log_message(f"Pre-merge 3D fluorescence plot written: {out_html}")
            except Exception as e:
                functions.log_message(f"[warn] could not write pre-merge 3D plot: {e}")

        def _premerge_plot_callback(raw_df):
            out_html = os.path.join(self.plots_dir, "premerge_3d_fluorescence.html")
            try:
                from flowcytometer_tool.misc.functions import plot_3d_fluorescence_premerge  # uses same columns as inspect_overlap
                plot_3d_fluorescence_premerge(raw_df, label_col="source_label", out_html=out_html)
                functions.log_message(f"Pre-merge 3D fluorescence plot written: {out_html}")
            except Exception as e:
                functions.log_message(f"[warn] could not write pre-merge 3D plot: {e}")

        self.df = functions.combine_csvs(
            self.url_entry,
            self.root,
            self.output_path,
            expertise_matrix_path=expertise_matrix_path,
            max_per_class_entry=self.max_per_class_entry.get(),
            nogui=False,
            prompt_merge_fn=self.prompt_class_grouping,
            premerge_plot_fn= _premerge_plot_callback,  
            delete_labels_fn=self.prompt_delete_labels
        )

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

            listbox = tk.Listbox(top, selectmode=tk.MULTIPLE, width=50, height = 50)
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
                    record_label_merge(self.label_change_log_path, selected_labels, new_label)
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



    def handle_predict_test_set(self):
        if self.df is None:
            messagebox.showerror("Error", "No dataset loaded. Please load or combine CSVs first.")
            return
        try:
            predict_name = os.path.join(self.tool_dir, "test_predictions.csv")
            cm_filename = os.path.join(self.tool_dir, "confusion_matrix.csv")
            report_filename = os.path.join(self.tool_dir, "classification_report.csv")
            text_file = open(os.path.join(self.tool_dir, "prediction_log.txt"), "w")
            from flowcytometer_tool.tabs.download_train.custom_functions_for_python import predictTestSet
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
            #self.update_plot()
            #self.update_summary_table()
            messagebox.showinfo("Success", "Test set predictions completed and saved.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict test set:\n{e}")


    def _browse_gates_xml(self):
        path = filedialog.askopenfilename(
            title="Select gates XML file",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
        )
        if path:
            self.gates_xml_entry.delete(0, tk.END)
            self.gates_xml_entry.insert(0, path)

    def handle_classify_with_gates(self):
        xml_path = self.gates_xml_entry.get().strip()
        if not xml_path:
            messagebox.showwarning("No XML", "Please select a gates XML file first.")
            return
        if not os.path.isfile(xml_path):
            messagebox.showerror("File not found", f"Gates XML not found:\n{xml_path}")
            return
        try:
            combined = functions.combine_csvs_with_gates(self.output_path, xml_path)
            if combined is None or combined.empty:
                messagebox.showwarning("No data", "No listmode CSV files found in the output folder, or all were skipped.")
                return
            self.df = combined
            classes = sorted(self.df["source_label"].unique())
            messagebox.showinfo(
                "Classify with Gates",
                f"Classification complete.\n{len(self.df)} particles classified into {len(classes)} classes:\n"
                + ", ".join(classes),
            )
        except Exception as e:
            messagebox.showerror("Gate Error", f"Failed to classify with gates:\n{e}")


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
        self.url_entry.insert(0, "https://citprodflowcytosa.blob.core.windows.net/public/miniexampledata/")
        #self.url_entry.insert(0, "https://citprodflowcytosa.blob.core.windows.net/labelledcyz/multipleexperts3seas/") # This dataset depends on an SAS token having been passed in on the blob tools tab.
        #self.url_entry.insert(0, "https://citprodflowcytosa.blob.core.windows.net/mnceacyzfilesforthomasrutten/manuallypairedxmlsandcyzs/exportedindividuallyfromcytoclus/") # This dataset depends on an SAS token having been passed in on the blob tools tab.
        self.url_entry.pack(pady=5)
        tk.Button(self.tab_download, text="Download Files (optional / depreciating)", command=self.download_blob_directory).pack(pady=5)
        tk.Button(self.tab_download, text="Download cyz2json (optional / depreciating)", command=self.install_all_requirements).pack(pady=5)
        #tk.Button(self.tab_download, text="Cyz2json", command=self.cyz2json).pack(pady=5)
        #tk.Button(self.tab_download, text="To listmode", command=self.to_listmode).pack(pady=5)
        # --- XML gate-based classification ---
        #tk.Label(self.tab_download, text="Gates XML file (for class assignment):").pack(pady=(10, 0))
        #xml_frame = tk.Frame(self.tab_download)
        #xml_frame.pack(pady=2)
        #self.gates_xml_entry = tk.Entry(xml_frame, width=60)
        #self.gates_xml_entry.pack(side="left", padx=(0, 4))
        #tk.Button(xml_frame, text="Browse…", command=self._browse_gates_xml).pack(side="left")
        #tk.Button(self.tab_download, text="Classify with Gates", command=self.handle_classify_with_gates).pack(pady=5)
        tk.Button(self.tab_download, text="Build training set from selected CYZs + per-person XMLs", command=self.handle_build_training_from_selected_cyz_and_xmls).pack(pady=5)
        self.nn_clean_var = tk.BooleanVar(value=False)# Keep this false unless we have a way of logging it in the modelsettingsjson - though I don't think it is desired anyway
        #tk.Checkbutton(self.tab_download,text="Apply NN-cleaning before classes are merged in combine csvs",variable=self.nn_clean_var).pack(pady=5) # Keep this false unless we have a way of logging it in the modelsettingsjson - though I don't think it is desired anyway
        #tk.Button(self.tab_download, text="Combine CSVs", command=self.handle_combine_csvs).pack(pady=5)
        tk.Button(    self.tab_download,    text="Run NN cleaning post-merge (on the renamed classes)",    command=self.handle_nn_cleaning).pack(pady=5)
        tk.Label(self.tab_download, text="Max samples per class:").pack(pady=5)
        self.max_per_class_entry = tk.Entry(self.tab_download, width=10)
        self.max_per_class_entry.insert(0, "1000")
        self.max_per_class_entry.config(state="disabled")# needs to be tracked if changed in the app - currently just logged on combine_csvs
        self.max_per_class_entry.pack(pady=5)
        self.calibration_var = tk.BooleanVar(value=True)  # default ON; controls probabilistic classifier calibration
        tk.Checkbutton(self.tab_download, text="Enable Probabilistic Calibration", variable=self.calibration_var).pack(pady=5)           
        tk.Button(self.tab_download, text="Authenticate and train Model", command=self.handle_train_model).pack(pady=5)
        tk.Button(self.tab_download, text="Predict Test Set", command=self.handle_predict_test_set).pack(pady=5)



    def build_blob_tools_tab(self):
        self.tab_blob_tools = ttk.Frame(self.root)
        self.root.nametowidget(".!notebook").add(self.tab_blob_tools, text="Blob Tools")


        # Sample Rate Input
        tk.Label(self.tab_blob_tools, text="Sample Rate (e.g., 0.005):").pack(pady=5)
        self.sample_rate_entry = tk.Entry(self.tab_blob_tools, width=20)
        self.sample_rate_entry.insert(0, "0.005")
        self.sample_rate_entry.pack(pady=5)

        # Container URLs
        tk.Label(self.tab_blob_tools, text="Blob Container URL:").pack(pady=5)
        self.url_entry_blob = tk.Entry(self.tab_blob_tools, width=100)
        self.url_entry_blob.insert(0, "https://citprodflowcytosa.blob.core.windows.net/hdduploaddec2025")
        self.url_entry_blob.pack(pady=5)
        
        tk.Label(self.tab_blob_tools, text="Output Container Name:").pack(pady=5)
        self.output_blob_folder = tk.Entry(self.tab_blob_tools, width=100)
        self.output_blob_folder.insert(0, "results")  # default value
        self.output_blob_folder.pack(pady=5)

        # Buttons
        tk.Button(self.tab_blob_tools, text="Process all cyz files in blob store", command=self.process_all).pack(pady=10)
        tk.Button(self.tab_blob_tools, text="Generate Mixfile of prediction csvs", command=self.generate_mixfile).pack(pady=5)
        
        # --- Active Model (Get/Set) -------------------------------------------------
        sep = ttk.Separator(self.tab_blob_tools, orient="horizontal")
        sep.pack(fill="x", pady=(12, 8))

        tk.Label(self.tab_blob_tools, text="Trained models container URL:").pack(pady=(0, 3))
        self.trained_models_container_entry = tk.Entry(self.tab_blob_tools, width=100)
        self.trained_models_container_entry.insert(0, load_app_config().get("trained_models_container_url"))
        self.trained_models_container_entry.pack(pady=(0, 8))

        row = tk.Frame(self.tab_blob_tools)
        row.pack(pady=2, fill="x")
        tk.Button(row, text="Refresh Versions", command=self.refresh_model_versions).pack(side="left")
        self.model_version_cb = ttk.Combobox(row, width=40)
        self.model_version_cb.pack(side="left", padx=6)
        tk.Button(row, text="Download & Set Active (Raw/Uncalibrated)", command=self.handle_set_active_model).pack(side="left", padx=(6,0))

        self.active_model_label = tk.Label(self.tab_blob_tools, text="", fg="gray")
        self.active_model_label.pack(pady=(6, 0))
        self.update_active_model_label()


    def refresh_model_versions(self):
        try:
            container_url = self.trained_models_container_entry.get().strip()
            versions = list_available_model_versions(container_url)
            self.model_version_cb["values"] = versions
            if versions:
                self.model_version_cb.set(versions[-1])  # preselect latest by lexical
            messagebox.showinfo("Versions", f"Found {len(versions)} version(s).")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to list model versions:\n{e}")

    def handle_set_active_model(self):
        try:
            version = self.model_version_cb.get().strip()
            if not version:
                messagebox.showerror("Select a version", "Pick a version in the dropdown first.")
                return
            container_url = self.trained_models_container_entry.get().strip()
            primary_path = set_active_model(version, container_url=container_url)
            # Optionally sync a UI variable; inference will resolve from config anyway
            self.model_path = primary_path
            self.update_active_model_label()
            messagebox.showinfo("Active Raw/Uncalibrated Model", f"Active raw/uncalibrated model set to {version}.\n\nFiles downloaded into:\n{active_model_dir()}")
        except Exception as e:
            messagebox.showerror("Set Active Model", f"Failed to set active model:\n{e}")

    def update_active_model_label(self):
        cfg = load_app_config()
        am = cfg.get("active_uncalibrated_model") or cfg.get("active_model")
        if am:
            try:
                path = resolve_active_raw_model_path()
                self.active_model_label.config(
                    text=f"Active raw/uncalibrated model: v{am.get('version')}  →  {os.path.basename(path)}",
                    fg="green"
                )
            except Exception as _:
                self.active_model_label.config(text="Raw/uncalibrated model invalid/missing. Please set again.", fg="red")
        else:
            self.active_model_label.config(text="No raw/uncalibrated model set.", fg="red")

    # ------------------------------------------------------------------
    # Continuous sample analyser tab: model management helpers
    # ------------------------------------------------------------------

    def _watcher_refresh_model_versions(self):
        try:
            container_url = self.watcher_trained_models_container_entry.get().strip()
            versions = list_available_model_versions(container_url)
            self.watcher_raw_model_version_cb["values"] = versions
            self.watcher_beadcal_model_version_cb["values"] = versions
            if versions:
                self.watcher_raw_model_version_cb.set(versions[-1])
                self.watcher_beadcal_model_version_cb.set(versions[-1])
            messagebox.showinfo("Versions", f"Found {len(versions)} version(s).")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to list model versions:\n{e}")

    def _watcher_set_active_raw_model(self):
        try:
            version = self.watcher_raw_model_version_cb.get().strip()
            if not version:
                messagebox.showerror("Select a version", "Pick a version in the dropdown first.")
                return
            container_url = self.watcher_trained_models_container_entry.get().strip()
            set_active_uncalibrated_model(version, container_url=container_url)
            self._update_watcher_model_labels()
            messagebox.showinfo("Raw Model", f"Active uncalibrated model set to version {version}.")
        except Exception as e:
            messagebox.showerror("Set Raw Model", f"Failed:\n{e}")

    def _watcher_set_active_beadcal_model(self):
        try:
            version = self.watcher_beadcal_model_version_cb.get().strip()
            if not version:
                messagebox.showerror("Select a version", "Pick a version in the dropdown first.")
                return
            container_url = self.watcher_trained_models_container_entry.get().strip()
            set_active_beadcalibrated_model(version, container_url=container_url)
            self._update_watcher_model_labels()
            messagebox.showinfo("Bead-Calibrated Model", f"Active bead-calibrated model set to version {version}.")
        except Exception as e:
            messagebox.showerror("Set Bead-Calibrated Model", f"Failed:\n{e}")

    def _update_watcher_model_labels(self):
        # Raw model label
        cfg = load_app_config()
        raw_slot = cfg.get("active_uncalibrated_model")
        if raw_slot and isinstance(raw_slot, dict):
            try:
                path = resolve_active_raw_model_path()
                self.watcher_raw_model_label.config(
                    text=f"Raw model: v{raw_slot.get('version')}  →  {os.path.basename(path)}",
                    fg="green",
                )
            except Exception:
                self.watcher_raw_model_label.config(
                    text="Raw model invalid/missing. Please set again.", fg="red"
                )
        else:
            self.watcher_raw_model_label.config(text="No raw model set.", fg="red")

        # Bead-calibrated model label
        beadcal_slot = cfg.get("active_beadcalibrated_model")
        if beadcal_slot and isinstance(beadcal_slot, dict):
            try:
                path = resolve_active_beadcalibrated_model_path()
                if path:
                    self.watcher_beadcal_model_label.config(
                        text=f"Bead-calibrated model: v{beadcal_slot.get('version')}  →  {os.path.basename(path)}",
                        fg="green",
                    )
                else:
                    raise FileNotFoundError
            except Exception:
                self.watcher_beadcal_model_label.config(
                    text="Bead-calibrated model invalid/missing. Please set again.", fg="red"
                )
        else:
            self.watcher_beadcal_model_label.config(
                text="No bead-calibrated model set (optional).", fg="gray"
            )



    def download_blob_directory(self):
        try:
            blob_url = self.url_entry.get()
            sas_token = None
            if blob_url == "https://citprodflowcytosa.blob.core.windows.net/public/miniexampledata/":
                download_blobs(blob_url, self.download_path)
            else:
                download_blobs(blob_url, self.download_path, sas_token)
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

    def process_all(self):
        """
        New AAD-based Blob processing pipeline (no SAS tokens).
        Downloads each .cyz from the container, converts to JSON & listmode,
        applies the Python model, uploads outputs, updates QC plots, and logs progress.
        """
        import os
        import pandas as pd
        from tkinter import messagebox, filedialog, simpledialog

        # Storage helpers (AAD-based)
        from flowcytometer_tool.tabs.blob_tools.storage_clients import _split_blob_url, get_blob_client, get_container_client

        # Existing utility functions from your codebase
        from flowcytometer_tool.misc.functions import (
            upload_to_blob,
            extract_processed_url,
            log_message,
            delete_file,
            apply_python_model,
            to_listmode,
            load_file,
        )
        import flowcytometer_tool.tabs.continuous_sample_analyser.qc_plots as qc_plots

        try:
            container_url = self.url_entry_blob.get().strip()  # e.g., https://<acct>.blob.core.windows.net/<container>
            output_blob_folder = self.output_blob_folder.get().strip()  # destination container name for outputs

            account_url, container_name, _ = _split_blob_url(container_url)

            # Authenticated container client (browser sign-in on first use)
            cc = get_container_client(account_url, container_name, anonymous=False)

            # Track previously processed blobs (by URL without SAS)
            processed_files: Set[str] = set()
            log_file_path = "process_log.txt"
            if os.path.exists(log_file_path):
                with open(log_file_path, "r") as log_file:
                    for line in log_file:
                        processed_url = extract_processed_url(line)
                        if processed_url:
                            processed_files.add(processed_url)

            # Iterate .cyz blobs only
            for blob in cc.list_blobs():
                if not blob.name.lower().endswith(".cyz"):
                    continue

                blob_name = blob.name
                blob_url_no_token = f"{container_url}/{blob_name}"

                # Skip if already processed
                if blob_url_no_token in processed_files:
                    continue

                log_message(f"Starting: {blob_url_no_token}")

                # ---- DOWNLOAD CYZ (no SAS, AAD credential) ----
                with open(self.cyz_file, "wb") as fh:
                    get_blob_client(account_url, container_name, blob_name, anonymous=False).download_blob().readinto(fh)
                log_message(f"Success: Blob downloaded for {blob_url_no_token}")

                # ---- CONVERT: cyz -> json ----
                try:
                    load_file(self.path_entry.get(), self.cyz_file, self.json_file)
                    log_message(f"Success: Cyz2json applied {blob_url_no_token}")
                except Exception as e:
                    log_message(f"Error: cyz2json failed for {blob_url_no_token}: {e}")
                    continue

                # ---- CONVERT: json -> listmode CSV ----
                try:
                    to_listmode(self.json_file, self.listmode_file)
                    log_message(f"Success: Listmode applied {blob_url_no_token}")
                except Exception as e:
                    log_message(f"Error: to_listmode failed for {blob_url_no_token}: {e}")
                    continue

                # ---- BEAD PROTOCOL POST-PROCESSING BEFORE CLASSIFICATION ----
                try:
                    with open(self.json_file, "r", encoding="utf-8-sig") as jf:
                        full_js = json.load(jf)
                    print('measurement_settings = full_js.get("instrument", {}).get("measurementSettings", {})')
                    measurement_settings = full_js.get("instrument", {}).get("measurementSettings", {})
                    cyto_settings = measurement_settings.get("CytoSettings", {})
                    protocol_packet = {
                        "instrument.measurementSettings.CytoSettings.IIFCheck": cyto_settings.get("IIFCheck"),
                        "instrument.measurementSettings.CytoSettings.IsBeadsMeasurement": cyto_settings.get(
                            "IsBeadsMeasurement",
                            measurement_settings.get("beads_measurement_2"),
                        ),
                        "instrument.measurementSettings.beads_measurement_2": measurement_settings.get("beads_measurement_2"),
                    }
                    detected_protocol = detect_sampling_protocol(protocol_packet)
                except Exception as e:
                    detected_protocol = "unknownprotocol"
                    log_message(f"Warning: could not detect sampling protocol for {blob_url_no_token}: {e}")

                if detected_protocol == "beadsprotocol":
                    try:
                        qc_plots.update_after_file(
                            self.json_file,
                            None,
                            self.plots_dir,
                            listmode_csv=self.listmode_file,
                        )
                        log_message(
                            "Success: bead calibration updated before classification "
                            f"for {blob_url_no_token}"
                        )
                    except Exception as e:
                        log_message(f"Error: bead calibration failed for {blob_url_no_token}: {e}")
                        continue
                    log_message(f"Success: counted {blob_url_no_token}")
                    continue

                # ---- APPLY PYTHON MODEL -> predictions CSV ----
                try:
                    predictions_file = os.path.join(
                        self.tool_dir,
                        f"{os.path.splitext(os.path.basename(self.cyz_file))[0]}_predictions.csv",
                    )
                    apply_python_model(self.listmode_file, predictions_file, self.model_path)
                    log_message(f"Success: Inferences made for {blob_url_no_token}")
                except Exception as e:
                    log_message(f"Error: model inference failed for {blob_url_no_token}: {e}")
                    continue

                # ---- COUNTS CSV ----
                try:
                    predictions_df = pd.read_csv(predictions_file)
                    prediction_counts_path = predictions_file + "_counts.csv"
                    counts = predictions_df["predicted_label"].value_counts().reset_index()
                    counts.columns = ["class", "count"]
                    counts.to_csv(prediction_counts_path, index=False)
                except Exception as e:
                    log_message(f"Warning: could not compute class counts for {blob_url_no_token}: {e}")
                    prediction_counts_path = None

                # ---- QC plots + dashboard packet ----
                try:
                    qc_plots.update_after_file(self.json_file, predictions_file, self.plots_dir, listmode_csv=self.listmode_file)
                except Exception as e:
                    log_message(f"Warning: QC update failed for {blob_url_no_token}: {e}")

                # ---- OPTIONAL: 3D plot HTML per file (your existing code may generate this) ----
                plot3d_prediction_path = predictions_file + "_3d.html"
                
                
               
                
                if os.path.exists(plot3d_prediction_path):
                    # leave for upload
                    pass
                else:
                    # If not created earlier, skip silently
                    plot3d_prediction_path = None

                # ---- UPLOAD OUTPUTS (AAD, no SAS) ----
                try:
                    upload_to_blob(predictions_file, None, container_url, output_blob_folder)

                    if prediction_counts_path:
                        upload_to_blob(prediction_counts_path, None, container_url, output_blob_folder)

                    if plot3d_prediction_path and os.path.exists(plot3d_prediction_path):
                        upload_to_blob(plot3d_prediction_path, None, container_url, output_blob_folder)

                    log_message(f"Success: Uploaded {blob_url_no_token}")

                except Exception as e:
                    log_message(f"Error: upload failed for {blob_url_no_token}: {e}")

                # ---- HOUSEKEEPING ----
                try:
                    if plot3d_prediction_path and os.path.exists(plot3d_prediction_path):
                        delete_file(plot3d_prediction_path)

                    delete_file(predictions_file)

                    if prediction_counts_path and os.path.exists(prediction_counts_path):
                        delete_file(prediction_counts_path)

                except Exception as e:
                    log_message(f"Warning: cleanup failed for {blob_url_no_token}: {e}")
                    
                # ---- LOG SUCCESS ----
                log_message(f"Success: counted {blob_url_no_token}")

            # Done
            messagebox.showinfo("Success", "All files processed and uploaded.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed during processing: {e}")
            raise FileNotFoundError





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