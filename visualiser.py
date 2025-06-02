import pandas as pd
import numpy as np
from tkinter import *
from tkinter import ttk, filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import json


plot_area = None
color_key = None
df = None  
pumped_volume = 1
selector = None
polygons = []  # List to store polygons
predictions_datas = []  # List to store predictions_datas
current_polygon = None  # Store the current polygon being drawn
current_predictions_data = None  # Store the current predictions_data being entered

def load_csv(file_path=None):
    global df
    if file_path is None:
        file_path = filedialog.askopenfilename()
    if file_path:
        df = pd.read_csv(file_path)
        if 'label' in df.columns and 'predictions_data' in df.columns:
            df['agreement'] = df['label'] == df['predictions_data']
        else:
            df['agreement'] = pd.NA
        if 'predictions_data' not in df.columns:
            df['predictions_data'] = pd.NA
        color_options = []
        if 'label' in df.columns:
            color_options.append('label')
        if 'predictions_data' in df.columns:
            color_options.append('predictions_data')
        if 'agreement' in df.columns:
            color_options.append('agreement')
        variables = df.columns.tolist()
        x_variable_combobox['values'] = variables
        y_variable_combobox['values'] = variables
        color_variable_combobox['values'] = color_options if color_options else variables
        if 'FWS_total' in variables and 'FWS_maximum' in variables:
            x_variable_combobox.set('FWS_total')
            y_variable_combobox.set('FWS_maximum')
        else:
            x_variable_combobox.set(variables[0])
            y_variable_combobox.set(variables[1])
        color_variable_combobox.set('label' if 'label' in df.columns else (color_options[0] if color_options else variables[0]))  # Default color variable
        update_plot()
        update_summary_table()

def load_json(file_path=None):
    global pumped_volume
    if file_path is None:
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        try:
            with open(file_path, 'r') as f:
                lines = []
                for _ in range(100):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                json_content = ''.join(lines)
                start_index = json_content.find('"pumpedVolume": ')
                if start_index != -1:
                    end_index = json_content.find(',', start_index)
                    if end_index != -1:
                        pumped_volume_str = json_content[start_index + len('"pumpedVolume": '):end_index]
                        pumped_volume = float(pumped_volume_str.strip())
                        pumped_volume_label.config(text=f"Dividing totals by pumped volume of {pumped_volume} unknown units!")
                    else:
                        pumped_volume_label.config(text="Comma not found after pumpedVolume")
                else:
                    pumped_volume_label.config(text="pumpedVolume not found in JSON")
        except FileNotFoundError:
            pumped_volume_label.config(text=f"File not found: {file_path}")
        except ValueError:
            pumped_volume_label.config(text="Unable to parse pumpedVolume value")
        update_summary_table()

def onselect(verts):
    global current_polygon, current_predictions_data
    current_polygon = verts
    current_predictions_data = simpledialog.askstring("Input", "Enter your predictions_data for the selected points:")
    commit_polygon()
    update_plot()

def commit_polygon():
    global polygons, predictions_datas, current_polygon, current_predictions_data, selector
    if current_polygon is not None and current_predictions_data is not None:
        path = Path(current_polygon)
        selected_points = df.apply(lambda row: path.contains_point((row[x_variable_combobox.get()], row[y_variable_combobox.get()])), axis=1)
        polygons.append({
            "coordinates": current_polygon,
            "x_axis": x_variable_combobox.get(),
            "y_axis": y_variable_combobox.get()
        })
        predictions_datas.append(current_predictions_data)
        # Add the new category to the predictions_data column if it's not already present
        if current_predictions_data not in df['predictions_data'].cat.categories:
            df['predictions_data'] = df['predictions_data'].cat.add_categories([current_predictions_data])
        df.loc[selected_points, 'predictions_data'] = current_predictions_data
        current_polygon = None
        current_predictions_data = None
        update_summary_table()
        # Reset the PolygonSelector
        selector.disconnect_events()
        selector = PolygonSelector(ax, onselect)


def update_plot():
    global plot_area, color_key, selector, ax

    if df is None:
        return  # Return if no dataframe is loaded
    x_variable = x_variable_combobox.get()
    y_variable = y_variable_combobox.get()
    color_variable = color_variable_combobox.get()
    if 'label' in df.columns:
        df["label"] = pd.Categorical(df["label"], ordered=True)
    if 'predictions_data' in df.columns:
        df["predictions_data"] = pd.Categorical(df["predictions_data"], ordered=True)
    categories = set()
    if 'label' in df.columns:
        categories.update(df["label"].dropna().cat.categories)
    if 'predictions_data' in df.columns:
        categories.update(df["predictions_data"].dropna().cat.categories)
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    num_colors = len(colors)
    color_key = dict(zip(categories, colors[:len(categories)]))
    if len(categories) > num_colors:
        color_key = {cat: colors[i % num_colors] for i, cat in enumerate(categories)}
    x_min = np.percentile(df[x_variable], 1.5)
    x_max = np.percentile(df[x_variable], 98.5)
    y_min = np.percentile(df[y_variable], 1.5)
    y_max = np.percentile(df[y_variable], 98.5)
    try:
        x_min = float(xmin_entry.get())
    except ValueError:
        pass
    try:
        x_max = float(xmax_entry.get())
    except ValueError:
        pass
    try:
        y_min = float(ymin_entry.get())
    except ValueError:
        pass
    try:
        y_max = float(ymax_entry.get())
    except ValueError:
        pass
    if plot_area:
        plot_area.get_tk_widget().destroy()
    fig, ax = plt.subplots()
    for category in categories:
        subset = df[df[color_variable] == category]
        ax.scatter(subset[x_variable], subset[y_variable], label=category, color=color_key[category], alpha=0.4, s=0.4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)
    if log_x_var.get():
        ax.set_xscale('log')
    if log_y_var.get():
        ax.set_yscale('log')
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
                       for label, color in color_key.items()]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(right=0.6)
    plot_area = FigureCanvasTkAgg(fig, root)
    plot_area.get_tk_widget().grid(row=6, column=0, columnspan=6, padx=10, pady=10)

    # Add PolygonSelector
    selector = PolygonSelector(ax, onselect)

def update_summary_table():
    if df is None:
        return  # Return if no dataframe is loaded
    if 'label' in df.columns:
        unique_labels = pd.unique(df[['label']].values.ravel('K'))
    if 'predictions_data' in df.columns:
        unique_labels = pd.unique(df[['predictions_data']].values.ravel('K'))
    if 'label' in df.columns and 'predictions_data' in df.columns:
        unique_labels = pd.unique(df[['label', 'predictions_data']].values.ravel('K'))
    summary_data = []
    for item in unique_labels:
        labelled_count = (df['label'] == item).sum() / pumped_volume if 'label' in df.columns else 0
        predictions_count = (df['predictions_data'] == item).sum() / pumped_volume if 'predictions_data' in df.columns else 0
        if 'label' in df.columns and 'predictions_data' in df.columns:
            percentage = (predictions_count / labelled_count * 100) if labelled_count > 0 else 0
        else:
            percentage = 0
        summary_data.append((item, labelled_count, predictions_count, percentage))
    for row in summary_table.get_children():
        summary_table.delete(row)
    for item, labelled_count, predictions_count, percentage in summary_data:
        summary_table.insert("", "end", values=(item, labelled_count, predictions_count, f"{percentage:.2f}%"))


def save_metadata(file_path):
    metadata = {
        "polygons": polygons,
        "predictions_datas": predictions_datas
    }
    metadata_file_path = file_path.replace(".csv", "_metadata.json")
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f)

def save_csv():
    if df is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df.to_csv(file_path, index=False)
            save_metadata(file_path)
            
            
root = Tk()
root.title("Data Plotter")

load_button = Button(root, text="Load predictions.csv", command=lambda: load_csv())
load_button.grid(row=0, column=0, padx=10, pady=10)

load_json_button = Button(root, text="Load .json to divide counts by volume", command=lambda: load_json())
load_json_button.grid(row=0, column=3, padx=10, pady=10)

save_button = Button(root, text="Save CSV", command=save_csv)
save_button.grid(row=0, column=4, padx=10, pady=10)

pumped_volume_label = Label(root, text="")
pumped_volume_label.grid(row=1, column=3, padx=10, pady=10)

x_variable_combobox = ttk.Combobox(root)
x_variable_combobox.grid(row=0, column=1, padx=10, pady=10)

y_variable_combobox = ttk.Combobox(root)
y_variable_combobox.grid(row=0, column=2, padx=10, pady=10)

color_variable_combobox = ttk.Combobox(root)
color_variable_combobox.grid(row=1, column=0, padx=10, pady=10)

xmin_label = Label(root, text="X min")
xmin_label.grid(row=2, column=0, padx=5, pady=5)
xmin_entry = Entry(root)
xmin_entry.grid(row=2, column=1, padx=5, pady=5)

xmax_label = Label(root, text="X max")
xmax_label.grid(row=2, column=2, padx=5, pady=5)
xmax_entry = Entry(root)
xmax_entry.grid(row=2, column=3, padx=5, pady=5)

ymin_label = Label(root, text="Y min")
ymin_label.grid(row=3, column=0, padx=5, pady=5)
ymin_entry = Entry(root)
ymin_entry.grid(row=3, column=1, padx=5, pady=5)

ymax_label = Label(root, text="Y max")
ymax_label.grid(row=3, column=2, padx=5, pady=5)
ymax_entry = Entry(root)
ymax_entry.grid(row=3, column=3, padx=5, pady=5)

log_x_var = BooleanVar()
log_x_check = Checkbutton(root, text="Log X-axis", variable=log_x_var)
log_x_check.grid(row=4, column=0, padx=5, pady=5)

log_y_var = BooleanVar()
log_y_check = Checkbutton(root, text="Log Y-axis", variable=log_y_var)
log_y_check.grid(row=4, column=1, padx=5, pady=5)

update_button = Button(root, text="Update Plot", command=update_plot)
update_button.grid(row=1, column=1, columnspan=2, padx=10, pady=10)

# Create Treeview for summary table
summary_table = ttk.Treeview(root, columns=("Instance", "Total in Labelled", "Total in Predictions", "Predicted as % of Labelled"), show='headings')
summary_table.heading("Instance", text="Instance")
summary_table.heading("Total in Labelled", text="Total in Labelled")
summary_table.heading("Total in Predictions", text="Total in Predictions")
summary_table.heading("Predicted as % of Labelled", text="Predicted as % of Labelled")
summary_table.grid(row=5, column=0, columnspan=6, padx=10, pady=10)

# Load default CSV on start
default_csv_path = '..\\..\\outputs\\predictions.csv'
try:
    load_csv(default_csv_path)
except FileNotFoundError:
    print(f"Default file '{default_csv_path}' not found. Please load a CSV file manually.")

root.mainloop()