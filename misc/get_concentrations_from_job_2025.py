import pandas as pd
from azure.storage.blob import BlobServiceClient
import io
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import plotly.express as px
import pandas as pd



def get_sas_token(file_path):
    with open(file_path, 'r') as file:
        sas_token = file.read().strip()
    return sas_token

sas_token = get_sas_token('C:/Users/JR13/Documents/authenticationkeys/flowcytosaSAS.txt')
account_url = "https://citprodflowcytosa.blob.core.windows.net"
blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
container_name = "hdduploadnov2024"
container_client = blob_service_client.get_container_client(container_name)
master_table = pd.DataFrame()
all_blobs = list(container_client.list_blobs())


for blob in all_blobs:
    print(blob.name)
    if blob.name.endswith("counts.csv"):
        blob_client = container_client.get_blob_client(blob)
        download_stream = blob_client.download_blob()
        csv_content = download_stream.readall()
        df = pd.read_csv(io.BytesIO(csv_content))
        df_transposed = df.T.reset_index()
        df_transposed.columns = df_transposed.iloc[0]
        df_transposed = df_transposed[1:]
        df_transposed['filename'] = blob.name
        base_name = blob.name.split(".cyz_")[0]
        instruments_blob_name = None
        for b in all_blobs:
            if base_name in b.name and "instrument" in b.name:
                instruments_blob_name = b.name
                break
        if instruments_blob_name:
            instruments_blob_client = container_client.get_blob_client(instruments_blob_name)
            try:
                instruments_download_stream = instruments_blob_client.download_blob()
                instruments_csv_content = instruments_download_stream.readall()
                instruments_df = pd.read_csv(io.BytesIO(instruments_csv_content))
                print(instruments_df)
                analysed_volume = instruments_df.at[0, "measurementResults_analysedVolume"]
                start_time = instruments_df.at[0, "measurementResults_start"]
                duration = instruments_df.at[0, "measurementResults_duration"]
                pump_speed = instruments_df.at[0, "measurementSettings_CytoSettings_ConfiguredSamplePumpSpeedSetting"]
                df_transposed['analysed_volume'] = analysed_volume
                df_transposed['start_time'] = start_time
                df_transposed['duration'] = duration
                df_transposed['pump_speed'] = pump_speed
            except Exception as e:
                print(f"Could not load instruments file for {blob.name}: {e}")
                df_transposed['analysed_volume'] = None
                df_transposed['start_time'] = None
                df_transposed['duration'] = None
                df_transposed['pump_speed'] = None
        else:
            df_transposed['analysed_volume'] = None
            df_transposed['start_time'] = None
            df_transposed['duration'] = None
            df_transposed['pump_speed'] = None
        master_table = pd.concat([master_table, df_transposed], ignore_index=True, sort=False)

columns_to_calculate_concentration = ['RedPico', 'Orapicoprok', 'Other', 'RedNano','nophytoplankton', 'noiseum', 'no_phytoplankton', 'YB_um_beads','Plant_detritus', 'OraNano_crypto', 'RWS_um_beads', 'Beads_rest','RedMicro', 'Bubbles', 'OraNano', 'C_undetermined']

for column in columns_to_calculate_concentration:
    master_table[f'{column}_concentration'] = master_table[column].astype(float) / master_table['analysed_volume'].astype(float) * 1000


master_table['start_time'] = pd.to_datetime(master_table['start_time'])

for column in columns_to_calculate_concentration:
    plt.scatter(master_table['start_time'], master_table[f'{column}_concentration'], label=f'{column}_concentration',s=0.1)

master_table.to_csv("master_table.csv", index=False)



fig = px.scatter(master_table, x='start_time', y=[f'{column}_concentration' for column in columns_to_calculate_concentration],
                 labels={'value': 'Concentration particle/ul', 'start_time': 'Time'},
                 title=' ')

fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Concentration particle/ul',
    legend_title='Concentration',
    xaxis=dict(tickangle=45),
    yaxis=dict(range=[0, 100000]),
    template='plotly_white'
)


fig.add_annotation(
    text='Dominant Class Concentration Map<br>The training data for these predictions were sets (labels) defined by Veronique Creach and exported from cytoclus.<br>The prediction model was trained using the script https://github.com/JoeRibeiro/lucinda-flow-cytometry/blob/NEW-FLOW-CYTOMETER-FILE/scripts/train_then_predict_no_python_using_lucinda_variables.Rmd<br>... at this point in time (commit SHA): 25923dd90c4cfec14d6634c0e3bdc23de73c8bf3<br>... applied to this training file: https://citprodflowcytosa.blob.core.windows.net/public/Mersey_LV01_9%202024-02-21%2012h27.cyz<br>This plot is a scrape of the inference folder at https://citprodflowcytosa.blob.core.windows.net/hdduploadnov2024/inference/. <br> Intermediate data files including individual particle data, instrument metadata and 3D plots of each individual sample can also be found there.<br>It should be obvious, but any GPS data are fictional at the moment.',
    xref='paper', yref='paper',
    x=0.5, y=1,
    showarrow=False,
    font=dict(size=12, color="Blue"),
    align="left"
)

fig.write_html("C:/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/coco_labeller_dev_fork/outputs/concentrations_timeseries.html")
fig.show()






# The gps from the plankton imager very rarely matches with the flow cytometer. Can we get all the gps from the ferrybox somewhere? For now we plot this and accept it is wrong.
# Read the GPS data from the CSV file
gps_data = []
with open(r'C:\Users\JR13\Documents\LOCAL_NOT_ONEDRIVE\coco_labeller_dev_fork\PI_gps.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        gps_data.append(row)

# Convert the GPS data to a DataFrame
gps_df = pd.DataFrame(gps_data[1:], columns=gps_data[0])

# Extract latitude, longitude, and datetime from 'backgroundgps'
gps_df[['latitude', 'longitude', 'datetime']] = gps_df['backgroundgps'].str.extract(r'\(([^,]+), ([^,]+), \'([^\']+)\'\)')

# Convert 'datetime' to datetime format, and 'latitude' and 'longitude' to numeric, coercing errors
gps_df['datetime'] = pd.to_datetime(gps_df['datetime'], format='%Y:%m:%d %H:%M:%S', errors='coerce')
gps_df['latitude'] = pd.to_numeric(gps_df['latitude'], errors='coerce')
gps_df['longitude'] = pd.to_numeric(gps_df['longitude'], errors='coerce')

# Drop rows with any NaN or NaT values resulting from coercion
gps_df.dropna(subset=['latitude', 'longitude', 'datetime'], inplace=True)

# Load the master_table.csv file into a DataFrame
master_table = pd.read_csv('master_table.csv')

# Convert the 'start_time' column to datetime, coercing errors
master_table['start_time'] = pd.to_datetime(master_table['start_time'], errors='coerce')

# Drop rows with NaN or NaT values in 'start_time'
master_table.dropna(subset=['start_time'], inplace=True)

# Convert timezone-aware datetime objects to UTC
master_table['start_time'] = master_table['start_time'].apply(lambda x: x.tz_convert('UTC') if x.tzinfo else x)

# Ensure both 'start_time' and 'datetime' columns are in datetime format
master_table['start_time'] = pd.to_datetime(master_table['start_time'], utc=True)
gps_df['datetime'] = pd.to_datetime(gps_df['datetime'], utc=True)

# Define the maximum allowable difference (e.g., 5 minutes)
max_diff = pd.Timedelta('1000 day')

# Merge the master_table with the GPS data on the nearest datetime within the maximum difference
merged_df = pd.merge_asof(
    master_table.sort_values('start_time'),
    gps_df.sort_values('datetime'),
    left_on='start_time',
    right_on='datetime',
    direction='nearest',
    tolerance=max_diff
)

timedifference = merged_df['start_time']-merged_df['datetime']


# Since it is nonsense, let's just make it look continuous
time_diffs = (merged_df['start_time'] - merged_df['start_time'][0]).dt.total_seconds().abs()
normalized_latitudes = (time_diffs - time_diffs.min()) / (time_diffs.max() - time_diffs.min()) * 90
normalized_longitudes = (time_diffs - time_diffs.min()) / (time_diffs.max() - time_diffs.min()) * 90
merged_df['latitude'] = normalized_latitudes
merged_df['longitude'] = normalized_longitudes
merged_df['dominant_class'] = merged_df[[f'{col}_concentration' for col in columns_to_calculate_concentration]].idxmax(axis=1)
geometry = [Point(xy) for xy in zip(merged_df['longitude'], merged_df['latitude'])]
geo_df = gpd.GeoDataFrame(merged_df, geometry=geometry)









fig = px.scatter_geo(
    geo_df,
    lat=geo_df.geometry.y,
    lon=geo_df.geometry.x,
    color='dominant_class',
    color_continuous_scale='tab20',
    labels={'dominant_class': 'Dominant Class'},
    projection='natural earth'
)



fig.add_annotation(
    text='Dominant Class Concentration Map<br>The training data for these predictions were sets (labels) defined by Veronique Creach and exported from cytoclus.<br>The prediction model was trained using the script https://github.com/JoeRibeiro/lucinda-flow-cytometry/blob/NEW-FLOW-CYTOMETER-FILE/scripts/train_then_predict_no_python_using_lucinda_variables.Rmd<br>... at this point in time (commit SHA): 25923dd90c4cfec14d6634c0e3bdc23de73c8bf3<br>... applied to this training file: https://citprodflowcytosa.blob.core.windows.net/public/Mersey_LV01_9%202024-02-21%2012h27.cyz<br>This plot is a scrape of the inference folder at https://citprodflowcytosa.blob.core.windows.net/hdduploadnov2024/inference/. <br> Intermediate data files including individual particle data, instrument metadata and 3D plots of each individual sample can also be found there.<br>It should be obvious, but any GPS data are fictional at the moment.',
    xref='paper', yref='paper',
    x=0.5, y=0.05,
    showarrow=False,
    font=dict(size=12, color="Blue"),
    align="left"
)

fig.update_layout(
    geo=dict(
        showland=True,
        landcolor='lightgrey',
    ),
    legend_title='Dominant Class',
    margin={"r":0,"t":0,"l":0,"b":0}
)

fig.write_html("C:/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/coco_labeller_dev_fork/outputs/dominant_class_map.html")
fig.show()

