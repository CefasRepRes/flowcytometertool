import os
import json
import pandas as pd
from flowcytometer_tool.tabs.download_train.listmode import extract


def convert_json_to_listmode(output_path):
    for root, _, files in os.walk(output_path):
        for file in files:
            if file.lower().endswith(".json"):
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
                    print(f"Converted: {json_file} → {listmode_file}")
                except Exception as e:
                    print(f"Error processing file: {json_file}")
                    print(f"Exception: {e}")
