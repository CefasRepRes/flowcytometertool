# metadata_handler.py

import os
import json

class MetadataHandler:
    def __init__(self, temp_dir):
        self.metadata_dir = os.path.join(temp_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

    def save_metadata(self, image_file, confidence, species):
        metadata = {
            "confidence": confidence,
            "species": species
        }
        metadata_file = os.path.join(self.metadata_dir, f"{image_file}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

    def load_metadata(self, image_file):
        metadata_file = os.path.join(self.metadata_dir, f"{image_file}.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {"confidence": "", "species": ""}  # Return empty defaults if no metadata exists
