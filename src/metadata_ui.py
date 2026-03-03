# metadata_ui.py

import tkinter as tk
from tkinter import messagebox

class MetadataUI:
    def __init__(self, on_save):
        self.on_save = on_save  # Callback function for saving metadata

        # Create a new window for metadata entry
        self.window = tk.Toplevel()
        self.window.title("Metadata Entry")
        self.window.geometry("400x300")

        # Metadata Input
        self.confidence_label = tk.Label(self.window, text="confidence (Y/N):")
        self.confidence_label.pack(pady=5)

        self.confidence_entry = tk.Entry(self.window, width=10)
        self.confidence_entry.pack(pady=5)

        self.species_label = tk.Label(self.window, text="Suspected Species:")
        self.species_label.pack(pady=5)

        self.species_entry = tk.Entry(self.window, width=100)
        self.species_entry.pack(pady=5)

        # Navigation Buttons
        self.prev_button = tk.Button(self.window, text="Previous", command=self.prev_image, state=tk.DISABLED)
        self.next_button = tk.Button(self.window, text="Next", command=self.next_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=20)
        self.next_button.pack(side=tk.RIGHT, padx=20)

    def prev_image(self):
        # Placeholder for navigation logic
        self.on_save()  # Automatically save metadata before switching images

    def next_image(self):
        if self.current_image_index < len(self.tif_files) - 1:
            self.save_metadata()  # Automatically save metadata before switching images
            self.current_image_index += 1
            self.display_image(self.tif_files[self.current_image_index])
            self.update_navigation_buttons()
            
    def display_metadata(self, image_file, metadata):
        self.confidence_entry.delete(0, tk.END)
        self.confidence_entry.insert(0, metadata["confidence"])
        self.species_entry.delete(0, tk.END)
        self.species_entry.insert(0, metadata["species"])

    def update_navigation_buttons(self, current_index, total_images):
        self.prev_button.config(state=tk.NORMAL if current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if current_index < total_images - 1 else tk.DISABLED)

    def show(self):
        self.window.deiconify()  # Show the metadata window
