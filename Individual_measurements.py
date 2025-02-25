import os
import sys
import csv
import numpy as np
import tifffile as tiff
from glob import glob
import matplotlib.pyplot as plt
# Ensure the parent directory is in the Python path
sys.path.append("/Volumes/sils-mc/13776452/Python_scripts")

from Cell_tracking import segment_and_extract_centroids  # Import the tracking function
from Intensity_measurements import measure_intensity, create_cytoplasm_roi, segment_nucleus  # Import the intensity measurement functions

def measure_cell_intensities(image_stack, tracking_data):
    cell_intensity_data = {label: {"nucleus": [], "cytoplasm": []} for label in tracking_data.keys()}

    for time_index in range(image_stack.shape[0]):
        frame_image = image_stack[time_index, 0]  # Channel index for nucleus and cytoplasm data
        nucleus_mask = segment_nucleus(frame_image)  # Create nucleus mask for the frame
        cytoplasm_mask = create_cytoplasm_roi(nucleus_mask)  # Create cytoplasm mask for the frame

        # Debug: Print unique labels in masks
        print(f"Frame {time_index}: Unique labels in nucleus mask: {np.unique(nucleus_mask)}")
        print(f"Frame {time_index}: Unique labels in cytoplasm mask: {np.unique(cytoplasm_mask)}")

        for label, centroids in tracking_data.items():
            centroid = centroids[time_index]
            if centroid is not None:
                y, x = int(centroid[0]), int(centroid[1])  # Convert centroid to coordinates

                # Debug: Print centroid
                print(f"Frame {time_index}, Label {label}, Centroid: {centroid}")

                # Find the region corresponding to the cell
                nucleus_region_mask = (nucleus_mask == nucleus_mask[y, x]).astype(np.uint8)
                cytoplasm_region_mask = (cytoplasm_mask == cytoplasm_mask[y, x]).astype(np.uint8)

                # Debug: Visualize region masks
                plt.figure(figsize=(8, 8))
                plt.title(f"Nucleus Region for Cell {label}")
                plt.imshow(nucleus_region_mask, cmap="gray")
                plt.show()

                plt.figure(figsize=(8, 8))
                plt.title(f"Cytoplasm Region for Cell {label}")
                plt.imshow(cytoplasm_region_mask, cmap="gray")
                plt.show()

                # Measure intensities
                nucleus_intensity = np.mean(frame_image[nucleus_region_mask == 1])
                cytoplasm_intensity = np.mean(frame_image[cytoplasm_region_mask == 1])

                # Store intensities
                cell_intensity_data[label]["nucleus"].append(nucleus_intensity)
                cell_intensity_data[label]["cytoplasm"].append(cytoplasm_intensity)
            else:
                # If the cell is lost, append None
                cell_intensity_data[label]["nucleus"].append(None)
                cell_intensity_data[label]["cytoplasm"].append(None)

    return cell_intensity_data

def save_individual_intensities_to_csv(cell_intensity_data, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cell_Label", "Frame", "Nucleus_Intensity", "Cytoplasm_Intensity"])
        for label, intensities in cell_intensity_data.items():
            for frame, (nucleus_intensity, cytoplasm_intensity) in enumerate(zip(intensities["nucleus"], intensities["cytoplasm"])):
                print(f"Saving: Label {label}, Frame {frame}: Nucleus Intensity = {nucleus_intensity}, Cytoplasm Intensity = {cytoplasm_intensity}")
                writer.writerow([label, frame, nucleus_intensity, cytoplasm_intensity])

# Loop through each image stack in the folder
# for file_path in glob(os.path.join(input_folder, "*.tif")):  # Adjust file extension if needed
#     image_stack = tiff.imread(file_path)
#     print(f"Processing file: {file_path}")

#     # Track cells and get tracking data
#     tracking_data = segment_and_extract_centroids(image_stack)
#     print(f"Tracking data for {file_path}:", tracking_data)

#     # Measure cell intensities
#     cell_intensity_data = measure_cell_intensities(image_stack, tracking_data)
#     print(f"Intensity data for {file_path}:", cell_intensity_data)

#     # Save intensity data to CSV
#     output_csv = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + "_intensities.csv")
#     save_intensities_to_csv(cell_intensity_data, output_csv)
#     print(f"Intensity data saved to {output_csv}")
