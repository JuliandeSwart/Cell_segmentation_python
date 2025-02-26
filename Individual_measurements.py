import os
import sys
import csv
import numpy as np
import tifffile as tiff
from glob import glob
import matplotlib.pyplot as plt
from scipy.ndimage import label
# Ensure the parent directory is in the Python path
sys.path.append("/Volumes/sils-mc/13776452/Python_scripts")

from Cell_tracking import segment_and_extract_centroids  # Import the tracking function
from Intensity_measurements import measure_intensity, create_cytoplasm_roi, segment_nucleus  # Import the intensity measurement functions

def measure_cell_intensities(image_stack, tracking_data):
    cell_intensity_data = {cell_label: {"nucleus": [], "cytoplasm": []} for cell_label in tracking_data.keys()}

    for time_index in range(image_stack.shape[0]):
        frame_image = image_stack[time_index, 0]  # Channel index for nucleus and cytoplasm data
        nucleus_mask = segment_nucleus(frame_image)  # Create nucleus mask for the frame
        
        # Label connected components in the nucleus mask
        labeled_nucleus_mask, num_nucleus_labels = label(nucleus_mask)
        
        print(f"Frame {time_index}: Found {num_nucleus_labels} nucleus components")
        
        # Visualization of labeled nuclei
        plt.figure(figsize=(8, 8))
        plt.title("Labeled Nuclei")
        plt.imshow(labeled_nucleus_mask, cmap='nipy_spectral')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        # Create an empty cytoplasm mask
        full_cytoplasm_mask = np.zeros_like(nucleus_mask)

        for cell_label, centroids in tracking_data.items():
            centroid = centroids[time_index]
            if centroid is not None:
                y, x = int(centroid[0]), int(centroid[1])  # Convert centroid to coordinates

                # Debug: Print centroid
                print(f"Frame {time_index}, Label {cell_label}, Centroid: {y, x}")
                
                # Check if coordinates are within image bounds
                if y < 0 or y >= labeled_nucleus_mask.shape[0] or x < 0 or x >= labeled_nucleus_mask.shape[1]:
                    print(f"Warning: Centroid {y, x} is outside image bounds ({labeled_nucleus_mask.shape})")
                    cell_intensity_data[cell_label]["nucleus"].append(None)
                    cell_intensity_data[cell_label]["cytoplasm"].append(None)
                    continue

                # Find the region corresponding to the cell
                nucleus_region_label = labeled_nucleus_mask[y, x]
                
                print(f"Cell {cell_label} at {y, x} has nucleus label {nucleus_region_label}")
                
                # Skip if the centroid is in the background (label = 0)
                if nucleus_region_label == 0:
                    print(f"Warning: Centroid {y, x} for cell {cell_label} is in the background of the nucleus mask")
                    cell_intensity_data[cell_label]["nucleus"].append(None)
                    cell_intensity_data[cell_label]["cytoplasm"].append(None)
                    continue

                # Create a mask for this specific nucleus
                nucleus_region_mask = (labeled_nucleus_mask == nucleus_region_label).astype(np.uint8)
                
                # Create a cytoplasm mask specifically for this nucleus
                cytoplasm_region_mask = create_cytoplasm_roi(nucleus_region_mask, dilation_radius=5)
                
                # Ensure the cytoplasm mask doesn't overlap with any nucleus
                cytoplasm_region_mask = cytoplasm_region_mask & ~nucleus_mask
                
                # Count pixels in each mask to verify
                nucleus_pixels = np.sum(nucleus_region_mask)
                cytoplasm_pixels = np.sum(cytoplasm_region_mask)
                print(f"Cell {cell_label}: Nucleus mask has {nucleus_pixels} pixels, Cytoplasm mask has {cytoplasm_pixels} pixels")

                # Visualize the individual cell masks
                # plt.figure(figsize=(12, 6))
                # plt.subplot(1, 2, 1)
                # plt.title(f"Nucleus Region for Cell {cell_label}")
                # plt.imshow(nucleus_region_mask, cmap="gray")
                
                # plt.subplot(1, 2, 2)
                # plt.title(f"Cytoplasm Region for Cell {cell_label}")
                # plt.imshow(cytoplasm_region_mask, cmap="gray")
                # plt.tight_layout()
                # plt.show(block=False)
                # plt.waitforbuttonpress()  
                
                # Measure intensities only for this cell
                if nucleus_pixels > 0:
                    nucleus_intensity = np.mean(frame_image[nucleus_region_mask == 1])
                else:
                    nucleus_intensity = None
                    print(f"Warning: No nucleus pixels found for cell {cell_label}")
                
                if cytoplasm_pixels > 0:
                    cytoplasm_intensity = np.mean(frame_image[cytoplasm_region_mask == 1])
                else:
                    cytoplasm_intensity = None
                    print(f"Warning: No cytoplasm pixels found for cell {cell_label}")

                # Store intensities
                cell_intensity_data[cell_label]["nucleus"].append(nucleus_intensity)
                cell_intensity_data[cell_label]["cytoplasm"].append(cytoplasm_intensity)
                print(f"Cell {cell_label} intensities: Nucleus={nucleus_intensity}, Cytoplasm={cytoplasm_intensity}")
            else:
                # If the cell is lost, append None
                cell_intensity_data[cell_label]["nucleus"].append(None)
                cell_intensity_data[cell_label]["cytoplasm"].append(None)

    return cell_intensity_data

def save_individual_intensities_to_csv(cell_intensity_data, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cell_Label", "Frame", "Nucleus_Intensity", "Cytoplasm_Intensity"])
        for cell_label, intensities in cell_intensity_data.items():
            for frame, (nucleus_intensity, cytoplasm_intensity) in enumerate(zip(intensities["nucleus"], intensities["cytoplasm"])):
                print(f"Saving: Label {cell_label}, Frame {frame}: Nucleus Intensity = {nucleus_intensity}, Cytoplasm Intensity = {cytoplasm_intensity}")
                writer.writerow([cell_label, frame, nucleus_intensity, cytoplasm_intensity])

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
#     save_individual_intensities_to_csv(cell_intensity_data, output_csv)
#     print(f"Intensity data saved to {output_csv}")
