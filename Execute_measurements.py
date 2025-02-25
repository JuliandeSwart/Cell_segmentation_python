import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, disk, binary_dilation
import os
from glob import glob
import csv
import sys
sys.path.append("/Volumes/sils-mc/13776452/Python_scripts")

from Intensity_measurements import (segment_nucleus, create_cytoplasm_roi, 
                                    measure_intensities_for_all_timepoints, 
                                    save_intensities_to_csv)

from Individual_measurements import (segment_and_extract_centroids, measure_cell_intensities, save_individual_intensities_to_csv)

from Cell_tracking import (extract_centroids, segment_and_extract_centroids, visualize_tracked_centroids)

input_folder = "/Volumes/sils-mc/13776452/Python_scripts/Data_input_test"
output_folder = "/Volumes/sils-mc/13776452/Python_scripts/Data_output_test"
os.makedirs(output_folder, exist_ok=True)

for file_path in glob(os.path.join(input_folder, "*.tif")):
    image_stack = tiff.imread(file_path)
    print(f"Processing file: {file_path}")

    num_timepoints, num_channels, height, width = image_stack.shape
    time_index = 0
    original_image = image_stack[:, 0][time_index]
    segmented_mask = segment_nucleus(original_image)

    nucleus_channel = image_stack[:, 1]
    segmented_masks = [segment_nucleus(nucleus_channel[time_index]) for time_index in range(nucleus_channel.shape[0])]
    cytoplasm_rois = [create_cytoplasm_roi(mask, dilation_radius=5) for mask in segmented_masks]

    cyan_channel = image_stack[:, 0]
    nucleus_intensities_cyan, cytoplasm_intensities_cyan = measure_intensities_for_all_timepoints(cyan_channel, segmented_masks, cytoplasm_rois)

    green_channel = image_stack[:, 2]
    nucleus_intensities_green, cytoplasm_intensities_green = measure_intensities_for_all_timepoints(green_channel, segmented_masks, cytoplasm_rois)

    timepoints = range(len(nucleus_intensities_cyan))
    output_csv = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + "_results.csv")

    save_intensities_to_csv(nucleus_intensities_green, cytoplasm_intensities_green, timepoints, output_csv)

    image_stack = tiff.imread(file_path)
    print(f"Processing file: {file_path}")

    tracking_data = segment_and_extract_centroids(image_stack)
    # print(f"Tracking data for {file_path}:", tracking_data)

    #visualize_tracked_centroids(image_stack, tracking_data)
    cell_intensity_data = measure_cell_intensities(image_stack, tracking_data)
    # print(f"Intensity data for {file_path}:", cell_intensity_data)

    for label, intensities in cell_intensity_data.items():
        for frame, (nucleus_intensity, cytoplasm_intensity) in enumerate(zip(intensities["nucleus"], intensities["cytoplasm"])):
            # print(f"Label {label}, Frame {frame}: Nucleus Intensity = {nucleus_intensity}, Cytoplasm Intensity = {cytoplasm_intensity}")  # Commented out
            pass

    output_csv = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + "_intensities.csv")
    save_individual_intensities_to_csv(cell_intensity_data, output_csv)
    print(f"Intensity data saved to {output_csv}")
    print(f"Results saved to {output_csv}")
