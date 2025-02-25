import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, disk, binary_dilation
import os
from glob import glob
import csv

# Enable interactive mode
plt.ion()

# Function to visualize a specific time point and channel
def visualize_timepoint(stack, time_index, channel_index):
    plt.figure(figsize=(6, 6))
    plt.title(f"Time Point: {time_index}, Channel: {channel_index}")
    plt.imshow(stack[time_index, channel_index], cmap="gray")
    plt.axis("off")
    plt.show(block=False)
    plt.waitforbuttonpress()  # Wait for a button press to continue

# Function to segment nucleus
def segment_nucleus(image):
    thresh = threshold_otsu(image)
    binary_mask = image > thresh
    clean_mask = remove_small_objects(binary_mask, min_size=30)
    clean_mask = remove_small_holes(clean_mask, area_threshold=50)
    opened_mask = binary_opening(clean_mask, footprint=disk(2))
    return opened_mask

# Function to create cytoplasm ROI
def create_cytoplasm_roi(nucleus_mask, dilation_radius=5):
    dilated_mask = binary_dilation(nucleus_mask, footprint=disk(dilation_radius))
    cytoplasm_ring = dilated_mask ^ nucleus_mask
    return cytoplasm_ring

# Function to measure mean intensity in a given mask (nucleus or cytoplasm)
def measure_intensity(image, mask):
    return np.mean(image[mask == 1])

# Function to measure intensities for both nucleus and cytoplasm for all time points
def measure_intensities_for_all_timepoints(image_stack, nucleus_masks, cytoplasm_masks):
    nucleus_intensities = []
    cytoplasm_intensities = []
    for time_index in range(image_stack.shape[0]):
        current_image = image_stack[time_index]
        nucleus_intensity = measure_intensity(current_image, nucleus_masks[time_index])
        cytoplasm_intensity = measure_intensity(current_image, cytoplasm_masks[time_index])
        nucleus_intensities.append(nucleus_intensity)
        cytoplasm_intensities.append(cytoplasm_intensity)
    return nucleus_intensities, cytoplasm_intensities

# Function to save intensities to CSV
def save_intensities_to_csv(nucleus_intensities, cytoplasm_intensities, timepoints, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timepoint", "Nucleus", "Cytoplasm"])
        for timepoint, nucleus_intensity, cytoplasm_intensity in zip(timepoints, nucleus_intensities, cytoplasm_intensities):
            writer.writerow([timepoint, nucleus_intensity, cytoplasm_intensity])
