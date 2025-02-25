import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, disk
from skimage.measure import label, regionprops
import os
from glob import glob
from scipy.spatial import distance
import sys
# Ensure the parent directory is in the Python path
sys.path.append("/Volumes/sils-mc/13776452/Python_scripts")

from Intensity_measurements import segment_nucleus

# Function to extract centroids from segmented mask

def extract_centroids(segmented_mask):
    # Label the segmented mask
    labeled_mask = label(segmented_mask)
    
    # Reassign labels to make them consecutive
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    reassigned_mask = np.zeros_like(labeled_mask)
    
    for new_label, old_label in enumerate(unique_labels, start=1):
        reassigned_mask[labeled_mask == old_label] = new_label
    
    # Extract centroids from the reassigned mask
    properties = regionprops(reassigned_mask)
    centroids = {prop.label: prop.centroid for prop in properties}
    
    return centroids, reassigned_mask


# Function to segment nuclei and extract centroids for all time points
def segment_and_extract_centroids(image_stack):
    # Segment the first frame and get the centroids and labels
    first_frame = image_stack[0, 1]  # Adjust channel index if needed
    segmented_mask_first_frame = segment_nucleus(first_frame)
    centroids_first_frame, labeled_mask_first_frame = extract_centroids(segmented_mask_first_frame)
    
    # Initialize tracking data with cells from the first frame
    labels_to_track = np.unique(labeled_mask_first_frame)
    labels_to_track = labels_to_track[labels_to_track != 0]  # Exclude background
    tracking_data = {label: [centroids_first_frame[label]] for label in labels_to_track}

    # Process subsequent frames
    for time_index in range(1, image_stack.shape[0]):
        nucleus_channel = image_stack[time_index, 1]  # Adjust channel index if needed
        segmented_mask = segment_nucleus(nucleus_channel)
        current_centroids, _ = extract_centroids(segmented_mask)
        
        # Match centroids to cells from the first frame
        for label_to_track in labels_to_track:
            previous_centroid = tracking_data[label_to_track][-1]
            if previous_centroid is not None and current_centroids:
                # Find the closest centroid in the current frame
                distances = {label: distance.euclidean(previous_centroid, centroid) for label, centroid in current_centroids.items()}
                nearest_label = min(distances, key=distances.get)
                nearest_distance = distances[nearest_label]
                
                # Set a distance threshold to avoid incorrect matches
                distance_threshold = 20  # Adjust based on your data
                if nearest_distance <= distance_threshold:
                    tracking_data[label_to_track].append(current_centroids[nearest_label])
                    del current_centroids[nearest_label]  # Remove matched centroid
                else:
                    tracking_data[label_to_track].append(None)  # Cell lost
            else:
                tracking_data[label_to_track].append(None)  # Cell lost

    return tracking_data
#Visualize tracked centroids
def visualize_tracked_centroids(image_stack, tracking_data):
    for time_index in range(image_stack.shape[0]):
        plt.figure(figsize=(8, 8))
        plt.title(f"Tracked Centroids: Time Point {time_index}")
        plt.imshow(image_stack[time_index, 1], cmap="gray")  # Adjust channel index if needed
        for label, centroids in tracking_data.items():
            if centroids[time_index] is not None:
                plt.plot(centroids[time_index][1], centroids[time_index][0], 'ro')  # Red dots for centroids
                plt.text(centroids[time_index][1], centroids[time_index][0], str(label), color="yellow")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(1)  # Pause to show each frame

# Directory containing image stacks
input_folder = "/Volumes/sils-mc/13776452/Python_scripts/Data_input_test"  # Replace with your folder path

# Loop through each image stack in the folder
# for file_path in glob(os.path.join(input_folder, "*.tif")):  # Adjust file extension if needed
#     image_stack = tiff.imread(file_path)
#     print(f"Processing file: {file_path}")

#     tracking_data = segment_and_extract_centroids(image_stack)
#     print(f"Tracking data for {file_path}:", tracking_data)

   
