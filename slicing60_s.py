import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Define paths to the folders
ct_scan_folder = "C:/s7/rib nii files"  # Folder containing CT scans
label_folder = "C:/s7/ct_labels"  # Folder containing label files
output_folder = "C:/s7 60 rework/output_slices(sagittal)"  # Base folder where all slices will be saved

# Fracture label mappings
fracture_mapping = {
    1: "displaced_rib_fracture",
    2: "non_displaced_rib_fracture",
    3: "buckle_rib_fracture",
    4: "segmental_rib_fracture",
    0: "no_fractures"  # Class for images with no fractures
}

# Create subfolders for each fracture type
for label_name in fracture_mapping.values():
    folder_path = os.path.join(output_folder, label_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Get sorted lists of CT scan and label files
ct_scan_files = sorted(os.listdir(ct_scan_folder))
label_files = sorted(os.listdir(label_folder))

# Traverse through both CT scan and label files
for ct_scan_file, label_file in zip(ct_scan_files, label_files):
    # Full paths to the current CT scan and label files
    ct_scan_path = os.path.join(ct_scan_folder, ct_scan_file)
    label_path = os.path.join(label_folder, label_file)
    print(ct_scan_file)
    
    # Read the CT scan and label images
    ct_scan = sitk.ReadImage(ct_scan_path)
    label_image = sitk.ReadImage(label_path)

    # Convert both images to NumPy arrays
    ct_scan_array = sitk.GetArrayFromImage(ct_scan)
    label_array = sitk.GetArrayFromImage(label_image)

    # Skip the file if it contains only ambiguous labels (-1)
    if -1 in label_array:
        print(f"Skipping file {ct_scan_file} due to presence of label -1.")
        continue

    # Loop through slices along each axis
    #for axis, axis_name in zip(range(3), ["coronal", "sagittal", "axial"]):
    for axis, axis_name in zip(range(3), ["sagittal"]):  
        max_index = label_array.shape[axis]
        for i in range(0, max_index, 1):
            # Get the 2D slice along the current axis
            label_slice = np.take(label_array, i, axis=axis)
            ct_slice = np.take(ct_scan_array, i, axis=axis)

            # Identify unique labels in the slice
            unique_labels = np.unique(label_slice)

            # Determine folder name based on the label
            if np.array_equal(unique_labels, [0]):  # No fractures
                folder_name = "no_fractures"
            else:  # Process fracture labels
                for label_value in unique_labels:
                    if label_value in fracture_mapping:  # Valid label
                        folder_name = fracture_mapping[label_value]
                    else:
                        continue

            # Save slice as image
            output_subfolder = os.path.join(output_folder, folder_name)
            output_filename = f"{os.path.splitext(ct_scan_file)[0]}_{axis_name}_slice_{i}.jpg"
            output_path = os.path.join(output_subfolder, output_filename)
            rotated_slice = np.rot90(ct_slice, k=2)
            plt.imsave(output_path, rotated_slice, cmap='gray', format='jpg', dpi=50)

            print(f"Saved slice {i} to folder {folder_name}.")


print("\n Everything is hopefully done")