"""
===============================================================
 Script: Resample + Intensity-Clip CT NIfTI Scans (HU preserved)
===============================================================

Description
-----------
Batch-preprocesses 3D CT volumes (.nii / .nii.gz) in patient subfolders:
1) Resamples each scan to isotropic spacing (default 1.0 mm).
2) Clips intensities to a HU window (default [-100, 400]) WITHOUT scaling.
3) Saves the result next to the original with the suffix "_preprocessed.nii.gz".

How it works
------------
- Looks in `BASE_FOLDER`, which must contain per-patient subfolders.
- For each .nii or .nii.gz in a patient folder:
  a) Read with SimpleITK
  b) Resample to `NEW_SPACING` with linear interpolation
  c) Clip to [LOWER_BOUND, UPPER_BOUND] HU (no normalization to [0,1])
  d) Save as "<original_name>_preprocessed.nii.gz"

What you need to fill in
------------------------
- BASE_FOLDER   : path to the root data folder containing patient subfolders
- (Optional) NEW_SPACING, LOWER_BOUND, UPPER_BOUND to match your pipeline

Notes
-----
- File names are preserved except for appending the "_preprocessed.nii.gz" suffix.
- Spatial metadata (origin, direction, spacing) is preserved; spacing becomes NEW_SPACING.
- This script processes scans (not label maps). For segmentations, use nearest-neighbor.
===============================================================
"""

import os
import SimpleITK as sitk
import numpy as np



# Config (EDIT THESE if needed)
BASE_FOLDER   = "/path/to/DATA_ALL"   # <-- Folder with patient subfolders containing .nii/.nii.gz
NEW_SPACING   = [1.0, 1.0, 1.0]       # <-- Output isotropic spacing (mm)
LOWER_BOUND   = -100                  # <-- HU lower clip
UPPER_BOUND   = 400                   # <-- HU upper clip
OUTPUT_SUFFIX = "_preprocessed.nii.gz"

def resample_image(image, new_spacing=NEW_SPACING):
    """
    Resample a 3D image to isotropic spacing (linear interpolation).
    Preserves origin and direction; updates spacing and size accordingly.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image)

def intensity_normalization(image, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND):
    """
    Clip intensities to [lower_bound, upper_bound] in HU without scaling.
    """
    arr = sitk.GetArrayFromImage(image)
    arr = np.clip(arr, lower_bound, upper_bound)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)  # keep spacing/origin/direction
    return out


# Main loop
for patient in os.listdir(BASE_FOLDER):
    patient_folder = os.path.join(BASE_FOLDER, patient)
    if not os.path.isdir(patient_folder):
        continue

    print(f"\nProcessing patient: {patient}")

    for file in os.listdir(patient_folder):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            file_path = os.path.join(patient_folder, file)
            print(f"  ➤ Found scan: {file}")

            # Load image
            image = sitk.ReadImage(file_path)
            spacing = image.GetSpacing()
            print(f"    Original spacing: {spacing}")

            # Resample and clip
            resampled  = resample_image(image, new_spacing=NEW_SPACING)
            normalized = intensity_normalization(resampled, LOWER_BOUND, UPPER_BOUND)

            # Build output name (preserve original base name; just append suffix)
            if file.endswith(".nii.gz"):
                output_name = file.replace(".nii.gz", OUTPUT_SUFFIX)
            elif file.endswith(".nii"):
                output_name = file.replace(".nii", OUTPUT_SUFFIX)
            else:
                continue

            output_path = os.path.join(patient_folder, output_name)
            sitk.WriteImage(normalized, output_path)
            print(f"✅ Saved: {output_name}")
