"""
===============================================================
 Script: nnUNet Batch Inference Automation
===============================================================

Description:
------------
This script automates running nnUNet inference on a batch of 
preprocessed medical imaging files (e.g., CT scans) organized 
by patient folders. It:

1. Collects preprocessed `.nii.gz` image files from patient folders.
2. Renames and copies them to the nnUNet `imagesTs` folder.
3. Saves a mapping between nnUNet names and original file names.
4. Runs multiple nnUNet models (lowres and fullres) for different datasets.
5. Renames prediction outputs to match original file names.
6. Moves processed images to an archive folder.
7. Logs the entire process with timing information.

How it works:
-------------
- The script assumes each patient has their own folder in `SOURCE_DIR`.
- Files must end with `_resampled.nii.gz` (you can change the pattern if needed).
- nnUNet model predictions are run one image at a time, using a 
  temporary folder (`imagesTs_onecase`).
- Predictions from multiple datasets are renamed immediately for clarity.
- A CSV mapping file is saved in `imagesTs` for reference.

What you need to fill in:
-------------------------
- **LOG_FILE**: Path to where you want the log file saved.
- **SOURCE_DIR**: Path to the folder containing patient subfolders with preprocessed images.
- **IMAGES_TS, IMAGES_TS_ONECASE, PROCESSED_IMAGES**: nnUNet dataset folders for inference.
- **RESULTS_XXX**: Paths to your desired output prediction folders for each model.

Dependencies:
-------------
- Python 3.x
- nnUNet v2 installed and in your PATH
- Required Python packages: os, shutil, subprocess, re, sys, time, csv

Note:
-----
- Update dataset IDs (`Dataset102_PDACStudent`, `Dataset103_PDACvessels`) 
  to match your own nnUNet dataset setup.
- This script processes one image at a time to manage memory and 
  simplify mapping/prediction tracking.
===============================================================
"""

import os
import shutil
import subprocess
import re
import sys
import time
import csv

# ===============================
# 1. Logging Setup
# ===============================
LOG_FILE = "/path/to/log_file.txt"  # <-- Replace with your log file path
sys.stdout = open(LOG_FILE, "w", encoding="utf-8")
sys.stderr = sys.stdout  # redirect errors to log

# ===============================
# 2. Define Paths
# ===============================
# Source directory where preprocessed files reside (organized by patient folders)
SOURCE_DIR = "/path/to/source_data"  # <-- Replace with folder containing patient folders

# nnUNet directories (make sure these match your setup)
IMAGES_TS = "/path/to/nnUNet_raw/DatasetXXX/imagesTs"  # <-- Replace with your nnUNet imagesTs folder
IMAGES_TS_ONECASE = "/path/to/nnUNet_raw/DatasetXXX/imagesTs_onecase"  # <-- Temporary single-case folder
PROCESSED_IMAGES = "/path/to/nnUNet_raw/DatasetXXX/processed_images"  # <-- Where processed images are archived

RESULTS_102_LOWRES = "/path/to/nnUNet_results/Dataset102/pred_lowres"  # <-- Replace dataset ID & path
RESULTS_102_FULLRES = "/path/to/nnUNet_results/Dataset102/pred_fullres"
RESULTS_103_LOWRES = "/path/to/nnUNet_results/Dataset103/pred_lowres"
RESULTS_103_FULLRES = "/path/to/nnUNet_results/Dataset103/pred_fullres"

# Ensure directories exist
for folder in [IMAGES_TS, IMAGES_TS_ONECASE, PROCESSED_IMAGES,
               RESULTS_102_LOWRES, RESULTS_102_FULLRES,
               RESULTS_103_LOWRES, RESULTS_103_FULLRES]:
    os.makedirs(folder, exist_ok=True)

# ===============================
# 3. Build and Save Mapping Dictionary
# ===============================
mapping_dict = {}
print("Collecting and renaming preprocessed files...")

for patient_folder in sorted(os.listdir(SOURCE_DIR)):
    patient_path = os.path.join(SOURCE_DIR, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    m = re.search(r'R(\d+)', patient_folder)
    if not m:
        print(f"Warning: Could not extract patient id from folder {patient_folder}")
        continue
    patient_id = m.group(1).zfill(3)

    files = sorted([f for f in os.listdir(patient_path) if f.endswith("_resampled.nii.gz")])
    for idx, f in enumerate(files):
        index_str = str(idx).zfill(4)
        nnunet_name = f"case_{patient_id}_{index_str}.nii.gz"
        src_file = os.path.join(patient_path, f)
        dst_file = os.path.join(IMAGES_TS, nnunet_name)
        shutil.copy(src_file, dst_file)
        print(f"Moved {src_file} -> {dst_file}")
        desired_final_name = f.replace("_resampled", "")
        mapping_dict[nnunet_name[:-7]] = desired_final_name

mapping_csv = os.path.join(IMAGES_TS, "nnunet_mapping.csv")
with open(mapping_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["nnunet_name_base", "final_name"])
    for key, val in mapping_dict.items():
        writer.writerow([key, val])
print(f"Mapping saved to {mapping_csv}")

# ===============================
# 4. Inference Loop
# ===============================
image_files = sorted([f for f in os.listdir(IMAGES_TS) if f.endswith(".nii.gz")])
timing_info = []
total_start_time = time.time()

for image in image_files:
    case_start_time = time.time()
    print(f"\nProcessing: {image}")
    src = os.path.join(IMAGES_TS, image)
    dst = os.path.join(IMAGES_TS_ONECASE, image)
    shutil.move(src, dst)

    full_base = image[:-7]
    desired_final_name = mapping_dict.get(full_base, "unknown_original.nii.gz")
    patient_base = "case_" + image.split("_")[1]

    # Run Inference
    subprocess.run([
        "nnUNetv2_predict", "-i", IMAGES_TS_ONECASE, "-o", RESULTS_102_LOWRES,
        "-d", "Dataset102_PDACStudent", "-c", "3d_lowres", "-f", "4", "-npp", "16", "-nps", "8"
    ], stdout=sys.stdout, stderr=sys.stderr)

    subprocess.run([
        "nnUNetv2_predict", "-i", IMAGES_TS_ONECASE, "-o", RESULTS_102_FULLRES,
        "-d", "Dataset102_PDACStudent", "-c", "3d_cascade_fullres", "-f", "4",
        "-prev_stage_predictions", RESULTS_102_LOWRES, "-npp", "16", "-nps", "8"
    ], stdout=sys.stdout, stderr=sys.stderr)

    subprocess.run([
        "nnUNetv2_predict", "-i", IMAGES_TS_ONECASE, "-o", RESULTS_103_LOWRES,
        "-d", "Dataset103_PDACvessels", "-c", "3d_lowres", "-f", "4", "-npp", "16", "-nps", "8"
    ], stdout=sys.stdout, stderr=sys.stderr)

    subprocess.run([
        "nnUNetv2_predict", "-i", IMAGES_TS_ONECASE, "-o", RESULTS_103_FULLRES,
        "-d", "Dataset103_PDACvessels", "-c", "3d_cascade_fullres", "-f", "4",
        "-prev_stage_predictions", RESULTS_103_LOWRES, "-npp", "16", "-nps", "8"
    ], stdout=sys.stdout, stderr=sys.stderr)

    # Rename Predictions
    for res_dir, suffix in [
        (RESULTS_102_LOWRES, "_lowres.nii.gz"),
        (RESULTS_102_FULLRES, "_fullres.nii.gz"),
        (RESULTS_103_LOWRES, "_vessel_lowres.nii.gz"),
        (RESULTS_103_FULLRES, "_vessel_fullres.nii.gz")
    ]:
        pred_file = os.path.join(res_dir, f"{patient_base}.nii.gz")
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found in {res_dir}.")
            continue
        new_pred_name = desired_final_name.replace(".nii.gz", suffix)
        new_pred_path = os.path.join(res_dir, new_pred_name)
        counter = 1
        base_new, ext = os.path.splitext(new_pred_name)
        while os.path.exists(new_pred_path):
            new_pred_name = f"{base_new}_{counter}{ext}"
            new_pred_path = os.path.join(res_dir, new_pred_name)
            counter += 1
        os.rename(pred_file, new_pred_path)
        print(f"Renamed {pred_file} -> {new_pred_path}")

    shutil.move(dst, os.path.join(PROCESSED_IMAGES, image))
    case_time = time.time() - case_start_time
    timing_info.append((image, case_time))
    print(f"Finished: {image} (Time: {case_time:.2f} sec)")

# ===============================
# 5. Summary
# ===============================
total_time = time.time() - total_start_time
print("\nProcessing Summary:")
for case, duration in timing_info:
    print(f"{case}: {duration:.2f} seconds")
print(f"\nTotal time: {total_time:.2f} seconds")
print("All images processed successfully!")

sys.stdout.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
print(f"Logging complete! Log saved at: {LOG_FILE}")
