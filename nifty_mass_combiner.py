"""
===============================================================
 Script: Combine Organ + Vessel NIfTI Segmentations
===============================================================

Description
-----------
Takes per-case organ and vessel segmentation volumes and merges them
into a single label map. Vessel labels are offset so they don’t collide
with organ labels (i.e., new_label = max_organ_label + vessel_label).

Expected inputs
---------------
- Two folders with per-case NIfTI files (gzipped .nii.gz):
  1) Organ segmentations (e.g., "..._fullres.nii.gz")
  2) Vessel segmentations (matching filenames but with a vessel suffix,
     e.g., "..._vessel_fullres.nii.gz")

How it works
------------
1) Enumerates all organ files ending with "_fullres.nii.gz".
2) Locates the corresponding vessel file by replacing the suffix with
   "_vessel_fullres.nii.gz".
3) Loads both volumes, checks shape (and warns if affines differ).
4) Computes max organ label, then assigns vessel labels starting at
   (max_organ_label + 1), preserving background=0.
5) Writes merged volume to `output_dir` using a canonical filename:
   "<base_name>_segmentation.nii.gz" (base_name is the organ filename
   without the "_fullres" or "_vessel_fullres" part).

What you need to fill in
------------------------
- organ_seg_dir  : path to the organ segmentation folder
- vessel_seg_dir : path to the vessel segmentation folder
- output_dir     : path where combined segmentations will be written

Notes
-----
- Array dtype is set to uint16 to avoid overflow if labels > 255.
- If your file naming differs, adapt the suffix logic where indicated.
- Affine/header are taken from the organ file.
===============================================================
"""

import os
import nibabel as nib
import numpy as np

# -------------------------------
# 1) Define paths (EDIT THESE)
# -------------------------------
organ_seg_dir = "/path/to/nnUNet_results/Dataset102/pred_fullres"      # <-- folder with organ *_fullres.nii.gz
vessel_seg_dir = "/path/to/nnUNet_results/Dataset103/pred_fullres"     # <-- folder with vessel *_vessel_fullres.nii.gz
output_dir    = "/path/to/output/combined_segmentations"               # <-- destination for merged segmentations

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 2) Discover organ files
#    (Adapt the suffix if your naming differs)
# -------------------------------
organ_files = [f for f in os.listdir(organ_seg_dir) if f.endswith("_fullres.nii.gz")]

# -------------------------------
# 3) Process each case
# -------------------------------
for organ_file in sorted(organ_files):
    organ_path = os.path.join(organ_seg_dir, organ_file)

    # Build matching vessel filename by suffix replacement
    # If your pattern differs, adjust here:
    vessel_file = organ_file.replace("_fullres.nii.gz", "_vessel_fullres.nii.gz")
    vessel_path = os.path.join(vessel_seg_dir, vessel_file)

    if not os.path.exists(vessel_path):
        print(f"Skipping {organ_file}: No matching vessel file found at {vessel_path}.")
        continue

    # Load volumes
    organ_img = nib.load(organ_path)
    vessel_img = nib.load(vessel_path)

    # Convert to arrays (use uint16 to safely hold shifted labels)
    organ_data = organ_img.get_fdata().astype(np.uint16)
    vessel_data = vessel_img.get_fdata().astype(np.uint16)

    # Basic checks
    if organ_data.shape != vessel_data.shape:
        print(f"Skipping {organ_file}: Shape mismatch {organ_data.shape} vs {vessel_data.shape}.")
        continue

    # Optional: warn if affines differ (we still proceed, using organ affine/header)
    if not np.allclose(organ_img.affine, vessel_img.affine):
        print(f"Warning: Affines differ for {organ_file}. Proceeding with organ affine.")

    # Combine: offset vessel labels above current organ max
    combined = organ_data.copy()
    max_organ_label = int(organ_data.max())

    # Vectorized remap: for each non-zero vessel label, set to max_organ_label + label
    # (If you have a label map to specific targets, apply it here instead.)
    vessel_labels = np.unique(vessel_data)
    vessel_labels = vessel_labels[vessel_labels > 0]  # ignore background

    for vl in vessel_labels:
        combined[vessel_data == vl] = max_organ_label + vl

    # Build output filename:
    # Remove "_fullres" (and any lingering "_vessel" if present), then append "_segmentation.nii.gz"
    base_name = organ_file.replace("_fullres.nii.gz", "")
    base_name = base_name.replace("_vessel", "")
    output_filename = f"{base_name}_segmentation.nii.gz"
    output_path = os.path.join(output_dir, output_filename)

    # Save using organ's affine/header
    combined_img = nib.Nifti1Image(combined, affine=organ_img.affine, header=organ_img.header)
    nib.save(combined_img, output_path)

    print(f"Combined segmentation saved at: {output_path}")
