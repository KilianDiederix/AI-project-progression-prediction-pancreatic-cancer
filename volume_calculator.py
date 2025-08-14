import os
import nibabel as nib
import numpy as np

data_dir      = r"C:\...\test_set_nifty_combined" # to directory outcome of nifty_mass_combiner.py (whatever the output is set to)
output_file   = "registered_volume_alignment2.0.txt"
tumor_label   = 10  # change here if some files really do use 9

results = []

for fname in sorted(os.listdir(data_dir)):
    if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
        continue

    img  = nib.load(os.path.join(data_dir, fname))
    data = img.get_fdata()

    # round off any tiny float errors and cast to int
    data_int = np.rint(data).astype(np.int32)
    voxel_vol = np.prod(img.header.get_zooms())

    # check label presence
    labels = np.unique(data_int)
    if tumor_label not in labels:
        print(f"⚠️  {fname}: no voxels with label {tumor_label} (found {labels})")
        vol_mm3 = 0.0
    else:
        n_vox   = np.sum(data_int == tumor_label)
        vol_mm3 = n_vox * voxel_vol

    results.append(f"{fname} {vol_mm3:.2f}")

# write output
with open(output_file, "w") as f:
    f.write("Filename Volume (mm3)\n")
    f.write("\n".join(results))

# preview
print("Filename Volume (mm3)")
print("\n".join(results))
