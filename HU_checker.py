""" 
HU_checker.py

Checks volume shape, voxel spacing and HU value range per nifty file. 
Add path to nifty scan below in `nii_path`  
"""

import nibabel as nib
import numpy as np

# Load the NIfTI file
nii_path = r"C:\..." #path to .nii.gz (individual scan)
img = nib.load(nii_path)
data = img.get_fdata()  # 3D numpy array

# Get voxel spacing (resolution)
# Assumes the affine matrix is in RAS+ coordinate convention
affine = img.affine
voxel_spacing = np.abs(affine[:3, :3].diagonal())  # voxel size in mm (x, y, z)

# Get HU range
hu_min = np.min(data)
hu_max = np.max(data)

# Output
print(f"Volume shape: {data.shape}")
print(f"Voxel spacing (x, y, z in mm): {voxel_spacing}")
print(f"HU value range: [{hu_min}, {hu_max}]")
