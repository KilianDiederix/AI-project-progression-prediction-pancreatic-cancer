#!/usr/bin/env python3
"""
Script for QC (metadata & HU checks) of CT scans using SimpleITK.

Modify the `scan_dir` variable below to point directly at your data.

This will:
 1. Print out spacing, dimensions, orientation, origin, and HU range for each scan.
"""
import SimpleITK as sitk
import os
import glob
import numpy as np


scan_dir = r"/..."  # directory with all CT scans


def check_scans(scan_dir):
    """
    Print metadata and intensity stats for each NIfTI in `scan_dir`.
    """
    files = glob.glob(os.path.join(scan_dir, '*.nii')) + glob.glob(os.path.join(scan_dir, '*.nii.gz'))
    if not files:
        print(f"No NIfTI files found in {scan_dir}")
        return
    print(f"QC Report for scans in: {scan_dir}\n")
    for filepath in sorted(files):
        img = sitk.ReadImage(filepath)
        arr = sitk.GetArrayFromImage(img)
        print(f"File: {os.path.basename(filepath)}")
        print(f"  - Dimensions (voxels): {img.GetSize()}")
        print(f"  - Spacing (mm): {img.GetSpacing()}")
        print(f"  - Orientation (direction cosines): {img.GetDirection()}")
        print(f"  - Origin (mm): {img.GetOrigin()}")
        print(f"  - Intensity (HU) min/mean/max: {arr.min():.1f}/{arr.mean():.1f}/{arr.max():.1f}\n")
        print(f"  - Shape (z, y, x): {arr.shape}")


if __name__ == "__main__":
    check_scans(scan_dir)
