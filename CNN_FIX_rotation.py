import os
from pathlib import Path
import nibabel as nib
from nibabel.orientations import aff2axcodes

def canonicalize_folder(root):
    nii_files = list(Path(root).rglob("*.nii*"))
    print(f"Found {len(nii_files)} files in {root}")
    for file_path in nii_files:
        img = nib.load(str(file_path))
        orig_ax = aff2axcodes(img.affine)
        if orig_ax == ('R', 'A', 'S'):
            print(f"{file_path} already RAS.")
            continue
        img_ras = nib.as_closest_canonical(img)
        new_ax = aff2axcodes(img_ras.affine)
        print(f"{file_path.name}: {orig_ax} -> {new_ax}")
        nib.save(img_ras, str(file_path))

# === MAIN ===
scans_root = Path("/...")
masks_root = Path("/...")

print("Canonicalizing scans...")
canonicalize_folder(scans_root)

print("Canonicalizing masks...")
canonicalize_folder(masks_root)

print("Done. All images now RAS (MedicalNet standard).")
