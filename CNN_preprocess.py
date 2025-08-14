# preprocessing.py  (nested Scans/ & Masks/ folder layout)
"""
Pancreatic-cancer CT preprocessing script
========================================

Folder layout
-------------
  Scans/<PID>/<PID>-..._preprocessed.nii.gz
  Masks/<PID>/<PID>-..._segmentation.nii.gz

For every patient:
1. Load the scan and tumour mask.
2. Resample both volumes to **1 mm³** isotropic spacing.
3. Take the tumour bounding-box, enlarge it by --margin mm.
4. Crop, then pad / resize to a fixed cube (default 128x3 voxels).
5. Z-score-normalise and save   Cubes/<PID>.npz   with keys:
     • image – float16  (1, D, H, W)
     • mask  – uint8    (same shape, 0/1)

Example
-------
python CNN_preprocess.py \
  --scan_root "/..." \
  --mask_root "/..." \
  --out_root  "/..." \
  --cube 128 --margin 15
"""
from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_nii(path: Path):
    img = nib.load(str(path))
    print(f"{path}: shape {img.shape}, affine: {img.affine}")
    data = img.get_fdata(dtype=np.float32)
    data = np.clip(data, -100, 400)   # Clip after converting to numpy!
    spacing = img.header.get_zooms()[:3]
    return data, spacing



def save_npz(out_path: Path, image: np.ndarray, mask: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving: {out_path}")
    np.savez_compressed(out_path,
                        image=image.astype(np.float16),
                        mask=mask.astype(np.uint8))


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def resample(vol: np.ndarray, spacing, new=(1.0, 1.0, 1.0), order=3):
    factors = np.array(spacing) / np.array(new)
    return zoom(vol, zoom=factors, order=order)


def tumour_bbox(mask: np.ndarray, margin_vox: int):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise RuntimeError("Tumour mask is empty – check input.")
    zmin, ymin, xmin = coords.min(0)
    zmax, ymax, xmax = coords.max(0)
    return (max(zmin - margin_vox, 0), min(zmax + margin_vox + 1, mask.shape[0]),
            max(ymin - margin_vox, 0), min(ymax + margin_vox + 1, mask.shape[1]),
            max(xmin - margin_vox, 0), min(xmax + margin_vox + 1, mask.shape[2]))


def fit_cube(vol: np.ndarray, target=(128, 128, 128), order=3):
    cur = vol.shape
    if all(c <= t for c, t in zip(cur, target)):
        # central pad
        out = np.zeros(target, dtype=vol.dtype)
        starts = [(t - c) // 2 for c, t in zip(cur, target)]
        slices = tuple(slice(s, s + c) for s, c in zip(starts, cur))
        out[slices] = vol
        return out
    # isotropic resize
    factors = [t / c for c, t in zip(cur, target)]
    return zoom(vol, zoom=factors, order=order)


# --------------------------------------------------------------------------
# Per-patient routine
# --------------------------------------------------------------------------
def preprocess(scan_path: Path, mask_path: Path, out_path: Path,
               cube: int, margin: int):
    img, spacing = load_nii(scan_path)
    msk, _      = load_nii(mask_path)

    # 1 mm³ isotropic
    img = resample(img, spacing, new=(1, 1, 1), order=3)
    msk = resample(msk, spacing, new=(1, 1, 1), order=0)

    # tumour-centred crop
    z0, z1, y0, y1, x0, x1 = tumour_bbox(msk, margin)
    img = img[z0:z1, y0:y1, x0:x1]
    msk = msk[z0:z1, y0:y1, x0:x1]

    # pad / resize to cube³
    img = fit_cube(img, (cube, cube, cube), order=3)
    msk = fit_cube(msk, (cube, cube, cube), order=0)
        # ---- SAVE PROCESSED (CROPPED & PADDED) VOLUMES AS NIFTI FOR QC ----
    out_scan_nii = out_path.parent / (out_path.stem + "_cubeSCAN.nii.gz")
    out_mask_nii = out_path.parent / (out_path.stem + "_cubeMASK.nii.gz")
    # Using identity affine since spacing is now (1,1,1)
    nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)), str(out_scan_nii))
    nib.save(nib.Nifti1Image(msk.astype(np.uint8), np.eye(4)), str(out_mask_nii))
    print(f"Saved NIfTI QC: {out_scan_nii} and {out_mask_nii}")

    # ---- >>>>> SAVE COPIES FOR R001 <<<<< ----
    # if "R001" in scan_path.parts or "R001" in str(scan_path):
    #     # Save as NIfTI with identity affine (since resampled)
    #     out_scan_nii = out_path.parent / (out_path.stem + "_debugSCAN.nii.gz")
    #     out_mask_nii = out_path.parent / (out_path.stem + "_debugMASK.nii.gz")
    #     # Using identity affine since spacing is now (1,1,1)
    #     nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)), str(out_scan_nii))
    #     nib.save(nib.Nifti1Image(msk.astype(np.uint8), np.eye(4)), str(out_mask_nii))
    #     print(f"Saved debug NIfTI for R001: {out_scan_nii} and {out_mask_nii}")


    img = img.astype(np.float32)
    mu, sigma = img.mean(), img.std()
    print(f"Before norm: mean={mu}, std={sigma}")
    if sigma < 1e-4:
        print(f"Warning: Very low std ({sigma}) for {out_path.name} — skipping normalization.")
    else:
        img = (img - mu) / (sigma + 1e-6)
    print(f"After norm: mean={img.mean()}, std={img.std()}")
    if not np.isfinite(img).all():
        print(f"!!! Non-finite values in {out_path.name} !!!")
    save_npz(out_path, img, msk)




# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_root", type=Path, required=True,
                    help="Root folder containing Scans/<PID>/ sub-dirs")
    ap.add_argument("--mask_root", type=Path, required=True,
                    help="Root folder containing Masks/<PID>/ sub-dirs")
    ap.add_argument("--out_root",  type=Path, required=True,
                    help="Output folder for Cubes/*.npz")
    ap.add_argument("--cube",   type=int, default=128,
                    help="Edge length of the output cube (voxels)")
    ap.add_argument("--margin", type=int, default=15,
                    help="Padding beyond tumour bounding-box in mm")
    args = ap.parse_args()

    patients = sorted([p.name for p in args.scan_root.iterdir() if p.is_dir()])
    for pid in tqdm(patients, desc="Patients"):
        scan_folder = args.scan_root / pid
        mask_folder = args.mask_root / pid

        scan_files = sorted(scan_folder.glob("*_preprocessed.nii*"))
        if not scan_files:
            tqdm.write(f"Skip {pid}: no scan files found")
            continue

        for scan_path in scan_files:
            scan_name = scan_path.name
            # Replace only the FIRST "_preprocessed" with "_segmentation"
            mask_name = scan_name.replace("_preprocessed", "_segmentation", 1)
            mask_path = mask_folder / mask_name

            if not mask_path.exists():
                tqdm.write(f"Skip {pid}: mask not found for {scan_name}")
                continue

            # e.g., Cubes/R001_myseries.npz (drop file extension)
            out_base = scan_name.rsplit('.', 2)[0]  # Removes .nii.gz or .nii
            out_path = args.out_root / f"{pid}_{out_base}.npz"

            try:
                preprocess(scan_path, mask_path, out_path,
                           cube=args.cube, margin=args.margin)
            except Exception as e:
                tqdm.write(f"Error {pid}/{scan_name}: {e}")



if __name__ == "__main__":
    main()
