# Pancreatic Cancer ML & Imaging Analysis Toolkit

This repository contains a collection of scripts and notebooks for processing CT scan data, preparing tabular datasets, training machine learning models, and running statistical comparisons for pancreatic cancer research.  
The project combines **deep learning on imaging**, **classical ML on tabular features**, and **statistical validation tools** into a single workflow.

---

## Project Overview

The workflow can be broadly split into three areas:

1. **Imaging Pipeline (CNN-based)**  
   - Preprocessing and organizing raw CT/MRI scan data  
   - Fixing orientation/rotation issues  
   - Building sequence models for temporal imaging data  
   - Checking DICOM/NIfTI integrity and intensity ranges

2. **Tabular Data ML Models**  
   - Logistic Regression, Random Forest, XGBoost pipelines  
   - Custom Wilcoxon + correlation-based feature selector  
   - Nested cross-validation with random search  
   - SHAP explainability  
   - Statistical tests to compare models/datasets

3. **Utilities & Plotting**  
   - Mass combination of NIfTI segmentations  
   - Volume calculation from segmentations  
   - Longitudinal tumor volume plots  
   - Baseline volume doubling time calculations

---

## File-by-File Summary

### Imaging / CNN
- **`CNN_FIX_rotation.py`**  
  Corrects scan orientation/rotation inconsistencies before preprocessing.
  
- **`CNN_preprocess.py`**  
  Main preprocessing script for CNN inputs (resizing, normalization, cropping).
  
- **`CNN_sequence_34.py`**  
  Model training script for CNN sequence model using MedicalNet (Resnet34 backbone).
  
- **`ct_check.py`**  
  Quick validation tool for CT datasets — checks file counts, folder structure, and slice consistency.
  
- **`HU_checker.py`**  
  Histogram/voxel intensity checker for ensuring correct HU (Hounsfield Unit) calibration.
  
- **`inference.py`**  
  Gets segmentations for all scans based on Bereska et al. segmentation model. (see script for comments on how to use).

- **`preprocess.py`**  
  HU clipping and normalization for all CT scans, needed before running inference.py.
  
---

### Imaging Utilities
- **`nifty_mass_combiner.py`**  
  Combines multiple NIfTI segmentations into a single file for batch evaluation, Bereska model has 2 output .nii.gz files, one for the organs, another for the vessels. If only one of the models is used, this file is not needed.
  
- **`slicer_renamer.py`**  
  This script can be copied into Slicer3D to rename all segmentation slices according to Bereska plans.json files.

---

### Plotting & Visualization
- **`plot_progressive.py`**  
  Generates longitudinal plots showing tumor volume progression over time for each patient.

---

### Tabular ML & Statistics
- **`tabular_all_features.py`**  
  ML training script (LR, RF, XGB) on all available tabular features, with nested CV and random search.
  
- **`tabular_no_extracted.py`**  
  Same as above, but excluding extracted radiomics/segmentation-based features.
  
- **`shap.ipynb`**  
  Jupyter notebook for SHAP value computation and visualizations on trained tabular models.
  
- **`significance_testing.py`**  
  Runs repeated CV to compare two specific pipelines (paired t-test), e.g. full vs. base dataset.

---

### Tumor Volume & Growth
- **`VDT_baseline.py`**  
  Calculates baseline and follow-up Volume Doubling Time (VDT) from segmentation data.
  
- **`volume_calculator.py`**  
  Converts NIfTI segmentations into absolute tumor volumes (cm³).


---

### Requirements
- **`requirements.txt`**  
  Python package list to replicate the environment used for this project.

---

## Getting Started

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
