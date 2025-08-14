# This is based on comparing to baseline, so first Pre to baseline and baseline to last FU. Intstead of comparing first pre treatment to baseline and first FU to last FU. 

import pandas as pd
import re
import numpy as np
from math import log
import matplotlib.pyplot as plt

# === Paths ===
input_path = r"C:\...\registered_volume_alignment2.0.txt" # to volume (see volume_calculator.py)
output_path = r"C:\...\vdt_results_baseline.txt"

# === Step 1: Load and parse data ===
with open(input_path, "r") as f:
    lines = f.readlines()

records = []

for line in lines[1:]:  # Skip header
    parts = line.strip().split()
    if len(parts) != 2:
        continue

    filename, volume = parts[0], float(parts[1])

    # Extract patient id
    patient_match = re.match(r"(R\d+)", filename)
    if not patient_match:
        continue
    patient_id = patient_match.group(1)

    # Extract months using a case-insensitive regex (matches m or M)
    month_match = re.search(r"(\d{1,2})m", filename, flags=re.IGNORECASE)
    months = int(month_match.group(1)) if month_match else 0

    # Determine phase using a normalized (lowercase) filename
    lower_filename = filename.lower()
    if "fu" in lower_filename or "post" in lower_filename:
        phase = "post"
    else:
        phase = "pre"

    # Scan type (art/ven/nonc)
    scan_type_match = re.search(r"-(ven|art|nonc)_segmentation\.nii\.gz", filename, flags=re.IGNORECASE)
    scan_type = scan_type_match.group(1).lower() if scan_type_match else "nonc"
    scan_label = f"{months}m-{scan_type}"

    records.append({
        "patient_id": patient_id,
        "filename": filename,
        "volume": volume,
        "months": months,
        "phase": phase,
        "scan_label": scan_label,
        "scan_type": scan_type
    })

df = pd.DataFrame(records)

# === Step 2: Average volume if multiple scans in the same month, phase ===
df_avg = df.groupby(["patient_id", "phase", "months"]).agg({
    "volume": "mean",
    "scan_label": lambda x: ",".join(sorted(set(x))),
    "filename": "first",
    "scan_type": "first"
}).reset_index()

# === Step 3: VDT Calculation as described ===
def get_vdt_custom(patient_df):
    # --- Find baseline: prefer nonc, else earliest pre ---
    pre_scans = patient_df[(patient_df['phase'] == 'pre')]
    baseline = None
    nonc_scan = pre_scans[pre_scans['scan_type'] == 'nonc']
    if not nonc_scan.empty:
        baseline = nonc_scan.sort_values('months').iloc[0]
    elif not pre_scans.empty:
        baseline = pre_scans.sort_values('months').iloc[0]
    else:
        baseline = None

    # --- Pre VDT: earliest pre (not baseline) to baseline ---
    vdt_pre, scan_pair_pre = np.nan, None
    if baseline is not None and len(pre_scans) > 1:
        # Exclude baseline from earliest pre if needed
        pre_others = pre_scans[pre_scans['filename'] != baseline['filename']]
        if not pre_others.empty:
            earliest_pre = pre_others.sort_values('months').iloc[0]
            pre_first, pre_last = earliest_pre, baseline
            delta_t_pre = abs(pre_last['months'] - pre_first['months']) * 30
            ratio_pre = pre_last['volume'] / pre_first['volume'] if pre_first['volume'] != 0 else np.nan
            scan_pair_pre = f"{pre_first['scan_label']} → {pre_last['scan_label']}"
            if ratio_pre > 0 and ratio_pre != 1 and delta_t_pre > 0:
                vdt_pre = round(delta_t_pre * log(2) / log(ratio_pre), 2)
    # --- Post VDT: baseline to last post (FU) ---
    post_scans = patient_df[(patient_df['phase'] == 'post')]
    vdt_post, scan_pair_post = np.nan, None
    if baseline is not None and not post_scans.empty:
        last_post = post_scans.sort_values('months').iloc[-1]
        delta_t_post = abs(last_post['months'] - baseline['months']) * 30
        ratio_post = last_post['volume'] / baseline['volume'] if baseline['volume'] != 0 else np.nan
        scan_pair_post = f"{baseline['scan_label']} → {last_post['scan_label']}"
        if ratio_post > 0 and ratio_post != 1 and delta_t_post > 0:
            vdt_post = round(delta_t_post * log(2) / log(ratio_post), 2)
    return pd.Series({
        'vdt_pre_days': vdt_pre,
        'scan_pair_pre': scan_pair_pre,
        'vdt_post_days': vdt_post,
        'scan_pair_post': scan_pair_post
    })

results = df_avg.groupby("patient_id").apply(get_vdt_custom).reset_index()

# === Step 4: Save results ===
results.to_csv(output_path, sep="\t", index=False)
print(f" VDT results saved to: {output_path}")

# === Optional: Pivot for plotting ===
df_pivot = results.set_index('patient_id')[['vdt_pre_days', 'vdt_post_days']]

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df_pivot.index))
width = 0.35

bars_pre = ax.bar(x - width/2, df_pivot['vdt_pre_days'], width, label='Pre')
bars_post = ax.bar(x + width/2, df_pivot['vdt_post_days'], width, label='Post')

ax.set_xlabel("Patient ID")
ax.set_ylabel("VDT (days)")
ax.set_title("Volume Doubling Time (VDT) by Patient (Pre and Post)")
ax.set_xticks(x)
ax.set_xticklabels(df_pivot.index, rotation=90)
ax.legend()
ax.set_ylim(-1000, 1500)

def annotate_bars_all(bars):
    for bar in bars:
        height = bar.get_height()
        y = height
        if height > 1500:
            y = 1500
            text = f'{height:.0f} ↑'
        elif height < -1000:
            y = -1000
            text = f'{height:.0f} ↓'
        else:
            text = f'{height:.2f}'
        ax.annotate(text,
                    xy=(bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

annotate_bars_all(bars_pre)
annotate_bars_all(bars_post)
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.show()
