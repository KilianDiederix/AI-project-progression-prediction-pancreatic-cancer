import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import os

# === 1) Load the two-column table (Filename + Volume) ===
records = []
with open(r"C:\....\registered_volume_alignment2.0.txt", "r") as f: # to volume calculations (see volume_calculator.py)
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        filename = parts[0]
        try:
            volume = float(parts[1])
        except ValueError:
            continue
        records.append((filename, volume))

df = pd.DataFrame(records, columns=["Filename", "Volume"])

# === 2) Parse PatientID, Months, Phase from filename ===
def parse_row(filename):
    pid_match = re.match(r"(R\d+)", filename)
    patient_id = pid_match.group(1) if pid_match else None

    month_match = re.search(r"(\d{1,2})[mM]", filename)
    months = int(month_match.group(1)) if month_match else 0

    lower = filename.lower()
    if "fu" in lower:
        phase = "post"
    elif "pre" in lower:
        phase = "pre"
    elif months == 0:
        phase = "baseline"
    else:
        phase = "pre"
    return patient_id, months, phase

parsed = df["Filename"].apply(parse_row)
df[["PatientID", "Months", "Phase"]] = pd.DataFrame(parsed.tolist(), index=df.index)

# === 3) Compute relative x-axis (months from treatment) ===
def compute_x(row):
    if row["Phase"] == "baseline":
        return 0
    elif row["Phase"] == "pre":
        return -row["Months"]
    else:
        return row["Months"]

df["X_rel"] = df.apply(compute_x, axis=1)

# === 4) Load VDT data ===
vdt_df = pd.read_csv(r"C:\...\vdt_results_baseline.txt", sep="\t") # to vdt calculations (see VDT_baseline.py)
vdt_df.set_index("patient_id", inplace=True)

# === 5) Create output directory ===
output_dir = "plots_progressive"
os.makedirs(output_dir, exist_ok=True)

# === 6) Loop over each patient and plot ===
for pid, group in df.groupby("PatientID"):
    baseline = group[group["Phase"] == "baseline"]
    pre_treatment = group[group["Phase"] == "pre"]
    post_treatment = group[group["Phase"] == "post"]

    plt.figure(figsize=(6, 4))

    # Plot all points as black 'x'
    plt.scatter(group["X_rel"], group["Volume"], color="black", marker="x", label="Measured")

    y0_pre = None  # for fallback if needed

    # Pre-treatment trendline (blue)
    if not pre_treatment.empty:
        coeffs_pre = np.polyfit(pre_treatment["X_rel"], pre_treatment["Volume"], deg=1)
        x_pre_line = np.linspace(pre_treatment["X_rel"].min(), 0, 50)
        y_pre_line = coeffs_pre[0] * x_pre_line + coeffs_pre[1]
        y0_pre = coeffs_pre[0] * 0 + coeffs_pre[1]
        plt.plot(x_pre_line, y_pre_line, color="#1f77b4", linestyle="--", label="Pre‐treatment")

    # Post-treatment trendline (green)
    if not post_treatment.empty:
        x_post = post_treatment["X_rel"].values
        y_post = post_treatment["Volume"].values

        if np.max(x_post) == 0 or len(x_post) == 0 or np.all(np.isnan(y_post)):
            pass
        else:
            x_post_line = np.linspace(0, np.max(x_post), 50)

            if len(post_treatment) >= 2:
                coeffs_post = np.polyfit(x_post, y_post, deg=1)
                y_post_line = coeffs_post[0] * x_post_line + coeffs_post[1]
                plt.plot(x_post_line, y_post_line, color="#2ca02c", linestyle="--", label="Post‐treatment")
            else:
                x1 = x_post[0]
                y1 = y_post[0]

                if not baseline.empty:
                    x0 = 0
                    y0 = baseline["Volume"].iloc[0]
                elif y0_pre is not None:
                    x0 = 0
                    y0 = y0_pre
                else:
                    x0 = 0
                    y0 = y1

                if x1 == x0:
                    x_post_line = np.array([x0, x0 + 1])
                    y_post_line = np.array([y0, y1])
                else:
                    x_post_line = np.linspace(x0, x1, 50)
                    y_post_line = np.linspace(y0, y1, 50)

                plt.plot(x_post_line, y_post_line, color="#2ca02c", linestyle="--", label="Post‐treatment")

    # === VDT Annotation ===
    vdt_text_left = ""
    vdt_text_right = ""
    if pid in vdt_df.index:
        row = vdt_df.loc[pid]

        if pd.notna(row["vdt_pre_days"]):
            vdt_months = round(row["vdt_pre_days"] / 30, 2)
            vdt_text_left = f"VDT Pre: {vdt_months} mo"

        if pd.notna(row["vdt_post_days"]):
            vdt_months = round(row["vdt_post_days"] / 30, 2)
            vdt_text_right = f"VDT Post: {vdt_months} mo"

    if vdt_text_left:
        plt.text(0.01, 0.97, vdt_text_left, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    if vdt_text_right:
        plt.text(0.99, 0.97, vdt_text_right, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Plot formatting
    plt.title(f"Volume Trend for Patient {pid}")
    plt.xlabel("Months Relative to Treatment Start (0)")
    plt.ylabel("Volume (mm³)")
    plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f"{pid}_volume_trend.png")
    plt.savefig(plot_path)
    plt.close()
