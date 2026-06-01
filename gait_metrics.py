import pandas as pd
import numpy as np
import scipy.signal as signal
import os

SKIP_STEPS = 100
FRAMERATE = 20

variants = [
    ("variantBaseline",      "Baseline"),
    ("variantUpperLegShort", "Upper leg short"),
    ("variantUpperLegLong",  "Upper leg long"),
    ("variantLowerLegShort", "Lower leg short"),
    ("variantLowerLegLong",  "Lower leg long"),
    ("variantBodyLong1",     "Body ext 1"),
    ("variantBodyLong2",     "Body ext 2"),
]

rows = []

for run, label in variants:
    path = f"runs/{run}/data.csv"
    if not os.path.exists(path):
        print(f"Skipping {run} — no data.csv")
        continue

    df = pd.read_csv(path).iloc[SKIP_STEPS:].reset_index(drop=True)
    t = df["t"].values

    # --- stride period via contact onset detection (foot_1) ---
    contact = df["foot_1_contact"].values.astype(int)
    onsets = np.where(np.diff(contact) == 1)[0] + 1  # rising edges
    if len(onsets) < 2:
        print(f"  {label}: fewer than 2 contact onsets detected")
        stride_periods = np.array([np.nan])
    else:
        stride_periods = np.diff(t[onsets])  # seconds between consecutive onsets

    mean_period = np.nanmean(stride_periods)
    cv_period   = np.nanstd(stride_periods) / mean_period if mean_period > 0 else np.nan
    frequency   = 1.0 / mean_period if mean_period > 0 else np.nan

    # --- stride amplitude via peak-trough of fore-aft foot position ---
    fore_aft = df["foot_1_x"].values - df["torso_x"].values
    peaks,  _ = signal.find_peaks( fore_aft, distance=FRAMERATE // 2)
    troughs,_ = signal.find_peaks(-fore_aft, distance=FRAMERATE // 2)
    if len(peaks) > 0 and len(troughs) > 0:
        amplitude = np.mean(fore_aft[peaks]) - np.mean(fore_aft[troughs])
    else:
        amplitude = np.nan

    rows.append({
        "Variant":             label,
        "Mean stride period (s)": round(mean_period, 3),
        "Stride CV":           round(cv_period, 3),
        "Stride frequency (Hz)": round(frequency, 3),
        "Stride amplitude (m)":  round(amplitude, 3),
        "N strides":           len(stride_periods),
    })

df_out = pd.DataFrame(rows)

print("\n=== Stride Regularity ===")
print(df_out[["Variant", "Mean stride period (s)", "Stride CV", "N strides"]].to_string(index=False))

print("\n=== Gait Kinematics ===")
print(df_out[["Variant", "Stride frequency (Hz)", "Stride amplitude (m)"]].to_string(index=False))

df_out.to_csv("runs/gait_metrics.csv", index=False)
print("\nSaved: runs/gait_metrics.csv")
