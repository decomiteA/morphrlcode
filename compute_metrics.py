import numpy as np
import pandas as pd
from compare_rsquare import compute_rsquare

variants = [
    ("variantBaseline",      "Baseline"),
    ("variantUpperLegLong",  "Upper leg long"),
    ("variantLowerLegShort", "Lower leg short"),
    ("variantLowerLegLong",  "Lower leg long"),
]

rows = []
for run, label in variants:
    print(f"Processing {label}...")
    rsquare_body, rsquare_self = compute_rsquare(run)
    feedback_median   = np.nanmedian(rsquare_body, axis=0)
    feedforward_median = np.nanmedian(rsquare_self, axis=0)
    magnitude  = float(np.nanmax(feedback_median))
    dominance  = float(np.nanmean(feedback_median - feedforward_median))
    rows.append({
        "Variant":             label,
        "Peak Feedback R²":    round(magnitude, 3),
        "Feedback Dominance":  round(dominance, 3),
    })

df = pd.DataFrame(rows)
print("\n=== Foot Placement Control Metrics ===")
print(df.to_string(index=False))
df.to_csv("runs/control_metrics.csv", index=False)
print("\nSaved: runs/control_metrics.csv")
