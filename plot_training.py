import pandas as pd
import matplotlib.pyplot as plt
import os

# Add your run names here — one per variant
runs = [
    ("variantBaseline", "Baseline"),
    ("variantBodyLong1", "Body ext 1"),
    ("variantBodyLong2", "Body ext 2"),
    ("variantUpperLegShort", "Upper leg short"),
    ("variantUpperLegLong", "Upper leg long"),
    #("variantLowerLegShort", "Lower leg short"),
    ("variantLowerLegLong", "Lower leg long"),
]

fig, ax = plt.subplots(figsize=(7, 4))

for run, label in runs:
    metrics_path = f"runs/{run}/metrics.csv"
    if not os.path.exists(metrics_path):
        print(f"Skipping {run} — no metrics.csv found")
        continue
    metrics = pd.read_csv(metrics_path)
    ax.plot(metrics["num_steps"], metrics["reward"], label=label)

ax.set_xlabel("Timesteps")
ax.set_ylabel("Episode Reward")
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.tight_layout()
plt.savefig("runs/training_curves_all.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: runs/training_curves_all.png")
