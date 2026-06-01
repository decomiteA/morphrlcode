import os
import numpy as np
import pandas as pd

from config import PPOConfig


def load(run_name, cfg=None):
    if cfg is None:
        cfg = PPOConfig()
    path = os.path.join(cfg.runs_dir, run_name, "data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data at {path} — run rollout.py first")
    df = pd.read_csv(path)

    df["vel_x_smooth"] = df["torso_vel_x"].rolling(20, center=True, min_periods=1).mean()

    qw, qx, qy, qz = df["torso_qw"], df["torso_qx"], df["torso_qy"], df["torso_qz"]
    df["pitch"] = np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qy**2 + qx**2))
    df["roll"]  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))

    df["distance"] = df["torso_x"].values - df["torso_x"].values[0]

    h_mean = df["torso_z"].mean()
    omega_0 = np.sqrt(9.81 / h_mean)
    df["xcom_x"] = df["torso_x"] + df["torso_vel_x"] / omega_0
    df["xcom_y"] = df["torso_y"] + df["torso_vel_y"] / omega_0

    return df
