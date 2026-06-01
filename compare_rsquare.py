from analyze import load
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy.signal as signal
from sklearn.decomposition import PCA
from utils import *
import os

SKIP_STEPS = 100  # change to higher number if model in rollout needs more time to adjust, for anaylzable results
FOOT_ORDER = [4, 1, 3, 2]
N_SEEDS = 20  # number of rollout seeds per variant (set to 1 to use single run)

groups = [
    ("baseline", "Baseline", [
        ("variantBaseline", "Baseline", "k"),
    ]),
    ("upper_leg_long", "Upper Leg Long", [
        ("variantUpperLegLong", "Long (0.25)", "#b92929"),
    ]),
    ("lower_leg", "Lower Leg Length", [
        ("variantLowerLegShort", "Short (0.30)", "#350904"),
        ("variantLowerLegLong",  "Long (0.50)",  "#2980b9"),
    ]),
]


def get_run_names(base):
    seeded = [f"{base}/r{i}" for i in range(N_SEEDS)]
    available = [r for r in seeded if os.path.exists(f"runs/{r}/data.csv")]
    if available:
        return available
    return [base]  # fall back to single run


def get_foot_contact(raw_data, framerate=20):
    local_data = copy.deepcopy(raw_data)
    foot_contact_matrix = np.zeros((2, 4, local_data.shape[-1]))
    bool_up = (np.nanmean(local_data[2, 0, :]) > 0).astype(int)

    diffs = [local_data[0, i, :] - local_data[0, 0, :] for i in range(1, 5)]
    maxes = [signal.find_peaks( d, distance=framerate / 10)[0] for d in diffs]
    mines = [signal.find_peaks(-d, distance=framerate / 10)[0] for d in diffs]

    bool_contact = np.zeros((len(diffs[0]), 4))
    for leg in range(4):
        bool_contact[maxes[leg], leg] =  1
        bool_contact[mines[leg], leg] = -1

    for leg in range(4):
        for time in range(bool_contact.shape[0]):
            cdt1 = 1 if bool_up else -1
            cdt2 = -1 if bool_up else 1
            lb = min(5, time)
            la = min(5, bool_contact.shape[0] - time)
            if (bool_contact[time, leg] == cdt1 and
                    np.sum(np.abs(bool_contact[time - lb:time, leg])) == 0 and
                    np.sum(np.abs(bool_contact[time + 1:time + 1 + la, leg])) == 0):
                idx_toe = np.where(bool_contact[time:, leg] == cdt2)[0]
                idx_end = idx_toe[0] if len(idx_toe) > 0 else bool_contact.shape[0] - 1 - time
                x_pos = np.nanmean(local_data[0, leg + 1, time:time + idx_end])
                y_pos = np.nanmean(local_data[1, leg + 1, time:time + idx_end])
                foot_contact_matrix[0, leg, time:time + idx_end] = x_pos
                foot_contact_matrix[1, leg, time:time + idx_end] = y_pos
    return foot_contact_matrix


def build_matrix(run):
    """Load one run and return its PCA-rotated matrix."""
    df = load(run)
    df = df.iloc[SKIP_STEPS:].reset_index(drop=True)
    N = len(df)
    matrix = np.zeros((4, 5, N))
    for ti in range(N):
        for pos, foot_num in enumerate(FOOT_ORDER, start=1):
            matrix[0, pos, ti] = df[f"foot_{foot_num}_x"].values[ti]
            matrix[1, pos, ti] = df[f"foot_{foot_num}_y"].values[ti]
            matrix[2, pos, ti] = df[f"foot_{foot_num}_vx"].values[ti]
            matrix[3, pos, ti] = df[f"foot_{foot_num}_vy"].values[ti]
        matrix[0, 0, ti] = df["torso_x"].values[ti]
        matrix[1, 0, ti] = df["torso_y"].values[ti]
        matrix[2, 0, ti] = df["torso_vel_x"].values[ti]
        matrix[3, 0, ti] = df["torso_vel_y"].values[ti]
    pca = PCA(n_components=2)
    pca.fit(np.concatenate([matrix[0].reshape(-1, 1), matrix[1].reshape(-1, 1)], axis=1))
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    R = np.array([[np.cos(-angle), -np.sin(-angle)],
                  [np.sin(-angle),  np.cos(-angle)]])
    rot = np.zeros(matrix.shape)
    for ti in range(N):
        for m in range(5):
            rot[0, m, ti] = R[0, 0] * matrix[0, m, ti] + R[0, 1] * matrix[1, m, ti]
            rot[1, m, ti] = R[1, 0] * matrix[0, m, ti] + R[1, 1] * matrix[1, m, ti]
            rot[2, m, ti] = R[0, 0] * matrix[2, m, ti] + R[0, 1] * matrix[3, m, ti]
            rot[3, m, ti] = R[1, 0] * matrix[2, m, ti] + R[1, 1] * matrix[3, m, ti]
    return rot


def process_single_run(run, seed_id=0):
    rot = build_matrix(run)
    foot_data = get_foot_contact(rot, framerate=20)
    nan_col = np.full((rot.shape[0], rot.shape[1], 1), np.nan)
    input_raw_nan = np.concatenate([nan_col, rot, nan_col], axis=2)
    foot_data_nan = np.concatenate([np.full((2, 4, 1), np.nan), foot_data,
                                    np.full((2, 4, 1), np.nan)], axis=2)
    dummy_vector = np.full((input_raw_nan.shape[2], 1), seed_id)

    output_metrics = extract_metrics(input_raw_nan, foot_data_nan, dummy_vector, framerate=20)
    idx_to_keep = np.where((output_metrics[:, 1] == 0) & (output_metrics[:, 2] == 0))[0]

    total_input = np.full((len(idx_to_keep), 21, 8), np.nan)
    total_output = np.full((len(idx_to_keep), 6), np.nan)
    total_input_self = np.full((len(idx_to_keep), 21, 8), np.nan)
    total_animal = output_metrics[idx_to_keep, 0]

    for line in range(len(idx_to_keep)):
        tmp_input, tmp_output = get_io_time_model_cycle_fr(
            foot_data_nan, input_raw_nan, output_metrics[idx_to_keep[line], :], output_metrics)
        tmp_input_self, _ = get_io_time_model_cycle_self(
            foot_data_nan, input_raw_nan, output_metrics[idx_to_keep[line], :], output_metrics)
        if tmp_output is not None and tmp_output.shape[1] != 0:
            total_input[line, :] = tmp_input
            total_input_self[line, :] = tmp_input_self
            for col in range(6):
                total_output[line, col] = tmp_output[col][0]

    return total_input, total_output, total_input_self, total_animal


def compute_rsquare(base_run):
    run_names = get_run_names(base_run)
    print(f"  Pooling {len(run_names)} run(s) for {base_run}")

    all_input, all_output, all_input_self, all_animal = [], [], [], []
    for seed_id, run in enumerate(run_names):
        ti, to, tis, ta = process_single_run(run, seed_id=seed_id)
        all_input.append(ti)
        all_output.append(to)
        all_input_self.append(tis)
        all_animal.append(ta)

    total_input = np.concatenate(all_input, axis=0)
    total_output = np.concatenate(all_output, axis=0)
    total_input_self = np.concatenate(all_input_self, axis=0)
    total_animal = np.concatenate(all_animal, axis=0)
    print(f"  {base_run}: {len(total_animal)} gait cycles total")

    rsquare_body, _ = get_rsquare_matrix_feedback(
        total_animal, total_input, total_output, bool_hind=False, bool_lat=True)
    rsquare_self = get_rsquare_matrix_self(
        total_animal, total_input_self, total_output, bool_hind=False, bool_lat=True)
    return rsquare_body, rsquare_self


for fname, title, variants in groups:
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (run, label, color) in zip(axes, variants):
        rsquare_body, rsquare_self = compute_rsquare(run)
        x = np.linspace(0, 1, 21)
        ax.plot(x, np.nanmedian(rsquare_body, axis=0), color='k', label='Body state')
        ax.plot(x, np.nanmedian(rsquare_self, axis=0), color='r', label='Foot self-history')
        ax.fill_between(x,
                        np.nanpercentile(rsquare_body, q=25, axis=0),
                        np.nanpercentile(rsquare_body, q=75, axis=0),
                        color='k', alpha=0.3)
        ax.fill_between(x,
                        np.nanpercentile(rsquare_self, q=25, axis=0),
                        np.nanpercentile(rsquare_self, q=75, axis=0),
                        color='r', alpha=0.3)
        ax.axvline(x=0.75, color='gray', linestyle='--', linewidth=1, label='3/4 cycle')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("Gait cycle phase")
        ax.set_title(label)
        ax.legend(fontsize=7)

    axes[0].set_ylabel(r"$R^2$")
    fig.suptitle(title, fontsize=11)
    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    plt.savefig(f"runs/rsquare_{fname}.png", dpi=150)
    plt.close()
    print(f"Saved: runs/rsquare_{fname}.png")
