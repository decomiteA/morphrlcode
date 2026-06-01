import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.decomposition import PCA
from analyze import load
from utils import *

# Change accordingly to whichever runs need to be plotted
runs = [
    "variantBaseline",
    #"variantBodyLong1",
    #"variantBodyLong2",
    #"variantUpperLegShort",
    #"variantUpperLegLong",
    #"variantLowerLegShort",
    #"variantLowerLegLong",
]


def get_foot_contact(raw_data, framerate=20):
    local_data = copy.deepcopy(raw_data)
    foot_contact_matrix = np.zeros((2, 4, local_data.shape[-1]))

    bool_up = (np.nanmean(local_data[2, 0, :]) > 0).astype(int)

    diff_leg1 = local_data[0, 1, :] - local_data[0, 0, :]
    diff_leg2 = local_data[0, 2, :] - local_data[0, 0, :]
    diff_leg3 = local_data[0, 3, :] - local_data[0, 0, :]
    diff_leg4 = local_data[0, 4, :] - local_data[0, 0, :]

    max_leg1, _ = signal.find_peaks(diff_leg1, distance=framerate / 10)
    max_leg2, _ = signal.find_peaks(diff_leg2, distance=framerate / 10)
    max_leg3, _ = signal.find_peaks(diff_leg3, distance=framerate / 10)
    max_leg4, _ = signal.find_peaks(diff_leg4, distance=framerate / 10)

    min_leg1, _ = signal.find_peaks(-diff_leg1, distance=framerate / 10)
    min_leg2, _ = signal.find_peaks(-diff_leg2, distance=framerate / 10)
    min_leg3, _ = signal.find_peaks(-diff_leg3, distance=framerate / 10)
    min_leg4, _ = signal.find_peaks(-diff_leg4, distance=framerate / 10)

    bool_contact = np.zeros((len(diff_leg1), 4))
    for leg in range(4):
        bool_contact[max_leg1, 0], bool_contact[min_leg1, 0] = 1, -1
        bool_contact[max_leg2, 1], bool_contact[min_leg2, 1] = 1, -1
        bool_contact[max_leg3, 2], bool_contact[min_leg3, 2] = 1, -1
        bool_contact[max_leg4, 3], bool_contact[min_leg4, 3] = 1, -1

    for leg in range(4):
        for time in range(bool_contact.shape[0]):
            cdt1 = (1 if bool_up else -1)
            cdt2 = (-1 if bool_up else 1)
            lb = min(5, len(bool_contact[:time, leg]))
            la = min(5, len(bool_contact[time:, leg]))
            if ((bool_contact[time, leg] == cdt1) &
                    (np.sum(np.abs(bool_contact[time - lb:time, leg])) == 0) &
                    (np.sum(np.abs(bool_contact[time + 1:time + 1 + la, leg])) == 0)):
                idx_next_toeoff = np.where(bool_contact[time:, leg] == cdt2)[0]
                if len(idx_next_toeoff) == 0:
                    idx_endc = bool_contact[time:, leg].shape[0] - 1
                else:
                    idx_endc = idx_next_toeoff[0]
                x_pos = np.nanmean(local_data[0, leg + 1, time:time + idx_endc])
                y_pos = np.nanmean(local_data[1, leg + 1, time:time + idx_endc])
                foot_contact_matrix[0, leg, time:time + idx_endc] = x_pos
                foot_contact_matrix[1, leg, time:time + idx_endc] = y_pos

    return foot_contact_matrix


# foot_1: front left, foot_2: back left, foot_3: back right, foot_4: front right
labels = ["front left", "back left", "back right", "front right"]
colors = ["blue", "red", "green", "orange"]
FOOT_ORDER = [4, 1, 3, 2]  # front right, front left, hind right, hind left
SKIP_STEPS = 100
PLOT_STEPS = 200


for run in runs:
    if not os.path.exists(f"runs/{run}/data.csv"):
        print(f"Skipping {run} — no data.csv found")
        continue
    print(f"\n=== {run} ===")

    df = load(run)
    df = df.iloc[SKIP_STEPS:].reset_index(drop=True)
    N = len(df)
    t = df["t"].values

    for i in range(1, 5):
        x = df[f"foot_{i}_x"].values - df["torso_x"].values
        c = df[f"foot_{i}_contact"].values.astype(bool)
        plt.plot(t[:PLOT_STEPS], x[:PLOT_STEPS], color=colors[i-1], label=labels[i-1])
        plt.scatter(t[:PLOT_STEPS][c[:PLOT_STEPS]], x[:PLOT_STEPS][c[:PLOT_STEPS]],
                    color=colors[i-1], s=20, zorder=5)
    plt.xlabel("Time (s)")
    plt.ylabel("Foot X (m)")
    plt.legend()
    plt.savefig(f"runs/{run}/foot_x.png", dpi=150)
    plt.close()
    print(f"Saved: runs/{run}/foot_x.png")

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

    # PCA rotation to align direction of travel with x-axis
    pca_data = np.concatenate((matrix[0].reshape(-1, 1), matrix[1].reshape(-1, 1)), axis=1)
    pca = PCA(n_components=2)
    pca.fit(pca_data)
    local_angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    rot_matrix = np.array([[np.cos(-local_angle), -np.sin(-local_angle)],
                            [np.sin(-local_angle),  np.cos(-local_angle)]])
    rotated_matrix = np.zeros(matrix.shape)
    for time in range(rotated_matrix.shape[2]):
        for marker in range(rotated_matrix.shape[1]):
            rotated_matrix[0, marker, time] = rot_matrix[0, 0] * matrix[0, marker, time] + rot_matrix[0, 1] * matrix[1, marker, time]
            rotated_matrix[1, marker, time] = rot_matrix[1, 0] * matrix[0, marker, time] + rot_matrix[1, 1] * matrix[1, marker, time]
            rotated_matrix[2, marker, time] = rot_matrix[0, 0] * matrix[2, marker, time] + rot_matrix[0, 1] * matrix[3, marker, time]
            rotated_matrix[3, marker, time] = rot_matrix[1, 0] * matrix[2, marker, time] + rot_matrix[1, 1] * matrix[3, marker, time]

    foot_data = get_foot_contact(rotated_matrix, framerate=20)

    nan_col = np.full((rotated_matrix.shape[0], rotated_matrix.shape[1], 1), np.nan)
    input_raw_nan = np.concatenate([nan_col, rotated_matrix, nan_col], axis=2)
    foot_data_nan = np.concatenate([np.full((2, 4, 1), np.nan), foot_data, np.full((2, 4, 1), np.nan)], axis=2)
    dummy_vector = np.zeros((input_raw_nan.shape[2], 1))

    output_metrics = extract_metrics(input_raw_nan, foot_data_nan, dummy_vector, framerate=20)

    idx_to_keep = np.where((output_metrics[:, 1] == 0) & (output_metrics[:, 2] == 0))[0]
    print(f"  {len(idx_to_keep)} gait cycles found")

    total_input = np.zeros((len(idx_to_keep), 21, 8))
    total_output = np.zeros((len(idx_to_keep), 6))
    total_input_self = np.zeros((len(idx_to_keep), 21, 8))
    total_animal = output_metrics[idx_to_keep, 0]

    for line in tqdm(range(len(idx_to_keep))):
        tmp_input, tmp_output = get_io_time_model_cycle_fr(
            foot_data_nan, input_raw_nan, output_metrics[idx_to_keep[line], :], output_metrics)
        tmp_input_self, _ = get_io_time_model_cycle_self(
            foot_data_nan, input_raw_nan, output_metrics[idx_to_keep[line], :], output_metrics)
        if tmp_output is None:
            total_input[line, :], total_output[line, :], total_input_self[line, :] = np.nan, np.nan, np.nan
        elif tmp_output.shape[1] != 0:
            total_input[line, :] = tmp_input
            total_input_self[line, :] = tmp_input_self
            total_output[line, 0] = tmp_output[0][0]
            total_output[line, 1] = tmp_output[1][0]
            total_output[line, 2] = tmp_output[2][0]
            total_output[line, 3] = tmp_output[3][0]
            total_output[line, 4] = tmp_output[4][0]
            total_output[line, 5] = tmp_output[5][0]
        else:
            total_input[line, :], total_output[line, :], total_input_self[line, :] = np.nan, np.nan, np.nan

    rsquare_body, gains_body = get_rsquare_matrix_feedback(
        total_animal, total_input, total_output, bool_hind=False, bool_lat=True)
    rsquare_self = get_rsquare_matrix_self(
        total_animal, total_input_self, total_output, bool_hind=False, bool_lat=True)

    x = np.linspace(0, 1, 21)
    fig, axs = plt.subplots(1, 1, figsize=(3.5, 3.5))
    axs.spines[['top', 'right']].set_visible(False)
    axs.set_ylim([-0.05, 1.05])
    axs.plot(x, np.nanmedian(rsquare_body, axis=0), color='k', label='Body state')
    axs.plot(x, np.nanmedian(rsquare_self, axis=0), color='r', label='Foot self-history')
    axs.fill_between(x,
                     np.nanpercentile(rsquare_body, q=25, axis=0),
                     np.nanpercentile(rsquare_body, q=75, axis=0), color='k', alpha=0.3)
    axs.fill_between(x,
                     np.nanpercentile(rsquare_self, q=25, axis=0),
                     np.nanpercentile(rsquare_self, q=75, axis=0), color='r', alpha=0.3)
    axs.axvline(x=0.75, color='gray', linestyle='--', linewidth=1, label='3/4 cycle')
    axs.set_xlabel("Gait cycle phase")
    axs.set_ylabel(r"$R^2$")
    axs.set_title(run)
    axs.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"runs/{run}/rsquare.png", dpi=150)
    plt.close()
    print(f"Saved: runs/{run}/rsquare.png")
