# This is the main script for the feedforward analysis 


import os, sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import scipy.signal
import warnings
warnings.filterwarnings('ignore')

input_path = # Select here the path to the condition you want to analysis 
output_path = # Select here the path where you want to save the results

n_seeds, n_runs = 8, 20
dt = 0.05
for seed in range(n_seeds):
    local_path = os.path.join(input_path,f'baseline_seed{seed}','runs')
    local_output = os.path.join(output_path,f"baseline_seed{seed}")
    list_target, list_true = [], []
    list_duration_1, list_duration_2, list_duration_3, list_duration_4 = [], [], [], []
    list_distance_1, list_distance_2, list_distance_3, list_distance_4 = [], [], [], []
    for run in range(n_runs):
        local_data = pd.read_csv(os.path.join(local_path,f'data_run{run}.csv'))
        list_target.append(local_data['target_speed'].values[0])
        list_true.append(np.nanmean(np.diff(local_data["torso_x"].values))/dt)
    
        diff_leg1 = local_data['foot_1_x'].values - local_data['torso_x'].values
        diff_leg2 = local_data['foot_2_x'].values - local_data['torso_x'].values
        diff_leg3 = local_data['foot_3_x'].values - local_data['torso_x'].values
        diff_leg4 = local_data['foot_4_x'].values - local_data['torso_x'].values
        p_1, _ = scipy.signal.find_peaks(diff_leg1)
        p_2, _ = scipy.signal.find_peaks(diff_leg2)
        p_3, _ = scipy.signal.find_peaks(diff_leg3)
        p_4, _ = scipy.signal.find_peaks(diff_leg4)
        list_duration_1.append(np.nanmean(np.diff(p_1)))
        list_duration_2.append(np.nanmean(np.diff(p_2)))
        list_duration_3.append(np.nanmean(np.diff(p_3)))
        list_duration_4.append(np.nanmean(np.diff(p_4)))
        list_distance_1.append(np.nanmean(np.diff(local_data['foot_1_x'].values[p_1])))
        list_distance_2.append(np.nanmean(np.diff(local_data['foot_2_x'].values[p_2])))
        list_distance_3.append(np.nanmean(np.diff(local_data['foot_3_x'].values[p_3])))
        list_distance_4.append(np.nanmean(np.diff(local_data['foot_4_x'].values[p_4])))

    # Generate and save the figures 

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(list_target,list_true, color='k',s=5)
    axs.set_xlabel('Target speed'), axs.set_ylabel('Measured speed')
    plt.tight_layout()
    fig.savefig(os.path.join(local_output,f'speed_tracking_run{run}.png'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(list_target,list_duration_1, color='k',s=5)
    axs.scatter(list_target,list_duration_2, color='r',s=5)
    axs.scatter(list_target,list_duration_3, color='g',s=5)
    axs.scatter(list_target,list_duration_4, color='b',s=5)
    axs.set_xlabel("Target speed"), axs.set_ylabel("Stride duration")
    plt.tight_layout()
    fig.savefig(os.path.join(local_output,f'stride_duration_run{run}.png'),bbox_inches='tight')
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(list_target,list_distance_1, color='k',s=5)
    axs.scatter(list_target,list_distance_2, color='r',s=5)
    axs.scatter(list_target,list_distance_3, color='g',s=5)
    axs.scatter(list_target,list_distance_4, color='b',s=5)
    axs.set_xlabel("Target speed"), axs.set_ylabel("Stride duration")
    plt.tight_layout()
    fig.savefig(os.path.join(local_output,f'stride_length_run{run}.png'),bbox_inches='tight')
    plt.close('all')

    
