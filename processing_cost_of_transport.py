import os, sys 
import numpy as np 
import scipy.signal 
import pickle
import scipy.signal
import pandas as pd 
from utils_analysis import *
import matplotlib.pyplot as plt 


dt = 0.01
n_seeds, n_runs = 8, 50
input_path = os.path.join(os.getcwd(),'final_runs')
tot_list_target, tot_list_cot = [], [] 
for seed in range(n_seeds):
    local_output_path = os.path.join(input_path,f'baseline_lowrange_d7p5_seed_{seed+1}','results')
    os.makedirs(local_output_path, exist_ok=True)
    list_target, list_cot_power = [], []

    for run in range(n_runs):
        local_data = pd.read_csv(os.path.join(input_path,f'baseline_lowrange_d7p5_seed_{seed+1}',f'data_run{run}.csv'))
        actions_data = get_actions(local_data)
        power_vector = get_power(local_data)
        print(np.nanmean(power_vector))
        distance = np.max(local_data['torso_x'].values) - np.min(local_data['torso_x'].values)
        tot_energy = np.sum(np.square(actions_data))
        local_speed = np.mean(np.diff(local_data['torso_x'].values))/0.01
        list_target.append(local_speed)
        list_cot_power.append(np.nanmean(power_vector)/(local_speed*9.81*0.9108))
    tot_list_target.append(list_target), tot_list_cot.append(list_cot_power)


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
for seed in range(n_seeds):
    axs.scatter(tot_list_target[seed], tot_list_cot[seed], s=2, alpha=0.5)
axs.set_xlabel('target speed'), axs.set_ylabel('cost of transport')
plt.tight_layout()
fig.savefig('cost_of_transport_d7p5.png',bbox_inches='tight')
    # fig, axs = plt.subplots(1,1,figsize=(3,3))
    # axs.spines[['top','right']].set_visible(False)
    # axs.scatter(list_target, list_cot, color='k',s=5,alpha=0.5)
    # # axs.scatter(list_target, list_cot_power, color='r', s=5, alpha=0.5)
    # axs.set_xlabel('speed'), axs.set_ylabel('cost of transport')
    # plt.tight_layout()
    # fig.savefig(f'dummy_fig.png',bbox_inches='tight')
    # sys.exit()

    