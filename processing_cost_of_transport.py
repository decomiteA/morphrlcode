import os, sys 
import numpy as np 
import scipy.signal 
import pickle
import scipy.signal
import pandas as pd 
from utils_analysis import *
import matplotlib.pyplot as plt 


str_group = 'baseline_lowrange_d5'

dt = 0.01
n_seeds, n_runs = 8, 50
input_path = os.path.join(os.getcwd(),'final_runs')
output_path = os.path.join(os.getcwd(),'FinalFigures')
os.makedirs(output_path,exist_ok=True)
tot_list_target, tot_list_cot = [], [] 
bins_limits = np.linspace(0.5,2,11)
bins_centers = bins_limits[0:-1] + (np.mean(np.diff(bins_limits))/2)
tot_matrix_cot = np.zeros((8, len(bins_centers)))
for seed in range(n_seeds):
    local_output_path = os.path.join(input_path,f'{str_group}_seed_{seed+1}','results')
    os.makedirs(local_output_path, exist_ok=True)
    list_target, list_cot_power = [], []

    for run in range(n_runs):
        local_data = pd.read_csv(os.path.join(input_path,f'{str_group}_seed_{seed+1}',f'data_run{run}.csv'))
        actions_data = get_actions(local_data)
        power_vector = get_power(local_data)
        tot_energy = np.sum(np.square(actions_data))
        local_speed = np.mean(np.diff(local_data['torso_x'].values))/0.01
        list_target.append(local_speed)
        list_cot_power.append(np.nanmean(power_vector)/(local_speed*9.81*0.9108))
    tot_list_target.append(list_target), tot_list_cot.append(list_cot_power)

    for bin in range(len(bins_centers)):
        idx_local = np.where((list_target>bins_limits[bin]) & (list_target<bins_limits[bin+1]))[0]
        tot_matrix_cot[seed,bin] = np.nanmedian(np.array(list_cot_power)[idx_local])


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers, np.nanmedian(tot_matrix_cot, axis=0), color='k',lw=2)
axs.fill_between(bins_centers, np.nanpercentile(tot_matrix_cot,axis=0,q=25), np.nanpercentile(tot_matrix_cot,axis=0,q=75), color='k', alpha=0.5)
axs.set_xlabel('speed'), axs.set_ylabel('cost of transport')
fig.savefig(os.path.join(output_path,f'{str_group}_cost_of_transport.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_cost_of_transport.svg'),bbox_inches='tight')
plt.show()