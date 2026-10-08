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
input_path = os.path.join(os.getcwd(),'runs')
for seed in range(n_seeds):
    local_output_path = os.path.join(input_path,f'baseline_lowrange_seed_{seed+1}','results')
    os.makedirs(local_output_path, exist_ok=True)
    list_target, list_cot = [], []

    for run in range(n_runs):
        local_data = pd.read_csv(os.path.join(input_path,f'baseline_lowrange_seed_{seed+1}',f'data_run{run}.csv'))
        actions_data = get_actions(local_data)
        distance = np.max(local_data['torso_x'].values) - np.min(local_data['torso_x'].values)
        tot_energy = np.sum(np.square(actions_data))
        list_target.append(local_data['target_speed'].values[0])
        list_cot.append(tot_energy/distance)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(list_target, list_cot, color='k',s=5)
    axs.set_xlabel('speed'), axs.set_ylabel('cost of transport')
    plt.tight_layout()
    fig.savefig('dummy_fig.png',bbox_inches='tight')
    # sys.exit()

    