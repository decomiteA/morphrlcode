import os, sys 
import numpy as np 
from scipy.fft import fft, fftfreq, fftshift
import scipy.signal
import pickle
import pandas as pd 
import scipy as sp
from tqdm import tqdm
from utils_analysis import *
import matplotlib.pyplot as plt 


n_seeds, n_runs = 8, 20
input_path = os.path.join(os.getcwd(),'runs')
for seed in range(n_seeds):
    local_output_path = os.path.join(input_path,f'baseline_seed{1}','results')
    os.makedirs(local_output_path, exist_ok=True)
    list_target, list_true = [], []
    list_duration_1, list_duration_2, list_duration_3, list_duration_4 = [], [], [], []
    list_distance_1, list_distance_2, list_distance_3, list_distance_4 = [], [], [], []
    list_corr2, list_corr3 ,list_corr4 = [], [], []
    list_input_body, list_input_self, list_output = [], [], []
    for run in range(n_runs):
        local_data = pd.read_csv(os.path.join(input_path,f'baseline_seed{1}',f'data_run{run}.csv'))
        input_data = reshape_data(local_data)
        input_data = compute_velocity_markers(input_data)
        foot_contact_matrix = get_foot_contact(input_data)

        # Getting the foot contact metrics 
        matrix_metrics = extract_metrics(input_data, foot_contact_matrix, run)
        total_metrics = np.concatenate((total_metrics, matrix_metrics), axis=0)
        # For the summary statistics (high level)
    
        phasor_metrics_leg_0 = extract_phasor_metrics(input_data, foot_contact_matrix, run, leg_id=0)
        # phasor_metrics_leg_1 = extract_phasor_metrics(input_data, foot_contact_matrix, run, leg_id=1)
        # phasor_metrics_leg_2 = extract_phasor_metrics(input_data, foot_contact_matrix, run, leg_id=2)
        # phasor_metrics_leg_3 = extract_phasor_metrics(input_data, foot_contact_matrix, run, leg_id=3)
        total_phase = np.concatenate((total_phase, phasor_metrics_leg_0),axis=0)
    
        # Getting the foot placement control data 
        idx_to_keep = np.where((matrix_metrics[:,1]==0) & (matrix_metrics[:,2]==0))[0]
    
        total_input, total_output, total_input_self = np.zeros((len(idx_to_keep),11,8)), np.zeros((len(idx_to_keep),6)), np.zeros((len(idx_to_keep),11,8))
        total_animal = matrix_metrics[idx_to_keep,0]
        idx_nans = np.where(np.isnan(input_data[0,0,:]))[0]
        for line in tqdm(range(len(idx_to_keep))):
            tmp_input, tmp_output = get_io_time_model_fr(foot_contact_matrix, input_data, matrix_metrics[idx_to_keep[line],:], matrix_metrics)
            tmp_self, _ = get_io_time_model_fr_self(foot_contact_matrix, input_data, matrix_metrics[idx_to_keep[line],:], matrix_metrics)
            if tmp_output is None:
                total_input[line,:], total_output[line,:], total_input_self[line,:] = np.nan, np.nan, np.nan
            elif tmp_output.shape[1]!=0:
                total_input[line,:] = tmp_input
                total_input_self[line,:] = tmp_self
                total_output[line,0] = tmp_output[0][0]
                total_output[line,1] = tmp_output[1][0]
                total_output[line,2] = tmp_output[2][0]
                total_output[line,3] = tmp_output[3][0]
                total_output[line,4] = tmp_output[4][0]
                total_output[line,5] = tmp_output[5][0]
            else:
                total_input[line,:], total_output[line,:], total_input_self[line,:] = np.nan, np.nan, np.nan
    
        list_input_body.append(total_input)
        list_input_self.append(total_input_self)
        list_output.append(total_output)
        
        local_speed = np.mean(np.diff(local_data['torso_x'].values))/dt
        list_target.append(local_data['target_speed'].values[0])
        list_true.append(local_speed)
    
        diff_leg1 = local_data['foot_1_x'].values - local_data['torso_x'].values
        diff_leg2 = local_data['foot_2_x'].values - local_data['torso_x'].values
        diff_leg3 = local_data['foot_3_x'].values - local_data['torso_x'].values
        diff_leg4 = local_data['foot_4_x'].values - local_data['torso_x'].values
        diff_leg1 = diff_leg1 - np.nanmean(diff_leg1)
        diff_leg2 = diff_leg2 - np.nanmean(diff_leg2)
        diff_leg3 = diff_leg3 - np.nanmean(diff_leg3)
        diff_leg4 = diff_leg4 - np.nanmean(diff_leg4)
        p_1, _ = scipy.signal.find_peaks(diff_leg1, distance=10)
        p_2, _ = scipy.signal.find_peaks(diff_leg2, distance=10)
        p_3, _ = scipy.signal.find_peaks(diff_leg3, distance=10)
        p_4, _ = scipy.signal.find_peaks(diff_leg4, distance=10)
        list_duration_1.append(np.nanmean(np.diff(p_1)))
        list_duration_2.append(np.nanmean(np.diff(p_2)))
        list_duration_3.append(np.nanmean(np.diff(p_3)))
        list_duration_4.append(np.nanmean(np.diff(p_4)))
        list_distance_1.append(np.nanmean(np.diff(local_data['foot_1_x'].values[p_1])))
        list_distance_2.append(np.nanmean(np.diff(local_data['foot_2_x'].values[p_2])))
        list_distance_3.append(np.nanmean(np.diff(local_data['foot_3_x'].values[p_3])))
        list_distance_4.append(np.nanmean(np.diff(local_data['foot_4_x'].values[p_4])))
        local_lags = scipy.signal.correlation_lags(len(diff_leg1),len(diff_leg2))
        list_corr2.append(local_lags[np.argmax(scipy.signal.correlate(diff_leg1,diff_leg2,'full'))]/list_duration_1[-1])
        list_corr3.append(local_lags[np.argmax(scipy.signal.correlate(diff_leg1,diff_leg3,'full'))]/list_duration_1[-1])
        list_corr4.append(local_lags[np.argmax(scipy.signal.correlate(diff_leg1,diff_leg4,'full'))]/list_duration_1[-1])

    # Generate the figures and save the data for that seed 
    total_metrics = total_metrics[1:,:]
    total_phase = total_phase[1:,:]
    idx_same = np.where((total_metrics[:,1]==0) & (total_metrics[:,2]==0))[0]

    with open(os.path.join(local_output_path,'list_input_body.pkl'),'wb') as f1:
        pickle.dump(list_input_body, f1)
    with open(os.path.join(local_output_path,'list_input_self.pkl'),'wb') as f2:
        pickle.dump(list_input_self, f2)
    with open(os.path.join(local_output_path,'list_output.pkl'),'wb') as f3:
        pickle.dump(list_output, f3)

    # Saving the data once and for all
    np.save(os.path.join(local_output_path,'total_metrics.npy'), total_metrics)
    np.save(os.path.join(local_output_path,'phase_metrics.npy'), total_phase)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(list_target,list_true, color='k',s=5)
    axs.plot([1,2],[1,2],color='r',ls=':')
    axs.set_xlabel('target speed'), axs.set_ylabel('measured speed')
    plt.tight_layout()
    fig.savefig(os.path.join(local_output_path,'figure_speed_tracking.png'),bbox_inches='tight')
    fig.savefig(os.path.join(local_output_path,'figure_speed_tracking.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(list_target,list_duration_1, color='k',s=5,label='Front left')
    axs.scatter(list_target,list_duration_2, color='r',s=5,label='Hind left')
    axs.scatter(list_target,list_duration_3, color='g',s=5,label='Hind right')
    axs.scatter(list_target,list_duration_4, color='b',s=5,label='Front right')
    axs.set_xlabel('speed'), axs.set_ylabel('stride duration')
    plt.tight_layout()
    fig.savefig(os.path.join(local_output_path,'figure_speed_duration.png'),bbox_inches='tight')
    fig.savefig(os.path.join(local_output_path,'figure_speed_duration.svg'),bbox_inches='tight')
    


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(list_target,list_distance_1, color='k',s=5)
    axs.scatter(list_target,list_distance_2, color='r',s=5)
    axs.scatter(list_target,list_distance_3, color='g',s=5)
    axs.scatter(list_target,list_distance_4, color='b',s=5)
    axs.set_xlabel('speed'), axs.set_ylabel('stride length')
    plt.tight_layout()
    fig.savefig(os.path.join(local_output_path,'figure_speed_length.png'),bbox_inches='tight')
    fig.savefig(os.path.join(local_output_path,'figure_speed_length.svg'),bbox_inches='tight')
    



