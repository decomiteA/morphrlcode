import os, sys 
import copy, pickle
import numpy as np 
import matplotlib.cm as cm
import scipy.stats 
import matplotlib.pyplot as plt 
from utils_analysis import *

output_path = os.path.join(os.getcwd(),'ResultsFigures')


n_seeds = 8
bins_limits = np.linspace(0.5,2,11)
bins_centers = bins_limits[0:-1] + np.nanmean(np.diff(bins_limits)/2) 
print(bins_limits)
matrix_stance_duration = np.zeros((n_seeds, len(bins_centers)))
matrix_duty_cycle = np.zeros((n_seeds, len(bins_centers)))
matrix_contact_mode = np.zeros((n_seeds, len(bins_centers),6))
for seed in range(n_seeds):
    input_path = os.path.join(os.getcwd(),'runs',f'baseline_lowrange_seed_{seed+1}','results')

    with open(os.path.join(input_path,'list_input_stride.pkl'),'rb') as f1:
        list_input_stride = pickle.load(f1)
    with open(os.path.join(input_path,'list_output_stride.pkl'),'rb') as f2:
        list_output_stride = pickle.load(f2)
    with open(os.path.join(input_path,'list_animal_stride.pkl'),'rb') as f3:
        list_animal_stride = pickle.load(f3)


    total_data = list_input_stride[0]
    total_metrics_data = list_output_stride[0]
    total_speed = np.expand_dims(np.nanmean(list_input_stride[0][:,:,2],axis=1),axis=-1)
    for run in range(1,50):
        local_data = list_input_stride[run]
        local_metrics_data = list_output_stride[run]
        local_speed = np.nanmean(local_data[:,:,2],axis=1)
        total_data = np.concatenate((total_data, local_data),axis=0)
        total_metrics_data = np.concatenate((total_metrics_data, local_metrics_data),axis=0)
        total_speed = np.vstack((total_speed,np.expand_dims(local_speed,axis=-1)))


    for bin in range(len(bins_centers)):
        idx_local = np.where((total_speed>bins_limits[bin]) & (total_speed<bins_limits[bin+1]))[0]
        matrix_stance_duration[seed, bin] = np.nanmean(total_metrics_data[idx_local,3]) 
        matrix_duty_cycle[seed, bin] = 0.01*np.nanmean(total_metrics_data[idx_local,3]) /np.nanmean(total_metrics_data[idx_local,2])
        for ii in range(matrix_contact_mode.shape[-1]):
            matrix_contact_mode[seed,bin,ii] = np.nanmean(total_metrics_data[idx_local,7+ii])


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers,np.nanmedian(matrix_stance_duration,axis=0), color='k', lw=2)
axs.fill_between(bins_centers,np.nanpercentile(matrix_stance_duration,axis=0,q=25), np.nanpercentile(matrix_stance_duration,axis=0,q=75), color='k', alpha=0.5)
axs.set_xlabel('speed'), axs.set_ylabel('stance duration')
plt.tight_layout()
fig.savefig(os.path.join(output_path,'baseline_lowrange_stance_duration.png'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers,np.nanmedian(matrix_duty_cycle,axis=0), color='k', lw=2)
axs.fill_between(bins_centers,np.nanpercentile(matrix_duty_cycle,axis=0,q=25), np.nanpercentile(matrix_duty_cycle,axis=0,q=75), color='k', alpha=0.5)
axs.set_xlabel('speed'), axs.set_ylabel('duty cycle')
plt.tight_layout()
fig.savefig(os.path.join(output_path,'baseline_lowrange_duty_cycle.png'),bbox_inches='tight')

labels = ['0 paw','1 paw','2 diago','2 others','3 paws','4 paws']
cmap = cm.gray(np.linspace(0,0.7,7))
fig, axs = plt.subplots(1,1,figsize=(6,6))
axs.spines[['top','right']].set_visible(False)
axs.stackplot(bins_centers,np.nanmean(matrix_contact_mode,axis=0).T,colors=cmap,labels=labels)
axs.set_ylabel('proportion'), axs.set_xlabel('speed')
axs.legend(frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(output_path,'baseline_lowrange_contact_mode.png'),bbox_inches='tight')
