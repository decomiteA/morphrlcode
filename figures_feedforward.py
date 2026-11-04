import os, sys 
import pickle
import numpy as np 
import scipy.signal
import matplotlib.pyplot as plt 
from utils_analysis import *
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')
str_group = 'baseline_lowrange_d5'
output_path = os.path.join(os.getcwd(),'FinalFigures')
os.makedirs(output_path,exist_ok=True)
folder_hildebrand = os.path.join(output_path,f'hildebrandfigures_{str_group}')
os.makedirs(folder_hildebrand,exist_ok=True)
n_seeds = 8
list_metrics_data, list_phasor_data, list_hildebrand, list_speed = [], [], [], []
list_input_stride, list_output_stride, list_animal_stride = [], [], []
for seed in range(n_seeds):
    local_input_path = os.path.join(os.getcwd(),'final_runs',f'{str_group}_seed_{seed+1}','results')
    

    list_metrics_data.append(np.load(os.path.join(local_input_path,'total_metrics.npy')))
    list_phasor_data.append(np.load(os.path.join(local_input_path,'phase_metrics.npy')))
    list_hildebrand.append(np.load(os.path.join(local_input_path,'hildebrand_data.npy')))
    list_speed.append(get_speed(list_metrics_data[-1]))

    with open(os.path.join(local_input_path,'list_input_stride.pkl'),'rb') as f1:
        list_input_stride.append(pickle.load(f1))
    with open(os.path.join(local_input_path,'list_output_stride.pkl'),'rb') as f2:
        list_output_stride.append(pickle.load(f2))
    with open(os.path.join(local_input_path,'list_animal_stride.pkl'),'rb') as f3:
        list_animal_stride.append(pickle.load(f3))


bins_limits = np.linspace(0.5,2,11)
bins_centers = bins_limits[0:-1] + (np.mean(np.diff(bins_limits))/2)

matrix_length, matrix_duration = np.zeros((n_seeds,len(bins_centers))), np.zeros((n_seeds,len(bins_centers)))
matrix_length_front, matrix_duration_front = np.zeros((n_seeds,len(bins_centers))), np.zeros((n_seeds,len(bins_centers)))

matrix_binned_hildebrand = np.zeros((n_seeds,len(bins_centers),4,51))

matrix_relative_time = np.zeros((n_seeds, len(bins_centers),3))

matrix_stance_duration = np.zeros((n_seeds, len(bins_centers)))
matrix_duty_cycle = np.zeros((n_seeds, len(bins_centers)))
matrix_contact_mode = np.zeros((n_seeds, len(bins_centers),6))

for seed in range(n_seeds):
    local_data = list_metrics_data[seed]
    local_speed = list_speed[seed]
    local_hildebrand = list_hildebrand[seed]
    local_input_stride, local_output_stride, local_animal_stride = list_input_stride[seed], list_output_stride[seed], list_animal_stride[seed]
    for bin in range(len(bins_centers)):
        idx_local = np.where((local_data[:,1]==0) & (local_data[:,2]==0) & (local_data[:,6]>bins_limits[bin]) & (local_data[:,6]<bins_limits[bin+1]))
        matrix_length[seed, bin] = np.nanmedian(local_data[idx_local,4])
        matrix_duration[seed, bin] = np.nanmedian(local_data[idx_local,3])

        idx_front = np.where((local_data[:,1]==0) & (local_data[:,2]==1) & (local_data[:,6]>bins_limits[bin]) & (local_data[:,6]<bins_limits[bin+1]))
        matrix_length_front[seed, bin] = np.nanmedian(local_data[idx_front,4])
        matrix_duration_front[seed, bin] = np.nanmedian(local_data[idx_front,3])
        phasor_data = list_phasor_data[seed]
        idx_local = np.where((np.abs(phasor_data[:,-1])>bins_limits[bin]) & (np.abs(phasor_data[:,-1])<bins_limits[bin+1]))[0]
        matrix_relative_time[seed,bin,0] = scipy.stats.circmean(phasor_data[idx_local,3]/phasor_data[idx_local,2], low=0, high=1, nan_policy='omit')
        matrix_relative_time[seed,bin,1] = scipy.stats.circmean(phasor_data[idx_local,4]/phasor_data[idx_local,2], low=0, high=1, nan_policy='omit')
        matrix_relative_time[seed,bin,2] = scipy.stats.circmean(phasor_data[idx_local,5]/phasor_data[idx_local,2], low=-0.5, high=0.5, nan_policy='omit')

        idx_hildebrand = np.where((local_speed>bins_limits[bin]) & (local_speed<bins_limits[bin+1]))[0]
        matrix_binned_hildebrand[seed,bin,:,:] = np.nanmean(local_hildebrand[idx_hildebrand,:,:],axis=0)
    total_data = local_input_stride[0]
    total_metrics_data = local_output_stride[0]
    total_speed = np.expand_dims(np.nanmean(local_input_stride[0][:,:,2],axis=1),axis=-1)
    for run in range(1,50):
        local_data = local_input_stride[run]
        local_metrics_data = local_output_stride[run]
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


matrix_relative_time *= 2*np.pi


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
for seed in range(n_seeds):
    local_data = list_metrics_data[seed]
    idx_same = np.where((local_data[:,1]==0) & (local_data[:,2]==0))[0]
    axs.scatter(local_data[idx_same,6], local_data[idx_same,4],s=5,alpha=0.1)
axs.set_xlabel('speed [m/s]'), axs.set_ylabel('stride length [m]')
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_stride_length_raw.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_stride_length_raw.svg'),bbox_inches='tight')


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
for seed in range(n_seeds):
    local_data = list_metrics_data[seed]
    idx_same = np.where((local_data[:,1]==0) & (local_data[:,2]==0))[0]
    axs.scatter(local_data[idx_same,6], local_data[idx_same,3],s=5,alpha=0.1)
axs.set_xlabel('speed [m/s]'), axs.set_ylabel('stride duration [s]')
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_stride_duration_raw.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_stride_duration_raw.svg'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers,np.nanmedian(matrix_length,axis=0), color='k',lw=2)
axs.fill_between(bins_centers, np.nanpercentile(matrix_length,axis=0,q=25),np.nanpercentile(matrix_length,axis=0,q=75),color='k',alpha=0.5)
axs.set_xlabel('speed [m/s]'), axs.set_ylabel('stride length [m]')
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_stride_length_binned.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_stride_length_binned.svg'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers,np.nanmedian(matrix_duration,axis=0), color='k',lw=2)
axs.fill_between(bins_centers, np.nanpercentile(matrix_duration,axis=0,q=25),np.nanpercentile(matrix_duration,axis=0,q=75),color='k',alpha=0.5)
axs.set_xlabel('speed [m/s]'), axs.set_ylabel('stride duration [s]')
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_stride_duration_binned.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_stride_duration_binned.svg'),bbox_inches='tight')


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers,np.nanmedian(matrix_length_front,axis=0), color='k',lw=2)
axs.fill_between(bins_centers, np.nanpercentile(matrix_length_front,axis=0,q=25),np.nanpercentile(matrix_length_front,axis=0,q=75),color='k',alpha=0.5)
axs.set_xlabel('speed [m/s]'), axs.set_ylabel('step length [m]')
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_frontstep_length_binned.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_frontstep_length_binned.svg'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers,np.nanmedian(matrix_duration_front,axis=0), color='k',lw=2)
axs.fill_between(bins_centers, np.nanpercentile(matrix_duration_front,axis=0,q=25),np.nanpercentile(matrix_duration_front,axis=0,q=75),color='k',alpha=0.5)
axs.set_xlabel('speed [m/s]'), axs.set_ylabel('step duration [s]')
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_frontstep_duration_binned.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_frontstep_duration_binned.svg'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(5,5),subplot_kw={'projection':'polar'})
axs.plot(scipy.stats.circmean(matrix_relative_time[:,:,0],low=0,high=2*np.pi,nan_policy='omit',axis=0), bins_centers, color='k')
axs.plot(scipy.stats.circmean(matrix_relative_time[:,:,1],low=0,high=2*np.pi,nan_policy='omit',axis=0), bins_centers, color='r')
axs.plot(scipy.stats.circmean(matrix_relative_time[:,:,2],low=-np.pi,high=np.pi,nan_policy='omit',axis=0), bins_centers, color='b')
axs.fill_betweenx(bins_centers, scipy.stats.circmean(matrix_relative_time[:,:,0],low=-np.pi,high=np.pi,nan_policy='omit',axis=0)+scipy.stats.circstd(matrix_relative_time[:,:,0],low=-np.pi,high=np.pi,nan_policy='omit',axis=0),
scipy.stats.circmean(matrix_relative_time[:,:,0],low=-np.pi,high=np.pi,nan_policy='omit',axis=0)-scipy.stats.circstd(matrix_relative_time[:,:,0],low=-np.pi,high=np.pi,nan_policy='omit',axis=0),color='k',alpha=0.5)
axs.fill_betweenx(bins_centers, scipy.stats.circmean(matrix_relative_time[:,:,1],low=-np.pi,high=np.pi,nan_policy='omit',axis=0)+scipy.stats.circstd(matrix_relative_time[:,:,1],low=-np.pi,high=np.pi,nan_policy='omit',axis=0),
scipy.stats.circmean(matrix_relative_time[:,:,1],low=-np.pi,high=np.pi,nan_policy='omit',axis=0)-scipy.stats.circstd(matrix_relative_time[:,:,1],low=-np.pi,high=np.pi,nan_policy='omit',axis=0),color='r',alpha=0.5)
axs.fill_betweenx(bins_centers, scipy.stats.circmean(matrix_relative_time[:,:,2],low=-np.pi,high=np.pi,nan_policy='omit',axis=0)+scipy.stats.circstd(matrix_relative_time[:,:,2],low=-np.pi,high=np.pi,nan_policy='omit',axis=0),
scipy.stats.circmean(matrix_relative_time[:,:,2],low=-np.pi,high=np.pi,nan_policy='omit',axis=0)-scipy.stats.circstd(matrix_relative_time[:,:,2],low=-np.pi,high=np.pi,nan_policy='omit',axis=0),color='b',alpha=0.5)
plt.tight_layout
fig.savefig(os.path.join(output_path,f'{str_group}_coordination.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_coordination.svg'),bbox_inches='tight')

for bin in range(len(bins_centers)):
    fig, axs = plt.subplots(1,1,figsize=(6,3))
    axs.spines[['top','right']].set_visible(False)
    axs.pcolormesh(1-np.squeeze(np.nanmean(matrix_binned_hildebrand[:,bin,:,:],axis=0)),cmap=cm.gray)
    plt.tight_layout()
    fig.savefig(os.path.join(folder_hildebrand,f'hildebrand_bin{bin}.png'),bbox_inches='tight')


fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers,np.nanmedian(matrix_stance_duration,axis=0), color='k', lw=2)
axs.fill_between(bins_centers,np.nanpercentile(matrix_stance_duration,axis=0,q=25), np.nanpercentile(matrix_stance_duration,axis=0,q=75), color='k', alpha=0.5)
axs.set_xlabel('speed'), axs.set_ylabel('stance duration')
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_stance_duration.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_stance_duration.svg'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers,np.nanmedian(matrix_duty_cycle,axis=0), color='k', lw=2)
axs.fill_between(bins_centers,np.nanpercentile(matrix_duty_cycle,axis=0,q=25), np.nanpercentile(matrix_duty_cycle,axis=0,q=75), color='k', alpha=0.5)
axs.set_xlabel('speed'), axs.set_ylabel('duty cycle')
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_duty_cycle.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_duty_cycle.svg'),bbox_inches='tight')

labels = ['0 paw','1 paw','2 diago','2 others','3 paws','4 paws']
cmap = cm.gray(np.linspace(0,0.7,7))
fig, axs = plt.subplots(1,1,figsize=(6,6))
axs.spines[['top','right']].set_visible(False)
axs.stackplot(bins_centers,np.nanmean(matrix_contact_mode,axis=0).T,colors=cmap,labels=labels)
axs.set_ylabel('proportion'), axs.set_xlabel('speed')
axs.legend(frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_contact_mode.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_contact_mode.svg'),bbox_inches='tight')

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(bins_centers,np.nanmedian(matrix_contact_mode[:,:,2],axis=0),color='k',lw=2)
axs.fill_between(bins_centers,np.nanpercentile(matrix_contact_mode[:,:,2],axis=0,q=25),np.nanpercentile(matrix_contact_mode[:,:,2],axis=0,q=75),color='k',alpha=0.5)
axs.set_xlabel('speed'), axs.set_ylabel('diagonal contact [%]')
plt.tight_layout()
fig.savefig(os.path.join(output_path,f'{str_group}_diagonal_contact.png'),bbox_inches='tight')
fig.savefig(os.path.join(output_path,f'{str_group}_diagonal_contact.svg'),bbox_inches='tight')

plt.show()