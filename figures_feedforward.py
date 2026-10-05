import os, sys 
import pickle
import numpy as np 
import scipy.signal
import matplotlib.pyplot as plt 
from utils_analysis import *
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')
str_group = 'baseline_highrange'
output_path = os.path.join(os.getcwd(),'ResultsFigures')
os.makedirs(output_path,exist_ok=True)
folder_hildebrand = os.path.join(output_path,f'hildebrandfigures_{str_group}')
os.makedirs(folder_hildebrand,exist_ok=True)
n_seeds = 8
list_metrics_data, list_phasor_data, list_hildebrand, list_speed = [], [], [], []
for seed in range(n_seeds):
    local_input_path = os.path.join(os.getcwd(),'runs',f'{str_group}_seed_{seed+1}','results')
    

    list_metrics_data.append(np.load(os.path.join(local_input_path,'total_metrics.npy')))
    list_phasor_data.append(np.load(os.path.join(local_input_path,'phase_metrics.npy')))
    list_hildebrand.append(np.load(os.path.join(local_input_path,'hildebrand_data.npy')))
    list_speed.append(get_speed(list_metrics_data[-1]))

bins_limits = np.linspace(0.5,2,11)
bins_centers = bins_limits[0:-1] + (np.mean(np.diff(bins_limits))/2)

matrix_length, matrix_duration = np.zeros((n_seeds,len(bins_centers))), np.zeros((n_seeds,len(bins_centers)))
matrix_length_front, matrix_duration_front = np.zeros((n_seeds,len(bins_centers))), np.zeros((n_seeds,len(bins_centers)))

matrix_binned_hildebrand = np.zeros((n_seeds,len(bins_centers),4,51))

matrix_relative_time = np.zeros((n_seeds, len(bins_centers),3))

for seed in range(n_seeds):
    local_data = list_metrics_data[seed]
    local_speed = list_speed[seed]
    local_hildebrand = list_hildebrand[seed]
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
