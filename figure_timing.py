import os, sys 
import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt 



input_path = os.path.join(os.getcwd(),'results')
phasor_data = np.load(os.path.join(input_path,'phase_metrics.npy'))

metrics_data = np.load(os.path.join(input_path,'total_metrics.npy'))

idx_same = np.where((metrics_data[:,1]==0) & (metrics_data[:,2]==0) & (metrics_data[:,3]<1))[0]

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(metrics_data[idx_same,6], metrics_data[idx_same,6]/metrics_data[idx_same, 4], color='k',s=5, alpha=0.5)
axs.set_xlabel('speed'), axs.set_ylabel('stride duration')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.hist(phasor_data[:,-1],bins=30, color='k',alpha=0.5, density=True)
plt.tight_layout()
# plt.show()

print(phasor_data)

bins_limits = np.linspace(1,2,6)
bins_centers = np.diff(bins_limits)[0]/2 + bins_limits[:-1]
list_phasor_data = []
matrix_relative_time = np.zeros((len(bins_centers),3))
for bin in range(len(bins_centers)):
    idx_local = np.where((np.abs(phasor_data[:,-1])>bins_limits[bin]) & (np.abs(phasor_data[:,-1])<bins_limits[bin+1]))[0]
    matrix_relative_time[bin,0] = scipy.stats.circmean(phasor_data[idx_local,3]/phasor_data[idx_local,2], low=0, high=1, nan_policy='omit')
    matrix_relative_time[bin,1] = scipy.stats.circmean(phasor_data[idx_local,4]/phasor_data[idx_local,2], low=0, high=1, nan_policy='omit')
    matrix_relative_time[bin,2] = scipy.stats.circmean(phasor_data[idx_local,5]/phasor_data[idx_local,2], low=-0.5, high=0.5, nan_policy='omit')
matrix_relative_time *= 2*np.pi

print(matrix_relative_time)

fig, axs = plt.subplots(1,1,figsize=(3,3),subplot_kw={"projection":"polar"})
axs.plot(matrix_relative_time[:,0], bins_centers, color='k')
axs.plot(matrix_relative_time[:,1], bins_centers, color='r')
axs.plot(matrix_relative_time[:,2], bins_centers, color='b')
plt.tight_layout()
plt.show()

print(matrix_relative_time)