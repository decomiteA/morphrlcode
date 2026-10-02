import os, sys 
import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt 



input_path = os.path.join(os.getcwd(),'runs','baseline_highfreq_largerange_1','results')
phasor_data = np.load(os.path.join(input_path,'phase_metrics.npy'))

metrics_data = np.load(os.path.join(input_path,'total_metrics.npy'))


print(np.nanpercentile(phasor_data[:,-1],q=25), np.nanpercentile(phasor_data[:,-1],q=75))

bins_limits = np.linspace(0.5,3.0,20)
bins_centers = np.diff(bins_limits)[0]/2 + bins_limits[:-1]
list_phasor_data = []
matrix_relative_time = np.zeros((len(bins_centers),3))
for bin in range(len(bins_centers)):
    idx_local = np.where((np.abs(phasor_data[:,-1])>bins_limits[bin]) & (np.abs(phasor_data[:,-1])<bins_limits[bin+1]))[0]
    print(len(idx_local))
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
fig.savefig('polar.png')