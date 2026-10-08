import os, sys 
import copy, pickle
import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt 
from utils_analysis import *

output_path = os.path.join('ResultsFigures')
n_seeds = 1
tot_rsquare_body, tot_rsquare_self, tot_avg_speed = np.zeros((1,11)), np.zeros((1,11)), np.zeros((1,1))
for seed in range(n_seeds):
    input_path = os.path.join(os.getcwd(),'runs',f'baseline_lowrange_w5_seed_{seed+1}','results')

    with open(os.path.join(input_path,'list_animal.pkl'),'rb') as f1:
        list_animal = pickle.load(f1)
    with open(os.path.join(input_path,'list_input_body.pkl'),'rb') as f1:
        list_input_body = pickle.load(f1)
    with open(os.path.join(input_path,'list_input_self.pkl'),'rb') as f2:
        list_input_self = pickle.load(f2)
    with open(os.path.join(input_path,'list_output.pkl'),'rb') as f3:
        list_output = pickle.load(f3)

    tot_animal_data = copy.deepcopy(list_animal[0])
    tot_input_body = copy.deepcopy(list_input_body[0])
    tot_input_self = copy.deepcopy(list_input_self[0])
    tot_output = copy.deepcopy(list_output[0])



    for run in range(1,len(list_input_body)):       
        # fig, axs = plt.subplots(1,1,figsize=(20,3))
        # axs.spines[['top','right']].set_visible(False)
        # for ii in range(list_input_body[run].shape[0]):
        #     axs.plot(list_input_body[run][ii,:,4],list_input_body[run][ii,:,5], color='k',lw=1,alpha=0.5)
        #     axs.plot(list_input_self[run][ii,:,4],list_input_self[run][ii,:,5], color='b',lw=1,alpha=0.5)
        # axs.plot(np.nanmedian(list_input_body[run][:,:,4],axis=0), np.nanmedian(list_input_body[run][:,:,5],axis=0), color='r',lw=2)
        # axs.scatter(list_output[run][:,3],list_output[run][:,4],color='k',s=5,alpha=0.5)
        # axs.scatter(list_output[run][:,0],list_output[run][:,1],color='r',s=5,alpha=0.5)

        # plt.tight_layout()
        

        # fig, axs = plt.subplots(1,1,figsize=(3,3))
        # axs.spines[['top','right']].set_visible(False)
        # axs.scatter(list_input_self[run][:,25,5], list_output[run][:,4], color='k',s=5,alpha=.5)
        # axs.scatter(list_input_self[run][:,37,5], list_output[run][:,4], color='r',s=5,alpha=.5)
        # axs.scatter(list_input_self[run][:,48,5], list_output[run][:,4], color='b',s=5,alpha=.5)
        # plt.tight_layout()
        # plt.close('all')


        tot_animal_data = np.concatenate((tot_animal_data, list_animal[run]),axis=0)
        tot_input_body = np.concatenate((tot_input_body, list_input_body[run]),axis=0)
        tot_input_self = np.concatenate((tot_input_self, list_input_self[run]),axis=0)
        tot_output = np.concatenate((tot_output, list_output[run]),axis=0)


    avg_speed = get_avg_speed(tot_animal_data[1:], tot_input_body[1:,:,:])
    print(avg_speed.shape)

    rsquare_lateral, gains_lateral = get_rsquare_matrix_feedback(tot_animal_data[1:], tot_input_body[1:,:,:], tot_output[1:,:], bool_hind=True, bool_lat=True)
    rsquare_body = get_rsquare_matrix_self(tot_animal_data[1:], tot_input_self[1:,:,:], tot_output[1:,:], bool_hind=True, bool_lat=True)


    tot_rsquare_body = np.concatenate((tot_rsquare_body,rsquare_lateral),axis=0)
    tot_rsquare_self = np.concatenate((tot_rsquare_self,rsquare_body),axis=0)
    tot_avg_speed = np.concatenate((tot_avg_speed, avg_speed),axis=0)


tot_rsquare_body = tot_rsquare_body[1:,:]
tot_rsquare_self = tot_rsquare_self[1:,:]
tot_avg_speed = tot_avg_speed[1:,:]


bins_limits = np.linspace(0.5,2,11)
bins_centers = bins_limits[0:-1] + (np.mean(np.diff(bins_limits))/2)
matrix_body = np.zeros((len(bins_limits), 11))
matrix_self = np.zeros((len(bins_limits), 11))
print(tot_avg_speed.shape, tot_rsquare_body.shape, tot_rsquare_self.shape)
for bin in range(len(bins_centers)):
    idx_local = np.where((avg_speed>bins_limits[bin]) & (avg_speed<bins_limits[bin+1]))[0]
    matrix_body[bin,:] = np.nanmedian(tot_rsquare_body[idx_local,:],axis=0)
    matrix_self[bin,:] = np.nanmedian(tot_rsquare_self[idx_local,:],axis=0)




    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.arange(11),matrix_body[bin,:],'k',lw=2)
    axs.plot(np.arange(11),matrix_self[bin,:],'r',lw=2)
    axs.set_ylim([-0.05,1.05])
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(output_path,f'AAAAAA_{bin}.png'),bbox_inches='tight')


    # diff = rsquare_lateral - rsquare_body
    # ctrl_ampl = np.max(diff, axis=1)

    # print(np.nanmedian(ctrl_ampl), np.nanpercentile(ctrl_ampl, q=25), np.nanpercentile(ctrl_ampl, q=75))

    # fig, axs = plt.subplots(1,1,figsize=(3,3))
    # axs.spines[['top','right']].set_visible(False)
    # axs.plot(np.arange(51), np.nanmedian(rsquare_lateral, axis=0),'k',lw=2,label='body')
    # axs.plot(np.arange(51), np.nanmedian(rsquare_body, axis=0),'r',lw=2,label='baseline')
    # axs.fill_between(np.arange(51), np.nanpercentile(rsquare_lateral, axis=0, q=25), np.nanpercentile(rsquare_lateral, axis=0, q=75),color='k',alpha=0.5)
    # axs.fill_between(np.arange(51), np.nanpercentile(rsquare_body, axis=0, q=25), np.nanpercentile(rsquare_body, axis=0, q=75),color='r',alpha=0.5)
    # axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    # # axs.set_xticks([0.0,2.5,5.0,7.5,10.0]), axs.set_xticklabels(['0','0.25','0.5','0.75','1'])
    # axs.legend(frameon=False)
    # axs.set_ylim([-0.05,1.05])
    # plt.tight_layout()
    # fig.savefig(os.path.join(input_path,'figure_rsquared.png'),bbox_inches='tight')
    # fig.savefig(os.path.join(input_path,'figure_rsquared.svg'),bbox_inches='tight')
