import os, sys 
import copy, pickle
import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt 
from utils_analysis import *

input_path = os.path.join(os.getcwd(),'results')


with open(os.path.join(input_path,'list_input_body.pkl'),'rb') as f1:
    list_input_body = pickle.load(f1)
with open(os.path.join(input_path,'list_input_self.pkl'),'rb') as f2:
    list_input_self = pickle.load(f2)
with open(os.path.join(input_path,'list_output.pkl'),'rb') as f3:
    list_output = pickle.load(f3)

tot_animal_data = np.zeros((list_input_self[0].shape[0],1))
tot_input_body = copy.deepcopy(list_input_body[0])
tot_input_self = copy.deepcopy(list_input_self[0])
tot_output = copy.deepcopy(list_output[0])

for run in range(1,len(list_input_body)):
    tot_animal_data = np.concatenate((tot_animal_data, run*np.ones((list_input_body[run].shape[0],1))),axis=0)
    tot_input_body = np.concatenate((tot_input_body, list_input_body[run]),axis=0)
    tot_input_self = np.concatenate((tot_input_self, list_input_self[run]),axis=0)
    tot_output = np.concatenate((tot_output, list_output[run]),axis=0)


rsquare_lateral, gains_lateral = get_rsquare_matrix_feedback(tot_animal_data, tot_input_body, tot_output, bool_hind=False, bool_lat=True)
rsquare_body = get_rsquare_matrix_self(tot_animal_data, tot_input_self, tot_output, bool_hind=False, bool_lat=True)


print(rsquare_lateral.shape)

diff = rsquare_lateral - rsquare_body
ctrl_ampl = np.max(diff, axis=1)

print(np.nanmedian(ctrl_ampl), np.nanpercentile(ctrl_ampl, q=25), np.nanpercentile(ctrl_ampl, q=75))

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.plot(np.arange(11), np.nanmedian(rsquare_lateral, axis=0),'k',lw=2,label='body')
axs.plot(np.arange(11), np.nanmedian(rsquare_body, axis=0),'r',lw=2,label='baseline')
axs.fill_between(np.arange(11), np.nanpercentile(rsquare_lateral, axis=0, q=25), np.nanpercentile(rsquare_lateral, axis=0, q=75),color='k',alpha=0.5)
axs.fill_between(np.arange(11), np.nanpercentile(rsquare_body, axis=0, q=25), np.nanpercentile(rsquare_body, axis=0, q=75),color='r',alpha=0.5)
axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
axs.set_xticks([0.0,2.5,5.0,7.5,10.0]), axs.set_xticklabels(['0','0.25','0.5','0.75','1'])
axs.legend(frameon=False)
axs.set_ylim([-0.05,1.05])
plt.tight_layout()
plt.show()