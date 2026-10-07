import os, sys, copy
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.signal as signal
import scipy.stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_actions(input_data):
    """
    Extracts the actions from the raw data
    """
    tmp = input_data['act_1'].values
    output_actions = np.zeros((8,len(tmp)))
    for act in range(8):
        output_actions[act,:] = input_data[f'act_{act}'].values
    return output_actions

def reshape_data(input_data):
    """
    Reshapes the input data in the format that allows for proper analysis
    """
    tmp = input_data['foot_1_x'].values
    output_data = np.zeros((2,5,len(tmp)))
    vec_str = ['torso_','foot_1_','foot_2_','foot_3_','foot_4_']
    for marker in range(len(vec_str)):
        output_data[0, marker,:] = input_data[vec_str[marker]+'x'].values
        output_data[1, marker,:] = input_data[vec_str[marker]+'y'].values

    return output_data

def compute_velocity_markers(input_data, framerate=100):
    """
    Computes the velocity of the markers contained in the input_data
    """
    output_data = np.zeros((input_data.shape[0]*2, input_data.shape[1], input_data.shape[2]))
    output_data[:2,:,:] = copy.deepcopy(input_data)
    dt = 1/framerate
    tmp_data = np.zeros((output_data.shape[0]//2, output_data.shape[1], output_data.shape[2]))
    tmp_data[:2,:,2:-3] = (-output_data[:2,:,4:-1] + 8*output_data[:2,:,3:-2] - 8*output_data[:2,:,1:-4] + output_data[:2,:,0:-5]) / (12*dt)
    output_data[2:4,:,:] = tmp_data
    return output_data

def get_foot_contact(input_data):
    """
    Extracts the foot contact matrix from the raw kinematics data
    """

    local_data = copy.deepcopy(input_data)
    foot_contact_matrix = np.zeros((2, 4, local_data.shape[-1]))
     
    bool_up = (np.nanmean(local_data[2,0,:]) > 0).astype(int)
    diff_leg1 = local_data[0,1,:] - local_data[0,0,:]
    diff_leg2 = local_data[0,2,:] - local_data[0,0,:]
    diff_leg3 = local_data[0,3,:] - local_data[0,0,:]
    diff_leg4 = local_data[0,4,:] - local_data[0,0,:]

    b,a = signal.butter(6,0.5,'low')
    diff_leg1 = signal.filtfilt(b,a,diff_leg1)
    diff_leg2 = signal.filtfilt(b,a,diff_leg2)
    diff_leg3 = signal.filtfilt(b,a,diff_leg3)
    diff_leg4 = signal.filtfilt(b,a,diff_leg4)

    max_leg1, _ = signal.find_peaks(diff_leg1, distance=50)
    max_leg2, _ = signal.find_peaks(diff_leg2, distance=50)
    max_leg3, _ = signal.find_peaks(diff_leg3, distance=50)
    max_leg4, _ = signal.find_peaks(diff_leg4, distance=50)

    min_leg1, _ = signal.find_peaks(-diff_leg1, distance=50)
    min_leg2, _ = signal.find_peaks(-diff_leg2, distance=50)
    min_leg3, _ = signal.find_peaks(-diff_leg3, distance=50)
    min_leg4, _ = signal.find_peaks(-diff_leg4, distance=50)

    bool_contact = np.zeros((len(diff_leg1), 4))
    for leg in range(4):
        bool_contact[max_leg1,0], bool_contact[min_leg1,0] = 1, -1
        bool_contact[max_leg2,1], bool_contact[min_leg2,1] = 1, -1
        bool_contact[max_leg3,2], bool_contact[min_leg3,2] = 1, -1
        bool_contact[max_leg4,3], bool_contact[min_leg4,3] = 1, -1
        
    for leg in range(4):
        for time in range(bool_contact.shape[0]):
            cdt1 = (1 if bool_up else -1)
            cdt2 = (-1 if bool_up else 1)
            len_before_time = len(bool_contact[:time,leg])
            len_after_time = len(bool_contact[time:,leg])
            min_look_back = min(1, len_before_time)
            min_look_after = min(1, len_after_time)
            if ((bool_contact[time, leg]==cdt1) & (np.sum(np.abs(bool_contact[time-min_look_back:time,leg]))==0) & (np.sum(np.abs(bool_contact[time+1:time+1+min_look_after,leg]))==0)): 
                idx_next_toeoff = np.where(bool_contact[time:,leg]==cdt2)[0]
                if len(idx_next_toeoff)==0:
                    idx_endc = bool_contact[time:,leg].shape[0]-1
                else:
                    idx_endc = idx_next_toeoff[0]
                x_pos = np.nanmean(local_data[0,leg+1,time:time+idx_endc])
                y_pos = np.nanmean(local_data[1,leg+1,time:time+idx_endc])
                foot_contact_matrix[0,leg,time:time+idx_endc] = x_pos
                foot_contact_matrix[1,leg,time:time+idx_endc] = y_pos
        
    return foot_contact_matrix

def foot_contact_detection(input_data):
    """
    Detects the timings of contact for each individual foot
    """
    foot_contact_matrix = np.zeros((input_data.shape[2], input_data.shape[1]))
    for ii in range(foot_contact_matrix.shape[0]-1):
        for jj in range(foot_contact_matrix.shape[1]):
            if (input_data[0,jj,ii]==0 and input_data[0,jj,ii+1]!=0):
                foot_contact_matrix[ii,jj] = 1
            elif (input_data[0,jj,ii]!=0 and input_data[0,jj,ii+1]==0):
                foot_contact_matrix[ii,jj] = -1

    return foot_contact_matrix

def extract_metrics(input_raw, input_foot, input_video, framerate=100):
    # This has to be updated for the data we are working with here ...
    """
    Extracts the metrics for the simulated locomotion data
    """
    output_matrix = np.zeros((1,8))
    for time in tqdm(range(1,input_raw.shape[-1])):
        if (input_foot[0,0,time-1]==0) and (input_foot[0,0,time]!=0):
            init_pos_x = input_foot[0,0,time]
            init_pos_y = input_foot[1,0,time]
            tmp = foot_contact_detection(input_foot[:,:,time:])
            idx_leg0 = np.where(tmp[:,0]==1)[0]
            idx_leg1 = np.where(tmp[:,1]==1)[0]
            idx_leg2 = np.where(tmp[:,2]==1)[0]
            idx_leg3 = np.where(tmp[:,3]==1)[0]
            if len(idx_leg0)!=0:
                final_pos_x = input_foot[0,0,time+1+idx_leg0[0]]
                final_pos_y = input_foot[1,0,time+1+idx_leg0[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 0,0,1/framerate*idx_leg0[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg0[0]+1]), time])))
            if len(idx_leg1)!=0:
                final_pos_x = input_foot[0,1,time+1+idx_leg1[0]]
                final_pos_y = input_foot[1,1,time+1+idx_leg1[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 0,1,1/framerate*idx_leg1[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg1[0]+1]), time])))
            if len(idx_leg2)!=0:
                final_pos_x = input_foot[0,2,time+1+idx_leg2[0]]
                final_pos_y = input_foot[1,2,time+1+idx_leg2[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 0,2,1/framerate*idx_leg2[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg2[0]+1]), time])))
            if len(idx_leg3)!=0:
                final_pos_x = input_foot[0,3,time+1+idx_leg3[0]]
                final_pos_y = input_foot[1,3,time+1+idx_leg3[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 0,3,1/framerate*idx_leg3[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg3[0]+1]), time])))
        if (input_foot[0,1,time-1]==0) and (input_foot[0,1,time]!=0):
            init_pos_x = input_foot[0,1,time]
            init_pos_y = input_foot[1,1,time]
            tmp = foot_contact_detection(input_foot[:,:,time:])
            idx_leg0 = np.where(tmp[:,0]==1)[0]
            idx_leg1 = np.where(tmp[:,1]==1)[0]
            idx_leg2 = np.where(tmp[:,2]==1)[0]
            idx_leg3 = np.where(tmp[:,3]==1)[0]
            if len(idx_leg0)!=0:
                final_pos_x = input_foot[0,0,time+1+idx_leg0[0]]
                final_pos_y = input_foot[1,0,time+1+idx_leg0[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 1,0,1/framerate*idx_leg0[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg0[0]+1]), time])))
            if len(idx_leg1)!=0:
                final_pos_x = input_foot[0,1,time+1+idx_leg1[0]]
                final_pos_y = input_foot[1,1,time+1+idx_leg1[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 1,1,1/framerate*idx_leg1[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg1[0]+1]), time])))
            if len(idx_leg2)!=0:
                final_pos_x = input_foot[0,2,time+1+idx_leg2[0]]
                final_pos_y = input_foot[1,2,+time+1+idx_leg2[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 1,2,1/framerate*idx_leg2[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg2[0]+1]), time])))
            if len(idx_leg3)!=0:
                final_pos_x = input_foot[0,3,time+1+idx_leg3[0]]
                final_pos_y = input_foot[1,3,time+1+idx_leg3[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 1,3,1/framerate*idx_leg3[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg3[0]+1]), time])))
        if (input_foot[0,2,time-1]==0) and (input_foot[0,2,time]!=0):
            init_pos_x = input_foot[0,2,time]
            init_pos_y = input_foot[1,2,time]
            tmp = foot_contact_detection(input_foot[:,:,time:])
            idx_leg0 = np.where(tmp[:,0]==1)[0]
            idx_leg1 = np.where(tmp[:,1]==1)[0]
            idx_leg2 = np.where(tmp[:,2]==1)[0]
            idx_leg3 = np.where(tmp[:,3]==1)[0]
            if len(idx_leg0)!=0:
                final_pos_x = input_foot[0,0,time+1+idx_leg0[0]]
                final_pos_y = input_foot[1,0,time+1+idx_leg0[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 2,0,1/framerate*idx_leg0[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg0[0]+1]), time])))
            if len(idx_leg1)!=0:
                final_pos_x = input_foot[0,1,time+1+idx_leg1[0]]
                final_pos_y = input_foot[1,1,time+1+idx_leg1[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 2,1,1/framerate*idx_leg1[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg1[0]+1]), time])))
            if len(idx_leg2)!=0:
                final_pos_x = input_foot[0,2,time+1+idx_leg2[0]]
                final_pos_y = input_foot[1,2,time+1+idx_leg2[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 2,2,1/framerate*idx_leg2[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg2[0]+1]), time])))
            if len(idx_leg3)!=0:
                final_pos_x = input_foot[0,3,time+1+idx_leg3[0]]
                final_pos_y = input_foot[1,3,time+1+idx_leg3[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 2,3,1/framerate*idx_leg3[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg3[0]+1]), time])))
        if (input_foot[0,3,time-1]==0) and (input_foot[0,3,time]!=0):
            init_pos_x = input_foot[0,3,time]
            init_pos_y = input_foot[1,3,time]
            tmp = foot_contact_detection(input_foot[:,:,time:])
            idx_leg0 = np.where(tmp[:,0]==1)[0]
            idx_leg1 = np.where(tmp[:,1]==1)[0]
            idx_leg2 = np.where(tmp[:,2]==1)[0]
            idx_leg3 = np.where(tmp[:,3]==1)[0]
            if len(idx_leg0)!=0:
                final_pos_x = input_foot[0,0,time+1+idx_leg0[0]]
                final_pos_y = input_foot[1,0,time+1+idx_leg0[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 3,0,1/framerate*idx_leg0[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg0[0]+1]), time])))
            if len(idx_leg1)!=0:
                final_pos_x = input_foot[0,1,time+1+idx_leg1[0]]
                final_pos_y = input_foot[1,1,time+1+idx_leg1[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 3,1,1/framerate*idx_leg1[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg1[0]+1]), time])))
            if len(idx_leg2)!=0:
                final_pos_x = input_foot[0,2,time+1+idx_leg2[0]]
                final_pos_y = input_foot[1,2,time+1+idx_leg2[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 3,2,1/framerate*idx_leg2[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg2[0]+1]), time])))
            if len(idx_leg3)!=0:
                final_pos_x = input_foot[0,3,time+1+idx_leg3[0]]
                final_pos_y = input_foot[1,3,time+1+idx_leg3[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, 3,3,1/framerate*idx_leg3[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,time+1:time+idx_leg3[0]+1]), time])))
    return output_matrix[1:,:]


def extract_phasor_metrics(input_raw, input_foot, input_video, framerate=100, leg_id=0):
    """
    Extracts the relative contact timing information
    """
    output_matrix = np.zeros((1,7))
    for time in range(1,input_raw.shape[-1]):
        if (input_foot[0,leg_id,time-1]==0) and (input_foot[0,leg_id,time]!=0): # identifies a front right to front right contact 
            tmp = foot_contact_detection(input_foot[:,:,time:])
            idx_leg0 = np.where(tmp[:,0]==1)[0]
            idx_leg1 = np.where(tmp[:,1]==1)[0]
            idx_leg2 = np.where(tmp[:,2]==1)[0]
            idx_leg3 = np.where(tmp[:,3]==1)[0]
            if (len(idx_leg0)!=0) and (len(idx_leg1)!=0) and (len(idx_leg2)!=0) and (len(idx_leg3)!=0):
                vec_idx = [idx_leg0[0], idx_leg1[0], idx_leg2[0], idx_leg3[0]]
                output_matrix = np.vstack((output_matrix, np.array([input_video, leg_id, 1/framerate*idx_leg0[0], 1/framerate*idx_leg1[0], 1/framerate*idx_leg2[0], 1/framerate*idx_leg3[0],np.nanmean(input_raw[2,0,time+1:time+vec_idx[leg_id]+1])])))

    return output_matrix[1:,:]


def get_io_time_model_fr(input_foot, input_raw, line, output_metrics):
    """
    Computes the inputs and outputs for the feedback analyses
    """
    idx_meta = line[-1].astype(int)
    idx_next_fc0 =  np.where((np.abs(input_foot[0,1,idx_meta+1:]) - np.abs(input_foot[0,1,idx_meta:-1]))>0)[0] # detecting the next foot contact of leg 0
    idx_next_fc1 =  np.where((np.abs(input_foot[0,3,idx_meta+1:]) - np.abs(input_foot[0,3,idx_meta:-1]))>0)[0] # detecting the next foot contact of leg 1
    idx_prev_fc0 = np.where((np.flip(np.abs(input_foot[0,1,:idx_meta-1])) - np.flip(np.abs(input_foot[0,1,1:idx_meta])))<0)[0] # detecting the previous contact of leg 0
    idx_prev_fc1 = np.where((np.flip(np.abs(input_foot[0,3,:idx_meta-1])) - np.flip(np.abs(input_foot[0,3,1:idx_meta])))<0)[0] # detecting the previous contact of leg 1
    if ((len(idx_next_fc0)==0) | (len(idx_next_fc1)==0) | (len(idx_prev_fc0)==0) | (len(idx_prev_fc1)==0)):
        return None, None
    ref_position_x, ref_position_y = input_foot[0,0,idx_meta], input_foot[1,0,idx_meta] 
    time_input1 = np.arange(idx_meta-idx_prev_fc0[0], idx_meta + idx_next_fc0[0]+1)
    time_input2 = np.arange(idx_meta-idx_prev_fc1[0], idx_meta + idx_next_fc1[0]+1)
    ##############
    ### INPUTS ###
    ##############
    head_position_x1, head_position_y1 = input_raw[0,0,time_input1] - ref_position_x, input_raw[1,0,time_input1] - ref_position_y 
    head_velocity_x1, head_velocity_y1 = input_raw[2,0,time_input1], input_raw[3,0,time_input1]
    head_position_x2, head_position_y2 = input_raw[0,0,time_input2] - ref_position_x, input_raw[1,0,time_input2] - ref_position_y 
    head_velocity_x2, head_velocity_y2 = input_raw[2,0,time_input2], input_raw[3,0,time_input2]
    # Interpolation of the inputs 
    output_time = np.linspace(0,1,51)
    head_position_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x1)), head_position_x1),-1)
    head_position_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y1)), head_position_y1),-1)
    head_velocity_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x1)), head_velocity_x1),-1)
    head_velocity_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y1)), head_velocity_y1),-1)
    head_position_x2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x2)), head_position_x2),-1)
    head_position_y2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y2)), head_position_y2),-1)
    head_velocity_x2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x2)), head_velocity_x2),-1)
    head_velocity_y2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y2)), head_velocity_y2),-1)
    input_array = np.hstack((head_position_x1_, head_position_y1_, head_velocity_x1_, head_velocity_y1_,
                             head_position_x2_, head_position_y2_, head_velocity_x2_, head_velocity_y2_))

    ###############
    ### OUTPUTS ###
    ###############
    idx_same1 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,2]==1) & (output_metrics[:,-1]==line[-1]))[0]
    if len(idx_same1)==0:
        return input_array, None
    final_position_x_1, final_position_y_1 = input_foot[0,1,idx_meta+idx_next_fc0[0]+2] - ref_position_x, input_foot[1,1,idx_meta+idx_next_fc0[0]+2] - ref_position_y
    time_contact_1 = output_metrics[idx_same1,3][0]

    idx_same2 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,2]==3) & (output_metrics[:,-1]==line[-1]))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2, final_position_y_2 = input_foot[0,3,idx_meta+idx_next_fc1[0]+2] - ref_position_x, input_foot[1,3,idx_meta+idx_next_fc1[0]+2] - ref_position_y
    time_contact_2 = output_metrics[idx_same2,3][0]


    output_array = np.expand_dims(np.array([final_position_x_1, final_position_y_1, time_contact_1, final_position_x_2, final_position_y_2, time_contact_2]),-1)
    return input_array, output_array 

def get_io_time_model_fr_self(input_foot, input_raw, line, output_metrics):
    """
    Computes the inputs and outputs for the feedback analyses
    """
    idx_meta = line[-1].astype(int)

    idx_next_fc0 =  np.where((np.abs(input_foot[0,1,idx_meta+1:]) - np.abs(input_foot[0,1,idx_meta:-1]))>0)[0] # detecting the next foot contact of leg 0
    idx_next_fc1 =  np.where((np.abs(input_foot[0,3,idx_meta+1:]) - np.abs(input_foot[0,3,idx_meta:-1]))>0)[0] # detecting the next foot contact of leg 1
    idx_prev_fc0 = np.where((np.flip(np.abs(input_foot[0,1,:idx_meta-1])) - np.flip(np.abs(input_foot[0,1,1:idx_meta])))<0)[0] # detecting the previous contact of leg 0
    idx_prev_fc1 = np.where((np.flip(np.abs(input_foot[0,3,:idx_meta-1])) - np.flip(np.abs(input_foot[0,3,1:idx_meta])))<0)[0] # detecting the previous contact of leg 1
    if ((len(idx_next_fc0)==0) | (len(idx_next_fc1)==0) | (len(idx_prev_fc0)==0) | (len(idx_prev_fc1)==0)):
        return None, None
    ref_position_x, ref_position_y = input_foot[0,0,idx_meta], input_foot[1,0,idx_meta] 
    time_input1 = np.arange(idx_meta-idx_prev_fc0[0], idx_meta + idx_next_fc0[0]+1)
    time_input2 = np.arange(idx_meta-idx_prev_fc1[0], idx_meta + idx_next_fc1[0]+1)

    ##############
    ### INPUTS ###
    ##############
    foot_position_x1, foot_position_y1 = input_raw[0,2,time_input1] - ref_position_x, input_raw[1,2,time_input1] - ref_position_y 
    foot_velocity_x1, foot_velocity_y1 = input_raw[2,2,time_input1], input_raw[3,2,time_input1]
    foot_position_x2, foot_position_y2 = input_raw[0,4,time_input2] - ref_position_x, input_raw[1,4,time_input2] - ref_position_y 
    foot_velocity_x2, foot_velocity_y2 = input_raw[2,4,time_input2], input_raw[3,4,time_input2]
    # Interpolation of the inputs 
    output_time = np.linspace(0,1,51)
    foot_position_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_position_x1)), foot_position_x1),-1)
    foot_position_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_position_y1)), foot_position_y1),-1)
    foot_velocity_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_velocity_x1)), foot_velocity_x1),-1)
    foot_velocity_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_velocity_y1)), foot_velocity_y1),-1)
    foot_position_x2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_position_x2)), foot_position_x2),-1)
    foot_position_y2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_position_y2)), foot_position_y2),-1)
    foot_velocity_x2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_velocity_x2)), foot_velocity_x2),-1)
    foot_velocity_y2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_velocity_y2)), foot_velocity_y2),-1)
    input_array = np.hstack((foot_position_x1_, foot_position_y1_, foot_velocity_x1_, foot_velocity_y1_,
                             foot_position_x2_, foot_position_y2_, foot_velocity_x2_, foot_velocity_y2_))

    ###############
    ### OUTPUTS ###
    ###############
    idx_same1 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,2]==1) & (output_metrics[:,-1]==line[-1]))[0]
    if len(idx_same1)==0:
        return input_array, None
    final_position_x_1, final_position_y_1 = input_foot[0,1,idx_meta+idx_next_fc0[0]+2] - ref_position_x, input_foot[1,1,idx_meta+idx_next_fc0[0]+2] - ref_position_y
    time_contact_1 = output_metrics[idx_same1,3][0]

    idx_same2 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,2]==3) & (output_metrics[:,-1]==line[-1]))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2, final_position_y_2 = input_foot[0,3,idx_meta+idx_next_fc1[0]+2] - ref_position_x, input_foot[1,3,idx_meta+idx_next_fc1[0]+2] - ref_position_y
    time_contact_2 = output_metrics[idx_same2,3][0]


    output_array = np.expand_dims(np.array([final_position_x_1, final_position_y_1, time_contact_1, final_position_x_2, final_position_y_2, time_contact_2]),-1)
    return input_array, output_array 


def get_rsquare_matrix_feedback(tot_animal, tot_input_list, tot_output_list, bool_hind, bool_lat):
    """
    Computes the rsquare matrix for the linear prediction of the foot contact location around the nominal 
    """


    n_animal = np.max(tot_animal).astype(int)+1
    rsquare_diagonal = np.zeros((n_animal, 51))
    gains_diagonal = np.zeros((n_animal,51,5))
    for animal in tqdm(range(n_animal)):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list[idx_animal,37,0]))[0]
        local_input = tot_input_list[idx_animal[idx_nan],:,4*bool_hind:4+4*bool_hind]
        local_output = tot_output_list[idx_animal[idx_nan],3*bool_hind+bool_lat]
        # Normalization of the inputs 
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        tmp_vel = np.nanmean(local_input[:,:,2],1)
        if local_input.shape[0]==0:
            rsquare_diagonal[animal,:] = np.nan
            gains_diagonal[animal,:] = np.nan
            continue
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(51)
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Normalization of the outputs
        if bool_lat:
            subjectlin = scipy.stats.linregress(tmp_vel, local_output)
            local_output = local_output - (tmp_vel * subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        for time in range(51):
            if bool_lat:
                idx_plot = np.where(local_input[:,time,1]!=0)[0]
            else:
                idx_plot = np.where(local_input[:,time,1]!=0)[0]
            design_mat = np.hstack((np.ones((local_input[idx_plot].shape[0],1)),local_input[idx_plot,time,:]))
            if not bool_lat:
                design_mat_y = design_mat[:,[0,1,2,3,4]]
            else:
                design_mat_y = design_mat[:,[0,1,2,3,4]]
            if design_mat_y.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                a, b = multilinear_ols_rsquare_gains(design_mat_y, local_output[idx_plot])
                pred_output = b @ design_mat_y.T
                gains_diagonal[animal,time,:] = b
                rsquare_diagonal[animal,time] = a #multilinear_ols_rsquare(design_mat_y, local_output[idx_plot])
    
    return rsquare_diagonal, gains_diagonal


def multilinear_ols_rsquare_gains(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare, theta_hat

def multilinear_ols_rsquare(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare

def get_speed(input_metrics):
    output_speed = np.zeros(len(np.unique(input_metrics[:,0])))
    for group in range(len(output_speed)):
        idx_local = np.where((input_metrics[:,0]==group))[0]
        output_speed[group] = np.nanmedian(input_metrics[idx_local,6])
    return output_speed

def get_rsquare_matrix_self(tot_animal, tot_input_self, tot_output_self, bool_hind, bool_lat):
    """
    Computes the rsquares matrix for the self prediction
    """
    n_animal = np.max(tot_animal).astype(int) + 1
    rsquare_diagonal = np.zeros((n_animal,51))
    for animal in range(n_animal):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_self[idx_animal,37,0]))[0]
        local_input = tot_input_self[idx_animal[idx_nan],:,4*bool_hind:4+4*bool_hind]
        local_output = tot_output_self[idx_animal[idx_nan],bool_lat+3*bool_hind] - np.nanmean(tot_output_self[idx_animal[idx_nan],bool_lat+3*bool_hind])
        for time in range(51):
            tmp_input_ = local_input[:,time,:]
            design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
            design_mat_y = design_mat
            if design_mat_y.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                rsquare_diagonal[animal,time] = multilinear_ols_rsquare(design_mat_y, local_output)
    
    return rsquare_diagonal

def get_hildebrand_data(tot_foot_contact_data, tot_metrics):
    """
    Represents the hildebrand data from the contact information and the 
    """
    interpolation_x = np.linspace(0,1,51)
    hildebrand_matrix = np.zeros((1,4, len(interpolation_x)))
    idx_same = np.where((tot_metrics[:,1]==0) & (tot_metrics[:,2]==0))[0] # We grab the cycles we are interested in.
    # For each of those, we compute grab the corresponding contact matrix ... 
    for cycle in range(len(idx_same)):
        idx_begin = tot_metrics[idx_same[cycle],-1].astype(int)
        idx_next_contact = np.where((tot_foot_contact_data[0,0,idx_begin+1:]-tot_foot_contact_data[0,0,idx_begin:-1])>0)[0]
        if len(idx_next_contact)==0:
            continue
        # print(tot_foot_contact_data[0,:,idx_begin:idx_begin+idx_next_contact[0]])
        local_data = (tot_foot_contact_data[0,:,idx_begin:idx_begin+idx_next_contact[0]]!=0).astype(int)
        # Interpolation 
        local_interp = np.zeros((4,len(interpolation_x)))
        for foot in range(4):
            local_interp[foot,:] = np.interp(interpolation_x, np.linspace(0,1,local_data.shape[-1]),local_data[foot,:])
        hildebrand_matrix = np.concatenate((hildebrand_matrix, np.expand_dims(local_interp,axis=0)),axis=0)


    return hildebrand_matrix[1:,:,:]

def get_io_time_model_cycle_front(input_foot, input_raw, line, output_metrics):
    """
    Computes the inputs and outputs at the gait cycle level for the marmoset data
    The outputs contains the following information
    Stride length
    Stride width
    Stride duration
    Stance duration
    Contact timing of the other three legs (relative)
    Percentage of time with 0, 1, 2 diag, 2 non diag, 3 and 4 legs on the floor
    """
    idx_meta = line[-1].astype(int)
    idx_next_fc0 =  np.where((np.abs(input_foot[0,0,idx_meta+1:]) - np.abs(input_foot[0,0,idx_meta:-1]))>0)[0] # detecting the next foot contact of leg 0
    idx_next_fc1 =  np.where((np.abs(input_foot[0,1,idx_meta+1:]) - np.abs(input_foot[0,1,idx_meta:-1]))>0)[0] # detecting the next foot contact of leg 1
    idx_prev_fc0 = np.where((np.flip(np.abs(input_foot[0,0,idx_meta:-1])) - np.flip(np.abs(input_foot[0,0,idx_meta+1:])))<0)[0] # detecting the previous contact of leg 0
    idx_prev_fc1 = np.where((np.flip(np.abs(input_foot[0,1,idx_meta:-1])) - np.flip(np.abs(input_foot[0,1,idx_meta+1:])))<0)[0] # detecting the previous contact of leg 1
    idx_next_to0 = np.where((np.abs(input_foot[0,0,idx_meta+1:]) - np.abs(input_foot[0,0,idx_meta:-1]))<0)[0] # detecting the next toe-off of leg 0
    if ((len(idx_next_fc0)==0) | (len(idx_prev_fc1)==0)):
        return None, None
    ref_position_x, ref_position_y = input_foot[0,0,idx_meta], input_foot[1,0,idx_meta]
    time_input1 = np.arange(idx_meta, idx_meta + idx_next_fc0[0]+1)    ##############
    ### INPUTS ###
    ##############
    head_position_x1, head_position_y1 = input_raw[0,0,time_input1] - ref_position_x, input_raw[1,0,time_input1] - ref_position_y
    head_velocity_x1, head_velocity_y1 = input_raw[2,0,time_input1], input_raw[3,0,time_input1]
    # Interpolation of the inputs
    output_time = np.linspace(0,1,51)
    head_position_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x1)), head_position_x1),-1)
    head_position_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y1)), head_position_y1),-1)
    head_velocity_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x1)), head_velocity_x1),-1)
    head_velocity_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y1)), head_velocity_y1),-1)
    input_array = np.hstack((head_position_x1_, head_position_y1_, head_velocity_x1_, head_velocity_y1_))    ###############
    ### OUTPUTS ###
    ###############
    idx_same1 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,2]==0) & (output_metrics[:,-1]==line[-1]))
    if len(idx_same1)==0:
        return input_array, None
    final_position_x_1, final_position_y_1 = line[4], line[5]
    time_contact_1 = line[3]
    stance_duration_1 = idx_next_to0[0] if len(idx_next_to0)!=0 else np.nan
    idx_same2 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,-1]==line[-1]) & (output_metrics[:,2]==1))[0]
    idx_same3 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,-1]==line[-1]) & (output_metrics[:,2]==2))[0]
    idx_same4 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,-1]==line[-1]) & (output_metrics[:,2]==3))[0]
    time_contact_2 = output_metrics[idx_same2[0],3] if len(idx_same2)!=0 else np.nan
    time_contact_3 = output_metrics[idx_same3[0],3] if len(idx_same3)!=0 else np.nan
    time_contact_4 = output_metrics[idx_same4[0],3] if len(idx_same4)!=0 else np.nan    # Contact pattern
    local_foot_mat = (np.squeeze(input_foot[0,:4,time_input1])!=0).astype(int)
    n_foot_time = np.sum(local_foot_mat, 1)
    n_0_foot = len(np.where(n_foot_time==0)[0])/len(n_foot_time)
    n_1_foot = len(np.where(n_foot_time==1)[0])/len(n_foot_time)
    n_2_foot = len(np.where(n_foot_time==2)[0])/len(n_foot_time)
    n_3_foot = len(np.where(n_foot_time==3)[0])/len(n_foot_time)
    n_4_foot = len(np.where(n_foot_time==4)[0])/len(n_foot_time)
    # Differentiate between 2 diagonals and others
    idx_2_feet = np.where(n_foot_time==2)[0]
    if len(idx_2_feet)==0:
        n_2_diago, n_2_nondiago = 0, 0
    else:
        idx_diago = np.where(((local_foot_mat[idx_2_feet,0]==1) & (local_foot_mat[idx_2_feet,2]==1)) | ((local_foot_mat[idx_2_feet,1]==1) & (local_foot_mat[idx_2_feet,3]==1)))[0]
        if len(idx_diago)==0:
            n_2_diago = 0
            n_2_nondiago = n_2_foot
        else:
            n_2_diago = len(idx_diago)/len(n_foot_time)
            n_2_nondiago = (len(idx_2_feet) - len(idx_diago))/len(n_foot_time)
    output_array = np.expand_dims(np.array([final_position_x_1, final_position_y_1, time_contact_1, stance_duration_1, time_contact_2, time_contact_3, time_contact_4, n_0_foot, n_1_foot, n_2_diago, n_2_nondiago, n_3_foot, n_4_foot]),-1)
    return input_array, output_array
