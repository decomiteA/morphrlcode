import numpy as np 
import os, sys 
import copy
import scipy
import scipy.signal, scipy.stats
import scipy.io as spio
from tqdm import tqdm
import matplotlib.pyplot as plt 
import scipy.signal as signal
import scipy as sp
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

def fill_missing_7markers(Y, kind='linear'):
    """
    Fills missing values independently along each dimension after the first 
    With error handling for tail end
    """
    initial_shape = Y.shape
    Y = Y.reshape((initial_shape[0], -1))
    for i in range(Y.shape[-1]-2):
        y = Y[:, i]
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        Y[:, i] = y
    Y = Y.reshape(initial_shape)
    return Y

def fill_missing(Y, kind="linear"):
    """
    Fills missing values independently along each dimension after the first.
    """
    initial_shape = Y.shape
    Y = Y.reshape((initial_shape[0], -1))
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        Y[:, i] = y
    Y = Y.reshape(initial_shape)
    return Y


def findHighSpeed(time_vector, head_velocity):
    """
    Finds the indices of the high speed movements 
    """
    count, threshold = 0, 0.1
    idx_above_threshold = np.where(head_velocity>threshold)[0]
    idx_below_threshold = np.where(head_velocity<-threshold)[0]
  
    bool_tmp = np.zeros_like(time_vector)
    bool_tmp[idx_above_threshold] = 1
    bool_tmp[idx_below_threshold] = -1
    timings = np.zeros((1,2))

    for ii in range(len(bool_tmp)):
        if (bool_tmp[ii]!=0 and count==0):
            tmp = np.where(bool_tmp[ii:]!=bool_tmp[ii])[0]
            if (len(tmp)!=0):
                end_idx = tmp[0]
                if np.ptp(bool_tmp[ii:ii+end_idx])!=0:
                    continue
            else:
                end_idx = -1
            timings = np.vstack((timings, np.hstack((ii,ii+1+end_idx))))
            count = end_idx
        elif count > 0:
            count -= 1
    return timings


def compute_velocity_markers(input_data, framerate=120):
    """
    Computes the velocity of the markers contained in the input_data
    """
    output_data = np.zeros((input_data.shape[0], input_data.shape[1], 4))
    output_data[:,:,:2] = copy.deepcopy(input_data[:,:,:2])
    dt = 1/framerate
    tmp_data = np.zeros((output_data.shape[0], output_data.shape[1], 2))
    tmp_data[2:-3,:,:2] = (-output_data[4:-1,:,:2] + 8*output_data[3:-2,:,:2] - 8*output_data[1:-4,:,:2] + output_data[0:-5,:,:2]) / (12*dt)
    output_data[:,:,2:4] = tmp_data
    return output_data

def get_foot_contact(raw_data,framerate=120):
    """
    Extracts the foot contact matrix for the raw kinematics data. 
    
    For each leg, it computes the fore-aft difference in position between the foot and the nose marker. 
    It identifies the minima and maxima of that difference, respectively corresponding to toe off and heel strike.
    The foot is in contact for the time interval between the heel strike and the toe off. The contact location is defined as the average position of the foot marker during that time interval.
    There is a cleaning mechanism that refuse to identify a contact if another one has been detected less than 7 timessteps before (corresponding to 0.05s).
    """

    # Detect movements 
    local_data = copy.deepcopy(raw_data)
    foot_contact_matrix = np.zeros((2, 4, local_data.shape[-1]))
     
    bool_up = (np.nanmean(local_data[2,0,:]) > 0).astype(int)
    diff_leg1 = local_data[0,1,:] - local_data[0,0,:]
    diff_leg2 = local_data[0,2,:] - local_data[0,0,:]
    diff_leg3 = local_data[0,3,:] - local_data[0,0,:]
    diff_leg4 = local_data[0,4,:] - local_data[0,0,:]


    max_leg1, _ = signal.find_peaks(diff_leg1, distance=framerate/10)
    max_leg2, _ = signal.find_peaks(diff_leg2, distance=framerate/10)
    max_leg3, _ = signal.find_peaks(diff_leg3, distance=framerate/10)
    max_leg4, _ = signal.find_peaks(diff_leg4, distance=framerate/10)

    min_leg1, _ = signal.find_peaks(-diff_leg1, distance=framerate/10)
    min_leg2, _ = signal.find_peaks(-diff_leg2, distance=framerate/10)
    min_leg3, _ = signal.find_peaks(-diff_leg3, distance=framerate/10)
    min_leg4, _ = signal.find_peaks(-diff_leg4, distance=framerate/10)

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
            min_look_back = min(5, len_before_time)
            min_look_after = min(5, len_after_time)
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


def extract_phasor_metrics(input_raw, input_foot, input_video, framerate=120, leg_id=0):
    """
    Extracts the relative contact timing information
    Returns a Nx7 matrix where each row contains in order 
    - animal id 
    - leg id (used to define the gait cycle)
    - front right contact
    - front left contact 
    - hind right contact 
    - hind left contact 
    - average velocity during the gait cycle
    """
    output_matrix = np.zeros((1,7))
    idx_nans = np.where(np.isnan(input_raw[0,0,:]))[0]
    for ii in tqdm(range(len(idx_nans)-1)):
        idx_begin = idx_nans[ii]+1
        idx_end = idx_nans[ii+1]
        len_bout = idx_end - idx_begin
        for time in range(1,len_bout):
            if (input_foot[0,leg_id,idx_begin+time-1]==0) and (input_foot[0,leg_id,idx_begin+time]!=0): # identifies a front right to front right contact 
                tmp = foot_contact_detection(input_foot[:,:,idx_begin+time:idx_end])
                idx_leg0 = np.where(tmp[:,0]==1)[0]
                idx_leg1 = np.where(tmp[:,1]==1)[0]
                idx_leg2 = np.where(tmp[:,2]==1)[0]
                idx_leg3 = np.where(tmp[:,3]==1)[0]
                if (len(idx_leg0)!=0) and (len(idx_leg1)!=0) and (len(idx_leg2)!=0) and (len(idx_leg3)!=0):
                    vec_idx = [idx_leg0[0], idx_leg1[0], idx_leg2[0], idx_leg3[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], leg_id, 1/framerate*idx_leg0[0], 1/framerate*idx_leg1[0], 1/framerate*idx_leg2[0], 1/framerate*idx_leg3[0],np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+vec_idx[leg_id]+1])])))
 
    return output_matrix[1:,:]


def extract_metrics(input_raw, input_foot, input_video, framerate=120):
    """
    Extracts the metrics for the shank3 mouse data 
    """
    output_matrix = np.zeros((1,8))
    idx_nans = np.where(np.isnan(input_raw[0,0,:]))[0] # this is not the number of videos ...
    for ii in tqdm(range(len(idx_nans)-1)):
        idx_begin = idx_nans[ii]+1
        idx_end = idx_nans[ii+1]
        len_bout = idx_end - idx_begin
        for time in range(1,len_bout):
            if (input_foot[0,0,idx_begin+time-1]==0) and (input_foot[0,0,idx_begin+time]!=0):
                init_pos_x = input_foot[0,0,idx_begin+time]
                init_pos_y = input_foot[1,0,idx_begin+time]
                tmp = foot_contact_detection(input_foot[:,:,idx_begin+time:idx_end])
                idx_leg0 = np.where(tmp[:,0]==1)[0]
                idx_leg1 = np.where(tmp[:,1]==1)[0]
                idx_leg2 = np.where(tmp[:,2]==1)[0]
                idx_leg3 = np.where(tmp[:,3]==1)[0]
                if len(idx_leg0)!=0:
                    final_pos_x = input_foot[0,0,idx_begin+time+1+idx_leg0[0]]
                    final_pos_y = input_foot[1,0,idx_begin+time+1+idx_leg0[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 0,0,1/framerate*idx_leg0[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg0[0]+1]), idx_begin+time])))
                if len(idx_leg1)!=0:
                    final_pos_x = input_foot[0,1,idx_begin+time+1+idx_leg1[0]]
                    final_pos_y = input_foot[1,1,idx_begin+time+1+idx_leg1[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 0,1,1/framerate*idx_leg1[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg1[0]+1]), idx_begin+time])))
                if len(idx_leg2)!=0:
                    final_pos_x = input_foot[0,2,idx_begin+time+1+idx_leg2[0]]
                    final_pos_y = input_foot[1,2,idx_begin+time+1+idx_leg2[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 0,2,1/framerate*idx_leg2[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg2[0]+1]), idx_begin+time])))
                if len(idx_leg3)!=0:
                    final_pos_x = input_foot[0,3,idx_begin+time+1+idx_leg3[0]]
                    final_pos_y = input_foot[1,3,idx_begin+time+1+idx_leg3[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 0,3,1/framerate*idx_leg3[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg3[0]+1]), idx_begin+time])))
            if (input_foot[0,1,idx_begin+time-1]==0) and (input_foot[0,1,idx_begin+time]!=0):
                init_pos_x = input_foot[0,1,idx_begin+time]
                init_pos_y = input_foot[1,1,idx_begin+time]
                tmp = foot_contact_detection(input_foot[:,:,idx_begin+time:idx_end])
                idx_leg0 = np.where(tmp[:,0]==1)[0]
                idx_leg1 = np.where(tmp[:,1]==1)[0]
                idx_leg2 = np.where(tmp[:,2]==1)[0]
                idx_leg3 = np.where(tmp[:,3]==1)[0]
                if len(idx_leg0)!=0:
                    final_pos_x = input_foot[0,0,idx_begin+time+1+idx_leg0[0]]
                    final_pos_y = input_foot[1,0,idx_begin+time+1+idx_leg0[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 1,0,1/framerate*idx_leg0[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg0[0]+1]), idx_begin+time])))
                if len(idx_leg1)!=0:
                    final_pos_x = input_foot[0,1,idx_begin+time+1+idx_leg1[0]]
                    final_pos_y = input_foot[1,1,idx_begin+time+1+idx_leg1[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 1,1,1/framerate*idx_leg1[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg1[0]+1]), idx_begin+time])))
                if len(idx_leg2)!=0:
                    final_pos_x = input_foot[0,2,idx_begin+time+1+idx_leg2[0]]
                    final_pos_y = input_foot[1,2,idx_begin+time+1+idx_leg2[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 1,2,1/framerate*idx_leg2[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg2[0]+1]), idx_begin+time])))
                if len(idx_leg3)!=0:
                    final_pos_x = input_foot[0,3,idx_begin+time+1+idx_leg3[0]]
                    final_pos_y = input_foot[1,3,idx_begin+time+1+idx_leg3[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 1,3,1/framerate*idx_leg3[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg3[0]+1]), idx_begin+time])))
            if (input_foot[0,2,idx_begin+time-1]==0) and (input_foot[0,2,idx_begin+time]!=0):
                init_pos_x = input_foot[0,2,idx_begin+time]
                init_pos_y = input_foot[1,2,idx_begin+time]
                tmp = foot_contact_detection(input_foot[:,:,idx_begin+time:idx_end])
                idx_leg0 = np.where(tmp[:,0]==1)[0]
                idx_leg1 = np.where(tmp[:,1]==1)[0]
                idx_leg2 = np.where(tmp[:,2]==1)[0]
                idx_leg3 = np.where(tmp[:,3]==1)[0]
                if len(idx_leg0)!=0:
                    final_pos_x = input_foot[0,0,idx_begin+time+1+idx_leg0[0]]
                    final_pos_y = input_foot[1,0,idx_begin+time+1+idx_leg0[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 2,0,1/framerate*idx_leg0[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg0[0]+1]), idx_begin+time])))
                if len(idx_leg1)!=0:
                    final_pos_x = input_foot[0,1,idx_begin+time+1+idx_leg1[0]]
                    final_pos_y = input_foot[1,1,idx_begin+time+1+idx_leg1[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 2,1,1/framerate*idx_leg1[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg1[0]+1]), idx_begin+time])))
                if len(idx_leg2)!=0:
                    final_pos_x = input_foot[0,2,idx_begin+time+1+idx_leg2[0]]
                    final_pos_y = input_foot[1,2,idx_begin+time+1+idx_leg2[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 2,2,1/framerate*idx_leg2[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg2[0]+1]), idx_begin+time])))
                if len(idx_leg3)!=0:
                    final_pos_x = input_foot[0,3,idx_begin+time+1+idx_leg3[0]]
                    final_pos_y = input_foot[1,3,idx_begin+time+1+idx_leg3[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 2,3,1/framerate*idx_leg3[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg3[0]+1]), idx_begin+time])))
            if (input_foot[0,3,idx_begin+time-1]==0) and (input_foot[0,3,idx_begin+time]!=0):
                init_pos_x = input_foot[0,3,idx_begin+time]
                init_pos_y = input_foot[1,3,idx_begin+time]
                tmp = foot_contact_detection(input_foot[:,:,idx_begin+time:idx_end])
                idx_leg0 = np.where(tmp[:,0]==1)[0]
                idx_leg1 = np.where(tmp[:,1]==1)[0]
                idx_leg2 = np.where(tmp[:,2]==1)[0]
                idx_leg3 = np.where(tmp[:,3]==1)[0]
                if len(idx_leg0)!=0:
                    final_pos_x = input_foot[0,0,idx_begin+time+1+idx_leg0[0]]
                    final_pos_y = input_foot[1,0,idx_begin+time+1+idx_leg0[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 3,0,1/framerate*idx_leg0[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg0[0]+1]), idx_begin+time])))
                if len(idx_leg1)!=0:
                    final_pos_x = input_foot[0,1,idx_begin+time+1+idx_leg1[0]]
                    final_pos_y = input_foot[1,1,idx_begin+time+1+idx_leg1[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 3,1,1/framerate*idx_leg1[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg1[0]+1]), idx_begin+time])))
                if len(idx_leg2)!=0:
                    final_pos_x = input_foot[0,2,idx_begin+time+1+idx_leg2[0]]
                    final_pos_y = input_foot[1,2,idx_begin+time+1+idx_leg2[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 3,2,1/framerate*idx_leg2[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg2[0]+1]), idx_begin+time])))
                if len(idx_leg3)!=0:
                    final_pos_x = input_foot[0,3,idx_begin+time+1+idx_leg3[0]]
                    final_pos_y = input_foot[1,3,idx_begin+time+1+idx_leg3[0]]
                    output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin+time+1][0], 3,3,1/framerate*idx_leg3[0],np.abs(final_pos_x-init_pos_x),final_pos_y-init_pos_y,np.nanmean(input_raw[2,0,idx_begin+time+1:idx_begin+time+idx_leg3[0]+1]), idx_begin+time])))
    return output_matrix[1:,:]


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


def align_gait_segmented_data(input_data,input_self,output_data):
    """
    Rotates the input data so that each entry is pointing in the same direction
    """

    input_flipped = copy.deepcopy(input_data)
    input_self_flipped = copy.deepcopy(input_self)
    output_flipped = copy.deepcopy(output_data)
    idx_flip = np.where(np.nanmean(input_data[:,:,2],axis=1)<0)[0]
    input_flipped[idx_flip,:,:] = - input_flipped[idx_flip,:,:]
    input_self_flipped[idx_flip,:,:] = - input_self[idx_flip,:,:]
    col_flip = [0,1,3,4]
    for col in col_flip:
        output_flipped[idx_flip,col] = -output_flipped[idx_flip,col]

    return input_flipped, input_self_flipped ,output_flipped


def get_io_time_model_cycle_fr(input_foot, input_raw, line, output_metrics):
    """
    Computes the inputs and outputs at the gait cycle level for the mouse data to perform the foot placement analysis
    """
    idx_meta = line[-1].astype(int)
    idx_next_nan = np.where(np.isnan(input_foot[0,0,idx_meta:]))[0]
    idx_prev_nan = np.where(np.isnan(np.flip(np.squeeze(input_foot[0,0,:idx_meta]))))[0]
    idx_next_fc0 =  np.where((np.abs(input_foot[0,1,idx_meta+1:idx_meta+idx_next_nan[0]]) - np.abs(input_foot[0,1,idx_meta:idx_meta+idx_next_nan[0]-1]))>0)[0] # detecting the next foot contact of leg 0
    idx_next_fc1 =  np.where((np.abs(input_foot[0,2,idx_meta+1:idx_meta+idx_next_nan[0]]) - np.abs(input_foot[0,2,idx_meta:idx_meta+idx_next_nan[0]-1]))>0)[0] # detecting the next foot contact of leg 1
    idx_prev_fc0 = np.where((np.flip(np.abs(input_foot[0,1,idx_meta-idx_prev_nan[0]:idx_meta-1])) - np.flip(np.abs(input_foot[0,1,idx_meta+1-idx_prev_nan[0]:idx_meta])))<0)[0] # detecting the previous contact of leg 0
    idx_prev_fc1 = np.where((np.flip(np.abs(input_foot[0,2,idx_meta-idx_prev_nan[0]:idx_meta-1])) - np.flip(np.abs(input_foot[0,2,idx_meta+1-idx_prev_nan[0]:idx_meta])))<0)[0] # detecting the previous contact of leg 1
    if ((len(idx_next_fc0)==0) | (len(idx_prev_fc1)==0) | (len(idx_next_fc1)==0) | (len(idx_prev_fc0)==0)):
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
    output_time = np.linspace(0,1,21)
    head_position_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x1)), head_position_x1),-1)
    head_position_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y1)), head_position_y1),-1)
    head_velocity_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x1)), head_velocity_x1),-1)
    head_velocity_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y1)), head_velocity_y1),-1)
    head_position_x2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x2)), head_position_x2),-1)
    head_position_y2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y2)), head_position_y2),-1)
    head_velocity_x2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x2)), head_velocity_x2),-1)
    head_velocity_y2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y2)), head_velocity_y2),-1)
    input_array = np.hstack((head_position_x1_, head_position_y1_, head_velocity_x1_, head_velocity_y1_,head_position_x2_, head_position_y2_, head_velocity_x2_, head_velocity_y2_))

    ###############
    ### OUTPUTS ###
    ###############
    idx_same1 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,2]==1) & (output_metrics[:,-1]==line[-1]))[0]
    if len(idx_same1)==0:
        return input_array, None
    final_position_x_1, final_position_y_1 = input_foot[0,1,idx_meta+idx_next_fc0[0]+2] - ref_position_x, input_foot[1,1,idx_meta+idx_next_fc0[0]+2] - ref_position_y
    time_contact_1 = output_metrics[idx_same1,3][0]

    idx_same2 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,2]==2) & (output_metrics[:,-1]==line[-1]))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2, final_position_y_2 = input_foot[0,2,idx_meta+idx_next_fc1[0]+2] - ref_position_x, input_foot[1,2,idx_meta+idx_next_fc1[0]+2] - ref_position_y
    time_contact_2 = output_metrics[idx_same2,3][0]

    output_array = np.expand_dims(np.array([final_position_x_1, final_position_y_1, time_contact_1, final_position_x_2, final_position_y_2, time_contact_2]),-1)
    return input_array, output_array 


def get_io_time_model_cycle_self(input_foot, input_raw, line, output_metrics):
    """
    Computes the inputs and outputs at the gait cycle level for the mouse data to perform the foot placement analysis
    """
    idx_meta = line[-1].astype(int)
    idx_next_nan = np.where(np.isnan(input_foot[0,0,idx_meta:]))[0]
    idx_prev_nan = np.where(np.isnan(np.flip(np.squeeze(input_foot[0,0,:idx_meta]))))[0]
    idx_next_fc0 =  np.where((np.abs(input_foot[0,1,idx_meta+1:idx_meta+idx_next_nan[0]]) - np.abs(input_foot[0,1,idx_meta:idx_meta+idx_next_nan[0]-1]))>0)[0] # detecting the next foot contact of leg 0
    idx_next_fc1 =  np.where((np.abs(input_foot[0,2,idx_meta+1:idx_meta+idx_next_nan[0]]) - np.abs(input_foot[0,2,idx_meta:idx_meta+idx_next_nan[0]-1]))>0)[0] # detecting the next foot contact of leg 1
    idx_prev_fc0 = np.where((np.flip(np.abs(input_foot[0,1,idx_meta-idx_prev_nan[0]:idx_meta-1])) - np.flip(np.abs(input_foot[0,1,idx_meta+1-idx_prev_nan[0]:idx_meta])))<0)[0] # detecting the previous contact of leg 0
    idx_prev_fc1 = np.where((np.flip(np.abs(input_foot[0,2,idx_meta-idx_prev_nan[0]:idx_meta-1])) - np.flip(np.abs(input_foot[0,2,idx_meta+1-idx_prev_nan[0]:idx_meta])))<0)[0] # detecting the previous contact of leg 1
    if ((len(idx_next_fc0)==0) | (len(idx_prev_fc1)==0) | (len(idx_next_fc1)==0) | (len(idx_prev_fc0)==0)):
        return None, None
    ref_position_x, ref_position_y = input_foot[0,0,idx_meta], input_foot[1,0,idx_meta] 
    time_input1 = np.arange(idx_meta-idx_prev_fc0[0], idx_meta + idx_next_fc0[0]+1)
    time_input2 = np.arange(idx_meta-idx_prev_fc1[0], idx_meta + idx_next_fc1[0]+1)

    ##############
    ### INPUTS ###
    ##############
    foot_position_x1, foot_position_y1 = input_raw[0,2,time_input1]-ref_position_x, input_raw[1,2,time_input1]-ref_position_y
    foot_velocity_x1, foot_velocity_y1 = input_raw[2,2,time_input1], input_raw[3,2,time_input1]
    foot_position_x2, foot_position_y2 = input_raw[0,3,time_input2]-ref_position_x, input_raw[1,3,time_input2]-ref_position_y
    foot_velocity_x2, foot_velocity_y2 = input_raw[2,3,time_input2], input_raw[3,3,time_input2]
    # Interpolation of the inputs 
    output_time = np.linspace(0,1,21)
    foot_position_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_position_x1)), foot_position_x1),-1)
    foot_position_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_position_y1)), foot_position_y1),-1)
    foot_velocity_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_velocity_x1)), foot_velocity_x1),-1)
    foot_velocity_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_velocity_y1)), foot_velocity_y1),-1)
    foot_position_x2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_position_x2)), foot_position_x2),-1)
    foot_position_y2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_position_y2)), foot_position_y2),-1)
    foot_velocity_x2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_velocity_x2)), foot_velocity_x2),-1)
    foot_velocity_y2_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(foot_velocity_y2)), foot_velocity_y2),-1)
    input_array = np.hstack((foot_position_x1_, foot_position_y1_, foot_velocity_x1_, foot_velocity_y1_,foot_position_x2_, foot_position_y2_, foot_velocity_x2_, foot_velocity_y2_))

    ###############
    ### OUTPUTS ###
    ###############
    idx_same1 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,2]==1) & (output_metrics[:,-1]==line[-1]))[0]
    if len(idx_same1)==0:
        return input_array, None
    final_position_x_1, final_position_y_1 = input_foot[0,1,idx_meta+idx_next_fc0[0]+2] - ref_position_x, input_foot[1,1,idx_meta+idx_next_fc0[0]+2] - ref_position_y
    time_contact_1 = output_metrics[idx_same1,3][0]

    idx_same2 = np.where((output_metrics[:,0]==line[0]) & (output_metrics[:,2]==2) & (output_metrics[:,-1]==line[-1]))[0]
    if len(idx_same2)==0:
        return input_array, None
    final_position_x_2, final_position_y_2 = input_foot[0,2,idx_meta+idx_next_fc1[0]+2] - ref_position_x, input_foot[1,2,idx_meta+idx_next_fc1[0]+2] - ref_position_y
    time_contact_2 = output_metrics[idx_same2,3][0]

    output_array = np.expand_dims(np.array([final_position_x_1, final_position_y_1, time_contact_1, final_position_x_2, final_position_y_2, time_contact_2]),-1)
    return input_array, output_array 

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
    idx_next_nan = np.where(np.isnan(input_foot[0,0,idx_meta:]))[0]
    idx_prev_nan = np.where(np.isnan(np.flip(np.squeeze(input_foot[0,0,:idx_meta]))))[0]
    local_foot_data = input_foot[:,:,idx_meta-idx_prev_nan[0]:idx_meta+idx_next_nan[0]] # this line is useless - to be removed
    local_raw_data = input_foot[:,:,idx_meta-idx_prev_nan[0]:idx_meta+idx_next_nan[0]] # this line is useless - to be removed
    idx_next_fc0 =  np.where((np.abs(input_foot[0,0,idx_meta+1:idx_meta+idx_next_nan[0]]) - np.abs(input_foot[0,0,idx_meta:idx_meta+idx_next_nan[0]-1]))>0)[0] # detecting the next foot contact of leg 0
    idx_next_fc1 =  np.where((np.abs(input_foot[0,1,idx_meta+1:idx_meta+idx_next_nan[0]]) - np.abs(input_foot[0,1,idx_meta:idx_meta+idx_next_nan[0]-1]))>0)[0] # detecting the next foot contact of leg 1
    idx_prev_fc0 = np.where((np.flip(np.abs(input_foot[0,0,idx_meta-idx_prev_nan[0]:idx_meta-1])) - np.flip(np.abs(input_foot[0,0,idx_meta+1-idx_prev_nan[0]:idx_meta])))<0)[0] # detecting the previous contact of leg 0
    idx_prev_fc1 = np.where((np.flip(np.abs(input_foot[0,1,idx_meta-idx_prev_nan[0]:idx_meta-1])) - np.flip(np.abs(input_foot[0,1,idx_meta+1-idx_prev_nan[0]:idx_meta])))<0)[0] # detecting the previous contact of leg 1
    idx_next_to0 = np.where((np.abs(input_foot[0,0,idx_meta+1:idx_meta+idx_next_nan[0]]) - np.abs(input_foot[0,0,idx_meta:idx_meta+idx_next_nan[0]-1]))<0)[0] # detecting the next toe-off of leg 0
    if ((len(idx_next_fc0)==0) | (len(idx_prev_fc1)==0)):
        return None, None
    ref_position_x, ref_position_y = input_foot[0,0,idx_meta], input_foot[1,0,idx_meta] 
    time_input1 = np.arange(idx_meta, idx_meta + idx_next_fc0[0]+1)

    ##############
    ### INPUTS ###
    ##############
    head_position_x1, head_position_y1 = input_raw[0,0,time_input1] - ref_position_x, input_raw[1,0,time_input1] - ref_position_y 
    head_velocity_x1, head_velocity_y1 = input_raw[2,0,time_input1], input_raw[3,0,time_input1]
    # Interpolation of the inputs 
    output_time = np.linspace(0,1,11)
    head_position_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_x1)), head_position_x1),-1)
    head_position_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_position_y1)), head_position_y1),-1)
    head_velocity_x1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_x1)), head_velocity_x1),-1)
    head_velocity_y1_ = np.expand_dims(np.interp(output_time, np.linspace(0,1,len(head_velocity_y1)), head_velocity_y1),-1)
    input_array = np.hstack((head_position_x1_, head_position_y1_, head_velocity_x1_, head_velocity_y1_))

    ###############
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
    time_contact_4 = output_metrics[idx_same4[0],3] if len(idx_same4)!=0 else np.nan

    # Contact pattern
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
        idx_diago = np.where(((local_foot_mat[idx_2_feet,0]==1) & (local_foot_mat[idx_2_feet,3]==1)) | ((local_foot_mat[idx_2_feet,1]==1) & (local_foot_mat[idx_2_feet,2]==1)))[0]
        if len(idx_diago)==0:
            n_2_diago = 0
            n_2_nondiago = n_2_foot
        else:
            n_2_diago = len(idx_diago)/len(n_foot_time)
            n_2_nondiago = (len(idx_2_feet) - len(idx_diago))/len(n_foot_time)
    output_array = np.expand_dims(np.array([final_position_x_1, final_position_y_1, time_contact_1, stance_duration_1, time_contact_2, time_contact_3, time_contact_4, n_0_foot, n_1_foot, n_2_diago, n_2_nondiago, n_3_foot, n_4_foot]),-1)
    return input_array, output_array 


def load_total_metrics(input_path):
    """
    This function loads the mouse dataset containing the Cnptnap2 and L7 groups

    Output matrix is a n x 8 matrix where each row contains information about a specific pair of contacts as follows
    - animal id 
    - first contact leg 
    - second contact leg
    - time duration of the contact 
    - fore-aft distance of the contact 
    - lateral distance of the contact 
    - average velocity during the contact 
    - time stamp of the first contact
    """
    total_matrix = spio.loadmat(os.path.join(input_path,'matrix_metrics_notcontrol.mat'))['matrix_metrics']

    return total_matrix


def get_rsquare_matrix_feedback(tot_animal, tot_input_list, tot_output_list, bool_hind, bool_lat):
    """
    Computes the rsquare matrix for the linear prediction of the foot contact location around the nominal 
    """
    n_animal = np.max(tot_animal).astype(int)+1
    rsquare_diagonal = np.zeros((n_animal, 21))
    gains_diagonal = np.zeros((n_animal,21,5))
    for animal in tqdm(range(n_animal)):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list[idx_animal,15,0]))[0]
        local_input = tot_input_list[idx_animal[idx_nan],:,4*bool_hind:4+4*bool_hind]
        local_output = tot_output_list[idx_animal[idx_nan],3*bool_hind+bool_lat]
        # Normalization of the inputs 
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        tmp_vel = np.nanmean(local_input[:,:,2],1)
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(21)
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Normalization of the outputs
        if bool_lat:
            if len(tmp_vel) == 0 or np.all(tmp_vel == tmp_vel[0]):
                local_output = local_output - np.nanmean(local_output)
            else:
                subjectlin = scipy.stats.linregress(tmp_vel, local_output)
                local_output = local_output - (tmp_vel * subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        for time in range(21):
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
    theta_hat = np.linalg.pinv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare, theta_hat

def multilinear_ols_rsquare(X,y):
    theta_hat = np.linalg.pinv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare

def get_rsquare_matrix_self(tot_animal, tot_input_self, tot_output_self, bool_hind, bool_lat):
    """
    Computes the rsquares matrix for the self prediction
    """
    n_animal = np.max(tot_animal).astype(int) + 1
    rsquare_diagonal = np.zeros((n_animal,21))
    for animal in range(n_animal):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_self[idx_animal,15,0]))[0]
        local_input = tot_input_self[idx_animal[idx_nan],:,4*bool_hind:4+4*bool_hind]
        local_output = tot_output_self[idx_animal[idx_nan],bool_lat+3*bool_hind] - np.nanmean(tot_output_self[idx_animal[idx_nan],bool_lat+3*bool_hind])
        print(local_input.shape, local_output.shape)
        for time in range(21):
            tmp_input_ = local_input[:,time,:]
            design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
            # print(design_mat.shape)
            design_mat_y = design_mat
            if design_mat_y.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                rsquare_diagonal[animal,time] = multilinear_ols_rsquare(design_mat_y, local_output)
    
    return rsquare_diagonal
