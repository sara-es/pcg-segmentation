# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:49:42 2023

Liang1997 heart sound segmentation algorithm

@author: hssdwo
"""

import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))
 
import scipy as sp
import numpy as np
from Utilities.create_segmentation_array import create_segmentation_array
from matplotlib import pyplot as plt


def get_wav_and_tsv(patient_name, dataset_dir):
    fs, recording = sp.io.wavfile.read(os.path.join(dataset_dir, patient_name + ".wav"))
    tsv = np.loadtxt(os.path.join(dataset_dir, patient_name + ".tsv"), delimiter="\t")

    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                tsv,
                                                                recording_frequency=4000,
                                                                feature_frequency=4000)
    try:
        return clipped_recording[0], segmentations[0], fs
    except:
        return [], [], 0

def average_envelope(env, f, t = 0.02, overlap = 0.01):
    ave_env = []
    # calculates the average envelope over epochs of time t, with overlap = overlap
    N = np.round(f*t)
    step = np.round(f*overlap)
    step = int(step)
    i = 0
    while i < len(env) - N + 1:
        window = env[i : i + step]
        window_ave = np.sum(window)/N
        ave_env.append(window_ave)
        i = i + step
    
    # deal with the last window? Not clear if the paper does this or not.
    return ave_env

def get_peaks(env, threshold):
        # from paper, looks like a peak is counted as the first point (in time) above the threshold - i.e. not actually at the peak?
       peaks, props = sp.signal.find_peaks(env, height = threshold)
       energies = props['peak_heights']
       return peaks, energies

def get_peak_limits(peaks, a=1, b=1):
    # a and b are parameters that set the upper and lower allowed limits for time
    # intervals between peaks. These are not specified in Liang 1997, so making
    # a reasonable guess here.
    Ts = np.diff(peaks)
    Ts = Ts[Ts>5] #ignore any super short peaks
    Ts_mean = np.mean(Ts)
    Ts_std = np.std(Ts)
    high_limit = Ts_mean + a*Ts_std # to see if there are gaps/missing peaks
    low_limit = Ts_mean - b*Ts_std # to reject peaks too close together (in addition to 50ms 'largest splotted normal sound interval')
    return high_limit, low_limit

def reject_peaks(peaks, energies, low_limit, fs=100, e_thresh = 0.8):
    # low_limit sets the minimum allowable time between peak.
    # fs is the sample rate
    # e_thresh is the minimum allowable ratio for a 'split sound' (not defined in paper)
    split_limit = np.round(0.05 * fs) # 'largest splitted sound interval
    clean_peaks = []
    clean_energies = []
   
    i = 0
    int(i)
    while i < len(peaks)-1:
        peak_i = peaks[i]
        peak_j = peaks[i+1]
        energy_i = energies[i]
        energy_j = energies[i+1]
        
        if (peak_j-peak_i < split_limit): # peaks are part of the same sound
            #compare energy of the two peaks
            if energy_i/energy_j > e_thresh:
                # add energy_i and peak_i to the clean lists
                clean_peaks.append(peak_i)
                clean_energies.append(energy_i)
            else:
                # add energy_j and peak_j to the clean lists
                clean_peaks.append(peak_j)
                clean_energies.append(energy_j)
                
            i = i+2 # skip ahead as we have dealt with the ith and jth peak
        # todo - deal with 3 peaks close together.
        
        elif (peak_j-peak_i < low_limit): # peaks are too close together
            # n.b. paper says: 'if the last intervalt meets meets certain consistency
            # of every second interval' - I have no idea what this means
            if energy_i/energy_j > 1:
                # add energy_i and peak_i to the clean lists
                clean_peaks.append(peak_i)
                clean_energies.append(energy_i)
            else:
                # add energy_j and peak_j to the clean lists
                clean_peaks.append(peak_j)
                clean_energies.append(energy_j)
        
            i = i+2
        else:
            clean_peaks.append(peak_i)
            clean_energies.append(energy_i)
            i = i+1
            
    return clean_peaks, clean_energies

def recover_peaks(peaks, energies, env, low_limit, high_limit, threshold = 0.7, max_iter=4):
    #"When the interval exceeds the high-level time limit, it is assumed that a peak has been lost
    # and the threshold will be decreased by a certain amount. This reduction will be iterated until
    # the losing peaks are picked up or the iteration limit is reached" - Liang 97

    Ts = np.diff(peaks)
    Tgap = Ts[Ts>high_limit]
    i=0
    
    while((i < max_iter) & (Tgap.size != 0)):
        # try a more sensitive peak threshold
        threshold = threshold - (i+1)*0.05
        peaks_a, energies_a = get_peaks(env, threshold)
        
        # consider only the new peaks
        mask = ~np.isin(peaks_a,peaks) # nb. note that this is NOT(isin)
        peaks_a = peaks_a[mask]
        energies_a = energies_a[mask]
        
        # do any of these new peaks fit in the gaps? if so, append to the list of peaks + energies
        for j,this_peak in enumerate(peaks_a):
            p_mask = peaks_a < this_peak
            idx = (sum(p_mask))
            if this_peak > peaks[idx] & this_peak < peaks[idx] + high_limit:
                peaks.append(this_peak)
                energies.append(energies_a[j])
        
        # reorder in peak index order
        sort_index = np.argsort(peaks)
        peaks = [peaks[i] for i in sort_index]
        energies = [energies[i] for i in sort_index]
        
        # "Here the above criteria to eliminate the extra peaks are used again to delete all “extra” peaks picked up
        # in the procedure of finding lost ones." - Liang 97
        for k in range(3):
            peaks, energies = reject_peaks(peaks, energies, low_limit)
            
        
        
        # update time gaps array and iteration
        Ts = np.diff(peaks)
        Tgap = Ts[Ts>high_limit]
        i = i+1
    
    return peaks, energies

def assign_s1s2(peaks, energies, c1=0.15, c2=0.3):
# (1) largest interval is the diastolic period, (2) systolic period is relatively constant
# c1, c2 are percentage tolerance in sys and dias periods
# Liang et al. contains a 'pthr' parameter in table 2, but this is not mentioned in the text

# it's not at all clear what the paper does.
# ignore outliers (probably still missing adjacent peaks)
        
    s1_idx = []
    s2_idx = []
    other_idx = []
    

    #select largest non-outlier and assign as diastolic period
    #select preceding Ts as systolic period 
    Ts = np.diff(peaks)
    Ts_mean = np.mean(Ts)
    Ts_std = np.std(Ts)
    # max_interval = np.max(Ts[Ts<(Ts_mean+2*Ts_std)])
    # idx = np.where(Ts == max_interval)
    # s2_idx.append(idx)
    # s1_idx.append(idx-1)
    
    # #from longest Ts, look backwards in pairs
    # i = idx-2
    # while i > 0:
    #         t2 = Ts[i]
    #         t1 = Ts[i-1]
    #         #assign s2 if within tolerance
    #         if t2 > 
                
    #         if t2>t1:
    #             if (t1 > np.mean(s1_idx) - np.std(s1_idx)) & t1 < np.mean 
    #         else
    
    s1_idx = []
    s2_idx = []
    
    if Ts[1] > Ts[0]:
        s2_idx.append(1)
        s1_idx.append(0)
        prev = 2
    else:
        s2_idx.append(0)
        s1_idx.append(1)
        prev = 1
        
    s1_T = Ts[s1_idx]
    s2_T = Ts[s2_idx]
    
    i = 2
    while i < Ts.size:
        if ((prev == 1) & (Ts[i]>=Ts[i-1])):
            s2_idx.append(i)
            prev = 2
        elif ((prev == 2) & (Ts[i]<=Ts[i-1])):
            s1_idx.append(i)
            prev = 1           
        i = i+1
    
    print(Ts)
    print(s1_idx)
    print(s2_idx)
    
    s1_locs = [peaks[i] for i in s1_idx]
    s2_locs = [peaks[i] for i in s2_idx]
    other_locs = []
    return s1_locs, s2_locs, other_locs

def generate_pcg_envelope(wav, fs, t = 0.02, overlap = 0.01):
    
    wav = wav / np.max(np.abs(wav)) # normalise PCG w.r.t to maximum of the signal
    wav[wav==0] = 1.e-10 # add a small number to any zeros so that logs don't break later
    s_env = -np.square(wav) * np.log(np.square(wav)) # Get the shannon energy envelope

    # Calculate the average shannon energy over n-seconds, with m-seconds overlap
    # n.b. this effectively downsamples to 100Hz
    ave_s_env = average_envelope(s_env,fs, t, overlap)
    ave_s_env = (ave_s_env - np.mean(ave_s_env)) / np.std(ave_s_env) # Normalised average shannon energy envelope
    return ave_s_env

def find_pcg_peaks(env, fs, threshold = 0.7):
    # This takes the shannon entry envelope as input, and outputs the indices of the onset of S1, S2, and any unidentified peaks
    # It takes parameter threshold, threshold = 0.5. The default value is set based on eye-balling the paper.
    
    # 6. Threshold to get candidate peaks
    peaks, energies = get_peaks(env, threshold)
    
    
    # 7. peak rejection
    # "The low-level time limit and high-level time limit, which are used for deleting extra peaks and
    # finding lost sounds respectively, are computed for each recording based on the mean value and standard
    # deviation of these intervals" - Liang et al.
    high_limit, low_limit = get_peak_limits(peaks)
    
    # running peak rejection multiple times is an addition to Liang. The original description does
    # not consider n>2 peaks in a short time window
    for i in range(2):
        peaks, energies = reject_peaks(peaks, energies, low_limit)
    
    peaks, energies = recover_peaks(peaks, energies, env, low_limit, high_limit)
    plt.plot(env)
    plt.scatter(peaks,energies)
    plt.show()
    s1_idx, s2_idx, other_idx = assign_s1s2(peaks, energies)

    return s1_idx, s2_idx, other_idx


# # 0. get some dummy data to test
# dataset_dir = 'C:\\Users\\hssdwo\\Documents\\Physionet_2022\\smoke_data'
# files = ["2530_AV"]#, "2530_MV", "2530_PV", "2530_TV"]

# for file in files:
#     wav, true_seg, fs = get_wav_and_tsv(file, dataset_dir)

# ave_s_env = generate_pcg_envelope(wav, fs)

# # 6. find pcg peaks
# s1_locs, s2_locs, other_locs = find_pcg_peaks(ave_s_env, fs)


