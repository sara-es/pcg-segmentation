import numpy as np 
import sys
import scipy as sp
import statistics
import os 

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from Utilities import constants
from Utilities.create_segmentation_array import create_segmentation_array
from SegmentationCNN.Models.Envelope_CNN.DataPreprocessing import DataPreprocessing

SAMPLE_FREQ = 4000
DOWNSAMPLE_FREQ = 50

def downsample_segmentation_array(segmentation_array):
    labels_per_sample = int(SAMPLE_FREQ / DOWNSAMPLE_FREQ)
    downsample_segment = [] 
    for i in range(0, segmentation_array.shape[0], labels_per_sample):
        modal_val = statistics.mode(segmentation_array[i:i+labels_per_sample])
        downsample_segment.append(modal_val)
    return downsample_segment

def upsample_states(original_qt, old_fs, new_fs, new_length):

    original_qt = original_qt.reshape(-1)
    expanded_qt = np.zeros(new_length)

    indices_of_changes =  np.nonzero(np.diff(original_qt))[0]
    indices_of_changes = np.concatenate((indices_of_changes, [original_qt.shape[0] - 1]))

    start_index = 0
    for idx in range(len(indices_of_changes)):
        end_index = indices_of_changes[idx]

        value_at_midpoint = original_qt[end_index]


        expanded_start_index = int(np.round((start_index) / old_fs * new_fs)) + 1
        expanded_end_index = int(np.round((end_index) / old_fs * new_fs))

        if expanded_end_index > new_length:
            expanded_end_index = new_length
        if idx == len(indices_of_changes) - 1:
            expanded_end_index = new_length + 1

        expanded_qt[expanded_start_index - 1:expanded_end_index] = value_at_midpoint
        start_index = end_index

    return expanded_qt

def make_sample_prediction(patches, new_length, window, stride):
    index_options = {key: [] for key in range(new_length)}
    if (window > new_length):
        return np.array(patches[0][:new_length])
    for i in range(len(patches)):
        for j in range(len(patches[i])):

            try: 
                index_options[j+(stride*i)].append(patches[i][j].item())
            except KeyError: 
                break
 
    prediction = np.zeros(new_length)
    for (key, value) in index_options.items():  
        mode = statistics.mode(value)
        if key == 0:
            prediction[key] = mode 
        elif mode != (prediction[key-1] + 1) % 4:
            prediction[key] = prediction[key-1]
        else:
            prediction[key] = mode
    return prediction

def get_true_segmentations(wav, tsv):
    fs, recording = sp.io.wavfile.read(wav)
    tsv = np.loadtxt(tsv, delimiter="\t")
    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                    tsv,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=4000)  


    
    try:
        segmentations = np.array(segmentations[0])
        dp = DataPreprocessing(clipped_recording[0], segmentations, fs, window=64, stride=8)
        dp.extract_segmentation_patches()
        return segmentations, dp.seg_patches
    except:
        return [], []


def extract_segmentation_patches(segmentation_array):
    seg_patches = [] 
    for i in range(0, len(segmentation_array), 8):
        padding = i+64 - len(segmentation_array)
        if i+64 >= len(segmentation_array):
            segmentation_array = np.pad(segmentation_array, pad_width=(0,padding), mode="constant", constant_values=(segmentation_array[-1]))
            patch = segmentation_array[i:i+64]
            seg_patches.append(patch)
            break
        else: 
            patch = segmentation_array[i:i+64]
            seg_patches.append(patch)
    seg_patches = np.array(seg_patches)
    return seg_patches


dataset_dir = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data"
patch_extract_combine_accuracy = []
upsample_accuracy = [] 
for file_ in sorted(os.listdir(dataset_dir)):
    root, extension = os.path.splitext(file_)
    if "50782_MV" in root:
        print("Ignoring")
        continue 
    elif extension == ".wav":
        wav_file = os.path.join(dataset_dir, file_)
        segmentation_file = os.path.join(dataset_dir, root + ".tsv")
        true_segmentations, seg_patches = get_true_segmentations(wav_file, segmentation_file)
        if len(true_segmentations) > 0 : 
            downsampled = np.array(downsample_segmentation_array(true_segmentations))
            recombined = make_sample_prediction(seg_patches-1, len(downsampled), 64,8)+1
            upsampled = upsample_states(recombined, 50, 4000, len(true_segmentations))

            patch_extract_combine = (recombined==downsampled).sum()/len(downsampled)

            patch_extract_combine_accuracy.append((recombined==downsampled).sum()/len(downsampled))
            upsample_accuracy.append((upsampled==true_segmentations).sum()/len(true_segmentations))

print(np.mean(patch_extract_combine_accuracy))    
print(np.mean(upsample_accuracy))
print(np.std(patch_extract_combine_accuracy))
print(np.std(upsample_accuracy))
print("DONE")


