import statistics
import numpy as np 
import sys 
import scipy as sp 
import os 



sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from Utilities import constants
from Utilities.create_segmentation_array import create_segmentation_array
from SegmentationCNN.Models.Envelope_CNN.DataPreprocessing import DataPreprocessing


SAMPLE_FREQ = 4000
DOWNSAMPLE_FREQ = 50
PATCH_SIZE = 64
STRIDE = 8 

def downsample_segmentation_array(segmentation_array):
    labels_per_sample = int(SAMPLE_FREQ / DOWNSAMPLE_FREQ)
    downsample_segment = [] 
    for i in range(0, segmentation_array.shape[0], labels_per_sample):
        modal_val = statistics.mode(segmentation_array[i:i+labels_per_sample])
        downsample_segment.append(modal_val)
    return downsample_segment

def upsample_states(original_qt, old_fs, new_fs, new_length):
    """

    Parameters
    ----------
    original_qt
       The states inferred from the recording features (sampled at old_fs)
    old_fs
        The sampling frequency of the features from which the states were derived
    new_fs
        The desired sampling frequency
    new_length
        The desired length of the new signal

    Returns
    -------
    expanded_qt
        the inferred states resampled to be at frequency new_fs

    """

    original_qt = original_qt.reshape(-1)
    expanded_qt = np.zeros(new_length)

    indices_of_changes =  np.nonzero(np.diff(original_qt))[0]
    indices_of_changes = np.concatenate((indices_of_changes, [original_qt.shape[0] - 1]))

    start_index = 0
    for idx in range(len(indices_of_changes)):
        end_index = indices_of_changes[idx]

        # because division by 2 only has 0.0 and 0.5 as fractional parts, we can use ceil instead of round to stay faithful to MATLAB
        #mid_point = int(np.ceil((end_index - start_index) / 2) + start_index)
        # We don't need value at midpoint, we can just use the value at the start_index

        # value_at_midpoint = original_qt[mid_point]
        value_at_midpoint = original_qt[end_index]
#        if start_index != 0:
#            assert original_qt[start_index + 1] == original_qt[end_index]

        expanded_start_index = int(np.round((start_index) / old_fs * new_fs)) + 1
        expanded_end_index = int(np.round((end_index) / old_fs * new_fs))

        if expanded_end_index > new_length:
            expanded_end_index = new_length
        if idx == len(indices_of_changes) - 1:
            expanded_end_index = new_length + 1

        expanded_qt[expanded_start_index - 1:expanded_end_index] = value_at_midpoint
        start_index = end_index


    # expanded_qt = expanded_qt.reshape(-1, 1)
    return expanded_qt

def make_sample_prediction(patches, new_length):
    index_options = {key: [] for key in range(new_length)}

    if (PATCH_SIZE > new_length):
        return patches[0][:new_length]
    for i in range(len(patches)):
        for j in range(len(patches[i])):
            max_index = (PATCH_SIZE-1)+(8*i)
            if max_index >= new_length:
                shift = max_index - new_length + 1 
                index_options[j+(8*i)-shift].append(patches[i][j])
            else: 
                index_options[j+(8*i)].append(patches[i][j])
    prediction = [statistics.mode(value) for (key, value) in index_options.items()] 
    return np.array(prediction)

def get_true_segmentations(file):
    patientID = file.split(".")[0]
    fs, recording = sp.io.wavfile.read(os.path.join(constants.TRAINING_DATA_PATH_2022 + "training_data/", file))
    tsv = np.loadtxt(constants.TRAINING_DATA_PATH_2022 + "training_data/" + patientID + ".tsv", delimiter="\t")
    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                    tsv,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=4000)  


    dp = DataPreprocessing(clipped_recording[0], segmentations[0], fs)
    x_patches = dp.extract_env_patches()
    y_patches = dp.extract_segmentation_patches()

    return np.array(segmentations[0])


def extract_segmentation_patches(segmentation_array):
    patch_list = [] 
    if len(segmentation_array) < PATCH_SIZE: 
        padding = PATCH_SIZE - len(segmentation_array)
        segmentation_array = np.pad(segmentation_array, pad_width=(0,padding), mode="constant", constant_values=(segmentation_array[-1]))
    for i in range(0, len(segmentation_array), STRIDE):
        if i+PATCH_SIZE >= len(segmentation_array):
            patch = segmentation_array[-PATCH_SIZE:]
            patch_list.append(patch)
            break
        else: 
            patch = segmentation_array[i:i+PATCH_SIZE]
            patch_list.append(patch)
    return patch_list 


file = "14241_MV.wav"
true_segmentations = np.array(get_true_segmentations(file))
# print(true_segmentations)
downsampled = np.array(downsample_segmentation_array(true_segmentations))
patches = extract_segmentation_patches(downsampled)

recombined = make_sample_prediction(patches, len(downsampled))

print((recombined==downsampled).sum())
print(len(downsampled))
upsampled = upsample_states(recombined, 50, 4000, len(true_segmentations))

correct = (upsampled == true_segmentations).sum()
print(correct / len(upsampled))
