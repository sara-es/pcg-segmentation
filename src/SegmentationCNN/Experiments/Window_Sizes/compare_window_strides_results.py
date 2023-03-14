import pickle 
import numpy as np 
import scipy as sp 
import statistics 
import sys 
import os 

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from DataManipulation.DataPresentation import * 
from DataManipulation.PatientFrame import PatientFrame
from Utilities.create_segmentation_array import *
from Utilities.constants import *



RESULTS_PATH = '/Users/serenahuston/GitRepos/ThirdYearProject/Results/CNN_Results_'
WINDOWS = [64, 128, 256, 512]
STRIDES = [8, 16, 32, 64]
results_dict = {64 : [], 128: [], 256 : [], 512 : []}
accuracy_dict = {64 : [], 128: [], 256 : [], 512 : []}

def get_model_results(window, stride, fold):
    results_dir = RESULTS_PATH + str(window) + "_" + str(stride) + "/"
    results_file = "cnn_" + str(window) + "_" + str(stride) + "_results_" + str(fold)
    with open(results_dir + results_file, 'rb') as f:
        results = pickle.load(f)
        return results 
    
def get_true_segmentations(file):
    patientID = file.split(".")[0]
    fs, recording = sp.io.wavfile.read(os.path.join(TRAINING_DATA_PATH_2022 + "training_data/", file))
    tsv = np.loadtxt(TRAINING_DATA_PATH_2022 + "training_data/" + patientID + ".tsv", delimiter="\t")
    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                    tsv,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=4000)

    return np.array(segmentations[0])

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


def make_sample_prediction(file_results, new_length, window, stride):
    index_options = {key: [] for key in range(new_length)}
    sorted_windows = sorted(file_results,key=lambda l:l[2])
    if (window > new_length):
        return np.array(sorted_windows[0][0][:new_length])
    for i in range(len(sorted_windows)):
        for j in range(len(sorted_windows[i][0])):
       
            try: 
                index_options[j+(stride*i)].append(sorted_windows[i][0][j].item())
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




for i in range(1, 6):
    for j in range(len(WINDOWS)):
        results_dict[WINDOWS[j]] = get_model_results(WINDOWS[j], STRIDES[j], i)

    for file in results_dict[64].keys():
        true_segmentations = get_true_segmentations(file)
        for j in range(len(WINDOWS)):
            cnn_window_predictions = results_dict[WINDOWS[j]][file]
            cnn_downsample_prediction = make_sample_prediction(cnn_window_predictions, math.ceil(len(true_segmentations)/(4000/50)), WINDOWS[j], STRIDES[j])
            cnn_prediction = upsample_states(cnn_downsample_prediction, 50, 4000, len(true_segmentations)) + 1 
            accuracy_dict[WINDOWS[j]].append((cnn_prediction == true_segmentations).sum() / len(cnn_prediction))
    print([np.mean(value) for (key, value) in accuracy_dict.items()])


  