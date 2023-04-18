import numpy as np
import statistics
import torch
import torch.nn.functional as F 

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

def combine_windows(patches, new_length, window=64, stride=8):
    index_options = {key: [] for key in range(new_length)}
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

def make_window_prediction(window_probabilities):
    softmax = F.softmax(window_probabilities, dim=0)
    _, yhat = torch.max(softmax, 0)
    return yhat 

def make_ensemble_window_prediction(env_probabilites, stft_probabilities):
    env_softmax = F.softmax(env_probabilites, dim=0)
    stft_softmax = F.softmax(stft_probabilities, dim=0)
    avg = (stft_softmax + env_softmax)/2
    _, yhat = torch.max(avg, 0)
    return yhat 