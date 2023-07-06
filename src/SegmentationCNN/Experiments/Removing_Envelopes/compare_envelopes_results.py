import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import pickle 
import numpy as np 
import statistics 

from DataManipulation.DataPresentation import * 
from Utilities.create_segmentation_array import *
from Utilities.constants import *
from Utilities.prediction_helper_functions import * 
from SegmentationCNN.Experiments.performance_metrics import * 
from SegmentationCNN.Experiments.Removing_Envelopes.DataPreprocessing import * 
from SegmentationCNN.Experiments.Removing_Envelopes.Envelope_Enum import * 

RESULTS_PATH = '/Results/Envelope_Experiments/'
ENVELOPES = [Envelope.HOMO, Envelope.HILB, Envelope.WAVE, Envelope.PSD]



def get_model_results(envelope, fold):
    if envelope == Envelope.HOMO:
        results_file = RESULTS_PATH + "Homo/cnn_homo_results_" + str(fold)
    elif envelope == Envelope.HILB:
        results_file = RESULTS_PATH + "Hilb/cnn_hilb_results_" + str(fold)
    elif envelope == Envelope.WAVE:
        results_file = RESULTS_PATH + "Wave/cnn_wave_results_" + str(fold)
    elif envelope == Envelope.PSD:
        results_file = RESULTS_PATH + "PSD/cnn_psd_results_" + str(fold)
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
        return results 

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

def get_upsampled_prediction(cnn_window_predictions, true_segmentations):
    cnn_downsample_prediction = make_sample_prediction(cnn_window_predictions, math.ceil(len(true_segmentations)/(4000/50)), 128, 16)
    cnn_prediction = upsample_states(cnn_downsample_prediction, 50, 4000, len(true_segmentations)) + 1 
    return cnn_prediction 

def get_accuracies():
    results_dict = {Envelope.HOMO: [], Envelope.HILB: [], Envelope.WAVE : [], Envelope.PSD : []}
    us_accuracy_dict = {Envelope.HOMO: dict(), Envelope.HILB: dict(), Envelope.WAVE : dict(), Envelope.PSD : dict()}
    ds_accuracy_dict = {Envelope.HOMO: dict(), Envelope.HILB: dict(), Envelope.WAVE : dict(), Envelope.PSD : dict()}

    for i in range(1, 6):
        for j in range(len(ENVELOPES)):
            results_dict[ENVELOPES[j]] = get_model_results(ENVELOPES[j], i)
            us_accuracy_dict[ENVELOPES[j]][i] = []
            ds_accuracy_dict[ENVELOPES[j]][i] = []


        for file in results_dict[Envelope.HOMO].keys():
            true_segmentations = get_true_segmentations(file)
            for j in range(len(ENVELOPES)):
                cnn_prediction = get_upsampled_prediction(results_dict[ENVELOPES[j]][file], true_segmentations)
                us_accuracy_dict[ENVELOPES[j]][i].append((cnn_prediction == true_segmentations).sum() / len(cnn_prediction))
                ds_accuracy_dict[ENVELOPES[j]][i].extend([k[1] for k in results_dict[ENVELOPES[j]][file]])
    
    return us_accuracy_dict, ds_accuracy_dict

def print_accuracy_results(accuracy_dict): 
    overall_accuracies =  {Envelope.HOMO: [], Envelope.HILB: [], Envelope.WAVE : [], Envelope.PSD : []}
    for (envelope, fold_dict) in accuracy_dict.items():
        print("------", envelope, "------")
        for (fold, accuracies) in fold_dict.items():
            print("------", fold, "------")
            overall_accuracies[envelope].extend(accuracies)
            print(np.mean(accuracies), np.std(accuracies))
    print("OVERALL:", [np.mean(overall_accuracies[window]) for window in overall_accuracies.keys()])
    print("OVERALL:", [np.std(overall_accuracies[window]) for window in overall_accuracies.keys()])

def get_upsampled_confusion_matrices(envelope):
    cnn_totals = [0,0,0,0]
    cnn_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(1, 6):
        results_for_window = get_model_results(envelope, i)
    
        for file in results_for_window.keys():
            true_segmentations = get_true_segmentations(file)-1
            cnn_prediction = get_upsampled_prediction(results_for_window[file], true_segmentations)-1
            if envelope == Envelope.HILB and i == 5:
                    continue 
            else: 
                for j in range(0,4):
                    cnn_totals[j], cnn_confusion_mat_all_classes[j] = update_confusion_matrix(cnn_prediction, true_segmentations, cnn_totals[j], cnn_confusion_mat_all_classes[j], j)
    cnn_metrics = dict()
    cnn_metrics = build_metric_dict(cnn_confusion_mat_all_classes)

    means = []
    stds = [] 
    print("TP", "TN", "FP", "FN")
    print([np.array(cnn_confusion_mat_all_classes[i])/cnn_totals[i] for i in range(4)])

    for (metric, vals) in cnn_metrics.items():
        print("----",metric, "----")
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    print(means, stds)

def get_downsampled_confusion_matrices(envelope):
    cnn_totals = [0,0,0,0]
    cnn_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(1, 6):
        results_for_window = get_model_results(envelope, i)
    
        for file in results_for_window.keys():
            wav, true_segmentations = get_true_segmentations(file, return_recording=True)
            dp = DataPreprocessing(wav, true_segmentations-1, 4000, envelopes=[envelope])
            dp.extract_segmentation_patches()
            for k in range(len(results_for_window[file])):
                patch = results_for_window[file][k][0]
                patch_num = results_for_window[file][k][2]

                if envelope == Envelope.HILB and i == 5:
                    continue 
                else: 
                    for j in range(0,4):
                        cnn_totals[j], cnn_confusion_mat_all_classes[j] = update_confusion_matrix(patch, dp.seg_patches[patch_num], cnn_totals[j], cnn_confusion_mat_all_classes[j], j)
    cnn_metrics = dict()
    cnn_metrics = build_metric_dict(cnn_confusion_mat_all_classes)

    means = []
    stds = [] 
    print("TP", "TN", "FP", "FN")
    print([np.array(cnn_confusion_mat_all_classes[i])/cnn_totals[i] for i in range(4)])

    for (metric, vals) in cnn_metrics.items():
        print("----",metric, "----")
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    print(means, stds)



# us_accuracy_dict, ds_accuracy_dict = get_accuracies()
# print("-----UPSAMPLED-----")
# print_accuracy_results(us_accuracy_dict)
# print("-----DOWNSAMPLED-----")
# print_accuracy_results(ds_accuracy_dict)

for i in range(len(ENVELOPES)):
    print("-----", ENVELOPES[1], "-----")
    print("-----DOWNSAMPLED-----")
    get_downsampled_confusion_matrices(ENVELOPES[1])

    print("-----UPSAMPLED-----")
    get_upsampled_confusion_matrices(ENVELOPES[1])

    
    


  