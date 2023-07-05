import sys, os
sys.path.append(os.path.join(sys.path[0], '..', '..', '..'))

import pickle 
import numpy as np 
import statistics 

from DataManipulation.DataPresentation import * 
from Utilities.create_segmentation_array import *
from Utilities.constants import *
from Utilities.prediction_helper_functions import * 
from SegmentationCNN.Experiments.performance_metrics import * 
from SegmentationCNN.Models.Envelope_CNN.DataPreprocessing import * 

RESULTS_PATH = '/Results/CNN_Results_'
WINDOWS = [128, 256, 512]
STRIDES = [ 16, 32, 64]


def get_model_results(window, stride, fold):
    results_dir = RESULTS_PATH + str(window) + "_" + str(stride) + "_23_03_2023/"
    results_file = "cnn_" + str(window) + "_" + str(stride) + "_results_23_03_2023_" + str(fold)
    with open(results_dir + results_file, 'rb') as f:
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

def get_upsampled_prediction(cnn_window_predictions, true_segmentations, window, stride):
    cnn_downsample_prediction = make_sample_prediction(cnn_window_predictions, math.ceil(len(true_segmentations)/(4000/50)), window, stride)
    cnn_prediction = upsample_states(cnn_downsample_prediction, 50, 4000, len(true_segmentations)) + 1 
    return cnn_prediction 

def get_accuracies():
    results_dict = {64: [], 128: [], 256 : [], 512 : []}
    us_accuracy_dict = {64: dict(), 128: dict(), 256 : dict(), 512 : dict()}
    ds_accuracy_dict = {64: dict(), 128: dict(), 256 : dict(), 512 : dict()}

    for i in range(1, 6):
        for j in range(len(WINDOWS)):
            results_dict[WINDOWS[j]] = get_model_results(WINDOWS[j], STRIDES[j], i)
            us_accuracy_dict[WINDOWS[j]][i] = []
            ds_accuracy_dict[WINDOWS[j]][i] = []


        for file in results_dict[128].keys():
            true_segmentations = get_true_segmentations(file)
            for j in range(len(WINDOWS)):
                cnn_prediction = get_upsampled_prediction(results_dict[WINDOWS[j]][file], true_segmentations, WINDOWS[j], STRIDES[j])
                us_accuracy_dict[WINDOWS[j]][i].append((cnn_prediction == true_segmentations).sum() / len(cnn_prediction))
                ds_accuracy_dict[WINDOWS[j]][i].extend([k[1] for k in results_dict[WINDOWS[j]][file]])
    
    return us_accuracy_dict, ds_accuracy_dict

def print_accuracy_results(accuracy_dict): 
    overall_accuracies = {64: [], 128: [], 256 : [], 512 : []}
    for (window,fold_dict) in accuracy_dict.items():
        print("------", window, "------")
        for (fold, accuracies) in fold_dict.items():
            print("------", fold, "------")
            overall_accuracies[window].extend(accuracies)
            print(np.mean(accuracies), np.std(accuracies))
    print("OVERALL:", [np.mean(overall_accuracies[window]) for window in overall_accuracies.keys()])
    print("OVERALL:", [np.std(overall_accuracies[window]) for window in overall_accuracies.keys()])

def get_upsampled_confusion_matrices(window, stride):
    cnn_totals = [0,0,0,0]
    cnn_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(1, 6):
        results_for_window = get_model_results(window, stride, i)
    
        for file in results_for_window.keys():
            true_segmentations = get_true_segmentations(file)-1
            cnn_prediction = get_upsampled_prediction(results_for_window[file], true_segmentations, window, stride)-1
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

def get_downsampled_confusion_matrices(window, stride):
    cnn_totals = [0,0,0,0]
    cnn_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(1, 6):
        results_for_window = get_model_results(window, stride, i)
    
        for file in results_for_window.keys():
            wav, true_segmentations = get_true_segmentations(file, return_recording=True)
            dp = DataPreprocessing(wav, true_segmentations-1, 4000, window=window, stride=stride)
            dp.extract_segmentation_patches()
            for k in range(len(results_for_window[file])):
                patch = results_for_window[file][k][0]
                patch_num = results_for_window[file][k][2]

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

for i in range(len(WINDOWS)):
    print("-----", WINDOWS[i], "-----")
    print("-----DOWNSAMPLED-----")
    get_downsampled_confusion_matrices(WINDOWS[i], STRIDES[i])
    print("-----UPSAMPLED----")
    get_upsampled_confusion_matrices(WINDOWS[i], STRIDES[i])
# us_accuracy_dict, ds_accuracy_dict = get_accuracies()
# print("-----UPSAMPLED-----")
# print_accuracy_results(us_accuracy_dict)
# print("-----DOWNSAMPLED-----")
# print_accuracy_results(ds_accuracy_dict)
    
    


  