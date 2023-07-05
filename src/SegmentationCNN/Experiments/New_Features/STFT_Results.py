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
from SegmentationCNN.Models.STFT_CNN.STFT_DataPreprocessing import * 


RESULTS_PATH = "/Results/STFT_64_8_Results_26_03_2023/"
WINDOW = 64
STRIDE = 8


def get_model_results(fold):
    results_file = "stft_cnn_results_" + str(fold)
    with open(RESULTS_PATH + results_file, 'rb') as f:
        
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
    cnn_downsample_prediction = make_sample_prediction(cnn_window_predictions, math.ceil(len(true_segmentations)/(4000/50)), WINDOW, STRIDE)
    cnn_prediction = upsample_states(cnn_downsample_prediction, 50, 4000, len(true_segmentations)) + 1 
    return cnn_prediction 

def get_accuracies():
    cnn_us_accuracy_dict = {1: [], 2: [], 3 : [], 4 : [], 5: []}
    cnn_ds_accuracy_dict = {1: [], 2: [], 3 : [], 4 : [], 5: []}
    results_dict = dict() 
    dp = DataPresentation()
    acc_more_90 = 0 
    total = 0

    for i in range(1, 6):
        results_dict= get_model_results(i)

        for file in results_dict.keys():
            recording, true_segmentations = get_true_segmentations(file, return_recording=True)
            cnn_prediction = get_upsampled_prediction(results_dict[file], true_segmentations)
            
            cnn_us_accuracy_dict[i].append((cnn_prediction == true_segmentations).sum() / len(cnn_prediction))
            cnn_ds_accuracy_dict[i].extend([k[1] for k in results_dict[file]])

            acc = (cnn_prediction == true_segmentations).sum() / len(cnn_prediction)

            
            

    print(acc_more_90, total)
    return cnn_us_accuracy_dict, cnn_ds_accuracy_dict

def print_accuracy_results(accuracy_dict): 
    overall_accuracies = []
    for (fold, accuracies) in accuracy_dict.items():
        print("------", fold, "------")
        overall_accuracies.extend(accuracies)
        print(np.mean(accuracies), np.std(accuracies))
    print("OVERALL:", np.mean(overall_accuracies))
    print("OVERALL:", np.std(overall_accuracies))

def get_upsampled_confusion_matrices():
    totals = [0,0,0,0]
    confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(1, 6):
        results_dict = get_model_results(i)
    
        for file in results_dict.keys():
            true_segmentations = get_true_segmentations(file)-1
            prediction = get_upsampled_prediction(results_dict[file], true_segmentations)-1
            for j in range(0,4):
                totals[j], confusion_mat_all_classes[j] = update_confusion_matrix(prediction, true_segmentations, totals[j], confusion_mat_all_classes[j], j)
    metrics = dict()
    metrics = build_metric_dict(confusion_mat_all_classes)

    means = []
    stds = [] 
    print("TP", "TN", "FP", "FN")
    print([np.array(confusion_mat_all_classes[i])/totals[i] for i in range(4)])

    for (metric, vals) in metrics.items():
        print("----",metric, "----")
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    print(means, stds)

def get_downsampled_confusion_matrices():
    cnn_totals = [0,0,0,0]
    cnn_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(1, 6):
        results_for_window = get_model_results(i)
    
        for file in results_for_window.keys():
            wav, true_segmentations = get_true_segmentations(file, return_recording=True)
            dp = STFT_DataPreprocessing(wav, true_segmentations-1, 4000, window=WINDOW*80, stride=STRIDE*80)
            dp.extract_wav_and_seg_patches()
            for k in range(len(results_for_window[file])):
                patch = results_for_window[file][k][0]
                patch_num = results_for_window[file][k][2]

                for j in range(0,4):
                    try: 
                        cnn_totals[j], cnn_confusion_mat_all_classes[j] = update_confusion_matrix(patch, dp.output_patches[patch_num], cnn_totals[j], cnn_confusion_mat_all_classes[j], j)
                    except: 
                        print(len(dp.seg_patches), patch_num)
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

# for model in ["cnn", "hmm"]:
#     print("-----", model, "-----")
#     get_confusion_matrices(model)

# get_downsampled_confusion_matrices("cnn")
# cnn_us_accuracy_dict, cnn_ds_accuracy_dict = get_accuracies()
print("-----CNN UPSAMPLED-----")
get_upsampled_confusion_matrices()
# print_accuracy_results(cnn_us_accuracy_dict)
print("-----CNN DOWNSAMPLED-----")
# print_accuracy_results(cnn_ds_accuracy_dict)
get_downsampled_confusion_matrices()
  
    


  