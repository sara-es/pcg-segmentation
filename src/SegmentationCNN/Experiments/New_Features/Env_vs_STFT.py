import pickle 
import numpy as np 
import statistics 
import sys 

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from DataManipulation.DataPresentation import * 
from Utilities.create_segmentation_array import *
from Utilities.constants import *
from Utilities.prediction_helper_functions import * 
from SegmentationCNN.Experiments.performance_metrics import * 
from SegmentationCNN.Models.Envelope_CNN.DataPreprocessing import * 


RESULTS_PATH = "/Users/serenahuston/GitRepos/ThirdYearProject/Results/"
WINDOW = 64
STRIDE = 8


def get_model_results(model, fold):
    if model == "env":
        results_file = "HMM_vs_CNN_22_03_2023/cnn_results_22_03_23__" + str(fold)
    elif model == "hmm":
        results_file = "HMM_vs_CNN_22_03_2023/hmm_results_22_03_23__" + str(fold)
    elif model == "stft":
        results_file = "STFT_64_8_Results_26_03_2023/stft_cnn_results_" + str(fold)
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
    hmm_accuracy_dict = {1: [], 2: [], 3 : [], 4 : [], 5: []}
    results_dict = {"env" : dict(), "stft" : dict(), "hmm": dict()}
    prop_murmurs = {"env" : [], "stft" : [], "hmm": []}
    dp = DataPresentation()
    for i in range(1, 6):
        for model in results_dict.keys():
            results_dict[model] = get_model_results(model, i)

        for file in results_dict["env"].keys():
            recording, true_segmentations = get_true_segmentations(file, return_recording=True)
            env_prediction = get_upsampled_prediction(results_dict["env"][file], true_segmentations)
            stft_prediction = get_upsampled_prediction(results_dict["stft"][file], true_segmentations)
            hmm_prediction = results_dict["hmm"][file][0]

            env_accuracy = (env_prediction == true_segmentations).sum() / len(env_prediction)
            stft_accuracy = (stft_prediction == true_segmentations).sum() / len(stft_prediction)
            hmm_accuracy = results_dict["hmm"][file][1]

            if env_accuracy > 0.8:
                prop_murmurs["env"].append(file)
            elif stft_accuracy > 0.8: 
                prop_murmurs["stft"].append(file)
            elif hmm_accuracy > 0.8: 
                prop_murmurs["hmm"].append(file)
    
    for model in prop_murmurs.keys():
        print(model)
        calc_proportion_murmurs(prop_murmurs[model])
    
    return cnn_us_accuracy_dict, cnn_ds_accuracy_dict, hmm_accuracy_dict

def print_accuracy_results(accuracy_dict): 
    overall_accuracies = []
    for (fold, accuracies) in accuracy_dict.items():
        print("------", fold, "------")
        overall_accuracies.extend(accuracies)
        print(np.mean(accuracies), np.std(accuracies))
    print("OVERALL:", np.mean(overall_accuracies))
    print("OVERALL:", np.std(overall_accuracies))

def get_upsampled_confusion_matrices(model):
    totals = [0,0,0,0]
    confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(1, 6):
        results_dict = get_model_results(model, i)
    
        for file in results_dict.keys():
            true_segmentations = get_true_segmentations(file)-1
            if model == "cnn":
                prediction = get_upsampled_prediction(results_dict[file], true_segmentations)-1
            else:
                prediction = results_dict[file][0]-1
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

def get_downsampled_confusion_matrices(model):
    cnn_totals = [0,0,0,0]
    cnn_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(1, 6):
        results_for_window = get_model_results(model, i)
    
        for file in results_for_window.keys():
            wav, true_segmentations = get_true_segmentations(file, return_recording=True)
            dp = DataPreprocessing(wav, true_segmentations-1, 4000, window=WINDOW, stride=STRIDE)
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

# for model in ["cnn", "hmm"]:
#     print("-----", model, "-----")
#     get_confusion_matrices(model)

# get_downsampled_confusion_matrices("cnn")
cnn_us_accuracy_dict, cnn_ds_accuracy_dict, hmm_accuracy_dict = get_accuracies()
# print("-----CNN UPSAMPLED-----")
# print_accuracy_results(cnn_us_accuracy_dict)
# print("-----CNN DOWNSAMPLED-----")
# print_accuracy_results(cnn_ds_accuracy_dict)
# print("-----HMM UPSAMPLED-----")
# print_accuracy_results(hmm_accuracy_dict)    
    


  