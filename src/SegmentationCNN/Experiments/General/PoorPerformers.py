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


RESULTS_PATH = "/Results/CNN_Results_256_32_26_03_2023/"
WINDOW = 256
STRIDE = 32


def get_model_results(fold):
    results_file = "cnn_256_32_results_26_03_2023_"+ str(fold)
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
    cnn_downsample_prediction = make_sample_prediction(cnn_window_predictions, math.ceil(len(true_segmentations)/(4000/50)), 256, 32)
    cnn_prediction = upsample_states(cnn_downsample_prediction, 50, 4000, len(true_segmentations)) + 1 
    return cnn_prediction 

def get_accuracies():
    cnn_us_accuracy_dict = {1: [], 2: [], 3 : [], 4 : [], 5: []}
    cnn_ds_accuracy_dict = {1: [], 2: [], 3 : [], 4 : [], 5: []}
   
    results_dict = dict() 
    dp = DataPresentation()
    
    for i in range(1, 6):
        results_dict = get_model_results(i)

        audio_len = 0 
        for file in results_dict.keys():
            recording, true_segmentations = get_true_segmentations(file, return_recording=True)
            cnn_prediction = get_upsampled_prediction(results_dict[file], true_segmentations)
            cnn_us_accuracy_dict[i].append((cnn_prediction == true_segmentations).sum() / len(cnn_prediction))
            accuracies = [k[1] for k in results_dict[file]]
            cnn_ds_accuracy_dict[i].extend(accuracies)
            
            if file == "78592_TV.wav" or file == "50219_MV.wav":
                results_dir = "/DataPresentation/SegmentationModelPerformance/"
                dp.plot_PCG_segmentations(file.split(".")[0], results_dir, recording, true_segmentations-1, cnn_prediction-1, clip=False)
            
            audio_len += (len(recording)/4000)

        print(audio_len/len(results_dict.keys()))
            # if np.mean(accuracies) <= 0.5:
            #     results_dir = "/Users/serenahuston/GitRepos/ThirdYearProject/DataPresentation/SegmentationModelPerformance/Poor_256_32_Env/"
            #     dp.plot_PCG_segmentations(file.split(".")[0], results_dir, recording, true_segmentations, cnn_prediction, clip=False)
            # elif np.mean(accuracies) >= 0.95:
            #     results_dir = "/Users/serenahuston/GitRepos/ThirdYearProject/DataPresentation/SegmentationModelPerformance/Good_256_32_Env/"
            #     dp.plot_PCG_segmentations(file.split(".")[0], results_dir, recording, true_segmentations, cnn_prediction, clip=False)
    
    
    return cnn_us_accuracy_dict, cnn_ds_accuracy_dict

def print_accuracy_results(accuracy_dict): 
    overall_accuracies = []
    for (fold, accuracies) in accuracy_dict.items():
        print("------", fold, "------")
        overall_accuracies.extend(accuracies)
        print(np.mean(accuracies), np.std(accuracies))
    print("OVERALL:", np.mean(overall_accuracies))
    print("OVERALL:", np.std(overall_accuracies))

def get_confusion_matrices(model):
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

# for model in ["cnn", "hmm"]:
#     print("-----", model, "-----")
#     get_confusion_matrices(model)

cnn_us_accuracy_dict, cnn_ds_accuracy_dict = get_accuracies()
# print("-----CNN UPSAMPLED-----")
# print_accuracy_results(cnn_us_accuracy_dict)
# print("-----CNN DOWNSAMPLED-----")
# print_accuracy_results(cnn_ds_accuracy_dict)
# print("-----HMM UPSAMPLED-----")
# print_accuracy_results(hmm_accuracy_dict)    


files = ["84714_AV", "64715_PV", "84738_MV"]
  