import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import pickle 
import numpy as np 
import statistics 
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from DataManipulation.DataPresentation import * 
from Utilities.create_segmentation_array import *
from Utilities.constants import *
from Utilities.prediction_helper_functions import * 
from SegmentationCNN.Experiments.performance_metrics import * 
from SegmentationCNN.Models.Envelope_CNN.DataPreprocessing import * 

WINDOW = 64
STRIDE = 8


def get_model_results(model, fold):
    if model == "cnn_env":
        results_file = os.path.join(RESULTS_PATH, "cnn_results_22_03_23__" + str(fold))
    elif model == "cnn_stft":
        results_file = os.path.join(RESULTS_PATH, "stft_cnn_results_" + str(fold))
    elif model == "hmm":
        results_file = os.path.join(RESULTS_PATH, "hmm_results_22_03_23__" + str(fold))
    else:
        print(f"Results file for {model} not found.")
        return None
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
        # print(dict(list(results.items())[:1])) # horrible way to get around hashing
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
    results_dict = {"cnn_env" : dict(), "hmm" : dict()}
    dp = DataPresentation()

    for i in range(1, 6):
        for model in results_dict.keys():
            results_dict[model] = get_model_results(model, i)
            if results_dict[model] is None:
                return -1

        for file in results_dict["cnn_env"].keys():
            recording, true_segmentations = get_true_segmentations(file, return_recording=True)
            cnn_prediction = get_upsampled_prediction(results_dict["cnn_env"][file], true_segmentations)
            
            cnn_us_accuracy_dict[i].append((cnn_prediction == true_segmentations).sum() / len(cnn_prediction))
            cnn_ds_accuracy_dict[i].extend([k[1] for k in results_dict["cnn_env"][file]])
            hmm_accuracy_dict[i].append(results_dict["hmm"][file][1])

            cnn_accuracy = (cnn_prediction == true_segmentations).sum() / len(cnn_prediction)
            hmm_accuracy = results_dict["hmm"][file][1]
            hmm_segs = results_dict["hmm"][file][0]

            if cnn_accuracy <=0.5 and hmm_accuracy >=0.8:
                print(file, cnn_accuracy, hmm_accuracy)
                # results_dir = "/DataPresentation/SegmentationModelPerformance/CNN_vs_HMM/"
                # dp.plot_PCG_HMM_vs_CNN_segmentations(file.split(".")[0], results_dir, recording, true_segmentations, cnn_prediction-1, hmm_segs-1, clip=True)
    
    return cnn_us_accuracy_dict, cnn_ds_accuracy_dict, hmm_accuracy_dict


def print_accuracy_results(accuracy_dict): 
    overall_accuracies = []
    for (fold, accuracies) in accuracy_dict.items():
        print("------", fold, "------")
        overall_accuracies.extend(accuracies)
        print(np.mean(accuracies), np.std(accuracies))
    print(f"OVERALL: {np.mean(overall_accuracies):.2f}")
    print(f"OVERALL: {np.std(overall_accuracies):.2f}")


def get_upsampled_confusion_matrices(model):
    print("Generating confusion matrices for upsampled recordings segmentations...")
    totals = [0,0,0,0]
    confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    acc_by_len = []

    for i in range(1, 6):
        results_dict = get_model_results(model, i)
        if results_dict is None:
            return -1

        acc_by_len_split = np.zeros((len(results_dict.keys()), 2))
        for idx, file in enumerate(tqdm(results_dict.keys())):
            true_segmentations = get_true_segmentations(file)-1
            if model == "cnn_env" or model == "cnn_stft":
                prediction = get_upsampled_prediction(results_dict[file], true_segmentations)-1
            else:
                prediction = results_dict[file][0]-1
            for j in range(0,4):
                totals[j], confusion_mat_all_classes[j] = update_confusion_matrix(prediction, true_segmentations, totals[j], confusion_mat_all_classes[j], j)

            # accuracy by recording length
            acc_by_len_split[idx,0] = len(true_segmentations) # length
            acc_by_len_split[idx,1] = (np.logical_and((prediction == true_segmentations), (true_segmentations != 0)).mean()) # accuracy
        
        acc_by_len.append(acc_by_len_split)
        break

    metrics = dict()
    metrics = build_metric_dict(confusion_mat_all_classes)

    means = []
    stds = [] 
    print("TP", "TN", "FP", "FN")
    print([np.array(confusion_mat_all_classes[i])/totals[i] for i in range(4)])

    for (metric, vals) in metrics.items():
        print("----",metric, "----")
        print(f"{np.mean(vals):.2f}, {np.std(vals):.2f}")
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    return acc_by_len


def get_downsampled_confusion_matrices(model):
    print("Generating confusion matrices for downsampled recordings segmentations...")
    cnn_totals = [0,0,0,0]
    cnn_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(1, 6):
        results_for_window = get_model_results(model, i)
        if results_for_window is None:
            return -1
    
        for file in tqdm(results_for_window.keys()):
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
        print(f"{np.mean(vals):.2f}, {np.std(vals):.2f}")
        means.append(np.mean(vals))
        stds.append(np.std(vals))


def plot_acc_by_len(model_performances : dict):
    split = 0
    fig, ax1 = plt.subplots(1, 1)
    # fig.set_size_inches(18, 10)
    ax1.set_title('Accuracy as a function of recording duration')
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Recording duration")
    colors = ["b", "g", "r"]
    # print(model_performances.keys())
    for model, color in zip(model_performances.keys(), colors):
        acc_by_len = model_performances[model][0] # just take the first split for now
        ax1.scatter(acc_by_len[:,0], acc_by_len[:,1], label=model, color=color, s=20)
        # regression fit
        res = stats.linregress(acc_by_len[:,0], acc_by_len[:,1])
        print(f"{model} R-squared: {res.rvalue**2:.6f}")
        ax1.plot(acc_by_len[:,0], res.intercept + res.slope*acc_by_len[:,0], color=color)
    ax1.legend(loc="upper right")
    fig.savefig(os.path.join("src","Experiments","acc_by_duration_split_0.png"))


def main():
    print("----UPSAMPLED----")
    model_perf = {}
    for model in ["cnn_env", "cnn_stft", "hmm"]:
        print("-----", model, "-----")
        model_perf[model] = get_upsampled_confusion_matrices(model)
    plot_acc_by_len(model_perf)

    # print("DOWNSAMPLED CNN")
    # get_downsampled_confusion_matrices("cnn_env")
    # cnn_us_accuracy_dict, cnn_ds_accuracy_dict, hmm_accuracy_dict = get_accuracies()
    # print("-----CNN UPSAMPLED-----")
    # print_accuracy_results(cnn_us_accuracy_dict)
    # print("-----CNN DOWNSAMPLED-----")
    # print_accuracy_results(cnn_ds_accuracy_dict)
    # print("-----HMM UPSAMPLED-----")
    # print_accuracy_results(hmm_accuracy_dict) 


if __name__ == "__main__":
    main()



  