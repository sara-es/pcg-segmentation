import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import pickle 
import numpy as np 
import statistics 
from tqdm import tqdm, trange
import matplotlib as mpl
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


def get_upsampled_results(models: list, fold: int):
    model_outputs = {}
    # get keys (file names)
    filenames = get_model_results(models[0], fold).keys()

    for model in models:
        print(f"Extracting results from {model}...")
        results_dict = get_model_results(model, fold)
        if results_dict is None:
            sys.exit("Could not retrieve results dictionary.")
        if set(results_dict.keys()) != set(filenames):
            sys.exit("Inconsistent filenames in results dictionary; are you using the same train-test split across models?")

        y_preds = []
        true_outputs = []

        for file in tqdm(filenames):
            true_segmentations = get_true_segmentations(file)-1
            true_outputs.append(true_segmentations)
            
            if model == "cnn_env" or model == "cnn_stft":
                prediction = get_upsampled_prediction(results_dict[file], true_segmentations)-1
            elif model == "hmm":
                prediction = results_dict[file][0]-1
            else:
                sys.exit("Unknown model or no results found.")
            
            y_preds.append(prediction)

        model_outputs[model] = y_preds

    return model_outputs, true_outputs


def accuracy_length_array_gen(predictions: list, true_segmentations: list):
    print("Calculating accuracies...")
    acc_by_len = np.zeros((len(true_segmentations), 2))

    # accuracy by recording length
    for i in range(len(true_segmentations)):
        # recording length
        acc_by_len[i,0] = len(true_segmentations[i]) 
        # accuracy: exclude recording segments with 0 values, ie those where manual annotation was only done for some portion
        acc_by_len[i,1] = (np.logical_and((predictions[i] == true_segmentations[i]), (true_segmentations[i] != 0)).mean()) 

    return acc_by_len


def majority_voting_three_way(cnn_env, cnn_stft, hmm):
    print("Gathering consensus vote...")
    yhat_consensus = []
    if len(cnn_env) != len(cnn_stft) or len(cnn_env) != len(hmm):
        sys.exit("Model results are not of equal length.")
    for i in trange(len(cnn_env)):
        preds = np.vstack((cnn_env[i], cnn_stft[i], hmm[i]))
        consensus = stats.mode(preds)[0]
        yhat_consensus.append(consensus)

    return yhat_consensus


def get_cmap(n, name='winter'):
    # cmap = mpl.colormaps.get_cmap(name).resampled(n)
    # colors = cmap(np.arange(0, cmap.N)) 
    colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
    return colors[:n]


def plot_acc_by_len(model_performances : dict, fold : int):
    print("Plotting results...")
    fig, ax1 = plt.subplots(1, 1)
    # fig.set_size_inches(18, 10)
    ax1.set_title('Accuracy as a function of recording duration')
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Recording duration")
    colors = get_cmap(len(model_performances.keys()))

    for model, color in zip(model_performances.keys(), colors):
        acc_by_len = model_performances[model]
        ax1.scatter(acc_by_len[:,0], acc_by_len[:,1], label=model, color=color, s=20)
        # regression fit
        res = stats.linregress(acc_by_len[:,0], acc_by_len[:,1])
        print(f"{model} R-squared: {res.rvalue**2:.6f}")
        ax1.plot(acc_by_len[:,0], res.intercept + res.slope*acc_by_len[:,0], color=color)

    ax1.legend(loc="upper right")
    fig.savefig(os.path.join("src","Experiments",f"acc_by_duration_fold{fold}.png"))


def bin_acc_by_len(model_performances: dict, fold : int):
    print("Plotting results (histogram)...")
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title('Average accuracy as a function of recording duration in 2s bins')
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Recording duration")
    colors = get_cmap(len(model_performances.keys()))

    for model, color in zip(model_performances.keys(), colors):
        acc_by_len = model_performances[model] 
        n_bins = int((acc_by_len[:,0].max() - acc_by_len[:,0].min())/(4000*2)) # fs=4000, approx 2s bins
        bin_means, bin_edges, binnumber = stats.binned_statistic(acc_by_len[:,0], acc_by_len[:,1], bins=n_bins)
        ax1.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors=color, lw=2,
           label=model)

    ax1.legend(loc="upper right")
    fig.savefig(os.path.join("src","Experiments",f"acc_by_duration_binned_fold{fold}.png"))


def main():
    # dict of model performances, list of ground truth segmentations
    models = ["cnn_env", "cnn_stft", "hmm"]
    fold = 4
    model_outputs, true_outputs = get_upsampled_results(models, fold)

    # find majority vote. WARNING: very slow (~15 minutes)
    model_outputs["consensus"] = majority_voting_three_way(**model_outputs)

    model_perf = {} # dictionary with accuracy vs segmentation length for each model
    for model in model_outputs.keys(): 
        model_perf[model] = accuracy_length_array_gen(model_outputs[model], true_outputs)
    
    # plot
    plot_acc_by_len(model_perf, fold)
    bin_acc_by_len(model_perf, fold)


if __name__ == "__main__":
    main()



  