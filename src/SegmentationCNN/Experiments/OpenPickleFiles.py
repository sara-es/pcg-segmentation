import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys 
import statistics
import scipy as sp
import os 
from librosa import resample 

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from DataManipulation.DataPresentation import * 
from DataManipulation.PatientFrame import PatientFrame
from Utilities.create_segmentation_array import *
from Utilities.constants import *

csv_file = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data.csv"

PATCH_SIZE = 64
STRIDE = 8

def plot_sound_and_segmentations(file_root, hmm_segmentations, cnn_segmentations):
    
    fs, recording = sp.io.wavfile.read(TRAINING_DATA_PATH_2022 + "training_data/"+ file_root + ".wav")

    tsv = np.loadtxt(TRAINING_DATA_PATH_2022 + "training_data/" + file_root + ".tsv", delimiter="\t")

    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                tsv,
                                                                recording_frequency=fs,
                                                                feature_frequency=fs)



    time_4000 = np.linspace(0, len(clipped_recording[0]) / 4000, num=len(clipped_recording[0]))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
    fig.set_size_inches(16, 8)
    fig.subplots_adjust(hspace=0)
    
    ax1.plot(time_4000, clipped_recording[0], color="#611d91")        
    ax2.plot(time_4000, segmentations[0], color="#611d91")     
    ax3.plot(time_4000, hmm_segmentations, color="#611d91")
    ax4.plot(time_4000, cnn_segmentations, color="#611d91")
    start, end = ax1.get_xlim()
    ax1.xaxis.set_ticks(np.arange(0, end, 1))
    ax2.xaxis.set_ticks(np.arange(0, end, 1))
    ax3.xaxis.set_ticks(np.arange(0, end, 1))
    ax4.xaxis.set_ticks(np.arange(0, end, 1))
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()

    ax3.set_xlabel("Time (Seconds)")
    ax1.set_ylabel("Clipped PCG Recording")
    ax2.set_ylabel("True FHS Segmentations")
    ax3.set_ylabel("HMM Segmentations")
    ax4.set_ylabel("CNN Segmentations")
    plt.savefig(DATA_PRESENTATION_PATH +"SegmentationModelPerformance/"+ file_root + "_Segmentation_Model_Performance")                                

def calc_proportion_murmurs(bad_files):
    pf = PatientFrame(csv_file)
    murmurs = pf.get_murmur_status_by_patient([int(file.split("_")[0]) for file in bad_files.keys()])
    present = (murmurs == 'Present').sum()
    unknown = (murmurs == 'Unknown').sum()
    absent = (murmurs == 'Absent').sum()
    print(present, unknown, absent)

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

def make_sample_prediction(file_results, new_length):
    index_options = {key: [] for key in range(new_length)}
    sorted_windows = sorted(file_results,key=lambda l:l[2])
    if (PATCH_SIZE > new_length):
        return np.array(sorted_windows[0][0][:new_length])
    for i in range(len(sorted_windows)):
        for j in range(len(sorted_windows[i][0])):
            max_index = (PATCH_SIZE-1)+(STRIDE*i)
            if max_index >= new_length:
                shift = max_index - new_length + 1 
                index_options[j+(STRIDE*i)-shift].append(sorted_windows[i][0][j].item())
            else: 
                index_options[j+(STRIDE*i)].append(sorted_windows[i][0][j].item())

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

def calc_sensitivity(tp, fn):
    return tp/(tp+fn)

def calc_specificity(tn, fp):
    return tn/(tn+fp)

def calc_ppv(tp, fp):
    return tp/(tp+fp)

def calc_npv(tn, fn):
    return tn/(fn+tn)

def build_metric_dict(confusion_mat):
    metrics = dict()
    metrics["Sensitivity"] = [calc_sensitivity(fhs[0], fhs[3]) for fhs in confusion_mat]
    metrics["Specificity"] = [calc_specificity(fhs[1], fhs[2]) for fhs in confusion_mat]
    metrics["PPV"] = [calc_ppv(fhs[0], fhs[2]) for fhs in confusion_mat]
    metrics["NPV"] = [calc_npv(fhs[1], fhs[3]) for fhs in confusion_mat]
    return metrics 

def get_true_segmentations(file):
    patientID = file.split(".")[0]
    fs, recording = sp.io.wavfile.read(os.path.join(TRAINING_DATA_PATH_2022 + "training_data/", file))
    tsv = np.loadtxt(TRAINING_DATA_PATH_2022 + "training_data/" + patientID + ".tsv", delimiter="\t")
    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                    tsv,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=4000)

    return np.array(segmentations[0])

# def downsample_segmentation_array(segmentation_array):
#     labels_per_sample = int(4000 / 50)
#     downsample_segment = [] 
#     for i in range(0, segmentation_array.shape[0], labels_per_sample):
#         modal_val = statistics.mode(segmentation_array[i:i+labels_per_sample])
#         downsample_segment.append(modal_val)
#     return np.array(downsample_segment)

def calc_confusion_matrix(prediction, truth, fhs):
    positive_pred = np.where(prediction == fhs)[0]
    negative_pred = np.where(prediction != fhs)[0]
    positive_truth = np.where(truth==fhs)[0]
    negative_truth = np.where(truth!=fhs)[0]
    tp = np.in1d(positive_pred, positive_truth, assume_unique=True).sum()
    tn = np.in1d(negative_pred, negative_truth, assume_unique=True).sum()
    fp = np.in1d(positive_pred, negative_truth, assume_unique=True).sum()
    fn = np.in1d(negative_pred, positive_truth, assume_unique=True).sum()

    return tp, tn, fp, fn

def update_confusion_matrix(prediction, truth, total, confusion_mat, fhs):
    tp, tn, fp, fn = calc_confusion_matrix(prediction, truth, fhs)
    total += len(prediction)
    confusion_mat[0] += tp
    confusion_mat[1] += tn
    confusion_mat[2] += fp 
    confusion_mat[3] += fn 
    return total, confusion_mat
   
def clip_results(prediction, truth):
    clipping_len = min(len(prediction), len(truth))
    prediction = prediction[:clipping_len]
    truth = truth[:clipping_len]
    return prediction, truth 


dp = DataPresentation() 
cnn_poor = dict()
hmm_poor = dict() 
cnn_good = dict()
hmm_good = dict()
cnn_totals, hmm_totals = [0,0,0,0], [0,0,0,0]
cnn_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
hmm_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]


for i in range(1, 6):
    with open('/Users/serenahuston/GitRepos/ThirdYearProject/Results/Full_Dataset_Results_20_02_2023/cnn_results_'+str(i), 'rb') as f:
        cnn_results = pickle.load(f)
    with open('/Users/serenahuston/GitRepos/ThirdYearProject/Results/Full_Dataset_Results_20_02_2023/hmm_results_'+str(i), 'rb') as f:
        hmm_results = pickle.load(f)


    cnn_accs = [] 
    hmm_accs = [] 
    for (file, value) in cnn_results.items():
        true_segmentations = get_true_segmentations(file) - 1 
        cnn_downsample_prediction = make_sample_prediction(value, math.ceil(len(true_segmentations)/(4000/50)))
        cnn_prediction = upsample_states(cnn_downsample_prediction, 50, 4000, len(true_segmentations)) 
        hmm_prediction = hmm_results.get(file)[0] - 1

        cnn_accs.append((cnn_prediction == true_segmentations).sum() / len(cnn_prediction))
        hmm_accs.append(hmm_results.get(file)[1])

        # plot_sound_and_segmentations(file.split(".")[0], hmm_results.get(file)[0], cnn_prediction)

        if cnn_accs[-1] < 0.5: 
            cnn_poor[file] = [cnn_prediction, cnn_accs]
            # plot_sound_and_segmentations(file.split(".")[0], hmm_prediction, cnn_prediction)
        if hmm_accs[-1] < 0.5: 
            hmm_poor[file] = [hmm_prediction, hmm_results.get(file)[1]] 

        if cnn_accs[-1] > 0.85: 
            cnn_good[file] = [cnn_prediction, cnn_accs]
            # plot_sound_and_segmentations(file.split(".")[0], hmm_prediction, cnn_prediction)
        if hmm_accs[-1] > 0.85: 
            hmm_good[file] = [hmm_prediction, hmm_results.get(file)[1]] 


        for i in range(0,4):
            cnn_totals[i], cnn_confusion_mat_all_classes[i] = update_confusion_matrix(cnn_prediction, true_segmentations, cnn_totals[i], cnn_confusion_mat_all_classes[i], i)
            hmm_totals[i], hmm_confusion_mat_all_classes[i] = update_confusion_matrix(hmm_prediction, true_segmentations, hmm_totals[i], hmm_confusion_mat_all_classes[i], i)

    # dp.plot_model_comp_box_plots(cnn_accs, hmm_accs, i)
    
    cnn_avg = np.mean(cnn_accs)
    hmm_avg = np.mean(hmm_accs)

    cnn_std = np.std(cnn_accs)
    hmm_std = np.std(hmm_accs)

    print(cnn_avg, hmm_avg)
    print(cnn_std, hmm_std)
# print(cnn_accs)
# calc_proportion_murmurs(cnn_poor)
# calc_proportion_murmurs(hmm_poor)

print([np.array(cnn_confusion_mat_all_classes[i])/cnn_totals[i] for i in range(4)])
print([np.array(hmm_confusion_mat_all_classes[i])/hmm_totals[i] for i in range(4)])

# print("SENSITIVITY")
# print("CNN", [calc_sensitivity(fhs[0], fhs[3]) for fhs in cnn_confusion_mat_all_classes])
# print("HMM", [calc_sensitivity(fhs[0], fhs[3]) for fhs in hmm_confusion_mat_all_classes])

# print("SPECIFICITY")
# print("CNN", [calc_specificity(fhs[1], fhs[2]) for fhs in cnn_confusion_mat_all_classes])
# print("HMM", [calc_sensitivity(fhs[1], fhs[2]) for fhs in hmm_confusion_mat_all_classes])

# print("PPV")
# print("CNN", [calc_ppv(fhs[0], fhs[2]) for fhs in cnn_confusion_mat_all_classes])
# print("HMM", [calc_ppv(fhs[0], fhs[2]) for fhs in hmm_confusion_mat_all_classes])

# print("NPV")
# print("CNN", [calc_npv(fhs[1], fhs[3]) for fhs in cnn_confusion_mat_all_classes])
# print("HMM", [calc_npv(fhs[1], fhs[3]) for fhs in hmm_confusion_mat_all_classes])

metrics = dict()
metrics["CNN"] = dict()
metrics["HMM"] = dict()

for (k,v) in metrics.items():
    if k == "CNN":
        metrics[k] = build_metric_dict(cnn_confusion_mat_all_classes)
    else:
        metrics[k] = build_metric_dict(hmm_confusion_mat_all_classes)

means = []
stds = [] 
for (model, results) in metrics.items():
    means.append([])
    stds.append([])
    print(model)
    for (metric, vals) in results.items():
        means[-1].append(np.mean(vals))
        stds[-1].append(np.std(vals))

print(means, stds)
# dp.plot_multi_bar_chart(["Sensitivity", "Specificity", "PPV", "NPV"], "Evaluation Metric", "Score", means, stds, ["CNN", "HMM"],
#                         "Comparison Between HMM and CNN of\nVarious Evaluation Metrics")

