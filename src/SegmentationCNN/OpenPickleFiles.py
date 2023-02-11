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
from create_segmentation_array import *
from Utilities.constants import *

csv_file = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data.csv"


def plot_sound_and_segmentations(file_root, hmm_segmentations, cnn_segmentations):
    # plt.plot(downsample_segmentation_array(true_segments))
    
    fs, recording = sp.io.wavfile.read(TRAINING_DATA_PATH_2022 + "training_data/"+ file_root + ".wav")

    tsv = np.loadtxt(TRAINING_DATA_PATH_2022 + "training_data/" + file_root + ".tsv", delimiter="\t")

    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                tsv,
                                                                recording_frequency=fs,
                                                                feature_frequency=fs)


    hmm_segmentations = downsample_segmentation_array(hmm_segmentations)
    true_segmentations = downsample_segmentation_array(segmentations[0])-1
    # clipped_recording = resample(np.array(clipped_recording[0],dtype = float), 4000, 50)
    minimum = min(len(hmm_segmentations), len(cnn_segmentations), len(true_segmentations))

    time_4000 = np.linspace(0, len(clipped_recording[0]) / 4000, num=len(clipped_recording[0]))

    time_50 = np.linspace(0, minimum/50, num=minimum)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
    fig.set_size_inches(16, 8)
    fig.subplots_adjust(hspace=0)
    
    ax1.plot(time_4000, clipped_recording[0], color="#611d91")        
    ax2.plot(time_50, true_segmentations[:minimum], color="#611d91")     
    ax3.plot(time_50, hmm_segmentations[:minimum], color="#611d91")
    ax4.plot(time_50, cnn_segmentations[:minimum], color="#611d91")
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

def make_sample_prediction(file_results):
    overlappers = dict()
    sorted_windows = sorted(file_results,key=lambda l:l[2])
    for i in range(len(sorted_windows)):
        for j in range(len(sorted_windows[i][0])):
            if overlappers.get(j+8*i) == None:
                overlappers[j+8*i] = []
            overlappers[j+8*i].append(sorted_windows[i][0][j].item())
    
    prediction = [statistics.mode(value) for (key, value) in overlappers.items()] 
    return np.array(prediction)

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

def downsample_segmentation_array(segmentation_array):
    labels_per_sample = int(4000 / 50)
    downsample_segment = [] 
    for i in range(0, segmentation_array.shape[0], labels_per_sample):
        modal_val = statistics.mode(segmentation_array[i:i+labels_per_sample])
        downsample_segment.append(modal_val)
    return np.array(downsample_segment)

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
cnn_totals, hmm_totals = [0,0,0,0], [0,0,0,0]
cnn_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
hmm_confusion_mat_all_classes = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]

for i in range(1, 6):
    with open('/Users/serenahuston/GitRepos/ThirdYearProject/Results/cnn_results_'+str(i), 'rb') as f:
        cnn_results = pickle.load(f)
    with open('/Users/serenahuston/GitRepos/ThirdYearProject/Results/hmm_results_'+str(i), 'rb') as f:
        hmm_results = pickle.load(f)


    cnn_accs = [] 
    hmm_accs = [] 
 
    for (file, value) in cnn_results.items():
        segmentations = get_true_segmentations(file)
        cnn_prediction = make_sample_prediction(value)
        
        cnn_truth = downsample_segmentation_array(segmentations)-1
        hmm_truth = segmentations - 1 
        cnn_prediction, cnn_truth = clip_results(cnn_prediction, cnn_truth)
        hmm_prediction, hmm_truth = clip_results(np.array(hmm_results.get(file)[0])-1, hmm_truth)

        cnn_accs.append((cnn_prediction == cnn_truth).sum() / len(cnn_prediction))
        hmm_accs.append(hmm_results.get(file)[1])

        # if cnn_accs[-1] < 0.5: 
        #     # cnn_poor[file] = [cnn_prediction, cnn_truth]
        #     plot_sound_and_segmentations(file.split(".")[0], hmm_prediction, cnn_prediction)
        # if hmm_accs[-1] < 0.5: 
        #     hmm_poor[file] = [hmm_prediction, hmm_truth] 



        # for i in range(0,4):
        #     cnn_totals[i], cnn_confusion_mat_all_classes[i] = update_confusion_matrix(cnn_prediction, cnn_truth, cnn_totals[i], cnn_confusion_mat_all_classes[i], i)
        #     hmm_totals[i], hmm_confusion_mat_all_classes[i] = update_confusion_matrix(hmm_prediction, hmm_truth, hmm_totals[i], hmm_confusion_mat_all_classes[i], i)


    # dp.plot_model_comp_box_plots(cnn_accs, hmm_accs, i)
    
    # cnn_avg = np.mean(cnn_accs)
    # hmm_avg = np.mean(hmm_accs)

    # cnn_std = np.std(cnn_accs)
    # hmm_std = np.std(hmm_accs)

   
print(cnn_accs)
# calc_proportion_murmurs(cnn_poor)
# calc_proportion_murmurs(hmm_poor)

# print([np.array(cnn_confusion_mat_all_classes[i])/cnn_totals[i] for i in range(4)])
# print([np.array(hmm_confusion_mat_all_classes[i])/hmm_totals[i] for i in range(4)])

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

# metrics = dict()
# metrics["CNN"] = dict()
# metrics["HMM"] = dict()


# for (k,v) in metrics.items():
#     if k == "CNN":
#         metrics[k] = build_metric_dict(cnn_confusion_mat_all_classes)
#     else:
#         metrics[k] = build_metric_dict(hmm_confusion_mat_all_classes)

# means = []
# stds = [] 
# for (model, results) in metrics.items():
#     means.append([])
#     stds.append([])
#     print(model)
#     for (metric, vals) in results.items():
#         means[-1].append(np.mean(vals))
#         stds[-1].append(np.std(vals))

# print(means, stds)
# dp.plot_multi_bar_chart(["Sensitivity", "Specificity", "PPV", "NPV"], "Evaluation Metric", "Score", means, stds, ["CNN", "HMM"],
#                         "Comparison Between HMM and CNN of\nVarious Evaluation Metrics")

