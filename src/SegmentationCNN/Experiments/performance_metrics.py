import sys, os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

import numpy as np
import scipy as sp 

from DataManipulation.PatientFrame import PatientFrame
from Utilities.constants import * 
from Utilities.create_segmentation_array import * 

CSV_FILE = DATA_CSV_PATH

def calc_proportion_murmurs(files):
    pf = PatientFrame(CSV_FILE)
    murmurs = pf.get_murmur_status_by_patient([int(file.split("_")[0]) for file in files])
    present = (murmurs == 'Present').sum()
    unknown = (murmurs == 'Unknown').sum()
    absent = (murmurs == 'Absent').sum()
    print(present, unknown, absent)

def calc_sensitivity(tp, fn):
    return tp/(tp+fn)

def calc_specificity(tn, fp):
    return tn/(tn+fp)

def calc_ppv(tp, fp):
    return tp/(tp+fp)

def calc_npv(tn, fn):
    return tn/(fn+tn)

def calc_f1(tp, fp, fn):
    return tp / (tp + (0.5 * (fp + fn)))

def build_metric_dict(confusion_mat):
    metrics = dict()
    metrics["Sensitivity"] = [calc_sensitivity(fhs[0], fhs[3]) for fhs in confusion_mat]
    metrics["PPV"] = [calc_ppv(fhs[0], fhs[2]) for fhs in confusion_mat]
    metrics["F1"] = [calc_f1(fhs[0], fhs[2], fhs[3]) for fhs in confusion_mat]
    return metrics 

def update_confusion_matrix(prediction, truth, total, confusion_mat, fhs):
    tp, tn, fp, fn = calc_confusion_matrix(prediction, truth, fhs)
    total += len(prediction)
    confusion_mat[0] += tp
    confusion_mat[1] += tn
    confusion_mat[2] += fp 
    confusion_mat[3] += fn 
    return total, confusion_mat

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

def get_true_segmentations(file, return_recording=False):
    patientID = file.split(".")[0]
    fs, recording = sp.io.wavfile.read(os.path.join(TRAINING_DATA_PATH, file))
    tsv = np.loadtxt(TRAINING_DATA_PATH + patientID + ".tsv", delimiter="\t")
    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                    tsv,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=4000)
    if return_recording: 
        return np.array(clipped_recording[0]), np.array(segmentations[0])
    else:
        return np.array(segmentations[0])
    
