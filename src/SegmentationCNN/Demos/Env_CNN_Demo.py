import sys 
import math 
import os 
import scipy as sp
import pickle

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from SegmentationCNN.Models.Envelope_CNN.GitHubUNet import * 
from SegmentationCNN.Models.Envelope_CNN.DataPreprocessing import * 
from Utilities.create_segmentation_array import *
from Utilities.prediction_helper_functions import * 
from DataManipulation.DataPresentation import * 

dataset_dir = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data"
model_path = "/Users/serenahuston/GitRepos/ThirdYearProject/Models/Demos/Env_CNN/model_weights_2022_env_cnn_128_16_5.pt"

def get_wav_and_tsv(patient_name):
    fs, recording = sp.io.wavfile.read(os.path.join(dataset_dir, patient_name + ".wav"))
    tsv = np.loadtxt(os.path.join(dataset_dir, patient_name + ".tsv"), delimiter="\t")

    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                tsv,
                                                                recording_frequency=4000,
                                                                feature_frequency=4000)
    
    try:
        return clipped_recording[0], segmentations[0], fs
    except:
        return [], [], 0

def load_model(model_path):
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model 


def make_prediction(audio_data, fs, model_path, window=128, stride=16):
    predictions = [] 
    model = load_model(model_path)
    dp = DataPreprocessing(audio_data, np.array([]), fs, window=window, stride=stride)
    dp.extract_env_patches()
    window_predictions = []
    for patch in dp.env_patches: 
        window_probabilities = model(torch.tensor(patch, requires_grad=True).type(torch.float32))
        window_predictions.append(make_window_prediction(window_probabilities))
    combined_windows = combine_windows(window_predictions, math.ceil(len(audio_data)/(fs/50)), window=window, stride=stride)
    predictions = upsample_states(combined_windows, 50, fs, len(audio_data)) + 1 
    return np.array(predictions) 



# Get files not in folds of these models
files = ["84996_TV", "50029_PV", "49653_MV"]

dp = DataPresentation()

results_dir = "/Users/serenahuston/GitRepos/ThirdYearProject/DataPresentation/SegmentationModelPerformance/Demos/Env_CNN/"
for file in files:
    wav, true_seg, fs = get_wav_and_tsv(file)
    predict_seg = make_prediction(wav, fs, model_path, window=128, stride=16)
    print("---- PATIENT: " + file +" ----")
    print("Accuracy: " + str((true_seg==predict_seg).sum()/len(true_seg)))
    print("Plotting Results...")
    dp.plot_PCG_segmentations(file, results_dir, wav, true_seg, predict_seg)
    print("______________________________")
