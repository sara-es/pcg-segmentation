import sys 
import math 
import os 
import scipy as sp

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from SegmentationCNN.Models.STFT_CNN.STFT_GitHubUNet import * 
from SegmentationCNN.Models.STFT_CNN.STFT_DataPreprocessing import * 
from Utilities.create_segmentation_array import *
from Utilities.prediction_helper_functions import * 
from DataManipulation.DataPresentation import * 

dataset_dir = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data"
model_path = "/Users/serenahuston/GitRepos/ThirdYearProject/Models/Demos/STFT_CNN/model_weights_2022_stft_cnn_5.pt"

def get_wav_and_tsv(patient_name):
    fs, recording = sp.io.wavfile.read(os.path.join(dataset_dir, patient_name + ".wav"))
    tsv = np.loadtxt(os.path.join(dataset_dir, patient_name +  ".tsv"), delimiter="\t")

    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                tsv,
                                                                recording_frequency=4000,
                                                                feature_frequency=4000)
    
    try:
        return clipped_recording[0], segmentations[0], fs
    except:
        return [], [], 0

def load_model(saved_model):
    model = STFT_UNet(n_channels=8)
    model.load_state_dict(torch.load(saved_model))
    model.eval()
    return model 


def make_prediction(audio_data, fs, saved_model, window=5120, stride=640):
    predictions = [] 
    model = load_model(saved_model)
    # for recording in audio_data:
    dp = STFT_DataPreprocessing(audio_data, np.array([]), fs, window=window, stride=stride)
    dp.extract_wav_patches_only()
    window_predictions = []
    for patch in dp.stft_patches: 
        input = torch.tensor(patch, requires_grad=True).type(torch.float32)[None, :]
        window_probabilities = model(input)
        window_predictions.append(make_window_prediction(window_probabilities[0]))
    combined_windows = combine_windows(window_predictions, math.ceil(len(audio_data)/(fs/50)), window=64, stride=8)
    predictions = upsample_states(combined_windows, 50, fs, len(audio_data)) + 1 
    return np.array(predictions) 


# Get files not in folds of these models
files = ["84996_TV", "50029_PV", "49653_MV"]
data_pres = DataPresentation()

results_dir = "/Users/serenahuston/GitRepos/ThirdYearProject/DataPresentation/SegmentationModelPerformance/Demos/STFT_CNN/"
for file in files:
    wav, true_seg, fs = get_wav_and_tsv(file)
    predict_seg = make_prediction(wav, fs, model_path, window=5120, stride=640)
    print("---- PATIENT: " + file +" ----")
    print("Accuracy: " + str((true_seg==predict_seg).sum()/len(true_seg)))
    print("Plotting Results...")
    data_pres.plot_PCG_segmentations(file, results_dir, wav, true_seg, predict_seg)
    print("______________________________")
