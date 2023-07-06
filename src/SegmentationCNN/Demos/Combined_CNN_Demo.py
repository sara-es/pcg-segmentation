import sys, os
sys.path.append(os.path.join(sys.path[0], '..', '..'))

import math 
from scipy.io import wavfile

from SegmentationCNN.Models.Envelope_CNN.GitHubUNet import * 
from SegmentationCNN.Models.STFT_CNN.STFT_GitHubUNet import * 
from SegmentationCNN.Models.Envelope_CNN.DataPreprocessing import * 
from SegmentationCNN.Models.STFT_CNN.STFT_DataPreprocessing import * 
from Utilities.create_segmentation_array import *
from Utilities.prediction_helper_functions import * 
from DataManipulation.DataPresentation import * 
from Utilities.constants import *

dataset_dir = TRAINING_DATA_PATH
model_path = MODEL_PATH
saved_env_model = os.path.join(model_path, "model_weights_2022_env_cnn_64_8_3.pt")
saved_stft_model = os.path.join(model_path, "model_weights_2022_stft_cnn_3.pt")

def get_wav_and_tsv(patient_name):
    fs, recording = wavfile.read(os.path.join(dataset_dir, patient_name + ".wav"))
    tsv = np.loadtxt(os.path.join(dataset_dir, patient_name+ ".tsv"), delimiter="\t")

    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                tsv,
                                                                recording_frequency=4000,
                                                                feature_frequency=4000)
    
    try:
        return clipped_recording[0], segmentations[0], fs
    except:
        return [], [], 0

def load_models(saved_env_model, saved_stft_model):
    env_model = UNet()
    env_model.load_state_dict(torch.load(saved_env_model))
    env_model.eval()
    STFT_model = STFT_UNet(n_channels=8)
    STFT_model.load_state_dict(torch.load(saved_stft_model))
    STFT_model.eval()
    return env_model, STFT_model

def make_prediction(audio_data, fs, saved_env_model, saved_STFT_model, window=64, stride=8):
    predictions = [] 
    env_model, stft_model = load_models(saved_env_model, saved_STFT_model)
    # for recording in audio_data:
    env_dp = DataPreprocessing(audio_data, np.array([]), fs, window=window, stride=stride)
    stft_dp = STFT_DataPreprocessing(audio_data, np.array([]), fs, window=window*80, stride=stride*80)
    env_dp.extract_env_patches()
    stft_dp.extract_wav_patches_only()
    window_predictions = []
    for i in range(len(env_dp.env_patches)):
        env_probabilities = env_model(torch.tensor(env_dp.env_patches[i], requires_grad=True).type(torch.float32))
        stft_input =  torch.tensor(stft_dp.stft_patches[i], requires_grad=True).type(torch.float32)[None, :]
        stft_probabilities = stft_model(stft_input)[0]
        window_predictions.append(make_ensemble_window_prediction(env_probabilities, stft_probabilities))
    combined_windows = combine_windows(window_predictions, math.ceil(len(audio_data)/(fs/50)), window=window, stride=stride)
    predictions = upsample_states(combined_windows, 50, fs, len(audio_data)) + 1 
    return np.array(predictions) 


# Get files not in folds of these models
files = ["84996_TV", "50029_PV", "49653_MV"]
data_pres = DataPresentation()
results_dir = os.path.join(RESULTS_PATH, "Combined_CNN")

for file in files:
    wav, true_seg, fs = get_wav_and_tsv(file)
    predict_seg = make_prediction(wav, fs, saved_env_model, saved_stft_model, window=64, stride=8)
    print("---- PATIENT: " + file +" ----")
    print("Accuracy: " + str((true_seg==predict_seg).sum()/len(true_seg)))
    print("Plotting Results...")
    data_pres.plot_PCG_segmentations(file, results_dir, wav, true_seg, predict_seg)
    print("______________________________")