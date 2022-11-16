from DataPresentation import DataPresentation
from PhysionetCode import helper_code
from constants import *
import numpy as np 
import os
import pandas as pd
from scipy.io import wavfile
from PatientFrame import PatientFrame
from matplotlib import pyplot as plt

os.chdir(TRAINING_DATA_PATH)

pf = PatientFrame(TRAINING_DATA_PATH + "training_data.csv")

def load_tsv_files_for_patient_location(patient_ID, location):
    tsv_file_path = TRAINING_DATA_PATH + "/training_data/" + patient_ID + "_" + location + ".tsv"
    if os.path.exists(tsv_file_path):
        return pd.read_csv(tsv_file_path, sep='\t').to_numpy()
    return None 

def segment_audio_for_patient_location(patient_ID, location):
    wav_file_path = TRAINING_DATA_PATH + "/training_data/" + patient_ID + "_" + location + ".wav"
    if os.path.exists(wav_file_path):
        recording, freq = helper_code.load_wav_file(wav_file_path)
    return recording, freq

def get_audio_duration(self, wav_file_paths):
        durations = []
        for w in wav_file_paths:
            audio, sample_rate = helper_code.load_wav_file(w)
            durations.append(len(audio) / sample_rate)
        return durations

fhs_locs = load_tsv_files_for_patient_location("2530", "AV")

dp = DataPresentation()
dp.plot_patient_audio_file_with_fhs_locs("2530", TRAINING_DATA_PATH + "/training_data/2530_AV.wav", fhs_locs[:,1])