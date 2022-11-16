from constants import *
import glob
import numpy as np 
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from scipy.io import wavfile

os.chdir(TRAINING_DATA_PATH)

class AudioData(Dataset):

    def __init__(self, file):
        patient_frame = prepare_frame(file)

        self.num_recordings_per_patient = self.get_num_recordings_per_patient(patient_frame)
        self.patient_IDs = torch.tensor(self.get_vector_from_df(patient_frame, "Patient ID"))
   
        murmurs = self.get_vector_from_df(patient_frame, "Murmur")
        outcomes = self.get_vector_from_df(patient_frame, "Outcome")

        self.murmurs = torch.tensor(np.array(pd.Categorical(murmurs).codes), dtype=torch.long)
     
        self.outcomes = torch.tensor(np.array(pd.Categorical(outcomes).codes), dtype=torch.long)
        self.set_audio_tensor()

    def get_num_recordings_per_patient(self, df):
        return df["Recording locations"].str.len()

    def get_vector_from_df(self, df, col_name):
        return np.repeat(df[col_name], self.num_recordings_per_patient).reset_index(drop=True)

    def set_audio_tensor(self):
        
        audio_recordings = [] 
        
        self.wav_files = pd.Series(glob.glob(AUDIO_FILE_REGEX))
        self.num_recordings = len(self.wav_files)

        for file in self.wav_files:
            tensor, _ = torchaudio.load(TRAINING_DATA_PATH + file, num_frames=NUM_FRAMES)
            audio_recordings.append(tensor)
        
        self.audio_tensor = torch.cat(audio_recordings, dim=0)

    def __len__(self):
        return len(self.patient_IDs)

    def __getitem__(self, index):
        return self.audio_tensor[index], self.murmurs[index]

    def get_item_by_patient_ID(self, patient_ID):
        return self.__getitem__((self.patient_IDs == patient_ID).nonzero(as_tuple=True)[0])

    def get_audio_duration(self, wav_files):
        durations = []
        for w in wav_files:
            sample_rate, data = wavfile.read(w)
            len_data = len(data)  # holds length of the numpy array
            durations.append(len_data / sample_rate)
        return durations
