import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

from Utilities.constants import *

import glob
import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torchaudio


os.chdir(TRAINING_DATA_PATH)

class PatientData(Dataset):

    def __init__(self, file):
        self.file = file
        self.patient_frame = self.prepare_frame()
            
        # Output Labels
        self.outcomes = self.patient_frame.pop("Outcome")
        self.murmurs = self.patient_frame.pop("Murmur")


    def __getitem__(self, index):
        return self.patient_frame[index], self.outcomes[index]
    
    def __len__(self):
        return len(self.patient_frame)

    def get_patient_data(self, patient_num):
        return self.patient_frame.where(self.patient_frame["Patient ID"] == patient_num)

    def get_patients_with_murmur_label(self, label):
        return self.patient_frame.where(self.murmurs == label).dropna(how="all")

    def get_patient_audio_file_names(self, patient_num):
        wav_files = [] 
        file_regex = TRAINING_DATA_PATH + str(patient_num) + '_*.wav'
        for file in glob.glob(file_regex):
            wav_files.append(file)
        return sorted(wav_files)

    def convert_patient_df_to_feature_matrix(self):
        feature_df = self.patient_frame[["Age", "Sex", "Height", "Weight", "Pregnancy status"]]
        return feature_df.fillna(feature_df.mean()).to_numpy()

    def prepare_frame(self):
        patient_frame = pd.read_csv(self.file)
        patient_frame["Age"].replace(AGES, inplace=True)
        patient_frame["Sex"].replace(SEXES, inplace=True)
        patient_frame["Pregnancy status"] = patient_frame["Pregnancy status"].astype(int)
        patient_frame["Recording locations"] = patient_frame["Recording locations:"].apply(lambda x: x.split("+"))
        patient_frame.drop("Recording locations:", inplace=True, axis=1)        
        return patient_frame 

    def get_audio_tensor_from_file(self, filename):
        if os.path.isfile(TRAINING_DATA_PATH + filename):
            return torchaudio.load(TRAINING_DATA_PATH + filename, num_frames=1000)
        else:
            return torch.zeros(1000), None

    def get_num_audio_samples(self):
        return len(self.patient_frame["Recording locations"].sum())

    def murmurs_to_numerical_classes(self):
        self.murmurs = pd.Categorical(self.murmurs)
        return self.murmurs.codes

    def outcomes_to_numerical_classes(self):
        self.outcomes = pd.Categorical(self.outcomes)
        return self.outcomes.codes


