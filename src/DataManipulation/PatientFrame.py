import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import glob 
import pandas as pd
import shutil

from Utilities.constants import * 

class PatientFrame():

    def __init__(self, file):
        self.file = file
        self.patient_frame = self.prepare_frame()
    
    def __len__(self):
        return len(self.patient_frame)

    def get_patient_data(self, patient_num):
        return self.patient_frame.where(self.patient_frame["Patient ID"] == patient_num).dropna(axis=0, how="all")

    def get_patient_audio_file_names(self, patient_num):
        wav_files = [] 
        file_regex = TRAINING_DATA_PATH + str(patient_num) + '_*.wav'
        for file in glob.glob(file_regex):
            wav_files.append(file)
        return sorted(wav_files)

    def prepare_frame(self):
        patient_frame = pd.read_csv(self.file)
        patient_frame["Age"].replace(AGES, inplace=True)
        patient_frame["Sex"].replace(SEXES, inplace=True)
        patient_frame["Pregnancy status"] = patient_frame["Pregnancy status"].astype(int)
        patient_frame["Recording locations"] = patient_frame["Recording locations:"].apply(lambda x: x.split("+"))
        patient_frame.drop("Recording locations:", inplace=True, axis=1)        
        return patient_frame 

    def get_num_audio_samples(self):
        return len(self.patient_frame["Recording locations"].sum())

    def get_patients_with_murmur_status(self, status):
        return (self.patient_frame.loc[self.patient_frame['Murmur'] == status])["Patient ID"]

    def get_murmur_status_by_patient(self, patientID):
        return (self.patient_frame.loc[self.patient_frame['Patient ID'].isin(patientID)])["Murmur"]

    def create_dataset_dir_from_IDs(self):
        dir = TRAINING_DATA_PATH + "DataSubset_" + str(len(self))
        os.mkdir(dir)
        for id in self.patient_frame["Patient ID"]:
            files = glob.glob(TRAINING_DATA_PATH + str(id) + "_*")
            for f in files:
                shutil.copy(f, dir)