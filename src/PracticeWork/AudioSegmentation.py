from constants import *
import csv
import glob
import numpy as np 
import os
import pandas as pd
from pydub import AudioSegment
import torch
import torchaudio
from torch.utils.data import Dataset
from scipy.io import wavfile
import array 

os.chdir(TRAINING_DATA_PATH)

class SegmentedAudio(Dataset):



    def __init__(self, file):
        patient_frame = prepare_frame(file)

        self.num_recordings_per_patient = self.get_num_recordings_per_patient(patient_frame)
        self.patient_IDs = torch.tensor(self.get_vector_from_df(patient_frame, "Patient ID"))

        self.set_audio_tensor()

    def get_num_recordings_per_patient(self, df):
        return df["Recording locations"].str.len()

    def get_murmur_locations(self, df, patient_ID):
        return df["Recording locations"].where(df["Patient ID"] == patient_ID)

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

    def get_audio_segments(self):
        self.audio_segments = []
        self.segment_classes = [] 
        self.wav_files = pd.Series(glob.glob(AUDIO_FILE_REGEX))
        self.segmentation_files = pd.Series(glob.glob(SEGMENTATION_FILE_REGEX))
        self.num_recordings = len(self.wav_files)

        for a_file, s_file in zip(self.wav_files, self.segmentation_files):
            # tensor, _ = torchaudio.load(TRAINING_DATA_PATH + a_file)
            segments = self.get_segment_data(s_file)

            audio_segment_list, classes = self.split_audio_file(a_file, segments)
            self.audio_segments.extend(audio_segment_list)
            self.segment_classes.extend(classes)
   

    def split_audio_file(self, audio_file, segments):
  
        audio_segment_list = []
        segment_classes = []
        whole_audio = AudioSegment.from_wav(TRAINING_DATA_PATH + audio_file)
        for i in range(len(segments)):
            t1 = float(segments[i][0])*1000 
            t2 = float(segments[i][1])*1000
            tensor_segment = torch.tensor(whole_audio[t1:t2].get_array_of_samples(), dtype=torch.float)
            audio_segment_list.append(tensor_segment)
            segment_classes.append(int(segments[i][2]))
        print("WHOLE LIST LENGTH")
        print(len(audio_segment_list))
        print(len(audio_segment_list[0]))
        return audio_segment_list, segment_classes 

    def get_segment_data(self, segmentation_file):
        file = open(TRAINING_DATA_PATH + segmentation_file)
        csvreader = csv.reader(file)
        segments = [] 
        for row in csvreader:
            segments.append(row[0].split("\t"))
        return segments

a = SegmentedAudio("training_data.csv")
a.get_audio_segments()
