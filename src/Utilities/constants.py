import pandas as pd
import math
from torch.utils.data import random_split

AGES = {"Neonate" : 0.5, "Infant": 6, "Child" : 72, "Adolescent": 180, "Young Adult": 240}
AUDIO_FILE_REGEX = "training_data/*.wav"
SEGMENTATION_FILE_REGEX = "training_data/*.tsv"
SEXES = {"Male" : 0, "Female": 1}
LOCATIONS = ["AV", "PV", "TV", "MV"]
MURMUR_PRESENCE = ["Present", "Absent", "Unknown"]
OUTCOMES = ["Normal", "Abnormal"]
NUM_FRAMES = 1000 
DATA_PATH = "/Users/serenahuston/GitRepos/Data/"
TRAINING_DATA_PATH_2022 = DATA_PATH + "PhysioNet_2022/"
DATA_PRESENTATION_PATH = "/Users/serenahuston/GitRepos/python-classifier-2022/DataPresentation/"

def prepare_frame(file):
    patient_frame = pd.read_csv(file)
    patient_frame["Age"].replace(AGES, inplace=True)
    patient_frame["Sex"].replace(SEXES, inplace=True)
    patient_frame["Pregnancy status"] = patient_frame["Pregnancy status"].astype(int)
    patient_frame["Recording locations"] = patient_frame["Recording locations:"].apply(lambda x: x.split("+"))
    patient_frame.drop("Recording locations:", inplace=True, axis=1)        
    return patient_frame 

def split_data(dataset, train_data_split):
    dataset_size = len(dataset)
    train_data_len = math.floor(dataset_size * train_data_split)
    train_dataset, val_dataset = random_split(dataset, (train_data_len, dataset_size-train_data_len))
    return train_dataset, val_dataset 