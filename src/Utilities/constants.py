import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import pandas as pd
import math
from torch.utils.data import random_split

AGES = {"Neonate" : 0.5, "Infant": 6, "Child" : 72, "Adolescent": 180, "Young Adult": 240}
AUDIO_FILE_REGEX = "*.wav"
SEGMENTATION_FILE_REGEX = "*.tsv"
SEXES = {"Male" : 0, "Female": 1}
LOCATIONS = ["AV", "PV", "TV", "MV"]
MURMUR_PRESENCE = ["Present", "Absent", "Unknown"]
OUTCOMES = ["Normal", "Abnormal"]
NUM_FRAMES = 1000 
# DATA_PATH = "/Users/serenahuston/GitRepos/Data/"
TRAINING_DATA_PATH = os.path.join("physionet.org","files","circor-heart-sound","1.0.3","training_data")
TINY_TEST_DATA_PATH = os.path.join("physionet.org","files","circor-heart-sound","1.0.3","tiny_test")  # for a small test set, 10 patients
DATA_CSV_PATH = os.path.join("physionet.org","files","circor-heart-sound","1.0.3","training_data.csv")
TINY_TEST_CSV_PATH = os.path.join("physionet.org","files","circor-heart-sound","1.0.3","tiny_test","tiny_test_data.csv") # for a small test set, 10 patients
# TRAINING_DATA_PATH_2022 = DATA_PATH + "PhysioNet_2022/"
# DATA_PRESENTATION_PATH = "/Users/serenahuston/GitRepos/ThirdYearProject/DataPresentation/"
DATA_PRESENTATION_PATH = "DataPresentation"
# RESULTS_PATH = "/Users/serenahuston/GitRepos/ThirdYearProject/Results/"
RESULTS_PATH = os.path.join("src","SegmentationCNN","Results")
# MODEL_PATH = "/Users/serenahuston/GitRepos/ThirdYearProject/Models/"
MODEL_PATH = os.path.join("src","SegmentationCNN","Models")

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