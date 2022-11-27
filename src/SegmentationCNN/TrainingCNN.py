import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset 
import sys 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold



sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from Utilities.constants import * 
from DataManipulation.PatientFrame import * 

from DataManipulation.PatientFrame import PatientFrame
from CNNData import CNNData 
from GitHubUNet import UNet, init_weights
from DataPreprocessing import DataPreprocessing
from utils import get_wavs_and_tsvs, get_wavs_and_tsvs_by_regex

# dataset_dir = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data"
dataset_dir = "/Users/serenahuston/GitRepos/Data/DataSubset_48"
# csv_file = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data.csv"
csv_file = "/Users/serenahuston/GitRepos/Data/training_data_subset.csv"

epoch_count = 0 

def set_up_model():
    global model, optimiser, criterion 
    model = UNet()
    model.apply(init_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

def get_dataset_by_IDs(patient_IDs):
    string_ids = np.array2string(patient_IDs.values, separator="|")
    file_pattern = "(" + string_ids[1:-1] + ")_*"

    wavs, tsvs, fs, names = get_wavs_and_tsvs_by_regex(file_pattern, TRAINING_DATA_PATH_2022 + "/training_data", return_names=True)

    x_patches = []
    y_patches = [] 

    for i in range(len(wavs)):
        dp = DataPreprocessing(wavs[i], tsvs[i], fs[i], names[i])
        if len(dp.wav) >0  and len(dp.segmentation_array) >0:
            x_patches += dp.extract_env_patches()
            y_patches += dp.extract_segmentation_patches()

    dataset = CNNData(np.array(x_patches), np.array(y_patches))
    
    return dataset

def get_patient_data_dict(dataset_dir):

    wavs, tsvs, fs, names = get_wavs_and_tsvs(dataset_dir, return_names=True)
    x_patches = []
    y_patches = [] 

    patient_data_dict = dict()
    for i in range(len(wavs)):
        dp = DataPreprocessing(wavs[i], tsvs[i], fs[i], names[i])
        patient_ID = int(names[i].split("_")[0])
        if len(dp.wav) >0  and len(dp.segmentation_array) >0:
            x_patches = dp.extract_env_patches()
            y_patches = dp.extract_segmentation_patches()
        
        if patient_data_dict.get(patient_ID):
            patient_data_dict[patient_ID].append(CNNData(np.array(x_patches), np.array(y_patches)))
        else:
            patient_data_dict[patient_ID] = [CNNData(np.array(x_patches), np.array(y_patches))]
       
    return patient_data_dict

def stratified_sample(csv_file, dataset_dir, folds=10):
    pf = PatientFrame(csv_file)

    patient_data_dict = get_patient_data_dict(dataset_dir)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
  
    for train_index, test_index in skf.split(pf.patient_frame["Patient ID"], pf.patient_frame["Murmur"]):
        x_train_fold, x_test_fold = pf.patient_frame["Patient ID"][train_index], pf.patient_frame["Patient ID"][test_index]

        train_data_list = [ConcatDataset(patient_data_dict[id]) for id in x_train_fold]
        train_data = ConcatDataset(train_data_list)
       
        validation_data_list = [ConcatDataset(patient_data_dict[id]) for id in x_test_fold]
        validation_data = ConcatDataset(validation_data_list)

        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
        validation_loader = DataLoader(dataset=validation_data, batch_size=1, shuffle=True)

        set_up_model()
        print("TRAINING")
        train(train_loader, validation_loader, len(validation_data))

def train(train_loader, validation_loader, validation_size, epochs=15):
    loss_list = []
    accuracy_list = [] 
    model.train(True)

    epochs = 20 

    for epoch in range(epochs):
        model.train()
        for x,y in train_loader:
            optimiser.zero_grad()
            yhat = model(x[0])
            try:
                loss = criterion(torch.t(yhat), y[0])
                loss.backward()
                optimiser.step()
            except:
                # -1 value appearing here 
                print(yhat, y[0])
            

        correct = 0 
        model.train(False)
        for x_test, y_test in validation_loader:
            
            z = model(x_test[0])
            softmax = F.softmax(z, dim=1)
            _, yhat = torch.max(softmax, 0)
   
            for i in range(1, yhat.shape[0]): 
                if yhat[i] != (yhat[i-1] + 1) % 4:
                    yhat[i] = yhat[i-1]
            
            correct += (yhat == y_test[0]).sum().item()      
        accuracy = correct / (validation_size * 64) 
        loss_list.append(loss.data)
        accuracy_list.append(accuracy)

    plot_loss_and_accuracy(loss_list, accuracy_list)
        

def plot_loss_and_accuracy(loss, accuracy):
    global epoch_count
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle('Model Loss VS Accuracy Across Epochs', fontsize=12)
    ax1.plot(loss)
    ax2.plot(accuracy)
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Epochs")
    ax1.grid()
    ax2.grid()
    plt.savefig("/Users/serenahuston/GitRepos/ThirdYearProject/DataPresentation/" + "Loss VS Accuracy" + str(epoch_count))
    
    epoch_count += 1



# dataset = get_dataset()
# print(dataset.keys())
# cross_validation(dataset=dataset)

stratified_sample(csv_file, dataset_dir)
