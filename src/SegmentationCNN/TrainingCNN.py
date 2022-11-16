


import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.utils.data import Subset 
import sys 
import matplotlib.pyplot as plt 



sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")


from DataManipulation.PatientFrame import PatientFrame
from CNNData import CNNData 
from GitHubUNet import UNet, init_weights
from DataPreprocessing import DataPreprocessing
from utils import get_wavs_and_tsvs

# dataset_file = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data"
dataset_file = "/Users/serenahuston/GitRepos/Data/Data_297_Samples"

epoch_count = 0 

def set_up_model():
    global model, optimiser, criterion 
    model = UNet()
    model.apply(init_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

def get_dataset():

    wavs, tsvs, fs, names = get_wavs_and_tsvs(dataset_file,
                                    return_names=True)

    
    x_patches = []
    y_patches = [] 

    for i in range(len(wavs)):
        dp = DataPreprocessing(wavs[i], tsvs[i], fs[i], names[i])
        if len(dp.wav) >0  and len(dp.segmentation_array) >0:
            x_patches += dp.extract_env_patches()
            y_patches += dp.extract_segmentation_patches()

    dataset = CNNData(np.array(x_patches), np.array(y_patches))
    
    return dataset


def cross_validation(dataset=None,k_fold=10):
    
    # train_score = pd.Series()
    # val_score = pd.Series()
    
    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)

    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_set = Subset(dataset,train_indices)
        val_set = Subset(dataset,val_indices)

        train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
        validation_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=True)
        set_up_model()
        train(train_loader, validation_loader, len(val_set))

def train(train_loader, validation_loader, validation_size, epochs=15):
    loss_list = []
    accuracy_list = [] 
    model.train(True)
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
        accuracy = correct / (validation_size * 256) 
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



dataset = get_dataset()
cross_validation(dataset=dataset)
