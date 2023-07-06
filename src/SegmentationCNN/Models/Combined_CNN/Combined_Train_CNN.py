import sys, os
sys.path.append(os.path.join(sys.path[0], '..', '..', '..'))

import numpy as np 
import pickle
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from Utilities.constants import * 
from DataManipulation.PatientFrame import * 
from DataManipulation.PatientFrame import PatientFrame
from Combined_PatientInfo import * 
from SegmentationCNN.Models.STFT_CNN.STFT_GitHubUNet import STFT_UNet, init_weights
from SegmentationCNN.Models.Envelope_CNN.GitHubUNet import UNet

dataset_dir = TINY_TEST_DATA_PATH #TRAINING_DATA_PATH
csv_file = TINY_TEST_CSV_PATH #DATA_CSV_PATH
# dataset_dir = "/Users/serenahuston/GitRepos/Data/DataSubset_21_Patients"
# csv_file = "/Users/serenahuston/GitRepos/Data/training_data_subset_21.csv"


epoch_count = 0 

def set_up_STFT_model():
    global stft_model, stft_optimiser, stft_criterion 
    stft_model = STFT_UNet(n_channels=15)
    stft_model.apply(init_weights)
    stft_optimiser = torch.optim.Adam(stft_model.parameters(), lr=0.0001)
    stft_criterion = nn.CrossEntropyLoss()

def set_up_env_model():
    global env_model, env_optimiser, env_criterion 
    env_model = UNet()
    env_model.load_state_dict(torch.load(MODEL_PATH + "model_weights_2016_64_8.pt"))
    # env_model.apply(init_weights)
    env_optimiser = torch.optim.Adam(env_model.parameters(), lr=0.0001)
    env_criterion = nn.CrossEntropyLoss()


def stratified_sample(csv_file, dataset_dir, folds=5):
    pf = PatientFrame(csv_file)
    print("RUNNING")
    patient_info = Combined_PatientInfo(dataset_dir)
    patient_info.get_data()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    fold_num = 1
    for train_index, test_index in skf.split(pf.patient_frame["Patient ID"], pf.patient_frame["Murmur"]):
        patients_train, patients_test = pf.patient_frame["Patient ID"][train_index], pf.patient_frame["Patient ID"][test_index]
        training_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_train)]
        val_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_test)]
        env_CNN_train_loader, env_CNN_validation_loader =prep_env_CNN(training_df, val_df)
        STFT_CNN_train_loader, STFT_CNN_validation_loader =prep_STFT_CNN(training_df, val_df)
        results = train_models(env_CNN_train_loader, env_CNN_validation_loader, STFT_CNN_train_loader, STFT_CNN_validation_loader)
        save_results(results, "stft_env_cnn", fold_num)
        # fold_num += 1 


def prep_STFT_CNN(training_df, val_df):
    train_data = ConcatDataset(training_df["STFT_CNN_Data"])
    validation_data = ConcatDataset(val_df["STFT_CNN_Data"])   
  
    train_loader = DataLoader(dataset=train_data, batch_size=50, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=1, shuffle=True)

    set_up_STFT_model()
    return train_loader, validation_loader

def prep_env_CNN(training_df, val_df):
    train_data = ConcatDataset(training_df["Env_CNN_Data"])
    validation_data = ConcatDataset(val_df["Env_CNN_Data"])   
  
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=1, shuffle=True)

    set_up_env_model()
    return train_loader, validation_loader
    
        
def train_STFT_model(train_loader):
    global stft_model, stft_criterion, stft_optimiser
    train_loss = [] 
    stft_model.train()
    for x,y,name,ordering in train_loader:
        stft_optimiser.zero_grad()
        yhat = stft_model(x)
        loss = stft_criterion(yhat, y)
        # print(loss)
        train_loss.append(loss.item())
        loss.backward()
        stft_optimiser.step()
    return train_loss 


def train_env_model(train_loader):
    global env_model, env_criterion, env_optimiser
    train_loss = [] 
    env_model.train()
    for x,y,name,ordering in train_loader:
        env_optimiser.zero_grad()
        yhat = train_loss(x[0])
        loss = env_criterion(torch.t(yhat), y[0])
        train_loss.append(loss.item())
        loss.backward()
        env_optimiser.step()
    return train_loss 

def evaluate_stft_model(validation_loader):
    global stft_model 
    stft_predictions = dict() 
    for x_test, y_test, name, ordering in validation_loader:
        yhat = stft_model(x_test)
        softmax = F.softmax(yhat, dim=1)
        if stft_predictions.get(name) == None:
            stft_predictions[name] = [] 
        stft_predictions[name].append([ordering, softmax])
    return stft_predictions

def evaluate_env_model(validation_loader):
    global env_model 
    env_predictions = dict() 
    for x_test, y_test, name, ordering in validation_loader:
        yhat = env_model(x_test[0])
        softmax = F.softmax(yhat, dim=1)
        if env_predictions.get(name) == None:
            env_predictions[name] = [] 
        env_predictions[name].append([ordering, softmax])
    return env_predictions

def train_models(env_CNN_train_loader, env_CNN_validation_loader, STFT_CNN_train_loader, STFT_CNN_validation_loader):
    
    avg_train_loss = []
    avg_validation_loss = [] 

    accuracy_list = [] 
    

    epochs = 20

    for epoch in range(epochs):
        global stft_model, env_model 
        validation_loss = [] 
        
        stft_train_loss = train_STFT_model(STFT_CNN_train_loader)
        env_train_loss = train_env_model(env_CNN_train_loader)

        correct = 0 
        stft_model.eval()
        env_model.eval()
        num_test_points = 0
        results = dict() 
                

        env_predictions = dict() 
        for x_test, y_test, name, ordering in env_CNN_validation_loader:
            yhat = env_model(x_test[0])
            softmax = F.softmax(yhat, dim=1)
            if env_predictions.get(name) == None:
                env_predictions[name] = dict()
            env_predictions[name][ordering] = softmax

        for x_test, y_test, name, ordering in STFT_CNN_validation_loader:
            yhat = env_model(x_test)
            stft_softmax = F.softmax(yhat, dim=0)
            env_softmax = env_predictions[name][ordering]
            combined_softmax = F.softmax(stft_softmax + env_softmax, dim=0)
            _, yhat = torch.max(combined_softmax, 1)

            for i in range(1, yhat.shape[0]): 
                if yhat[i] != (yhat[i-1] + 1) % 4:
                    yhat[i] = yhat[i-1]
            
            correct += (yhat == y_test[0]).sum().item()      
            mean_acc = (yhat == y_test[0]).sum() / len(y_test[0])
            num_test_points += len(y_test[0])
            if results.get(name[0]) != None: 
                results[name[0]].append([yhat, mean_acc, ordering])
            else: 
                results[name[0]] = [[yhat, mean_acc, ordering]]
            
        accuracy = correct / (num_test_points) 
        accuracy_list.append(accuracy)
        avg_validation_loss.append(np.average(validation_loss))


    print("HERE")
    plot_loss_and_accuracy(stft_train_loss, env_train_loss, avg_validation_loss, accuracy_list)
    return results 
        

def plot_loss_and_accuracy(stft_train_loss, env_train_loss, valid_loss, accuracy):
    global epoch_count
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fig.set_size_inches(24, 10)
    fig.suptitle('Model Loss VS Accuracy Across Epochs', fontsize=12)

    ax1.plot(stft_train_loss, color="#611d91")
    ax2.plot(env_train_loss, color="#611d91")
    ax3.plot(valid_loss, color="#a260d1")
    ax4.plot(accuracy, color="#a260d1")

    ax1.set_ylabel("STFT CNN Training Loss")
    ax2.set_ylabel("Envelope CNN Training Loss")
    ax3.set_ylabel("Ensemble Validation Loss")
    ax4.set_ylabel("Validation Accuracy")
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Epochs")
    ax3.set_xlabel("Epochs")
    ax4.set_xlabel("Epochs")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    fig_name = DATA_PRESENTATION_PATH + "Loss_VS_Accuracy_STFT_Env" + str(epoch_count)
    print(fig_name)
    plt.savefig(fig_name)
    
    epoch_count += 1

def save_results(results_dict, model, fold_num):
    outfile = open(RESULTS_PATH + model+ "results_" + str(fold_num),'wb')
    pickle.dump(results_dict, outfile)
    outfile.close()

stratified_sample(csv_file, dataset_dir)
