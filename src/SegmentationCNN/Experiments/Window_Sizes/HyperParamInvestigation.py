import sys, os
sys.path.append(os.path.join(sys.path[0], '..', '..', '..'))

import numpy as np 
import pickle
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

from Utilities.constants import * 
from DataManipulation.PatientFrame import * 
from DataManipulation.PatientFrame import PatientFrame
from DataManipulation.DataPresentation import DataPresentation 

from SegmentationCNN.Models.Envelope_CNN.GitHubUNet import UNet, init_weights
from SegmentationCNN.Models.Envelope_CNN.PatientInfo import * 
from SegmentationCNN.Models.Envelope_CNN.EarlyStopping import EarlyStopping

dataset_dir = TRAINING_DATA_PATH
csv_file = DATA_CSV_PATH
# dataset_dir = "/Users/serenahuston/GitRepos/Data/DataSubset_21_Patients"
# csv_file = "/Users/serenahuston/GitRepos/Data/training_data_subset_21.csv"

data_pres = DataPresentation()
fold_num = 1
data_pres_folder = ""

def set_up_model(window, stride):
    global model, optimiser, criterion 
    model = UNet()
    model_file = "model_weights_2016_" + str(window) + "_" + str(stride) + ".pt"
    # model.load_state_dict(torch.load(MODEL_PATH + model_file))
    model.apply(init_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()


def stratified_sample(csv_file, dataset_dir, folds=10):
    global fold_num, data_pres_folder
    pf = PatientFrame(csv_file)
    print("RUNNING")
    patient_info = PatientInfo(dataset_dir)
    patient_info.get_data()

    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    fold_num = 1

    windows = [64]
    strides = [8]

    for i in range(len(windows)):
        patient_info = PatientInfo(dataset_dir, window=windows[i], stride=strides[i])
        patient_info.get_data()
        data_pres_folder = DATA_PRESENTATION_PATH + "results_26_03_2023_" + str(windows[i]) + "_" + str(strides[i]) + "/"
        fold_num = 1
        for train_index, test_index in skf.split(pf.patient_frame["Patient ID"], pf.patient_frame["Murmur"]):
            if fold_num < 3:
                fold_num += 1
                print("SKIPPING")
                continue
            patients_train, patients_test = pf.patient_frame["Patient ID"][train_index], pf.patient_frame["Patient ID"][test_index]
            training_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_train)]
            val_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_test)]
            cnn_results =prep_CNN(training_df, val_df, windows[i], strides[i])
            # save_results(cnn_results, "cnn", fold_num, windows[i], strides[i])
            save_model(fold_num, windows[i], strides[i])
            fold_num += 1 
            break


def prep_CNN(training_df, val_df, window, stride):
    train_data = ConcatDataset(training_df["CNN_Data"])

    validation_data = ConcatDataset(val_df["CNN_Data"])
  
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=1, shuffle=True)

    set_up_model(window, stride)
    return train(train_loader, validation_loader, len(validation_data), window)        

def train(train_loader, validation_loader, validation_size, window, epochs=15, patience=5):
    global fold_num, data_pres, data_pres_folder
    
    avg_train_loss = []
    avg_validation_loss = [] 

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    accuracy_list = [] 
    model.train(True)

    epochs = 30 

    for epoch in range(epochs):
        training_loss = [] 
        validation_loss = [] 
        model.train()
        for x,y,name,ordering in train_loader:
            optimiser.zero_grad()
            yhat = model(x[0])
            loss = criterion(torch.t(yhat), y[0])
            training_loss.append(loss.item())
            loss.backward()
            optimiser.step()


        correct = 0 
        model.eval()
        results = dict() 
        for x_test, y_test, name, ordering in validation_loader:
            
            z = model(x_test[0])
            loss = criterion(torch.t(z), y_test[0])
            validation_loss.append(loss.item())

            softmax = F.softmax(z, dim=0)
            _, yhat = torch.max(softmax, 0)
        
            for i in range(1, yhat.shape[0]): 
                if yhat[i] != (yhat[i-1] + 1) % 4:
                    yhat[i] = yhat[i-1]
            
            correct += (yhat == y_test[0]).sum().item()      
            mean_acc = (yhat == y_test[0]).sum() / len(y_test[0])
            if results.get(name[0]) != None: 
                results[name[0]].append([yhat, mean_acc, ordering])
            else: 
                results[name[0]] = [[yhat, mean_acc, ordering]]
            
        accuracy = correct / (validation_size * window) 
        accuracy_list.append(accuracy)
        avg_train_loss.append(np.average(training_loss))
        avg_validation_loss.append(np.average(validation_loss))

        early_stopping(np.average(validation_loss), model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("HERE")
    data_pres.plot_loss_and_accuracy(avg_train_loss, avg_validation_loss, accuracy_list, data_pres_folder, fold_num)
    return results 
        

def save_results(results_dict, model, fold_num, window, stride):
    filename = model + "_" + str(window) + "_" + str(stride) + "_results_26_03_2023_" + str(fold_num)
    outfile = open(RESULTS_PATH + filename,'wb')
    pickle.dump(results_dict, outfile)
    outfile.close()

def save_model(fold_num, window, stride):
    global model
    # torch.save(model.state_dict(), "/Users/serenahuston/GitRepos/ThirdYearProject/Models/model_weights_2022_env_cnn_" + str(window) + "_" + str(stride) + "_" + str(fold_num) + ".pt")
    torch.save(model.state_dict(), "/Models/model_weights_2022_env_cnn_" + str(window) + "_" + str(stride) + "_" + str(fold_num) + ".pt")

stratified_sample(csv_file, dataset_dir)
