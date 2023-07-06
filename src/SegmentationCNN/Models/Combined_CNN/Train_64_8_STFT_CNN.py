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
from SegmentationCNN.Models.STFT_CNN.STFT_PatientInfo import * 
from SegmentationCNN.Models.STFT_CNN.STFT_GitHubUNet import STFT_UNet, init_weights

dataset_dir = TRAINING_DATA_PATH
csv_file = DATA_CSV_PATH
# dataset_dir = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data"
# csv_file = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data.csv"

epoch_count = 0 

def set_up_model():
    global model, optimiser, criterion 
    model = STFT_UNet(n_channels=8)
    model.apply(init_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()


def stratified_sample(csv_file, dataset_dir, folds=5):
    print("Loading data and initializing trials...")
    pf = PatientFrame(csv_file)
    patient_info = PatientInfo_STFT(dataset_dir, window=5120, stride=640)
    patient_info.get_data()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    fold_num = 1
    for train_index, test_index in skf.split(pf.patient_frame["Patient ID"], pf.patient_frame["Murmur"]):
        print(f"#### FOLD {fold_num} ####")
        patients_train, patients_test = pf.patient_frame["Patient ID"][train_index], pf.patient_frame["Patient ID"][test_index]
        training_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_train)]
        val_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_test)]
        print(f"Training CNN...")
        cnn_results, avg_validation_loss, avg_train_loss, accuracy_list =prep_CNN(training_df, val_df)
        print(f"Saving results...")
        save_results(cnn_results, "stft_cnn_for_ensemble_64_8", fold_num)
        save_epoch_stats(avg_validation_loss, avg_train_loss, accuracy_list, "stft_cnn_for_ensemble_64_8", fold_num)
        save_model(fold_num)
        fold_num += 1 


def prep_CNN(training_df, val_df):
    train_data = ConcatDataset(training_df["STFT_Data"])
    validation_data = ConcatDataset(val_df["STFT_Data"])   
  
    train_loader = DataLoader(dataset=train_data, batch_size=50, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=1, shuffle=True)

    set_up_model()
    return train(train_loader, validation_loader, len(validation_data))
        


def train(train_loader, validation_loader, validation_size, epochs=15):
    
    avg_train_loss = []
    avg_validation_loss = [] 

    

    accuracy_list = [] 
    model.train(True)

    epochs = 20

    for epoch in range(epochs):
        training_loss = [] 
        validation_loss = [] 
        model.train()
        for x,y,name,ordering in train_loader:
            optimiser.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            # print(loss)
            training_loss.append(loss.item())
            loss.backward()
            optimiser.step()



        correct = 0 
        model.eval()
        num_test_points = 0
        results = dict() 
        for x_test, y_test, name, ordering in validation_loader:
            
            z = model(x_test)
            loss = criterion(z, y_test)
            validation_loss.append(loss.item())

            softmax = F.softmax(z[0], dim=0)

            _, yhat = torch.max(softmax, 0)

            for i in range(len(yhat)):
                
                if yhat[i] != (yhat[i] + 1) % 4:
                    yhat[i] = yhat[i]

            correct += (yhat == y_test[0]).sum().item()  
            mean_acc = torch.sum(yhat == y_test[0]) / (len(y_test[0]))
            num_test_points += len(y_test[0])
            if results.get(name[0]) != None: 
                results[name[0]].append([yhat, mean_acc, ordering])
            else: 
                results[name[0]] = [[yhat, mean_acc, ordering]]
            
        accuracy = correct / (num_test_points) 
        accuracy_list.append(accuracy)
        avg_train_loss.append(np.average(training_loss))
        avg_validation_loss.append(np.average(validation_loss))


    # print("HERE")
    return results, avg_validation_loss, avg_train_loss, accuracy_list
        

def save_results(results_dict, model, fold_num):
    outfile = open(RESULTS_PATH + model+ "results_" + str(fold_num),'wb')
    pickle.dump(results_dict, outfile)
    outfile.close()

def save_epoch_stats(avg_validation_loss, avg_train_loss, accuracy_list, model, fold_num):
    f = RESULTS_PATH + model+ "_epoch_results_" + model + "_30_03_2023_" + str(fold_num)
    np.savetxt(f, (avg_validation_loss, avg_train_loss, accuracy_list), delimiter=',')  

def save_model(fold_num):
    global model
    # torch.save(model.state_dict(),"/Users/serenahuston/GitRepos/ThirdYearProject/Models/model_weights_2022_stft_cnn_for_ensemble_64_8_" + str(fold_num) + ".pt")
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "model_weights_2022_stft_cnn_for_ensemble_64_8_", str(fold_num), ".pt"))

stratified_sample(csv_file, dataset_dir)
