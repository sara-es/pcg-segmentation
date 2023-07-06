import sys, os
sys.path.append(os.path.join(sys.path[0], '..', '..', '..'))

import numpy as np 
import pickle
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

import sys 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from EarlyStopping import EarlyStopping
from tqdm.contrib import tzip

from Utilities.constants import * 
from DataManipulation.PatientFrame import * 
from DataManipulation.PatientFrame import PatientFrame
from DataManipulation.DataPresentation import DataPresentation

from GitHubUNet import UNet, init_weights

from SegmentationHMM import train_segmentation, run_segmentation
from PatientInfo import * 

dataset_dir = TRAINING_DATA_PATH
csv_file = DATA_CSV_PATH
# dataset_dir = TINY_TEST_DATA_PATH
# csv_file = TINY_TEST_CSV_PATH

# dataset_dir = "/Users/serenahuston/GitRepos/Data/DataSubset_21_Patients"
# csv_file = "/Users/serenahuston/GitRepos/Data/training_data_subset_21.csv"

epoch_count = 0 

data_pres = DataPresentation()
data_pres_folder = DATA_PRESENTATION_PATH + "CNN_vs_HMM_Full_Data_22_03_2023/"
fold_num = 1 

def set_up_model():
    global model, optimiser, criterion 
    model = UNet()
    # model.apply(init_weights)
    model.load_state_dict(torch.load(MODEL_PATH + "model_weights_2016.pt")) # No such file or directory: '/Models/model_weights_2016.pt'
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()


def stratified_sample(csv_file, dataset_dir, folds=5):
    print("Loading data and initializing trials...")
    global fold_num 
    pf = PatientFrame(csv_file)
    # print("RUNNING")
    patient_info = PatientInfo(dataset_dir)
    patient_info.get_data()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    
    for train_index, test_index in skf.split(pf.patient_frame["Patient ID"], pf.patient_frame["Murmur"]):
        print(f"#### FOLD {fold_num} ####")
        patients_train, patients_test = pf.patient_frame["Patient ID"][train_index], pf.patient_frame["Patient ID"][test_index]
        training_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_train)]
        val_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_test)]
        print(f"Training HMM...")
        hmm_results = train_eval_HMM(training_df, val_df)
        print(f"Training CNN...")
        cnn_results =prep_CNN(training_df, val_df)
        print(f"Saving results...")
        save_results(hmm_results, "hmm", fold_num)
        save_results(cnn_results, "cnn", fold_num)
        fold_num += 1 


def prep_CNN(training_df, val_df):
    train_data = ConcatDataset(training_df["CNN_Data"])

    validation_data = ConcatDataset(val_df["CNN_Data"])
  
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=1, shuffle=True)

    set_up_model()
    return train(train_loader, validation_loader, len(validation_data))

def train_eval_HMM(training_df, val_df):

    models, pi_vector, total_obs_distribution= train_segmentation.train_hmm_segmentation(training_df["Raw_WAV"].tolist(), training_df["TSV"].tolist())
    results = dict()

    for rec, seg, name in tzip(val_df["Clipped_WAV"].tolist(), val_df["Segmentations"].tolist(), val_df["Filename"].tolist()):
        yhat = run_segmentation.run_hmm_segmentation(rec,
                                              models,
                                              pi_vector,
                                              total_obs_distribution,
                                              use_psd=True,
                                              return_heart_rate=False,
                                              try_multiple_heart_rates=False)
        results[name] = [yhat, (seg == yhat).mean()]

    return results
        


def train(train_loader, validation_loader, validation_size, epochs=15, patience=5):
    global data_pres_folder, fold_num 
    
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
            
        accuracy = correct / (validation_size * 64) 
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
        

def plot_loss_and_accuracy(valid_loss, train_loss, accuracy):
    global epoch_count
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(18, 10)
    fig.suptitle('Model Loss VS Accuracy Across Epochs', fontsize=12)
    ax1.plot(train_loss)
    ax2.plot(valid_loss)
    ax3.plot(accuracy)
    ax1.set_ylabel("Training Loss")
    ax2.set_ylabel("Validation Loss")
    ax3.set_ylabel("Accuracy")
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Epochs")
    ax3.set_xlabel("Epochs")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    fig_name = DATA_PRESENTATION_PATH + "Loss VS Accuracy_22_03_23__" + str(epoch_count)
    print(fig_name)
    plt.savefig(fig_name)
    
    epoch_count += 1

def save_results(results_dict, model, fold_num):
    outfile = open(RESULTS_PATH + model+ "_results_22_03_23__" + str(fold_num),'wb')
    pickle.dump(results_dict, outfile)
    outfile.close()



stratified_sample(csv_file, dataset_dir)
