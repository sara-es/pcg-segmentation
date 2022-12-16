import numpy as np 
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
from create_segmentation_array import create_segmentation_array

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from Utilities.constants import * 
from DataManipulation.PatientFrame import * 

from DataManipulation.PatientFrame import PatientFrame

from GitHubUNet import UNet

from utils import get_all_patient_info
from SegmentationHMM import train_segmentation, run_segmentation

# dataset_dir = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data"
dataset_dir = "/Users/serenahuston/GitRepos/Data/DataSubset_100"
# csv_file = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data.csv"
csv_file = "/Users/serenahuston/GitRepos/Data/training_data_subset_100.csv"

epoch_count = 0 

def set_up_model():
    global model, optimiser, criterion 
    model = UNet()
    # model.apply(init_weights)
    model.load_state_dict(torch.load("/Users/serenahuston/GitRepos/ThirdYearProject/Models/model_weights_2016.pt"))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()


def stratified_sample(csv_file, dataset_dir, folds=10):
    pf = PatientFrame(csv_file)
    print("RUNNING")
    patient_info = get_all_patient_info(dataset_dir)

    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
  
    for train_index, test_index in skf.split(pf.patient_frame["Patient ID"], pf.patient_frame["Murmur"]):
        patients_train, patients_test = pf.patient_frame["Patient ID"][train_index], pf.patient_frame["Patient ID"][test_index]

        training_info = dict([ (k,v) for (k,v) in patient_info.items() if k in patients_train.values])
        validation_info = dict([ (k,v) for (k,v) in patient_info.items() if k in patients_test.values])

        hmm_correct_results, hmm_incorrect_results, accuracies = train_eval_HMM(training_info, validation_info)
        print(len(hmm_correct_results.keys()), len(hmm_incorrect_results), accuracies)
        # prep_CNN(wavs, tsvs, fs, names, patients_train, patients_test)


def prep_CNN(wavs, tsvs, fs, names, patients_train, patients_test):
    # Remember to -1 from segmentations, not done yet 
    patient_data_dict = get_patient_data_dict(wavs, tsvs, fs, names)
    train_data_list = [ConcatDataset(patient_data_dict[id]) for id in patients_train]
    train_data = ConcatDataset(train_data_list)
    
    validation_data_list = [ConcatDataset(patient_data_dict[id]) for id in patients_test]
    validation_data = ConcatDataset(validation_data_list)

    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=1, shuffle=True)

    set_up_model()
    train(train_loader, validation_loader, len(validation_data))

def train_eval_HMM(training_info, validation_info):

    train_wavs, train_segments = [], []
    valid_wavs, valid_segments = [], []
    valid_files = [] 
    for (patient_id, patient_file_info) in training_info.items():
        for (k,v) in patient_file_info.items():
            train_wavs.append(v[0])
            train_segments.append(v[1])

    for (patient_id, patient_file_info) in validation_info.items():
        for (k,v) in patient_file_info.items():
            valid_wavs.append(v[0])
            valid_segments.append(v[1])
            valid_files.append(k)
        
    models, pi_vector, total_obs_distribution= train_segmentation.train_hmm_segmentation(train_wavs, train_segments)

    idx = 0
    correct_results = dict()
    incorrect_results = dict()
    accuracies = np.zeros(len(valid_wavs))

    for wav, segments, name in tzip(valid_wavs, valid_segments, valid_files):

        y_hat = run_segmentation.run_hmm_segmentation(wav,
                                    models,
                                    pi_vector,
                                    total_obs_distribution,
                                    use_psd=True,
                                    return_heart_rate=False,
                                    try_multiple_heart_rates=False)

        accuracies[idx] = (segments == y_hat).mean() 

        if accuracies[idx] > 0.5: 
            correct_results[name] = y_hat
        else:
            incorrect_results[name] = y_hat
        idx += 1
    return correct_results, incorrect_results, accuracies
        


def train(train_loader, validation_loader, validation_size, epochs=15, patience=5):
    
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
        for x,y in train_loader:
            optimiser.zero_grad()
            yhat = model(x[0])
            try:
                loss = criterion(torch.t(yhat), y[0])
                training_loss.append(loss.item())
                loss.backward()
                optimiser.step()
            except:
                # -1 value appearing here 
                print(yhat, y[0])
            

        correct = 0 
        model.eval()
        for x_test, y_test in validation_loader:
            
            z = model(x_test[0])
            loss = criterion(torch.t(z), y_test[0])
            validation_loss.append(loss.item())

            softmax = F.softmax(z, dim=1)
            _, yhat = torch.max(softmax, 0)
        
            for i in range(1, yhat.shape[0]): 
                if yhat[i] != (yhat[i-1] + 1) % 4:
                    yhat[i] = yhat[i-1]
            
            correct += (yhat == y_test[0]).sum().item()      
            
        accuracy = correct / (validation_size * 64) 
        accuracy_list.append(accuracy)
        avg_train_loss.append(np.average(training_loss))
        print(len(validation_loss))
        avg_validation_loss.append(np.average(validation_loss))

        early_stopping(np.average(validation_loss), model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("HERE")
    plot_loss_and_accuracy(avg_validation_loss, avg_train_loss, accuracy_list)
        

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
    fig_name = "/Users/serenahuston/GitRepos/ThirdYearProject/DataPresentation/" + "Loss VS Accuracy" + str(epoch_count)
    print(fig_name)
    plt.savefig(fig_name)
    
    epoch_count += 1



stratified_sample(csv_file, dataset_dir)
