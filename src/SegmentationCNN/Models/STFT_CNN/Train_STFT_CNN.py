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

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from Utilities.constants import * 
from DataManipulation.PatientFrame import * 
from DataManipulation.PatientFrame import PatientFrame
from PatientInfoSTFT import * 
from STFT_GitHubUNet import STFT_UNet, init_weights

dataset_dir = "/Users/serenahuston/GitRepos/Data/DataSubset_48"
csv_file = "/Users/serenahuston/GitRepos/Data/training_data_subset_48.csv"


epoch_count = 0 

def set_up_model():
    global model, optimiser, criterion 
    model = STFT_UNet(n_channels=17)
    model.apply(init_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()


def stratified_sample(csv_file, dataset_dir, folds=10):
    pf = PatientFrame(csv_file)
    print("RUNNING")
    patient_info = PatientInfo_STFT(dataset_dir)
    patient_info.get_data()

    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    fold_num = 1
    for train_index, test_index in skf.split(pf.patient_frame["Patient ID"], pf.patient_frame["Murmur"]):
        patients_train, patients_test = pf.patient_frame["Patient ID"][train_index], pf.patient_frame["Patient ID"][test_index]
        training_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_train)]
        val_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_test)]
        cnn_results =prep_CNN(training_df, val_df)
        save_results(cnn_results, "stft_cnn", fold_num)
        # fold_num += 1 


def prep_CNN(training_df, val_df):
    train_data = ConcatDataset(training_df["STFT_Data"])
    validation_data = ConcatDataset(val_df["STFT_Data"])   
  
    train_loader = DataLoader(dataset=train_data, batch_size=50, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=50, shuffle=True)

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

            softmax = F.softmax(z, dim=1)

            _, yhat = torch.max(softmax, 1)

            for i in range(len(yhat)):
                for j in range(1, len(yhat[i])):
                    if yhat[i][j] != (yhat[i][j-1] + 1) % 4:
                        yhat[i][j] = yhat[i][j-1]

            

            correct += (yhat == y_test).sum().item()  
            testing =   (yhat == y_test)  
            sum_testing = torch.sum(yhat == y_test, dim=1)
            mean_acc = torch.sum(yhat == y_test, dim=1) / (len(y_test[0]))
            num_test_points += len(y_test) * len(y_test[0])
            # print(mean_acc)
            if results.get(name[0]) != None: 
                results[name[0]].append([yhat, mean_acc, ordering])
            else: 
                results[name[0]] = [[yhat, mean_acc, ordering]]
            
        accuracy = correct / (num_test_points) 
        accuracy_list.append(accuracy)
        avg_train_loss.append(np.average(training_loss))
        avg_validation_loss.append(np.average(validation_loss))


    print("HERE")
    plot_loss_and_accuracy(avg_validation_loss, avg_train_loss, accuracy_list)
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
    fig_name = DATA_PRESENTATION_PATH + "Loss_VS_Accuracy_STFT" + str(epoch_count)
    print(fig_name)
    plt.savefig(fig_name)
    
    epoch_count += 1

def save_results(results_dict, model, fold_num):
    outfile = open(RESULTS_PATH + model+ "results_" + str(fold_num),'wb')
    pickle.dump(results_dict, outfile)
    outfile.close()

stratified_sample(csv_file, dataset_dir)
