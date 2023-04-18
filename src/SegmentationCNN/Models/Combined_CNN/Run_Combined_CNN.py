from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import sys 
import pickle 

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from SegmentationCNN.Models.Envelope_CNN.GitHubUNet import UNet
from SegmentationCNN.Models.STFT_CNN.STFT_GitHubUNet import STFT_UNet
from DataManipulation.PatientFrame import PatientFrame
from SegmentationCNN.Models.Combined_CNN.Combined_PatientInfo import *

stft_model = None
env_model = None

MODEL_PATH = "/Users/serenahuston/GitRepos/ThirdYearProject/Models/Ensemble_Weights/"
RESULTS_PATH = "/Users/serenahuston/GitRepos/ThirdYearProject/Results/Ensemble_Results/Combined/"
dataset_dir = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data"
csv_file = "/Users/serenahuston/GitRepos/Data/PhysioNet_2022/training_data.csv"

def load_model(model, fold):
    global stft_model, env_model 
    if model == "env":
        env_model = UNet(n_channels=4)
        env_model.eval()
        env_model.load_state_dict(torch.load(MODEL_PATH + "Env/model_weights_2022_env_cnn_" + str(fold) + ".pt"))
    elif model == "stft":
        stft_model = STFT_UNet(n_channels=8)
        stft_model.eval()
        stft_model.load_state_dict(torch.load(MODEL_PATH + "STFT/model_weights_2022_stft_cnn_" + str(fold) + ".pt"))


def stratified_sample(csv_file, dataset_dir, folds=10):
    global fold_num, data_pres_folder
    pf = PatientFrame(csv_file)
    print("RUNNING")

    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    fold_num = 1

    patient_info = Combined_PatientInfo(dataset_dir)
    patient_info.get_data()

    for train_index, test_index in skf.split(pf.patient_frame["Patient ID"], pf.patient_frame["Murmur"]):
        _, patients_test = pf.patient_frame["Patient ID"][train_index], pf.patient_frame["Patient ID"][test_index]
        val_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_test)]
        load_model("env", fold_num)
        load_model("stft", fold_num)
        env_validation_loader = get_validation_loader("env", val_df)
        stft_validation_loader = get_validation_loader("stft", val_df)
        results = run_combined_model(env_validation_loader, stft_validation_loader)
        save_results(results, fold_num)
        fold_num += 1 

def get_validation_loader(model, val_df):
    if model == "env":
        validation_data = ConcatDataset(val_df["Env_CNN_Data"])
        print(len(validation_data))
    elif model == "stft":
        validation_data = ConcatDataset(val_df["STFT_CNN_Data"])
        print(len(validation_data))
    validation_loader = DataLoader(dataset=validation_data, batch_size=1, shuffle=True)
    return validation_loader 
    
def run_combined_model(env_validation_loader, stft_validation_loader):
    global env_model, stft_model
    env_predictions = dict() 
    results = dict() 

    correct = 0 
    num_test_points = 0
    

    for x_test, y_test, name, ordering in env_validation_loader:
        yhat = env_model(x_test[0])
        softmax = F.softmax(yhat, dim=0)
        if env_predictions.get(name) == None:
            env_predictions[name] = dict()
        env_predictions[name][ordering.item()] = softmax

    for x_test, y_test, name, ordering in stft_validation_loader:
        if env_predictions[name].get(ordering.item()) != None: 
            yhat = stft_model(x_test)
            stft_softmax = F.softmax(yhat[0], dim=0)
            env_softmax = env_predictions[name][ordering.item()]
            combined_softmax = (stft_softmax + env_softmax)/2
            _, yhat = torch.max(combined_softmax, 0)

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
        else:
            print(ordering.item())
            print(max(env_predictions[name].keys()))
    return results

def save_results(results_dict, fold_num):
    filename = "combined_cnn_64_8_results_04_04_2023_" + str(fold_num)
    outfile = open(RESULTS_PATH + filename, 'wb')
    pickle.dump(results_dict, outfile)
    outfile.close()

stratified_sample(csv_file, dataset_dir)