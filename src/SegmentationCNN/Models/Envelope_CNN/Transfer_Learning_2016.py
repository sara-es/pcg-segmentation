import sys, os
sys.path.append(os.path.join(sys.path[0], '..', '..', '..'))

import pandas as pd
import numpy as np 

import numpy as np
from scipy.io import wavfile, loadmat
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from CNNData import CNNData 
from GitHubUNet import *
from Utilities.create_segmentation_array import create_segmentation_array
from DataPreprocessing import DataPreprocessing
# from TrainingCNN import * 
import re 


MAT_FILE_SAMPLE_RATE = 2000 


def get_df_from_mat(filename):
    mat = loadmat(filename)

    mdata = mat['state_ans']
    # pqr=pd.Series(mdata)
    df = pd.DataFrame(mdata, columns=["Time", "Period"])


    df["Time"] = df["Time"].apply(lambda x: float(re.sub(r"[\([{})\]]", "", str(x)))/MAT_FILE_SAMPLE_RATE)
    df["Period"] = df["Period"].apply(lambda x: re.sub(r"[\([{})\]]", "", str(x)))
    df = df.replace({'N': 0}, regex=True)
    df = df.replace({'S1': 1}, regex=True)
    df = df.replace({'systole': 2}, regex=True)
    df = df.replace({'s': 2}, regex=True)
    df = df.replace({'S2': 3}, regex=True)
    df = df.replace({'diastole': 4}, regex=True)
    df = df.replace({'d': 4}, regex=True)
    return df 

def get_tsv_from_df(df):
    tsvs = [] 
    for i in range(1, len(df)):
        tsvs.append([0,0,0])
        tsvs[i-1][0] = df.loc[i-1]["Time"]
        tsvs[i-1][1] = df.loc[i]["Time"]
        if df.loc[i-1]["Period"] == "S":
            tsvs[i-1][2] = int((df.loc[i-2]["Period"]+1)%4)
        else: 
            tsvs[i-1][2] = int(df.loc[i-1]["Period"])
    return tsvs

def get_wavs_tsvs_2016(wav_dir, annotate_dir, return_names=False):
    wav_arrays, tsv_arrays, fs_arrays, names = [], [], [], []
    for file_ in tqdm(sorted(os.listdir(wav_dir))):
        root, extension = os.path.splitext(file_)
        if extension == ".wav":
            segmentation_file = os.path.join(annotate_dir, root + "_StateAns.mat")
            if not os.path.exists(segmentation_file):
                continue
            fs, recording = wavfile.read(os.path.join(wav_dir, file_))
            wav_arrays.append(recording)
            fs_arrays.append(fs)
            if return_names:
                names.append(file_)

            df = get_df_from_mat(segmentation_file)
            tsv_segmentation = get_tsv_from_df(df)
            tsv_arrays.append(np.array(tsv_segmentation))
    if return_names:
        return wav_arrays, tsv_arrays, fs_arrays, names
    return wav_arrays, tsv_arrays, fs_arrays

def get_data_for_CNN(wav_root, annotate_root, dir_extensions):
    datasets = [] 
    for training_sample in dir_extensions:
        wav_dir = wav_root + "training-" + training_sample + "/"
        annotate_dir = annotate_root + "training-" + training_sample + "_StateAns/"
        wavs, tsvs, fs, names = get_wavs_tsvs_2016(wav_dir, annotate_dir, return_names=True)

        for i in range(len(wavs)):
            clipped_recording, segmentations = create_segmentation_array(wavs[i],
                                                                    tsvs[i],
                                                                    recording_frequency=fs[i],
                                                                    feature_frequency=fs[i])
            if len(clipped_recording) > 0 and len(segmentations) > 0:
                dp = DataPreprocessing(clipped_recording[0], segmentations[0], fs[i], window=64, stride=8)
                dp.extract_env_patches()
                dp.extract_segmentation_patches()
                datasets.append(CNNData(dp.env_patches, dp.seg_patches, names[i], range(len(dp.env_patches))))
    return ConcatDataset(datasets)

def train_2016(train_loader, epochs=2):
    
    model.train(True)

    for epoch in range(epochs):
        print("NEW EPOCH")
        training_loss = []  
        for x,y,name,ordering in train_loader:
            optimiser.zero_grad()
            yhat = model(x[0])
            try:
                loss = criterion(torch.t(yhat), y[0])
                training_loss.append(loss) 
                loss.backward()
                optimiser.step()
            except:
                # -1 value appearing here 
                print(yhat, y[0])
        

def set_up_model_2016():
    global model, optimiser, criterion 
    model = UNet()
    model.apply(init_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

dir_extensions = ["a", "b", "c", "d", "e", "f"]
# wav_root = "/Users/serenahuston/GitRepos/Data/PhysioNet_2016/"
wav_root = "physionet.org/files/challenge-2016/1.0.0/"
# annotate_root = "/Users/serenahuston/GitRepos/Data/annotations/hand_corrected/"
annotate_root = "physionet.org/files/challenge-2016/1.0.0/annotations/hand_corrected/"

dataset = get_data_for_CNN(wav_root, annotate_root, dir_extensions)
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)


set_up_model_2016()
print("TRAINING")
train_2016(train_loader, epochs=2)
print("SAVING")
torch.save(model.state_dict(), "src/SegmentationCNN/Models/model_weights_2016_64_8.pt")