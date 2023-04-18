
import os 
import scipy as sp 
import numpy as np
import sys
from scipy.fft import fft

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from Utilities.constants import * 
from Utilities.create_segmentation_array import * 
from DataManipulation.DataPresentation import *
from SegmentationCNN.Models.STFT_CNN.STFT_DataPreprocessing import * 


patient = "23625_AV"

# dp = DataPresentation()
# dp.plot_STFT_shorter_window(patient)

fs, recording = sp.io.wavfile.read(os.path.join(TRAINING_DATA_PATH_2022 + "training_data/", patient + ".wav"))
tsv = np.loadtxt(TRAINING_DATA_PATH_2022 + "training_data/" + patient + ".tsv", delimiter="\t")
clipped_recording, segmentations = create_segmentation_array(recording, tsv,
                                                                recording_frequency=4000,
                                                                feature_frequency=4000)


dp = STFT_DataPreprocessing(clipped_recording[0], segmentations[0]-1, fs)
dp.extract_wav_and_seg_patches()


dp = DataPresentation()
accuracy = [0.853, 0.813, 0.835, 0.863, 0.824, 0.868, 0.869, 0.8645, 0.821, 0.81]
train_loss = [0.81, 0.49, 0.46, 0.440, 0.43, 0.425, 0.42,0.415,0.41, 0.405]
valid_loss = [0.405, 0.479, 0.43, 0.375, 0.457, 0.35, 0.35, 0.365, 0.452, 0.482]
data_pres_folder = "/Users/serenahuston/GitRepos/ThirdYearProject/DataPresentation/STFT_Full_Data_15_Channel"
dp.plot_loss_and_accuracy(train_loss, valid_loss, accuracy, data_pres_folder, 0)