import scipy as sp 
import os 
import numpy as np 

from Utilities import constants
from SegmentationCNN import STFT_DataPreprocessing
from Utilities.create_segmentation_array import *


def get_segments(recording):
    
    patch_list = [] 
    if len(recording) < 64: 
        padding = 64 - len(recording)
        recording = np.pad(recording, pad_width=(0,padding), mode="constant", constant_values=(recording[-1]))
    for i in range(0, len(recording), 8):
        if i+64 >= len(recording):
            padding = (i+64) - len(recording)
            patch = np.pad(recording[i:len(recording)], pad_width=(0,padding), mode="constant", constant_values=(0))
            patch_list.append(patch)
            break
        else: 
            patch = recording[i:i+64]
            patch_list.append(patch)
    return np.array(patch_list)


filenames = ["84946_PV", "50296_AV", "50628_AV", "85140_AV", "50115_PV_2",
            "50048_PV", "50822_PV", "50789_AV", "85241_AV", "49712_PV"]

for f in filenames:
    # SHORT TIME FOURIER TRANSFORM 
    
    fs, recording = sp.io.wavfile.read(os.path.join(constants.TRAINING_DATA_PATH_2022, "training_data/", (f+".wav")))
    tsv = np.loadtxt(os.path.join(constants.TRAINING_DATA_PATH_2022, "training_data/", (f + ".tsv")), delimiter="\t")

    clipped_recording, segmentations = create_segmentation_array(recording,
                                                                    tsv,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=4000)
    dp = STFT_DataPreprocessing.DataPreprocessing_STFT(clipped_recording[0], segmentations[0], fs)
    dp.extract_wav_and_seg_patches()



    # plt.plot(dp.stft_patches[0][:,1])
    # plt.show()
    # plt.plot(dp.stft_patches[0][:,-2])
    # plt.show()

    break
    # pip install antropy
    # se = entropy.spectral_entropy(recording, fs, normalize=True)
    # print(se)


    # mfccs = mfcc(y=np.array(recording, dtype=np.float64), sr=fs,)
    # print(len(mfccs))
    

# Shannon energy envelope 