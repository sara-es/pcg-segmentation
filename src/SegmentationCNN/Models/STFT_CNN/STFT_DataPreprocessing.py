
import numpy as np 
from scipy.signal import stft
import statistics
import librosa
import matplotlib.pyplot as plt 

class STFT_DataPreprocessing: 


    def __init__(self, wav, segmentation_array, fs, window=5120, stride=640):

        self.wav = wav
        self.segmentation_array = segmentation_array
        self.sampling_frequency = fs
        self.PATCH_SIZE = window
        self.STRIDE = stride 
        self.stft_patches = [] 
        self.wav_patches = [] 
        self.seg_patches = []
        self.output_patches = [] 
        
    def extract_wav_and_seg_patches(self):
        for i in range(0, len(self.segmentation_array), self.STRIDE):
            padding = i+self.PATCH_SIZE - len(self.segmentation_array)
            if i+self.PATCH_SIZE >= len(self.segmentation_array):
                self.segmentation_array = np.pad(self.segmentation_array, pad_width=(0,padding), mode="constant", constant_values=(0))
                self.wav = np.pad(self.wav, pad_width=(0,padding), mode="constant", constant_values=(0))
            seg_patch = self.segmentation_array[i:i+self.PATCH_SIZE]
            wav_patch = self.wav[i:i+self.PATCH_SIZE]

            self.extract_stft_patches(wav_patch)
            self.downsample_seg_patch(seg_patch)
        self.stft_patches = np.array(self.stft_patches)
        self.output_patches = np.array(self.output_patches)

    def extract_wav_patches_only(self):
        for i in range(0, len(self.wav), self.STRIDE):
            padding = i+self.PATCH_SIZE - len(self.wav)
            if i+self.PATCH_SIZE >= len(self.wav):
                self.wav = np.pad(self.wav, pad_width=(0,padding), mode="constant", constant_values=(0))
            wav_patch = self.wav[i:i+self.PATCH_SIZE]
            self.extract_stft_patches(wav_patch)
        self.stft_patches = np.array(self.stft_patches)


    def extract_stft_patches(self, patch):
        f, t, Zxx = stft(patch, fs=self.sampling_frequency, nperseg=576, noverlap=504, boundary=None, padded=False)
        shortened_Zxx = Zxx[:150, :]

        results = [] 
        for i in range(0, len(shortened_Zxx), 20):
            mean = np.mean(shortened_Zxx[i:i+10, :], axis=0)
            results.append(np.abs(mean))
        self.stft_patches.append(results)

    def downsample_seg_patch(self, patch):
        downsample_segment = [] 
        for i in range(0, len(patch), 80):
            modal_val = statistics.mode(patch[i:i+80])
            downsample_segment.append(modal_val)
        self.output_patches.append(downsample_segment)

