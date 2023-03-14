
import numpy as np 
from scipy.signal import stft
import statistics


class DataPreprocessing_STFT: 


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


    def extract_stft_patches(self, patch):
        f, t, Zxx=stft(patch, fs=self.sampling_frequency, nperseg=640, noverlap=560, padded=False)
        # Potentially complex vals
        # half = int(len(Zxx) / 2)
        # shortened_Zxx = np.abs(Zxx[:half, :])
        # print(Zxx.shape)
        results = [] 
        for i in range(0, len(Zxx), 20):
            mean = np.mean(Zxx[i:i+20, :], axis=0)
            results.append(np.abs(mean))
        self.stft_patches.append(results)


    # def extract_output_patches(self, patch):
    #     for i in range(0, len(patch), 160):
    #         padding = i+1280 - len(patch)
    #         if i+1280 >= len(patch):
    #             patch = np.pad(patch, pad_width=(0,padding), mode="constant", constant_values=(0))
    #             sub_patch = patch[i:i+1280]
    #             break
    #         else: 
    #             sub_patch = patch[i:i+1280]
    #         self.output_patches[-1].append(sub_patch)



    
    def downsample_seg_patch(self, patch):
        intervals = np.linspace(0, 5120, num=65, endpoint=False) 
        downsample_segment = [] 
        for i in range(0, len(intervals)):
            try:
                modal_val = statistics.mode(patch[int(intervals[i]):int(intervals[i+1])])
            except: 
                modal_val = statistics.mode(patch[int(intervals[i]):len(patch)])
            downsample_segment.append(modal_val)
        self.output_patches.append(downsample_segment)
