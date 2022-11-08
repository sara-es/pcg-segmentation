import torch
from torch.utils.data import Dataset
from DataPreprocessing import DataPreprocessing
from utils import get_wavs_and_tsvs


class CNNData(Dataset):
    
    def __init__(self, env_patches, segmentation_patches): 
    
        self.x = torch.from_numpy(env_patches).type(torch.float32)
        
        self.y = segmentation_patches
        
        # Sets length
        self.len = self.x.shape[1]
        
    
    # Overrides get method
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        return sample
    
    # Overrides len() method
    def __len__(self):
        return self.len

wavs, tsvs, fs, names = get_wavs_and_tsvs("/Users/serenahuston/GitRepos/python-classifier-2022/physionet.org/files/circor-heart-sound/1.0.3/training_data",
                                return_names=True)


dp = DataPreprocessing(wavs[0], tsvs[0], fs[0], names[0])
print(len(dp.extract_env_patches()))
print(len(dp.extract_segmentation_patches()))