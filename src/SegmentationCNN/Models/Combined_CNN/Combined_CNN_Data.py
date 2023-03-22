import torch
from torch.utils.data import Dataset


class Combined_CNN_Data(Dataset):
    
    def __init__(self, env_patches, stft_patches, segmentation_patches, file_name, ordering): 
    
        self.stft_patches = torch.tensor(stft_patches, requires_grad=True).type(torch.float32)

        self.env_patches = torch.tensor(env_patches, requires_grad=True).type(torch.float32)

        self.seg_patches = torch.from_numpy(segmentation_patches).type(torch.int64)

        self.file_name = file_name

        self.ordering = ordering
        # Sets length
        self.len = self.env_patches.shape[0]
        
    
    # Overrides get method
    def __getitem__(self, index):
        sample = self.env_patches[index], self.stft_patches[index], self.seg_patches[index], self.file_name, self.ordering[index]
        return sample
    
    # Overrides len() method
    def __len__(self):
        return self.len
