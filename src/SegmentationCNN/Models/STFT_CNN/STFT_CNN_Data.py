import torch
from torch.utils.data import Dataset


class STFT_CNN_Data(Dataset):
    
    def __init__(self, stft, segmentation_patches, file_name, ordering): 
    
        self.x = torch.tensor(stft, requires_grad=True).type(torch.float32)
        
        self.y = torch.from_numpy(segmentation_patches).type(torch.int64)

        self.file_name = file_name

        self.ordering = ordering
        # Sets length
        self.len = self.x.shape[0]
        
    
    # Overrides get method
    def __getitem__(self, index):
        sample = self.x[index], self.y[index], self.file_name, self.ordering[index]
        return sample
    
    # Overrides len() method
    def __len__(self):
        return self.len
