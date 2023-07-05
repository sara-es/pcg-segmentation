import sys, os
sys.path.append(os.path.join(sys.path[0], '..', '..', '..'))

from tqdm import tqdm
import scipy as sp
import pandas as pd 

from Utilities.create_segmentation_array import *
from SegmentationCNN.Models.Envelope_CNN.DataPreprocessing import * 
from SegmentationCNN.Models.Envelope_CNN.CNNData import * 

class PatientInfo: 

    def __init__(self, dataset_dir, window=64, stride=8):
        self.dataset_dir = dataset_dir 
        self.info_dict = {"ID": [], "Filename": [], "Raw_WAV": [], "Frequency" : [], 
                          "TSV": [], "Clipped_WAV" : [], "Segmentations": [], 
                          "CNN_Data": [], }
        self.window = window
        self.stride = stride


    def get_data(self):
        
        for file_ in tqdm(sorted(os.listdir(self.dataset_dir))):
            root, extension = os.path.splitext(file_)
            # Sketchy files - ignore 
            if "50782_MV" in root:
                print("Ignoring")
                continue 
            if extension == ".wav":
                wav_file = file_
                segmentation_file = os.path.join(self.dataset_dir, root + ".tsv")
                if os.path.exists(segmentation_file):
                    if (self.update_info_dict_entry(root, wav_file, segmentation_file)): 
                        self.update_CNN_data()
        
        self.patient_df = pd.DataFrame(self.info_dict)

                             
    def update_info_dict_entry(self, root, wav_file, segmentation_file):
        fs, recording = sp.io.wavfile.read(os.path.join(self.dataset_dir, wav_file))
        tsv = np.loadtxt(segmentation_file, delimiter="\t")

        
        clipped_recording, segmentations = create_segmentation_array(recording,
                                                                    tsv,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=4000)
        
        if len(clipped_recording) > 0 and len(segmentations) > 0: 
            if len(segmentations[0] > 0):
                self.info_dict["TSV"].append(tsv)
                self.info_dict["Raw_WAV"].append(recording)
                self.info_dict["Frequency"].append(fs)
                self.info_dict["Filename"].append(wav_file)
                self.info_dict["Clipped_WAV"].append(clipped_recording[0])
                self.info_dict["Segmentations"].append(segmentations[0])
                self.info_dict["ID"].append(int(root.split("_")[0]))
                return True
            else:
                print("HERE")
                return False
        

    def update_CNN_data(self):
        dp = DataPreprocessing(self.info_dict["Clipped_WAV"][-1], self.info_dict["Segmentations"][-1]-1, self.info_dict["Frequency"][-1],
                               self.window, self.stride)
        
        dp.extract_env_patches()
        dp.extract_segmentation_patches()

        self.info_dict["CNN_Data"].append(CNNData(dp.env_patches, dp.seg_patches, self.info_dict["Filename"][-1], range(0, len(dp.env_patches))))
        