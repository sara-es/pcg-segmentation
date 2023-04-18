

import os 
import scipy as sp 
import numpy as np
import sys

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from Utilities.constants import * 
from Utilities.create_segmentation_array import * 
from SegmentationCNN.Models.Envelope_CNN.DataPreprocessing import * 
from DataManipulation.DataPresentation import *

patient = "36327_AV"


fs, recording = sp.io.wavfile.read(os.path.join(TRAINING_DATA_PATH_2022 + "training_data/", patient + ".wav"))
tsv = np.loadtxt(TRAINING_DATA_PATH_2022 + "training_data/" + patient + ".tsv", delimiter="\t")
clipped_recording, segmentations = create_segmentation_array(recording,
                                                                    tsv,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=4000)


print(len(clipped_recording[0]))
dp = DataPreprocessing(clipped_recording[0], segmentations[0], fs)
data_pres = DataPresentation()

data_pres.plot_signal(clipped_recording[0], "Original 36327_AV PCG")
filtered_signal = dp.filter_signal(clipped_recording[0])
print("HERE")
data_pres.plot_signal(filtered_signal, "36327_AV Filtered by 400Hz and 25Hz Band-Pass")
spike_rem_signal = dp.spike_removal(filtered_signal)
data_pres.plot_signal(spike_rem_signal, "36327_AV Following Schmidt Spike Removal")
# homo_env = dp.get_homomorphic_envelope(spike_rem_signal)
# data_pres.plot_signal(homo_env, "Homomorphic Envelope of 36327_AV")
hilb_env = dp.get_hilbert_envelope(spike_rem_signal)
# data_pres.plot_signal(hilb_env, "Hilbert Envelope of 36327_AV")
wave_env = dp.get_wavelet_envelope(spike_rem_signal)
print(len(wave_env))
# data_pres.plot_signal(wave_env, "Wavelet Envelope of 36327_AV")
# power_spec_env = dp.get_power_spectral_density_envelope(spike_rem_signal)


# data_pres.plot_signal(dp.power_spec_env, "Normalised and Down-Sampled Power Spectral Density Envelope of 36327_AV")
# data_pres.plot_signal(dp.wave_env, "Normalised and Down-Sampled Wavelet Envelope of 36327_AV")
# data_pres.plot_signal(dp.homo_env, "Normalised and Down-Sampled Homomorphic Envelope of 36327_AV")
# data_pres.plot_signal(dp.hilb_env, "Normalised and Down-Sampled Hilbert Envelope of 36327_AV")