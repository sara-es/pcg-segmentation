import numpy as np
import scipy as sp
import math 
import sys 
import librosa 
import os 
import matplotlib.pyplot as plt 
from librosa import resample

sys.path.append("/Users/serenahuston/GitRepos/ThirdYearProject/src/")
from Utilities import constants

f= "29045_MV"

fs, recording = sp.io.wavfile.read(os.path.join(constants.TRAINING_DATA_PATH_2022, "training_data/", (f+".wav")))

#extract the maximum signal
maximum_signal = max(np.abs(recording))
#normalize the signal
normalized_signal = np.array([(abs(signal) / maximum_signal) for signal in recording])

plt.plot(np.linspace(0, len(recording) / fs, num=len(recording)), recording)
plt.show()

plt.plot(np.linspace(0, len(recording) / fs, num=len(recording)), normalized_signal)
plt.show()


#iterate through the normalized signal
for x in range(0, len(normalized_signal)):
    #power the signal by 2 
    signal_sample = abs(normalized_signal[x]) ** 2 
    if signal_sample <= 0: #set the signal to 1 if it is empty
       signal_sample = 1.0
    
    #calculate Shannon energy
    shannon_energy = signal_sample * math.log(signal_sample)
    
    #replace the normalized signal with Shannon energy       
    normalized_signal[x] = shannon_energy

#obtain the length of signal
length_of_signal = len(normalized_signal)
#Initialize the signal
segment_signal = int(fs*0.02)
segment_energy = [] #initialize the array
for x in range(0, len(normalized_signal), segment_signal):
    sum_signal = 0 
    #retrieve the signal in a segment of 0.02 seconds
    current_segment_energy=normalized_signal[x:x+segment_signal]
    for i in range(0, len(current_segment_energy)):
        #sum up the Shannon energy
        sum_signal += current_segment_energy[i]
    #assign the average Shannon energy to array    
    segment_energy.append(-(sum_signal/segment_signal))
#convert to numpy array
segment_energy_signal = np.array(segment_energy)


#calculate mean
mean_SE = np.mean(segment_energy_signal)
#calculate standard deviation
std_SE = np.std(segment_energy_signal)
#calculate Shannon Envelope
for x in range(0, len(segment_energy_signal)):
    envelope = 0
    envelope = (segment_energy_signal[x] - mean_SE) / std_SE
    segment_energy_signal[x] = envelope
shannon_envelope = segment_energy_signal
#calculate envelope size
envelope_size = range(0, shannon_envelope.size)
#calculate envelope time
envelope_time = librosa.frames_to_time(envelope_size,hop_length=442)

plt.plot(shannon_envelope)
plt.show()

print(len(shannon_envelope))
print(len(recording))
print(len(resample(np.array(recording, dtype=np.float64), orig_sr=fs, target_sr=50)))