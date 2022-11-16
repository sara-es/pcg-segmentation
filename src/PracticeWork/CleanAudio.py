from scipy.fftpack import rfft, rfftfreq, irfft
import numpy as np 
from FirstModel import PatientData
import matplotlib.pyplot as plt
import torch


dataset = PatientData("training_data.csv")
audio_filenames = dataset.get_patient_audio_file_names("29378")

audio_tensor, _ = dataset.get_audio_tensor_from_file(audio_filenames[0])



data_step = 0.001
t = np.arange(start=0,stop=1,step=data_step)


n = len(t)
yf = rfft(torch.flatten(audio_tensor).numpy())
xf = rfftfreq(n,data_step)

plt.plot(xf,np.abs(yf))
plt.savefig("Cleaning1")
plt.clf()   

yf_abs = np.abs(yf) 
indices = yf_abs>10   # filter out those value under 300
yf_clean = indices * yf # noise frequency will be set to 0
plt.plot(xf,np.abs(yf_clean))
plt.savefig("Cleaning2")
plt.clf()   

new_f_clean = irfft(yf_clean)
plt.plot(t,new_f_clean)
plt.ylim(-6,8)

plt.savefig("Cleaning3")