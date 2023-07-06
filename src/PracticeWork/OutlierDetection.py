import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

from sklearn.ensemble import IsolationForest
from DataManipulation.AudioDataset import AudioData
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from Utilities.constants import DATA_PRESENTATION_PATH
import statistics


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from DataPresentation import DataPresentation

audio_data = AudioData("training_data.csv")

X_train, X_test, wav_files_train, wav_files_test = train_test_split(audio_data.audio_tensor.numpy(), audio_data.wav_files, test_size=0.3, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = IsolationForest(random_state=47)

param_grid = {'n_estimators': [1000, 1500], 
              'max_samples': [10], 
              'contamination': ['auto', 0.025, 0.05, 0.075, 0.1], 
              'max_features': [10, 15], 
              'bootstrap': [True], 
              'n_jobs': [-1]}

# grid_search = GridSearchCV(model, 
#                             param_grid,
#                             scoring="neg_mean_squared_error", 
#                             refit=True,
#                             cv=10, 
#                             return_train_score=True)

# grid_search.fit(X_train, y_train)

# best_model = grid_search.fit(X_train, y_train)
# print('Optimum parameters', best_model.best_params_)

iforest = IsolationForest(bootstrap=True,
                          contamination=0.07, 
                          max_features=10, 
                          max_samples=10, 
                          n_estimators=1000, 
                          n_jobs=-1,
                         random_state=1)

iforest.fit(X_train)
y_pred = pd.Series(iforest.predict(X_test))


noisy_wav_files = wav_files_test.iloc[y_pred.index[y_pred==-1]].sort_values(ascending=True)
clean_wav_files = wav_files_test.iloc[y_pred.index[y_pred==1]].sort_values(ascending=True)


data_pres = DataPresentation()


# sample = noisy_wav_files.sample(n=6).reset_index(drop=True)

# print(sample)
# data_pres.plot_patient_audio_file("SAMPLE", sample)


# Investigating the Length of Noisy WAV Files
boxplot_data = {"Noisy Files":audio_data.get_audio_duration(noisy_wav_files), "Clean Files":audio_data.get_audio_duration(clean_wav_files)}

data_pres.plot_boxplot(boxplot_data,
                       "Boxplots of WAV File Durations, partitioned by the Outlier Detection Results",
                       "Duration (Seconds)")



plt.clf()
print(statistics.mean(boxplot_data.get("Noisy Files")))
print(statistics.mean(boxplot_data.get("Clean Files")))
plt.figure(figsize=(40, 25))
plt.bar(noisy_wav_files.str.split(pat="/").str[-1], boxplot_data.get("Noisy Files"), color="#611d91", zorder=3)
plt.axhline(y=statistics.mean(boxplot_data.get("Clean Files")), color='r', linestyle='-')
plt.ylabel("Duration (Seconds)", fontsize=20)
plt.xlabel("WAV Files", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(rotation=45, ha="right", fontsize=20)
plt.title("A Bar Chart to Show the Durations of Recordings Deemed 'Noisy' by the Outlier Detection Process", wrap=True, fontsize=24)
plt.grid(zorder=0)
plt.savefig(DATA_PRESENTATION_PATH + "NoisyRecordingDurations")