import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

from Utilities.constants import * 
import matplotlib.pyplot as plt
import numpy as np
import math 
import wave
import scipy as sp 
from scipy.fft import fft
from scipy.signal import stft 

import seaborn as sn
import pandas as pd

from Utilities.create_segmentation_array import * 

class DataPresentation:

    def __init__(self):
        self.colour_scheme = ["#611d91", "#a260d1", "#e4cbf5"]
        self.fig_width = 18.5
        self.fig_row_height = 4
        self.title_size = 14
        self.subtitle_size = 12
        self.ax_size = 10

    def plot_patient_audio_file(self, patient_num, wav_files):
        num_rows = math.ceil(len(wav_files)/2)
        fig, axs = plt.subplots(nrows=num_rows, ncols=2, constrained_layout = True)

        fig.suptitle('Audio recordings for patient ' + str(patient_num), fontsize=self.title_size)
        fig.set_size_inches(self.fig_width, num_rows*self.fig_row_height)

        row_count = 0
        col_count = 0 
        print(wav_files)
        for i in range(len(wav_files)):
            spf = wave.open(wav_files[i], "r")
            # Extract Raw Audio from Wav File
            signal = spf.readframes(-1)
            signal = np.frombuffer(signal, "int16")
            fs = spf.getframerate()

            # If Stereo
            if spf.getnchannels() == 2:
                print("Just mono files")
                sys.exit(0)

            time = np.linspace(0, len(signal) / fs, num=len(signal))

            if len(np.array(axs).shape) == 1:
                ax_list = axs[col_count]
            else:
                ax_list = axs[row_count, col_count]

            ax_list.set_title(wav_files[i], fontsize=self.subtitle_size)
            ax_list.plot(time, signal, color=self.colour_scheme[0])
            ax_list.grid()
            ax_list.set_ylabel("Amplitude", fontsize=self.ax_size)
            ax_list.set_xlabel("Time (Seconds)", fontsize=self.ax_size)
            ax_list.tick_params(axis='x', labelsize=self.ax_size)
            ax_list.tick_params(axis='y', labelsize=self.ax_size)
            ax_list.set_xlim(left=0, right=35)
            ax_list.set_ylim(bottom=-35000, top=35000)
            
            col_count = (col_count + 1) % 2 
            if (col_count == 0):
                row_count += 1

        if len(wav_files) % 2 == 1:
            if len(np.array(axs).shape) == 1:
                fig.delaxes(axs[-1])
            if len(np.array(axs).shape) > 1:
                fig.delaxes(axs[-1][-1])
        plt.savefig(DATA_PRESENTATION_PATH + str(patient_num) + "_audio_plots")

    def plot_signal(self, signal, title):
        homo_col = "#CA03BA"
        hilb_col = "#4E46FF"
        wav_col = "#0D712B"
        psd_col = "#E67F35"
    
        fig, axs = plt.subplots(nrows=1, ncols=1, constrained_layout = True)
        fig.set_size_inches(self.fig_width, self.fig_row_height)
        time = np.linspace(0, 19.35, num=len(signal))
        axs.set_title(title, fontsize=self.subtitle_size)
        axs.plot(time, signal, color=self.colour_scheme[0])
        axs.grid()
        axs.set_ylabel("Amplitude", fontsize=self.ax_size)
        axs.set_xlabel("Time (Seconds)", fontsize=self.ax_size)
        axs.tick_params(axis='x', labelsize=self.ax_size)
        axs.tick_params(axis='y', labelsize=self.ax_size)
        axs.set_xlim(left=0, right=20)
        axs.set_ylim(bottom=-8000, top=8000)
        plt.savefig(DATA_PRESENTATION_PATH + title.replace(" ", "_"))


    def plot_patient_audio_file_with_fhs_locs(self, patient_num, wav_file, fhs_locs):
        fig, ax = plt.subplots()
        fig.suptitle('Audio recordings for patient ' + str(patient_num), fontsize=self.title_size)
        fig.set_size_inches(self.fig_width, self.fig_row_height)

        spf = wave.open(wav_file, "r")
        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, "int16")
        fs = spf.getframerate()

        # If Stereo
        if spf.getnchannels() == 2:
            print("Just mono files")
            sys.exit(0)

        time = np.linspace(0, len(signal) / fs, num=len(signal))

        ax.set_title(wav_file, fontsize=self.subtitle_size)
        ax.plot(time, signal, color=self.colour_scheme[0])
        ax.vlines(fhs_locs, color=self.colour_scheme[2], ymin=-35000, ymax=35000)
        ax.grid()
        ax.set_ylabel("Amplitude", fontsize=self.ax_size)
        ax.set_xlabel("Time (Seconds)", fontsize=self.ax_size)
        ax.tick_params(axis='x', labelsize=self.ax_size)
        ax.tick_params(axis='y', labelsize=self.ax_size)
        ax.set_xlim(left=0, right=35)
        ax.set_ylim(bottom=-35000, top=35000)
        
        plt.savefig(DATA_PRESENTATION_PATH + str(patient_num) + "_audio_plot_with_FHS")

    def plot_multi_bar_chart(self, x_labels, x_label_title, y_label_title, data, errors, bar_labels, title):
        plt.title(title, fontsize=self.title_size)
        x_ticks = np.arange(len(x_labels))
        bar_spacing = np.linspace(-0.2, 0.2, num=len(bar_labels))
        for i in range(len(bar_labels)):
            plt.bar(x_ticks+bar_spacing[i], data[i], yerr=errors[i], capsize=10, width=0.4, 
                    label=bar_labels[i], color=self.colour_scheme[i], zorder=3)


        plt.xticks(x_ticks, x_labels, fontsize=self.ax_size)
        plt.xlabel(x_label_title, fontsize=self.ax_size)
        plt.ylabel(y_label_title, fontsize=self.ax_size)
        plt.grid(zorder=0)
        plt.legend(loc='lower right')    
        plt.savefig(DATA_PRESENTATION_PATH + title.replace(" ", "_"))

    def plot_confusion_matrix(self, confusion_matrix, title, xlabel, ylabel, xticks, yticks):
        rows = len(confusion_matrix)
        cols = len(confusion_matrix[0])
        df_cm = pd.DataFrame(confusion_matrix, range(rows), range(cols))
        plt.figure(figsize = (max(rows,cols)*2.5,max(rows,cols)*2.5))
        ax= sn.heatmap(df_cm, annot=True, xticklabels=xticks, yticklabels=yticks)
        ax.set_yticklabels(labels=ax.get_yticklabels(), va='center')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.title(title, fontsize=self.title_size, wrap=True)
        plt.xlabel(xlabel, fontsize=self.ax_size, labelpad=10)
        plt.ylabel(ylabel, fontsize=self.ax_size, labelpad=10)
        plt.savefig(DATA_PRESENTATION_PATH + title.replace(" ", "_"))

    def plot_boxplot(self, data_dict, title, ylabel):
        fig, ax = plt.subplots()
        ax.boxplot(data_dict.values(),
                    boxprops=dict( color="black"),
                    capprops=dict(color=self.colour_scheme[1]),
                    whiskerprops=dict(color="black"),
                    flierprops=dict(color="black", markeredgecolor=self.colour_scheme[1]),
                    medianprops=dict(color=self.colour_scheme[1]))
        ax.set_xticklabels(data_dict.keys(), fontsize=self.ax_size)
        ax.set_ylabel(ylabel, fontsize=self.ax_size)
        ax.set_title(title, fontsize=self.title_size, wrap=True)
        ax.grid()
        plt.savefig(DATA_PRESENTATION_PATH + title.replace(" ", "_"))

    def plot_model_comp_box_plots(self, cnn_accuracies, hmm_accuracies, fold_num):
        fig, ax = plt.subplots()
        ax.set_title('Distribution of the Model Accuracies in Fold ' + str(fold_num))
        ax.boxplot([cnn_accuracies, hmm_accuracies])
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Model")

        ax.set_ylim(-0.1, 1)

        ax.boxplot([cnn_accuracies, hmm_accuracies],
                        boxprops=dict( color="black"),
                        capprops=dict(color=self.colour_scheme[1]),
                        whiskerprops=dict(color="black"),
                        flierprops=dict(markeredgecolor=self.colour_scheme[1], markersize=3),
                        medianprops=dict(color=self.colour_scheme[1]))
        ax.grid()
        plt.savefig(DATA_PRESENTATION_PATH + "model_comp_accs_" + str(fold_num))

    def plot_loss_and_accuracy(self, train_loss, valid_loss, accuracy, data_pres_folder, fold_num):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(18, 10)
        fig.suptitle('Model Loss VS Accuracy Across Epochs', fontsize=self.title_size)
        ax1.plot(train_loss, color=self.colour_scheme[0])
        ax2.plot(valid_loss, self.colour_scheme[0])
        ax3.plot(accuracy, self.colour_scheme[1])
        ax1.set_ylabel("Training Loss")
        ax2.set_ylabel("Validation Loss")
        ax3.set_ylabel("Accuracy")
        ax1.set_xlabel("Epochs")
        ax2.set_xlabel("Epochs")
        ax3.set_xlabel("Epochs")
        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.savefig(data_pres_folder + "Loss VS Accuracy"  + str(fold_num))

    def plot_patient_fft(self, patient_ID):
        fs, recording = sp.io.wavfile.read(os.path.join(TRAINING_DATA_PATH, patient_ID + ".wav"))
        tsv = np.loadtxt(TRAINING_DATA_PATH + patient_ID + ".tsv", delimiter="\t")
        clipped_recording, segmentations = create_segmentation_array(recording,
                                                                            tsv,
                                                                            recording_frequency=4000,
                                                                            feature_frequency=4000)

        fourier = fft(clipped_recording[0])
        n = len(clipped_recording[0])
        timestep = 1/fs
        freq = np.fft.fftfreq(n, d=timestep)
        print(np.max(np.abs(fourier)))

        fig, ax = plt.subplots()
        fig.set_size_inches(self.fig_width, self.fig_row_height)
        # ax.plot(np.abs(fft_recording), color=self.colour_scheme[0])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Fast Fourier Transform of " + patient_ID)
        ax.grid()
        ax.set_xlim(left=0, right=700)
        ax.plot(freq[:int(n/2)], np.abs(fourier[:int(n/2)])/3388, color=self.colour_scheme[0])

        plt.savefig(DATA_PRESENTATION_PATH + "FFT_" + patient_ID)

    def plot_patient_stft(self, patient_ID):
        fs, recording = sp.io.wavfile.read(os.path.join(TRAINING_DATA_PATH, patient_ID + ".wav"))
        tsv = np.loadtxt(TRAINING_DATA_PATH + patient_ID + ".tsv", delimiter="\t")
        clipped_recording, segmentations = create_segmentation_array(recording,
                                                                            tsv,
                                                                            recording_frequency=4000,
                                                                            feature_frequency=4000)

        f, t, Zxx = stft(clipped_recording[0], fs=fs, nperseg=576, noverlap=504)
        print(np.max(np.abs(Zxx)))
        
        fig, ax = plt.subplots()
        fig.set_size_inches(self.fig_width, self.fig_row_height)
        # ax.plot(np.abs(fft_recording), color=self.colour_scheme[0])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel('Magnitude')
        ax.set_title("Fast Fourier Transform taken at Windows From " + patient_ID)
        ax.grid()

        
        # ax.set_ylim(-0.1, 600)
        ax.plot(np.abs(Zxx))



        # pxx, freq, t, cax = ax.specgram(clipped_recording[0], # first channel
        #                         Fs=fs,      # to get frequency axis in Hz
        #                         vmin=-40)
        # cbar = fig.colorbar(cax)
        # cbar.set_label('Intensity (dB)')
        # ax.axis("tight")



        plt.savefig(DATA_PRESENTATION_PATH + "STFT_2_" + patient_ID)


    def plot_STFT_shorter_window(self, patient_ID):
        fs, recording = sp.io.wavfile.read(os.path.join(TRAINING_DATA_PATH, patient_ID + ".wav"))
        tsv = np.loadtxt(TRAINING_DATA_PATH + patient_ID + ".tsv", delimiter="\t")
        clipped_recording, segmentations = create_segmentation_array(recording, tsv,
                                                                        recording_frequency=4000,
                                                                        feature_frequency=4000)

        for i in range(3):
            patch = clipped_recording[0][i*640:(i*640)+5120]
            f, t, Zxx = stft(patch, fs=fs, nperseg=576, noverlap=504, boundary=None, padded=False)
            plt.plot(np.abs(Zxx))
            plt.show()
            shortened_Zxx = Zxx[:150, :]
            plt.plot(np.abs(shortened_Zxx))
            plt.show()
            results = [] 
            print(shortened_Zxx)
            for i in range(0, len(shortened_Zxx), 20):
                mean = np.mean(shortened_Zxx[i:i+20, :], axis=0)
                results.append(np.abs(mean))
            plt.plot(results)
            plt.show()


    def plot_PCG_HMM_vs_CNN_segmentations(self, patient_ID, results_dir, signal, true_segmentation, cnn_seg, hmm_seg, clip=True):

        time_4000 = np.linspace(0, len(signal) / 4000, num=len(signal))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
        fig.set_size_inches(16, 8)
        fig.subplots_adjust(hspace=0)
        
        if clip: 
            ax1.plot(time_4000[:40000], signal[:40000], color=self.colour_scheme[0])        
            ax2.plot(time_4000[:40000], true_segmentation[:40000], color=self.colour_scheme[1])     
            ax3.plot(time_4000[:40000], cnn_seg[:40000], color=self.colour_scheme[1])
            ax4.plot(time_4000[:40000], hmm_seg[:40000], color=self.colour_scheme[1])
        else: 
            ax1.plot(time_4000, signal, color=self.colour_scheme[0])        
            ax2.plot(time_4000, true_segmentation, color=self.colour_scheme[1])     
            ax3.plot(time_4000, cnn_seg, color=self.colour_scheme[1])
            ax4.plot(time_4000, hmm_seg, color=self.colour_scheme[1])
    
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(0, end, 1))
        ax2.xaxis.set_ticks(np.arange(0, end, 1))
        ax3.xaxis.set_ticks(np.arange(0, end, 1))

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax1.set_title("Plot of PCG recording and segmentations for " + patient_ID)

        ax3.set_xlabel("Time (Seconds)")
        ax1.set_ylabel("PCG Recording")
        ax2.set_ylabel("True Segmentations")
        ax3.set_ylabel("Env Segmentations")
        ax4.set_ylabel("STFT Segmentations")

        plt.savefig(results_dir + patient_ID + "_True_VS_CNN_VS_HSMM")   
        plt.cla()
        plt.clf()   


    
    def plot_PCG_segmentations(self, patient_ID, results_dir, signal, true_segmentation, predicted_segmentation, clip=True):

        time_4000 = np.linspace(0, len(signal) / 4000, num=len(signal))

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
        fig.set_size_inches(16, 8)
        fig.subplots_adjust(hspace=0)
        
        if clip: 
            ax1.plot(time_4000[:40000], signal[:40000], color=self.colour_scheme[0])        
            ax2.plot(time_4000[:40000], true_segmentation[:40000], color=self.colour_scheme[1])     
            ax3.plot(time_4000[:40000], predicted_segmentation[:40000], color=self.colour_scheme[1])
        else: 
            ax1.plot(time_4000, signal, color=self.colour_scheme[0])        
            ax2.plot(time_4000, true_segmentation, color=self.colour_scheme[1])     
            ax3.plot(time_4000, predicted_segmentation, color=self.colour_scheme[1])
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(0, end, 1))
        ax2.xaxis.set_ticks(np.arange(0, end, 1))
        ax3.xaxis.set_ticks(np.arange(0, end, 1))

        ax1.grid()
        ax2.grid()
        ax3.grid()

        ax1.set_title("Plot of PCG recording and segmentations for " + patient_ID)


        ax3.set_xlabel("Time (Seconds)")
        ax1.set_ylabel(" PCG Recording")
        ax2.set_ylabel("True Segmentations")
        ax3.set_ylabel("Predicted Segmentations")

        plt.savefig(results_dir + patient_ID + "_True_VS_CNN")   
        plt.cla()
        plt.clf()   

    def plot_PCG_HMM_vs_CNN_vs_STFT_segmentations(self, patient_ID, results_dir, signal, true_segmentation, 
                                                  hmm_seg, cnn_seg, stft_seg, clip=True):
        time_4000 = np.linspace(0, len(signal) / 4000, num=len(signal))

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True)
        fig.set_size_inches(16, 8)
        fig.subplots_adjust(hspace=0)

        if clip: 
            ax1.plot(time_4000[20000:60000], signal[20000:60000], color=self.colour_scheme[0])        
            ax2.plot(time_4000[20000:60000], true_segmentation[20000:60000], color=self.colour_scheme[1])     
            ax3.plot(time_4000[20000:60000], hmm_seg[20000:60000], color=self.colour_scheme[1])
            ax4.plot(time_4000[20000:60000], cnn_seg[20000:60000], color=self.colour_scheme[1])
            ax5.plot(time_4000[20000:60000], stft_seg[20000:60000], color=self.colour_scheme[1])
        else: 
            ax1.plot(time_4000, signal, color=self.colour_scheme[0])        
            ax2.plot(time_4000, true_segmentation, color=self.colour_scheme[1])     
            ax3.plot(time_4000, hmm_seg, color=self.colour_scheme[1])
            ax4.plot(time_4000, cnn_seg, color=self.colour_scheme[1])
            ax5.plot(time_4000, stft_seg, color=self.colour_scheme[1])
    
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(5, end, 1))
        ax2.xaxis.set_ticks(np.arange(5, end, 1))
        ax3.xaxis.set_ticks(np.arange(5, end, 1))

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax5.grid()
        ax1.set_title("Plot of PCG recording and segmentations for " + patient_ID)

        ax3.set_xlabel("Time (Seconds)")
        ax1.set_ylabel("PCG Recording")
        ax2.set_ylabel("True Segmentations")
        ax3.set_ylabel("HSMM Segmentations")
        ax4.set_ylabel("Env Segmentations")
        ax5.set_ylabel("STFT Segmentations")

        plt.savefig(results_dir + patient_ID + "_True_VS_CNN_VS_HSMM_VS_STFT")   
        plt.cla()
        plt.clf()   



    


