import matplotlib.pyplot as plt 
from scipy.fft import ifft
import math 
import numpy as np 


class PlotSignals:

    DATA_PRESENTATION_PATH = "/Users/serenahuston/GitRepos/ThirdYearProject/DataPresentation/"

    def __init__(self):
        self.colour_scheme = ["#611d91", "#a260d1", "#e4cbf5"]
        self.fig_width = 18.5
        self.fig_row_height = 4
        self.title_size = 14
        self.subtitle_size = 12
        self.ax_size = 10

    def plot_envelopes(self, signals, filename):
        titles = ["Homomorphic Envelope", "Hilbert Envelope", "Wavelet Envelope", "PSD Envelope"]
        num_rows = math.ceil(len(signals)/2)
        fig, axs = plt.subplots(nrows=num_rows, ncols=2, constrained_layout = True)

        fig.suptitle('Normalised Envelopes of audio: ' + filename  , fontsize=self.title_size)
        fig.set_size_inches(self.fig_width, num_rows*self.fig_row_height)

        row_count = 0
        col_count = 0 
        for i in range(len(signals)):
            
            if len(np.array(axs).shape) == 1:
                ax_list = axs[col_count]
            else:
                ax_list = axs[row_count, col_count]

            ax_list.set_title(titles[i], fontsize=self.subtitle_size)
            ax_list.plot(signals[i], color=self.colour_scheme[0])
            ax_list.grid()
            ax_list.set_ylabel("Amplitude", fontsize=self.ax_size)
            ax_list.set_xlabel("Time (Seconds)", fontsize=self.ax_size)
            ax_list.tick_params(axis='x', labelsize=self.ax_size)
            ax_list.tick_params(axis='y', labelsize=self.ax_size)

            
            col_count = (col_count + 1) % 2 
            if (col_count == 0):
                row_count += 1

        if len(signals) % 2 == 1:
            if len(np.array(axs).shape) == 1:
                fig.delaxes(axs[-1])
            if len(np.array(axs).shape) > 1:
                fig.delaxes(axs[-1][-1])
        plt.savefig(self.DATA_PRESENTATION_PATH + filename.split(".")[0]+ "_envelope_plots")
