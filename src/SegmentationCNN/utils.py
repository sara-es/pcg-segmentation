import os
import sys

import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
import re 


def get_wavs_and_tsvs(input_folder=None, return_names=False):
    """

    Parameters
    ----------
    input_folder

    Returns
    -------

    """
    wav_arrays = []
    tsv_arrays = []
    fs_arrays = [] 
    if return_names:
        names = []
    if input_folder is None:
        input_folder = sys.argv[1]

    for file_ in tqdm(sorted(os.listdir(input_folder))):
        root, extension = os.path.splitext(file_)
        if extension == ".wav":
            segmentation_file = os.path.join(input_folder, root + ".tsv")
            if not os.path.exists(segmentation_file):
                continue
            fs, recording = scipy.io.wavfile.read(os.path.join(input_folder, file_))
            wav_arrays.append(recording)
            fs_arrays.append(fs)
            if return_names:
                names.append(file_)

            tsv_segmentation = np.loadtxt(segmentation_file, delimiter="\t")
            tsv_arrays.append(tsv_segmentation)
    if return_names:
        return wav_arrays, tsv_arrays, fs_arrays, names
    return wav_arrays, tsv_arrays, fs_arrays


def get_wavs_and_tsvs_by_regex(regex, input_folder, return_names=False):
    wav_arrays = []
    tsv_arrays = []
    fs_arrays = [] 
    if return_names:
        names = []
    if input_folder is None:
        input_folder = sys.argv[1]

    file_list = sorted(filter(re.compile(regex ).match, os.listdir(input_folder)))
    for file_ in tqdm(file_list):
        root, extension = os.path.splitext(file_)
        if extension == ".wav":
            segmentation_file = os.path.join(input_folder, root + ".tsv")
            if not os.path.exists(segmentation_file):
                continue
            fs, recording = scipy.io.wavfile.read(os.path.join(input_folder, file_))
            wav_arrays.append(recording)
            fs_arrays.append(fs)
            if return_names:
                names.append(file_)

            tsv_segmentation = np.loadtxt(segmentation_file, delimiter="\t")
            tsv_arrays.append(tsv_segmentation)
    if return_names:
        return wav_arrays, tsv_arrays, fs_arrays, names
    return wav_arrays, tsv_arrays, fs_arrays

