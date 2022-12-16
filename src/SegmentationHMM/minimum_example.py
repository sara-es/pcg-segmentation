from utils import get_wavs_and_tsvs
from train_segmentation import train_hmm_segmentation
from run_segmentation import run_hmm_segmentation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm.contrib import tzip
from train_segmentation import train_hmm_segmentation, create_segmentation_array, get_full_recordings

# get list of recordings and corresponding segmentations (in the format given in the tsv)

wavs_and_tsvs_path = "/Users/serenahuston/GitRepos/python-classifier-2022/physionet.org/files/circor-heart-sound/1.0.3/training_data"
test_wavs_and_tsvs = "/Users/serenahuston/GitRepos/Springer-Segmentation-Python-main/test_wavs"
wavs, tsvs, names = get_wavs_and_tsvs(wavs_and_tsvs_path, return_names=True)

# wav_train, wav_test, tsv_train, tsv_test, name_train, name_test = train_test_split(wavs, tsvs, names, test_size=0.5, random_state=42)

# # train the model
# models, pi_vector, total_obs_distribution = train_hmm_segmentation(wav_train, tsv_train)


# for i in range(len(wav_test)):

#     # get segmentations out of the model for the first wav file in our list
#     annotation, heart_rate = run_hmm_segmentation(wav_test[i],
#                                         models,
#                                         pi_vector,
#                                         total_obs_distribution,
#                                         min_heart_rate=60,
#                                         max_heart_rate= 200,
#                                         return_heart_rate=True)
#     plt.plot(annotation)
#     plt.savefig(f"images/{name_test[i]}.pdf")
#     plt.clf()
def does_it_segment(data_folder, n_train=10):
    recordings, segmentations, names = get_wavs_and_tsvs(data_folder, return_names=True)

    models, pi_vector, total_obs_distribution= train_hmm_segmentation(recordings[:n_train], segmentations[:n_train])

    all_recordings, names = get_full_recordings(data_folder)
    for recording, name in tzip(all_recordings, names):
        annotation, hr = run_hmm_segmentation(recording,
                                              models,
                                              pi_vector,
                                              total_obs_distribution,
                                              use_psd=True,
                                              return_heart_rate=True)
        plt.plot(annotation)
        plt.savefig(f"images/{name}.pdf")
        plt.clf()
    


def how_does_it_do(data_folder=wavs_and_tsvs_path):
    # Get recordings and segmentations
    recordings, segmentations, names = get_wavs_and_tsvs(data_folder, return_names=True)
    ground_truth_segmentations = []
    clipped_recordings = []

    # Train HMM
    models, pi_vector, total_obs_distribution= train_hmm_segmentation(recordings[:5], segmentations[:5])

    # Get ground truth
    for rec, seg in zip(recordings[:500], segmentations[:500]):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                                    seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=4000)
  
        ground_truth_segmentations.append(ground_truth[0])
        clipped_recordings.append(clipped_recording[0])
        
    idx = 0
    accuracies = np.zeros(len(clipped_recordings[:500]))
    for rec, seg, name in tzip(clipped_recordings[:500], ground_truth_segmentations, names):
     
        annotation, hr = run_hmm_segmentation(rec,
                                              models,
                                              pi_vector,
                                              total_obs_distribution,
                                              use_psd=True,
                                              return_heart_rate=True,
                                              try_multiple_heart_rates=False)
     
        accuracies[idx] = (seg == annotation).mean()
        if accuracies[idx] < 0.5:
            print(name)

        idx += 1
    plt.hist(accuracies, color = "#e4cbf5", ec="#611d91")
    plt.xlabel("Accuracy")
    plt.ylabel("Number of Samples")
    plt.title("A Histogram to show the Accuracy Frequency on 500 Samples")
    plt.grid()
    
    print(f"average accuracy: {accuracies.mean()}")
    plt.show()


how_does_it_do()