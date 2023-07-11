import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

from tqdm import tqdm, trange
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from Experiments.SegmentationLiang import generate_pcg_envelope, find_pcg_peaks, get_wav_and_tsv
from DataManipulation.PatientFrame import PatientFrame
from SegmentationCNN.Models.Envelope_CNN import PatientInfo

from Utilities.constants import TRAINING_DATA_PATH, DATA_CSV_PATH, TINY_TEST_DATA_PATH, TINY_TEST_CSV_PATH


def get_segmentation_array(files):
    for file in files:
        wav, true_seg, fs = get_wav_and_tsv(file)

    # return results_dict


def stratified_sample(csv_file, dataset_dir, folds=5):
    print("Loading data and initializing trials...")
    global fold_num 
    pf = PatientFrame(csv_file)
    # print("RUNNING")
    patient_info = PatientInfo(dataset_dir)
    patient_info.get_data()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    
    for train_index, test_index in skf.split(pf.patient_frame["Patient ID"], pf.patient_frame["Murmur"]):
        print(f"#### FOLD {fold_num} ####")
        patients_train, patients_test = pf.patient_frame["Patient ID"][train_index], pf.patient_frame["Patient ID"][test_index]
        training_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_train)]
        val_df = patient_info.patient_df.loc[patient_info.patient_df['ID'].isin(patients_test)]
        print(f"Training HMM...")
        # hmm_results = train_eval_HMM(training_df, val_df)
        print(f"Training CNN...")
        # cnn_results =prep_CNN(training_df, val_df)
        print(f"Saving results...")
        # save_results(hmm_results, "hmm", fold_num)
        # save_results(cnn_results, "cnn", fold_num)
        fold_num += 1 


def main():
    dataset_dir = TRAINING_DATA_PATH
    # csv_file = DATA_CSV_PATH
    # dataset_dir = TINY_TEST_DATA_PATH
    csv_file = TINY_TEST_CSV_PATH

    # 0. get some dummy data to test
    files = ["2530_AV"]#, "2530_MV", "2530_PV", "2530_TV"]

    for file in files:
        wav, true_seg, fs = get_wav_and_tsv(file, dataset_dir)
        ave_s_env = generate_pcg_envelope(wav, fs, t=0.02, overlap=0.01)
    
    plt.plot(ave_s_env)
    
    # 6. find pcg peaks
    s1_locs, s2_locs, other_locs = find_pcg_peaks(ave_s_env, fs)
    print(f"S1 locs array len {len(s1_locs)}")
    print(f"S2 locs array len {len(s2_locs)}")
    print(f"other locs array len {len(other_locs)}")


if __name__ == "__main__":
    main()