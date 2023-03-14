import numpy as np 

def create_segmentation_array(recording,
                              tsv_segmentation,
                              recording_frequency,
                              feature_frequency=50):
    """

    Parameters
    ----------
    recording
    tsv_segmentation
    recording_frequency : int
        Frequency at which the recording is sampled
    feature_frequency : int
        Frequency of the features extracted in order to train the segmentation. The default, 50, is
        the frequency used in the matlab implementation

    Returns
    -------

    """

    full_segmentation_array = np.zeros(int(recording.shape[0] * feature_frequency / recording_frequency))

    for row_idx in range(0, tsv_segmentation.shape[0]):
        row = tsv_segmentation[row_idx, :]
        start_point = np.round(row[0] * feature_frequency).astype(int)
        end_point = np.round(row[1] * feature_frequency).astype(int)
        full_segmentation_array[start_point:end_point] = int(row[2])

    start_indices = []
    end_indices = []
    segmentations = []
    segment_started = False
    TOLERANCE = 5
    for idx in range(full_segmentation_array.shape[0]):
        if not segment_started and full_segmentation_array[idx] == 0:
            continue
        if full_segmentation_array[idx] != 0:
            if not segment_started:
                start_indices.append(idx)
                segment_started = True
                tol_counter = 0
            if tol_counter > 0:
                for adjust_idx in range(tol_counter):
                    full_segmentation_array[idx - adjust_idx - 1] = full_segmentation_array[idx - tol_counter - 1]
                tol_counter = 0
        if segment_started and full_segmentation_array[idx] == 0:
            tol_counter += 1
        if tol_counter == TOLERANCE or idx == full_segmentation_array.shape[0] - 1:
            end_indices.append(idx - tol_counter)
            if end_indices[-1] - start_indices[-1] > feature_frequency:
                segmentations.append(full_segmentation_array[start_indices[-1]:end_indices[-1]].astype(int))
            else:
                end_indices = end_indices[:-1]
                start_indices = start_indices[:-1]
            segment_started = False

    clipped_recordings = []
    for start, end in zip(start_indices, end_indices):
        clip = recording[int(start * recording_frequency / feature_frequency):int(end * recording_frequency / feature_frequency)]
        clipped_recordings.append(clip)

    return clipped_recordings, segmentations