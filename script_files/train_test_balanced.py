from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

def train_test_balanced_function(        
        normal_spectra,
        normal_labels,
        normal_paths,
        anomaly_spectra,
        anomaly_labels,
        anomaly_paths,
        normal_voxel_weighted_concentrations_of_voxels,
        anomaly_voxel_weighted_concentrations_of_voxels
        ):

    # Split NORMAL data
    normal_train, normal_test, normal_train_labels, normal_test_labels, normal_train_paths, normal_test_paths, normal_concentrations_of_voxelsentrations_of_train_data, normal_test_concentrations_of_voxels = train_test_split(
        normal_spectra,
        normal_labels,
        normal_paths,
        normal_voxel_weighted_concentrations_of_voxels,
        test_size=0.2,
        random_state=21
    )

    # Split ANOMALY data
    anomaly_train, anomaly_test, anomaly_train_labels, anomaly_test_labels, anomaly_train_paths, anomaly_test_paths, anomaly_concentrations_of_voxelsentrations_of_train_data, anomaly_test_concentrations_of_voxels = train_test_split(
        anomaly_spectra,
        anomaly_labels,
        anomaly_paths,
        anomaly_voxel_weighted_concentrations_of_voxels,
        test_size=0.2,
        random_state=21
    )

    # Balance test set
    fifty_fifty_anom_norm_data = min(len(normal_test), len(anomaly_test))

    normal_test = normal_test[:fifty_fifty_anom_norm_data]
    anomaly_test = anomaly_test[:fifty_fifty_anom_norm_data]

    normal_test_labels = normal_test_labels[:fifty_fifty_anom_norm_data]
    anomaly_test_labels = anomaly_test_labels[:fifty_fifty_anom_norm_data]

    normal_test_paths = normal_test_paths[:fifty_fifty_anom_norm_data]
    anomaly_test_paths = anomaly_test_paths[:fifty_fifty_anom_norm_data]

    normal_test_concentrations_of_voxels = normal_test_concentrations_of_voxels[:fifty_fifty_anom_norm_data]
    anomaly_test_concentrations_of_voxels = anomaly_test_concentrations_of_voxels[:fifty_fifty_anom_norm_data]

    # TRAIN SET
    train_data = normal_train
    train_labels = normal_train_labels
    train_paths = normal_train_paths
    concentrations_of_train_data = normal_concentrations_of_voxelsentrations_of_train_data

    # TEST SET
    test_data = np.concatenate([normal_test, anomaly_test], axis=0)
    test_labels = np.concatenate([normal_test_labels, anomaly_test_labels], axis=0)
    test_paths = normal_test_paths + anomaly_test_paths
    test_concentrations_of_voxels = normal_test_concentrations_of_voxels + anomaly_test_concentrations_of_voxels

    return train_data, train_labels, train_paths, test_data, test_labels, test_paths, test_concentrations_of_voxels