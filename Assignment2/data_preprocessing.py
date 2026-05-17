"""
MEG Signal Data Preprocessing Module

This module provides functions to load and preprocess MEG signal data from H5 files.
It includes normalization, downsampling, and label extraction utilities.
"""

import h5py
import numpy as np
import os
from collections import Counter


# ============================================================================
# Label mapping
# ============================================================================

LABEL_MAP = {
    "rest": 0,
    "task_story_math": 1,
    "task_working_memory": 2,
    "task_motor": 3
}


# ============================================================================
# Core preprocessing functions
# ============================================================================

def get_dataset_name(file_name_with_dir):
    """Extract dataset name from file path.
    
    Args:
        file_name_with_dir (str): Full file path
        
    Returns:
        str: Dataset name (filename without directory and extension)
    """
    filename_without_dir = os.path.basename(file_name_with_dir)
    name_no_ext, _ = os.path.splitext(filename_without_dir)
    dataset_name = '_'.join(name_no_ext.split('_')[:-1])
    return dataset_name


def read_h5_file(filename_path):
    """Read MEG signal data from H5 file.
    
    Args:
        filename_path (str): Path to H5 file
        
    Returns:
        np.ndarray: MEG signal matrix
    """
    dataset_name = get_dataset_name(filename_path)

    with h5py.File(filename_path, "r") as f:
        matrix = f.get(dataset_name)[()]

    return matrix


def zscore_normalize(matrix):
    """Normalize matrix using z-score normalization.
    
    Args:
        matrix (np.ndarray): Input matrix of shape (n_channels, n_timepoints)
        
    Returns:
        np.ndarray: Normalized matrix
    """
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)

    normalized_matrix = (matrix - mean) / (std + 1e-6)

    return normalized_matrix


def average_pool_downsample(matrix, factor=10):
    """Downsample matrix using average pooling.
    
    Args:
        matrix (np.ndarray): Input matrix of shape (n_channels, n_timepoints)
        factor (int): Downsampling factor (default: 10)
        
    Returns:
        np.ndarray: Downsampled matrix
    """
    n_channels, n_timepoints = matrix.shape

    usable_length = (n_timepoints // factor) * factor
    matrix = matrix[:, :usable_length]

    matrix = matrix.reshape(n_channels, usable_length // factor, factor)

    downsampled_matrix = matrix.mean(axis=2)

    return downsampled_matrix


def extract_label(filename_path):
    """Extract task label from filename.
    
    Args:
        filename_path (str): Path to H5 file
        
    Returns:
        int: Label (0=rest, 1=story_math, 2=working_memory, 3=motor)
        
    Raises:
        ValueError: If filename doesn't match any known task type
    """
    filename = os.path.basename(filename_path)

    if filename.startswith("rest"):
        return LABEL_MAP["rest"]
    elif filename.startswith("task_story_math"):
        return LABEL_MAP["task_story_math"]
    elif filename.startswith("task_working_memory"):
        return LABEL_MAP["task_working_memory"]
    elif filename.startswith("task_motor"):
        return LABEL_MAP["task_motor"]
    else:
        raise ValueError(f"Unknown task type in file: {filename}")


def preprocess_file(filename_path, downsample_factor=1):
    """Preprocess a single MEG signal file.
    
    Args:
        filename_path (str): Path to H5 file
        downsample_factor (int): Downsampling factor (default: 1, no downsampling)
        
    Returns:
        tuple: (X, y) where
            X (np.ndarray): Preprocessed MEG signal (float32)
            y (int): Task label
    """
    matrix = read_h5_file(filename_path)
    matrix = average_pool_downsample(
        matrix,
        factor=downsample_factor
    )
    matrix = zscore_normalize(matrix)

    matrix = matrix.astype(np.float32)

    label = extract_label(filename_path)

    return matrix, label


def load_and_preprocess_folder(folder_path, downsample_factor=1, verbose=True):
    """Load and preprocess all H5 files in a folder.
    
    Args:
        folder_path (str): Path to folder containing H5 files
        downsample_factor (int): Downsampling factor (default: 1)
        verbose (bool): Print processing information (default: True)
        
    Returns:
        tuple: (X, y, file_names) where
            X (np.ndarray): Preprocessed signals, shape (N, n_channels, n_timepoints)
            y (np.ndarray): Task labels, shape (N,)
            file_names (list): Original H5 filenames
    """
    X = []
    y = []
    file_names = []

    h5_files = [
        file for file in os.listdir(folder_path)
        if file.endswith(".h5")
    ]

    h5_files = sorted(h5_files)

    if verbose:
        print(f"\nProcessing folder: {folder_path}")
        print(f"Number of h5 files: {len(h5_files)}")

    for file in h5_files:
        filename_path = os.path.join(folder_path, file)

        x_i, y_i = preprocess_file(
            filename_path,
            downsample_factor=downsample_factor
        )

        X.append(x_i)
        y.append(y_i)
        file_names.append(file)

    X = np.stack(X).astype(np.float32)
    y = np.array(y)

    if verbose:
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Label distribution:", Counter(y))

    return X, y, file_names


def load_datasets(base_dir, downsample_factor=10, verbose=True):
    """Load and preprocess all datasets (train/test splits).
    
    Args:
        base_dir (str): Base directory containing 'Intra' and 'Cross' folders
        downsample_factor (int): Downsampling factor (default: 10)
        verbose (bool): Print processing information (default: True)
        
    Returns:
        dict: Dictionary with keys:
            'intra_train', 'intra_test', 'cross_train', 'cross_test1', 'cross_test2', 'cross_test3'
            Each value is a dict with keys: 'X', 'y', 'file_names'
            
    Example:
        >>> datasets = load_datasets(base_dir="/path/to/data", downsample_factor=10)
        >>> X_train = datasets["cross_train"]["X"]
        >>> y_train = datasets["cross_train"]["y"]
    """
    folders = {
        "intra_train": os.path.join(base_dir, "Intra", "train"),
        "intra_test": os.path.join(base_dir, "Intra", "test"),
        "cross_train": os.path.join(base_dir, "Cross", "train"),
        "cross_test1": os.path.join(base_dir, "Cross", "test1"),
        "cross_test2": os.path.join(base_dir, "Cross", "test2"),
        "cross_test3": os.path.join(base_dir, "Cross", "test3"),
    }

    datasets = {}
    for subject, folder_path in folders.items():
        X, y, file_names = load_and_preprocess_folder(
            folder_path,
            downsample_factor=downsample_factor,
            verbose=verbose
        )

        datasets[subject] = {
            "X": X,
            "y": y,
            "file_names": file_names
        }

    return datasets
