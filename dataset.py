import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def read_h5_file(filename_path):
    filename_without_dir = os.path.basename(filename_path)
    name_no_ext, _ = os.path.splitext(filename_without_dir)
    dataset_name = '_'.join(name_no_ext.split('_')[:-1])
    with h5py.File(filename_path, "r") as f:
        matrix = f.get(dataset_name)[()]
    return matrix

def zscore_normalize(matrix):
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / (std + 1e-6)

def average_pool_downsample(matrix, factor=10):
    n_channels, n_timepoints = matrix.shape
    usable_length = (n_timepoints // factor) * factor
    matrix = matrix[:, :usable_length]
    matrix = matrix.reshape(n_channels, usable_length // factor, factor)
    return matrix.mean(axis=2)

label_map = {
    "rest": 0,
    "task_story_math": 1,
    "task_working_memory": 2,
    "task_motor": 3
}

def extract_label(filename_path):
    filename = os.path.basename(filename_path)
    for k, v in label_map.items():
        if filename.startswith(k):
            return v
    raise ValueError(f"Unknown task type in file: {filename}")

class MEGDataset(Dataset):
    def __init__(self, folder_path, downsample_factor=10):
        self.folder_path = folder_path
        self.downsample_factor = downsample_factor
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        matrix = read_h5_file(file_path)
        
        matrix = average_pool_downsample(matrix, factor=self.downsample_factor)
        matrix = zscore_normalize(matrix)
        
        # Add channel dimension: (1, 248, 3562)
        matrix = np.expand_dims(matrix, axis=0).astype(np.float32)
        label = extract_label(file_path)
        
        return torch.from_numpy(matrix), torch.tensor(label, dtype=torch.long)
