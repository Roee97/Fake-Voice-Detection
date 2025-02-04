import os.path

import librosa
import random

import numpy as np
import pandas
import torch
import torch.nn.functional as F
import soundfile
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


def read_flac(file_path):
    """
    Reads a .frac file as raw audio data using torchaudio.
    Assumes the .frac file contains raw waveform data.
    """
    try:
        waveform, sample_rate = librosa.load(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

    return waveform, sample_rate


def preprocess_audio_fixed_length(waveform, sample_rate, target_sample_rate=16000, target_length=59049):
    """
    Preprocesses the audio waveform:
    - Resamples to the target sample rate.
    - Trims or pads to the target length.
    """

    if waveform is None:
        return None

    # Resample to the target sample rate
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)

    # Convert waveform to a PyTorch tensor
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # Flatten waveform in case it's multi-channel
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    # Trim or pad waveform to the target length
    if waveform.size(0) > target_length:
        waveform = waveform[:target_length]  # Trim
    elif waveform.size(0) < target_length:
        waveform = F.pad(waveform, (0, target_length - waveform.size(0)))  # Pad

    return waveform


def augment_file(file_path):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
    ])

    # Apply augmentation
    y, sr = librosa.load(file_path, sr=None)
    y_aug = augment(samples=y, sample_rate=sr)

    dir_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    soundfile.write(f"{dir_name}/aug_{file_name}", y_aug, sr)


def has_augmented_files(dir_path):
    for file_name in os.listdir(dir_path):
        if file_name.startswith("aug_"):
            return True
    return False


def add_augmentations(files_dir, dir_df: pandas.DataFrame, percent_of_file_to_augment=0.25):
    """
    create augmentation files and
    """
    if has_augmented_files(files_dir):
        print(f"Already has augmented files in {files_dir}, not creating again")
        return

    all_files = os.listdir(files_dir)
    num_of_files_to_augment = int(len(dir_df) * percent_of_file_to_augment)

    selected_files = dir_df.sample(num_of_files_to_augment)
    new_rows = []
    for _, row in selected_files.iterrows():
        orig_path = os.path.join(files_dir, f"{row['file_name']}.flac")
        new_row = row.copy()
        new_row['file_name'] = f"aug_{row['file_name']}"
        new_rows.append(new_row)
        augment_file(orig_path)

    return pandas.concat([dir_df, pandas.DataFrame(new_rows)], ignore_index=True)


class AudioDataset(Dataset):
    def __init__(self, df, data_dir, target_sample_rate=16000, target_length=3):
        self.df = df
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = f"{self.data_dir}/{row['file_name']}.flac"  # Path to .frac file
        label = row['label']  # 0 for real, 1 for fake
        waveform, sample_rate = read_flac(file_path)
        waveform = preprocess_audio_fixed_length(waveform, self.target_sample_rate)
        if waveform is None:
            return np.zeros(self.target_sample_rate), torch.tensor(0, dtype=torch.float32)
        return waveform, torch.tensor(label, dtype=torch.float32)