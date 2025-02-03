import librosa
import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import resample
from torch.utils.data import Dataset


def read_frac(file_path):
    """
    Reads a .frac file as raw audio data using torchaudio.
    Assumes the .frac file contains raw waveform data.
    """
    # Load the audio file using torchaudio.load
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
        waveform, sample_rate = read_frac(file_path)
        waveform = preprocess_audio_fixed_length(waveform, self.target_sample_rate)
        return waveform, torch.tensor(label, dtype=torch.float32)