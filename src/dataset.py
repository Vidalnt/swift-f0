import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
from typing import Tuple, Optional
import random

class SwiftF0Dataset(Dataset):
    """
    Dataset class for SwiftF0 training.
    
    Handles loading of audio files and their corresponding pitch annotations,
    with data augmentation and preprocessing.
    """
    
    def __init__(self, 
                 data_paths: list,
                 sample_rate: int = 16000,
                 frame_length: int = 1024,
                 hop_length: int = 256,
                 augment: bool = True,
                 noise_factor: float = 0.001):
        """
        Initialize the dataset.
        
        Args:
            data_paths: List of paths to data files (audio + pitch annotations)
            sample_rate: Target sample rate for audio
            frame_length: STFT frame length
            hop_length: STFT hop length
            augment: Whether to apply data augmentation
            noise_factor: Factor for additive noise augmentation
        """
        self.data_paths = data_paths
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.augment = augment
        self.noise_factor = noise_factor
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            tuple: (spectrogram, pitch_classification_target, pitch_regression_target)
        """
        # Load data file
        data_path = self.data_paths[idx]
        audio_path = data_path.replace('.f0', '.wav')
        pitch_path = data_path.replace('.wav', '.f0')
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Apply data augmentation if enabled
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Compute STFT magnitude spectrogram
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft)  # Shape: (n_freq_bins, n_frames)
        
        # Load pitch annotations
        pitch_data = np.loadtxt(pitch_path)  # Assuming format: [time, f0, confidence]
        times = pitch_data[:, 0]
        f0_values = pitch_data[:, 1]
        confidences = pitch_data[:, 2]
        
        # Align pitch annotations with STFT frames
        n_frames = magnitude.shape[1]
        frame_times = np.arange(n_frames) * self.hop_length / self.sample_rate
        
        # Interpolate pitch values to match frame times
        f0_aligned = np.interp(frame_times, times, f0_values, left=0, right=0)
        confidence_aligned = np.interp(frame_times, times, confidences, left=0, right=0)
        
        # Create targets
        # Only use frames with high confidence
        voiced_mask = confidence_aligned > 0.5
        f0_voiced = f0_aligned[voiced_mask]
        
        if len(f0_voiced) == 0:
            # No voiced frames, return dummy data
            spectrogram = torch.zeros(1, magnitude.shape[0], magnitude.shape[1])
            classification_target = torch.zeros(360)  # Dummy target
            regression_target = torch.tensor([0.0])
            return spectrogram, classification_target, regression_target
        
        # Select a random voiced frame for training
        frame_idx = np.random.choice(np.where(voiced_mask)[0])
        target_f0 = f0_aligned[frame_idx]
        
        # Convert to classification target (one-hot-like distribution)
        classification_target = self._f0_to_classification_target(target_f0)
        
        # Regression target (log frequency)
        regression_target = torch.tensor([np.log(target_f0)])
        
        # Extract spectrogram for the selected frame
        # Use a context window around the frame
        context_frames = 5
        start_frame = max(0, frame_idx - context_frames)
        end_frame = min(n_frames, frame_idx + context_frames + 1)
        
        spectrogram_segment = magnitude[:, start_frame:end_frame]
        
        # Normalize spectrogram
        spectrogram_segment = (spectrogram_segment - np.mean(spectrogram_segment)) / (np.std(spectrogram_segment) + 1e-8)
        
        # Convert to tensor and add channel dimension
        spectrogram = torch.from_numpy(spectrogram_segment).float().unsqueeze(0)  # Add channel dim
        
        return spectrogram, classification_target, regression_target
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Augmented audio signal
        """
        # Additive noise
        if random.random() < 0.5:
            noise = np.random.randn(*audio.shape) * self.noise_factor * np.std(audio)
            audio = audio + noise
        
        # Pitch shift
        if random.random() < 0.3:
            n_steps = random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        
        # Time stretching
        if random.random() < 0.2:
            rate = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        return audio
    
    def _f0_to_classification_target(self, f0: float) -> torch.Tensor:
        """
        Convert F0 value to classification target using Gaussian distribution.
        
        Args:
            f0: Fundamental frequency in Hz
            
        Returns:
            Classification target as probability distribution
        """
        # Create log-spaced frequency bins
        f_min, f_max, n_bins = 46.875, 2093.75, 360
        freq_bins = f_min * (f_max / f_min) ** (np.arange(n_bins) / (n_bins - 1))
        
        # Convert F0 to log frequency
        log_f0 = np.log(f0 + 1e-8)  # Add small value to avoid log(0)
        log_freq_bins = np.log(freq_bins)
        
        # Create Gaussian distribution centered at the F0
        sigma = 0.1  # Standard deviation in log frequency space
        distances = (log_freq_bins - log_f0) ** 2
        probabilities = np.exp(-distances / (2 * sigma ** 2))
        
        # Normalize
        probabilities = probabilities / (np.sum(probabilities) + 1e-8)
        
        return torch.from_numpy(probabilities).float()

def create_dataloader(data_paths: list, batch_size: int = 32, shuffle: bool = True, **kwargs) -> DataLoader:
    """
    Create a DataLoader for SwiftF0 training.
    
    Args:
        data_paths: List of paths to data files
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments for dataset initialization
        
    Returns:
        DataLoader instance
    """
    dataset = SwiftF0Dataset(data_paths, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)