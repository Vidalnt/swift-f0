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
    with data augmentation and preprocessing matching the paper.
    """
    
    def __init__(self, 
                 data_paths: list,
                 sample_rate: int = 16000,
                 hop_length: int = 256,
                 n_fft: int = 1024,
                 n_bins: int = 200,
                 f_min: float = 46.875,
                 f_max: float = 2093.75,
                 augment: bool = True,
                 noise_snr_range: Tuple[float, float] = (10, 30),  # SNR range in dB
                 gain_db_range: Tuple[float, float] = (-6, 6)):   # Gain range in dB
        """
        Initialize the dataset.
        
        Args:
            data_paths: List of paths to data files (audio + pitch annotations)
            sample_rate: Target sample rate for audio
            hop_length: STFT hop length
            n_fft: STFT window size
            n_bins: Number of pitch bins
            f_min: Minimum frequency in Hz
            f_max: Maximum frequency in Hz
            augment: Whether to apply data augmentation
            noise_snr_range: Range of SNR values for noise addition (dB)
            gain_db_range: Range of gain adjustments (dB)
        """
        self.data_paths = data_paths
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_bins = n_bins
        self.f_min = f_min
        self.f_max = f_max
        self.augment = augment
        self.noise_snr_range = noise_snr_range
        self.gain_db_range = gain_db_range
        
        # Create log-spaced frequency bins
        self.pitch_bins = self._create_log_spaced_bins()
        
    def _create_log_spaced_bins(self) -> np.ndarray:
        """Create log-spaced frequency bins."""
        return self.f_min * (self.f_max / self.f_min) ** (np.arange(self.n_bins) / (self.n_bins - 1))
        
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            tuple: (audio, pitch_classification_target, target_f0)
                - audio: Raw audio tensor (1, audio_length)
                - pitch_classification_target: Target probability distribution (n_bins, n_frames)
                - target_f0: Target F0 values in Hz (n_frames,)
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
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # Add batch dimension
        
        # Load pitch annotations
        pitch_data = np.loadtxt(pitch_path)  # Assuming format: [time, f0, confidence]
        times = pitch_data[:, 0]
        f0_values = pitch_data[:, 1]
        confidences = pitch_data[:, 2]
        
        # Compute STFT to get frame times
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        n_frames = stft.shape[1]
        frame_times = np.arange(n_frames) * self.hop_length / self.sample_rate
        
        # Interpolate pitch values to match frame times
        f0_aligned = np.interp(frame_times, times, f0_values, left=0, right=0)
        confidence_aligned = np.interp(frame_times, times, confidences, left=0, right=0)
        
        # Create targets only for voiced frames with high confidence
        voiced_mask = confidence_aligned > 0.5
        f0_voiced = f0_aligned[voiced_mask]
        
        if len(f0_voiced) == 0:
            # No voiced frames, return dummy data
            dummy_target = torch.zeros(self.n_bins, n_frames)
            dummy_f0 = torch.zeros(n_frames)
            return audio_tensor, dummy_target, dummy_f0
        
        # Convert F0 values to classification targets
        classification_targets = self._f0_to_classification_targets(f0_aligned, voiced_mask)
        
        # Target F0 values (for regression loss)
        target_f0 = torch.from_numpy(f0_aligned).float()
        
        return audio_tensor, classification_targets, target_f0
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to audio signal, matching paper description.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Augmented audio signal
        """
        # Gain adjustment
        if random.random() < 0.5:
            gain_db = random.uniform(self.gain_db_range[0], self.gain_db_range[1])
            gain_factor = 10**(gain_db/20)  # Convert dB to linear scale
            audio = audio * gain_factor
            
        # Add noise with SNR control (matching paper description)
        if random.random() < 0.5:
            # Generate noise (could be white, pink, or brown noise)
            noise_type = random.choice(['white', 'pink', 'brown'])
            noise = self._generate_noise(len(audio), noise_type)
            
            # Calculate SNR
            snr_db = random.uniform(self.noise_snr_range[0], self.noise_snr_range[1])
            
            # Calculate signal and noise power
            signal_power = np.mean(audio**2)
            noise_power = np.mean(noise**2)
            
            # Calculate scaling factor for desired SNR
            if noise_power > 0:
                scaling_factor = np.sqrt(signal_power / (noise_power * 10**(snr_db/10)))
                audio = audio + scaling_factor * noise
                
        # Pitch shift (small variations)
        if random.random() < 0.3:
            n_steps = random.uniform(-0.5, 0.5)  # Small pitch shifts
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
            
        # Clipping to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _generate_noise(self, length: int, noise_type: str = 'white') -> np.ndarray:
        """
        Generate different types of noise.
        
        Args:
            length: Length of noise signal
            noise_type: Type of noise ('white', 'pink', 'brown')
            
        Returns:
            Noise signal
        """
        if noise_type == 'white':
            return np.random.randn(length)
        elif noise_type == 'pink':
            # Simple pink noise approximation
            white = np.random.randn(length)
            # Apply a simple filter to approximate pink noise
            pink = np.zeros_like(white)
            pink[0] = white[0]
            for i in range(1, length):
                pink[i] = 0.997 * pink[i-1] + white[i] * 0.033
            return pink
        elif noise_type == 'brown':
            # Brown noise (integration of white noise)
            white = np.random.randn(length)
            brown = np.cumsum(white)
            # Normalize to same power as white noise
            brown = brown / np.std(brown) * np.std(white)
            return brown
        else:
            return np.random.randn(length)
    
    def _f0_to_classification_targets(self, f0_values: np.ndarray, voiced_mask: np.ndarray) -> torch.Tensor:
        """
        Convert F0 values to classification targets using Gaussian distribution.
        
        Args:
            f0_values: Array of F0 values in Hz
            voiced_mask: Boolean mask indicating voiced frames
            
        Returns:
            Classification targets as probability distributions (n_bins, n_frames)
        """
        n_frames = len(f0_values)
        targets = np.zeros((self.n_bins, n_frames))
        
        # Convert pitch bins to log scale
        log_pitch_bins = np.log(self.pitch_bins)
        
        # Standard deviation in log frequency space (matching paper)
        sigma = 0.1
        
        for i in range(n_frames):
            if voiced_mask[i] and f0_values[i] > 0:
                # Convert F0 to log frequency
                log_f0 = np.log(f0_values[i])
                
                # Create Gaussian distribution centered at the F0
                distances = (log_pitch_bins - log_f0) ** 2
                probabilities = np.exp(-distances / (2 * sigma ** 2))
                
                # Normalize
                probabilities_sum = np.sum(probabilities)
                if probabilities_sum > 0:
                    probabilities = probabilities / probabilities_sum
                    targets[:, i] = probabilities
                else:
                    # If normalization fails, use a sharp peak
                    closest_bin = np.argmin(np.abs(self.pitch_bins - f0_values[i]))
                    targets[closest_bin, i] = 1.0
            # For unvoiced frames, targets remain zero (no probability mass)
        
        return torch.from_numpy(targets).float()

def create_dataloader(data_paths: list, 
                      batch_size: int = 32, 
                      shuffle: bool = True, 
                      **kwargs) -> DataLoader:
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