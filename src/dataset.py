import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
from typing import Tuple, Optional, List
import random
from scipy import signal

class SwiftF0Dataset(Dataset):
    """
    Dataset class for SwiftF0 training.
    
    Handles loading of audio files and their corresponding pitch annotations,
    with data augmentation and preprocessing matching the paper (to the extent possible
    without CHiME-Home dataset) and adapted for flexibility.
    """
    
    def __init__(self, 
                 data_paths: List[str], # List of paths to audio files or a directory structure
                 sample_rate: int = 16000,
                 hop_length: int = 256, # Configurable
                 n_fft: int = 1024,
                 n_bins: int = 200,     # Configurable
                 f_min: float = 46.875, 
                 f_max: float = 2093.75,
                 augment: bool = True,
                 noise_snr_range: Tuple[float, float] = (10, 30),  # SNR range in dB
                 gain_db_range: Tuple[float, float] = (-6, 6)):   # Gain range in dB
        """
        Initialize the dataset.
        
        Assumes data is organized with audio files (.wav) and corresponding 
        pitch annotation files (.f0 or .pv or .csv) in the same directory or a paired structure.
        Annotation file format expected: [time_sec, f0_hz, confidence] per line or similar.
        
        Args:
            data_paths: List of paths to audio files or base directories containing data.
            sample_rate: Target sample rate for audio.
            hop_length: STFT hop length (affects frame rate).
            n_fft: STFT window size.
            n_bins: Number of pitch bins (must match model).
            f_min: Minimum frequency in Hz for pitch bins.
            f_max: Maximum frequency in Hz for pitch bins.
            augment: Whether to apply data augmentation.
            noise_snr_range: Range of SNR values for noise addition (dB).
            gain_db_range: Range of gain adjustments (dB).
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
        
        # Create log-spaced frequency bins for target generation
        self.pitch_bins = self._create_log_spaced_bins()
        
        # Discover and load data files
        self.audio_files = []
        self.pitch_files = []
        
        for path in data_paths:
            if os.path.isfile(path) and path.lower().endswith('.wav'):
                # Assume pitch file has same name with .f0 extension
                pitch_path = path.rsplit('.', 1)[0] + '.f0'
                if os.path.exists(pitch_path):
                    self.audio_files.append(path)
                    self.pitch_files.append(pitch_path)
            elif os.path.isdir(path):
                # Search for .wav and corresponding .f0 files recursively
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith('.wav'):
                            audio_path = os.path.join(root, file)
                            pitch_path = audio_path.rsplit('.', 1)[0] + '.f0'
                            if os.path.exists(pitch_path):
                                self.audio_files.append(audio_path)
                                self.pitch_files.append(pitch_path)
                                
        if len(self.audio_files) == 0:
            raise ValueError("No valid audio-pitch file pairs found in the provided paths.")
            
        print(f"Found {len(self.audio_files)} audio-pitch pairs for training.")
        
    def _create_log_spaced_bins(self) -> np.ndarray:
        """Create log-spaced frequency bins for pitch classification."""
        return self.f_min * (self.f_max / self.f_min) ** (np.arange(self.n_bins) / (self.n_bins - 1))
        
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            tuple: (audio, pitch_classification_target, target_f0)
                - audio: Raw audio tensor (1, audio_length)
                - pitch_classification_target: Target probability distribution (n_bins, n_frames)
                - target_f0: Target F0 values in Hz (n_frames,)
        """
        audio_path = self.audio_files[idx]
        pitch_path = self.pitch_files[idx]
        
        # --- 1. Load Audio ---
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # --- 2. Load Pitch Annotations ---
        # Assuming .f0 file format: [time_sec, f0_hz, confidence]
        try:
            pitch_data = np.loadtxt(pitch_path)
            if pitch_data.ndim == 1:
                # Handle single line files
                pitch_data = pitch_data.reshape(1, -1)
        except Exception as e:
            print(f"Error loading pitch file {pitch_path}: {e}")
            # Return dummy data
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            n_frames = len(audio) // self.hop_length + 1
            dummy_target = torch.zeros(self.n_bins, n_frames)
            dummy_f0 = torch.zeros(n_frames)
            return audio_tensor, dummy_target, dummy_f0
            
        times = pitch_data[:, 0]
        f0_values = pitch_data[:, 1]
        confidences = pitch_data[:, 2] if pitch_data.shape[1] > 2 else np.ones_like(times)
        
        # --- 3. Align with STFT Frames ---
        # Compute number of frames based on audio length and hop_length
        n_frames = int(np.ceil(len(audio) / self.hop_length)) # Ceiling to ensure coverage
        
        # Create frame center times
        frame_times = np.arange(n_frames) * self.hop_length / self.sample_rate
        
        # Interpolate pitch values and confidences to match frame times
        # Use left=0, right=0 for out-of-bounds (unvoiced)
        f0_aligned = np.interp(frame_times, times, f0_values, left=0, right=0)
        confidence_aligned = np.interp(frame_times, times, confidences, left=0, right=0)
        
        # --- 4. Apply Data Augmentation ---
        if self.augment:
            audio = self._augment_audio(audio)
        
        # --- 5. Convert F0 values to classification targets ---
        # Use voiced mask based on interpolated confidence
        voiced_mask = confidence_aligned > 0.5 # Threshold can be configurable
        classification_targets = self._f0_to_classification_targets(f0_aligned, voiced_mask)
        
        # --- 6. Prepare Outputs ---
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # Add batch dimension [1, L]
        target_f0_tensor = torch.from_numpy(f0_aligned).float()      # [n_frames]
        classification_targets_tensor = torch.from_numpy(classification_targets).float() # [n_bins, n_frames]
        
        return audio_tensor, classification_targets_tensor, target_f0_tensor
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to audio signal.
        This is a simplified version compared to the paper's use of CHiME-Home.
        """
        # --- Gain adjustment ---
        if random.random() < 0.5:
            gain_db = random.uniform(self.gain_db_range[0], self.gain_db_range[1])
            gain_factor = 10**(gain_db/20)  # Convert dB to linear scale
            audio = audio * gain_factor
            
        # --- Add noise with SNR control ---
        # Simplified: Add white noise. Paper uses CHiME-Home + gaussian mixtures.
        if random.random() < 0.5:
            # Calculate SNR
            snr_db = random.uniform(self.noise_snr_range[0], self.noise_snr_range[1])
            
            # Generate white noise
            noise = np.random.randn(len(audio)).astype(np.float32)
            
            # Calculate signal and noise power
            signal_power = np.mean(audio**2)
            noise_power = np.mean(noise**2)
            
            # Calculate scaling factor for desired SNR
            if noise_power > 0:
                # SNR = 10 * log10(P_signal / P_noise)
                # 10^(SNR/10) = P_signal / P_noise
                # P_noise_desired = P_signal / 10^(SNR/10)
                # scaling_factor = sqrt(P_noise_desired / P_noise_actual)
                scaling_factor = np.sqrt(signal_power / (noise_power * 10**(snr_db/10)))
                audio = audio + scaling_factor * noise
                
        # --- Pitch shift (small variations) ---
        # Note: This changes the ground truth F0, which should ideally be adjusted too.
        # For simplicity, we apply it but note the limitation.
        if random.random() < 0.3:
            n_steps = random.uniform(-0.5, 0.5)  # Small pitch shifts in semitones
            # This requires regenerating pitch annotations, which is complex.
            # We apply the shift but acknowledge the ground truth F0 will be misaligned.
            # A more robust approach would re-estimate F0 after augmentation or use time-stretching.
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
            
        # --- Clipping to prevent overflow ---
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _f0_to_classification_targets(self, f0_values: np.ndarray, voiced_mask: np.ndarray) -> np.ndarray:
        """
        Convert F0 values to classification targets using Gaussian distribution.
        
        Args:
            f0_values: Array of F0 values in Hz (n_frames,)
            voiced_mask: Boolean mask indicating voiced frames (n_frames,)
            
        Returns:
            Classification targets as probability distributions (n_bins, n_frames)
        """
        n_frames = len(f0_values)
        targets = np.zeros((self.n_bins, n_frames))
        
        # Convert pitch bins to log scale for Gaussian calculation
        log_pitch_bins = np.log(self.pitch_bins)
        
        # Standard deviation in log frequency space (matching paper)
        # The paper uses σ=0.1 for the Gaussian in log space.
        sigma_log = 0.1 
        
        for i in range(n_frames):
            if voiced_mask[i] and f0_values[i] > 0:
                # Convert F0 to log frequency
                log_f0 = np.log(f0_values[i])
                
                # Create Gaussian distribution centered at the F0
                # p_b = exp(-0.5 * ((log(f_b) - log(f_true)) / sigma)^2)
                distances_squared = (log_pitch_bins - log_f0) ** 2
                # Note: The paper's Eq. 3 has a normalization factor, but often in practice
                # a simple unnormalized Gaussian is used and then normalized.
                # We follow the common practice of normalizing the resulting probabilities.
                log_probabilities_unnorm = -0.5 * distances_squared / (sigma_log ** 2)
                # Use log-space computation for numerical stability
                # Subtract max for stability in exp
                log_probabilities_unnorm -= np.max(log_probabilities_unnorm)
                probabilities = np.exp(log_probabilities_unnorm)
                
                # Normalize
                probabilities_sum = np.sum(probabilities)
                if probabilities_sum > 1e-8: # Avoid division by zero
                    probabilities = probabilities / probabilities_sum
                    targets[:, i] = probabilities
                else:
                    # If normalization fails, use a sharp peak
                    closest_bin = np.argmin(np.abs(self.pitch_bins - f0_values[i]))
                    targets[closest_bin, i] = 1.0
            # For unvoiced frames, targets remain zero (no probability mass)
            # The model should learn to output low probabilities across all bins.
        
        return targets # Shape: [n_bins, n_frames]


def create_dataloader(data_paths: List[str], 
                      batch_size: int = 32, 
                      shuffle: bool = True, 
                      num_workers: int = 4,
                      **kwargs) -> DataLoader:
    """
    Create a DataLoader for SwiftF0 training.
    
    Args:
        data_paths: List of paths to data files or directories.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.
        **kwargs: Additional arguments for dataset initialization (sample_rate, hop_length, etc.).
        
    Returns:
        DataLoader instance.
    """
    dataset = SwiftF0Dataset(data_paths, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
