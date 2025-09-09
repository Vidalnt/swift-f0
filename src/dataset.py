import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
from typing import Tuple, Optional, List
import random
from torch.nn.utils.rnn import pad_sequence

class SwiftF0Dataset(Dataset):
    """
    Dataset class for SwiftF0 training.
    
    Handles loading of audio files and their corresponding pitch annotations,
    with data augmentation and preprocessing matching the paper.
    """
    
    def __init__(self, 
                 data_paths: List[str],
                 sample_rate: int = 16000,
                 hop_length: int = 256,
                 n_fft: int = 1024,
                 n_bins: int = 200,
                 f_min: float = 46.875, 
                 f_max: float = 2093.75,
                 augment: bool = True,
                 noise_snr_range: Tuple[float, float] = (10, 30),
                 gain_db_range: Tuple[float, float] = (-6, 6)):
        """
        Initialize the dataset.
        
        Assumes data is organized with audio files (.wav) and corresponding 
        pitch annotation files (.f0, .pv, .csv) in the same directory.
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
        
        self.pitch_bins = self._create_log_spaced_bins()
        
        self.audio_files = []
        self.pitch_files = []
        
        for path in data_paths:
            if os.path.isfile(path) and path.lower().endswith('.wav'):
                pitch_path = path.rsplit('.', 1)[0] + '.f0'
                if os.path.exists(pitch_path):
                    self.audio_files.append(path)
                    self.pitch_files.append(pitch_path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith('.wav'):
                            audio_path = os.path.join(root, file)
                            pitch_path = audio_path.rsplit('.', 1)[0] + '.f0'
                            if os.path.exists(pitch_path):
                                self.audio_files.append(audio_path)
                                self.pitch_files.append(pitch_path)
                                
        if not self.audio_files:
            raise ValueError("No valid audio-pitch file pairs found in the provided paths.")
            
        print(f"Found {len(self.audio_files)} audio-pitch pairs.")
        
    def _create_log_spaced_bins(self) -> np.ndarray:
        return self.f_min * (self.f_max / self.f_min) ** (np.arange(self.n_bins) / (self.n_bins - 1))
        
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        MODIFIED: Returns (audio, target_indices, target_f0)
        - audio: Raw audio tensor (1, audio_length)
        - target_indices: Target class indices for cross-entropy (n_frames,)
        - target_f0: Target F0 values in Hz for regression loss (n_frames,)
        """
        audio_path = self.audio_files[idx]
        pitch_path = self.pitch_files[idx]
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            pitch_data = np.loadtxt(pitch_path)
            if pitch_data.ndim == 1:
                pitch_data = pitch_data.reshape(1, -1)
        except Exception as e:
            print(f"Error loading {audio_path} or {pitch_path}: {e}. Skipping.")
            # Return a valid-shaped dummy item to avoid crashing the DataLoader
            dummy_audio = torch.zeros(1, self.sample_rate) # 1 sec of audio
            n_frames = self.sample_rate // self.hop_length
            dummy_indices = torch.full((n_frames,), -100, dtype=torch.long)
            dummy_f0 = torch.zeros(n_frames)
            return dummy_audio, dummy_indices, dummy_f0
            
        times = pitch_data[:, 0]
        f0_values = pitch_data[:, 1]
        
        audio_tensor_for_stft = torch.from_numpy(audio).float()
        dummy_stft = torch.stft(
            audio_tensor_for_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        n_frames = dummy_stft.size(1)

        frame_times = np.arange(n_frames) * self.hop_length / self.sample_rate
        f0_aligned = np.interp(frame_times, times, f0_values, left=0, right=0)
        
        if self.augment:
            audio = self._augment_audio(audio)
            
        # MODIFIED: Generate target indices for cross-entropy
        voiced_mask = f0_aligned > 0
        target_indices = self._f0_to_target_indices(f0_aligned, voiced_mask)
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        target_f0_tensor = torch.from_numpy(f0_aligned).float()
        # MODIFIED: Target tensor must be of type Long
        target_indices_tensor = torch.from_numpy(target_indices).long()
        
        return audio_tensor, target_indices_tensor, target_f0_tensor
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply data augmentation: gain and noise."""
        if random.random() < 0.75:
            gain_db = random.uniform(self.gain_db_range[0], self.gain_db_range[1])
            audio = audio * (10**(gain_db/20))
            
        if random.random() < 0.75:
            snr_db = random.uniform(self.noise_snr_range[0], self.noise_snr_range[1])
            noise = np.random.randn(len(audio)).astype(np.float32)
            
            signal_power = np.mean(audio**2)
            if signal_power > 1e-8: # Avoid division by zero for silence
                noise_power = np.mean(noise**2)
                scaling_factor = np.sqrt(signal_power / (noise_power * 10**(snr_db/10)))
                audio = audio + scaling_factor * noise
                
        return np.clip(audio, -1.0, 1.0)
    
    # NEW: Modified function to generate one-hot indices
    def _f0_to_target_indices(self, f0_values: np.ndarray, voiced_mask: np.ndarray) -> np.ndarray:
        """
        Convert F0 values to target bin indices for cross-entropy loss.
        
        Args:
            f0_values: Array of F0 values in Hz (n_frames,).
            voiced_mask: Boolean mask indicating voiced frames (n_frames,).
            
        Returns:
            Target indices (n_frames,). Unvoiced frames are assigned -100, which is
            the default ignore_index for PyTorch's CrossEntropyLoss.
        """
        n_frames = len(f0_values)
        target_indices = np.full(n_frames, -100, dtype=np.int64) # Default to ignore_index
        
        # For each voiced frame, find the closest pitch bin
        voiced_f0s = f0_values[voiced_mask]
        if len(voiced_f0s) > 0:
            # Vectorized search for the closest bin for all voiced frames
            closest_bin_indices = np.argmin(np.abs(self.pitch_bins[:, np.newaxis] - voiced_f0s), axis=0)
            target_indices[voiced_mask] = closest_bin_indices
            
        return target_indices

def swiftf0_collate_fn(batch):
    """
    Collation function for SwiftF0Dataset that handles variable-length sequences
    by padding them to the length of the longest sequence in the batch.
    Args:
        batch: List of tuples (audio, target_indices, target_f0) from the dataset.
    Returns:
        Tuple of batched and padded tensors: (audio_batch, target_indices_batch, target_f0_batch)
    """
    audios, target_indices, target_f0s = zip(*batch) # Unpack the tuple

    # Convert to list if necessary (although zip already creates a kind of list)
    audios = list(audios)
    target_indices = list(target_indices)
    target_f0s = list(target_f0s)

    # --- 1. Padding for audios ---
    # audios is a list of tensors of shape [1, L_i]
    # Remove channel dimension: [1, L_i] -> [L_i]
    audios_1d = [audio.squeeze(0) for audio in audios] # List of [L_i]
    # Pad to the maximum length in the batch
    padded_audios_1d = pad_sequence(audios_1d, batch_first=True, padding_value=0.0) # [B, L_max]
    # Add channel dimension again: [B, L_max] -> [B, 1, L_max]
    audio_batch = padded_audios_1d.unsqueeze(1) # [B, 1, L_max]

    # --- 2. Padding for target_indices ---
    # target_indices is a list of tensors of shape [T_i]
    target_indices_batch = pad_sequence(target_indices, batch_first=True, padding_value=-100) # [B, T_max]

    # --- 3. Padding for target_f0 ---
    # target_f0s is a list of tensors of shape [T_i]
    target_f0_batch = pad_sequence(target_f0s, batch_first=True, padding_value=0.0) # [B, T_max]

    return audio_batch, target_indices_batch, target_f0_batch

def create_dataloader(data_paths: List[str], 
                      batch_size: int = 32, 
                      shuffle: bool = True, 
                      num_workers: int = 2, # Reduced to avoid warnings
                      **kwargs) -> DataLoader:
    """Create a DataLoader for SwiftF0 training."""
    dataset = SwiftF0Dataset(data_paths, **kwargs)
    # Pass the custom collate function
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=swiftf0_collate_fn, # <<< Use the custom collate_fn here
        pin_memory=True # Optional, can help with GPU performance
    )
