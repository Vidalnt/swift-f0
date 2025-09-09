import torch
import numpy as np
import os
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def find_audio_files(data_dir: str, extensions: List[str] = ['.wav']) -> List[str]:
    """
    Recursively find all audio files in a directory.
    
    Args:
        data_dir: The root directory to search.
        extensions: List of file extensions to include.
        
    Returns:
        A list of file paths.
    """
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

def split_data(data_files: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a list of data files into train, validation, and test sets.
    
    Args:
        data_files: List of data file paths.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        
    Returns:
        Tuple of (train_files, val_files, test_files).
    """
    import random
    # Shuffle the list
    random.shuffle(data_files)
    
    n_total = len(data_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    train_files = data_files[:n_train]
    val_files = data_files[n_train:n_train + n_val]
    test_files = data_files[n_train + n_val:]
    
    return train_files, val_files, test_files

def plot_pitch_contour(f0_hz: np.ndarray, confidence: np.ndarray, hop_length: int, sample_rate: int, 
                       title: str = "Predicted Pitch Contour"):
    """
    Plot the fundamental frequency (F0) contour over time.
    
    Args:
        f0_hz: Array of F0 values in Hz.
        confidence: Array of confidence values.
        hop_length: Hop length used in analysis.
        sample_rate: Sample rate of the audio.
        title: Title for the plot.
    """
    times = np.arange(len(f0_hz)) * hop_length / sample_rate
    
    plt.figure(figsize=(12, 4))
    
    # Mask unvoiced regions (e.g., where confidence is low)
    voiced_mask = confidence > 0.5 # Use a threshold
    f0_voiced = np.where(voiced_mask, f0_hz, np.nan)
    
    plt.plot(times, f0_voiced, label='Voiced F0')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, loss: float, filepath: str):
    """
    Save model checkpoint.
    
    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        epoch: Current epoch number.
        loss: Current loss value.
        filepath: Path to save the checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str):
    """
    Load model checkpoint.
    
    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        filepath: Path to the checkpoint file.
        
    Returns:
        epoch, loss from the checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {filepath} (Epoch {epoch}, Loss {loss:.4f})")
    return epoch, loss

# Note: ONNX export logic is in export.py
# This file can contain other utility functions for training/inference analysis.
