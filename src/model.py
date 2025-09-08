import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SwiftF0Model(nn.Module):
    """
    SwiftF0 model architecture based on the ONNX inspection and paper.
    
    Key features:
    - STFT-based processing with specific parameters (N=1024, H=256, fs=16kHz)
    - Spectral slicing to retain only relevant frequency bins (K_min=3, K_max=134)
    - Log compression of spectrogram
    - Compact CNN architecture with 6 conv layers (5x5 kernels, SAME padding)
    - Batch normalization after each conv layer
    - Configurable number of pitch bins
    - Joint classification and regression training
    """
    
    def __init__(self, 
                 n_bins: int = 200, 
                 f_min: float = 46.875, 
                 f_max: float = 2093.75, 
                 sample_rate: int = 16000,
                 hop_length: int = 256,
                 n_fft: int = 1024,
                 k_min: int = 3,
                 k_max: int = 134):
        """
        Initialize the SwiftF0 model.
        
        Args:
            n_bins: Number of pitch bins (default: 200)
            f_min: Minimum frequency in Hz (default: 46.875)
            f_max: Maximum frequency in Hz (default: 2093.75)
            sample_rate: Audio sample rate (default: 16000)
            hop_length: STFT hop length (default: 256)
            n_fft: STFT window size (default: 1024)
            k_min: Minimum frequency bin index to retain (default: 3)
            k_max: Maximum frequency bin index to retain (default: 134)
        """
        super(SwiftF0Model, self).__init__()
        
        self.n_bins = n_bins
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.k_min = k_min
        self.k_max = k_max
        
        # Create log-spaced frequency bins (matching the paper)
        self.register_buffer('pitch_bin_centers', torch.tensor(
            self._create_log_spaced_bins(n_bins, f_min, f_max)
        ))
        
        # Network architecture based on ONNX inspection
        # 6 conv layers with 5x5 kernels, SAME padding, BatchNorm, ReLU activation
        self.conv_layers = nn.Sequential(
            # Layer 1: 1 -> 8 channels
            nn.Conv2d(1, 8, kernel_size=5, padding=2),  # SAME padding
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Layer 2: 8 -> 16 channels
            nn.Conv2d(8, 16, kernel_size=5, padding=2),  # SAME padding
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Layer 3: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # SAME padding
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 4: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # SAME padding
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 5: 64 -> 1 channel
            nn.Conv2d(64, 1, kernel_size=5, padding=2),  # SAME padding
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        
        # Final projection layer: k_max-k_min -> n_bins channels, 1x1 kernel
        # According to ONNX inspection, freq_projection.weight has shape (200, 132, 1)
        # This means in_channels=132 (k_max-k_min=134-3=131, but ONNX shows 132)
        # We'll use k_max-k_min = 131 as the input channels
        self.freq_projection = nn.Conv1d(k_max - k_min, n_bins, kernel_size=1)
        
    def _create_log_spaced_bins(self, n_bins: int, f_min: float, f_max: float) -> np.ndarray:
        """Create log-spaced frequency bins matching the paper."""
        # Log-spaced from f_min to f_max
        return f_min * (f_max / f_min) ** (np.arange(n_bins) / (n_bins - 1))
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, audio_length)
                This is the raw audio input, STFT computed internally
            
        Returns:
            tuple: (pitch_logits, confidence)
                - pitch_logits: (batch_size, n_bins, n_frames)
                - confidence: (batch_size, n_frames)
        """
        # Compute STFT internally (matching paper parameters)
        # x is raw audio [batch, samples]
        stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=x.device),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Get magnitude spectrogram
        mag_spec = torch.abs(stft)  # [batch, freq_bins, time_frames]
        # For n_fft=1024, freq_bins = 513
        
        # Add channel dimension
        mag_spec = mag_spec.unsqueeze(1)  # [batch, 1, freq_bins, time_frames]
        
        # Slice the spectrogram to retain only relevant frequency bins
        # According to paper: K_min=3, K_max=134 (131 bins retained)
        mag_spec = mag_spec[:, :, self.k_min:self.k_max, :]  # [batch, 1, 131, time_frames]
        
        # Apply log compression (with small epsilon to avoid log(0))
        epsilon = 1e-8
        mag_spec = torch.log(mag_spec + epsilon)  # [batch, 1, 131, time_frames]
        
        # Apply convolutional layers
        features = self.conv_layers(mag_spec)  # [batch, 1, 131, time_frames]
        
        # Squeeze channel dimension and rearrange dimensions
        features = features.squeeze(1)  # [batch, 131, time_frames]
        
        # Apply frequency projection
        logits = self.freq_projection(features)  # [batch, n_bins, time_frames]
        
        # Apply softmax to get probability distribution
        probs = F.softmax(logits, dim=1)
        
        # Compute confidence as sum of probabilities (should be 1, but can be less if model is uncertain)
        confidence = torch.sum(probs, dim=1)  # [batch, n_frames]
        
        return logits, confidence
    
    def decode_pitch(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decode pitch from logits using local expected value method.
        
        Args:
            logits: Logits from the model (batch_size, n_bins, n_frames)
            
        Returns:
            pitch_hz: Pitch estimates in Hz (batch_size, n_frames)
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Compute expected frequency using pitch_bin_centers
        # Expand pitch_bin_centers to match batch and time dimensions
        batch_size, _, n_frames = logits.shape
        pitch_bins_expanded = self.pitch_bin_centers.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, n_frames)
        
        # Compute weighted sum
        expected_pitch = torch.sum(probs * pitch_bins_expanded, dim=1)  # [batch, n_frames]
        
        return expected_pitch

# Function to create model instance
def create_model(**kwargs) -> SwiftF0Model:
    """
    Create a SwiftF0 model instance.
    
    Args:
        **kwargs: Model parameters (n_bins, f_min, f_max, sample_rate, hop_length, n_fft, k_min, k_max)
        
    Returns:
        SwiftF0Model instance
    """
    return SwiftF0Model(**kwargs)