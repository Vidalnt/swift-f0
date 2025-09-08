import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SwiftF0Model(nn.Module):
    """
    SwiftF0 model architecture based on the ONNX inspection and paper.
    
    Key features:
    - STFT-based processing with configurable parameters
    - Compact CNN architecture with 6 conv layers (5x5 kernels, SAME padding)
    - Configurable number of pitch bins
    - Joint classification and regression training
    """
    
    def __init__(self, 
                 n_bins: int = 200, 
                 f_min: float = 46.875, 
                 f_max: float = 2093.75, 
                 sample_rate: int = 16000,
                 hop_length: int = 256,
                 n_fft: int = 1024):
        """
        Initialize the SwiftF0 model.
        
        Args:
            n_bins: Number of pitch bins (default: 200)
            f_min: Minimum frequency in Hz (default: 46.875)
            f_max: Maximum frequency in Hz (default: 2093.75)
            sample_rate: Audio sample rate (default: 16000)
            hop_length: STFT hop length (default: 256)
            n_fft: STFT window size (default: 1024)
        """
        super(SwiftF0Model, self).__init__()
        
        self.n_bins = n_bins
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Create log-spaced frequency bins (matching the paper)
        self.register_buffer('pitch_bin_centers', torch.tensor(
            self._create_log_spaced_bins(n_bins, f_min, f_max)
        ))
        
        # Network architecture based on ONNX inspection
        # 6 conv layers with 5x5 kernels, SAME padding, ReLU activation
        self.conv_layers = nn.Sequential(
            # Layer 1: 1 -> 8 channels
            nn.Conv2d(1, 8, kernel_size=5, padding=2),  # SAME padding
            nn.ReLU(),
            
            # Layer 2: 8 -> 16 channels
            nn.Conv2d(8, 16, kernel_size=5, padding=2),  # SAME padding
            nn.ReLU(),
            
            # Layer 3: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # SAME padding
            nn.ReLU(),
            
            # Layer 4: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # SAME padding
            nn.ReLU(),
            
            # Layer 5: 64 -> 1 channel
            nn.Conv2d(64, 1, kernel_size=5, padding=2),  # SAME padding
            nn.ReLU(),
        )
        
        # Final projection layer: 1 -> n_bins channels, 1x1 kernel
        self.freq_projection = nn.Conv1d(1, n_bins, kernel_size=1)
        
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
        
        # Add channel dimension
        mag_spec = mag_spec.unsqueeze(1)  # [batch, 1, freq_bins, time_frames]
        
        # Apply convolutional layers
        features = self.conv_layers(mag_spec)  # [batch, 1, freq_bins, time_frames]
        
        # Squeeze channel dimension and rearrange dimensions
        features = features.squeeze(1)  # [batch, freq_bins, time_frames]
        
        # Apply frequency projection
        logits = self.freq_projection(features)  # [batch, n_bins, time_frames]
        
        # Apply softmax to get probability distribution
        probs = F.softmax(logits, dim=1)
        
        # Compute confidence as maximum probability (following ONNX model pattern)
        confidence = torch.max(probs, dim=1)[0]  # [batch, time_frames]
        
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
        **kwargs: Model parameters (n_bins, f_min, f_max, sample_rate, hop_length, n_fft)
        
    Returns:
        SwiftF0Model instance
    """
    return SwiftF0Model(**kwargs)