import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SwiftF0Model(nn.Module):
    """
    SwiftF0 model architecture based on the paper.
    
    Key features:
    - STFT-based processing with specific parameters
    - Compact CNN architecture for efficiency
    - Joint classification and regression training
    """
    
    def __init__(self, n_bins=360, f_min=46.875, f_max=2093.75, sample_rate=16000):
        super(SwiftF0Model, self).__init__()
        
        self.n_bins = n_bins
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = sample_rate
        
        # Create frequency bins (log-spaced from f_min to f_max)
        self.register_buffer('freq_bins', torch.tensor(
            f_min * (f_max / f_min) ** (torch.arange(n_bins).float() / (n_bins - 1))
        ))
        
        # Network architecture based on paper description
        # Input: STFT magnitude spectrogram
        # Output: pitch probabilities + regression targets
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Output heads
        # Classification head (pitch bins)
        self.classification_head = nn.Linear(256, n_bins)
        
        # Regression head (continuous pitch)
        self.regression_head = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, 1, n_frames, n_freq_bins)
            
        Returns:
            tuple: (classification_logits, regression_output)
                - classification_logits: (batch_size, n_bins)
                - regression_output: (batch_size, 1)
        """
        # Feature extraction
        features = self.conv_layers(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc_layers(features)
        
        # Output heads
        classification_logits = self.classification_head(features)
        regression_output = self.regression_head(features)
        
        return classification_logits, regression_output
    
    def get_pitch_from_logits(self, logits):
        """
        Convert classification logits to pitch estimates using local expected value method.
        
        Args:
            logits: Classification logits of shape (batch_size, n_bins)
            
        Returns:
            pitch_hz: Pitch estimates in Hz (batch_size,)
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute expected frequency
        log_freq_bins = torch.log(self.freq_bins)
        expected_log_freq = torch.sum(probs * log_freq_bins, dim=-1)
        pitch_hz = torch.exp(expected_log_freq)
        
        return pitch_hz

# Function to create model instance
def create_model(**kwargs):
    return SwiftF0Model(**kwargs)