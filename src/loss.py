import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiftF0Loss(nn.Module):
    """
    Combined loss function for SwiftF0 training, matching the paper formulation.
    
    Combines classification loss (cross-entropy) and regression loss (L1 on log frequencies).
    Based on Eq. 4 and Eq. 5 from the SwiftF0 paper:
    LCE = -1/T * ΣΣ y_m,b * log(p̂_m,b)  (Categorical cross-entropy)
    Lcents = 1/T * Σ |f̂_log[m] - log(f_true[m])|  (L1 loss in log-frequency space)
    Ltotal = LCE + λ*Lcents  (with λ=1)
    """
    
    def __init__(self, classification_weight: float = 1.0, regression_weight: float = 1.0):
        """
        Initialize the loss function.
        
        Args:
            classification_weight: Weight for classification loss (λ in paper, default 1.0)
            regression_weight: Weight for regression loss (default 1.0, but paper uses λ=1 for both)
        """
        super(SwiftF0Loss, self).__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        
    def forward(self, 
                pitch_logits: torch.Tensor, 
                classification_targets: torch.Tensor,
                target_f0: torch.Tensor) -> tuple:
        """
        Compute the combined loss.
        
        Args:
            pitch_logits: Output logits from model (batch_size, n_bins, n_frames)
            classification_targets: Target probability distributions (batch_size, n_bins, n_frames)
            target_f0: Target F0 values in Hz (batch_size, n_frames)
            
        Returns:
            tuple: (total_loss, classification_loss, regression_loss)
        """
        # Classification loss (Cross-entropy with soft targets)
        batch_size, n_bins, n_frames = pitch_logits.shape
        
        # Reshape for cross_entropy: (N, C, d1, d2, ...) where C is number of classes
        # PyTorch cross_entropy expects (N, C, ...) for input and (N, ...) for targets
        logits_reshaped = pitch_logits.permute(0, 2, 1).reshape(-1, n_bins)  # (batch*n_frames, n_bins)
        targets_reshaped = classification_targets.permute(0, 2, 1).reshape(-1, n_bins)  # (batch*n_frames, n_bins)
        
        # Convert soft targets to hard targets for cross-entropy
        # Find the index of the maximum probability in each target distribution
        hard_targets = torch.argmax(targets_reshaped, dim=1)
        classification_loss = F.cross_entropy(logits_reshaped, hard_targets, reduction='mean')
        
        # Regression loss (L1 on log frequencies, matching paper Eq. 5)
        # First decode predicted pitch from logits using the same method as in model.decode_pitch
        # But we need to compute the expected value in log space as per the paper
        predicted_log_f0 = self._decode_log_f0_from_logits(pitch_logits, targets_reshaped.device)  # (batch_size, n_frames)
        
        # Compute log frequencies (add small epsilon to prevent log(0))
        epsilon = 1e-8
        log_target_f0 = torch.log(target_f0 + epsilon)
        
        # L1 loss on log frequencies (cents error up to constant factor)
        regression_loss = F.l1_loss(predicted_log_f0, log_target_f0)
        
        # Combined loss (Eq. 6 in paper)
        total_loss = (self.classification_weight * classification_loss + 
                     self.regression_weight * regression_loss)
        
        return total_loss, classification_loss, regression_loss
    
    def _decode_log_f0_from_logits(self, logits: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Internal helper to decode log F0 from logits using expected value method in log space.
        This matches the approach described in the paper for computing f̂_log[m].
        
        Args:
            logits: Logits from the model (batch_size, n_bins, n_frames)
            device: Device to create pitch bins on
            
        Returns:
            Decoded log F0 values (batch_size, n_frames)
        """
        batch_size, n_bins, n_frames = logits.shape
        
        # Create pitch bins (this should match the model's pitch_bin_centers)
        # These are hardcoded to match SwiftF0 paper parameters
        f_min, f_max = 46.875, 2093.75
        pitch_bins = f_min * (f_max / f_min) ** (torch.arange(n_bins, device=device) / (n_bins - 1))
        
        # Convert to log space
        log_pitch_bins = torch.log(pitch_bins)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Expand log_pitch_bins to match batch and time dimensions
        log_pitch_bins_expanded = log_pitch_bins.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, n_frames)
        
        # Compute weighted sum (expected value in log frequency domain)
        expected_log_pitch = torch.sum(probs * log_pitch_bins_expanded, dim=1)  # [batch, n_frames]
        
        return expected_log_pitch

# Function to create loss instance
def create_loss(**kwargs) -> SwiftF0Loss:
    """
    Create a SwiftF0 loss function instance.
    
    Args:
        **kwargs: Loss parameters (classification_weight, regression_weight)
        
    Returns:
        SwiftF0Loss instance
    """
    return SwiftF0Loss(**kwargs)