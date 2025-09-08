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
            device: Device to create pitch bins on (not used here as pitch_bin_centers is in the model)
            
        Returns:
            Decoded log F0 values (batch_size, n_frames)
        """
        # This function is kept for conceptual clarity in the loss, 
        # but in practice, the model's pitch_bin_centers buffer should be used.
        # The actual implementation for loss calculation would ideally get the model instance
        # or the pitch_bin_centers buffer passed to it.
        # For now, we assume the model's decode method or the dataset handles this correctly.
        # A more robust approach would be to pass the pitch_bin_centers to the loss function.
        # However, to keep the interface simple, we'll note this dependency.
        
        # This is a placeholder to indicate the dependency. 
        # The correct way is to compute E[log(f)] using the model's self.pitch_bin_centers
        # outside this function or by passing it as an argument.
        # For the purpose of this code structure, we'll raise a note.
        raise NotImplementedError("This method should ideally use the model's pitch_bin_centers. "
                                  "Consider computing E[log(f)] directly in the training loop "
                                  "using the model's buffer for accuracy.")

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

# --- Recommended approach for computing regression loss ---
# In your training loop, after getting logits from the model:
# 1. Get probs: probs = F.softmax(logits, dim=1)
# 2. Get log pitch bins from model buffer: log_pitch_bins = torch.log(model.pitch_bin_centers)
# 3. Expand: log_pitch_bins_expanded = log_pitch_bins.unsqueeze(0).unsqueeze(-1).expand_as(probs)
# 4. Compute E[log(f)]: predicted_log_f0 = torch.sum(probs * log_pitch_bins_expanded, dim=1)
# 5. Compute regression loss: regression_loss = F.l1_loss(predicted_log_f0, log_target_f0)
# This avoids the complexity and potential mismatch in the loss class itself.

# Simplified loss function for direct use in training loop context
def compute_swiftf0_loss(model: torch.nn.Module, 
                         pitch_logits: torch.Tensor, 
                         classification_targets: torch.Tensor,
                         target_f0: torch.Tensor,
                         classification_weight: float = 1.0,
                         regression_weight: float = 1.0) -> tuple:
    """
    Compute the combined SwiftF0 loss using the model's buffers for accuracy.
    This is the recommended way to compute the loss during training.
    
    Args:
        model: The SwiftF0Model instance (to access pitch_bin_centers).
        pitch_logits: Output logits from model (batch_size, n_bins, n_frames).
        classification_targets: Target probability distributions (batch_size, n_bins, n_frames).
        target_f0: Target F0 values in Hz (batch_size, n_frames).
        classification_weight: Weight for classification loss.
        regression_weight: Weight for regression loss.
        
    Returns:
        tuple: (total_loss, classification_loss, regression_loss)
    """
    batch_size, n_bins, n_frames = pitch_logits.shape
    
    # --- Classification Loss ---
    logits_reshaped = pitch_logits.permute(0, 2, 1).reshape(-1, n_bins)
    targets_reshaped = classification_targets.permute(0, 2, 1).reshape(-1, n_bins)
    hard_targets = torch.argmax(targets_reshaped, dim=1)
    classification_loss = F.cross_entropy(logits_reshaped, hard_targets, reduction='mean')
    
    # --- Regression Loss (Lcents) ---
    # Apply softmax to get probabilities
    probs = F.softmax(pitch_logits, dim=1) # Shape: [B, n_bins, n_frames]
    
    # Get log pitch bins from model buffer
    # model.pitch_bin_centers: [n_bins]
    log_pitch_bins = torch.log(model.pitch_bin_centers + 1e-8) # [n_bins]
    
    # Expand to match probability tensor dimensions
    # [n_bins] -> [1, n_bins, 1] -> [B, n_bins, n_frames]
    log_pitch_bins_expanded = log_pitch_bins.unsqueeze(0).unsqueeze(-1).expand_as(probs)
    
    # Compute expected value in log space: E[log(f)] = sum(p * log(f_b))
    # Sum over the n_bins dimension (dim=1)
    predicted_log_f0 = torch.sum(probs * log_pitch_bins_expanded, dim=1) # [B, n_frames]
    
    # Target log frequencies
    epsilon = 1e-8
    log_target_f0 = torch.log(target_f0 + epsilon) # [B, n_frames]
    
    # L1 loss on log frequencies
    regression_loss = F.l1_loss(predicted_log_f0, log_target_f0)
    
    # --- Combined Loss ---
    total_loss = classification_weight * classification_loss + regression_weight * regression_loss
    
    return total_loss, classification_loss, regression_loss
