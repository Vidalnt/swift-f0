# src/loss.py
import torch
import torch.nn.functional as F

def compute_swiftf0_loss(model: torch.nn.Module, 
                         pitch_logits: torch.Tensor, 
                         target_indices: torch.Tensor,
                         target_f0: torch.Tensor,
                         classification_weight: float = 1.0,
                         regression_weight: float = 1.0) -> tuple:
    """
    Compute the combined SwiftF0 loss using the model's buffers for accuracy.
    This matches the formulation in the SwiftF0 paper.
    
    Args:
        model: The SwiftF0Model instance (to access pitch_bin_centers).
        pitch_logits: Output logits from model (batch_size, n_bins, n_frames).
        target_indices: Target class indices for cross-entropy (batch_size, n_frames).
        target_f0: Target F0 values in Hz (batch_size, n_frames).
        classification_weight: Weight for classification loss (lambda in paper, default 1.0).
        regression_weight: Weight for regression loss (lambda in paper, default 1.0).
        
    Returns:
        tuple: (total_loss, classification_loss, regression_loss)
    """
    batch_size, n_bins, n_frames = pitch_logits.shape
    
    # --- Classification Loss (LCE) ---
    # Reshape for cross_entropy: (N, C) and (N,)
    # PyTorch cross_entropy expects (N, C, ...) for input and (N, ...) for targets
    logits_reshaped = pitch_logits.permute(0, 2, 1).reshape(-1, n_bins)  # (batch*n_frames, n_bins)
    target_indices_reshaped = target_indices.reshape(-1)  # (batch*n_frames,)
    
    # CrossEntropyLoss automatically ignores targets with value -100
    classification_loss = F.cross_entropy(logits_reshaped, target_indices_reshaped, ignore_index=-100)
    
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
    
    # --- MODIFICATION: Apply regression loss only on voiced frames ---
    # According to the SwiftF0 paper, Lcents is calculated on all frames,
    # but applying it only on voiced frames avoids affecting predictions on unvoiced frames.
    # Create a mask for voiced frames in the target
    voiced_mask = (target_f0 > 0) # Boolean mask [B, n_frames]
    
    if voiced_mask.any():
        # Calculate L1 loss only on elements where voiced_mask is True
        regression_loss = F.l1_loss(
            predicted_log_f0[voiced_mask], 
            log_target_f0[voiced_mask]
        )
    else:
        # If there are no voiced frames in the batch, the regression loss is zero
        # Make sure it's a tensor on the same device
        regression_loss = torch.tensor(0.0, device=pitch_logits.device)
    
    # --- Combined Loss (Ltotal = LCE + λ*Lcents, with λ=1 by default) ---
    total_loss = classification_weight * classification_loss + regression_weight * regression_loss
    
    return total_loss, classification_loss, regression_loss
