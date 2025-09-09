# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_swiftf0_loss(model: torch.nn.Module, 
                         pitch_logits: torch.Tensor, 
                         target_indices: torch.Tensor,
                         target_f0: torch.Tensor,
                         classification_weight: float = 1.0,
                         regression_weight: float = 1.0) -> tuple:
    """
    Compute the combined SwiftF0 loss using the model's buffers for accuracy.
    This is the recommended way to compute the loss during training.
    
    Args:
        model: The SwiftF0Model instance (to access pitch_bin_centers).
        pitch_logits: Output logits from model (batch_size, n_bins, n_frames).
        target_indices: Target class indices for cross-entropy (batch_size, n_frames).
        target_f0: Target F0 values in Hz (batch_size, n_frames).
        classification_weight: Weight for classification loss.
        regression_weight: Weight for regression loss.
        
    Returns:
        tuple: (total_loss, classification_loss, regression_loss)
    """
    batch_size, n_bins, n_frames = pitch_logits.shape
    
    # --- Classification Loss (CrossEntropy) ---
    # Reshape for cross_entropy: (N, C, d1, d2, ...) where C is number of classes
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
    
    # L1 loss on log frequencies
    regression_loss = F.l1_loss(predicted_log_f0, log_target_f0)
    
    # --- Combined Loss ---
    total_loss = classification_weight * classification_loss + regression_weight * regression_loss
    
    return total_loss, classification_loss, regression_loss