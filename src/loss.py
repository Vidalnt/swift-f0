import torch
import torch.nn.functional as F

def compute_swiftf0_loss(model: torch.nn.Module,
                         pitch_logits: torch.Tensor,
                         target_indices: torch.Tensor, # MODIFIED: Receives indices
                         target_f0: torch.Tensor,
                         classification_weight: float = 1.0,
                         regression_weight: float = 1.0) -> tuple:
    """
    Compute the combined SwiftF0 loss using the model's buffers for accuracy.
    This is the recommended way to compute the loss during training.
   
    Args:
        model: The SwiftF0Model instance (to access pitch_bin_centers).
        pitch_logits: Output logits from model (batch_size, n_bins, n_frames).
        target_indices: Target class indices (batch_size, n_frames).
        target_f0: Target F0 values in Hz (batch_size, n_frames).
        classification_weight: Weight for classification loss.
        regression_weight: Weight for regression loss.
       
    Returns:
        tuple: (total_loss, classification_loss, regression_loss)
    """
    batch_size, n_bins, n_frames = pitch_logits.shape
   
    # --- Classification Loss (LCE) ---
    # MODIFIED: Uses cross_entropy directly with indices
    # Reshape for cross_entropy: (N, C) and (N)
    logits_reshaped = pitch_logits.permute(0, 2, 1).reshape(-1, n_bins)
    target_indices_reshaped = target_indices.reshape(-1)
   
    # PyTorch's cross_entropy ignores target_index -100 by default
    classification_loss = F.cross_entropy(logits_reshaped, target_indices_reshaped, ignore_index=-100)
   
    # --- Regression Loss (Lcents) ---
    probs = F.softmax(pitch_logits, dim=1) # Shape: [B, n_bins, T]
   
    log_pitch_bins = torch.log(model.pitch_bin_centers + 1e-8) # [n_bins]
    log_pitch_bins_expanded = log_pitch_bins.unsqueeze(0).unsqueeze(-1).expand_as(probs)
   
    predicted_log_f0 = torch.sum(probs * log_pitch_bins_expanded, dim=1) # [B, T]
   
    log_target_f0 = torch.log(target_f0 + 1e-8) # [B, T]
   
    # MODIFIED: Apply regression loss only on voiced frames
    voiced_mask = (target_f0 > 0) # Boolean mask [B, T]
   
    if voiced_mask.any():
        regression_loss = F.l1_loss(predicted_log_f0[voiced_mask], log_target_f0[voiced_mask])
    else:
        # If there are no voiced frames in the batch, regression loss is zero
        regression_loss = torch.tensor(0.0, device=pitch_logits.device)
   
    # --- Combined Loss ---
    total_loss = classification_weight * classification_loss + regression_weight * regression_loss
   
    return total_loss, classification_loss, regression_loss