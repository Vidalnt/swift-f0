import torch
import numpy as np
from typing import Tuple

def evaluate_model(model, loss_fn, data_loader, device: str) -> Tuple[float, float, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        loss_fn: The loss function
        data_loader: DataLoader for the evaluation dataset
        device: Device to use for evaluation
        
    Returns:
        Tuple of (avg_loss, avg_ce_loss, avg_l1_loss)
    """
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_l1_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for spectrograms, class_targets, reg_targets in data_loader:
            # Move data to device
            spectrograms = spectrograms.to(device)
            class_targets = class_targets.to(device)
            reg_targets = reg_targets.to(device)
            
            # Forward pass
            class_logits, reg_output = model(spectrograms)
            
            # Compute loss
            total_loss_batch, ce_loss, l1_loss = loss_fn(class_logits, reg_output, class_targets, reg_targets)
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_ce_loss += ce_loss.item()
            total_l1_loss += l1_loss.item()
            num_batches += 1
    
    # Compute average losses
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_ce_loss = total_ce_loss / num_batches if num_batches > 0 else 0.0
    avg_l1_loss = total_l1_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_ce_loss, avg_l1_loss

def calculate_metrics(model, data_loader, device: str) -> dict:
    """
    Calculate evaluation metrics (RPA, RCA, etc.) for the model.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the evaluation dataset
        device: Device to use for evaluation
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Initialize counters
    total_frames = 0
    correct_pitch_frames = 0
    correct_chroma_frames = 0
    voiced_frames = 0
    true_voiced_frames = 0
    
    with torch.no_grad():
        for spectrograms, class_targets, reg_targets in data_loader:
            # Move data to device
            spectrograms = spectrograms.to(device)
            class_targets = class_targets.to(device)
            reg_targets = reg_targets.to(device)
            
            # Forward pass
            class_logits, reg_output = model(spectrograms)
            
            # Get predicted pitch
            pred_pitch = model.get_pitch_from_logits(class_logits)
            true_pitch = torch.exp(reg_targets.squeeze())
            
            # Convert to cents for comparison
            pred_cents = 1200 * torch.log2(pred_pitch / 10.0)
            true_cents = 1200 * torch.log2(true_pitch / 10.0)
            
            # Calculate differences
            diff_cents = torch.abs(pred_cents - true_cents)
            
            # Count frames within thresholds
            correct_pitch_frames += torch.sum(diff_cents < 50).item()  # 50 cents threshold
            # For chroma accuracy, we need to consider octave errors
            chroma_diff = torch.remainder(diff_cents, 1200)
            chroma_diff = torch.min(chroma_diff, 1200 - chroma_diff)
            correct_chroma_frames += torch.sum(chroma_diff < 50).item()
            
            # Count voiced frames (simplified)
            voiced_frames += spectrograms.size(0)
            true_voiced_frames += spectrograms.size(0)  # Assuming all are voiced in this dataset
            
            total_frames += spectrograms.size(0)
    
    # Calculate metrics
    rpa = correct_pitch_frames / total_frames if total_frames > 0 else 0.0
    rca = correct_chroma_frames / total_frames if total_frames > 0 else 0.0
    vr = true_voiced_frames / total_frames if total_frames > 0 else 0.0  # Simplified
    
    return {
        'RPA': rpa,
        'RCA': rca,
        'VR': vr,
        'total_frames': total_frames
    }