"""
Evaluation script for the SwiftF0 model.
This script evaluates a trained SwiftF0 model on a validation or test dataset
and computes standard pitch estimation metrics like RPA, RCA, OA, VFA, VR.
It is designed to work with the SwiftF0Dataset and SwiftF0Model.
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
# Import metrics from a standard library, e.g., mir_eval
# If mir_eval is not available, you can implement basic versions or use a similar library.
try:
    from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy, overall_accuracy, voicing_measures
    MIR_EVAL_AVAILABLE = True
except ImportError:
    print("Warning: mir_eval not found. Basic accuracy calculations will be used.")
    MIR_EVAL_AVAILABLE = False
    # Define dummy functions or implement basic versions if mir_eval is not available.
    def raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=50):
        # Simplified RPA calculation
        if len(ref_c) == 0 or len(est_c) == 0 or len(ref_v) != len(est_v) or len(ref_c) != len(est_c):
            return 0.0
        # Consider only voiced frames in reference
        ref_voiced_idx = ref_v == 1
        if not np.any(ref_voiced_idx):
            return 1.0 # If no voiced frames in reference, perfect if est also has none
        diff_cents = np.abs(ref_c[ref_voiced_idx] - est_c[ref_voiced_idx])
        correct = np.sum(diff_cents < cent_tolerance)
        return correct / np.sum(ref_voiced_idx)
        
    def raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=50):
        # Simplified RCA calculation (ignores octaves)
        if len(ref_c) == 0 or len(est_c) == 0 or len(ref_v) != len(est_v) or len(ref_c) != len(est_c):
             return 0.0
        ref_voiced_idx = ref_v == 1
        if not np.any(ref_voiced_idx):
            return 1.0
        ref_cents_mod = np.remainder(ref_c[ref_voiced_idx], 1200)
        est_cents_mod = np.remainder(est_c[ref_voiced_idx], 1200)
        # Handle wrap-around, e.g., difference between 1190 and 10 cents
        diff_cents = np.abs(ref_cents_mod - est_cents_mod)
        diff_cents = np.minimum(diff_cents, 1200 - diff_cents)
        correct = np.sum(diff_cents < cent_tolerance)
        return correct / np.sum(ref_voiced_idx)

    def overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=50):
        # Simplified OA calculation (includes unvoiced)
        if len(ref_c) == 0 or len(est_c) == 0 or len(ref_v) != len(est_v) or len(ref_c) != len(est_c):
            return 0.0
        # Voicing decisions
        ref_voiced = ref_v == 1
        est_voiced = est_v == 1
        
        # Correct voiced pitch (within tolerance)
        # We evaluate all frames, not just ref voiced
        diff_cents = np.abs(ref_c - est_c)
        # For unvoiced frames in ref, est should also be unvoiced (or pitch doesn't matter)
        # For voiced frames in ref, est pitch should be close
        correct_voiced_pitch = (ref_voiced & (diff_cents < cent_tolerance))
        # Correct unvoiced detection (both unvoiced)
        correct_unvoiced = (~ref_voiced & ~est_voiced)
        
        correct = np.sum(correct_voiced_pitch) + np.sum(correct_unvoiced)
        return correct / len(ref_c)
        
    def voicing_measures(ref_v, est_v):
        # Simplified VR (Recall) and VFA (False Alarm)
        if len(ref_v) == 0 or len(est_v) == 0 or len(ref_v) != len(est_v):
            return 0.0, 0.0
        ref_voiced = ref_v == 1
        est_voiced = est_v == 1
        
        # True Positives (Voiced correctly identified)
        tp = np.sum(ref_voiced & est_voiced)
        # False Negatives (Voiced but identified as unvoiced)
        fn = np.sum(ref_voiced & ~est_voiced)
        # False Positives (Unvoiced but identified as voiced)
        fp = np.sum(~ref_voiced & est_voiced)
        
        # Voicing Recall (VR) = TP / (TP + FN)
        vr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Voicing False Alarm (VFA) = FP / (FP + TN) where TN = ~ref_voiced & ~est_voiced
        # Simplified as FP / (Total Unvoiced in ref)
        total_unvoiced_ref = np.sum(~ref_voiced)
        vfa = fp / total_unvoiced_ref if total_unvoiced_ref > 0 else 0.0
        
        return vr, vfa

from src.model import create_model # Needed if loading model directly in this script
from src.dataset import create_dataloader # Needed if creating loader directly
from src.loss import compute_swiftf0_loss # For computing validation loss if needed

def evaluate_model(model: torch.nn.Module, 
                   data_loader: torch.utils.data.DataLoader, 
                   device: torch.device,
                   loss_params: dict,
                   cent_tolerance: float = 50.0) -> dict:
    """
    Evaluate the model on a given dataset and compute metrics.
    
    Args:
        model: The trained SwiftF0 model.
        data_loader: DataLoader for the evaluation dataset.
        device: The device to run evaluation on (cuda/cpu).
        loss_params: Parameters for the loss function (e.g., weights).
        cent_tolerance: Tolerance in cents for RPA/RCA metrics.
        
    Returns:
        A dictionary containing evaluation metrics.
    """
    model.eval() # Set model to evaluation mode
    
    # Initialize accumulators for metrics
    total_loss = 0.0
    total_ce_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    # For mir_eval metrics
    all_ref_voicing = []
    all_ref_cents = []
    all_est_voicing = []
    all_est_cents = []
    all_est_confidence = [] # Optional, for confidence-based voicing
    
    # Use tqdm for a progress bar
    with torch.no_grad(): # Disable gradient computation for efficiency
        for audio, class_targets, target_f0_hz in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            audio = audio.to(device)
            class_targets = class_targets.to(device)
            target_f0_hz = target_f0_hz.to(device) # Shape: [B, n_frames]

            # Forward pass
            logits, confidence = model(audio) # model.forward returns (logits, confidence)
            
            # --- 1. Compute Loss ---
            # It's common to report the validation loss as well
            val_loss, val_ce_loss, val_reg_loss = compute_swiftf0_loss(
                model, logits, class_targets, target_f0_hz,
                classification_weight=loss_params.get('classification_weight', 1.0),
                regression_weight=loss_params.get('regression_weight', 1.0)
            )
            total_loss += val_loss.item()
            total_ce_loss += val_ce_loss.item()
            total_reg_loss += val_reg_loss.item()
            num_batches += 1

            # --- 2. Decode Pitch and Confidence ---
            # Decode predicted pitch from logits (in Hz)
            # Use the model's decode_pitch method or re-implement
            predicted_f0_hz = model.decode_pitch(logits) # [B, n_frames]
            
            # Determine voicing decisions based on confidence
            # A common threshold is 0.5, but this can be tuned
            confidence_threshold = 0.5
            # confidence is [B, n_frames]
            predicted_voicing = (confidence > confidence_threshold).cpu().numpy() # [B, n_frames]
            
            # Target voicing: We can define it based on target_f0 being > 0 or target confidence
            # Assuming target_f0 is 0 for unvoiced frames
            target_voicing = (target_f0_hz > 0).cpu().numpy() # [B, n_frames]
            
            # Convert frequencies to cents for metric calculation
            # Use a reference frequency, e.g., 10Hz as in SwiftF0/RMVPE papers
            f_ref = 10.0
            # Handle log(0) by adding a small epsilon or masking
            eps = 1e-8
            # For target: mask unvoiced frames (where f0 is 0) to avoid incorrect cent calculation
            # mir_eval typically handles this via the voicing arrays
            target_f0_hz_np = target_f0_hz.cpu().numpy() # [B, n_frames]
            predicted_f0_hz_np = predicted_f0_hz.cpu().numpy() # [B, n_frames]
            
            # Convert to cents: 1200 * log2(f / f_ref)
            # We only care about voiced frames for pitch accuracy, but mir_eval handles this
            target_cents = 1200 * np.log2((target_f0_hz_np + eps) / f_ref) # [B, n_frames]
            predicted_cents = 1200 * np.log2((predicted_f0_hz_np + eps) / f_ref) # [B, n_frames]
            
            # Flatten arrays for mir_eval (it expects 1D arrays)
            # mir_eval.melody functions typically work on entire tracks, 
            # but we can accumulate results across all frames/batches
            all_ref_voicing.append(target_voicing.flatten()) 
            all_ref_cents.append(target_cents.flatten())
            all_est_voicing.append(predicted_voicing.flatten())
            all_est_cents.append(predicted_cents.flatten())
            all_est_confidence.append(confidence.cpu().numpy().flatten()) # Optional
            
    # --- 3. Calculate Final Metrics ---
    results = {}
    
    # Average Losses
    if num_batches > 0:
        results['avg_loss'] = total_loss / num_batches
        results['avg_ce_loss'] = total_ce_loss / num_batches
        results['avg_reg_loss'] = total_reg_loss / num_batches
    else:
        results['avg_loss'] = results['avg_ce_loss'] = results['avg_reg_loss'] = float('inf')
        
    # Concatenate all accumulated values for mir_eval
    if all_ref_voicing and all_ref_cents and all_est_voicing and all_est_cents:
        ref_v = np.concatenate(all_ref_voicing)
        ref_c = np.concatenate(all_ref_cents)
        est_v = np.concatenate(all_est_voicing)
        est_c = np.concatenate(all_est_cents)
        # est_conf = np.concatenate(all_est_confidence) # If needed for advanced voicing
        
        # Compute mir_eval metrics
        try:
            # Raw Pitch Accuracy (RPA)
            rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=cent_tolerance)
            results['rpa'] = rpa
            
            # Raw Chroma Accuracy (RCA)
            rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=cent_tolerance)
            results['rca'] = rca
            
            # Overall Accuracy (OA)
            oa = overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=cent_tolerance)
            results['oa'] = oa
            
            # Voicing Measures (VR, VFA)
            vr, vfa = voicing_measures(ref_v, est_v)
            results['vr'] = vr # Voicing Recall
            results['vfa'] = vfa # Voicing False Alarm
            
        except Exception as e:
            print(f"Error calculating mir_eval metrics: {e}")
            # Set default values or handle error
            results['rpa'] = results['rca'] = results['oa'] = results['vr'] = results['vfa'] = 0.0
    else:
        print("Warning: No data collected for evaluation metrics.")
        results['rpa'] = results['rca'] = results['oa'] = results['vr'] = results['vfa'] = 0.0
        
    model.train() # Set model back to train mode
    return results


def main():
    """
    Main function to run evaluation from command line.
    This is useful for evaluating a saved model checkpoint.
    """
    import argparse
    import os
    from config import MODEL_PARAMS, VAL_DATA_PATHS, DATALOADER_PARAMS, LOSS_PARAMS
    
    parser = argparse.ArgumentParser(description='Evaluate SwiftF0 Model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, 
                        help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--data_paths', '-d', type=str, nargs='+', 
                        default=VAL_DATA_PATHS, 
                        help='Paths to validation data directories or files')
    parser.add_argument('--batch_size', '-b', type=int, 
                        default=DATALOADER_PARAMS.get('batch_size', 32), 
                        help='Batch size for evaluation')
    parser.add_argument('--cent_tolerance', '-t', type=float, default=50.0,
                        help='Cent tolerance for RPA/RCA metrics (default: 50)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for evaluation (cuda/cpu). If None, defaults to cuda if available.')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Create model
    print("Creating model...")
    model = create_model(**MODEL_PARAMS).to(device)
    
    # 2. Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully.")
    
    # 3. Create data loader
    print("Creating data loader...")
    # Note: For final evaluation, shuffle is usually False
    eval_loader = create_dataloader(
        args.data_paths,
        batch_size=args.batch_size,
        shuffle=False, # Important for reproducible evaluation
        num_workers=DATALOADER_PARAMS.get('num_workers', 4),
        **MODEL_PARAMS # Pass model params like hop_length, n_bins to dataset
    )
    print(f"Data loader created with {len(eval_loader)} batches.")
    
    # 4. Run evaluation
    print("Starting evaluation...")
    metrics = evaluate_model(
        model=model,
        data_loader=eval_loader,
        device=device,
        loss_params=LOSS_PARAMS,
        cent_tolerance=args.cent_tolerance
    )
    
    # 5. Print results
    print("\n" + "="*40)
    print("Evaluation Results:")
    print("="*40)
    print(f"Average Loss:        {metrics.get('avg_loss', 'N/A'):.4f}")
    print(f"Classification Loss: {metrics.get('avg_ce_loss', 'N/A'):.4f}")
    print(f"Regression Loss:     {metrics.get('avg_reg_loss', 'N/A'):.4f}")
    print("-" * 40)
    print(f"Raw Pitch Accuracy (RPA @{args.cent_tolerance} cents):     {metrics.get('rpa', 0.0)*100:.2f}%")
    print(f"Raw Chroma Accuracy (RCA @{args.cent_tolerance} cents):    {metrics.get('rca', 0.0)*100:.2f}%")
    print(f"Overall Accuracy (OA @{args.cent_tolerance} cents):        {metrics.get('oa', 0.0)*100:.2f}%")
    print(f"Voicing Recall (VR):                                       {metrics.get('vr', 0.0)*100:.2f}%")
    print(f"Voicing False Alarm (VFA):                                 {metrics.get('vfa', 0.0)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
