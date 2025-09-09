"""
Main training script for the SwiftF0 model.
This script handles the training loop, validation, checkpointing, and logging.
"""

import torch
import torch.optim as optim
import os
import time
# Import tqdm
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

# Import our modules
from src.model import create_model
from src.loss import compute_swiftf0_loss 
from src.dataset import create_dataloader
from src.utils import save_checkpoint, load_checkpoint

# Import configuration
from config import (MODEL_PARAMS, TRAINING_PARAMS, DATALOADER_PARAMS, 
                    LOSS_PARAMS, MODEL_SAVE_PATH, LOG_DIR, 
                    TRAIN_DATA_PATHS, VAL_DATA_PATHS)

def validate_model(model, val_loader, device, loss_params):
    """Performs model validation."""
    model.eval()
    total_val_loss = 0.0
    total_val_ce_loss = 0.0
    total_val_reg_loss = 0.0
    num_batches = 0
    
    # Wrap the validation loader with tqdm for progress bar
    val_pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        # MODIFIED: Use tqdm progress bar for validation
        for audio, target_indices, target_f0 in val_pbar:
            audio = audio.to(device)
            target_indices = target_indices.to(device)
            target_f0 = target_f0.to(device)

            # Forward pass
            logits, _ = model(audio) # Model returns (logits, confidence)

            # MODIFIED: Pass target_indices to loss function
            val_loss, val_ce_loss, val_reg_loss = compute_swiftf0_loss(
                model, logits, target_indices, target_f0,
                classification_weight=loss_params['classification_weight'],
                regression_weight=loss_params['regression_weight']
            )

            total_val_loss += val_loss.item()
            total_val_ce_loss += val_ce_loss.item()
            total_val_reg_loss += val_reg_loss.item()
            num_batches += 1
            
            # Update progress bar description with current batch loss
            val_pbar.set_postfix({'Loss': f'{val_loss.item():.4f}'})

    model.train() # Return to training mode
    if num_batches > 0:
        avg_val_loss = total_val_loss / num_batches
        avg_val_ce_loss = total_val_ce_loss / num_batches
        avg_val_reg_loss = total_val_reg_loss / num_batches
        return avg_val_loss, avg_val_ce_loss, avg_val_reg_loss
    else:
        return float('inf'), float('inf'), float('inf')


def main():
    """Main training function."""
    print("Starting SwiftF0 training...")
    
    # --- 1. Setup Device ---
    device = torch.device(TRAINING_PARAMS['device'])
    print(f"Using device: {device}")
    
    # --- 2. Create Model ---
    print("Creating model...")
    model = create_model(**MODEL_PARAMS).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    # --- 3. Create DataLoaders ---
    print("Creating DataLoaders...")
    train_loader = create_dataloader(
        TRAIN_DATA_PATHS, 
        batch_size=TRAINING_PARAMS['batch_size'],
        shuffle=DATALOADER_PARAMS['shuffle_train'],
        num_workers=DATALOADER_PARAMS['num_workers'],
        **{k: v for k, v in MODEL_PARAMS.items() if k not in ("k_min", "k_max")} # Pass model parameters like hop_length to dataset
    )
    val_loader = create_dataloader(
        VAL_DATA_PATHS, 
        batch_size=TRAINING_PARAMS['batch_size'],
        shuffle=DATALOADER_PARAMS['shuffle_val'],
        num_workers=DATALOADER_PARAMS['num_workers'],
        **{k: v for k, v in MODEL_PARAMS.items() if k not in ("k_min", "k_max")}
    )
    print("DataLoaders created.")
    
    # --- 4. Setup Optimizer and Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_PARAMS['learning_rate'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.98) 
    
    # --- 5. Setup Logging ---
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # --- 6. Resume from Checkpoint (Optional) ---
    start_epoch = 0
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            print(f"Resuming from checkpoint: {MODEL_SAVE_PATH}")
            start_epoch, _ = load_checkpoint(model, optimizer, MODEL_SAVE_PATH)
            start_epoch += 1 # Start from next epoch
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
    
    # --- 7. Training Loop ---
    num_epochs = TRAINING_PARAMS['num_epochs']
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        
        total_train_loss = 0.0
        total_train_ce_loss = 0.0
        total_train_reg_loss = 0.0
        num_train_batches = 0
        
        epoch_start_time = time.time()
        
        # Wrap train_loader with tqdm
        # MODIFIED: Use tqdm progress bar for training loop
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        # MODIFIED: Dataloader unpacking with tqdm
        # for batch_idx, (audio, target_indices, target_f0) in enumerate(train_loader):
        for batch_idx, (audio, target_indices, target_f0) in enumerate(train_pbar):
            audio, target_indices, target_f0 = audio.to(device), target_indices.to(device), target_f0.to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(audio)
            
            # MODIFIED: Loss function call with indices
            loss, ce_loss, reg_loss = compute_swiftf0_loss(
                model, logits, target_indices, target_f0,
                classification_weight=LOSS_PARAMS['classification_weight'],
                regression_weight=LOSS_PARAMS['regression_weight']
            )
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_ce_loss += ce_loss.item()
            total_train_reg_loss += reg_loss.item()
            num_train_batches += 1
            
            # Update the progress bar description with current loss
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_train_loss / num_train_batches:.4f}'
            })
            
            # Optional: Still print every 100 batches if you prefer
            # if (batch_idx + 1) % 100 == 0:
            #     print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # --- End of Epoch ---
        epoch_time = time.time() - epoch_start_time
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_train_ce_loss = total_train_ce_loss / num_train_batches if num_train_batches > 0 else 0
        avg_train_reg_loss = total_train_reg_loss / num_train_batches if num_train_batches > 0 else 0
            
        # Update the epoch summary print
        print(f"  Epoch {epoch+1} Summary - "
              f"Training Loss: {avg_train_loss:.4f} "
              f"(CE: {avg_train_ce_loss:.4f}, Reg: {avg_train_reg_loss:.4f}) "
              f"Time: {epoch_time:.2f}s")

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Train_Classification', avg_train_ce_loss, epoch)
        writer.add_scalar('Loss/Train_Regression', avg_train_reg_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Time/Epoch', epoch_time, epoch) # Log epoch time

        # --- Validation ---
        if (epoch + 1) % TRAINING_PARAMS['validation_interval'] == 0:
            print("  Running validation...")
            avg_val_loss, avg_val_ce_loss, avg_val_reg_loss = validate_model(
                model, val_loader, device, LOSS_PARAMS
            )
            
            print(f"  Validation Loss: {avg_val_loss:.4f} "
                  f"(CE: {avg_val_ce_loss:.4f}, Reg: {avg_val_reg_loss:.4f})")
            
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Loss/Validation_Classification', avg_val_ce_loss, epoch)
            writer.add_scalar('Loss/Validation_Regression', avg_val_reg_loss, epoch)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = MODEL_SAVE_PATH.replace('.pth', '_best.pth')
                save_checkpoint(model, optimizer, epoch, avg_val_loss, best_model_path)
                print(f"  New best model saved with validation loss: {avg_val_loss:.4f}")

        # --- Checkpointing ---
        if (epoch + 1) % TRAINING_PARAMS['checkpoint_interval'] == 0:
            checkpoint_path = MODEL_SAVE_PATH.replace('.pth', f'_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_path)

        scheduler.step()
        
    # --- Training Completed ---
    print("\nTraining completed.")
    final_model_path = MODEL_SAVE_PATH.replace('.pth', '_final.pth')
    save_checkpoint(model, optimizer, num_epochs - 1, avg_train_loss, final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    writer.close()
    print(f"Logs saved at {LOG_DIR}")

if __name__ == "__main__":
    main()