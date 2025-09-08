"""
Main training script for the SwiftF0 model.
This script handles the training loop, validation, checkpointing, and logging.
"""

import torch
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

# Import our modules
from src.model import create_model
# Note: We will use the functional loss from src/loss.py
from src.loss import compute_swiftf0_loss 
from src.dataset import create_dataloader
from src.utils import save_checkpoint, load_checkpoint
# Import configuration
from config import MODEL_PARAMS, TRAINING_PARAMS, DATALOADER_PARAMS, LOSS_PARAMS, MODEL_SAVE_PATH, LOG_DIR


def validate_model(model, val_loader, device, loss_params):
    """Perform validation on the model."""
    model.eval()
    total_val_loss = 0.0
    total_val_ce_loss = 0.0
    total_val_reg_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for audio, class_targets, target_f0 in val_loader:
            audio = audio.to(device)
            class_targets = class_targets.to(device)
            target_f0 = target_f0.to(device)

            # Forward pass
            logits, _ = model(audio) # model returns (logits, confidence)

            # Compute loss using the recommended functional approach
            # This correctly uses the model's pitch_bin_centers buffer
            val_loss, val_ce_loss, val_reg_loss = compute_swiftf0_loss(
                model, logits, class_targets, target_f0,
                classification_weight=loss_params['classification_weight'],
                regression_weight=loss_params['regression_weight']
            )

            total_val_loss += val_loss.item()
            total_val_ce_loss += val_ce_loss.item()
            total_val_reg_loss += val_reg_loss.item()
            num_batches += 1

    model.train() # Set back to train mode
    if num_batches > 0:
        avg_val_loss = total_val_loss / num_batches
        avg_val_ce_loss = total_val_ce_loss / num_batches
        avg_val_reg_loss = total_val_reg_loss / num_batches
        return avg_val_loss, avg_val_ce_loss, avg_val_reg_loss
    else:
        return float('inf'), float('inf'), float('inf')


def main():
    """Main training function."""
    print("Starting SwiftF0 Training...")
    
    # --- 1. Setup Device ---
    device = TRAINING_PARAMS['device']
    print(f"Using device: {device}")
    
    # --- 2. Create Model ---
    print("Creating model...")
    model = create_model(**MODEL_PARAMS).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # --- 3. Create Data Loaders ---
    print("Creating data loaders...")
    # Assuming config.py has TRAIN_DATA_PATHS and VAL_DATA_PATHS defined
    # You might need to adjust config.py or pass paths differently
    # For now, let's assume they are lists of paths
    from config import TRAIN_DATA_PATHS, VAL_DATA_PATHS
    
    train_loader = create_dataloader(
        TRAIN_DATA_PATHS, 
        batch_size=TRAINING_PARAMS['batch_size'],
        shuffle=DATALOADER_PARAMS['shuffle_train'],
        num_workers=DATALOADER_PARAMS['num_workers'],
        **MODEL_PARAMS # Pass model params like hop_length, n_bins to dataset
    )
    
    val_loader = create_dataloader(
        VAL_DATA_PATHS, 
        batch_size=TRAINING_PARAMS['batch_size'], # Can be different
        shuffle=DATALOADER_PARAMS['shuffle_val'],
        num_workers=DATALOADER_PARAMS['num_workers'],
        **MODEL_PARAMS
    )
    print("Data loaders created.")
    
    # --- 4. Setup Optimizer and Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_PARAMS['learning_rate'])
    # Example scheduler: reduce LR by 0.98 every 10 epochs (as in RMVPE)
    # Adjust step_size based on your epochs
    scheduler = StepLR(optimizer, step_size=10, gamma=0.98) 
    
    # --- 5. Setup Logging ---
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # --- 6. Resume from Checkpoint (Optional) ---
    start_epoch = 0
    # Add logic to resume if a checkpoint exists
    # if os.path.exists(MODEL_SAVE_PATH):
    #     try:
    #         start_epoch, _ = load_checkpoint(model, optimizer, MODEL_SAVE_PATH)
    #         start_epoch += 1 # Start from the next epoch
    #     except Exception as e:
    #         print(f"Could not load checkpoint: {e}. Starting from scratch.")
    
    # --- 7. Training Loop ---
    num_epochs = TRAINING_PARAMS['num_epochs']
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train() # Set model to training mode
        
        total_train_loss = 0.0
        total_train_ce_loss = 0.0
        total_train_reg_loss = 0.0
        num_train_batches = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (audio, class_targets, target_f0) in enumerate(train_loader):
            # Move data to device
            audio = audio.to(device)
            class_targets = class_targets.to(device)
            target_f0 = target_f0.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(audio) # model.forward returns (logits, confidence)
            
            # Compute loss using the function that correctly accesses model buffers
            loss, ce_loss, reg_loss = compute_swiftf0_loss(
                model, logits, class_targets, target_f0,
                classification_weight=LOSS_PARAMS['classification_weight'],
                regression_weight=LOSS_PARAMS['regression_weight']
            )
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate losses for reporting
            total_train_loss += loss.item()
            total_train_ce_loss += ce_loss.item()
            total_train_reg_loss += reg_loss.item()
            num_train_batches += 1
            
            # Print progress every N batches
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

        # --- Epoch End ---
        epoch_time = time.time() - epoch_start_time
        
        # Calculate average training losses
        if num_train_batches > 0:
            avg_train_loss = total_train_loss / num_train_batches
            avg_train_ce_loss = total_train_ce_loss / num_train_batches
            avg_train_reg_loss = total_train_reg_loss / num_train_batches
        else:
            avg_train_loss = avg_train_ce_loss = avg_train_reg_loss = 0.0
            
        print(f"  Train Loss: {avg_train_loss:.4f} "
              f"(CE: {avg_train_ce_loss:.4f}, Reg: {avg_train_reg_loss:.4f}) "
              f"Time: {epoch_time:.2f}s")

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Train_Classification', avg_train_ce_loss, epoch)
        writer.add_scalar('Loss/Train_Regression', avg_train_reg_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # --- Validation ---
        if (epoch + 1) % TRAINING_PARAMS['validation_interval'] == 0:
            print("  Running validation...")
            val_start_time = time.time()
            avg_val_loss, avg_val_ce_loss, avg_val_reg_loss = validate_model(
                model, val_loader, device, LOSS_PARAMS
            )
            val_time = time.time() - val_start_time
            
            print(f"  Val Loss: {avg_val_loss:.4f} "
                  f"(CE: {avg_val_ce_loss:.4f}, Reg: {avg_val_reg_loss:.4f}) "
                  f"Time: {val_time:.2f}s")
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Loss/Validation_Classification', avg_val_ce_loss, epoch)
            writer.add_scalar('Loss/Validation_Regression', avg_val_reg_loss, epoch)
            
            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = MODEL_SAVE_PATH.replace('.pth', '_best.pth')
                save_checkpoint(model, optimizer, epoch, avg_val_loss, best_model_path)
                print(f"  New best model saved with val loss: {avg_val_loss:.4f}")

        # --- Checkpointing ---
        if (epoch + 1) % TRAINING_PARAMS['checkpoint_interval'] == 0:
            checkpoint_path = MODEL_SAVE_PATH.replace('.pth', f'_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_path)

        # Update learning rate
        scheduler.step()
        
    # --- Training Finished ---
    print("\nTraining completed.")
    # Save final model
    final_model_path = MODEL_SAVE_PATH.replace('.pth', '_final.pth')
    save_checkpoint(model, optimizer, num_epochs, avg_train_loss, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    writer.close()
    print(f"Logs saved to {LOG_DIR}")

if __name__ == "__main__":
    main()
