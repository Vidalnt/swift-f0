import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from typing import Optional
from .model import create_model
from .loss import create_loss
from .dataset import create_dataloader
from .evaluate import evaluate_model

def train_model(
    train_data_paths: list,
    val_data_paths: list,
    model_save_path: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_interval: int = 10,
    log_dir: str = "logs"
):
    """
    Train the SwiftF0 model.
    
    Args:
        train_data_paths: List of paths to training data files
        val_data_paths: List of paths to validation data files
        model_save_path: Path to save the trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use for training ('cuda' or 'cpu')
        checkpoint_interval: Save checkpoint every N epochs
        log_dir: Directory for TensorBoard logs
    """
    # Create model, loss function, and optimizer
    model = create_model().to(device)
    loss_fn = create_loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Create data loaders
    train_loader = create_dataloader(train_data_paths, batch_size=batch_size, augment=True)
    val_loader = create_dataloader(val_data_paths, batch_size=batch_size, augment=False)
    
    # Setup TensorBoard logging
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_ce_loss = 0.0
        train_l1_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (spectrograms, class_targets, reg_targets) in enumerate(train_loader):
            # Move data to device
            spectrograms = spectrograms.to(device)
            class_targets = class_targets.to(device)
            reg_targets = reg_targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            class_logits, reg_output = model(spectrograms)
            
            # Compute loss
            total_loss, ce_loss, l1_loss = loss_fn(class_logits, reg_output, class_targets, reg_targets)
            
            # Backward pass
            total_loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate losses
            train_loss += total_loss.item()
            train_ce_loss += ce_loss.item()
            train_l1_loss += l1_loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}: Loss={total_loss.item():.4f}")
        
        # Compute average losses
        avg_train_loss = train_loss / num_batches
        avg_train_ce_loss = train_ce_loss / num_batches
        avg_train_l1_loss = train_l1_loss / num_batches
        
        # Validation phase
        val_loss, val_ce_loss, val_l1_loss = evaluate_model(model, loss_fn, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Train_CE', avg_train_ce_loss, epoch)
        writer.add_scalar('Loss/Train_L1', avg_train_l1_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Loss/Validation_CE', val_ce_loss, epoch)
        writer.add_scalar('Loss/Validation_L1', val_l1_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"  Train Loss: {avg_train_loss:.4f} (CE: {avg_train_ce_loss:.4f}, L1: {avg_train_l1_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (CE: {val_ce_loss:.4f}, L1: {val_l1_loss:.4f})")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_save_path.replace('.pth', '_best.pth'))
            print(f"  Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = model_save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, model_save_path)
    print(f"Training completed. Final model saved to {model_save_path}")
    
    # Close TensorBoard writer
    writer.close()