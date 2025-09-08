import os
import sys
import torch
from src.train import train_model
from src.utils import find_data_files, split_data
from config import TRAIN_DATA_DIR, VAL_DATA_DIR, MODEL_SAVE_PATH, TRAINING_PARAMS, LOG_DIR

def main():
    # Create output directories
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Find training data files
    print("Finding training data files...")
    train_files = find_data_files(TRAIN_DATA_DIR, ['.wav'])
    val_files = find_data_files(VAL_DATA_DIR, ['.wav'])
    
    print(f"Found {len(train_files)} training files and {len(val_files)} validation files")
    
    # Check if we have data
    if len(train_files) == 0:
        print("No training files found. Please check your data directory.")
        sys.exit(1)
    
    # Start training
    print("Starting training...")
    train_model(
        train_data_paths=train_files,
        val_data_paths=val_files,
        model_save_path=MODEL_SAVE_PATH,
        **TRAINING_PARAMS
    )

if __name__ == "__main__":
    main()