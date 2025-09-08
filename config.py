import os

# --- Data Paths ---
# Define your dataset directories here
# The dataloader in dataset.py will search these paths for .wav/.f0 pairs
DATA_ROOT = "path/to/your/dataset" # e.g., "data/MIR1K" or "data"
TRAIN_DATA_PATHS = [os.path.join(DATA_ROOT, "train")] # Can be a list of dirs/files
VAL_DATA_PATHS = [os.path.join(DATA_ROOT, "val")]
# TEST_DATA_PATHS = [os.path.join(DATA_ROOT, "test")] # For final evaluation

# --- Model Parameters ---
# These are the key parameters you want to configure for your custom training
MODEL_PARAMS = {
    'n_bins': 360,        # <<< YOUR CUSTOM VALUE <<<
    'f_min': 46.875,      # Minimum frequency in Hz (standard for SwiftF0/CREPE)
    'f_max': 2093.75,     # Maximum frequency in Hz (standard for SwiftF0)
    'sample_rate': 16000, # Audio sample rate (standard)
    'hop_length': 160,    # <<< YOUR CUSTOM VALUE <<< (10ms at 16kHz)
    'n_fft': 1024,        # STFT window size (standard)
    'k_min': 3,           # Frequency bin slicing min index (from SwiftF0 paper)
    'k_max': 134          # Frequency bin slicing max index (from SwiftF0 paper)
    # Note: sliced_freq_bins = k_max - k_min = 131
    # freq_projection layer will be Conv1d(131, n_bins, kernel_size=1)
}

# --- Training Parameters ---
TRAINING_PARAMS = {
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'checkpoint_interval': 10, # Save checkpoint every N epochs
    'validation_interval': 5,  # Run validation every N epochs
    'use_augmentation': True,  # Enable data augmentation
    # Loss weights (as per SwiftF0 paper, often both are 1.0)
    'classification_weight': 1.0, 
    'regression_weight': 1.0
}

# --- Data Loading Parameters ---
DATALOADER_PARAMS = {
    'num_workers': 4, # Number of subprocesses for data loading
    'shuffle_train': True,
    'shuffle_val': False # Usually False for validation
}

# --- Loss Function Parameters ---
# These can be passed directly to the loss function if needed
# They are also included in TRAINING_PARAMS for convenience
LOSS_PARAMS = {
    'classification_weight': TRAINING_PARAMS['classification_weight'],
    'regression_weight': TRAINING_PARAMS['regression_weight']
}

# --- Output and Logging Paths ---
OUTPUT_DIR = "output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Default checkpoint filename (can be appended with epoch number)
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "swiftf0_model.pth")

# --- ONNX Export Parameters ---
# Used by export.py
EXPORT_PARAMS = {
    'opset_version': 11, # ONNX opset version
    '