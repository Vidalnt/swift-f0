import os

# Data paths
DATA_DIR = "data"
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
VAL_DATA_DIR = os.path.join(DATA_DIR, "val")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

# Model parameters
MODEL_PARAMS = {
    'n_bins': 360,
    'f_min': 46.875,
    'f_max': 2093.75,
    'sample_rate': 16000
}

# Training parameters
TRAINING_PARAMS = {
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'checkpoint_interval': 10
}

# Loss parameters
LOSS_PARAMS = {
    'classification_weight': 1.0,
    'regression_weight': 1.0
}

# Output paths
OUTPUT_DIR = "output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "swiftf0_model.pth")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")