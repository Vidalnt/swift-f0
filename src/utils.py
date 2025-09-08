import torch
import numpy as np
import os
from typing import List

def find_data_files(data_dir: str, extensions: List[str] = ['.wav']) -> List[str]:
    """
    Find all data files in a directory with specified extensions.
    
    Args:
        data_dir: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of file paths
    """
    data_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                data_files.append(os.path.join(root, file))
    return data_files

def split_data(data_files: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data files into train, validation, and test sets.
    
    Args:
        data_files: List of data file paths
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    import random
    random.shuffle(data_files)
    
    n_train = int(len(data_files) * train_ratio)
    n_val = int(len(data_files) * val_ratio)
    
    train_files = data_files[:n_train]
    val_files = data_files[n_train:n_train + n_val]
    test_files = data_files[n_train + n_val:]
    
    return train_files, val_files, test_files

def convert_to_onnx(model, input_shape, output_path):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor
        output_path: Path to save ONNX model
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['classification_output', 'regression_output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'classification_output': {0: 'batch_size'},
            'regression_output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")