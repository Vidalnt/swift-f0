"""
Export script for SwiftF0 PyTorch model to ONNX format.

This script loads a trained SwiftF0 PyTorch model and exports it to ONNX format,
ensuring compatibility with the original ONNX model structure observed.
"""

import torch
import torch.onnx
import argparse
import os
from src.model import create_model
from config import MODEL_PARAMS

def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    model_params: dict,
    opset_version: int = 11,
    simplify: bool = True
) -> None:
    """
    Export a trained SwiftF0 PyTorch model to ONNX format.

    Args:
        checkpoint_path: Path to the trained PyTorch model checkpoint (.pth file).
        output_path: Path to save the exported ONNX model.
        model_params: Dictionary of model parameters (n_bins, f_min, f_max, etc.).
        opset_version: ONNX opset version to use (default: 11, compatible with ONNX.js).
        simplify: Whether to simplify the ONNX model using onnxsim (requires onnxsim).
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # 1. Create model instance with specified parameters
    model = create_model(**model_params)
    
    # 2. Load trained weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume the checkpoint is directly the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully.")
    
    # 3. Create dummy input that matches SwiftF0 requirements
    # According to the paper and ONNX inspection:
    # - Input is raw audio
    # - STFT is computed internally by the model
    # - Sample rate is 16kHz
    # - Typical audio duration for inference is variable, but we need a fixed size for export
    
    # Use a short audio segment for dummy input (e.g., 1 second at 16kHz)
    dummy_audio_length = model_params.get('sample_rate', 16000) # 1 second of audio
    dummy_input = torch.randn(1, dummy_audio_length) # (batch_size=1, audio_samples)
    
    print(f"Created dummy input with shape: {dummy_input.shape}")
    
    # 4. Define input/output names to match the original ONNX model
    input_names = ['input_audio']
    output_names = ['pitch_hz', 'confidence']
    
    # 5. Export to ONNX
    print("Starting ONNX export...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input_audio': {1: 'audio_length'},  # Allow variable audio length
            'pitch_hz': {1: 'n_frames'},         # Allow variable number of frames
            'confidence': {1: 'n_frames'}        # Allow variable number of frames
        }
    )
    print(f"Model exported to ONNX format at: {output_path}")
    
    # 6. Optional: Simplify the ONNX model
    if simplify:
        try:
            import onnx
            from onnxsim import simplify
            
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified_model, check = simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, output_path)
                print("ONNX model simplified successfully.")
            else:
                print("Warning: ONNX model simplification check failed. Model not simplified.")
                
        except ImportError:
            print("Warning: onnxsim not installed. Skipping model simplification.")
            print("To install: pip install onnxsim")
        except Exception as e:
            print(f"Warning: Failed to simplify ONNX model: {e}")

def main():
    """Main function to parse arguments and run export."""
    parser = argparse.ArgumentParser(description="Export SwiftF0 PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to the trained PyTorch model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save the exported ONNX model"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX model simplification"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run export
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_params=MODEL_PARAMS,
        opset_version=args.opset,
        simplify=not args.no_simplify
    )
    
    print("Export completed successfully!")

if __name__ == "__main__":
    main()