"""
Export script for SwiftF0 PyTorch model to ONNX format.
This script loads a trained SwiftF0 PyTorch model and exports it to ONNX format.
It is designed to work with the model architecture defined in src/model.py.
Note: The exported model's output logic (pitch_hz, confidence) will depend on 
the `forward` method of the model. If `forward` only returns logits, 
the ONNX model will too. For a model that directly outputs pitch and confidence 
like the original, the model's `forward` needs to include that logic.
"""

import torch
import torch.onnx
import argparse
import os
from src.model import create_model # Import the model creation function
# from config import MODEL_PARAMS, EXPORT_PARAMS # Optional: for default params

def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    model_params: dict, # Should include n_bins, hop_length, n_fft, k_min, k_max etc.
    opset_version: int = 11,
    simplify: bool = True
) -> None:
    """
    Export a trained SwiftF0 PyTorch model to ONNX format.

    Args:
        checkpoint_path: Path to the trained PyTorch model checkpoint (.pth file).
        output_path: Path to save the exported ONNX model.
        model_params: Dictionary of model parameters (n_bins, f_min, f_max, etc.).
                      Must match the parameters used during training.
        opset_version: ONNX opset version to use (default: 11).
        simplify: Whether to simplify the ONNX model using onnxsim (requires onnxsim).
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # 1. Create model instance with specified parameters
    # It's crucial that these params match the training config
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
    print("Model loaded and set to evaluation mode.")
    
    # 3. Create dummy input that matches SwiftF0 requirements
    # Input is raw audio. Shape: (batch_size, audio_samples)
    # For export, we need a fixed length. A common choice is a 1-second signal.
    sample_rate = model_params.get('sample_rate', 16000)
    dummy_audio_length = sample_rate # 1 second of audio
    dummy_input = torch.randn(1, dummy_audio_length) # (batch_size=1, audio_samples)
    
    print(f"Created dummy input with shape: {dummy_input.shape}")
    
    # 4. Define input/output names 
    # These should match what the model's `forward` method expects/returns
    input_names = ['input_audio']
    # If model.forward returns (logits, confidence):
    output_names = ['pitch_logits', 'confidence'] 
    # If you modify model.forward to return (pitch_hz, confidence) like the ONNX:
    # output_names = ['pitch_hz', 'confidence']
    
    # 5. Export to ONNX
    print("Starting ONNX export...")
    try:
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
                # Adjust output names based on model.forward's actual return
                'pitch_logits': {2: 'n_frames'},     
                'confidence': {1: 'n_frames'}        
                # If outputs were pitch_hz, confidence:
                # 'pitch_hz': {1: 'n_frames'},         
                # 'confidence': {1: 'n_frames'}        
            }
        )
        print(f"Model successfully exported to ONNX format at: {output_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        raise

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
    # It's often easier to pass a config file or hardcode params for export
    # to ensure they match training. For simplicity here, we use defaults
    # or require them to be passed. A more robust way is to save model params
    # with the checkpoint.
    parser.add_argument(
        "--n_bins",
        type=int,
        default=360, # Your custom value
        help="Number of pitch bins (default: 360)"
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=160, # Your custom value
        help="STFT hop length (default: 160)"
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
    
    # Construct model_params dict for create_model
    # These must match the training configuration!
    # In a more robust setup, you'd save/load these with the checkpoint.
    model_params = {
        'n_bins': args.n_bins,
        'f_min': 46.875,
        'f_max': 2093.75,
        'sample_rate': 16000,
        'hop_length': args.hop_length,
        'n_fft': 1024,
        'k_min': 3,
        'k_max': 134
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run export
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_params=model_params,
        opset_version=args.opset,
        simplify=not args.no_simplify
    )
    
    print("Export completed successfully!")

if __name__ == "__main__":
    main()
