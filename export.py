import torch
from src.model import create_model
from src.utils import convert_to_onnx
from config import MODEL_SAVE_PATH, MODEL_PARAMS

def main():
    # Create model
    model = create_model(**MODEL_PARAMS)
    
    # Load trained weights
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Convert to ONNX
    input_shape = (1, 1, 100, 513)  # (batch_size, channels, time_frames, freq_bins)
    output_path = MODEL_SAVE_PATH.replace('.pth', '.onnx')
    
    convert_to_onnx(model, input_shape, output_path)

if __name__ == "__main__":
    main()