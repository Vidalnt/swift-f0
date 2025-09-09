import numpy as np
from pathlib import Path
from tqdm import tqdm

def midi_to_hz(midi):
    midi = np.asarray(midi, dtype=float)
    return np.where(midi <= 0.0, 0.0, 440.0 * 2.0 ** ((midi - 69.0) / 12.0))

def convert_mir1k_dataset(pv_directory, output_directory, hop_length, sample_rate):
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    pv_files = list(Path(pv_directory).glob('*.pv'))
    if not pv_files:
        print(f"No .pv files in: {pv_directory}")
        return
    for pv_path in tqdm(pv_files, desc="Processing"):
        try:
            midi_notes = np.loadtxt(pv_path, dtype=float)
            frame_times = np.arange(midi_notes.shape[0]) * hop_length / sample_rate
            hz_values = midi_to_hz(midi_notes)
            output_data = np.column_stack((frame_times, hz_values))
            output_path = out / (pv_path.stem + '.f0')
            np.savetxt(output_path, output_data, fmt='%.4f\t%.4f', comments='')
        except Exception as e:
            print(f"Error processing {pv_path}: {e}")

if __name__ == '__main__':
    PV_INPUT_DIR = 'path/to/MIR-1K/PitchLabel'
    F0_OUTPUT_DIR = 'path/to/MIR-1K/Wavfile'
    HOP_LENGTH = 160
    SAMPLE_RATE = 16000
    convert_mir1k_dataset(PV_INPUT_DIR, F0_OUTPUT_DIR, HOP_LENGTH, SAMPLE_RATE)
