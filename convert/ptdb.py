import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_ptdb_dataset_smart(base_data_dir, f0_folder_name='REF', wav_folder_name='MIC', time_step=0.010):
    if not base_data_dir.is_dir():
        print(f"Error: The base directory '{base_data_dir}' does not exist.")
        return

    f0_search_path = base_data_dir / '**' / f0_folder_name
    f0_files = list(f0_search_path.rglob('*.f0'))
    
    if not f0_files:
        print(f"Error: No .f0 files were found in '{f0_search_path}'.")
        return
    
    processed_count = 0
    error_count = 0
    
    for f0_path in tqdm(f0_files, desc="Processing"):
        try:
            data = np.loadtxt(f0_path, usecols=(0, 1), dtype=float)
            frequencies = data[:, 0]
            voicing_decision = data[:, 1]
            
            frequencies[voicing_decision == 0] = 0.0
            
            num_frames = len(frequencies)
            frame_times = np.arange(num_frames) * time_step
            output_data = np.vstack((frame_times, frequencies)).T
            
            f0_path_str = str(f0_path)
            output_path_str = f0_path_str.replace(f'/{f0_folder_name}/', f'/{wav_folder_name}/', 1)
            output_path = Path(output_path_str)

            wav_path = output_path.with_suffix('.wav')
            if not wav_path.exists():
                error_count += 1
                continue

            np.savetxt(output_path, output_data, fmt='%.4f\t%.4f', header='time_sec\tfreq_hz', comments='')
            processed_count += 1

        except Exception as e:
            print(f"Error processing {f0_path}: {e}")
            error_count += 1

if __name__ == '__main__':
    BASE_DATA_DIR = 'path/to/PTDB-TUG/SPEECH DATA'
    F0_FOLDER = 'REF'
    WAV_FOLDER = 'MIC'
    TIME_STEP = 0.010
    
    base_dir_path = Path(BASE_DATA_DIR)
    convert_ptdb_dataset_smart(base_dir_path, F0_FOLDER, WAV_FOLDER, TIME_STEP)
