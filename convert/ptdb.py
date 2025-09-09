from pathlib import Path
import numpy as np
from tqdm import tqdm

def convert_ptdb_dataset_smart(base_data_dir, f0_folder_name='REF', wav_folder_name='MIC', time_step=0.010):
    base = Path(base_data_dir)
    if not base.is_dir():
        print(f"Error: Base directory '{base}' does not exist.")
        return
    
    f0_files = list(base.glob(f'**/{f0_folder_name}/**/*.f0'))
    if not f0_files:
        print(f"Error: No .f0 files found in '{base}/**/{f0_folder_name}'.")
        return
    
    processed = 0
    missing_wav = 0
    errors = 0
    
    for f0_path in tqdm(f0_files, desc="Processing .f0"):
        try:
            data = np.loadtxt(f0_path, usecols=(0, 1), dtype=float)
            freqs = data[:, 0]
            voiced = data[:, 1]
            freqs = np.where(voiced == 0, 0.0, freqs)
            times = np.arange(freqs.shape[0]) * time_step
            out_data = np.column_stack((times, freqs))
            
            rel = f0_path.relative_to(base)
            parts = list(rel.parts)
            
            try:
                ref_idx = parts.index(f0_folder_name)
            except ValueError:
                errors += 1
                continue
            
            wav_parts = parts.copy()
            wav_parts[ref_idx] = wav_folder_name
            
            filename = wav_parts[-1]
            if filename.lower().startswith('ref_'):
                wav_filename = 'mic_' + filename[4:]
            else:
                wav_filename = filename
            
            wav_parts[-1] = wav_filename
            wav_path = base / Path(*wav_parts).with_suffix('.wav')
            
            if not wav_path.exists():
                missing_wav += 1
                continue
            
            out_parts = parts.copy()
            out_parts[ref_idx] = wav_folder_name
            out_parts[-1] = wav_filename
            out_path = base / Path(*out_parts).with_suffix('.f0')
            if not wav_path.exists():
                missing_wav += 1
                continue
            
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, out_data, fmt='%.4f\t%.4f', header='time_sec\tfreq_hz', comments='')
            processed += 1
        except Exception as e:
            print(f"Error processing {f0_path}: {e}")
            errors += 1
    
    print(f"Summary: processed={processed}, missing_wav={missing_wav}, errors={errors}")

if __name__ == '__main__':
    BASE_DATA_DIR = 'path/to/PTDB-TUG/SPEECH DATA'
    convert_ptdb_dataset_smart(BASE_DATA_DIR, f0_folder_name='REF', wav_folder_name='MIC', time_step=0.010)