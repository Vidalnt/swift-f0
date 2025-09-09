import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_vocadito_dataset(csv_input_dir: str, f0_output_dir: str):
    inp = Path(csv_input_dir)
    out = Path(f0_output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_files = list(inp.rglob('*.csv'))
    if not csv_files:
        print(f"No .csv files found in: {csv_input_dir}")
        return

    for p in tqdm(csv_files, desc="Processing"):
        try:
            data = np.loadtxt(p, delimiter=',', dtype=float)
            rel = p.relative_to(inp).with_suffix('.f0')
            stem_without_f0 = rel.stem.replace('_f0', '')
            rel_clean = rel.with_name(stem_without_f0 + rel.suffix)
            out_path = out / rel_clean
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, data, fmt='%.6f\t%.3f', comments='')
        except Exception as e:
            print(f"Error processing {p}: {e}")

if __name__ == '__main__':
    CSV_INPUT_DIR = 'path/to/vocadito/Annotations/F0'
    F0_OUTPUT_DIR = 'path/to/vocadito/Audio'
    convert_vocadito_dataset(CSV_INPUT_DIR, F0_OUTPUT_DIR)
