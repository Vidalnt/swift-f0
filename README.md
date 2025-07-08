# SwiftF0

**SwiftF0** is a fast and accurate F0 detector that works by first converting audio into a spectrogram using an STFT, then applying a 2D convolutional neural network to estimate pitch. It’s optimized for:

* ⚡ Real-time analysis (132 ms for 5 seconds of audio on CPU)
* 🎵 Music Information Retrieval
* 🗣️ Speech Analysis

In the [Pitch Detection Benchmark](https://github.com/lars76/pitch-benchmark/), SwiftF0 outperforms algorithms like CREPE in both speed and accuracy. It supports frequencies between **46.875 Hz and 2093.75 Hz** (G1 to C7).

## 🚀 Installation

```bash
pip install swift-f0
```

**Optional dependencies**:

```bash
pip install librosa     # audio loading & resampling
pip install matplotlib  # plotting utilities
```

## ⚡ Quick Start

```python
from swift_f0 import SwiftF0, plot_pitch, export_to_csv

# Initialize the detector
# For speech analysis, consider setting fmin=65 and fmax=400
detector = SwiftF0(fmin=46.875, fmax=2093.75, confidence_threshold=0.9)

# Run pitch detection from an audio file
result = detector.detect_from_file("audio.wav")

# For raw audio arrays (e.g., loaded via librosa or scipy)
# result = detector.detect_from_array(audio_data, sample_rate)

# Visualize and export results
plot_pitch(result, show=False, output_path="pitch.jpg")
export_to_csv(result, "pitch_data.csv")
```

## 📖 API Reference

### `SwiftF0(...)`
```python
SwiftF0(
    confidence_threshold: Optional[float] = 0.9,
    fmin: Optional[float] = 46.875,
    fmax: Optional[float] = 2093.75,
)
```
Initialize the pitch detector. Processes audio at 16kHz with 256-sample hop size. The model always detects pitch across its full range (46.875-2093.75 Hz), but these parameters control which detections are marked as "voiced" in the results.

### `SwiftF0.detect_from_array(...)`
```python
detect_from_array(
    audio_array: np.ndarray,
    sample_rate: int
) -> PitchResult
```
Detect pitch from numpy array. Automatically handles resampling to 16kHz (requires librosa) and converts multi-channel audio to mono by averaging.

### `SwiftF0.detect_from_file(...)`
```python
detect_from_file(
    audio_path: str
) -> PitchResult
```
Detect pitch from audio file. Requires librosa for file loading. Supports any audio format that librosa can read (WAV, MP3, FLAC, etc.).

### `class PitchResult`
```python
@dataclass
class PitchResult:
    pitch_hz: np.ndarray      # F0 estimates (Hz) for each frame
    confidence: np.ndarray    # Model confidence [0.0–1.0] for each frame
    timestamps: np.ndarray    # Frame centers in seconds for each frame
    voicing: np.ndarray       # Boolean voicing decisions for each frame
```
Container for pitch detection results. All arrays have the same length (n_frames). Timestamps are calculated accounting for STFT windowing for accurate frame positioning.

### `plot_pitch(...)`
```python
plot_pitch(
    result: PitchResult,
    output_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 4),
    style: str = "seaborn-v0_8",
) -> None
```
Plot pitch detection results with voicing information. Voiced regions are shown in blue, unvoiced in light gray. Automatically scales y-axis based on detected pitch range. Requires matplotlib.

### `export_to_csv(...)`
```python
export_to_csv(
    result: PitchResult,
    output_path: str
) -> None
```
Export pitch detection results to CSV file with columns: timestamp, pitch_hz, confidence, voiced. Timestamps are formatted to 4 decimal places, pitch to 2 decimal places, confidence to 4 decimal places.

## 📄 Citation

If you use SwiftF0 in your research, please cite:

```bibtex
@software{swiftf0,
    title={SwiftF0: Fast and Accurate Fundamental Frequency Detection},
    author={Lars Nieradzik},
    url={https://github.com/lars76/swift_f0},
    year={2025}
}
```