import numpy as np
import torch
import torch.nn as nn
import neurokit2 as nk
import wfdb
from collections import Counter
from image_processing import extract_signal_from_image

# ── Device setup ──────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Color and label maps ───────────────────────────────────────
CONDITION_COLORS = {
    'Normal':  '#2ecc71',
    'PVC':     '#e74c3c',
    'PAC':     '#e67e22',
    'Fusion':  '#9b59b6',
    'Paced':   '#3498db',
    'Unknown': '#95a5a6',
}

CONDITION_LABELS = {
    'Normal':  'Normal beat',
    'PVC':     'PVC — Premature Ventricular Contraction',
    'PAC':     'PAC — Premature Atrial Contraction',
    'Fusion':  'Fusion beat',
    'Paced':   'Paced beat',
    'Unknown': 'Unknown',
}

CONDITION_INFO = {
    'PVC': (
        "A premature ventricular contraction originates in the ventricles "
        "rather than the sinoatrial node. Occasional PVCs are common and "
        "often benign, but frequent PVCs may indicate underlying heart disease."
    ),
    'PAC': (
        "A premature atrial contraction originates in the atria earlier than "
        "expected. Usually benign but can trigger arrhythmias in susceptible patients."
    ),
    'Fusion': (
        "A fusion beat occurs when a normal and ventricular beat happen "
        "simultaneously, producing a hybrid waveform."
    ),
    'Paced': (
        "A paced beat is triggered by an implanted pacemaker rather than "
        "the heart's natural conduction system."
    ),
}


# ── CNN model definition ───────────────────────────────────────
class ECG_CNN(nn.Module):
    def __init__(self, num_classes):
        super(ECG_CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 54, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Load model from checkpoint ─────────────────────────────────
def load_model(model_path, device):
    checkpoint  = torch.load(model_path,
                             map_location=device,
                             weights_only=False)
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['encoder_classes']

    model = ECG_CNN(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, class_names


# ── Signal processing ──────────────────────────────────────────
def process_record(record_path):
    """Load and clean a wfdb record. Returns signal, fs, r_peaks."""
    record  = wfdb.rdrecord(record_path)
    fs      = record.fs
    signal  = record.p_signal[:, 0]
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)

    peaks_dict = nk.ecg_peaks(cleaned, sampling_rate=fs)
    r_peaks    = peaks_dict[1]['ECG_R_Peaks']

    return cleaned, fs, r_peaks


def process_uploaded_signal(signal, fs):
    """Clean a raw signal array and detect R-peaks."""
    cleaned    = nk.ecg_clean(signal, sampling_rate=fs)
    peaks_dict = nk.ecg_peaks(cleaned, sampling_rate=fs)
    r_peaks    = peaks_dict[1]['ECG_R_Peaks']
    return cleaned, r_peaks


# ── Inference ──────────────────────────────────────────────────
def predict_beats(signal, r_peaks, fs, model, class_names, device,
                  batch_size=64):
    """
    Run CNN inference on every beat using batched processing.
    Significantly faster than one-at-a-time inference.
    """
    pre  = int(0.2 * fs)
    post = int(0.4 * fs)

    # Build all valid beat windows first
    valid_peaks = []
    beat_windows = []

    for peak in r_peaks:
        if peak - pre < 0 or peak + post >= len(signal):
            continue

        beat = signal[peak - pre : peak + post]
        mean = beat.mean()
        std  = beat.std() + 1e-8
        beat = (beat - mean) / std

        valid_peaks.append(peak)
        beat_windows.append(beat)

    if len(beat_windows) == 0:
        return []

    # Convert to numpy array for easy batching
    beat_array = np.array(beat_windows, dtype=np.float32)
    results    = []

    # Process in batches
    for batch_start in range(0, len(beat_array), batch_size):
        batch_beats = beat_array[batch_start : batch_start + batch_size]
        batch_peaks = valid_peaks[batch_start : batch_start + batch_size]

        # Shape: (batch_size, 1, 216)
        tensor = torch.FloatTensor(batch_beats).unsqueeze(1).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()

        for i, (peak, prob) in enumerate(zip(batch_peaks, probs)):
            pred_idx   = prob.argmax()
            confidence = float(prob[pred_idx])
            prediction = class_names[pred_idx]

            results.append({
                'sample_start': int(peak - pre),
                'sample_end':   int(peak + post),
                'r_peak':       int(peak),
                'prediction':   prediction,
                'confidence':   confidence,
                'all_probs':    {c: float(p)
                                 for c, p in zip(class_names, prob)}
            })

    return results

# ── Summary helpers ────────────────────────────────────────────
def get_flagged(results, confidence_threshold=0.7):
    return [r for r in results
            if r['prediction'] != 'Normal'
            and r['confidence'] >= confidence_threshold]


def get_summary_counts(results, confidence_threshold=0.7):
    flagged = get_flagged(results, confidence_threshold)
    return Counter(r['prediction'] for r in flagged)

# Image Processing
def process_image_upload(image_array, recording_duration_sec=10):
    """
    Full pipeline for image input:
    image → signal → clean → r-peaks
    """
    # Extract signal from image
    raw_signal, fs = extract_signal_from_image(
        image_array,
        recording_duration_sec=recording_duration_sec
    )

    # Clean and detect peaks using existing pipeline
    cleaned, r_peaks = process_uploaded_signal(raw_signal, fs)

    return cleaned, fs, r_peaks