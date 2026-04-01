'''
Load the image
Detect the ECG trace region (crop out grid, labels, borders)
Extract the pixel y-coordinates of the signal line
Convert pixel positions → millivolt values
Resample to a standard frequency (360Hz)
Feed into your existing peak detection and CNN pipeline
'''

import numpy as np
import cv2
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d


def load_ecg_image(image_path):
    """Load an ECG image and return as numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img


def preprocess_image(img):
    """
    Convert to grayscale, denoise, and enhance contrast.
    Returns a cleaned grayscale image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (contrast limited adaptive histogram equalization)
    # This handles images with poor contrast or uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    return denoised


def remove_grid(gray_img):
    """
    Remove the ECG grid lines, leaving only the signal trace.
    Works by detecting and subtracting the periodic grid pattern.
    """
    # Threshold to binary
    _, binary = cv2.threshold(
        gray_img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove horizontal grid lines using morphological operations
    # Horizontal kernel detects lines wider than the signal trace
    h, w = binary.shape
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (w // 20, 1)
    )
    horizontal_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
    )

    # Remove vertical grid lines
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, h // 20)
    )
    vertical_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1
    )

    # Subtract grid from binary image
    grid = cv2.add(horizontal_lines, vertical_lines)
    signal_only = cv2.subtract(binary, grid)

    # Clean up small noise pixels
    cleanup_kernel = np.ones((2, 2), np.uint8)
    signal_only = cv2.morphologyEx(
        signal_only, cv2.MORPH_OPEN, cleanup_kernel
    )

    return signal_only


def extract_signal_trace(binary_img):
    """
    Convert binary image of ECG trace to 1D signal array.
    For each column, finds the y-position of the signal pixel.
    Returns array of y-positions (one per column).
    """
    h, w = binary_img.shape
    y_positions = np.full(w, np.nan)

    for col in range(w):
        column = binary_img[:, col]
        signal_pixels = np.where(column > 0)[0]

        if len(signal_pixels) == 0:
            continue

        # Use the centroid of signal pixels in this column
        # handles thick traces better than just taking the first pixel
        y_positions[col] = np.mean(signal_pixels)

    return y_positions


def interpolate_gaps(y_positions):
    """
    Fill in NaN gaps caused by missing pixels using linear interpolation.
    """
    x = np.arange(len(y_positions))
    valid = ~np.isnan(y_positions)

    if valid.sum() < 10:
        raise ValueError(
            "Too few signal pixels detected. "
            "Image may be too noisy or the trace is not visible."
        )

    interpolator = interp1d(
        x[valid], y_positions[valid],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )

    return interpolator(x)


def normalize_to_mv(y_positions, img_height):
    """
    Convert pixel y-positions to approximate millivolt values.
    ECG standard: 10mm = 1mV. We estimate scale from image height.
    Y is inverted in image coordinates (0 = top), so we flip it.
    """
    # Flip y-axis (image y=0 is top, ECG positive is up)
    flipped = img_height - y_positions

    # Normalize to roughly -2 to +2 mV range
    # (standard ECG display range)
    mid   = np.median(flipped)
    scale = (img_height * 0.1)   # rough pixel-to-mV scale

    mv_signal = (flipped - mid) / scale

    return mv_signal


def resample_to_target_fs(signal_array, source_fs, target_fs=360):
    """
    Resample signal from image pixel rate to target sampling frequency.
    source_fs is estimated from image width and assumed recording duration.
    """
    n_samples_target = int(len(signal_array) * target_fs / source_fs)
    resampled = scipy_signal.resample(signal_array, n_samples_target)
    return resampled


def extract_signal_from_image(image_input, recording_duration_sec=10):
    """
    Full pipeline: image → clean numpy signal array at 360Hz.

    Args:
        image_input: file path string OR numpy array (for Streamlit uploads)
        recording_duration_sec: estimated duration shown in the image

    Returns:
        signal (np.array): signal in mV at 360Hz
        fs (int): sampling rate (always 360)
    """
    # Load image
    if isinstance(image_input, np.ndarray):
        img = image_input
    else:
        img = load_ecg_image(image_input)

    h, w = img.shape[:2]

    # Process
    gray        = preprocess_image(img)
    signal_only = remove_grid(gray)
    y_positions = extract_signal_trace(signal_only)
    y_filled    = interpolate_gaps(y_positions)
    mv_signal   = normalize_to_mv(y_filled, h)

    # Estimate source sampling rate from image width and duration
    # e.g. a 10-second strip that is 1800px wide = 180px/sec
    source_fs = w / recording_duration_sec

    # Resample to 360Hz to match what the CNN was trained on
    resampled = resample_to_target_fs(mv_signal, source_fs, target_fs=360)

    return resampled.astype(np.float32), 360