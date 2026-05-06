"""
EXP-002: Italian PD openSMILE-equivalent eGeMAPS feature extraction
Uses scipy.io.wavfile + numpy to extract acoustic features from WAV files.
Since openSMILE/librosa aren't in the base image, we compute eGeMAPS-like features manually.

Features extracted (per file):
- F0 statistics (mean, std, CV, range, percentiles)
- Jitter (local, RAP, PPQ5)
- Shimmer (local, APQ3, APQ5)
- HNR (harmonics-to-noise ratio)
- Spectral features (centroid, bandwidth, rolloff, flux, slope)
- MFCC statistics (13 coefficients × mean/std)
- Energy/loudness features
- Zero crossing rate
- Voice quality measures

Output: features CSV + metadata CSV uploaded to S3
"""

import os
import json
import time
import gc
import io
import warnings
import struct
warnings.filterwarnings("ignore")

import boto3
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks, spectrogram, lfilter, butter
from scipy.fft import fft, rfft, rfftfreq

SEED = 42
np.random.seed(SEED)

S3_BUCKET = os.environ.get("DATA_S3_BUCKET", "YOUR_DATA_S3_BUCKET")
S3_PREFIX = "competitions/cognitrace-pd/datasets/italian-pd/"
OUTPUT_PREFIX = "competitions/cognitrace-pd/features/exp-002/"
OUTPUTS_DIR = "/work/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

s3 = boto3.client("s3")
start_time = time.time()


# ── Audio Feature Extraction Functions ──────────────────────────────────────

def safe_divide(a, b, default=0.0):
    return a / b if b != 0 else default


def compute_f0_autocorr(signal, sr, fmin=50, fmax=500):
    """Estimate F0 using autocorrelation method."""
    # Windowed autocorrelation
    n = len(signal)
    if n < sr // fmin:
        return 0.0

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    max_lag = min(max_lag, n - 1)

    if min_lag >= max_lag:
        return 0.0

    # Normalize signal
    signal = signal - np.mean(signal)
    norm = np.sum(signal ** 2)
    if norm < 1e-10:
        return 0.0

    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:]  # Take positive lags only
    autocorr = autocorr / norm

    # Find peak in valid lag range
    search = autocorr[min_lag:max_lag+1]
    if len(search) == 0:
        return 0.0

    peak_idx = np.argmax(search)
    lag = peak_idx + min_lag

    if lag == 0:
        return 0.0

    return sr / lag


def compute_f0_track(signal, sr, frame_len=0.03, hop=0.01, fmin=50, fmax=500):
    """Compute F0 track over frames."""
    frame_samples = int(frame_len * sr)
    hop_samples = int(hop * sr)
    n_frames = max(1, (len(signal) - frame_samples) // hop_samples + 1)

    f0_values = []
    for i in range(n_frames):
        start = i * hop_samples
        end = start + frame_samples
        if end > len(signal):
            break
        frame = signal[start:end]
        f0 = compute_f0_autocorr(frame, sr, fmin, fmax)
        if f0 > 0:
            f0_values.append(f0)

    return np.array(f0_values)


def compute_jitter(f0_track):
    """Compute jitter measures from F0 track."""
    if len(f0_track) < 3:
        return {"jitter_local": 0.0, "jitter_rap": 0.0, "jitter_ppq5": 0.0}

    periods = 1.0 / f0_track
    n = len(periods)

    # Local jitter (relative)
    diffs = np.abs(np.diff(periods))
    jitter_local = safe_divide(np.mean(diffs), np.mean(periods))

    # RAP (Relative Average Perturbation) - 3-point average
    if n >= 3:
        rap_diffs = []
        for i in range(1, n - 1):
            avg = (periods[i-1] + periods[i] + periods[i+1]) / 3
            rap_diffs.append(abs(periods[i] - avg))
        jitter_rap = safe_divide(np.mean(rap_diffs), np.mean(periods))
    else:
        jitter_rap = 0.0

    # PPQ5 (5-point Period Perturbation Quotient)
    if n >= 5:
        ppq_diffs = []
        for i in range(2, n - 2):
            avg = np.mean(periods[i-2:i+3])
            ppq_diffs.append(abs(periods[i] - avg))
        jitter_ppq5 = safe_divide(np.mean(ppq_diffs), np.mean(periods))
    else:
        jitter_ppq5 = 0.0

    return {"jitter_local": jitter_local, "jitter_rap": jitter_rap, "jitter_ppq5": jitter_ppq5}


def compute_shimmer(signal, sr, f0_track):
    """Compute shimmer from amplitude of voiced frames."""
    if len(f0_track) < 3:
        return {"shimmer_local": 0.0, "shimmer_apq3": 0.0, "shimmer_apq5": 0.0}

    periods = 1.0 / f0_track
    frame_len = int(np.mean(periods) * sr)
    if frame_len < 10:
        return {"shimmer_local": 0.0, "shimmer_apq3": 0.0, "shimmer_apq5": 0.0}

    # Get amplitude of each period
    hop = frame_len
    amps = []
    for i in range(0, len(signal) - frame_len, hop):
        frame = signal[i:i+frame_len]
        amps.append(np.max(np.abs(frame)))
        if len(amps) >= len(f0_track):
            break

    amps = np.array(amps)
    if len(amps) < 3:
        return {"shimmer_local": 0.0, "shimmer_apq3": 0.0, "shimmer_apq5": 0.0}

    n = len(amps)
    # Local shimmer
    diffs = np.abs(np.diff(amps))
    shimmer_local = safe_divide(np.mean(diffs), np.mean(amps))

    # APQ3
    if n >= 3:
        apq_diffs = []
        for i in range(1, n - 1):
            avg = (amps[i-1] + amps[i] + amps[i+1]) / 3
            apq_diffs.append(abs(amps[i] - avg))
        shimmer_apq3 = safe_divide(np.mean(apq_diffs), np.mean(amps))
    else:
        shimmer_apq3 = 0.0

    # APQ5
    if n >= 5:
        apq_diffs = []
        for i in range(2, n - 2):
            avg = np.mean(amps[i-2:i+3])
            apq_diffs.append(abs(amps[i] - avg))
        shimmer_apq5 = safe_divide(np.mean(apq_diffs), np.mean(amps))
    else:
        shimmer_apq5 = 0.0

    return {"shimmer_local": shimmer_local, "shimmer_apq3": shimmer_apq3, "shimmer_apq5": shimmer_apq5}


def compute_hnr(signal, sr, f0):
    """Estimate harmonics-to-noise ratio."""
    if f0 <= 0:
        return 0.0

    period = int(sr / f0)
    if period < 2 or period > len(signal) // 2:
        return 0.0

    n_periods = len(signal) // period
    if n_periods < 2:
        return 0.0

    # Average period waveform
    periods_matrix = []
    for i in range(n_periods):
        start = i * period
        end = start + period
        if end <= len(signal):
            periods_matrix.append(signal[start:end])

    if len(periods_matrix) < 2:
        return 0.0

    periods_matrix = np.array(periods_matrix)
    avg_period = np.mean(periods_matrix, axis=0)

    harmonic_energy = np.sum(avg_period ** 2) * len(periods_matrix)
    total_energy = np.sum(signal[:n_periods*period] ** 2)
    noise_energy = total_energy - harmonic_energy

    if noise_energy <= 0:
        return 40.0  # Cap at 40 dB

    hnr = 10 * np.log10(safe_divide(harmonic_energy, abs(noise_energy), default=1.0))
    return np.clip(hnr, -20, 40)


def compute_spectral_features(signal, sr, n_fft=2048):
    """Compute spectral features: centroid, bandwidth, rolloff, flux, slope."""
    # Compute magnitude spectrum
    S = np.abs(rfft(signal * np.hanning(len(signal)), n=n_fft))
    freqs = rfftfreq(n_fft, 1.0/sr)

    # Spectral centroid
    total = np.sum(S)
    if total < 1e-10:
        return {"spec_centroid": 0, "spec_bandwidth": 0, "spec_rolloff": 0, "spec_slope": 0}

    centroid = np.sum(freqs * S) / total

    # Spectral bandwidth
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * S) / total)

    # Spectral rolloff (85%)
    cumsum = np.cumsum(S)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    rolloff = freqs[min(rolloff_idx, len(freqs)-1)]

    # Spectral slope (linear regression of log magnitude)
    log_S = np.log10(S + 1e-10)
    if len(freqs) > 1:
        slope = np.polyfit(freqs, log_S, 1)[0]
    else:
        slope = 0.0

    return {
        "spec_centroid": centroid,
        "spec_bandwidth": bandwidth,
        "spec_rolloff": rolloff,
        "spec_slope": slope
    }


def compute_mfcc(signal, sr, n_mfcc=13, n_fft=2048, n_mels=40):
    """Compute MFCC using manual implementation."""
    # Pre-emphasis
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # Frame the signal
    frame_len = n_fft
    hop = frame_len // 2
    n_frames = max(1, (len(emphasized) - frame_len) // hop + 1)

    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(emphasized):
            frames[i, :len(emphasized)-start] = emphasized[start:]
        else:
            frames[i] = emphasized[start:end]

    # Window
    window = np.hanning(frame_len)
    frames *= window

    # FFT
    mag = np.abs(rfft(frames, n=n_fft, axis=1))
    power = mag ** 2 / n_fft

    # Mel filterbank
    fmin_mel = 0
    fmax_mel = 2595 * np.log10(1 + (sr/2) / 700)
    mel_points = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left = bin_points[m-1]
        center = bin_points[m]
        right = bin_points[m+1]

        for k in range(left, center):
            if center > left:
                fbank[m-1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fbank[m-1, k] = (right - k) / (right - center)

    # Apply filterbank
    mel_spec = np.dot(power, fbank.T)
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
    log_mel = np.log(mel_spec)

    # DCT (Type-II)
    n_coefs = log_mel.shape[1]
    dct_matrix = np.zeros((n_mfcc, n_coefs))
    for i in range(n_mfcc):
        for j in range(n_coefs):
            dct_matrix[i, j] = np.cos(np.pi * i * (2*j + 1) / (2 * n_coefs))

    mfccs = np.dot(log_mel, dct_matrix.T)

    # Return statistics per coefficient
    result = {}
    for i in range(n_mfcc):
        result[f"mfcc_{i}_mean"] = np.mean(mfccs[:, i])
        result[f"mfcc_{i}_std"] = np.std(mfccs[:, i])

    return result


def compute_energy_features(signal, sr):
    """Compute energy and loudness features."""
    # RMS energy
    rms = np.sqrt(np.mean(signal ** 2))
    # Log energy
    log_energy = np.log(np.sum(signal ** 2) + 1e-10)
    # Zero crossing rate
    zcr = np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal))
    # Peak amplitude
    peak = np.max(np.abs(signal))

    return {
        "rms_energy": rms,
        "log_energy": log_energy,
        "zcr": zcr,
        "peak_amplitude": peak
    }


def extract_features(signal, sr):
    """Extract all features from a single audio signal."""
    # Normalize
    signal = signal.astype(np.float64)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    features = {}

    # Duration
    features["duration_s"] = len(signal) / sr

    # F0 analysis
    f0_track = compute_f0_track(signal, sr)
    if len(f0_track) > 0:
        features["f0_mean"] = np.mean(f0_track)
        features["f0_std"] = np.std(f0_track)
        features["f0_cv"] = safe_divide(np.std(f0_track), np.mean(f0_track))
        features["f0_min"] = np.min(f0_track)
        features["f0_max"] = np.max(f0_track)
        features["f0_range"] = np.max(f0_track) - np.min(f0_track)
        features["f0_p25"] = np.percentile(f0_track, 25)
        features["f0_p75"] = np.percentile(f0_track, 75)
        features["f0_iqr"] = features["f0_p75"] - features["f0_p25"]
        features["voiced_fraction"] = len(f0_track) / max(1, len(signal) // int(0.01 * sr))
    else:
        for k in ["f0_mean", "f0_std", "f0_cv", "f0_min", "f0_max", "f0_range",
                   "f0_p25", "f0_p75", "f0_iqr", "voiced_fraction"]:
            features[k] = 0.0

    # Jitter
    jitter = compute_jitter(f0_track)
    features.update(jitter)

    # Shimmer
    shimmer = compute_shimmer(signal, sr, f0_track)
    features.update(shimmer)

    # HNR
    f0_mean = features.get("f0_mean", 0)
    features["hnr"] = compute_hnr(signal, sr, f0_mean)

    # Spectral features (compute on frames and aggregate)
    frame_len = min(2048, len(signal))
    hop = frame_len // 2
    n_frames = max(1, (len(signal) - frame_len) // hop + 1)

    spec_feats_list = []
    for i in range(min(n_frames, 100)):  # Cap at 100 frames for speed
        start = i * hop
        end = start + frame_len
        if end > len(signal):
            break
        frame = signal[start:end]
        sf = compute_spectral_features(frame, sr)
        spec_feats_list.append(sf)

    if spec_feats_list:
        for key in spec_feats_list[0]:
            vals = [sf[key] for sf in spec_feats_list]
            features[f"{key}_mean"] = np.mean(vals)
            features[f"{key}_std"] = np.std(vals)

    # MFCC
    mfcc = compute_mfcc(signal, sr)
    features.update(mfcc)

    # Energy features
    energy = compute_energy_features(signal, sr)
    features.update(energy)

    return features


# ── Main Pipeline ───────────────────────────────────────────────────────────

print("Listing Italian PD WAV files in S3...")
all_files = []
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.lower().endswith(".wav"):
            all_files.append(key)

print(f"Found {len(all_files)} WAV files")

# Parse subject info from filenames
# Expected format varies — discover structure
if all_files:
    print(f"Sample filenames: {[f.split('/')[-1] for f in all_files[:5]]}")

# Extract features for all files
results = []
errors = []

for idx, s3_key in enumerate(all_files):
    filename = s3_key.split("/")[-1]

    if idx % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Processing {idx+1}/{len(all_files)} ({elapsed:.0f}s elapsed): {filename}")

    try:
        # Download WAV to memory
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        wav_bytes = obj["Body"].read()

        # Read WAV
        sr, data = wavfile.read(io.BytesIO(wav_bytes))

        # Handle stereo → mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Extract features
        feats = extract_features(data, sr)
        feats["filename"] = filename
        feats["s3_key"] = s3_key
        feats["sample_rate"] = sr
        feats["n_samples"] = len(data)

        # Try to parse subject/condition from filename
        # Common patterns: "SubjectXX_TaskName.wav", "PDxx_task.wav", "CTxx_task.wav"
        name_lower = filename.lower()
        feats["has_pd_in_name"] = 1 if "pd" in name_lower else 0
        feats["has_ctrl_in_name"] = 1 if ("ctrl" in name_lower or "hc" in name_lower or
                                           "control" in name_lower or "young" in name_lower) else 0

        results.append(feats)

        del wav_bytes, data
        if idx % 100 == 0:
            gc.collect()

    except Exception as e:
        errors.append({"filename": filename, "error": str(e)})
        if idx < 5:
            print(f"  ERROR on {filename}: {e}")

print(f"\nExtraction complete: {len(results)} succeeded, {len(errors)} failed")

# ── Save Results ────────────────────────────────────────────────────────────

df_features = pd.DataFrame(results)
print(f"Feature matrix shape: {df_features.shape}")
print(f"Feature columns: {len([c for c in df_features.columns if c not in ['filename', 's3_key']])}")

# Save locally
features_path = os.path.join(OUTPUTS_DIR, "italian_pd_egemaps_features.csv")
df_features.to_csv(features_path, index=False)

# Upload to S3 for downstream experiments
csv_buffer = io.BytesIO()
df_features.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)
s3.put_object(
    Bucket=S3_BUCKET,
    Key=OUTPUT_PREFIX + "italian_pd_egemaps_features.csv",
    Body=csv_buffer.getvalue()
)
print(f"Uploaded features to s3://{S3_BUCKET}/{OUTPUT_PREFIX}italian_pd_egemaps_features.csv")

# Save errors
if errors:
    errors_path = os.path.join(OUTPUTS_DIR, "extraction_errors.json")
    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=2)

# Summary statistics
numeric_cols = [c for c in df_features.columns if df_features[c].dtype in [np.float64, np.float32, np.int64]]
summary = df_features[numeric_cols].describe().to_dict()

# Save summary
with open(os.path.join(OUTPUTS_DIR, "feature_summary.json"), "w") as f:
    json.dump({
        "n_files": len(results),
        "n_errors": len(errors),
        "n_features": len(numeric_cols),
        "feature_names": numeric_cols,
        "sample_filenames": [r["filename"] for r in results[:10]],
    }, f, indent=2)

# Compute metrics for the run
elapsed = round(time.time() - start_time, 1)

# For feature extraction, metrics are about coverage and quality
metrics_list = [
    {"metric_name": "files_processed", "metric_value": len(results), "split_name": "overall", "is_primary": True},
    {"metric_name": "files_failed", "metric_value": len(errors), "split_name": "overall", "is_primary": False},
    {"metric_name": "n_features_extracted", "metric_value": len(numeric_cols), "split_name": "overall", "is_primary": False},
    {"metric_name": "extraction_rate_pct", "metric_value": round(100 * len(results) / max(1, len(all_files)), 1), "split_name": "overall", "is_primary": False},
    {"metric_name": "pipeline_latency_seconds", "metric_value": elapsed, "split_name": "overall", "is_primary": False},
]

# Add acoustic biomarker means (PD vs control if we can distinguish)
if "f0_mean" in df_features.columns:
    metrics_list.append({"metric_name": "f0_mean_overall", "metric_value": round(df_features["f0_mean"].mean(), 2), "split_name": "overall", "is_primary": False})
if "jitter_local" in df_features.columns:
    metrics_list.append({"metric_name": "jitter_mean", "metric_value": round(df_features["jitter_local"].mean(), 6), "split_name": "overall", "is_primary": False})
if "shimmer_local" in df_features.columns:
    metrics_list.append({"metric_name": "shimmer_mean", "metric_value": round(df_features["shimmer_local"].mean(), 6), "split_name": "overall", "is_primary": False})
if "hnr" in df_features.columns:
    metrics_list.append({"metric_name": "hnr_mean", "metric_value": round(df_features["hnr"].mean(), 2), "split_name": "overall", "is_primary": False})

with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_list, f, indent=2)

print(f"\n{'='*60}")
print(f"EXP-002 COMPLETE in {elapsed}s")
print(f"  Files processed: {len(results)}/{len(all_files)}")
print(f"  Features extracted: {len(numeric_cols)}")
print(f"  Output: s3://{S3_BUCKET}/{OUTPUT_PREFIX}italian_pd_egemaps_features.csv")
print(f"{'='*60}")
