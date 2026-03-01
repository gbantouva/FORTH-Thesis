"""
Step 3a — Node Feature Extraction
===================================
Computes per-channel, per-epoch features to use as GNN node features.

For each epoch (n_epochs, 19, 1024) and each channel we extract:

  Band Power (5 features)
  ─────────────────────────
  delta   0.5–4  Hz
  theta   4–8    Hz
  alpha   8–15   Hz
  beta    15–30  Hz
  gamma   30–45  Hz

  Hjorth Parameters (3 features)
  ─────────────────────────────────
  activity   = variance of signal
  mobility   = std(first derivative) / std(signal)
  complexity = mobility(first derivative) / mobility(signal)

  Statistical Features (4 features)
  ─────────────────────────────────
  mean, variance, skewness, kurtosis

Total: 12 features per channel per epoch
Output shape per subject: (n_epochs, 19, 12)

Usage:
    python step3a_node_features.py \\
        --epochdir  path/to/preprocessed_epochs \\
        --outputdir path/to/node_features
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
FS = 256
EPOCH_SAMPLES = 1024

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2',
]

# Frequency bands (same as Step 2)
BANDS = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 15.0),
    'beta':  (15.0, 30.0),
    'gamma': (30.0, 45.0),
}

FEATURE_NAMES = [
    # Band power
    'bp_delta', 'bp_theta', 'bp_alpha', 'bp_beta', 'bp_gamma',
    # Hjorth
    'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
    # Statistical
    'stat_mean', 'stat_variance', 'stat_skewness', 'stat_kurtosis',
]

N_FEATURES = len(FEATURE_NAMES)  # 12


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def band_power(x, fs, f_low, f_high):
    """
    Compute mean power in a frequency band using Welch's method.

    Parameters
    ----------
    x      : 1D array (n_samples,)
    fs     : sampling frequency (Hz)
    f_low  : lower band edge (Hz)
    f_high : upper band edge (Hz)

    Returns
    -------
    float : mean power in band (uV² or arbitrary units)
    """
    nperseg = min(256, len(x))         # window length for Welch
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=nperseg)
    idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(psd[idx]))


def hjorth_parameters(x):
    """
    Compute Hjorth activity, mobility, and complexity.

    Activity   = variance of signal          → signal power
    Mobility   = sqrt(var(x') / var(x))     → mean frequency proxy
    Complexity = mobility(x') / mobility(x) → bandwidth proxy

    Parameters
    ----------
    x : 1D array (n_samples,)

    Returns
    -------
    activity, mobility, complexity : floats
    """
    var_x = np.var(x)
    if var_x < 1e-12:
        return 0.0, 0.0, 0.0

    dx  = np.diff(x)
    var_dx = np.var(dx)

    activity   = float(var_x)
    mobility   = float(np.sqrt(var_dx / var_x))

    if var_dx < 1e-12 or mobility < 1e-12:
        complexity = 0.0
    else:
        ddx     = np.diff(dx)
        var_ddx = np.var(ddx)
        mob_dx  = float(np.sqrt(var_ddx / var_dx))
        complexity = float(mob_dx / mobility)

    return activity, mobility, complexity


def statistical_features(x):
    """
    Mean, variance, skewness, kurtosis of the signal.

    Parameters
    ----------
    x : 1D array (n_samples,)

    Returns
    -------
    mean, variance, skewness, kurtosis : floats
    """
    return (
        float(np.mean(x)),
        float(np.var(x)),
        float(skew(x)),
        float(kurtosis(x)),   # Fisher definition (normal → 0)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE CHANNEL FEATURE VECTOR
# ══════════════════════════════════════════════════════════════════════════════

def extract_channel_features(x, fs):
    """
    Extract all 12 features for a single channel signal.

    Parameters
    ----------
    x  : 1D array (n_samples,)
    fs : sampling frequency

    Returns
    -------
    features : 1D array (12,)  in the order of FEATURE_NAMES
    """
    features = []

    # ── Band power ────────────────────────────────────────────────────────
    for band_name, (f_lo, f_hi) in BANDS.items():
        features.append(band_power(x, fs, f_lo, f_hi))

    # ── Hjorth ───────────────────────────────────────────────────────────
    act, mob, comp = hjorth_parameters(x)
    features.extend([act, mob, comp])

    # ── Statistical ───────────────────────────────────────────────────────
    mn, var, sk, kurt = statistical_features(x)
    features.extend([mn, var, sk, kurt])

    return np.array(features, dtype=np.float32)  # (12,)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SINGLE EPOCH FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def extract_epoch_features(epoch, fs):
    """
    Extract features for all channels in one epoch.

    Parameters
    ----------
    epoch : ndarray (n_channels, n_samples)  e.g. (19, 1024)
    fs    : sampling frequency

    Returns
    -------
    features : ndarray (n_channels, 12)
    """
    n_channels = epoch.shape[0]
    features   = np.zeros((n_channels, N_FEATURES), dtype=np.float32)

    for ch in range(n_channels):
        features[ch] = extract_channel_features(epoch[ch], fs)

    return features


# ══════════════════════════════════════════════════════════════════════════════
# 4. PROCESS ONE SUBJECT
# ══════════════════════════════════════════════════════════════════════════════

def process_subject(epochs_file, output_dir, fs):
    """
    Compute node features for all epochs of one subject.

    Saves:
      subject_XX_node_features.npy  — (n_epochs, 19, 12)
      subject_XX_node_features_normalized.npy — z-score normalized version

    Returns
    -------
    dict with summary statistics
    """
    subject_name = epochs_file.stem.replace('_epochs', '')

    # Load epochs: (n_epochs, 19, 1024)
    epochs   = np.load(epochs_file)
    n_epochs = len(epochs)

    # Load labels (for reference, not used in computation)
    labels_file = epochs_file.parent / f"{subject_name}_labels.npy"
    labels = np.load(labels_file) if labels_file.exists() else None

    # Allocate output: (n_epochs, 19, 12)
    all_features = np.zeros((n_epochs, epochs.shape[1], N_FEATURES),
                             dtype=np.float32)

    for ep_idx in range(n_epochs):
        all_features[ep_idx] = extract_epoch_features(epochs[ep_idx], fs)

    # ── Check for NaN/Inf ────────────────────────────────────────────────
    n_bad = int(np.sum(~np.isfinite(all_features)))
    if n_bad > 0:
        print(f"  ⚠️  {subject_name}: {n_bad} non-finite values → replaced with 0")
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Save raw features ─────────────────────────────────────────────────
    raw_path = output_dir / f"{subject_name}_node_features.npy"
    np.save(raw_path, all_features)  # (n_epochs, 19, 12)

    # ── Z-score normalization PER FEATURE across all epochs ──────────────
    # Shape: (n_epochs, 19, 12) → normalize over axis 0 (epochs)
    # Each feature independently normalized so mean=0, std=1 per channel
    feat_mean = all_features.mean(axis=0, keepdims=True)   # (1, 19, 12)
    feat_std  = all_features.std(axis=0,  keepdims=True)   # (1, 19, 12)
    feat_std[feat_std < 1e-8] = 1.0                         # avoid divide by zero

    normalized = (all_features - feat_mean) / feat_std     # (n_epochs, 19, 12)
    norm_path  = output_dir / f"{subject_name}_node_features_normalized.npy"
    np.save(norm_path, normalized)

    # ── Stats ─────────────────────────────────────────────────────────────
    summary = {
        'subject':    subject_name,
        'n_epochs':   int(n_epochs),
        'n_channels': int(epochs.shape[1]),
        'n_features': N_FEATURES,
        'shape':      list(all_features.shape),
        'n_bad_values': n_bad,
        'feature_means': feat_mean[0].tolist(),   # (19, 12)
        'feature_stds':  feat_std[0].tolist(),    # (19, 12)
    }

    if labels is not None:
        n_pre = int((labels == 0).sum())
        n_ict = int((labels == 1).sum())
        summary['n_pre_ictal'] = n_pre
        summary['n_ictal']     = n_ict

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract node features (band power + Hjorth + stats) for GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--epochdir',  required=True,
                        help='Directory with subject_XX_epochs.npy files')
    parser.add_argument('--outputdir', required=True,
                        help='Output directory for node feature files')
    parser.add_argument('--fs',  type=int, default=256,
                        help='Sampling frequency (default: 256)')
    args = parser.parse_args()

    epoch_dir  = Path(args.epochdir)
    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_files = sorted(epoch_dir.glob('subject_*_epochs.npy'))

    print("=" * 72)
    print("STEP 3a — NODE FEATURE EXTRACTION")
    print("=" * 72)
    print(f"  Input:      {epoch_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Subjects:   {len(epoch_files)}")
    print(f"  Fs:         {args.fs} Hz")
    print(f"  Features:   {N_FEATURES} per channel")
    print()
    print("  Features extracted:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"    [{i:02d}] {name}")
    print("=" * 72)

    all_summaries = []

    for ep_file in tqdm(epoch_files, desc="Subjects", unit="subject"):
        summary = process_subject(ep_file, output_dir, args.fs)
        all_summaries.append(summary)
        tqdm.write(
            f"  ✅ {summary['subject']} — "
            f"{summary['n_epochs']} epochs  "
            f"shape {summary['shape']}"
        )

    # ── Global metadata ───────────────────────────────────────────────────
    global_meta = {
        'description':   'Node features for GNN — per channel per epoch',
        'n_subjects':    len(all_summaries),
        'n_features':    N_FEATURES,
        'feature_names': FEATURE_NAMES,
        'feature_groups': {
            'band_power':  ['bp_delta', 'bp_theta', 'bp_alpha', 'bp_beta', 'bp_gamma'],
            'hjorth':      ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'],
            'statistical': ['stat_mean', 'stat_variance', 'stat_skewness', 'stat_kurtosis'],
        },
        'output_shapes': {
            'subject_XX_node_features.npy':            '(n_epochs, 19, 12)  — raw',
            'subject_XX_node_features_normalized.npy': '(n_epochs, 19, 12)  — z-score per channel per feature',
        },
        'normalization': 'z-score per feature per channel across epochs (per subject)',
        'sampling_rate': args.fs,
        'channels':      CHANNEL_NAMES,
        'subjects':      all_summaries,
    }

    meta_path = output_dir / 'node_features_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(global_meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    total_epochs = sum(s['n_epochs'] for s in all_summaries)

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(f"  Subjects processed:  {len(all_summaries)}")
    print(f"  Total epochs:        {total_epochs:,}")
    print(f"  Output shape:        (n_epochs, 19, {N_FEATURES})")
    print()
    print("  Files per subject:")
    print("    subject_XX_node_features.npy            ← raw features")
    print("    subject_XX_node_features_normalized.npy ← z-score normalized")
    print(f"  node_features_metadata.json")
    print()
    print("  Next steps:")
    print("    Step 3b: Baseline ML (SVM/RF on flattened features)")
    print("    Step 3c: Build PyTorch Geometric graphs")
    print("    Step 3d: Train GCN (supervised)")
    print("=" * 72)


if __name__ == '__main__':
    main()
