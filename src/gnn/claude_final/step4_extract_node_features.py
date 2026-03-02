"""
Step 4 — Extract Node Features for GNN
=======================================
For each epoch of each subject, computes a 9-dimensional feature vector
per EEG channel (node):

    [0]  Delta band power    (0.5 – 4 Hz)
    [1]  Theta band power    (4  – 8 Hz)
    [2]  Alpha band power    (8  – 15 Hz)
    [3]  Beta  band power    (15 – 30 Hz)
    [4]  Gamma band power    (30 – 45 Hz)
    [5]  DTF outflow         (sum of integrated-band DTF column)
    [6]  DTF inflow          (sum of integrated-band DTF row)
    [7]  PDC outflow         (sum of integrated-band PDC column)
    [8]  PDC inflow          (sum of integrated-band PDC row)

Output per subject:
    subject_XX_node_features.npy   shape (n_valid_epochs, 19, 9)

n_valid_epochs matches the number of epochs in subject_XX_graphs.npz
(i.e. epochs that passed the VAR stability check in step 2).

Usage:
    python step4_extract_node_features.py \
        --epochs_dir   F:/FORTH_Final_Thesis/FORTH-Thesis/final_preprocessed_epochs \
        --connectivity_dir F:/FORTH_Final_Thesis/FORTH-Thesis/connectivity \
        --output_dir   F:/FORTH_Final_Thesis/FORTH-Thesis/node_features \
        --subject_ids  1 2 3
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.signal import welch
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────
FS            = 256          # Hz
NFFT          = 512          # FFT length for Welch
N_CHANNELS    = 19
N_FEATURES    = 9

BANDS = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 15.0),
    'beta' : (15.0, 30.0),
    'gamma': (30.0, 45.0),
}
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']  # features 0-4


# ── helpers ───────────────────────────────────────────────────────────────────

def bandpower_welch(signal_1d: np.ndarray, flo: float, fhi: float) -> float:
    """
    Estimate band power using Welch's method.

    signal_1d : (n_samples,) single-channel EEG
    Returns a scalar (mean PSD in the band, in units^2/Hz).
    """
    freqs, psd = welch(signal_1d, fs=FS, nperseg=NFFT, noverlap=NFFT // 2)
    idx = np.logical_and(freqs >= flo, freqs <= fhi)
    # integrate over the band using the trapezoidal rule
    bp = np.trapz(psd[idx], freqs[idx])
    return float(bp)


def compute_band_powers(epoch: np.ndarray) -> np.ndarray:
    """
    epoch : (19, 1024)  — one preprocessed EEG epoch
    Returns : (19, 5)   — band power per channel per band
    """
    bp = np.zeros((N_CHANNELS, len(BAND_ORDER)), dtype=np.float32)
    for ch in range(N_CHANNELS):
        for b_idx, band in enumerate(BAND_ORDER):
            flo, fhi = BANDS[band]
            bp[ch, b_idx] = bandpower_welch(epoch[ch], flo, fhi)
    return bp


def compute_connectivity_summaries(dtf: np.ndarray, pdc: np.ndarray) -> np.ndarray:
    """
    dtf, pdc : (19, 19)  — integrated-band connectivity matrices
                           convention: matrix[i, j] = j → i

    Returns : (19, 4)
        col 0 : DTF outflow  per channel  (column sum = how much j drives others)
        col 1 : DTF inflow   per channel  (row    sum = how much others drive i)
        col 2 : PDC outflow  per channel
        col 3 : PDC inflow   per channel
    """
    dtf_out = dtf.sum(axis=0)   # sum over sink rows  → (19,) outflow per source
    dtf_in  = dtf.sum(axis=1)   # sum over source cols → (19,) inflow  per sink
    pdc_out = pdc.sum(axis=0)
    pdc_in  = pdc.sum(axis=1)

    return np.stack([dtf_out, dtf_in, pdc_out, pdc_in], axis=1).astype(np.float32)


def log_normalise(features: np.ndarray) -> np.ndarray:
    """
    Apply log1p to band-power features (columns 0-4) to reduce dynamic range.
    Connectivity features (columns 5-8) are already in [0, ~19] — leave them.
    """
    out = features.copy()
    out[:, :5] = np.log1p(out[:, :5])
    return out


def zscore_features(features: np.ndarray) -> np.ndarray:
    """
    Z-score each feature dimension across all (epoch × channel) observations.
    features : (n_epochs, 19, 9)
    Returns  : (n_epochs, 19, 9)  normalised
    """
    n_epochs, n_ch, n_feat = features.shape
    flat = features.reshape(-1, n_feat)           # (n_epochs*19, 9)
    mu   = flat.mean(axis=0, keepdims=True)
    sig  = flat.std( axis=0, keepdims=True) + 1e-8
    flat_norm = (flat - mu) / sig
    return flat_norm.reshape(n_epochs, n_ch, n_feat)


# ── main per-subject function ─────────────────────────────────────────────────

def process_subject(
    subj_id: int,
    epochs_dir: Path,
    connectivity_dir: Path,
    output_dir: Path,
) -> bool:

    subject_name  = f'subject_{subj_id:02d}'
    epochs_file   = epochs_dir       / f'{subject_name}_epochs.npy'
    graphs_file   = connectivity_dir / f'{subject_name}_graphs.npz'
    out_file      = output_dir       / f'{subject_name}_node_features.npy'

    # ── load epochs ───────────────────────────────────────────────────────────
    if not epochs_file.exists():
        print(f'  [SKIP] {subject_name}: epochs file not found')
        return False
    if not graphs_file.exists():
        print(f'  [SKIP] {subject_name}: graphs file not found')
        return False

    all_epochs = np.load(epochs_file)        # (N_total, 19, 1024)
    graphs     = np.load(graphs_file)

    # The connectivity step may have dropped unstable epochs.
    # The 'indices' array records which raw epoch indices survived.
    if 'indices' in graphs:
        valid_indices = graphs['indices'].astype(int)
    else:
        # Assume all epochs survived (rare — only if no unstable epochs)
        n_conn = graphs['dtf_integrated'].shape[0]
        valid_indices = np.arange(n_conn)

    dtf_all = graphs['dtf_integrated']   # (n_valid, 19, 19)
    pdc_all = graphs['pdc_integrated']   # (n_valid, 19, 19)
    n_valid = len(valid_indices)

    # ── extract features ──────────────────────────────────────────────────────
    node_features = np.zeros((n_valid, N_CHANNELS, N_FEATURES), dtype=np.float32)

    for ep_local, ep_raw in enumerate(tqdm(valid_indices,
                                           desc=f'  {subject_name}',
                                           leave=False)):
        epoch = all_epochs[ep_raw]           # (19, 1024)

        # Band powers — features 0-4
        bp = compute_band_powers(epoch)       # (19, 5)

        # Connectivity summaries — features 5-8
        cs = compute_connectivity_summaries(
            dtf_all[ep_local], pdc_all[ep_local]
        )                                     # (19, 4)

        node_features[ep_local] = np.concatenate([bp, cs], axis=1)  # (19, 9)

    # ── normalise ─────────────────────────────────────────────────────────────
    # 1. Log-compress band powers to handle the high dynamic range of PSD
    node_features = log_normalise(node_features)
    # 2. Z-score across all epochs and channels within this subject
    #    (subject-level normalisation keeps patient differences meaningful)
    node_features = zscore_features(node_features)

    # ── save ──────────────────────────────────────────────────────────────────
    np.save(out_file, node_features)
    print(f'  Saved: {out_file.name}  shape={node_features.shape}')
    return True


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract per-channel node features for GNN'
    )
    parser.add_argument('--epochs_dir',       required=True,
                        help='Directory with subject_XX_epochs.npy files')
    parser.add_argument('--connectivity_dir', required=True,
                        help='Directory with subject_XX_graphs.npz files')
    parser.add_argument('--output_dir',       required=True,
                        help='Where to save subject_XX_node_features.npy')
    parser.add_argument('--subject_ids', nargs='+', type=int,
                        default=list(range(1, 35)))
    args = parser.parse_args()

    epochs_dir       = Path(args.epochs_dir)
    connectivity_dir = Path(args.connectivity_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('STEP 4 — NODE FEATURE EXTRACTION')
    print('=' * 70)
    print(f'Epochs dir      : {epochs_dir}')
    print(f'Connectivity dir: {connectivity_dir}')
    print(f'Output dir      : {output_dir}')
    print(f'Subjects        : {args.subject_ids}')
    print(f'Features per node: {N_FEATURES}')
    print('  [0-4] Band power (delta, theta, alpha, beta, gamma) — log1p + zscore')
    print('  [5]   DTF outflow (integrated band)')
    print('  [6]   DTF inflow  (integrated band)')
    print('  [7]   PDC outflow (integrated band)')
    print('  [8]   PDC inflow  (integrated band)')
    print('=' * 70)

    success, errors = 0, 0
    for subj_id in args.subject_ids:
        ok = process_subject(subj_id, epochs_dir, connectivity_dir, output_dir)
        if ok:
            success += 1
        else:
            errors += 1

    print('\n' + '=' * 70)
    print(f'Done.  Success: {success}   Errors/Skipped: {errors}')
    print('=' * 70)
    print('\nNext step: python step5_build_dataset.py')


if __name__ == '__main__':
    main()
