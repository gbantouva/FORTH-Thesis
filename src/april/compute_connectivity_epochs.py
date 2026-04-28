"""
Effective Connectivity Pipeline — PDC & DTF
============================================

1) Cut the full EEG signal into non-overlapping 4-second epochs
2) Fit MVAR on each epoch, compute BIC → choose ONE model order for all epochs
3) Compute PDC and DTF for each epoch
4) Plot connectivity matrices per epoch (band-averaged + broadband)
   - Epochs labelled as Pre-ictal / Ictal / Post-ictal based on seizure annotation
   - Fixed color scale [0, 1] across all epochs for fair comparison

References:
    [1] Baccalá & Sameshima (2001). Partial directed coherence. Biol. Cybern. 84(6), 463–474.
    [2] Kamiński & Blinowska (1991). DTF. Biol. Cybern. 65, 203–210.
    [3] Schlögl & Supp (2006). Analyzing event-related EEG data with MVAR. Prog. Brain Res.

Author: [your name]
"""

import numpy as np
from scipy import linalg
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURE
# ═══════════════════════════════════════════════════════════════════════
mat_file = r"F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat"
out_dir  = r"F:\FORTH_Final_Thesis\FORTH-Thesis\figures\april\connectivity_epochs"

fs = 256                # sampling frequency (Hz)
EPOCH_SEC = 4           # epoch length in seconds
EPOCH_LEN = EPOCH_SEC * fs  # epoch length in samples
NFFT = 256              # frequency bins (1 Hz resolution)
MAX_ORDER = 20          # max MVAR order to test for BIC

# Standard EEG frequency bands
BANDS = {
    'Delta (0.5-4)':  (0.5, 4),
    'Theta (4-8)':    (4, 8),
    'Alpha (8-13)':   (8, 13),
    'Beta (13-30)':   (13, 30),
    'Gamma (30-45)':  (30, 45),
}


# ═══════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def fit_mvar(x, order):
    """
    Fit MVAR(p) model via OLS.

    Parameters
    ----------
    x : ndarray (T, K) — T samples, K channels
    order : int — model order p

    Returns
    -------
    A : ndarray (K, K*p) — coefficient matrices [A(1) ... A(p)]
    C : ndarray (K, K) — noise covariance
    """
    T, K = x.shape
    p = order
    Y = x[p:]
    X = np.column_stack([x[p - k - 1:T - k - 1] for k in range(p)])
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    A = B.T
    E = Y - X @ B
    C = np.cov(E, rowvar=False, bias=False)
    return A, C


def compute_bic(x, order):
    """Compute BIC for a given MVAR order."""
    T, K = x.shape
    n = T - order
    if n < K * order + 1:
        return np.inf
    A, C = fit_mvar(x, order)
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        return np.inf
    num_params = K * K * order
    return n * logdet + num_params * np.log(n)


def select_global_order(epochs, max_order=20):
    """
    Select a single MVAR order for all epochs using median BIC.

    For each candidate order, BIC is computed on every epoch.
    The order that minimizes the median BIC across epochs is chosen.

    Returns
    -------
    best_order : int
    bic_matrix : ndarray (max_order, n_epochs)
    """
    n_epochs = len(epochs)
    bic_matrix = np.full((max_order, n_epochs), np.inf)

    for p in range(1, max_order + 1):
        for ei, epoch in enumerate(epochs):
            bic_matrix[p - 1, ei] = compute_bic(epoch, p)

    median_bic = np.median(bic_matrix, axis=1)
    best_order = np.argmin(median_bic) + 1
    return best_order, bic_matrix, median_bic


def compute_pdc_dtf(A, C, nfft=256, fs=256):
    """
    Compute PDC and DTF from MVAR coefficients.

    Returns pdc, dtf each of shape (K, K, nfft), and freqs (nfft,).
    Convention: matrix[i, j, f] = influence FROM j TO i at frequency f.
    """
    K = C.shape[0]
    p = A.shape[1] // K
    freqs = np.linspace(0, fs / 2, nfft)

    pdc = np.zeros((K, K, nfft))
    dtf = np.zeros((K, K, nfft))

    for fi, f in enumerate(freqs):
        Af = np.eye(K, dtype=complex)
        for k in range(p):
            Ak = A[:, k * K:(k + 1) * K]
            Af -= Ak * np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)

        Hf = linalg.inv(Af)

        # PDC: column-normalized
        for j in range(K):
            col_norm = np.sqrt(np.real(np.vdot(Af[:, j], Af[:, j])))
            if col_norm > 0:
                pdc[:, j, fi] = np.abs(Af[:, j]) / col_norm

        # DTF: row-normalized
        for i in range(K):
            row_norm = np.sqrt(np.real(np.vdot(Hf[i, :], Hf[i, :])))
            if row_norm > 0:
                dtf[i, :, fi] = np.abs(Hf[i, :]) / row_norm

    return pdc, dtf, freqs


def band_average(conn, freqs, bands):
    """Average connectivity over frequency bands. Returns dict of (K, K) arrays."""
    matrices = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        matrices[name] = np.mean(conn[:, :, mask], axis=2) if mask.any() else np.zeros(conn.shape[:2])
    return matrices


def broadband_average(conn):
    """Average connectivity over all frequencies. Returns (K, K) array."""
    return np.mean(conn, axis=2)


def classify_epoch(epoch_start, epoch_stop, ann_start, ann_stop):
    """
    Classify an epoch as pre-ictal, ictal, or post-ictal.

    An epoch is ictal if it overlaps with the seizure annotation.
    Pre-ictal if it ends before seizure onset.
    Post-ictal if it starts after seizure offset.
    """
    if epoch_stop <= ann_start:
        return 'Pre-ictal'
    elif epoch_start >= ann_stop:
        return 'Post-ictal'
    else:
        return 'Ictal'


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_epoch_bands(matrices, chnames, suptitle, out_path):
    """Plot band-averaged connectivity matrices for one epoch. Fixed scale [0, 1]."""
    bands = list(matrices.keys())
    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n + 1, 4.5))
    if n == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        mat = matrices[band].copy()
        np.fill_diagonal(mat, 0)
        K = mat.shape[0]
        n_labels = min(K, len(chnames))

        im = ax.imshow(mat, cmap='hot_r', vmin=0, vmax=1,
                       aspect='equal', interpolation='nearest')
        ax.set_xticks(range(n_labels))
        ax.set_xticklabels(chnames[:n_labels], rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(n_labels))
        ax.set_yticklabels(chnames[:n_labels], fontsize=7)
        ax.set_xlabel('Source (j)', fontsize=9)
        ax.set_ylabel('Sink (i)', fontsize=9)
        ax.set_title(band, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_epoch_broadband(matrix, chnames, suptitle, out_path):
    """Plot broadband (all-frequency average) connectivity for one epoch. Fixed scale [0, 1]."""
    fig, ax = plt.subplots(figsize=(7, 6))
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    K = mat.shape[0]
    n_labels = min(K, len(chnames))

    im = ax.imshow(mat, cmap='hot_r', vmin=0, vmax=1,
                   aspect='equal', interpolation='nearest')
    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(chnames[:n_labels], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(chnames[:n_labels], fontsize=9)
    ax.set_xlabel('Source (j)', fontsize=11)
    ax.set_ylabel('Sink (i)', fontsize=11)
    ax.set_title(suptitle, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    data = sio.loadmat(mat_file, squeeze_me=False, struct_as_record=False)
    s = data['seizure'][0, 0]
    SUBJECTS = s.x.shape[0]

    os.makedirs(out_dir, exist_ok=True)

    for subj in range(SUBJECTS):
        x_full = s.x[subj, 0]       # (samples, channels)
        N, K = x_full.shape

        # Channel names
        chan_struct = s.chans[subj, 0][0, 0]
        sel = chan_struct.selected
        chnames = [str(sel[i, 0][0]) for i in range(sel.shape[0])]

        # Seizure annotation (1-based → 0-based)
        ann_start = int(s.annotation[subj, 0][0, 0]) - 1
        ann_stop  = int(s.annotation[subj, 1][0, 0]) - 1
        info = str(s.info[subj, 0][0])

        # ── Step 1: Cut into non-overlapping 4-second epochs ──────
        n_epochs = N // EPOCH_LEN
        epochs = []
        epoch_info = []  # (start_sample, stop_sample, label)

        for ei in range(n_epochs):
            start = ei * EPOCH_LEN
            stop  = start + EPOCH_LEN
            epoch_data = x_full[start:stop, :]
            label = classify_epoch(start, stop, ann_start, ann_stop)
            epochs.append(epoch_data)
            epoch_info.append((start, stop, label))

        n_pre   = sum(1 for _, _, l in epoch_info if l == 'Pre-ictal')
        n_ict   = sum(1 for _, _, l in epoch_info if l == 'Ictal')
        n_post  = sum(1 for _, _, l in epoch_info if l == 'Post-ictal')

        print(f'[{subj+1:2d}/{SUBJECTS}] {info:30s}  '
              f'epochs={len(epochs)} (pre={n_pre}, ict={n_ict}, post={n_post})')

        # ── Step 2: Select global MVAR order via BIC ──────────────
        best_p, bic_matrix, median_bic = select_global_order(epochs, MAX_ORDER)
        print(f'           Global MVAR order = {best_p} (BIC)')

        # Save BIC plot
        subj_dir = os.path.join(out_dir, f'subj_{subj+1:02d}')
        os.makedirs(subj_dir, exist_ok=True)

        fig_bic, ax_bic = plt.subplots(figsize=(7, 3.5))
        ax_bic.plot(range(1, MAX_ORDER + 1), median_bic, 'o-', ms=4)
        ax_bic.axvline(best_p, color='r', ls='--', label=f'Best order = {best_p}')
        ax_bic.set_xlabel('Model order p')
        ax_bic.set_ylabel('Median BIC')
        ax_bic.set_title(f'MVAR order selection — Subj {subj+1} ({info})')
        ax_bic.legend()
        plt.tight_layout()
        fig_bic.savefig(os.path.join(subj_dir, 'bic_order_selection.png'), dpi=130)
        plt.close(fig_bic)

        # ── Step 3: Compute PDC & DTF for each epoch ─────────────
        for ei, (epoch_data, (start, stop, label)) in enumerate(zip(epochs, epoch_info)):

            t_start = start / fs
            t_stop  = stop / fs

            A, C = fit_mvar(epoch_data, best_p)
            pdc, dtf, freqs = compute_pdc_dtf(A, C, NFFT, fs)

            # Band averages
            pdc_bands = band_average(pdc, freqs, BANDS)
            dtf_bands = band_average(dtf, freqs, BANDS)

            # Broadband averages
            pdc_broad = broadband_average(pdc)
            dtf_broad = broadband_average(dtf)

            # Epoch label for filenames and titles
            epoch_tag = f'epoch_{ei+1:03d}_{label.replace("-", "")}'
            title_tag = f'{label} | epoch {ei+1} ({t_start:.0f}–{t_stop:.0f}s)'

            # Plot band-averaged
            plot_epoch_bands(
                pdc_bands, chnames,
                f'PDC — {title_tag} — Subj {subj+1} ({info})',
                os.path.join(subj_dir, f'{epoch_tag}_PDC_bands.png')
            )
            plot_epoch_bands(
                dtf_bands, chnames,
                f'DTF — {title_tag} — Subj {subj+1} ({info})',
                os.path.join(subj_dir, f'{epoch_tag}_DTF_bands.png')
            )

            # Plot broadband
            plot_epoch_broadband(
                pdc_broad, chnames,
                f'PDC (broadband) — {title_tag} — Subj {subj+1}',
                os.path.join(subj_dir, f'{epoch_tag}_PDC_broadband.png')
            )
            plot_epoch_broadband(
                dtf_broad, chnames,
                f'DTF (broadband) — {title_tag} — Subj {subj+1}',
                os.path.join(subj_dir, f'{epoch_tag}_DTF_broadband.png')
            )

        print(f'           → saved to {subj_dir}/')

    print(f'\nDone — all subjects processed.')
