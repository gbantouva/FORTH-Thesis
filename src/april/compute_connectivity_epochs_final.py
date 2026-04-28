"""
Effective Connectivity Pipeline — PDC & DTF (with multiprocessing)
===================================================================

1) Cut the full EEG signal into non-overlapping 4-second epochs
2) Fit MVAR on each epoch across ALL subjects, compute BIC (parallel)
   → choose ONE global model order p for ALL subjects and epochs
3) Compute PDC and DTF for each epoch (parallel per subject)
4) Plot connectivity matrices per epoch (band-averaged + broadband)
   - Epochs labelled as Pre-ictal / Ictal / Post-ictal
   - Fixed color scale [0, 1] across all epochs for fair comparison

Output structure per subject:
    subj_XX/
    ├── dtf_broadband/
    ├── dtf_bands/
    ├── pdc_broadband/
    └── pdc_bands/

Frequency bands (standard EEG, matching data bandpass 0.5–45 Hz):
    Delta:  0.5 –  4 Hz
    Theta:  4   –  8 Hz
    Alpha:  8   – 13 Hz
    Beta:  13   – 30 Hz
    Gamma: 30   – 45 Hz

References:
    [1] Baccalá & Sameshima (2001). Partial directed coherence. Biol. Cybern. 84(6), 463–474.
    [2] Kamiński & Blinowska (1991). DTF. Biol. Cybern. 65, 203–210.

Author: [your name]
"""

import numpy as np
from scipy import linalg
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from functools import partial
import time

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURE
# ═══════════════════════════════════════════════════════════════════════
mat_file = r"F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat"
out_dir  = r"F:\FORTH_Final_Thesis\FORTH-Thesis\figures\april\connectivity_epochs"

fs = 256                # sampling frequency (Hz)
EPOCH_SEC = 4           # epoch length in seconds
EPOCH_LEN = EPOCH_SEC * fs  # epoch length in samples (1024)
NFFT = 256              # frequency bins (1 Hz resolution)
MAX_ORDER = 20          # max MVAR order to test for BIC
N_WORKERS = None        # None = use all available cores

# Standard EEG frequency bands
# Data bandpass: 0.5–45 Hz (from filename bw_0.5_45)
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
    try:
        A, C = fit_mvar(x, order)
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0:
            return np.inf
        num_params = K * K * order
        return n * logdet + num_params * np.log(n)
    except Exception:
        return np.inf


def _bic_for_one_epoch(epoch, max_order):
    """Compute BIC for all orders on one epoch. Returns array (max_order,)."""
    bics = np.full(max_order, np.inf)
    for p in range(1, max_order + 1):
        bics[p - 1] = compute_bic(epoch, p)
    return bics


def select_global_order_parallel(all_epochs, max_order=20, n_workers=None):
    """
    Select a SINGLE MVAR order across ALL epochs using median BIC.
    Parallelized: each worker processes one epoch across all orders.
    """
    n_epochs = len(all_epochs)
    print(f'  Computing BIC (orders 1–{max_order}) for {n_epochs} epochs...')

    worker_fn = partial(_bic_for_one_epoch, max_order=max_order)

    with mp.Pool(processes=n_workers) as pool:
        results = list(pool.imap(worker_fn, all_epochs, chunksize=16))

    bic_matrix = np.array(results)  # (n_epochs, max_order)
    median_bic = np.median(bic_matrix, axis=0)  # (max_order,)
    best_order = np.argmin(median_bic) + 1
    return best_order, median_bic


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
    """Classify epoch as Pre-ictal, Ictal, or Post-ictal."""
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
    """Plot broadband connectivity for one epoch. Fixed scale [0, 1]."""
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
#  PARALLEL SUBJECT PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def _process_one_subject(args):
    """
    Process all epochs of one subject: fit MVAR, compute PDC/DTF, save plots.
    Designed to run in a worker process.
    """
    (subj_idx, epochs_list, epoch_info_list, chnames, info,
     best_p, out_dir_root) = args

    subj_dir = os.path.join(out_dir_root, f'subj_{subj_idx+1:02d}')
    dir_pdc_bands     = os.path.join(subj_dir, 'pdc_bands')
    dir_pdc_broadband = os.path.join(subj_dir, 'pdc_broadband')
    dir_dtf_bands     = os.path.join(subj_dir, 'dtf_bands')
    dir_dtf_broadband = os.path.join(subj_dir, 'dtf_broadband')

    for d in [dir_pdc_bands, dir_pdc_broadband, dir_dtf_bands, dir_dtf_broadband]:
        os.makedirs(d, exist_ok=True)

    for ei in range(len(epochs_list)):
        epoch_data = epochs_list[ei]
        start, stop, label = epoch_info_list[ei]

        t_start = start / fs
        t_stop  = stop / fs

        try:
            A, C = fit_mvar(epoch_data, best_p)
            pdc, dtf, freqs = compute_pdc_dtf(A, C, NFFT, fs)
        except Exception:
            continue

        pdc_bands = band_average(pdc, freqs, BANDS)
        dtf_bands = band_average(dtf, freqs, BANDS)
        pdc_broad = broadband_average(pdc)
        dtf_broad = broadband_average(dtf)

        label_clean = label.replace('-', '')
        epoch_tag = f'epoch_{ei+1:03d}_{label_clean}'
        title_tag = f'{label} | epoch {ei+1} ({t_start:.0f}–{t_stop:.0f}s)'

        plot_epoch_bands(
            pdc_bands, chnames,
            f'PDC — {title_tag} — Subj {subj_idx+1} ({info})',
            os.path.join(dir_pdc_bands, f'{epoch_tag}.png')
        )
        plot_epoch_broadband(
            pdc_broad, chnames,
            f'PDC (broadband) — {title_tag} — Subj {subj_idx+1}',
            os.path.join(dir_pdc_broadband, f'{epoch_tag}.png')
        )
        plot_epoch_bands(
            dtf_bands, chnames,
            f'DTF — {title_tag} — Subj {subj_idx+1} ({info})',
            os.path.join(dir_dtf_bands, f'{epoch_tag}.png')
        )
        plot_epoch_broadband(
            dtf_broad, chnames,
            f'DTF (broadband) — {title_tag} — Subj {subj_idx+1}',
            os.path.join(dir_dtf_broadband, f'{epoch_tag}.png')
        )

    return subj_idx, len(epochs_list), info


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    mp.freeze_support()  # required on Windows

    t0 = time.time()

    n_workers = N_WORKERS or mp.cpu_count()
    print(f'Using {n_workers} CPU cores')

    data = sio.loadmat(mat_file, squeeze_me=False, struct_as_record=False)
    s = data['seizure'][0, 0]
    SUBJECTS = s.x.shape[0]

    os.makedirs(out_dir, exist_ok=True)

    # ==================================================================
    #  PHASE 1: Cut ALL subjects into epochs
    # ==================================================================
    print('\n' + '=' * 70)
    print('PHASE 1 — Cutting all subjects into 4-second epochs')
    print('=' * 70)

    all_epochs = []
    subject_data = []

    for subj in range(SUBJECTS):
        x_full = s.x[subj, 0]
        N, K = x_full.shape

        chan_struct = s.chans[subj, 0][0, 0]
        sel = chan_struct.selected
        chnames = [str(sel[i, 0][0]) for i in range(sel.shape[0])]

        ann_start = int(s.annotation[subj, 0][0, 0]) - 1
        ann_stop  = int(s.annotation[subj, 1][0, 0]) - 1
        info = str(s.info[subj, 0][0])

        n_epochs = N // EPOCH_LEN
        epochs = []
        epoch_info = []

        for ei in range(n_epochs):
            start = ei * EPOCH_LEN
            stop  = start + EPOCH_LEN
            epoch_data = x_full[start:stop, :]
            label = classify_epoch(start, stop, ann_start, ann_stop)
            epochs.append(epoch_data)
            epoch_info.append((start, stop, label))
            all_epochs.append(epoch_data)

        n_pre  = sum(1 for _, _, l in epoch_info if l == 'Pre-ictal')
        n_ict  = sum(1 for _, _, l in epoch_info if l == 'Ictal')
        n_post = sum(1 for _, _, l in epoch_info if l == 'Post-ictal')

        subject_data.append({
            'subj_idx': subj,
            'epochs': epochs,
            'epoch_info': epoch_info,
            'chnames': chnames,
            'info': info,
        })

        print(f'  [{subj+1:2d}/{SUBJECTS}] {info:30s}  '
              f'epochs={len(epochs)} (pre={n_pre}, ict={n_ict}, post={n_post})')

    print(f'\n  Total epochs across all subjects: {len(all_epochs)}')

    # ==================================================================
    #  PHASE 2: Global MVAR order selection via BIC (parallel)
    # ==================================================================
    print('\n' + '=' * 70)
    print('PHASE 2 — Selecting ONE global MVAR order (parallel BIC)')
    print('=' * 70)

    t1 = time.time()
    best_p, median_bic = select_global_order_parallel(all_epochs, MAX_ORDER, n_workers)
    t2 = time.time()
    print(f'\n  Global MVAR order = {best_p}  ({t2 - t1:.1f}s)')

    # Save global BIC plot
    fig_bic, ax_bic = plt.subplots(figsize=(8, 4))
    ax_bic.plot(range(1, MAX_ORDER + 1), median_bic, 'o-', ms=5, color='steelblue')
    ax_bic.axvline(best_p, color='r', ls='--', lw=2, label=f'Best order = {best_p}')
    ax_bic.set_xlabel('Model order p', fontsize=12)
    ax_bic.set_ylabel('Median BIC (across all epochs)', fontsize=12)
    ax_bic.set_title(f'Global MVAR order selection — {len(all_epochs)} epochs, {SUBJECTS} subjects')
    ax_bic.legend(fontsize=11)
    ax_bic.grid(alpha=0.3)
    plt.tight_layout()
    fig_bic.savefig(os.path.join(out_dir, 'global_bic_order_selection.png'), dpi=150)
    plt.close(fig_bic)
    print(f'  BIC plot saved')

    # ==================================================================
    #  PHASE 3: Compute PDC & DTF for every epoch (parallel per subject)
    # ==================================================================
    print('\n' + '=' * 70)
    print(f'PHASE 3 — Computing PDC & DTF (order={best_p}), {n_workers} workers')
    print('=' * 70)

    # Build argument list for parallel processing
    tasks = []
    for sd in subject_data:
        tasks.append((
            sd['subj_idx'],
            sd['epochs'],
            sd['epoch_info'],
            sd['chnames'],
            sd['info'],
            best_p,
            out_dir,
        ))

    t3 = time.time()

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_process_one_subject, tasks):
            subj_idx, n_ep, info = result
            print(f'  [{subj_idx+1:2d}/{SUBJECTS}] {info:30s}  {n_ep} epochs done')

    t4 = time.time()

    print(f'\n{"=" * 70}')
    print(f'DONE')
    print(f'  Global order:  p = {best_p}')
    print(f'  BIC time:      {t2 - t1:.1f}s')
    print(f'  Compute time:  {t4 - t3:.1f}s')
    print(f'  Total time:    {t4 - t0:.1f}s')
    print(f'  Output:        {out_dir}/')
    print(f'{"=" * 70}')
