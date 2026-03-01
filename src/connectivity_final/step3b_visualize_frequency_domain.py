"""
Frequency-Domain PDC Visualization (Baccalá Method)
=====================================================
Computes full-frequency PDC using Baccalá's method for every epoch,
applies training mask to exclude post-ictal epochs, then produces
three 19×19 grids:

  1. MEAN PRE-ICTAL  — average PDC(f) across all pre-ictal epochs
  2. MEAN ICTAL      — average PDC(f) across all ictal epochs
  3. DIFFERENCE      — ictal minus pre-ictal  (positive = increase, negative = decrease)

Matrix convention:
  pdc[i, j, f] = PDC FROM channel j TO channel i at frequency f
  → col j = outflow of j
  → row i = inflow  into i

Training mask:
  True  = epoch belongs to pre-ictal or ictal window (KEEP)
  False = post-ictal (EXCLUDE)
  Alignment handled via 'indices' array in graphs.npz if bad epochs were dropped.

Usage:
------
python step_freq_domain_pdc.py \
    --epochs_dir    F:/FORTH_Final_Thesis/FORTH-Thesis/preprocessed_epochs \
    --connectivity_dir F:/FORTH_Final_Thesis/FORTH-Thesis/connectivity \
    --output_dir    F:/FORTH_Final_Thesis/FORTH-Thesis/figures/freq_domain \
    --subject_ids   2 3 4
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# CHANNEL DEFINITIONS
# ==============================================================================

CHANNELS = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2'
]
N_CH = len(CHANNELS)

FS    = 256    # sampling frequency (Hz)
NFFT  = 512    # FFT length
ORDER = 12     # MVAR model order (same as used in main connectivity pipeline)


# ==============================================================================
# TRAINING MASK LOADER
# ==============================================================================

def load_training_mask(epochs_dir, subject_name, n_connectivity_epochs, graphs_data):
    """
    Load training mask and align to connectivity epoch indices.

    The mask is defined over ALL raw epochs before bad-epoch rejection.
    If bad epochs were dropped in step1, we need to map connectivity epochs
    back to the raw epoch numbering using the 'indices' array in graphs.npz.

    Parameters
    ----------
    epochs_dir : Path
    subject_name : str
    n_connectivity_epochs : int
    graphs_data : NpzFile  (the loaded .npz, may contain 'indices')

    Returns
    -------
    mask : np.ndarray of bool, shape (n_connectivity_epochs,)
        True = keep (pre-ictal / ictal), False = post-ictal
    n_excluded : int
    """
    mask_file = epochs_dir / f'{subject_name}_training_mask.npy'

    if not mask_file.exists():
        print(f'    No training mask found — using all {n_connectivity_epochs} epochs')
        return np.ones(n_connectivity_epochs, dtype=bool), 0

    raw_mask = np.load(mask_file)   # bool array over raw epochs

    # Align mask to connectivity epochs
    if len(raw_mask) == n_connectivity_epochs:
        mask = raw_mask
    elif 'indices' in graphs_data:
        indices = graphs_data['indices'].astype(int)
        if indices.max() < len(raw_mask):
            mask = raw_mask[indices]
        else:
            print('    Index mismatch — using all epochs')
            return np.ones(n_connectivity_epochs, dtype=bool), 0
    else:
        print('    Mask length mismatch and no indices — using all epochs')
        return np.ones(n_connectivity_epochs, dtype=bool), 0

    n_excluded = int((~mask).sum())
    return mask, n_excluded


# ==============================================================================
# BACCALÁ PDC — SINGLE EPOCH
# ==============================================================================

def compute_pdc_epoch(data, fs=FS, nfft=NFFT, order=ORDER):
    """
    Compute Baccalá's PDC for a single epoch using MVAR model.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_timepoints)
    fs, nfft, order : int

    Returns
    -------
    pdc : np.ndarray, shape (n_channels, n_channels, n_freqs)
        pdc[i, j, f] = FROM j TO i at frequency f
    freqs : np.ndarray, shape (n_freqs,)

    Notes
    -----
    Baccalá PDC is column-normalised:
        PDC[i,j,f] = |A[i,j,f]| / sqrt( sum_k |A[k,j,f]|^2 )
    where A(f) = I - sum_k AR_k * exp(-j2pi*f*k/fs)
    This ensures sum_i PDC[i,j,f]^2 = 1  (outflow of j sums to 1 per frequency).
    """
    from statsmodels.tsa.vector_ar.var_model import VAR

    n_freqs = nfft // 2 + 1
    freqs   = np.linspace(0, fs / 2, n_freqs)
    K       = data.shape[0]

    # Normalise to unit variance to improve VAR conditioning
    std = data.std(axis=1, keepdims=True)
    std[std < 1e-10] = 1.0
    data_n = data / std

    # Fit VAR
    model   = VAR(data_n.T)                                  # (T, K)
    results = model.fit(maxlags=order, trend='c', verbose=False)
    coefs   = results.coefs                                  # (p, K, K)
    p       = coefs.shape[0]

    # Build A(f) = I - sum_k AR_k * exp(-j2pi*f*k/fs)
    I  = np.eye(K, dtype=complex)
    Af = np.zeros((n_freqs, K, K), dtype=complex)

    for f_idx, f in enumerate(freqs):
        A_sum = np.zeros((K, K), dtype=complex)
        for k in range(p):
            A_sum += coefs[k] * np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
        Af[f_idx] = I - A_sum

    # Baccalá PDC: column-normalised
    pdc = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        col_norms = np.sqrt(np.sum(np.abs(Af[f_idx]) ** 2, axis=0))
        col_norms[col_norms < 1e-10] = 1e-10
        pdc[:, :, f_idx] = np.abs(Af[f_idx]) / col_norms[None, :]

    return pdc, freqs


# ==============================================================================
# COMPUTE AVERAGE PDC SPECTRUM PER CONDITION
# ==============================================================================

def compute_mean_spectra(epochs, labels, mask, fs=FS, nfft=NFFT, order=ORDER):
    """
    Compute mean PDC(f) separately for pre-ictal and ictal epochs
    after applying the training mask.

    Parameters
    ----------
    epochs : np.ndarray, shape (n_epochs, n_channels, n_timepoints)
    labels : np.ndarray, shape (n_epochs,)   0=pre-ictal, 1=ictal
    mask   : np.ndarray, shape (n_epochs,)   True=keep

    Returns
    -------
    pdc_pre  : (n_ch, n_ch, n_freqs)  mean pre-ictal PDC
    pdc_ict  : (n_ch, n_ch, n_freqs)  mean ictal PDC
    pdc_diff : (n_ch, n_ch, n_freqs)  ictal minus pre-ictal
    freqs    : (n_freqs,)
    counts   : dict with epoch counts
    """
    n_freqs  = nfft // 2 + 1
    freqs    = np.linspace(0, fs / 2, n_freqs)
    K        = epochs.shape[1]

    pdc_pre_sum = np.zeros((K, K, n_freqs))
    pdc_ict_sum = np.zeros((K, K, n_freqs))
    n_pre = n_ict = n_failed = 0

    # Apply mask first
    kept_epochs = epochs[mask]
    kept_labels = labels[mask]

    n_total = len(kept_epochs)
    print(f'    Computing PDC for {n_total} epochs '
          f'(pre={int((kept_labels==0).sum())}, ictal={int((kept_labels==1).sum())})')

    for idx in range(n_total):
        if idx % 10 == 0:
            print(f'    Epoch {idx+1}/{n_total}...', end='\r')

        data  = kept_epochs[idx]   # (K, T)
        label = kept_labels[idx]

        # Skip flat/constant channels
        if data.std() < 1e-10:
            n_failed += 1
            continue

        try:
            pdc, _ = compute_pdc_epoch(data, fs=fs, nfft=nfft, order=order)

            if label == 0:
                pdc_pre_sum += pdc
                n_pre += 1
            elif label == 1:
                pdc_ict_sum += pdc
                n_ict += 1

        except Exception as e:
            n_failed += 1
            continue

    print()   # newline after progress

    pdc_pre  = pdc_pre_sum / max(n_pre, 1)
    pdc_ict  = pdc_ict_sum / max(n_ict, 1)
    pdc_diff = pdc_ict - pdc_pre

    counts = {
        'pre':    n_pre,
        'ictal':  n_ict,
        'failed': n_failed,
    }

    print(f'    Pre-ictal: {n_pre}  |  Ictal: {n_ict}  |  Failed: {n_failed}')

    return pdc_pre, pdc_ict, pdc_diff, freqs, counts


# ==============================================================================
# PLOT 19×19 FREQUENCY-DOMAIN GRID
# ==============================================================================

def plot_freq_grid(pdc_matrix, freqs, output_path, title,
                   global_ymin=None, global_ymax=None,
                   cmap_line='steelblue', fill_alpha=0.25,
                   show_diff_zero=False):
    """
    Plot 19×19 grid where each cell shows PDC(f) for one channel pair.

    pdc_matrix : (19, 19, n_freqs)   pdc[i, j, :] = FROM j TO i
    """
    fig = plt.figure(figsize=(26, 26))
    gs  = GridSpec(N_CH, N_CH, figure=fig,
                   wspace=0.04, hspace=0.04,
                   left=0.06, right=0.98, top=0.96, bottom=0.05)

    if global_ymax is None:
        global_ymax = np.abs(pdc_matrix).max() * 1.1
    if global_ymin is None:
        global_ymin = pdc_matrix.min() * 1.1 if show_diff_zero else 0.0

    for i in range(N_CH):
        for j in range(N_CH):
            ax = fig.add_subplot(gs[i, j])

            if i == j:
                # Diagonal — channel name label
                ax.text(0.5, 0.5, CHANNELS[i],
                        ha='center', va='center',
                        fontsize=9, fontweight='bold')
                ax.axis('off')
            else:
                y = pdc_matrix[i, j, :]

                if show_diff_zero:
                    # Difference plot: colour positive/negative separately
                    ax.fill_between(freqs, y, 0,
                                    where=(y >= 0), alpha=fill_alpha,
                                    color='red',   interpolate=True)
                    ax.fill_between(freqs, y, 0,
                                    where=(y < 0),  alpha=fill_alpha,
                                    color='blue',  interpolate=True)
                    ax.plot(freqs, y, color='black', linewidth=0.7)
                    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
                else:
                    ax.fill_between(freqs, y, alpha=fill_alpha, color=cmap_line)
                    ax.plot(freqs, y, color=cmap_line, linewidth=0.8)

                # Frequency band background shading
                ax.axvspan(0.5,  4.0,  alpha=0.07, color='purple')
                ax.axvspan(4.0,  8.0,  alpha=0.07, color='blue')
                ax.axvspan(8.0,  15.0, alpha=0.07, color='green')
                ax.axvspan(15.0, 30.0, alpha=0.07, color='gold')
                ax.axvspan(30.0, 45.0, alpha=0.07, color='red')

                ax.set_xlim(0, 45)
                ax.set_ylim(global_ymin, global_ymax)

                # Only show tick labels on edges
                if i < N_CH - 1:
                    ax.set_xticklabels([])
                else:
                    ax.tick_params(labelsize=5)
                    ax.set_xlabel('Hz', fontsize=5)

                if j > 0:
                    ax.set_yticklabels([])
                else:
                    ax.tick_params(labelsize=5)

                ax.grid(alpha=0.2, linewidth=0.4)

    # Row labels (sink channels — receives FROM)
    for i, ch in enumerate(CHANNELS):
        fig.text(0.003, 1 - (i + 0.5) / N_CH * 0.91 - 0.04,
                 ch, va='center', ha='left', fontsize=7, fontweight='bold')

    # Column labels (source channels — sends TO)
    for j, ch in enumerate(CHANNELS):
        fig.text(0.06 + j / N_CH * 0.92,
                 0.025, ch,
                 va='center', ha='center', fontsize=7, fontweight='bold')

    fig.text(0.001, 0.5, 'Sink channel (receives FROM →)',
             va='center', rotation='vertical', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.002, 'Source channel (sends TO ↓)',
             ha='center', fontsize=11, fontweight='bold')

    # Band legend at bottom
    band_colors = [('δ 0.5–4',   'purple'),
                   ('θ 4–8',     'blue'),
                   ('α 8–15',    'green'),
                   ('β 15–30',   'gold'),
                   ('γ 30–45',   'red')]
    legend_x = 0.15
    for label, color in band_colors:
        fig.text(legend_x, 0.013, f'■ {label}',
                 color=color, fontsize=8, ha='left', va='center')
        legend_x += 0.14

    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {output_path.name}')


# ==============================================================================
# PROCESS ONE SUBJECT
# ==============================================================================

def process_subject(subj_id, epochs_dir, connectivity_dir, output_dir,
                    fs=FS, nfft=NFFT, order=ORDER):
    """Load epochs + mask, compute PDC per epoch, plot three grids."""

    subject_name = f'subject_{subj_id:02d}'
    graphs_file  = connectivity_dir / f'{subject_name}_graphs.npz'
    epochs_file  = epochs_dir / f'{subject_name}_epochs.npy'
    labels_file  = epochs_dir / f'{subject_name}_labels.npy'

    print(f'\n{"="*70}')
    print(f'Processing: {subject_name}')
    print(f'{"="*70}')

    # ------------------------------------------------------------------
    # Check files exist
    # ------------------------------------------------------------------
    missing = [f for f in [epochs_file, labels_file] if not f.exists()]
    if missing:
        print(f'  Missing files: {[f.name for f in missing]}')
        return False

    # ------------------------------------------------------------------
    # Load epochs and labels
    # ------------------------------------------------------------------
    epochs = np.load(epochs_file)   # (n_epochs, n_channels, n_timepoints)
    labels = np.load(labels_file)   # (n_epochs,)
    n_epochs = len(labels)

    print(f'  Epochs loaded: {n_epochs}  '
          f'(pre={int((labels==0).sum())}, ictal={int((labels==1).sum())})')

    if not np.any(labels == 1):
        print('  No ictal epochs — skipping')
        return False

    # ------------------------------------------------------------------
    # Load training mask
    # ------------------------------------------------------------------
    graphs_data = np.load(graphs_file) if graphs_file.exists() else {}
    mask, n_excluded = load_training_mask(
        epochs_dir, subject_name, n_epochs, graphs_data)

    print(f'  Training mask: {mask.sum()} kept, {n_excluded} post-ictal excluded')

    # ------------------------------------------------------------------
    # Compute PDC per epoch (Baccalá), averaged per condition
    # ------------------------------------------------------------------
    print('  Computing Baccalá PDC per epoch...')
    pdc_pre, pdc_ict, pdc_diff, freqs, counts = compute_mean_spectra(
        epochs, labels, mask, fs=fs, nfft=nfft, order=order)

    if counts['pre'] == 0 or counts['ictal'] == 0:
        print('  Not enough epochs in one condition — skipping plots')
        return False

    # ------------------------------------------------------------------
    # Shared y-axis limits for pre and ictal (same scale for comparison)
    # ------------------------------------------------------------------
    ymax_shared = max(pdc_pre.max(), pdc_ict.max()) * 1.1
    diff_abs    = np.abs(pdc_diff).max() * 1.1

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    subj_out = output_dir / subject_name
    subj_out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: Mean pre-ictal
    # ------------------------------------------------------------------
    print('  Plotting mean pre-ictal grid...')
    plot_freq_grid(
        pdc_pre, freqs,
        subj_out / f'{subject_name}_pdc_freq_preictal.png',
        title=f'{subject_name}  |  Mean PDC(f) — PRE-ICTAL  '
              f'(n={counts["pre"]} epochs)\n'
              f'Baccalá PDC  |  VAR order={order}  |  '
              f'Post-ictal excluded: {n_excluded} epochs',
        global_ymin=0, global_ymax=ymax_shared,
        cmap_line='steelblue',
    )

    # ------------------------------------------------------------------
    # Plot 2: Mean ictal
    # ------------------------------------------------------------------
    print('  Plotting mean ictal grid...')
    plot_freq_grid(
        pdc_ict, freqs,
        subj_out / f'{subject_name}_pdc_freq_ictal.png',
        title=f'{subject_name}  |  Mean PDC(f) — ICTAL  '
              f'(n={counts["ictal"]} epochs)\n'
              f'Baccalá PDC  |  VAR order={order}  |  '
              f'Post-ictal excluded: {n_excluded} epochs',
        global_ymin=0, global_ymax=ymax_shared,
        cmap_line='firebrick',
    )

    # ------------------------------------------------------------------
    # Plot 3: Difference (ictal − pre-ictal)
    # ------------------------------------------------------------------
    print('  Plotting difference grid (ictal − pre-ictal)...')
    plot_freq_grid(
        pdc_diff, freqs,
        subj_out / f'{subject_name}_pdc_freq_difference.png',
        title=f'{subject_name}  |  PDC(f) DIFFERENCE  (ictal − pre-ictal)\n'
              f'RED = connectivity increases during seizure  |  '
              f'BLUE = connectivity decreases\n'
              f'Baccalá PDC  |  VAR order={order}  |  '
              f'Post-ictal excluded: {n_excluded} epochs',
        global_ymin=-diff_abs, global_ymax=diff_abs,
        show_diff_zero=True,
    )

    print(f'  Done — outputs in: {subj_out}')
    return True


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Frequency-domain PDC (Baccalá) with training mask and '
                    'pre-ictal vs ictal comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--epochs_dir',        required=True,
                        help='Directory with subject_XX_epochs.npy + labels + mask')
    parser.add_argument('--connectivity_dir',  required=True,
                        help='Directory with subject_XX_graphs.npz '
                             '(used for training mask alignment)')
    parser.add_argument('--output_dir',        required=True)
    parser.add_argument('--subject_ids', nargs='+', type=int,
                        default=list(range(1, 35)),
                        help='Subject IDs to process (default: 2–34, skips 1)')
    parser.add_argument('--fs',    type=int, default=256, help='Sampling rate (default: 256)')
    parser.add_argument('--nfft',  type=int, default=512, help='FFT length (default: 512)')
    parser.add_argument('--order', type=int, default=12,  help='VAR model order (default: 12)')

    args = parser.parse_args()

    epochs_dir       = Path(args.epochs_dir)
    connectivity_dir = Path(args.connectivity_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('FREQUENCY-DOMAIN PDC — BACCALÁ METHOD')
    print('=' * 70)
    print(f'Epochs dir:     {epochs_dir}')
    print(f'Connectivity:   {connectivity_dir}')
    print(f'Output dir:     {output_dir}')
    print(f'Subjects:       {len(args.subject_ids)}')
    print(f'FS={args.fs} Hz  |  NFFT={args.nfft}  |  VAR order={args.order}')
    print('=' * 70)
    print()
    print('OUTPUTS PER SUBJECT:')
    print('  *_pdc_freq_preictal.png    — mean PDC(f) pre-ictal (blue)')
    print('  *_pdc_freq_ictal.png       — mean PDC(f) ictal     (red)')
    print('  *_pdc_freq_difference.png  — ictal − pre-ictal     (red=increase, blue=decrease)')
    print('=' * 70)

    success, errors = 0, 0

    for subj_id in args.subject_ids:
        #if subj_id == 1:
        #    print(f'\n  Skipping subject_01 — PAT 11 not in paper')
        #    continue
        try:
            ok = process_subject(
                subj_id,
                epochs_dir       = epochs_dir,
                connectivity_dir = connectivity_dir,
                output_dir       = output_dir,
                fs               = args.fs,
                nfft             = args.nfft,
                order            = args.order,
            )
            if ok:
                success += 1
            else:
                errors += 1
        except Exception as e:
            import traceback
            print(f'\n  Error on subject_{subj_id:02d}: {e}')
            traceback.print_exc()
            errors += 1

    print('\n' + '=' * 70)
    print(f'Success: {success}  |  Errors/skipped: {errors}')
    print('=' * 70)


if __name__ == '__main__':
    main()