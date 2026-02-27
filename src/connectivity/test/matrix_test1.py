"""
Spectral Matrix Plot — PDC and DTF across all frequencies
=========================================================
Produces the classic spectral matrix visualization (like Fig. 1 in the paper)
for a selected epoch from your TUC dataset.

Each cell [i, j] shows PDC_ij(f) or DTF_ij(f) as a function of frequency.
Row = target (sink), Column = source.

Usage:
    python plot_spectral_matrix.py \
        --inputdir F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs \
        --subject subject_01 \
        --epoch 0 \
        --fixedorder 12
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import linalg
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings

warnings.filterwarnings("ignore")

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3',  'C3',  'Cz', 'C4', 'T4',
    'T5',  'P3',  'Pz', 'P4', 'T6',
    'O1',  'O2',
]


# ==============================================================================
# CORE (same as your pipeline)
# ==============================================================================

def compute_dtf_pdc_from_var(coefs, fs=256.0, nfft=512):
    p, K, _ = coefs.shape
    n_freqs  = nfft // 2 + 1
    freqs    = np.linspace(0, fs / 2, n_freqs)
    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    H_f = np.zeros((n_freqs, K, K), dtype=complex)
    I   = np.eye(K)

    for f_idx, f in enumerate(freqs):
        A_sum = np.zeros((K, K), dtype=complex)
        for k in range(p):
            phase  = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
            A_sum += coefs[k] * phase
        A_f[f_idx] = I - A_sum
        try:
            H_f[f_idx] = linalg.inv(A_f[f_idx])
        except linalg.LinAlgError:
            H_f[f_idx] = linalg.pinv(A_f[f_idx])

    pdc = np.zeros((K, K, n_freqs))
    dtf = np.zeros((K, K, n_freqs))

    for f_idx in range(n_freqs):
        Af        = A_f[f_idx]
        col_norms = np.sqrt(np.sum(np.abs(Af) ** 2, axis=0))
        col_norms[col_norms == 0] = 1e-10
        pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]

        Hf        = H_f[f_idx]
        row_norms = np.sqrt(np.sum(np.abs(Hf) ** 2, axis=1))
        row_norms[row_norms == 0] = 1e-10
        dtf[:, :, f_idx] = np.abs(Hf) / row_norms[:, None]

    return dtf, pdc, freqs


# ==============================================================================
# SPECTRAL MATRIX PLOT
# ==============================================================================

def plot_spectral_matrix(spectrum, freqs, channel_names, title,
                         highlight_bands=True, outpath=None):
    """
    Plot the full K×K spectral matrix.
    Each cell [i,j] is a line plot of spectrum[i,j,:] vs freqs.

    Parameters
    ----------
    spectrum      : (K, K, n_freqs)  PDC or DTF
    freqs         : (n_freqs,)
    channel_names : list of K strings
    title         : figure title
    highlight_bands: shade standard EEG frequency bands
    outpath       : Path or None — save figure if provided
    """
    K = spectrum.shape[0]

    # Band shading colours
    bands = {
        'δ':  (0.5,  4.0, '#4477AA', 0.15),
        'θ':  (4.0,  8.0, '#66CCEE', 0.15),
        'α':  (8.0, 15.0, '#228833', 0.15),
        'β':  (15.0,30.0, '#CCBB44', 0.15),
        'γ':  (30.0,45.0, '#EE6677', 0.15),
    }

    fig = plt.figure(figsize=(K * 1.4, K * 1.4))
    gs  = gridspec.GridSpec(
        K, K,
        figure=fig,
        hspace=0.05, wspace=0.05,
        left=0.08, right=0.98,
        top=0.92,  bottom=0.06,
    )

    # Only plot up to 45 Hz (your analysis range)
    freq_mask = freqs <= 45.0
    f_plot    = freqs[freq_mask]

    for i in range(K):
        for j in range(K):
            ax  = fig.add_subplot(gs[i, j])
            val = spectrum[i, j, freq_mask]

            # Shade frequency bands
            if highlight_bands:
                for band_name, (f_lo, f_hi, color, alpha) in bands.items():
                    ax.axvspan(f_lo, f_hi, color=color, alpha=alpha, linewidth=0)

            # Fill under curve (paper style)
            ax.fill_between(f_plot, val, alpha=0.6, color='#333333')
            ax.plot(f_plot, val, color='#111111', linewidth=0.6)

            # Diagonal: highlight in red (self-connectivity artefact)
            if i == j:
                ax.fill_between(f_plot, val, alpha=0.4, color='#CC3311')
                ax.plot(f_plot, val, color='#CC3311', linewidth=0.8)

            ax.set_xlim(0, 45)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])

            # Channel labels — left column and bottom row only
            if j == 0:
                ax.set_ylabel(channel_names[i], fontsize=5.5,
                              rotation=0, labelpad=22, va='center')
            if i == K - 1:
                ax.set_xlabel(channel_names[j], fontsize=5.5, labelpad=3)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.97)

    # Shared axis labels
    fig.text(0.03, 0.50, 'Target (To i)',
             va='center', rotation='vertical', fontsize=8, color='#444444')
    fig.text(0.50, 0.02, 'Source (From j)',
             ha='center', fontsize=8, color='#444444')

    # Band legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.4, label=f'{n} {lo:.0f}–{hi:.0f} Hz')
        for n, (lo, hi, c, _) in bands.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper right', fontsize=6,
        ncol=5, framealpha=0.8,
        bbox_to_anchor=(0.98, 0.96),
    )

    if outpath:
        plt.savefig(outpath, dpi=180, bbox_inches='tight')
        print(f"  Saved: {outpath}")
    else:
        plt.show()

    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir",   required=True)
    parser.add_argument("--subject",    default="subject_01",
                        help="Subject name, e.g. subject_01")
    parser.add_argument("--epoch",      type=int, default=None,
                        help="Epoch index. Default: auto-pick first stable pre-ictal "
                             "and first stable ictal")
    parser.add_argument("--fixedorder", type=int, default=12)
    parser.add_argument("--outdir",     default="figures/spectral_matrix",
                        help="Output directory for saved plots")
    args = parser.parse_args()

    input_dir = Path(args.inputdir)
    out_dir   = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    epochs_file = input_dir / f"{args.subject}_epochs.npy"
    labels_file = input_dir / f"{args.subject}_labels.npy"

    if not epochs_file.exists():
        print(f"❌ File not found: {epochs_file}")
        return

    epochs = np.load(epochs_file)    # (n_epochs, 19, 1024)
    labels = np.load(labels_file)    # (n_epochs,)

    print(f"Loaded {args.subject}: {epochs.shape[0]} epochs")

    # Decide which epochs to plot
    if args.epoch is not None:
        epoch_list = [(args.epoch, labels[args.epoch])]
    else:
        # Auto-pick: first stable pre-ictal AND first stable ictal
        epoch_list = []
        for target_label, label_name in [(0, 'Pre-ictal'), (1, 'Ictal')]:
            candidates = np.where(labels == target_label)[0]
            for idx in candidates:
                data     = epochs[idx]
                data_std = np.std(data)
                if data_std < 1e-10:
                    continue
                try:
                    res = VAR(data / data_std).fit(
                        maxlags=args.fixedorder, trend='c', verbose=False
                    )
                    # Note: VAR expects (T, N) — need .T here too
                    res = VAR((data / data_std).T).fit(
                        maxlags=args.fixedorder, trend='c', verbose=False
                    )
                    if res.k_ar > 0 and res.is_stable():
                        epoch_list.append((idx, target_label))
                        print(f"  Auto-selected epoch {idx} ({label_name})")
                        break
                except Exception:
                    continue

    # Plot each selected epoch
    for epoch_idx, label in epoch_list:
        label_name = 'Ictal' if label == 1 else 'Pre-ictal'
        data       = epochs[epoch_idx]         # (19, 1024)
        data_std   = np.std(data)
        data_scaled = data / data_std

        print(f"\nProcessing epoch {epoch_idx} ({label_name})...")

        # Fit MVAR
        results = VAR(data_scaled.T).fit(
            maxlags=args.fixedorder, trend='c', verbose=False
        )

        if results.k_ar == 0:
            print(f"  ⚠️  k_ar=0, skipping")
            continue

        if not results.is_stable():
            print(f"  ⚠️  Unstable, skipping")
            continue

        # Compute spectra
        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(
            results.coefs, fs=256.0, nfft=512
        )

        base_title = (f"{args.subject}  |  Epoch {epoch_idx} ({label_name})  "
                      f"|  Order p={args.fixedorder}")

        # Plot PDC
        print(f"  Plotting PDC spectral matrix...")
        plot_spectral_matrix(
            pdc_spectrum, freqs, CHANNEL_NAMES,
            title=f"PDC — {base_title}",
            highlight_bands=True,
            outpath=out_dir / f"{args.subject}_epoch{epoch_idx}_{label_name}_PDC_spectral.png",
        )

        # Plot DTF
        print(f"  Plotting DTF spectral matrix...")
        plot_spectral_matrix(
            dtf_spectrum, freqs, CHANNEL_NAMES,
            title=f"DTF — {base_title}",
            highlight_bands=True,
            outpath=out_dir / f"{args.subject}_epoch{epoch_idx}_{label_name}_DTF_spectral.png",
        )

    print(f"\n✅ Done. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
