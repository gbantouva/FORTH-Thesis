"""
Mean Spectral Matrix — averaged across all epochs of each class
===============================================================
Produces the spectral matrix visualization but using the MEAN
spectrum across all pre-ictal epochs and all ictal epochs.

This is far more informative than single epochs.

Usage:
    python plot_mean_spectral_matrix.py \
        --inputdir F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs \
        --subject  subject_01 \
        --fixedorder 12 \
        --outdir   figures/spectral_matrix
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import linalg
from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

CHANNEL_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T3','C3','Cz','C4','T4',
    'T5','P3','Pz','P4','T6',
    'O1','O2',
]

BANDS = {
    'δ':  (0.5,  4.0, '#4477AA', 0.15),
    'θ':  (4.0,  8.0, '#66CCEE', 0.15),
    'α':  (8.0, 15.0, '#228833', 0.15),
    'β':  (15.0,30.0, '#CCBB44', 0.15),
    'γ':  (30.0,45.0, '#EE6677', 0.15),
}


# ==============================================================================
# CORE (same as pipeline)
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
            A_sum += coefs[k] * np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
        A_f[f_idx] = I - A_sum
        try:    H_f[f_idx] = linalg.inv(A_f[f_idx])
        except: H_f[f_idx] = linalg.pinv(A_f[f_idx])

    pdc = np.zeros((K, K, n_freqs))
    dtf = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Af = A_f[f_idx]
        cn = np.sqrt(np.sum(np.abs(Af)**2, axis=0)); cn[cn==0]=1e-10
        pdc[:,:,f_idx] = np.abs(Af) / cn[None,:]
        Hf = H_f[f_idx]
        rn = np.sqrt(np.sum(np.abs(Hf)**2, axis=1)); rn[rn==0]=1e-10
        dtf[:,:,f_idx] = np.abs(Hf) / rn[:,None]
    return dtf, pdc, freqs


def get_spectrum_for_epoch(data, fixed_order, fs=256.0, nfft=512):
    """Returns (dtf_spectrum, pdc_spectrum, freqs) or None if failed."""
    std = np.std(data)
    if std < 1e-10: return None
    try:
        res = VAR(data / std).fit(maxlags=fixed_order, trend='c', verbose=False)
        # need (T, N)
        res = VAR((data/std).T).fit(maxlags=fixed_order, trend='c', verbose=False)
        if res.k_ar == 0: return None
        if not res.is_stable(): return None
        dtf_s, pdc_s, freqs = compute_dtf_pdc_from_var(res.coefs, fs, nfft)
        return dtf_s, pdc_s, freqs
    except:
        return None


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_spectral_matrix(spectrum, freqs, channel_names, title,
                         n_epochs_used, outpath=None):
    """Plot K×K spectral matrix with band shading."""
    K        = spectrum.shape[0]
    freq_mask = freqs <= 45.0
    f_plot    = freqs[freq_mask]

    fig = plt.figure(figsize=(K * 1.5, K * 1.5))
    gs  = gridspec.GridSpec(K, K, figure=fig,
                            hspace=0.04, wspace=0.04,
                            left=0.07, right=0.98,
                            top=0.91,  bottom=0.05)

    for i in range(K):
        for j in range(K):
            ax  = fig.add_subplot(gs[i, j])
            val = spectrum[i, j, freq_mask]

            # Band shading
            for name, (f_lo, f_hi, color, alpha) in BANDS.items():
                ax.axvspan(f_lo, f_hi, color=color, alpha=alpha, linewidth=0)

            # Diagonal = red, off-diagonal = dark gray
            color = '#CC3311' if i == j else '#333333'
            ax.fill_between(f_plot, val, alpha=0.55, color=color)
            ax.plot(f_plot, val, color=color, linewidth=0.5)

            ax.set_xlim(0, 45); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(channel_names[i], fontsize=5,
                              rotation=0, labelpad=24, va='center')
            if i == K - 1:
                ax.set_xlabel(channel_names[j], fontsize=5, labelpad=2)

    fig.suptitle(f"{title}\n(averaged over {n_epochs_used} epochs)",
                 fontsize=10, fontweight='bold', y=0.97)
    fig.text(0.03, 0.50, 'Target (To i)',
             va='center', rotation='vertical', fontsize=7, color='#555')
    fig.text(0.50, 0.01, 'Source (From j)',
             ha='center', fontsize=7, color='#555')

    # Band legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc=c, alpha=0.5, label=f'{n} {lo:.0f}–{hi:.0f}Hz')
        for n,(lo,hi,c,_) in BANDS.items()
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               fontsize=5.5, ncol=5, framealpha=0.8,
               bbox_to_anchor=(0.98, 0.96))

    if outpath:
        plt.savefig(outpath, dpi=160, bbox_inches='tight')
        print(f"  Saved → {outpath.name}")
    plt.close()


def plot_difference_spectral(spectrum_pre, spectrum_ict, freqs,
                              channel_names, title, outpath=None):
    """
    Plot DIFFERENCE spectral matrix: ictal_mean - pre-ictal_mean.
    Red = ictal higher, Blue = pre-ictal higher.
    """
    K         = spectrum_pre.shape[0]
    freq_mask = freqs <= 45.0
    f_plot    = freqs[freq_mask]
    diff      = spectrum_ict - spectrum_pre            # (K, K, n_freqs)

    fig = plt.figure(figsize=(K * 1.5, K * 1.5))
    gs  = gridspec.GridSpec(K, K, figure=fig,
                            hspace=0.04, wspace=0.04,
                            left=0.07, right=0.98,
                            top=0.91,  bottom=0.05)

    for i in range(K):
        for j in range(K):
            ax  = fig.add_subplot(gs[i, j])
            val = diff[i, j, freq_mask]

            # Band shading
            for name, (f_lo, f_hi, color, alpha) in BANDS.items():
                ax.axvspan(f_lo, f_hi, color=color, alpha=alpha*0.7, linewidth=0)

            ax.axhline(0, color='black', linewidth=0.4, linestyle='--')

            # Fill positive (red = ictal>pre) and negative (blue = pre>ictal)
            ax.fill_between(f_plot, val, where=(val >= 0),
                            color='#CC3311', alpha=0.6)
            ax.fill_between(f_plot, val, where=(val < 0),
                            color='#3366AA', alpha=0.6)
            ax.plot(f_plot, val, color='black', linewidth=0.4)

            lim = np.abs(val).max() if np.abs(val).max() > 0 else 0.1
            ax.set_xlim(0, 45); ax.set_ylim(-lim, lim)
            ax.set_xticks([]); ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(channel_names[i], fontsize=5,
                              rotation=0, labelpad=24, va='center')
            if i == K - 1:
                ax.set_xlabel(channel_names[j], fontsize=5, labelpad=2)

    fig.suptitle(f"{title}\nRed = higher during seizure | Blue = higher pre-seizure",
                 fontsize=10, fontweight='bold', y=0.97)
    fig.text(0.03, 0.50, 'Target (To i)',
             va='center', rotation='vertical', fontsize=7, color='#555')
    fig.text(0.50, 0.01, 'Source (From j)',
             ha='center', fontsize=7, color='#555')

    if outpath:
        plt.savefig(outpath, dpi=160, bbox_inches='tight')
        print(f"  Saved → {outpath.name}")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir",    required=True)
    parser.add_argument("--subject",     default="subject_01")
    parser.add_argument("--fixedorder",  type=int, default=12)
    parser.add_argument("--outdir",      default="figures/spectral_matrix")
    parser.add_argument("--max_epochs",  type=int, default=30,
                        help="Max epochs per class to average (default 30, "
                             "use -1 for all)")
    args = parser.parse_args()

    input_dir = Path(args.inputdir)
    out_dir   = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.load(input_dir / f"{args.subject}_epochs.npy")   # (E, 19, 1024)
    labels = np.load(input_dir / f"{args.subject}_labels.npy")   # (E,)

    pre_idx = np.where(labels == 0)[0]
    ict_idx = np.where(labels == 1)[0]

    # Cap to max_epochs to keep runtime reasonable
    if args.max_epochs > 0:
        pre_idx = pre_idx[:args.max_epochs]
        ict_idx = ict_idx[:args.max_epochs]

    print(f"\n{args.subject}: using {len(pre_idx)} pre-ictal, "
          f"{len(ict_idx)} ictal epochs")

    # ------------------------------------------------------------------
    # Accumulate mean spectra  — sum valid epochs, divide at end
    # ------------------------------------------------------------------
    K      = 19
    nfft   = 512
    n_freq = nfft // 2 + 1

    sum_pdc = {'pre': np.zeros((K, K, n_freq)),
               'ict': np.zeros((K, K, n_freq))}
    sum_dtf = {'pre': np.zeros((K, K, n_freq)),
               'ict': np.zeros((K, K, n_freq))}
    count   = {'pre': 0, 'ict': 0}

    for class_name, idx_list in [('pre', pre_idx), ('ict', ict_idx)]:
        label_str = 'Pre-ictal' if class_name == 'pre' else 'Ictal'
        for i in tqdm(idx_list, desc=f"Computing {label_str} spectra"):
            result = get_spectrum_for_epoch(
                epochs[i], args.fixedorder, fs=256.0, nfft=nfft
            )
            if result is None:
                continue
            dtf_s, pdc_s, freqs = result
            sum_dtf[class_name] += dtf_s
            sum_pdc[class_name] += pdc_s
            count[class_name]   += 1

    print(f"\nValid epochs used: "
          f"pre-ictal={count['pre']}, ictal={count['ict']}")

    if count['pre'] == 0 or count['ict'] == 0:
        print("❌ Not enough valid epochs")
        return

    # Mean spectra
    mean_pdc_pre = sum_pdc['pre'] / count['pre']
    mean_pdc_ict = sum_pdc['ict'] / count['ict']
    mean_dtf_pre = sum_dtf['pre'] / count['pre']
    mean_dtf_ict = sum_dtf['ict'] / count['ict']

    base = f"{args.subject}_p{args.fixedorder}"

    # ------------------------------------------------------------------
    # Plot 1: Mean PDC — pre-ictal
    # ------------------------------------------------------------------
    print("\nPlotting mean PDC pre-ictal...")
    plot_spectral_matrix(
        mean_pdc_pre, freqs, CHANNEL_NAMES,
        title=f"PDC — {args.subject} — Mean Pre-ictal",
        n_epochs_used=count['pre'],
        outpath=out_dir / f"{base}_PDC_mean_preictal.png",
    )

    # ------------------------------------------------------------------
    # Plot 2: Mean PDC — ictal
    # ------------------------------------------------------------------
    print("Plotting mean PDC ictal...")
    plot_spectral_matrix(
        mean_pdc_ict, freqs, CHANNEL_NAMES,
        title=f"PDC — {args.subject} — Mean Ictal",
        n_epochs_used=count['ict'],
        outpath=out_dir / f"{base}_PDC_mean_ictal.png",
    )

    # ------------------------------------------------------------------
    # Plot 3: Mean DTF — pre-ictal
    # ------------------------------------------------------------------
    print("Plotting mean DTF pre-ictal...")
    plot_spectral_matrix(
        mean_dtf_pre, freqs, CHANNEL_NAMES,
        title=f"DTF — {args.subject} — Mean Pre-ictal",
        n_epochs_used=count['pre'],
        outpath=out_dir / f"{base}_DTF_mean_preictal.png",
    )

    # ------------------------------------------------------------------
    # Plot 4: Mean DTF — ictal
    # ------------------------------------------------------------------
    print("Plotting mean DTF ictal...")
    plot_spectral_matrix(
        mean_dtf_ict, freqs, CHANNEL_NAMES,
        title=f"DTF — {args.subject} — Mean Ictal",
        n_epochs_used=count['ict'],
        outpath=out_dir / f"{base}_DTF_mean_ictal.png",
    )

    # ------------------------------------------------------------------
    # Plot 5: DIFFERENCE spectral — PDC (ictal - pre-ictal)
    # ------------------------------------------------------------------
    print("Plotting PDC difference spectral (ictal - pre-ictal)...")
    plot_difference_spectral(
        mean_pdc_pre, mean_pdc_ict, freqs, CHANNEL_NAMES,
        title=f"PDC Difference (Ictal − Pre-ictal) — {args.subject}",
        outpath=out_dir / f"{base}_PDC_difference_spectral.png",
    )

    # ------------------------------------------------------------------
    # Plot 6: DIFFERENCE spectral — DTF (ictal - pre-ictal)
    # ------------------------------------------------------------------
    print("Plotting DTF difference spectral (ictal - pre-ictal)...")
    plot_difference_spectral(
        mean_dtf_pre, mean_dtf_ict, freqs, CHANNEL_NAMES,
        title=f"DTF Difference (Ictal − Pre-ictal) — {args.subject}",
        outpath=out_dir / f"{base}_DTF_difference_spectral.png",
    )

    print(f"\n✅ Done. All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
