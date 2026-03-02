"""
signals_test_19ch.py
====================
Synthetic 19-channel test with KNOWN connectivity:

    F7(3)  -> F3(4), Fz(5), F4(6)
    C3(9)  -> T4(12)
    Pz(15) -> C4(11)
    T3(8)  -> O2(19)

Uses EXACT step2 math. Saves diagnostic heatmaps identical to
save_diagnostic_plot() in step2_compute_connectivity.py.

Run:
    python signals_test_19ch.py
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import linalg
from statsmodels.tsa.vector_ar.var_model import VAR

warnings.filterwarnings("ignore")

# ── Constants (identical to step2) ────────────────────────────────────────────
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2',
]

BANDS = {
    'integrated': (0.5, 45.0),
    'delta':      (0.5,  4.0),
    'theta':      (4.0,  8.0),
    'alpha':      (8.0, 15.0),
    'beta':      (15.0, 30.0),
    'gamma1':    (30.0, 45.0),
}

GT_1IDX = [(3,4),(3,5),(3,6),(9,12),(15,11),(8,19)]
GT_0IDX = [(s-1, t-1) for s, t in GT_1IDX]

FS          = 256.0
NFFT        = 512
FIXED_ORDER = 12
N_SAMPLES   = 40_000
SEED        = 42
PLOT_DIR    = Path("synthetic_test_plots")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 MATH — verbatim copy, zero changes
# ══════════════════════════════════════════════════════════════════════════════

def compute_dtf_pdc_from_var(coefs, fs=256.0, nfft=512):
    p, K, _ = coefs.shape
    n_freqs  = nfft // 2 + 1
    freqs    = np.linspace(0, fs / 2, n_freqs)
    I        = np.eye(K)

    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    H_f = np.zeros((n_freqs, K, K), dtype=complex)

    for f_idx, f in enumerate(freqs):
        A_sum = np.zeros((K, K), dtype=complex)
        for k in range(p):
            A_sum += coefs[k] * np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
        A_f[f_idx] = I - A_sum
        try:
            H_f[f_idx] = linalg.inv(A_f[f_idx])
        except linalg.LinAlgError:
            H_f[f_idx] = linalg.pinv(A_f[f_idx])

    pdc = np.zeros((K, K, n_freqs))
    for fi in range(n_freqs):
        col_norms = np.sqrt((np.abs(A_f[fi]) ** 2).sum(axis=0))
        col_norms[col_norms < 1e-10] = 1e-10
        pdc[:, :, fi] = np.abs(A_f[fi]) / col_norms[np.newaxis, :]

    dtf = np.zeros((K, K, n_freqs))
    for fi in range(n_freqs):
        row_norms = np.sqrt((np.abs(H_f[fi]) ** 2).sum(axis=1))
        row_norms[row_norms < 1e-10] = 1e-10
        dtf[:, :, fi] = np.abs(H_f[fi]) / row_norms[:, np.newaxis]

    return dtf, pdc, freqs


def verify_spectrum(dtf_s, pdc_s, tol=1e-6):
    dtf2 = dtf_s ** 2
    pdc2 = pdc_s ** 2
    dtf_row_sums = dtf2.sum(axis=1)
    pdc_col_sums = pdc2.sum(axis=0)
    dtf_dev = float(np.abs(dtf_row_sums - 1.0).max())
    pdc_dev = float(np.abs(pdc_col_sums - 1.0).max())
    passed  = dtf_dev < tol and pdc_dev < tol
    return passed, dtf_dev, pdc_dev


def process_single_epoch(data, fs, fixed_order, nfft, verify=False, epoch_idx=None):
    data_std = np.std(data)
    if data_std < 1e-10:
        return None

    data_scaled = data / data_std

    try:
        model   = VAR(data_scaled.T)
        results = model.fit(maxlags=fixed_order, trend='c', verbose=False)

        if results.k_ar == 0:
            return None

        try:
            if not results.is_stable():
                return None
        except Exception:
            pass

        dtf_s, pdc_s, freqs = compute_dtf_pdc_from_var(results.coefs, fs, nfft)

        if verify:
            passed, dtf_dev, pdc_dev = verify_spectrum(dtf_s, pdc_s)
            tag   = "PASS" if passed else "FAIL"
            label = f"epoch {epoch_idx}" if epoch_idx is not None else "epoch"
            print(f"  [{tag}] {label} | DTF row-sum dev={dtf_dev:.2e} | "
                  f"PDC col-sum dev={pdc_dev:.2e}")

        dtf_bands = {}
        pdc_bands = {}

        for band_name, (f_lo, f_hi) in BANDS.items():
            idx = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
            if len(idx) == 0:
                return None
            dtf_band = dtf_s[:, :, idx].mean(axis=2)
            pdc_band = pdc_s[:, :, idx].mean(axis=2)
            np.fill_diagonal(dtf_band, 0.0)
            np.fill_diagonal(pdc_band, 0.0)
            dtf_bands[band_name] = dtf_band
            pdc_bands[band_name] = pdc_band

        return {
            'dtf_bands':    dtf_bands,
            'pdc_bands':    pdc_bands,
            'order':        fixed_order,
            'dtf_spectrum': dtf_s,
            'pdc_spectrum': pdc_s,
            'freqs':        freqs,
        }

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP — identical to save_diagnostic_plot() in step2
# ══════════════════════════════════════════════════════════════════════════════

def save_diagnostic_plot(dtf, pdc, fixed_order, subject_name, output_dir):
    """Exact copy of step2's save_diagnostic_plot()."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, mat, title in [
        (axes[0], dtf, f'DTF  integrated 0.5-45 Hz\n{subject_name}  p={fixed_order}'),
        (axes[1], pdc, f'PDC  integrated 0.5-45 Hz\n{subject_name}  p={fixed_order}'),
    ]:
        sns.heatmap(mat, ax=ax, cmap='viridis', square=True,
                    xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Connectivity (diagonal=0)'})
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Source  (From j)', fontsize=10)
        ax.set_ylabel('Sink  (To i)',     fontsize=10)
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0,  labelsize=7)

    fig.text(0.5, 0.01,
             'DTF bright cols = strong sources  |  PDC bright rows = strong sinks  '
             '|  Both are correct (not a bug)',
             ha='center', fontsize=9, style='italic', color='navy')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = output_dir / f'{subject_name}_integrated_connectivity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def save_band_plot(dtf_bands, pdc_bands, fixed_order, subject_name, output_dir):
    """One figure per metric with all 6 bands as subplots."""
    band_list = list(BANDS.keys())

    for metric, band_dict in [('DTF', dtf_bands), ('PDC', pdc_bands)]:
        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        axes = axes.ravel()

        for ax_i, band_name in enumerate(band_list):
            mat = band_dict[band_name]
            sns.heatmap(mat, ax=axes[ax_i], cmap='viridis', square=True,
                        xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
                        vmin=0, vmax=1,
                        cbar_kws={'label': 'Connectivity (diagonal=0)'})
            f_lo, f_hi = BANDS[band_name]
            axes[ax_i].set_title(f'{band_name}  {f_lo}-{f_hi} Hz',
                                 fontsize=11, fontweight='bold')
            axes[ax_i].set_xlabel('Source  (From j)', fontsize=8)
            axes[ax_i].set_ylabel('Sink  (To i)',     fontsize=8)
            axes[ax_i].tick_params(axis='x', rotation=90, labelsize=6)
            axes[ax_i].tick_params(axis='y', rotation=0,  labelsize=6)

        dim = 'bright cols = sources' if metric == 'DTF' else 'bright rows = sinks'
        fig.suptitle(f'{metric}  —  all bands  |  {subject_name}  p={fixed_order}\n'
                     f'({dim})',
                     fontsize=13, fontweight='bold')
        fig.text(0.5, 0.01,
                 'DTF bright cols = strong sources  |  PDC bright rows = strong sinks  '
                 '|  Both are correct (not a bug)',
                 ha='center', fontsize=9, style='italic', color='navy')

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        out = output_dir / f'{subject_name}_{metric.lower()}_all_bands.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_signal():
    K   = len(CHANNEL_NAMES)
    rng = np.random.default_rng(SEED)

    A = np.diag([0.30] * K).astype(float)
    for s0, t0 in GT_0IDX:
        A[t0, s0] = 0.75

    eigvals = np.abs(np.linalg.eigvals(A))
    assert eigvals.max() < 1.0, f"AR matrix unstable! max eigval={eigvals.max():.3f}"

    X = np.zeros((N_SAMPLES, K))
    X[0] = rng.standard_normal(K)
    for t in range(1, N_SAMPLES):
        X[t] = A @ X[t - 1] + 0.10 * rng.standard_normal(K)

    return X


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PLOT_DIR.mkdir(exist_ok=True)
    SUBJECT = "synth_19ch"

    print("=" * 65)
    print("19-Channel Synthetic Connectivity Test")
    print("=" * 65)
    print("Ground-truth connections (1-indexed):")
    for s1, t1 in GT_1IDX:
        print(f"  Ch{s1:2d} ({CHANNEL_NAMES[s1-1]:<4s}) -> Ch{t1:2d} ({CHANNEL_NAMES[t1-1]:<4s})")

    # 1. Generate
    print(f"\nGenerating {N_SAMPLES:,}-sample AR(1) signal ...", flush=True)
    X = generate_signal()
    print(f"  Shape  : {X.shape}")
    print(f"  Std/ch : min={X.std(axis=0).min():.3f}  max={X.std(axis=0).max():.3f}")

    # 2. Run step2 pipeline
    print(f"\nFitting MVAR(p={FIXED_ORDER}) — verify normalization at spectrum level ...")
    result = process_single_epoch(
        X.T,                    # (K, T) as step2 expects
        fs=FS,
        fixed_order=FIXED_ORDER,
        nfft=NFFT,
        verify=True,
        epoch_idx=0,
    )

    if result is None:
        print("\n[ERROR] process_single_epoch returned None.")
        raise SystemExit(1)

    # 3. Save plots
    print(f"\nSaving heatmaps to {PLOT_DIR.resolve()} ...")
    save_diagnostic_plot(
        result['dtf_bands']['integrated'],
        result['pdc_bands']['integrated'],
        FIXED_ORDER, SUBJECT, PLOT_DIR,
    )
    save_band_plot(
        result['dtf_bands'], result['pdc_bands'],
        FIXED_ORDER, SUBJECT, PLOT_DIR,
    )

    # 4. Recovery summary
    K       = len(CHANNEL_NAMES)
    pdc_int = result['pdc_bands']['integrated']
    dtf_int = result['dtf_bands']['integrated']
    THRESH  = 0.15

    print("\n" + "=" * 65)
    print("GROUND-TRUTH RECOVERY  (integrated band, threshold=0.15)")
    print("=" * 65)
    print(f"  {'Connection':<27} {'PDC':>7}  {'DTF':>7}  {'PDC ok?':>8}  {'DTF ok?':>8}")
    print("  " + "-" * 62)
    for (s0, t0), (s1, t1) in zip(GT_0IDX, GT_1IDX):
        pv  = pdc_int[t0, s0]
        dv  = dtf_int[t0, s0]
        pok = "YES" if pv >= THRESH else "NO"
        dok = "YES" if dv >= THRESH else "NO"
        name = f"Ch{s1}({CHANNEL_NAMES[s1-1]})->Ch{t1}({CHANNEL_NAMES[t1-1]})"
        print(f"  {name:<27}  {pv:>6.3f}   {dv:>6.3f}   {pok:>8}   {dok:>8}")

    pdc_bg = [pdc_int[r,c] for r in range(K) for c in range(K)
              if (c,r) not in GT_0IDX and r != c]
    dtf_bg = [dtf_int[r,c] for r in range(K) for c in range(K)
              if (c,r) not in GT_0IDX and r != c]
    print(f"\n  Noise floor  PDC mean={np.mean(pdc_bg):.4f}  DTF mean={np.mean(dtf_bg):.4f}")
    print("=" * 65)
    print(f"\nDone. All plots in: {PLOT_DIR.resolve()}")