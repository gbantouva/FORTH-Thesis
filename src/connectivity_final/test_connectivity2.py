"""
signals_test_19ch_hub.py
========================
Synthetic 19-channel signal designed to reproduce the pattern seen in
real ictal EEG (subject_01, epoch 075):

  DTF  : bright COLUMNS  (hub sources broadcast to many channels,
                           including via indirect paths)
  PDC  : bright ROWS     (only direct sinks light up, no column spread)

Ground-truth network
--------------------
  HUB 1 — Cz(10) directly drives:  C3(9), C4(11), T4(12), Pz(15), P4(16)
  HUB 2 — T4(12) directly drives:  T5(13), P3(14), T6(17)
  CHAIN  — Fp1(1) -> F3(4) -> Fz(5) -> F4(6) -> F8(7)
           (tests DTF column spread vs PDC local-only response)

Expected output
---------------
  DTF : full bright columns at Cz(10) and T4(12),
        partial bright column at Fp1(1) decaying down the chain
  PDC : bright rows at C3,C4,T4,Pz,P4 (Cz sinks)
        bright rows at T5,P3,T6       (T4 sinks)
        bright row  at F3             (Fp1 direct sink only)
        NO column-wide brightness in PDC

Run:
    python signals_test_19ch_hub.py
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

FS          = 256.0
NFFT        = 512
FIXED_ORDER = 12
N_SAMPLES   = 40_000
SEED        = 42
PLOT_DIR    = Path("synthetic_test_plots")

# ── Ground-truth network (1-indexed) ──────────────────────────────────────────
# HUB 1: Cz(10) -> many
HUB1_SRC  = 10
HUB1_TGTS = [9, 11, 12, 15, 16]      # C3, C4, T4, Pz, P4

# HUB 2: T4(12) -> many
HUB2_SRC  = 12
HUB2_TGTS = [13, 14, 17]             # T5, P3, T6

# CHAIN: Fp1->F3->Fz->F4->F8
CHAIN = [1, 4, 5, 6, 7]

GT_1IDX = (
    [(HUB1_SRC, t) for t in HUB1_TGTS] +
    [(HUB2_SRC, t) for t in HUB2_TGTS] +
    [(CHAIN[i], CHAIN[i+1]) for i in range(len(CHAIN)-1)]
)
GT_0IDX = [(s-1, t-1) for s, t in GT_1IDX]


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
        try:    H_f[f_idx] = linalg.inv(A_f[f_idx])
        except: H_f[f_idx] = linalg.pinv(A_f[f_idx])
    pdc = np.zeros((K, K, n_freqs))
    for fi in range(n_freqs):
        col_norms = np.sqrt((np.abs(A_f[fi])**2).sum(axis=0))
        col_norms[col_norms < 1e-10] = 1e-10
        pdc[:,:,fi] = np.abs(A_f[fi]) / col_norms[np.newaxis,:]
    dtf = np.zeros((K, K, n_freqs))
    for fi in range(n_freqs):
        row_norms = np.sqrt((np.abs(H_f[fi])**2).sum(axis=1))
        row_norms[row_norms < 1e-10] = 1e-10
        dtf[:,:,fi] = np.abs(H_f[fi]) / row_norms[:,np.newaxis]
    return dtf, pdc, freqs


def verify_spectrum(dtf_s, pdc_s, tol=1e-6):
    dtf_dev = float(np.abs((dtf_s**2).sum(axis=1) - 1.0).max())
    pdc_dev = float(np.abs((pdc_s**2).sum(axis=0) - 1.0).max())
    return dtf_dev < tol and pdc_dev < tol, dtf_dev, pdc_dev


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
            tag = "PASS" if passed else "FAIL"
            print(f"  [{tag}] epoch {epoch_idx} | DTF dev={dtf_dev:.2e} | PDC dev={pdc_dev:.2e}")
        dtf_bands, pdc_bands = {}, {}
        for band_name, (f_lo, f_hi) in BANDS.items():
            idx = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
            if len(idx) == 0:
                return None
            dtf_band = dtf_s[:,:,idx].mean(axis=2)
            pdc_band = pdc_s[:,:,idx].mean(axis=2)
            np.fill_diagonal(dtf_band, 0.0)
            np.fill_diagonal(pdc_band, 0.0)
            dtf_bands[band_name] = dtf_band
            pdc_bands[band_name] = pdc_band
        return {'dtf_bands': dtf_bands, 'pdc_bands': pdc_bands,
                'order': fixed_order, 'dtf_spectrum': dtf_s,
                'pdc_spectrum': pdc_s, 'freqs': freqs}
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP — identical to step2's save_diagnostic_plot()
# ══════════════════════════════════════════════════════════════════════════════

def save_diagnostic_plot(dtf, pdc, fixed_order, subject_name, output_dir, band='integrated'):
    f_lo, f_hi = BANDS[band]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mat, title in [
        (axes[0], dtf,
         f'DTF  {band} {f_lo}-{f_hi} Hz\n{subject_name}  p={fixed_order}\n'
         f'Bright COLUMNS = strong source channels\nrow-normalised | direct + indirect'),
        (axes[1], pdc,
         f'PDC  {band} {f_lo}-{f_hi} Hz\n{subject_name}  p={fixed_order}\n'
         f'Bright ROWS = strong sink channels\ncol-normalised | direct connections only'),
    ]:
        sns.heatmap(mat, ax=ax, cmap='viridis', square=True,
                    xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Connectivity (diagonal = 0)'})
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Source  (From j)', fontsize=10)
        ax.set_ylabel('Sink  (To i)',     fontsize=10)
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0,  labelsize=7)
    fig.text(0.5, 0.01,
             'DTF bright cols = strong sources  |  PDC bright rows = strong sinks  '
             '|  Both are correct (not a bug)',
             ha='center', fontsize=9, style='italic', color='navy')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = output_dir / f'{subject_name}_{band}_connectivity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def save_all_bands(dtf_bands, pdc_bands, fixed_order, subject_name, output_dir):
    band_list = list(BANDS.keys())
    for metric, band_dict in [('DTF', dtf_bands), ('PDC', pdc_bands)]:
        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        axes = axes.ravel()
        for ax_i, band_name in enumerate(band_list):
            mat = band_dict[band_name]
            sns.heatmap(mat, ax=axes[ax_i], cmap='viridis', square=True,
                        xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
                        vmin=0, vmax=1,
                        cbar_kws={'label': 'Connectivity (diagonal = 0)'})
            f_lo, f_hi = BANDS[band_name]
            axes[ax_i].set_title(f'{band_name}  {f_lo}-{f_hi} Hz', fontsize=11, fontweight='bold')
            axes[ax_i].set_xlabel('Source  (From j)', fontsize=8)
            axes[ax_i].set_ylabel('Sink  (To i)',     fontsize=8)
            axes[ax_i].tick_params(axis='x', rotation=90, labelsize=6)
            axes[ax_i].tick_params(axis='y', rotation=0,  labelsize=6)
        dim = 'bright cols = sources' if metric == 'DTF' else 'bright rows = sinks'
        fig.suptitle(f'{metric}  all bands  |  {subject_name}  p={fixed_order}  |  ({dim})',
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
    """
    AR(1) with:
      - Diagonal self-coupling 0.30
      - HUB1 (Cz)  -> C3, C4, T4, Pz, P4        coefficient 0.72
      - HUB2 (T4)  -> T5, P3, T6                 coefficient 0.70
      - CHAIN Fp1->F3->Fz->F4->F8                coefficient 0.68
    T4 is both a HUB2 source AND a HUB1 sink —
    this creates the indirect Cz->T4->T5/P3/T6 path that
    DTF will detect (bright Cz column extends to those rows)
    but PDC will NOT (PDC only shows direct T4->T5 etc.).
    """
    K   = len(CHANNEL_NAMES)
    rng = np.random.default_rng(SEED)

    A = np.diag([0.30] * K).astype(float)

    for t1 in HUB1_TGTS:
        A[t1-1, HUB1_SRC-1] = 0.72

    for t1 in HUB2_TGTS:
        A[t1-1, HUB2_SRC-1] = 0.70

    for i in range(len(CHAIN)-1):
        A[CHAIN[i+1]-1, CHAIN[i]-1] = 0.68

    eigvals = np.abs(np.linalg.eigvals(A))
    print(f"  AR matrix spectral radius: {eigvals.max():.4f}  (must be < 1)")
    assert eigvals.max() < 1.0, "AR matrix unstable!"

    X = np.zeros((N_SAMPLES, K))
    X[0] = rng.standard_normal(K)
    for t in range(1, N_SAMPLES):
        X[t] = A @ X[t-1] + 0.10 * rng.standard_normal(K)

    return X


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PLOT_DIR.mkdir(exist_ok=True)
    SUBJECT = "synth_19ch_hub"

    print("=" * 65)
    print("19-Channel Hub+Chain Synthetic Test")
    print("=" * 65)
    print("Ground-truth connections:")
    print(f"  HUB1 Cz(10)  -> {[CHANNEL_NAMES[t-1] for t in HUB1_TGTS]}")
    print(f"  HUB2 T4(12)  -> {[CHANNEL_NAMES[t-1] for t in HUB2_TGTS]}")
    print(f"  CHAIN        -> {[CHANNEL_NAMES[c-1] for c in CHAIN]}")
    print()
    print("Expected pattern:")
    print("  DTF : bright full COLUMNS at Cz, T4, Fp1")
    print("        (Cz column extends to T5/P3/T6 via indirect Cz->T4->...)")
    print("  PDC : bright isolated ROWS at direct sinks only")
    print("        NO full-column brightness")

    print(f"\nGenerating {N_SAMPLES:,}-sample AR(1) signal ...", flush=True)
    X = generate_signal()
    print(f"  Shape  : {X.shape}")
    print(f"  Std/ch : min={X.std(axis=0).min():.3f}  max={X.std(axis=0).max():.3f}")

    print(f"\nFitting MVAR(p={FIXED_ORDER}) ...")
    result = process_single_epoch(
        X.T, fs=FS, fixed_order=FIXED_ORDER, nfft=NFFT,
        verify=True, epoch_idx=0,
    )

    if result is None:
        print("\n[ERROR] process_single_epoch returned None.")
        raise SystemExit(1)

    print(f"\nSaving heatmaps to {PLOT_DIR.resolve()} ...")
    save_diagnostic_plot(
        result['dtf_bands']['integrated'],
        result['pdc_bands']['integrated'],
        FIXED_ORDER, SUBJECT, PLOT_DIR, band='integrated',
    )
    save_all_bands(result['dtf_bands'], result['pdc_bands'],
                   FIXED_ORDER, SUBJECT, PLOT_DIR)

    # Recovery summary
    K       = len(CHANNEL_NAMES)
    pdc_int = result['pdc_bands']['integrated']
    dtf_int = result['dtf_bands']['integrated']
    THRESH  = 0.10

    print("\n" + "=" * 65)
    print(f"GROUND-TRUTH RECOVERY  (integrated, threshold={THRESH})")
    print("=" * 65)
    print(f"  {'Connection':<30} {'PDC':>7}  {'DTF':>7}  {'PDC ok?':>8}  {'DTF ok?':>8}")
    print("  " + "-" * 65)
    for (s0, t0), (s1, t1) in zip(GT_0IDX, GT_1IDX):
        pv  = pdc_int[t0, s0]
        dv  = dtf_int[t0, s0]
        name = f"{CHANNEL_NAMES[s0]}({s1})->{CHANNEL_NAMES[t0]}({t1})"
        print(f"  {name:<30}  {pv:>6.3f}   {dv:>6.3f}"
              f"   {'YES' if pv>=THRESH else 'NO':>8}   {'YES' if dv>=THRESH else 'NO':>8}")

    # Indirect paths: Cz -> T5/P3/T6 (via T4)
    print("\nIndirect paths (Cz->T4->X) — DTF should detect, PDC should NOT:")
    indirect = [(9, 12), (9, 13), (9, 16)]   # Cz->T5, Cz->P3, Cz->T6  (0-idx)
    for s0, t0 in indirect:
        pv = pdc_int[t0, s0]
        dv = dtf_int[t0, s0]
        print(f"  {CHANNEL_NAMES[s0]}->{CHANNEL_NAMES[t0]:<5}  PDC={pv:.3f} (should be LOW)  "
              f"DTF={dv:.3f} (should be HIGH)")

    bg = [pdc_int[r,c] for r in range(K) for c in range(K)
          if (c,r) not in GT_0IDX and r!=c]
    print(f"\n  Noise floor PDC mean={np.mean(bg):.4f}")
    print("=" * 65)
    print(f"\nDone. Plots in: {PLOT_DIR.resolve()}")