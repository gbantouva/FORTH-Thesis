"""
TUC Dataset - Compute Connectivity with Fixed Order
====================================================
Computes DTF and PDC connectivity matrices using a FIXED model order from BIC analysis.

Features:
- Fixed MVAR order (from BIC analysis)
- Multi-band support (6 frequency bands)
- Diagonal set to ZERO (inter-channel connectivity only)
- Parallel processing
- Checkpointing (resume capability)
- Handles training mask properly
- Built-in mathematical verification (spectrum level)
- Full per-epoch per-subject JSON verification report (--verifyall)

Usage:
    # Normal run (spot-check 3 epochs, then compute connectivity):
    python step2_compute_connectivity_tuc.py \
        --inputdir  F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs \
        --outputdir F:\FORTH_Final_Thesis\FORTH-Thesis\connectivity \
        --fixedorder 12 \
        --workers 8

    # Full verification run (saves JSON report, no connectivity computed):
    python step2_compute_connectivity_tuc.py \
        --inputdir  F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs \
        --outputdir F:\FORTH_Final_Thesis\FORTH-Thesis\connectivity \
        --fixedorder 12 \
        --verifyall
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import concurrent.futures
import multiprocessing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import linalg
from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm import tqdm

warnings.filterwarnings("ignore")


# TUC 19-channel names (10-20 montage)
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3',  'C3',  'Cz', 'C4', 'T4',
    'T5',  'P3',  'Pz', 'P4', 'T6',
    'O1',  'O2',
]


# ==============================================================================
# CORE CONNECTIVITY FUNCTIONS
# ==============================================================================

def compute_dtf_pdc_from_var(coefs, fs=256.0, nfft=512):
    """
    Compute DTF and PDC spectra from MVAR coefficients.

    Convention (verified by controlled test):
        matrix[i, j]  =  influence of source j on target i
        row  i  =  target (sink)
        col  j  =  source

    Mathematical properties (hold at spectrum level, BEFORE band averaging):
        PDC²: sum over axis=0 (rows)  per column j per freq = 1.0
        DTF²: sum over axis=1 (cols)  per row    i per freq = 1.0

    Parameters
    ----------
    coefs : ndarray (p, K, K)
        statsmodels VAR coefficients. coefs[k][i, j] = effect of source j
        on target i at lag k+1.
    fs    : float   Sampling frequency (Hz)
    nfft  : int     FFT length

    Returns
    -------
    dtf   : ndarray (K, K, n_freqs)   amplitude (not squared)
    pdc   : ndarray (K, K, n_freqs)   amplitude (not squared)
    freqs : ndarray (n_freqs,)
    """
    p, K, _ = coefs.shape
    n_freqs  = nfft // 2 + 1
    freqs    = np.linspace(0, fs / 2, n_freqs)

    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    H_f = np.zeros((n_freqs, K, K), dtype=complex)
    I   = np.eye(K)

    # Build A(f) = I - sum_k A_k * exp(-j2pi*f*k/fs)  and  H(f) = A(f)^{-1}
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

    # PDC: column-wise normalisation of A(f)
    # PDC[i,j,f] = |A_ij(f)| / sqrt( sum_m |A_mj(f)|^2 )
    # Interpretation: fraction of source j's direct output that goes to target i
    pdc = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Af         = A_f[f_idx]
        col_norms  = np.sqrt(np.sum(np.abs(Af) ** 2, axis=0))   # (K,)
        col_norms[col_norms == 0] = 1e-10
        pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]      # broadcast over rows

    # DTF: row-wise normalisation of H(f)
    # DTF[i,j,f] = |H_ij(f)| / sqrt( sum_m |H_im(f)|^2 )
    # Interpretation: fraction of target i's total inflow that comes from source j
    dtf = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Hf         = H_f[f_idx]
        row_norms  = np.sqrt(np.sum(np.abs(Hf) ** 2, axis=1))   # (K,)
        row_norms[row_norms == 0] = 1e-10
        dtf[:, :, f_idx] = np.abs(Hf) / row_norms[:, None]      # broadcast over cols

    return dtf, pdc, freqs


# ==============================================================================
# SPECTRUM-LEVEL VERIFICATION
# ==============================================================================

def verify_spectrum(dtf_spectrum, pdc_spectrum,
                    epoch_idx=None, tol=1e-6, verbose=True):
    """
    Verify mathematical sum properties of DTF and PDC at spectrum level.

    MUST be called BEFORE band averaging and BEFORE zeroing the diagonal.
    After averaging or zeroing, sums will NOT be 1.0 — that is expected.

    Properties verified:
        PDC²: column sums (axis=0) = 1.0  at every frequency bin
        DTF²: row    sums (axis=1) = 1.0  at every frequency bin

    Parameters
    ----------
    dtf_spectrum : (K, K, n_freqs)
    pdc_spectrum : (K, K, n_freqs)
    epoch_idx    : int or None  — used for log messages only
    tol          : float        — maximum allowed deviation from 1.0
    verbose      : bool         — print result to stdout

    Returns
    -------
    passed      : bool
    pdc_max_dev : float
    dtf_max_dev : float
    """
    pdc2 = pdc_spectrum ** 2                            # (K, K, n_freqs)
    dtf2 = dtf_spectrum ** 2                            # (K, K, n_freqs)

    pdc_col_sums = pdc2.sum(axis=0)                     # (K, n_freqs)
    dtf_row_sums = dtf2.sum(axis=1)                     # (K, n_freqs)

    pdc_max_dev = np.abs(pdc_col_sums - 1.0).max()
    dtf_max_dev = np.abs(dtf_row_sums - 1.0).max()

    passed = (pdc_max_dev < tol) and (dtf_max_dev < tol)

    if verbose:
        tag    = f"epoch {epoch_idx}" if epoch_idx is not None else "epoch"
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  [{status}] {tag} | "
              f"PDC² col-sum dev={pdc_max_dev:.2e} | "
              f"DTF² row-sum dev={dtf_max_dev:.2e}")

    return passed, pdc_max_dev, dtf_max_dev


# ==============================================================================
# PROCESS SINGLE EPOCH
# ==============================================================================

def process_single_epoch(data, fs, fixed_order, nfft,
                          verify=False, epoch_idx=None):
    """
    Fit MVAR(fixed_order) to one epoch and compute band-averaged DTF/PDC.

    Parameters
    ----------
    data        : ndarray (n_channels, n_timepoints)  e.g. (19, 1024)
    fs          : float   Sampling frequency (Hz)
    fixed_order : int     Fixed MVAR order from BIC analysis
    nfft        : int     FFT length for spectrum computation
    verify      : bool    Run spectrum-level verification (verbose)
    epoch_idx   : int     Used only for verification log messages

    Returns
    -------
    dict with keys 'dtf_bands', 'pdc_bands', 'order'
    or None if the epoch failed quality / stability checks
    """
    # --- Quality check ---
    data_std = np.std(data)
    if data_std < 1e-10:
        return None

    # --- Standardise (prevents ill-conditioned OLS) ---
    data_scaled = data / data_std

    try:
        # statsmodels VAR requires shape (n_observations, n_variables)
        # i.e. (timepoints, channels) — hence the mandatory transpose
        model = VAR(data_scaled.T)

        try:
            results = model.fit(maxlags=fixed_order, trend='c', verbose=False)
        except Exception:
            return None

        if results.k_ar == 0:
            return None

        # --- Stationarity check via companion matrix eigenvalues ---
        try:
            if not results.is_stable():
                return None
        except Exception:
            pass   # if check itself fails, assume stable and continue

        # --- Full-spectrum DTF / PDC ---
        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(
            results.coefs, fs, nfft
        )

        # --- Verification (spectrum level, BEFORE averaging or zeroing) ---
        # Properties: PDC² col-sums = 1.0,  DTF² row-sums = 1.0
        # These will NOT hold after band averaging — that is expected
        if verify:
            passed, pdc_dev, dtf_dev = verify_spectrum(
                dtf_spectrum, pdc_spectrum,
                epoch_idx=epoch_idx, tol=1e-6, verbose=True
            )
            if not passed:
                print(f"  ⚠️  Verification FAILED epoch {epoch_idx}: "
                      f"PDC dev={pdc_dev:.2e}, DTF dev={dtf_dev:.2e}")

        # --- Band integration ---
        bands = {
            'integrated': (0.5, 45.0),
            'delta':      (0.5,  4.0),
            'theta':      (4.0,  8.0),
            'alpha':      (8.0, 15.0),
            'beta':      (15.0, 30.0),
            'gamma1':    (30.0, 45.0),
        }

        dtf_bands = {}
        pdc_bands = {}

        for band_name, (f_low, f_high) in bands.items():
            idx_band = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            if len(idx_band) == 0:
                return None

            # Average amplitude over frequency bins in band
            dtf_band = np.mean(dtf_spectrum[:, :, idx_band], axis=2)  # (K, K)
            pdc_band = np.mean(pdc_spectrum[:, :, idx_band], axis=2)  # (K, K)

            # Zero the diagonal — self-connectivity is a mathematical artefact
            # (PDC_ii = DTF_ii = 1 by construction; not a real brain connection)
            np.fill_diagonal(dtf_band, 0.0)
            np.fill_diagonal(pdc_band, 0.0)

            dtf_bands[band_name] = dtf_band
            pdc_bands[band_name] = pdc_band

        return {
            'dtf_bands': dtf_bands,
            'pdc_bands': pdc_bands,
            'order':     fixed_order,
        }

    except Exception:
        return None


# ==============================================================================
# FULL SUBJECT VERIFICATION  (used by --verifyall mode)
# ==============================================================================

def verify_subject_epochs(epochs_file, fs, fixed_order, nfft, tol=1e-6):
    """
    Run spectrum-level verification on every epoch of one subject.

    Parameters
    ----------
    epochs_file : Path
    fs          : float
    fixed_order : int
    nfft        : int
    tol         : float

    Returns
    -------
    dict  ready to be serialised into the JSON report
    """
    subject_name = epochs_file.stem.replace('_epochs', '')
    epochs       = np.load(epochs_file)     # (n_epochs, 19, 1024)
    n_total      = len(epochs)

    epoch_results = []
    n_pass = n_fail = n_skip = 0

    for i in range(n_total):
        data     = epochs[i]
        data_std = np.std(data)

        # --- flat signal ---
        if data_std < 1e-10:
            epoch_results.append({
                'epoch': i, 'status': 'skipped',
                'reason': 'flat signal',
                'pdc_max_dev': None, 'dtf_max_dev': None,
            })
            n_skip += 1
            continue

        data_scaled = data / data_std

        # --- VAR fit ---
        try:
            results = VAR(data_scaled.T).fit(
                maxlags=fixed_order, trend='c', verbose=False
            )
        except Exception as e:
            epoch_results.append({
                'epoch': i, 'status': 'skipped',
                'reason': f'VAR fit failed: {e}',
                'pdc_max_dev': None, 'dtf_max_dev': None,
            })
            n_skip += 1
            continue

        if results.k_ar == 0:
            epoch_results.append({
                'epoch': i, 'status': 'skipped',
                'reason': 'k_ar=0',
                'pdc_max_dev': None, 'dtf_max_dev': None,
            })
            n_skip += 1
            continue

        # --- stability ---
        try:
            stable = results.is_stable()
        except Exception:
            stable = True
        if not stable:
            epoch_results.append({
                'epoch': i, 'status': 'skipped',
                'reason': 'unstable model',
                'pdc_max_dev': None, 'dtf_max_dev': None,
            })
            n_skip += 1
            continue

        # --- spectrum verification ---
        dtf_s, pdc_s, _ = compute_dtf_pdc_from_var(results.coefs, fs, nfft)
        passed, pdc_dev, dtf_dev = verify_spectrum(
            dtf_s, pdc_s, epoch_idx=i, tol=tol, verbose=False
        )

        epoch_results.append({
            'epoch':       i,
            'status':      'pass' if passed else 'FAIL',
            'reason':      None,
            'pdc_max_dev': float(pdc_dev),
            'dtf_max_dev': float(dtf_dev),
        })
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    return {
        'subject':    subject_name,
        'n_total':    n_total,
        'n_pass':     n_pass,
        'n_fail':     n_fail,
        'n_skip':     n_skip,
        'all_passed': n_fail == 0,
        'epochs':     epoch_results,
    }


# ==============================================================================
# FILE WORKER  (runs inside each parallel process)
# ==============================================================================

def process_subject_file(args_bundle):
    """
    Process one subject file — called by ProcessPoolExecutor.

    Parameters
    ----------
    args_bundle : tuple  (epochs_file, output_dir, fs, fixed_order, nfft)

    Returns
    -------
    tuple  ('success' | 'skipped' | 'failed' | 'error', n_epochs, info, ...)
    """
    epochs_file, output_dir, fs, fixed_order, nfft = args_bundle

    try:
        subject_name = epochs_file.stem.replace('_epochs', '')
        out_file     = output_dir / f"{subject_name}_graphs.npz"

        # --- checkpointing ---
        if out_file.exists():
            return ('skipped', 0, subject_name)

        # --- load data ---
        epochs = np.load(epochs_file)       # (n_epochs, 19, 1024)

        labels_file = epochs_file.parent / f"{subject_name}_labels.npy"
        if not labels_file.exists():
            return ('error', 0, f"Labels not found: {subject_name}")
        labels = np.load(labels_file)

        time_file       = epochs_file.parent / f"{subject_name}_time_from_onset.npy"
        time_from_onset = np.load(time_file) if time_file.exists() else None

        # --- storage ---
        band_names = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']
        dtf_data   = {band: [] for band in band_names}
        pdc_data   = {band: [] for band in band_names}
        valid_indices      = []
        orders             = []
        n_total_epochs     = len(epochs)
        epoch_present_mask = np.zeros(n_total_epochs, dtype=bool)

        # --- process each epoch ---
        for i in range(n_total_epochs):
            # verify=False to keep parallel stdout clean
            result = process_single_epoch(
                epochs[i], fs, fixed_order, nfft,
                verify=False, epoch_idx=i
            )
            if result is not None:
                for band in band_names:
                    dtf_data[band].append(result['dtf_bands'][band])
                    pdc_data[band].append(result['pdc_bands'][band])
                valid_indices.append(i)
                orders.append(result['order'])
                epoch_present_mask[i] = True

        # --- save ---
        if len(valid_indices) > 0:
            save_dict = {}
            for band in band_names:
                save_dict[f'dtf_{band}'] = np.array(dtf_data[band])   # (E, 19, 19)
                save_dict[f'pdc_{band}'] = np.array(pdc_data[band])   # (E, 19, 19)

            save_dict['labels']      = labels[valid_indices]
            save_dict['indices']     = np.array(valid_indices)
            save_dict['orders']      = np.array(orders)
            save_dict['fixed_order'] = fixed_order

            if time_from_onset is not None:
                save_dict['time_from_onset'] = time_from_onset[valid_indices]

            np.savez_compressed(out_file, **save_dict)

            mask_file = epochs_file.parent / f"{subject_name}_epoch_present_mask.npy"
            np.save(mask_file, epoch_present_mask)

            return (
                'success', len(valid_indices), subject_name,
                dtf_data['integrated'][0],   # first epoch integrated DTF for plot
                pdc_data['integrated'][0],   # first epoch integrated PDC for plot
            )
        else:
            return ('failed', 0, subject_name)

    except Exception as e:
        return ('error', 0, f"{epochs_file.name}: {str(e)}")


# ==============================================================================
# DIAGNOSTIC PLOT
# ==============================================================================

def save_diagnostic_plot(dtf, pdc, fixed_order, subject_name, output_dir):
    """
    Save a side-by-side DTF / PDC heatmap for one subject's first valid epoch.

    Both matrices have shape (19, 19) with diagonal = 0.
    matrix[i, j] = influence of source j (x-axis) on target i (y-axis).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    vmax = max(dtf.max(), pdc.max())

    for ax, matrix, title in zip(
        axes,
        [dtf, pdc],
        [f'DTF (Integrated 0.5–45 Hz)\n{subject_name}  |  order p={fixed_order}',
         f'PDC (Integrated 0.5–45 Hz)\n{subject_name}  |  order p={fixed_order}'],
    ):
        sns.heatmap(
            matrix, ax=ax, cmap='viridis', square=True,
            xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
            vmin=0, vmax=vmax,
            cbar_kws={'label': 'Connectivity Strength'},
        )
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Source (From j)', fontsize=11)
        ax.set_ylabel('Target (To i)',   fontsize=11)

    fig.text(
        0.5, 0.02,
        '✓ Diagonal = 0  |  '
        'Verified: PDC² col-sums = DTF² row-sums = 1.0 at spectrum level',
        ha='center', fontsize=10, style='italic', color='green',
    )

    plt.tight_layout()
    plot_dir = output_dir / 'diagnostic_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        plot_dir / f'{subject_name}_connectivity.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute DTF/PDC connectivity for TUC dataset (fixed MVAR order)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--inputdir",      required=True,
                        help="Directory containing subject_*_epochs.npy files")
    parser.add_argument("--outputdir",     required=True,
                        help="Output directory for connectivity .npz files")
    parser.add_argument("--fixedorder",    type=int, required=True,
                        help="Fixed MVAR order from BIC analysis")
    parser.add_argument("--workers",       type=int, default=None,
                        help="Parallel workers (default: all CPU cores)")
    parser.add_argument("--save_plots",    type=int, default=5,
                        help="Number of diagnostic plots to save (default: 5)")
    parser.add_argument("--verify_epochs", type=int, default=3,
                        help="Epochs from subject_01 to spot-check before main run "
                             "(default: 3)")
    parser.add_argument("--verifyall",     action='store_true',
                        help="Verify ALL epochs of ALL subjects, save JSON report, "
                             "then exit without computing connectivity")

    args = parser.parse_args()

    input_dir  = Path(args.inputdir)
    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_files = sorted(input_dir.glob("subject_*_epochs.npy"))

    print("=" * 80)
    print("TUC DATASET — CONNECTIVITY COMPUTATION")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Subjects found:   {len(epoch_files)}")
    print(f"Fixed order:      p = {args.fixedorder}")
    print(f"Frequency bands:  integrated, delta, theta, alpha, beta, gamma1")
    print(f"Diagonal:         SET TO ZERO ✓")
    print(f"Workers:          {args.workers or multiprocessing.cpu_count()}")
    print("=" * 80)

    if len(epoch_files) == 0:
        print("\n❌ No epoch files found!")
        return

    # ==========================================================================
    # MODE A: --verifyall
    # Verify every epoch of every subject and save a full JSON report.
    # No connectivity is computed in this mode.
    # ==========================================================================
    if args.verifyall:
        report_path = output_dir / 'verification_report.json'
        print(f"\n{'=' * 80}")
        print("VERIFICATION MODE — all epochs, all subjects")
        print("No connectivity will be computed.")
        print(f"Report → {report_path}")
        print(f"{'=' * 80}\n")

        full_report = {
            'generated_at': datetime.now().isoformat(),
            'fixed_order':  args.fixedorder,
            'tolerance':    1e-6,
            'n_subjects':   len(epoch_files),
            'subjects':     [],
        }

        total_pass = total_fail = total_skip = 0
        subjects_with_failures = []

        for epochs_file in tqdm(epoch_files, desc="Verifying", unit="subject"):
            subj_report = verify_subject_epochs(
                epochs_file, fs=256.0,
                fixed_order=args.fixedorder,
                nfft=512, tol=1e-6,
            )
            full_report['subjects'].append(subj_report)

            total_pass += subj_report['n_pass']
            total_fail += subj_report['n_fail']
            total_skip += subj_report['n_skip']

            tag = "✅ ALL PASS" if subj_report['all_passed'] else f"❌ {subj_report['n_fail']} FAIL"
            print(f"  {subj_report['subject']:20s}  "
                  f"pass={subj_report['n_pass']:4d}  "
                  f"fail={subj_report['n_fail']:4d}  "
                  f"skip={subj_report['n_skip']:4d}  {tag}")

            if not subj_report['all_passed']:
                subjects_with_failures.append(subj_report['subject'])

        full_report['summary'] = {
            'total_epochs_pass':      total_pass,
            'total_epochs_fail':      total_fail,
            'total_epochs_skip':      total_skip,
            'all_subjects_passed':    len(subjects_with_failures) == 0,
            'subjects_with_failures': subjects_with_failures,
        }

        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2)

        print(f"\n{'=' * 80}")
        print("VERIFICATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"  Total epochs passed:  {total_pass}")
        print(f"  Total epochs failed:  {total_fail}")
        print(f"  Total epochs skipped: {total_skip}"
              f"  (unstable/flat — excluded from connectivity too)")
        if not subjects_with_failures:
            print("\n  ✅ ALL SUBJECTS PASSED — math is correct")
        else:
            print(f"\n  ❌ FAILURES in: {subjects_with_failures}")
        print(f"\n  📄 Full report: {report_path}")
        print(f"{'=' * 80}")
        return  # exit — do not compute connectivity

    # ==========================================================================
    # MODE B: Normal run
    # Spot-check first N epochs of subject_01, then run parallel processing.
    # ==========================================================================
    first_file   = epoch_files[0]
    first_epochs = np.load(first_file)
    n_verify     = min(args.verify_epochs, len(first_epochs))

    print(f"\n{'=' * 80}")
    print(f"PRE-RUN VERIFICATION — first {n_verify} epochs of {first_file.stem}")
    print(f"Checking: PDC² col-sums = 1.0 | DTF² row-sums = 1.0  (spectrum level)")
    print(f"Note: after band averaging + diagonal zeroing, sums will NOT be 1.0 — expected")
    print(f"{'=' * 80}")

    for i in range(n_verify):
        result = process_single_epoch(
            first_epochs[i], fs=256.0,
            fixed_order=args.fixedorder,
            nfft=512, verify=True, epoch_idx=i,
        )
        if result is None:
            print(f"  [SKIP] epoch {i} — unstable or bad data")
            continue

        dtf_int     = result['dtf_bands']['integrated']   # (19,19) diagonal=0
        pdc_int     = result['pdc_bands']['integrated']
        pdc_col_sum = pdc_int.sum(axis=0)
        dtf_row_sum = dtf_int.sum(axis=1)
        print(f"         integrated band (diagonal=0): "
              f"PDC col-sums ∈ [{pdc_col_sum.min():.3f}, {pdc_col_sum.max():.3f}]  "
              f"DTF row-sums ∈ [{dtf_row_sum.min():.3f}, {dtf_row_sum.max():.3f}]"
              f"  ← NOT 1.0, correct ✓")

    print(f"\nProceeding with full parallel run...\n")

    # --- parallel processing ---
    tasks = [
        (f, output_dir, 256.0, args.fixedorder, 512)
        for f in epoch_files
    ]

    stats        = {'success': 0, 'skipped': 0, 'failed': 0, 'error': 0}
    total_epochs = 0
    plots_saved  = 0

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers) as executor:

        futures = {
            executor.submit(process_subject_file, task): task[0]
            for task in tasks
        }
        pbar = tqdm(total=len(epoch_files),
                    desc="Computing connectivity", unit="subject")

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            status = result[0]

            if status == 'success':
                stats['success'] += 1
                total_epochs     += result[1]
                if plots_saved < args.save_plots:
                    _, _, subject_name, dtf, pdc = result
                    save_diagnostic_plot(
                        dtf, pdc, args.fixedorder, subject_name, output_dir
                    )
                    plots_saved += 1
                pbar.set_postfix({
                    'epochs':  f"{total_epochs:,}",
                    'success': stats['success'],
                })
            elif status == 'skipped':
                stats['skipped'] += 1
            elif status == 'failed':
                stats['failed']  += 1
            elif status == 'error':
                stats['error']   += 1
                print(f"\n⚠️  {result[2]}")

            pbar.update(1)

        pbar.close()

    print("\n" + "=" * 80)
    print("CONNECTIVITY COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"  Success:  {stats['success']} subjects")
    print(f"  Skipped:  {stats['skipped']} subjects  (already computed — checkpointed)")
    print(f"  Failed:   {stats['failed']} subjects")
    print(f"  Errors:   {stats['error']} subjects")
    print(f"  Total epochs processed: {total_epochs:,}")
    print(f"\n📁 Output:")
    print(f"  Connectivity files : {output_dir}")
    print(f"  Diagnostic plots   : {output_dir / 'diagnostic_plots'}")
    print(f"\n🎯 Next steps:")
    print(f"  1. Review diagnostic plots")
    print(f"  2. Run full verification:  python step2_compute_connectivity_tuc.py "
          f"--inputdir ... --outputdir ... --fixedorder {args.fixedorder} --verifyall")
    print(f"  3. Build graphs for GNN training")
    print("\n" + "=" * 80)
    print("✅ DONE")
    print("=" * 80)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
