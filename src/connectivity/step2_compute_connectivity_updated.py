"""
Step 2 — Compute DTF/PDC Connectivity (Fixed MVAR Order)
=========================================================
Computes DTF and PDC connectivity matrices per epoch using a fixed MVAR
order chosen from BIC analysis (step1).

KEY DESIGN DECISIONS (documented here for thesis):
──────────────────────────────────────────────────
1. FIXED order: one order for all subjects/epochs (from BIC mode in step1).
   Avoids per-epoch order selection which adds variance.

2. Normalization checked at SPECTRUM LEVEL (before band averaging):
     DTF²[i, :, f].sum(axis=1) = 1  for all sinks i and all freqs f
     PDC²[:, j, f].sum(axis=0) = 1  for all sources j and all freqs f
   After averaging over a band these sums will NOT be 1 — that is expected.

3. Diagonal set to ZERO AFTER band averaging:
   At spectrum level, DTF[i,i,f] = PDC[i,i,f] = 1 by construction.
   Self-connectivity is a mathematical artefact, not a brain connection.
   Zeroing BEFORE averaging would break the spectrum-level normalization check.

4. matrix[i, j] = influence of SOURCE j on SINK i
   rows = sinks,  columns = sources
   → DTF bright columns = strong source channels  (correct, not a bug)
   → PDC bright rows    = strong sink channels    (correct, not a bug)

5. All epochs computed (including post-ictal).
   Labels + indices saved so GNN loader can apply training_mask.

Usage:
    python step2_compute_connectivity.py \\
        --inputdir   path/to/preprocessed_epochs \\
        --outputdir  path/to/connectivity \\
        --fixedorder 12 \\
        --workers    8

    # Verify normalization on every epoch before computing:
    python step2_compute_connectivity.py ... --verifyall
"""

import argparse
import json
import warnings
import multiprocessing
import concurrent.futures
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Channel names ──────────────────────────────────────────────────────────────
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2',
]

# ── Frequency bands ────────────────────────────────────────────────────────────
BANDS = {
    'integrated': (0.5, 45.0),
    'delta':      (0.5,  4.0),
    'theta':      (4.0,  8.0),
    'alpha':      (8.0, 15.0),
    'beta':      (15.0, 30.0),
    'gamma1':    (30.0, 45.0),
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. CORE MATH
# ══════════════════════════════════════════════════════════════════════════════

def compute_dtf_pdc_from_var(coefs, fs=256.0, nfft=512):
    """
    Compute DTF and PDC spectra from MVAR coefficients.

    Parameters
    ----------
    coefs : ndarray (p, K, K)
        statsmodels coefs[k][i,j] = effect of source j on target i at lag k+1
    fs    : float   sampling frequency (Hz)
    nfft  : int     FFT length

    Returns
    -------
    dtf   : ndarray (K, K, n_freqs)  — amplitude, NOT squared
    pdc   : ndarray (K, K, n_freqs)  — amplitude, NOT squared
    freqs : ndarray (n_freqs,)

    Convention: matrix[i, j] = source j → sink i
    Normalization (spectrum level, before band avg):
        DTF²[i,:,f].sum(axis=0) = 1  ∀ i, f   (row norm via H)
        PDC²[:,j,f].sum(axis=0) = 1  ∀ j, f   (col norm via A)
    """
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

    # PDC: column-normalised |A(f)|   →  col sums of PDC² = 1
    pdc = np.zeros((K, K, n_freqs))
    for fi in range(n_freqs):
        col_norms = np.sqrt((np.abs(A_f[fi]) ** 2).sum(axis=0))   # (K,)
        col_norms[col_norms < 1e-10] = 1e-10
        pdc[:, :, fi] = np.abs(A_f[fi]) / col_norms[np.newaxis, :]

    # DTF: row-normalised |H(f)|   →  row sums of DTF² = 1
    dtf = np.zeros((K, K, n_freqs))
    for fi in range(n_freqs):
        row_norms = np.sqrt((np.abs(H_f[fi]) ** 2).sum(axis=1))   # (K,)
        row_norms[row_norms < 1e-10] = 1e-10
        dtf[:, :, fi] = np.abs(H_f[fi]) / row_norms[:, np.newaxis]

    return dtf, pdc, freqs


# ══════════════════════════════════════════════════════════════════════════════
# 2. SPECTRUM-LEVEL VERIFICATION
#    Call this BEFORE band averaging and BEFORE zeroing diagonal.
# ══════════════════════════════════════════════════════════════════════════════

def verify_spectrum(dtf_s, pdc_s, tol=1e-6):
    """
    Verify normalization at spectrum level.

    dtf_s, pdc_s : (K, K, n_freqs)

    DTF² row sums (axis=1, over sources) = 1  for every sink and freq
    PDC² col sums (axis=0, over sinks)   = 1  for every source and freq

    Returns: (passed: bool, dtf_max_dev: float, pdc_max_dev: float)
    """
    dtf2 = dtf_s ** 2   # (K, K, n_freqs)
    pdc2 = pdc_s ** 2

    dtf_row_sums = dtf2.sum(axis=1)   # (K, n_freqs) — sum over sources
    pdc_col_sums = pdc2.sum(axis=0)   # (K, n_freqs) — sum over sinks

    dtf_dev = float(np.abs(dtf_row_sums - 1.0).max())
    pdc_dev = float(np.abs(pdc_col_sums - 1.0).max())

    passed = dtf_dev < tol and pdc_dev < tol
    return passed, dtf_dev, pdc_dev


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROCESS ONE EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def process_single_epoch(data, fs, fixed_order, nfft, verify=False, epoch_idx=None):
    """
    Fit MVAR(fixed_order) to one epoch and return band-averaged DTF/PDC.

    Parameters
    ----------
    data        : ndarray (K, T)   e.g. (19, 1024)
    fs          : float
    fixed_order : int
    nfft        : int
    verify      : bool   run spectrum-level normalization check
    epoch_idx   : int    only used for log messages

    Returns
    -------
    dict with 'dtf_bands', 'pdc_bands', 'order', 'dtf_spectrum', 'pdc_spectrum', 'freqs'
    or None if the epoch is rejected (flat / unstable / VAR failed)
    """
    data_std = np.std(data)
    if data_std < 1e-10:          # flat signal — reject
        return None

    data_scaled = data / data_std  # global standardisation (scale cancels in DTF/PDC)

    try:
        model   = VAR(data_scaled.T)   # statsmodels expects (T, K)
        results = model.fit(maxlags=fixed_order, trend='c', verbose=False)

        if results.k_ar == 0:
            return None

        try:
            if not results.is_stable():
                return None
        except Exception:
            pass

        # ── full-spectrum DTF / PDC ─────────────────────────────────────────
        dtf_s, pdc_s, freqs = compute_dtf_pdc_from_var(results.coefs, fs, nfft)

        # ── VERIFY at spectrum level (BEFORE averaging, BEFORE diag zero) ───
        if verify:
            passed, dtf_dev, pdc_dev = verify_spectrum(dtf_s, pdc_s)
            tag = "✅ PASS" if passed else "❌ FAIL"
            label = f"epoch {epoch_idx}" if epoch_idx is not None else "epoch"
            print(f"  [{tag}] {label} | DTF row-sum dev={dtf_dev:.2e} | "
                  f"PDC col-sum dev={pdc_dev:.2e}")

        # ── band averaging ──────────────────────────────────────────────────
        dtf_bands = {}
        pdc_bands = {}

        for band_name, (f_lo, f_hi) in BANDS.items():
            idx = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
            if len(idx) == 0:
                return None

            dtf_band = dtf_s[:, :, idx].mean(axis=2)   # (K, K)
            pdc_band = pdc_s[:, :, idx].mean(axis=2)   # (K, K)

            # ── zero diagonal AFTER averaging ───────────────────────────────
            # At spectrum level DTF[i,i,f] = PDC[i,i,f] = 1 (math artefact).
            # Zeroing before averaging would invalidate the normalization check.
            # Zeroing after averaging removes self-edges for the GNN.
            np.fill_diagonal(dtf_band, 0.0)
            np.fill_diagonal(pdc_band, 0.0)

            dtf_bands[band_name] = dtf_band
            pdc_bands[band_name] = pdc_band

        return {
            'dtf_bands':    dtf_bands,
            'pdc_bands':    pdc_bands,
            'order':        fixed_order,
            'dtf_spectrum': dtf_s,     # (K, K, n_freqs) — kept for visualization
            'pdc_spectrum': pdc_s,
            'freqs':        freqs,
        }

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUBJECT WORKER (parallel)
# ══════════════════════════════════════════════════════════════════════════════

def process_subject_file(args_bundle):
    epochs_file, output_dir, fs, fixed_order, nfft = args_bundle

    try:
        subject_name = epochs_file.stem.replace('_epochs', '')
        out_file     = output_dir / f"{subject_name}_graphs.npz"

        if out_file.exists():
            return ('skipped', 0, subject_name)

        epochs = np.load(epochs_file)     # (n_epochs, K, T)

        labels_file = epochs_file.parent / f"{subject_name}_labels.npy"
        if not labels_file.exists():
            return ('error', 0, f"Labels not found: {subject_name}")
        labels = np.load(labels_file)

        tfo_file        = epochs_file.parent / f"{subject_name}_time_from_onset.npy"
        time_from_onset = np.load(tfo_file) if tfo_file.exists() else None

        band_names = list(BANDS.keys())
        dtf_data   = {b: [] for b in band_names}
        pdc_data   = {b: [] for b in band_names}
        valid_idx  = []
        orders     = []
        n_total    = len(epochs)
        epoch_mask = np.zeros(n_total, dtype=bool)

        for i in range(n_total):
            result = process_single_epoch(
                epochs[i], fs, fixed_order, nfft,
                verify=False, epoch_idx=i,
            )
            if result is not None:
                for b in band_names:
                    dtf_data[b].append(result['dtf_bands'][b])
                    pdc_data[b].append(result['pdc_bands'][b])
                valid_idx.append(i)
                orders.append(result['order'])
                epoch_mask[i] = True

        if len(valid_idx) == 0:
            return ('failed', 0, subject_name)

        save_dict = {}
        for b in band_names:
            save_dict[f'dtf_{b}'] = np.array(dtf_data[b])   # (E, K, K)
            save_dict[f'pdc_{b}'] = np.array(pdc_data[b])

        save_dict['labels']      = labels[valid_idx]
        save_dict['indices']     = np.array(valid_idx)
        save_dict['orders']      = np.array(orders)
        save_dict['fixed_order'] = fixed_order
        if time_from_onset is not None:
            save_dict['time_from_onset'] = time_from_onset[valid_idx]

        np.savez_compressed(out_file, **save_dict)

        mask_path = epochs_file.parent / f"{subject_name}_epoch_present_mask.npy"
        np.save(mask_path, epoch_mask)

        return (
            'success', len(valid_idx), subject_name,
            dtf_data['integrated'][0],
            pdc_data['integrated'][0],
        )

    except Exception as e:
        return ('error', 0, f"{epochs_file.name}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FULL VERIFICATION MODE (--verifyall)
# ══════════════════════════════════════════════════════════════════════════════

def verify_subject_all_epochs(epochs_file, fs, fixed_order, nfft, tol=1e-6):
    subject_name = epochs_file.stem.replace('_epochs', '')
    epochs       = np.load(epochs_file)
    n_total      = len(epochs)

    n_pass = n_fail = n_skip = 0
    epoch_results = []

    for i in range(n_total):
        data     = epochs[i]
        data_std = np.std(data)

        if data_std < 1e-10:
            epoch_results.append({'epoch': i, 'status': 'skipped',
                                  'reason': 'flat', 'dtf_dev': None, 'pdc_dev': None})
            n_skip += 1
            continue

        try:
            res = VAR((data / data_std).T).fit(maxlags=fixed_order, trend='c', verbose=False)
        except Exception as e:
            epoch_results.append({'epoch': i, 'status': 'skipped',
                                  'reason': str(e), 'dtf_dev': None, 'pdc_dev': None})
            n_skip += 1
            continue

        if res.k_ar == 0:
            epoch_results.append({'epoch': i, 'status': 'skipped',
                                  'reason': 'k_ar=0', 'dtf_dev': None, 'pdc_dev': None})
            n_skip += 1
            continue

        try:
            stable = res.is_stable()
        except Exception:
            stable = True

        if not stable:
            epoch_results.append({'epoch': i, 'status': 'skipped',
                                  'reason': 'unstable', 'dtf_dev': None, 'pdc_dev': None})
            n_skip += 1
            continue

        dtf_s, pdc_s, _ = compute_dtf_pdc_from_var(res.coefs, fs, nfft)
        passed, dtf_dev, pdc_dev = verify_spectrum(dtf_s, pdc_s, tol=tol)

        epoch_results.append({
            'epoch': i, 'status': 'pass' if passed else 'FAIL',
            'reason': None, 'dtf_dev': dtf_dev, 'pdc_dev': pdc_dev,
        })
        if passed: n_pass += 1
        else:      n_fail += 1

    return {
        'subject': subject_name, 'n_total': n_total,
        'n_pass': n_pass, 'n_fail': n_fail, 'n_skip': n_skip,
        'all_passed': n_fail == 0,
        'epochs': epoch_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. DIAGNOSTIC PLOT (one per subject)
# ══════════════════════════════════════════════════════════════════════════════

def save_diagnostic_plot(dtf, pdc, fixed_order, subject_name, output_dir):
    """One heatmap pair for the first valid epoch of each subject."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    vmax = max(dtf.max(), pdc.max(), 1e-6)

    for ax, mat, title in [
        (axes[0], dtf, f'DTF  integrated 0.5–45 Hz\n{subject_name}  p={fixed_order}'),
        (axes[1], pdc, f'PDC  integrated 0.5–45 Hz\n{subject_name}  p={fixed_order}'),
    ]:
        sns.heatmap(mat, ax=ax, cmap='viridis', square=True,
                    xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
                    vmin=0, vmax=vmax,
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
    plot_dir = output_dir / 'diagnostic_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'{subject_name}_epoch0_connectivity.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute DTF/PDC connectivity — fixed MVAR order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--inputdir',      required=True)
    parser.add_argument('--outputdir',     required=True)
    parser.add_argument('--fixedorder',    type=int, required=True)
    parser.add_argument('--workers',       type=int, default=None)
    parser.add_argument('--save_plots',    type=int, default=5,
                        help='Number of diagnostic heatmaps to save (default 5)')
    parser.add_argument('--verify_epochs', type=int, default=3,
                        help='Spot-check first N epochs of subject_01 before run')
    parser.add_argument('--verifyall',     action='store_true',
                        help='Verify normalization for ALL epochs, save JSON, then exit')
    args = parser.parse_args()

    input_dir  = Path(args.inputdir)
    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_files = sorted(input_dir.glob('subject_*_epochs.npy'))

    print('=' * 72)
    print('STEP 2 — DTF/PDC CONNECTIVITY')
    print('=' * 72)
    print(f'  Input:        {input_dir}')
    print(f'  Output:       {output_dir}')
    print(f'  Subjects:     {len(epoch_files)}')
    print(f'  Fixed order:  p = {args.fixedorder}')
    print(f'  Bands:        {", ".join(BANDS)}')
    print(f'  Diagonal:     zeroed AFTER band averaging')
    print(f'  Workers:      {args.workers or multiprocessing.cpu_count()}')
    print('=' * 72)

    if len(epoch_files) == 0:
        print('\n❌  No epoch files found.')
        return

    # ── MODE A: --verifyall ─────────────────────────────────────────────────
    if args.verifyall:
        report_path = output_dir / 'verification_report.json'
        print(f'\nVERIFICATION MODE — checking all epochs of all subjects')
        print(f'Report → {report_path}\n')

        report = {
            'generated_at': datetime.now().isoformat(),
            'fixed_order':  args.fixedorder,
            'tolerance':    1e-6,
            'subjects':     [],
        }
        total_pass = total_fail = total_skip = 0
        bad_subjects = []

        for ef in tqdm(epoch_files, desc='Verifying', unit='subject'):
            sr = verify_subject_all_epochs(ef, 256.0, args.fixedorder, 512)
            report['subjects'].append(sr)
            total_pass += sr['n_pass']
            total_fail += sr['n_fail']
            total_skip += sr['n_skip']
            tag = '✅' if sr['all_passed'] else f"❌ {sr['n_fail']} FAIL"
            print(f"  {sr['subject']:20s}  pass={sr['n_pass']:4d}  "
                  f"fail={sr['n_fail']:4d}  skip={sr['n_skip']:4d}  {tag}")
            if not sr['all_passed']:
                bad_subjects.append(sr['subject'])

        report['summary'] = {
            'total_pass': total_pass, 'total_fail': total_fail,
            'total_skip': total_skip,
            'all_passed': len(bad_subjects) == 0,
            'bad_subjects': bad_subjects,
        }
        with open(report_path, 'w') as fh:
            json.dump(report, fh, indent=2)

        print(f'\n{"=" * 72}')
        print(f'  Passed: {total_pass}   Failed: {total_fail}   Skipped: {total_skip}')
        print(f'  {"✅ ALL PASS" if not bad_subjects else "❌ FAILURES: " + str(bad_subjects)}')
        print(f'  Report: {report_path}')
        return

    # ── MODE B: spot-check then compute ────────────────────────────────────
    first_epochs = np.load(epoch_files[0])
    n_verify     = min(args.verify_epochs, len(first_epochs))

    print(f'\nSpot-checking first {n_verify} epochs of {epoch_files[0].stem}')
    print('Normalization checked at SPECTRUM LEVEL (before band avg, before diag zero)')
    print('─' * 72)

    for i in range(n_verify):
        result = process_single_epoch(
            first_epochs[i], fs=256.0,
            fixed_order=args.fixedorder, nfft=512,
            verify=True, epoch_idx=i,
        )
        if result is None:
            print(f'  [SKIP] epoch {i}')
            continue
        dtf_int = result['dtf_bands']['integrated']
        pdc_int = result['pdc_bands']['integrated']
        print(f'         integrated (diag=0): '
              f'DTF row-sums ∈ [{dtf_int.sum(axis=1).min():.3f}, '
              f'{dtf_int.sum(axis=1).max():.3f}]  '
              f'PDC col-sums ∈ [{pdc_int.sum(axis=0).min():.3f}, '
              f'{pdc_int.sum(axis=0).max():.3f}]  ← NOT 1.0, expected ✓')

    print('\nProceed with full parallel computation...\n')

    tasks  = [(f, output_dir, 256.0, args.fixedorder, 512) for f in epoch_files]
    stats  = {'success': 0, 'skipped': 0, 'failed': 0, 'error': 0}
    total  = 0
    n_plot = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_subject_file, t): t[0] for t in tasks}
        pbar    = tqdm(total=len(epoch_files), desc='Computing', unit='subject')

        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            stats[r[0]] = stats.get(r[0], 0) + 1
            if r[0] == 'success':
                total += r[1]
                if n_plot < args.save_plots:
                    _, _, sname, dtf, pdc = r
                    save_diagnostic_plot(dtf, pdc, args.fixedorder, sname, output_dir)
                    n_plot += 1
                pbar.set_postfix({'epochs': f'{total:,}', 'ok': stats['success']})
            elif r[0] == 'error':
                print(f'\n⚠️  {r[2]}')
            pbar.update(1)
        pbar.close()

    print('\n' + '=' * 72)
    print('DONE')
    print('=' * 72)
    print(f"  Success: {stats['success']}   Skipped: {stats['skipped']}   "
          f"Failed: {stats.get('failed',0)}   Errors: {stats.get('error',0)}")
    print(f'  Total valid epochs: {total:,}')
    print(f'\n  Next: python step2_tests.py   # confirm math')
    print(f'        python step2_visualize.py  # inspect per-epoch plots')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()