"""
Step 3 — Connectivity Visualization (Thesis Figures)
=====================================================
Produces three figures from the .npz files written by step2_compute_connectivity.py:

  Figure A  (grand_average_per_band.png)
      Grand-average DTF and PDC connectivity matrices, pre-ictal vs ictal,
      for every frequency band. One 6x4 grid per dataset.
      Template: Geng et al., J. Neurosci. Methods 2019, Fig. 3.

  Figure B  (channel_strength.png)
      In-strength (sink) and out-strength (source) per channel,
      pre-ictal vs ictal, for the integrated band.
      Shown as (1) grouped bar plot and (2) scalp topomap pair.
      Template: Narasimhan et al., 2020, Fig. 3 + Wilke et al., Epilepsia 2010, Fig. 3.

  Figure C  (time_locked_connectivity.png)
      Global mean DTF/PDC strength as a function of time-from-seizure-onset,
      averaged across subjects, ictal window shaded.
      Template: van Mierlo et al., NeuroImage 2013.

USAGE
-----
  # Standard: read real step2 outputs
  python step3_visualize_connectivity.py \
      --inputdir path/to/connectivity_results \
      --outputdir path/to/figures

  # Test without real data: generate synthetic .npz files and run on those
  python step3_visualize_connectivity.py --demo --outputdir demo_figures

CONVENTIONS
-----------
  matrix[i, j] = influence of SOURCE j on SINK i  (step2 convention)
  DTF outdegree of channel j  =  column-sum of DTF[:, j]   (how strongly j drives)
  DTF indegree  of channel i  =  row-sum    of DTF[i, :]   (= 1 at spectrum level,
                                                            != 1 after band avg)
  PDC outdegree of channel j  =  column-sum of PDC[:, j]   (= 1 at spectrum level)
  PDC indegree  of channel i  =  row-sum    of PDC[i, :]   (how strongly i is driven)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2',
]

# Bands in the order step2 saves them (matches BANDS dict there)
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma1', 'integrated']
BAND_LABEL = {
    'delta':      r'Delta (0.5–4 Hz)',
    'theta':      r'Theta (4–8 Hz)',
    'alpha':      r'Alpha (8–15 Hz)',
    'beta':       r'Beta (15–30 Hz)',
    'gamma1':     r'Gamma (30–45 Hz)',
    'integrated': r'Broadband (0.5–45 Hz)',
}

# Approximate 10–20 electrode positions on the unit disk (x, y).
# Used only for topomap; not the projection MNE uses, but close enough for a
# one-off figure and avoids introducing an MNE dependency.
# y>0 = front of head.
EEG_XY = {
    'Fp1': (-0.30,  0.92), 'Fp2': ( 0.30,  0.92),
    'F7':  (-0.80,  0.58), 'F3':  (-0.45,  0.58), 'Fz': ( 0.00,  0.58),
    'F4':  ( 0.45,  0.58), 'F8':  ( 0.80,  0.58),
    'T3':  (-0.95,  0.00), 'C3':  (-0.50,  0.00), 'Cz': ( 0.00,  0.00),
    'C4':  ( 0.50,  0.00), 'T4':  ( 0.95,  0.00),
    'T5':  (-0.80, -0.58), 'P3':  (-0.45, -0.58), 'Pz': ( 0.00, -0.58),
    'P4':  ( 0.45, -0.58), 'T6':  ( 0.80, -0.58),
    'O1':  (-0.30, -0.92), 'O2':  ( 0.30, -0.92),
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_subjects(input_dir: Path):
    """
    Read every <subject>_graphs.npz under input_dir and stack into big arrays.

    Returns
    -------
    dict with:
        'dtf'       : {band: (N_epochs_total, K, K)}
        'pdc'       : {band: (N_epochs_total, K, K)}
        'labels'    : (N_epochs_total,)   0=non-ictal, 1=ictal
        'tfo'       : (N_epochs_total,)   time from onset (seconds), or None if missing
        'subject'   : (N_epochs_total,)   string array of subject names
        'n_subjects': int
    """
    npz_files = sorted(input_dir.glob('subject_*_graphs.npz'))
    if not npz_files:
        raise FileNotFoundError(
            f'No subject_*_graphs.npz found in {input_dir}. '
            f'Run step2_compute_connectivity.py first.'
        )

    dtf_acc = {b: [] for b in BAND_ORDER}
    pdc_acc = {b: [] for b in BAND_ORDER}
    lab_acc = []
    tfo_acc = []
    sub_acc = []
    any_tfo_missing = False

    for f in npz_files:
        name = f.stem.replace('_graphs', '')
        with np.load(f, allow_pickle=True) as d:
            keys = set(d.files)
            for b in BAND_ORDER:
                k = f'dtf_{b}'
                if k in keys:
                    dtf_acc[b].append(d[k])
                    pdc_acc[b].append(d[f'pdc_{b}'])
            labels = d['labels']
            lab_acc.append(labels)
            if 'time_from_onset' in keys:
                tfo_acc.append(d['time_from_onset'])
            else:
                any_tfo_missing = True
                tfo_acc.append(np.full(len(labels), np.nan, dtype=np.float32))
            sub_acc.append(np.full(len(labels), name, dtype=object))

    dtf = {b: np.concatenate(dtf_acc[b], axis=0) for b in BAND_ORDER if dtf_acc[b]}
    pdc = {b: np.concatenate(pdc_acc[b], axis=0) for b in BAND_ORDER if pdc_acc[b]}
    labels = np.concatenate(lab_acc, axis=0).astype(int)
    tfo = np.concatenate(tfo_acc, axis=0).astype(float) if not any_tfo_missing else None
    subject = np.concatenate(sub_acc, axis=0)

    return {
        'dtf': dtf,
        'pdc': pdc,
        'labels': labels,
        'tfo': tfo,
        'subject': subject,
        'n_subjects': len(npz_files),
    }


# ============================================================================
# SYNTHETIC DATA (for --demo mode)
# ============================================================================

def make_synthetic_npz(output_dir: Path, n_subjects: int = 5, seed: int = 0):
    """
    Write fake step2 output files for testing the visualization pipeline
    without needing real data. Creates plausible-looking DTF/PDC matrices
    with a mild ictal-vs-preictal contrast around channels Fp1/F7/T3
    (pretending those are the focus).
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    K = 19
    focus = [0, 2, 7]  # Fp1, F7, T3 — pretend SOZ

    for s in range(1, n_subjects + 1):
        n_pre = rng.integers(60, 90)
        n_ict = rng.integers(10, 25)
        n = n_pre + n_ict
        labels = np.concatenate([np.zeros(n_pre), np.ones(n_ict)]).astype(int)
        # Time from onset: pre-ictal negative, ictal positive
        tfo = np.concatenate([
            np.linspace(-n_pre * 4, -4, n_pre),  # 4-s epochs before onset
            np.linspace(0, (n_ict - 1) * 4, n_ict),
        ]).astype(np.float32)

        save_dict = {}
        for b in BAND_ORDER:
            dtf = np.zeros((n, K, K))
            pdc = np.zeros((n, K, K))
            for e in range(n):
                base = 0.15 + 0.05 * rng.standard_normal((K, K))
                np.fill_diagonal(base, 0.0)
                # Ictal: focus channels drive others more strongly
                if labels[e] == 1:
                    for j in focus:
                        base[:, j] += 0.25 + 0.05 * rng.standard_normal(K)
                base = np.clip(base, 0, 1)
                np.fill_diagonal(base, 0.0)
                dtf[e] = base
                # PDC: make it look slightly different (less total flow, more focal)
                pdc[e] = 0.6 * base + 0.05 * rng.standard_normal((K, K))
                pdc[e] = np.clip(pdc[e], 0, 1)
                np.fill_diagonal(pdc[e], 0.0)
            save_dict[f'dtf_{b}'] = dtf.astype(np.float32)
            save_dict[f'pdc_{b}'] = pdc.astype(np.float32)

        save_dict['labels'] = labels
        save_dict['indices'] = np.arange(n)
        save_dict['orders'] = np.full(n, 12)
        save_dict['fixed_order'] = 12
        save_dict['time_from_onset'] = tfo

        out = output_dir / f'subject_{s:02d}_graphs.npz'
        np.savez_compressed(out, **save_dict)

    print(f'  Wrote {n_subjects} synthetic .npz files to {output_dir}')


# ============================================================================
# FIGURE A — Grand-average DTF/PDC matrices, pre vs ictal, per band
# ============================================================================

def figure_A_grand_average(data: dict, outpath: Path):
    """6 rows (bands) x 4 cols (DTF-pre, DTF-ict, PDC-pre, PDC-ict) of heatmaps."""
    labels = data['labels']
    pre_mask = labels == 0
    ict_mask = labels == 1

    bands = [b for b in BAND_ORDER if b in data['dtf']]
    n_rows = len(bands)

    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 3.4 * n_rows),
                             squeeze=False)

    col_titles = ['DTF — Pre-ictal', 'DTF — Ictal',
                  'PDC — Pre-ictal', 'PDC — Ictal']

    # Precompute all means so we can use a shared color scale per measure
    means = {}
    for b in bands:
        means[(b, 'dtf', 0)] = data['dtf'][b][pre_mask].mean(axis=0) if pre_mask.any() else np.zeros((19, 19))
        means[(b, 'dtf', 1)] = data['dtf'][b][ict_mask].mean(axis=0) if ict_mask.any() else np.zeros((19, 19))
        means[(b, 'pdc', 0)] = data['pdc'][b][pre_mask].mean(axis=0) if pre_mask.any() else np.zeros((19, 19))
        means[(b, 'pdc', 1)] = data['pdc'][b][ict_mask].mean(axis=0) if ict_mask.any() else np.zeros((19, 19))

    # Shared color scale across DTF and PDC columns
    dtf_vmax = max(m.max() for k, m in means.items() if k[1] == 'dtf')
    pdc_vmax = max(m.max() for k, m in means.items() if k[1] == 'pdc')

    for row, band in enumerate(bands):
        for col, (meas, state) in enumerate([('dtf', 0), ('dtf', 1),
                                             ('pdc', 0), ('pdc', 1)]):
            ax = axes[row, col]
            mat = means[(band, meas, state)]
            vmax = dtf_vmax if meas == 'dtf' else pdc_vmax
            sns.heatmap(
                mat, ax=ax, cmap='viridis', square=True,
                vmin=0, vmax=vmax, cbar=(col == 3 or col == 1),
                xticklabels=CHANNEL_NAMES if row == n_rows - 1 else False,
                yticklabels=CHANNEL_NAMES if col == 0 else False,
            )
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(BAND_LABEL[band] + '\nsink i',
                              fontsize=10, fontweight='bold')
            if row == n_rows - 1:
                ax.set_xlabel('source j', fontsize=9)
            ax.tick_params(axis='x', rotation=90, labelsize=6)
            ax.tick_params(axis='y', rotation=0,  labelsize=6)

    n_pre = int(pre_mask.sum())
    n_ict = int(ict_mask.sum())
    fig.suptitle(
        f'Grand-average connectivity matrices per band '
        f'(N = {data["n_subjects"]} subjects, {n_pre:,} pre-ictal + {n_ict:,} ictal epochs)\n'
        f'Rows = sinks, columns = sources. Diagonal zeroed.',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {outpath.name}')


# ============================================================================
# FIGURE B — Channel in/out-strength, pre vs ictal
# ============================================================================

def figure_B_channel_strength(data: dict, outpath: Path, band: str = 'integrated'):
    """Bar plot + scalp topomap of in/out-strength per channel, pre vs ictal."""
    if band not in data['dtf']:
        raise KeyError(f'Band {band!r} not in data. Available: {list(data["dtf"])}')

    labels = data['labels']
    pre = labels == 0
    ict = labels == 1

    dtf_mat = data['dtf'][band]  # (N, K, K)
    pdc_mat = data['pdc'][band]

    # Outdegree of channel j = column-sum over sinks i  (mean over epochs)
    # Indegree  of channel i = row-sum    over sources j
    def outdeg(M):  # (N, K, K) -> (K,)
        return M.sum(axis=1).mean(axis=0)

    def indeg(M):
        return M.sum(axis=2).mean(axis=0)

    dtf_out_pre = outdeg(dtf_mat[pre]) if pre.any() else np.zeros(19)
    dtf_out_ict = outdeg(dtf_mat[ict]) if ict.any() else np.zeros(19)
    pdc_in_pre  = indeg(pdc_mat[pre]) if pre.any() else np.zeros(19)
    pdc_in_ict  = indeg(pdc_mat[ict]) if ict.any() else np.zeros(19)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15],
                          hspace=0.35, wspace=0.25)

    # --- Row 1: grouped bar plots --------------------------------------------
    x = np.arange(19)
    w = 0.4

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - w/2, dtf_out_pre, w, label='Pre-ictal', color='steelblue',
            edgecolor='black', linewidth=0.4)
    ax1.bar(x + w/2, dtf_out_ict, w, label='Ictal', color='firebrick',
            edgecolor='black', linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=9)
    ax1.set_ylabel('DTF out-strength', fontsize=11, fontweight='bold')
    ax1.set_title('DTF out-strength per channel  (how strongly a channel drives the network)',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - w/2, pdc_in_pre, w, label='Pre-ictal', color='steelblue',
            edgecolor='black', linewidth=0.4)
    ax2.bar(x + w/2, pdc_in_ict, w, label='Ictal', color='firebrick',
            edgecolor='black', linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=9)
    ax2.set_ylabel('PDC in-strength', fontsize=11, fontweight='bold')
    ax2.set_title('PDC in-strength per channel  (how strongly a channel is driven)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # --- Row 2: scalp topomaps -----------------------------------------------
    # Ictal - Pre (difference) for the two measures — this is what a reviewer
    # wants to see: where does seizure change driving/receiving?
    dtf_diff = dtf_out_ict - dtf_out_pre
    pdc_diff = pdc_in_ict - pdc_in_pre

    ax3 = fig.add_subplot(gs[1, 0])
    _draw_topomap(ax3, dtf_diff,
                  title='DTF out-strength:  Ictal − Pre-ictal',
                  cmap='RdBu_r', symmetric=True)

    ax4 = fig.add_subplot(gs[1, 1])
    _draw_topomap(ax4, pdc_diff,
                  title='PDC in-strength:  Ictal − Pre-ictal',
                  cmap='RdBu_r', symmetric=True)

    fig.suptitle(
        f'Channel-level driving and receiving  —  {BAND_LABEL[band]}\n'
        f'(N = {data["n_subjects"]} subjects, {int(pre.sum()):,} pre-ictal + {int(ict.sum()):,} ictal epochs)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {outpath.name}')


def _draw_topomap(ax, values, title: str, cmap: str = 'viridis',
                  symmetric: bool = False):
    """
    Minimal scalp topomap: interpolates channel values on the unit disk
    using Gaussian RBF and draws a head outline with electrode markers.
    """
    from scipy.interpolate import Rbf

    xs = np.array([EEG_XY[c][0] for c in CHANNEL_NAMES])
    ys = np.array([EEG_XY[c][1] for c in CHANNEL_NAMES])

    # Interpolation grid
    ng = 120
    xi = np.linspace(-1.1, 1.1, ng)
    yi = np.linspace(-1.1, 1.1, ng)
    Xi, Yi = np.meshgrid(xi, yi)

    rbf = Rbf(xs, ys, values, function='multiquadric', smooth=0.05)
    Zi = rbf(Xi, Yi)

    # Mask outside head
    mask = np.sqrt(Xi**2 + Yi**2) > 1.0
    Zi_masked = np.ma.array(Zi, mask=mask)

    if symmetric:
        vmax = np.abs(values).max() + 1e-9
        vmin = -vmax
    else:
        vmin, vmax = values.min(), values.max()

    im = ax.contourf(Xi, Yi, Zi_masked, levels=40, cmap=cmap,
                     vmin=vmin, vmax=vmax)
    # Head outline, nose, ears
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5)
    # Nose
    ax.plot([-0.08, 0, 0.08], [1.0, 1.10, 1.0], 'k-', linewidth=1.5)
    # Ears
    for sign in (-1, 1):
        ear_t = np.linspace(-np.pi/3, np.pi/3, 30)
        ax.plot(sign * (1 + 0.05 * np.cos(ear_t)),
                0.15 * np.sin(ear_t), 'k-', linewidth=1.5)

    # Electrode dots + labels
    ax.plot(xs, ys, 'k.', markersize=4)
    for c, x, y in zip(CHANNEL_NAMES, xs, ys):
        ax.text(x, y + 0.06, c, ha='center', va='bottom', fontsize=7)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ============================================================================
# FIGURE C — Time-locked connectivity vs time-from-onset
# ============================================================================

def figure_C_time_locked(data: dict, outpath: Path,
                         bin_width_sec: float = 8.0):
    """Global mean DTF/PDC (off-diagonal) vs time-from-onset, averaged over epochs."""
    if data['tfo'] is None:
        print('  ! time_from_onset missing for some subjects — skipping Figure C.')
        return

    tfo = data['tfo']
    # Bin time-from-onset
    t_lo, t_hi = np.nanmin(tfo), np.nanmax(tfo)
    if not np.isfinite(t_lo) or not np.isfinite(t_hi):
        print('  ! time_from_onset has no finite values — skipping Figure C.')
        return

    edges = np.arange(t_lo - bin_width_sec, t_hi + bin_width_sec, bin_width_sec)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def global_mean(M):
        # M: (N, K, K) -> global mean of off-diagonals (diag is already 0)
        K = M.shape[-1]
        return M.sum(axis=(1, 2)) / (K * (K - 1))

    dtf_global = global_mean(data['dtf']['integrated'])
    pdc_global = global_mean(data['pdc']['integrated'])

    # Bin along time
    def bin_stats(series):
        mean_per_bin = np.full(len(centers), np.nan)
        sem_per_bin  = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            sel = (tfo >= edges[i]) & (tfo < edges[i + 1])
            if sel.sum() >= 2:
                mean_per_bin[i] = series[sel].mean()
                sem_per_bin[i]  = series[sel].std(ddof=1) / np.sqrt(sel.sum())
            elif sel.sum() == 1:
                mean_per_bin[i] = series[sel].mean()
                sem_per_bin[i]  = 0.0
        return mean_per_bin, sem_per_bin

    dtf_mean, dtf_sem = bin_stats(dtf_global)
    pdc_mean, pdc_sem = bin_stats(pdc_global)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, mean, sem, color, title in [
        (axes[0], dtf_mean, dtf_sem, 'steelblue',
         'DTF — global mean off-diagonal connectivity vs time from seizure onset'),
        (axes[1], pdc_mean, pdc_sem, 'firebrick',
         'PDC — global mean off-diagonal connectivity vs time from seizure onset'),
    ]:
        ok = np.isfinite(mean)
        ax.fill_between(centers[ok], (mean - sem)[ok], (mean + sem)[ok],
                        color=color, alpha=0.25)
        ax.plot(centers[ok], mean[ok], color=color, linewidth=2)
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2,
                   label='Seizure onset (t = 0)')
        ax.axvspan(0, max(centers[ok][-1], 0) if ok.any() else 0,
                   color='firebrick', alpha=0.07, label='Ictal (t > 0)')
        ax.set_ylabel('Mean connectivity', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

    axes[-1].set_xlabel('Time from seizure onset (seconds)',
                        fontsize=11, fontweight='bold')
    fig.suptitle(
        f'Time-locked connectivity (N = {data["n_subjects"]} subjects, '
        f'{bin_width_sec:.0f} s bins, shaded = SEM)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {outpath.name}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Produce thesis figures from step2 connectivity .npz files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--inputdir',  type=str, default=None,
                        help='Folder with subject_*_graphs.npz files')
    parser.add_argument('--outputdir', type=str, required=True,
                        help='Folder to write figures into')
    parser.add_argument('--band', type=str, default='integrated',
                        choices=BAND_ORDER,
                        help='Band for Figure B (default: integrated)')
    parser.add_argument('--bin_width_sec', type=float, default=8.0,
                        help='Time bin for Figure C (default: 8 s = two 4-s epochs)')
    parser.add_argument('--demo', action='store_true',
                        help='Generate synthetic .npz files and use those')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        demo_dir = output_dir / '_demo_npz'
        print(f'[DEMO] writing synthetic data to {demo_dir}')
        make_synthetic_npz(demo_dir, n_subjects=5)
        input_dir = demo_dir
    else:
        if args.inputdir is None:
            parser.error('--inputdir is required unless --demo is set.')
        input_dir = Path(args.inputdir)

    print('=' * 72)
    print('STEP 3 — THESIS FIGURES FROM CONNECTIVITY DATA')
    print('=' * 72)
    print(f'  Input:  {input_dir}')
    print(f'  Output: {output_dir}')
    print('=' * 72)

    print('\nLoading all subjects...')
    data = load_all_subjects(input_dir)
    n_total = len(data['labels'])
    print(f'  Subjects: {data["n_subjects"]}   Epochs: {n_total:,}   '
          f'Pre: {(data["labels"] == 0).sum():,}   Ict: {(data["labels"] == 1).sum():,}')
    print(f'  Bands available: {list(data["dtf"])}')

    print('\nFigure A — Grand-average matrices, pre vs ictal, per band')
    figure_A_grand_average(data, output_dir / 'figA_grand_average_per_band.png')

    print(f'\nFigure B — Channel strength, pre vs ictal ({args.band} band)')
    figure_B_channel_strength(data, output_dir / 'figB_channel_strength.png',
                              band=args.band)

    print(f'\nFigure C — Time-locked connectivity (bin = {args.bin_width_sec:.0f} s)')
    figure_C_time_locked(data, output_dir / 'figC_time_locked_connectivity.png',
                         bin_width_sec=args.bin_width_sec)

    print('\n' + '=' * 72)
    print('DONE')
    print('=' * 72)


if __name__ == '__main__':
    main()
