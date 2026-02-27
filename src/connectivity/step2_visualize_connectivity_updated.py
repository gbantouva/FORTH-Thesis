"""
Step 2 — Per-Epoch Connectivity Visualization
===============================================
Produces visualizations for every epoch of every subject from the .npz
connectivity files created by step2_compute_connectivity.py.

Three plot types per epoch
──────────────────────────
A) Heatmap pair (DTF + PDC)  — integrated band, colour-fixed [0, 1]
B) Multi-band heatmap grid   — 2 rows (DTF / PDC) × 6 bands
C) Baccalá-style grid        — each cell [i,j] = freq curve
   (produced per-subject only for a few representative epochs to avoid
    thousands of large files — configurable via --baccala_epochs)

Output structure
────────────────
output_dir/
  subject_01/
    heatmap/
      ep000_preictal.png
      ep005_ictal.png
      ...
    multiband/
      ep000_preictal.png
      ...
    baccala/           ← only for selected epochs
      ep000_preictal_dtf.png
      ep000_preictal_pdc.png

Usage
─────
  # All subjects, heatmap + multiband:
  python step2_visualize.py \\
      --npzdir   path/to/connectivity \\
      --epochdir path/to/preprocessed_epochs \\
      --outdir   path/to/figures

  # Also make Baccalá grids for first pre-ictal and first ictal epoch:
  python step2_visualize.py ... --baccala_epochs auto

  # Specific epochs:
  python step2_visualize.py ... --baccala_epochs 0 5 10 72

  # Limit to a few subjects for a quick check:
  python step2_visualize.py ... --subjects subject_01 subject_02
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from tqdm import tqdm

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2',
]

BANDS = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']
BAND_LABELS = {
    'integrated': 'Integrated\n0.5–45 Hz',
    'delta':      'Delta\n0.5–4 Hz',
    'theta':      'Theta\n4–8 Hz',
    'alpha':      'Alpha\n8–15 Hz',
    'beta':       'Beta\n15–30 Hz',
    'gamma1':     'Gamma1\n30–45 Hz',
}

LABEL_NAMES = {0: 'preictal', 1: 'ictal'}
LABEL_COLORS = {0: '#2980b9', 1: '#c0392b'}


# ══════════════════════════════════════════════════════════════════════════════
# A) HEATMAP PAIR  (integrated band, one epoch)
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap_pair(dtf, pdc, subject_name, epoch_idx, label,
                      time_from_onset, order, out_path):
    """
    DTF and PDC heatmaps side by side, colour-fixed [0, 1].
    Both (19, 19) matrices with diagonal = 0.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    label_str  = LABEL_NAMES.get(label, 'unknown')
    lcolor     = LABEL_COLORS.get(label, 'gray')
    t_str      = f'{time_from_onset:+.1f}s from onset' if time_from_onset is not None else ''
    K = dtf.shape[0]
    chan = CHANNEL_NAMES[:K]

    for ax, mat, mname, note in [
        (axes[0], dtf, 'DTF',
         'Bright COLUMNS = strong source channels\n'
         'row-normalised  |  direct + indirect'),
        (axes[1], pdc, 'PDC',
         'Bright ROWS = strong sink channels\n'
         'col-normalised  |  direct connections only'),
    ]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Connectivity  (diagonal = 0)')
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(chan, fontsize=7 if K > 10 else 9, rotation=90)
        ax.set_yticklabels(chan, fontsize=7 if K > 10 else 9)
        ax.set_xlabel('Source  (From j)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Sink  (To i)',     fontsize=10, fontweight='bold')
        ax.set_title(f'{mname}\n{note}', fontsize=10, fontweight='bold')

        # Annotate values for small K
        if K <= 10:
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, f'{mat[i,j]:.2f}',
                            ha='center', va='center', fontsize=6,
                            color='white' if mat[i,j] > 0.5 else 'black')

    fig.suptitle(
        f'{subject_name}  |  Epoch {epoch_idx:03d}  |  '
        f'{label_str.upper()}  |  {t_str}  |  order p={order}\n'
        'Colour scale fixed [0, 1] — same across all epochs and subjects',
        fontsize=12, fontweight='bold',
        color=lcolor,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# B) MULTI-BAND HEATMAP GRID  (2 × 6)
# ══════════════════════════════════════════════════════════════════════════════

def plot_multiband_grid(dtf_bands, pdc_bands, subject_name, epoch_idx,
                        label, time_from_onset, order, out_path):
    """
    2 rows (DTF / PDC) × 6 columns (bands).
    Fixed colour scale [0, 1].
    Last column has a shared colour bar.
    """
    label_str = LABEL_NAMES.get(label, 'unknown')
    lcolor    = LABEL_COLORS.get(label, 'gray')
    t_str     = f'{time_from_onset:+.1f}s' if time_from_onset is not None else ''
    K = dtf_bands['integrated'].shape[0]
    chan = CHANNEL_NAMES[:K]
    n_bands = len(BANDS)

    fig = plt.figure(figsize=(3.8 * n_bands + 1, 8.5))
    gs  = gridspec.GridSpec(2, n_bands + 1,
                             width_ratios=[1]*n_bands + [0.06],
                             hspace=0.35, wspace=0.12)

    for row, (metric, bands_dict, row_label) in enumerate([
            ('DTF', dtf_bands, 'DTF  (direct + indirect)'),
            ('PDC', pdc_bands, 'PDC  (direct only)'),
    ]):
        for col, band in enumerate(BANDS):
            ax  = fig.add_subplot(gs[row, col])
            mat = bands_dict[band]
            im  = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight='bold')
            if row == 0:
                ax.set_title(BAND_LABELS[band], fontsize=8, fontweight='bold')

        # Shared colour bar in last column
        cax = fig.add_subplot(gs[row, -1])
        plt.colorbar(im, cax=cax, label='[0, 1]')
        cax.tick_params(labelsize=7)

    fig.suptitle(
        f'{subject_name}  |  Epoch {epoch_idx:03d}  |  '
        f'{label_str.upper()}  |  {t_str}  |  p={order}\n'
        '2 rows (DTF / PDC)  ×  6 frequency bands  |  fixed scale [0, 1]',
        fontsize=11, fontweight='bold', color=lcolor, y=1.02,
    )

    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# C) BACCALÁ GRID  (per metric — one figure per DTF, one per PDC)
#    Each cell [i,j] = freq curve for sink i ← source j
#    Requires spectrum (K, K, n_freqs); loaded from the epoch file
#    and recomputed — or pre-saved if you extend step2 to save spectra.
#
#    Because recomputing costs time, this is only done for selected epochs.
# ══════════════════════════════════════════════════════════════════════════════

def plot_baccala_grid(dtf_s, pdc_s, freqs, subject_name, epoch_idx,
                      label, time_from_onset, order, out_dir):
    """
    Baccalá-style: cell [sink_i, source_j] = connectivity vs frequency.
    dtf_s / pdc_s : (K, K, n_freqs)  — spectrum-level (NOT band-averaged)
    """
    K         = dtf_s.shape[0]
    chan      = CHANNEL_NAMES[:K]
    label_str = LABEL_NAMES.get(label, 'unknown')
    t_str     = f'{time_from_onset:+.1f}s' if time_from_onset is not None else ''

    for metric_s, mname, mcolor in [
            (dtf_s, 'DTF', 'steelblue'), (pdc_s, 'PDC', 'tomato')]:
        fig, axes = plt.subplots(K, K, figsize=(2.8 * K, 2.4 * K))
        fig.suptitle(
            f'{subject_name}  |  Epoch {epoch_idx:03d}  |  '
            f'{label_str.upper()}  |  {t_str}  |  p={order}\n'
            f'{mname}  —  Baccalá style  |  '
            'Row = Sink (To i),  Col = Source (From j)',
            fontsize=10, fontweight='bold', y=0.999,
        )

        for snk_i in range(K):
            for src_j in range(K):
                ax   = axes[snk_i, src_j]
                vals = metric_s[snk_i, src_j, :]

                is_self = snk_i == src_j
                if is_self:
                    ax.fill_between(freqs, 0, vals, alpha=0.2, color='gray')
                    ax.plot(freqs, vals, color='gray', lw=0.8)
                else:
                    ax.fill_between(freqs, 0, vals, alpha=0.3, color=mcolor)
                    ax.plot(freqs, vals, color=mcolor, lw=1.0)

                ax.set_ylim(0, 1.05)
                ax.set_xlim(freqs[0], freqs[-1])
                ax.set_xticks([]); ax.set_yticks([0, 0.5, 1])
                ax.tick_params(labelsize=4); ax.grid(alpha=0.2)

                if src_j == 0:
                    ax.set_ylabel(chan[snk_i], fontsize=7, fontweight='bold',
                                  rotation=0, labelpad=28, va='center')
                if snk_i == K - 1:
                    ax.set_xlabel(chan[src_j], fontsize=7, fontweight='bold')

        fig.text(0.02, 0.5, 'Sink  →', va='center', rotation='vertical',
                 fontsize=10, fontweight='bold', color='navy')
        fig.text(0.5, 0.005, 'Source  →', ha='center',
                 fontsize=10, fontweight='bold', color='navy')
        plt.tight_layout(rect=[0.04, 0.03, 1, 0.97])

        fname = out_dir / f'ep{epoch_idx:03d}_{label_str}_{mname.lower()}.png'
        fig.savefig(fname, dpi=110, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Subject-level overview  (all epochs timeline)
# ══════════════════════════════════════════════════════════════════════════════

def plot_subject_timeline(dtf_integrated, labels, time_from_onset,
                          subject_name, out_path):
    """
    For one subject: mean connectivity strength over time.
    Shows how overall DTF evolves across epochs, coloured by label.
    """
    K = dtf_integrated.shape[1]
    # Mean off-diagonal connectivity per epoch
    mask = ~np.eye(K, dtype=bool)
    mean_conn = np.array([dtf_integrated[e][mask].mean()
                          for e in range(len(dtf_integrated))])

    fig, ax = plt.subplots(figsize=(16, 4))
    t = time_from_onset if time_from_onset is not None else np.arange(len(mean_conn))

    # Colour each epoch by label
    for lbl, lname, lc in [(0, 'Pre-ictal', '#2980b9'), (1, 'Ictal', '#c0392b')]:
        idx = np.where(labels == lbl)[0]
        if len(idx) > 0:
            ax.scatter(t[idx], mean_conn[idx], c=lc, s=15, label=lname,
                       alpha=0.7, zorder=3)

    ax.plot(t, mean_conn, color='gray', lw=0.6, alpha=0.5, zorder=2)
    ax.axvline(0, color='red', lw=2, linestyle='--', label='Seizure onset', zorder=4)
    ax.set_xlabel('Time from seizure onset (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean DTF\n(off-diagonal)', fontsize=11, fontweight='bold')
    ax.set_title(f'{subject_name} — Connectivity over time  (DTF integrated band)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Visualize DTF/PDC connectivity per epoch per subject',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--npzdir',   required=True,
                        help='Directory with subject_XX_graphs.npz files')
    parser.add_argument('--outdir',   required=True,
                        help='Output directory for figures')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Specific subjects (e.g. subject_01 subject_02). '
                             'Default: all.')
    parser.add_argument('--plot_types', nargs='+',
                        default=['heatmap', 'multiband'],
                        choices=['heatmap', 'multiband', 'baccala', 'timeline'],
                        help='Which plots to produce (default: heatmap multiband)')
    parser.add_argument('--baccala_epochs', nargs='+', default=['auto'],
                        help='Epoch indices for Baccalá plots, or "auto" for '
                             'first pre-ictal + first ictal.  '
                             'Requires --plot_types baccala AND the original '
                             'epoch files via --epochdir.')
    parser.add_argument('--epochdir', default=None,
                        help='Directory with subject_XX_epochs.npy files '
                             '(needed for Baccalá recomputation)')
    parser.add_argument('--fixedorder', type=int, default=12,
                        help='MVAR order used in step2 (for Baccalá recompute)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum epochs to visualize per subject (default: all)')
    args = parser.parse_args()

    npz_dir  = Path(args.npzdir)
    out_dir  = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob('subject_*_graphs.npz'))
    if args.subjects:
        npz_files = [f for f in npz_files
                     if f.stem.replace('_graphs', '') in args.subjects]

    print('=' * 72)
    print('STEP 2 — CONNECTIVITY VISUALIZATION')
    print('=' * 72)
    print(f'  NPZ dir:     {npz_dir}')
    print(f'  Output:      {out_dir}')
    print(f'  Subjects:    {len(npz_files)}')
    print(f'  Plot types:  {args.plot_types}')
    print('=' * 72)

    do_heatmap   = 'heatmap'   in args.plot_types
    do_multiband = 'multiband' in args.plot_types
    do_baccala   = 'baccala'   in args.plot_types
    do_timeline  = 'timeline'  in args.plot_types

    # Baccalá needs spectrum → need to re-fit MVAR from epoch data
    if do_baccala:
        if args.epochdir is None:
            print('⚠️  --baccala requires --epochdir.  Skipping Baccalá plots.')
            do_baccala = False
        else:
            from statsmodels.tsa.vector_ar.var_model import VAR
            from step2_compute_connectivity import compute_dtf_pdc_from_var
            epoch_dir = Path(args.epochdir)

    for npz_file in tqdm(npz_files, desc='Subjects', unit='subject'):
        subject_name = npz_file.stem.replace('_graphs', '')
        data         = np.load(npz_file)

        n_epochs       = len(data['labels'])
        labels         = data['labels']
        indices        = data.get('indices', np.arange(n_epochs))
        tfo            = data.get('time_from_onset', None)
        orders         = data.get('orders', np.full(n_epochs, args.fixedorder))
        fixed_order    = int(data.get('fixed_order', args.fixedorder))

        ep_range = range(n_epochs)
        if args.max_epochs is not None:
            ep_range = range(min(n_epochs, args.max_epochs))

        # ── Subject output dirs ───────────────────────────────────────────
        subj_dir  = out_dir / subject_name
        heat_dir  = subj_dir / 'heatmap';    heat_dir.mkdir(parents=True, exist_ok=True)
        mband_dir = subj_dir / 'multiband';  mband_dir.mkdir(parents=True, exist_ok=True)
        bacc_dir  = subj_dir / 'baccala';    bacc_dir.mkdir(parents=True, exist_ok=True)

        # ── Timeline (one per subject) ────────────────────────────────────
        if do_timeline:
            dtf_int = data['dtf_integrated']
            plot_subject_timeline(
                dtf_int, labels, tfo,
                subject_name, subj_dir / 'timeline.png',
            )

        # ── Determine Baccalá epoch list ──────────────────────────────────
        if do_baccala:
            if args.baccala_epochs == ['auto']:
                bacc_ep_list = []
                pre_idx = np.where(labels == 0)[0]
                ict_idx = np.where(labels == 1)[0]
                if len(pre_idx) > 0: bacc_ep_list.append(int(pre_idx[0]))
                if len(ict_idx) > 0: bacc_ep_list.append(int(ict_idx[0]))
            else:
                bacc_ep_list = [int(x) for x in args.baccala_epochs
                                if int(x) < n_epochs]

            # Load original epochs for re-fitting
            ep_file = epoch_dir / f'{subject_name}_epochs.npy'
            if ep_file.exists():
                orig_epochs = np.load(ep_file)   # (n_total_orig, K, T)
            else:
                print(f'  ⚠️  epoch file not found for {subject_name}, skip Baccalá')
                bacc_ep_list = []

        # ── Per-epoch plots ───────────────────────────────────────────────
        for ep in tqdm(ep_range, desc=f'  {subject_name}', leave=False, unit='epoch'):
            lbl      = int(labels[ep])
            lbl_name = LABEL_NAMES.get(lbl, 'unknown')
            t_val    = float(tfo[ep]) if tfo is not None else None
            order    = int(orders[ep])

            dtf_bands = {b: data[f'dtf_{b}'][ep] for b in BANDS}
            pdc_bands = {b: data[f'pdc_{b}'][ep] for b in BANDS}

            # ── A: Heatmap ────────────────────────────────────────────────
            if do_heatmap:
                plot_heatmap_pair(
                    dtf_bands['integrated'],
                    pdc_bands['integrated'],
                    subject_name, ep, lbl, t_val, order,
                    heat_dir / f'ep{ep:03d}_{lbl_name}.png',
                )

            # ── B: Multi-band ─────────────────────────────────────────────
            if do_multiband:
                plot_multiband_grid(
                    dtf_bands, pdc_bands,
                    subject_name, ep, lbl, t_val, order,
                    mband_dir / f'ep{ep:03d}_{lbl_name}.png',
                )

            # ── C: Baccalá (selected epochs only) ────────────────────────
            if do_baccala and ep in bacc_ep_list and ep_file.exists():
                orig_idx   = int(indices[ep])
                epoch_data = orig_epochs[orig_idx]   # (K, T)
                data_std   = np.std(epoch_data)
                if data_std > 1e-10:
                    try:
                        from statsmodels.tsa.vector_ar.var_model import VAR
                        res = VAR((epoch_data / data_std).T).fit(
                            maxlags=fixed_order, trend='c', verbose=False)
                        if res.k_ar > 0:
                            dtf_s, pdc_s, freqs = compute_dtf_pdc_from_var(
                                res.coefs, fs=256.0, nfft=512)
                            plot_baccala_grid(
                                dtf_s, pdc_s, freqs,
                                subject_name, ep, lbl, t_val, order,
                                bacc_dir,
                            )
                    except Exception as exc:
                        print(f'    Baccalá ep{ep} failed: {exc}')

    print(f'\n{"=" * 72}')
    print(f'  Done.  Figures saved to: {out_dir}')
    print(f'{"=" * 72}')
    print()
    print('  Folder structure:')
    print('    subject_XX/')
    print('      heatmap/   ep000_preictal.png  ep005_ictal.png  ...')
    print('      multiband/ ep000_preictal.png  ...')
    print('      baccala/   ep000_preictal_dtf.png  ep000_preictal_pdc.png  ...')
    print('      timeline.png   (connectivity over time)')
    print()
    print('  Colour scale [0, 1] is fixed across ALL epochs and subjects.')
    print('  Same colour = same connectivity value → direct comparison possible.')


if __name__ == '__main__':
    main()