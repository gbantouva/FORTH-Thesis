"""
Temporal Channel Evolution — Per-Channel Time Series
=====================================================
Produces ONE plot per subject: 19-channel grid showing total connectivity
strength over time with shared y-axis, mean lines, and significance markers.

All epochs are kept (pre-ictal, ictal, post-ictal).
Time axis taken directly from time_from_onset stored in the .npz.

Usage:
------
python step3a_temporal_channel_evolution.py \
    --connectivity_dir F:\FORTH_Final_Thesis\FORTH-Thesis\connectivity \
    --output_dir       F:\FORTH_Final_Thesis\FORTH-Thesis\figures\temporal \
    --band integrated \
    --metric pdc \
    --subject_ids 1 2 3
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

CHANNELS = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2',
]


def compute_total_strength(connectivity):
    """
    Total strength per channel per epoch = outflow + inflow.

    connectivity : (n_epochs, n_channels, n_channels)
      matrix[i, j] = source j -> sink i

    axis=1 sums over sink dimension   -> outflow per source channel
    axis=2 sums over source dimension -> inflow per sink channel

    Returns : (n_epochs, n_channels)
    """
    outflow = connectivity.sum(axis=1)
    inflow  = connectivity.sum(axis=2)
    return outflow + inflow


def plot_timeseries(graphs_file, output_dir, band, metric):
    data         = np.load(graphs_file)
    subject_name = Path(graphs_file).stem.replace('_graphs', '')

    conn_key = f'{metric}_{band}'
    if conn_key not in data:
        print(f'  Key {conn_key} not found in {graphs_file.name}')
        return

    connectivity = data[conn_key]   # (n_epochs, 19, 19)
    labels       = data['labels']   # (n_epochs,)
    n_epochs     = len(labels)

    if not np.any(labels == 1):
        print(f'  No ictal epochs for {subject_name} — skipping')
        return

    # ------------------------------------------------------------------
    # Time axis
    # ------------------------------------------------------------------
    if 'time_from_onset' in data:
        time = data['time_from_onset']
    else:
        # Fallback: reconstruct from first ictal epoch position
        first_ictal_idx = np.where(labels == 1)[0][0]
        time = np.arange(n_epochs) * 4.0
        time = time - time[first_ictal_idx]

    first_ictal_idx = np.where(labels == 1)[0][0]
    last_ictal_idx  = np.where(labels == 1)[0][-1]

    print(f'  {subject_name}: {n_epochs} epochs total '
          f'(pre={int((labels==0).sum())}, ictal={int((labels==1).sum())})')

    # ------------------------------------------------------------------
    # Compute total strength
    # ------------------------------------------------------------------
    total_strength = compute_total_strength(connectivity)  # (n_epochs, 19)

    # ------------------------------------------------------------------
    # Shared y-axis limits across all 19 channels
    # ------------------------------------------------------------------
    global_ymin = np.min(total_strength)
    global_ymax = np.max(total_strength)
    y_range      = global_ymax - global_ymin
    global_ymin -= 0.05 * y_range
    global_ymax += 0.05 * y_range

    # ------------------------------------------------------------------
    # Plot — 19 subplots in a 5x4 grid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(5, 4, figsize=(20, 16))
    axes_flat  = axes.flatten()

    for ch_idx, ch_name in enumerate(CHANNELS):
        ax = axes_flat[ch_idx]

        ax.plot(time, total_strength[:, ch_idx],
                linewidth=1.5, color='steelblue')

        # Shade ictal period
        ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
                   alpha=0.3, color='red')

        # Seizure onset at t=0
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Mean lines + Mann-Whitney significance (pre vs ictal only)
        pre = total_strength[labels == 0, ch_idx]
        ict = total_strength[labels == 1, ch_idx]

        sig_marker = ''
        if len(ict) > 0 and len(pre) > 0:
            _, p_val   = mannwhitneyu(pre, ict, alternative='two-sided')
            sig_marker = ('***' if p_val < 0.001 else
                          '**'  if p_val < 0.01  else
                          '*'   if p_val < 0.05  else '')
            ax.axhline(pre.mean(), color='steelblue', linestyle='--',
                       alpha=0.6, linewidth=1.0)
            ax.axhline(ict.mean(), color='red', linestyle='--',
                       alpha=0.6, linewidth=1.0)

        ax.set_ylim([global_ymin, global_ymax])
        ax.set_title(f'{ch_name} {sig_marker}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time from seizure onset (s)', fontsize=8)
        ax.set_ylabel('Strength', fontsize=8)
        ax.grid(alpha=0.3)

    for idx in range(len(CHANNELS), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle(
        f'{subject_name} — Per-Channel Temporal Evolution\n'
        f'{metric.upper()} {band} band  |  '
        f'*** p<0.001  ** p<0.01  * p<0.05  (pre-ictal vs ictal)\n'
        f'Shared y-axis [{global_ymin:.2f}, {global_ymax:.2f}]  |  '
        f'Red shading = ictal period',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    out_path = output_dir / f'{subject_name}_timeseries_{metric}_{band}.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def main():
    parser = argparse.ArgumentParser(
        description='Per-channel time series of connectivity strength'
    )
    parser.add_argument('--connectivity_dir', required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--band',   default='integrated',
                        choices=['integrated','delta','theta','alpha','beta','gamma1'])
    parser.add_argument('--metric', default='pdc', choices=['pdc','dtf'])
    parser.add_argument('--subject_ids', nargs='+', type=int,
                        default=list(range(1, 35)))
    args = parser.parse_args()

    connectivity_dir = Path(args.connectivity_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('TEMPORAL CHANNEL EVOLUTION — TIME SERIES')
    print('=' * 70)
    print(f'Metric:   {args.metric.upper()}')
    print(f'Band:     {args.band}')
    print(f'Output:   {output_dir}')
    print(f'Subjects: {len(args.subject_ids)}')
    print('=' * 70)

    success, errors = 0, 0

    for subj_id in args.subject_ids:
        subject_name = f'subject_{subj_id:02d}'
        graphs_file  = connectivity_dir / f'{subject_name}_graphs.npz'

        if not graphs_file.exists():
            print(f'\n  Skip {subject_name}: file not found')
            errors += 1
            continue

        print(f'\nProcessing {subject_name}...')
        try:
            plot_timeseries(graphs_file, output_dir, args.band, args.metric)
            success += 1
        except Exception as e:
            import traceback
            print(f'  Error: {e}')
            traceback.print_exc()
            errors += 1

    print('\n' + '=' * 70)
    print(f'Success: {success}  Errors: {errors}')
    print('=' * 70)


if __name__ == '__main__':
    main()