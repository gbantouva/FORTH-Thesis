"""
Per-Subject Channel-Specific Analysis
======================================
Analyzes EACH subject individually to identify their specific focal channels.

Supports --direction argument to analyse outflow, inflow, or total separately:
  outflow — identifies DRIVERS (channels that send during seizure)
  inflow  — identifies FOLLOWERS (channels that receive during seizure)
  total   — outflow + inflow combined (overall involvement)

For focal epilepsy, outflow is the most clinically meaningful direction:
the epileptic focus becomes a driver and recruits other regions.

Matrix convention:
  connectivity[i, j] = FROM channel j TO channel i
  → axis=0 sums over sinks   → outflow per source (column sum)
  → axis=1 sums over sources → inflow  per sink   (row sum)

Usage:
    python step3_channel_specific_connectivity_analysis.py \
        --connectivity_dir connectivity \
        --epochs_dir preprocessed_epochs \
        --output_dir figures/per_subject_analysis \
        --band integrated \
        --metric dtf \
        --direction outflow
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import json
import pandas as pd
from collections import Counter

# Channel names
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2']

# Channel groups
REGIONS = {
    'Frontal':   ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Temporal':  ['T3', 'T4', 'T5', 'T6'],
    'Central':   ['C3', 'Cz', 'C4'],
    'Parietal':  ['P3', 'Pz', 'P4'],
    'Occipital': ['O1', 'O2'],
}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING MASK LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_training_mask(epochs_dir, subject_name, n_epochs_in_connectivity):
    """
    Load training_mask for a subject and align it to the connectivity file.

    The connectivity .npz may have fewer epochs than the raw epochs file
    (bad epochs discarded during step1). We use the 'indices' array stored
    in the .npz to map connectivity epochs back to the original epoch indices,
    then apply the training_mask correctly.

    Parameters
    ----------
    epochs_dir : Path
    subject_name : str
    n_epochs_in_connectivity : int

    Returns
    -------
    mask : np.ndarray (n_epochs_in_connectivity,) bool
        True  = keep (pre-ictal or ictal)
        False = post-ictal, exclude
    found : bool
    """
    mask_path = epochs_dir / f'{subject_name}_training_mask.npy'

    if not mask_path.exists():
        print(f'  ⚠  training_mask not found for {subject_name} — using all epochs')
        return np.ones(n_epochs_in_connectivity, dtype=bool), False

    full_mask = np.load(mask_path)
    return full_mask.astype(bool), True


# ══════════════════════════════════════════════════════════════════════════════
# PER-CHANNEL STRENGTH  (fixed axis convention)
# ══════════════════════════════════════════════════════════════════════════════

def compute_channel_strength(connectivity_matrix, direction='total'):
    """
    Compute connectivity strength per channel for a single epoch matrix.

    Convention: connectivity_matrix[i, j] = FROM j TO i
      → outflow of channel j = sum over sinks i   = column sum = axis=0
      → inflow  of channel i = sum over sources j = row sum    = axis=1

    Parameters
    ----------
    connectivity_matrix : np.ndarray (n_channels, n_channels)
        Diagonal should already be zero.
    direction : str
        'outflow' — how much each channel SENDS  (column sum, axis=0)
        'inflow'  — how much each channel RECEIVES (row sum,  axis=1)
        'total'   — outflow + inflow

    Returns
    -------
    strength : np.ndarray (n_channels,)
    """
    # axis=0: sum over sink rows for each source column → outflow per source
    outflow = np.sum(connectivity_matrix, axis=0)   # (n_channels,)

    # axis=1: sum over source columns for each sink row → inflow per sink
    inflow  = np.sum(connectivity_matrix, axis=1)   # (n_channels,)

    if direction == 'outflow':
        return outflow
    elif direction == 'inflow':
        return inflow
    else:   # total
        return outflow + inflow


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE SUBJECT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_single_subject(file_path, band, metric, epochs_dir, direction='total'):
    """
    Analyze one subject, excluding post-ictal epochs via training_mask.

    Returns
    -------
    results : dict or None
    """
    data         = np.load(file_path)
    subject_name = Path(file_path).stem.replace('_graphs', '')

    conn_key = f'{metric}_{band}'
    if conn_key not in data:
        print(f'  ⚠  Key {conn_key} not found in {file_path.name}')
        return None

    connectivity = data[conn_key]          # (n_epochs, 19, 19)
    labels       = data['labels']          # (n_epochs,)
    n_epochs     = len(labels)

    # ------------------------------------------------------------------
    # Load and align training mask
    # ------------------------------------------------------------------
    raw_mask, found = load_training_mask(epochs_dir, subject_name, n_epochs)

    if found and len(raw_mask) != n_epochs:
        if 'indices' in data:
            indices = data['indices'].astype(int)
            if indices.max() < len(raw_mask):
                training_mask = raw_mask[indices]
            else:
                print(f'  ⚠  indices out of range for {subject_name} — using all')
                training_mask = np.ones(n_epochs, dtype=bool)
        else:
            print(f'  ⚠  no indices array in {file_path.name} — using all epochs')
            training_mask = np.ones(n_epochs, dtype=bool)
    else:
        training_mask = raw_mask

    # ------------------------------------------------------------------
    # Apply mask
    # ------------------------------------------------------------------
    n_total    = n_epochs
    n_excluded = int((~training_mask).sum())

    connectivity = connectivity[training_mask]
    labels       = labels[training_mask]
    n_epochs     = len(labels)

    print(f'  {subject_name}: {n_total} total, '
          f'{n_excluded} post-ictal excluded, '
          f'{n_epochs} kept  '
          f'(pre={int((labels==0).sum())}, ictal={int((labels==1).sum())})')

    if n_epochs == 0:
        print(f'  ⚠  No epochs left after masking for {subject_name}')
        return None

    # ------------------------------------------------------------------
    # Compute per-channel strength per epoch
    # ------------------------------------------------------------------
    channel_strength = np.zeros((n_epochs, 19))
    for i in range(n_epochs):
        channel_strength[i] = compute_channel_strength(
            connectivity[i], direction=direction)

    pre_ictal_strength = channel_strength[labels == 0]
    ictal_strength     = channel_strength[labels == 1]

    if len(ictal_strength) == 0:
        print(f'  ⚠  No ictal epochs for {subject_name} after masking')
        return None

    # ------------------------------------------------------------------
    # Per-channel statistics (Mann-Whitney U, pre vs ictal)
    # ------------------------------------------------------------------
    results = {
        'subject':     subject_name,
        'n_pre_ictal': int(len(pre_ictal_strength)),
        'n_ictal':     int(len(ictal_strength)),
        'n_excluded':  n_excluded,
        'direction':   direction,
        'channels':    {},
    }

    for ch_idx, ch_name in enumerate(CHANNELS):
        pre = pre_ictal_strength[:, ch_idx]
        ict = ictal_strength[:, ch_idx]

        pre_mean = float(pre.mean())
        ict_mean = float(ict.mean())
        diff     = ict_mean - pre_mean

        _, p_val = mannwhitneyu(pre, ict, alternative='two-sided')

        results['channels'][ch_name] = {
            'pre_mean':    pre_mean,
            'ictal_mean':  ict_mean,
            'difference':  diff,
            'p_value':     float(p_val),
            'significant': bool(p_val < 0.05),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL CHANNEL IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def identify_focal_channels(subject_results):
    """
    Identify channels with significant connectivity change (pre vs ictal).
    """
    channels = subject_results['channels']

    increases = []
    decreases = []

    for ch_name, stats in channels.items():
        if stats['significant']:
            if stats['difference'] > 0:
                increases.append((ch_name, stats['difference'], stats['p_value']))
            else:
                decreases.append((ch_name, stats['difference'], stats['p_value']))

    increases.sort(key=lambda x: x[1], reverse=True)
    decreases.sort(key=lambda x: x[1])

    if len(increases) == 0:
        seizure_type   = 'No focal pattern (all decrease or no change)'
        focal_channels = []
    elif len(increases) <= 4:
        seizure_type   = 'Focal seizure'
        focal_channels = [ch for ch, _, _ in increases]
    else:
        seizure_type   = 'Widespread/Generalized'
        focal_channels = [ch for ch, _, _ in increases[:4]]

    regions_involved = [
        region for region, region_channels in REGIONS.items()
        if any(ch in focal_channels for ch in region_channels)
    ]

    return {
        'seizure_type':     seizure_type,
        'focal_channels':   focal_channels,
        'regions_involved': regions_involved,
        'n_increased':      len(increases),
        'n_decreased':      len(decreases),
        'top_increases':    increases[:5],
        'top_decreases':    decreases[:5],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_subject_comparison(subject_results, output_path):
    """Bar chart: pre-ictal vs ictal per channel, with significance markers."""
    channels  = subject_results['channels']
    direction = subject_results.get('direction', 'total')

    ch_names    = list(CHANNELS)
    pre_means   = [channels[ch]['pre_mean']   for ch in ch_names]
    ictal_means = [channels[ch]['ictal_mean'] for ch in ch_names]
    p_values    = [channels[ch]['p_value']    for ch in ch_names]

    # Direction-specific colours
    color_map = {
        'outflow': ('steelblue',      'firebrick'),
        'inflow':  ('mediumseagreen', 'darkorange'),
        'total':   ('steelblue',      'crimson'),
    }
    pre_color, ict_color = color_map.get(direction, ('steelblue', 'crimson'))

    fig, ax = plt.subplots(figsize=(16, 6))
    x     = np.arange(len(ch_names))
    width = 0.35

    ax.bar(x - width/2, pre_means,   width,
           label='Pre-ictal', alpha=0.8, color=pre_color)
    ax.bar(x + width/2, ictal_means, width,
           label='Ictal',     alpha=0.8, color=ict_color)

    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            y_max  = max(pre_means[i], ictal_means[i])
            marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            ax.text(i, y_max * 1.05, marker, ha='center',
                    fontsize=12, fontweight='bold', color='red')

    direction_labels = {
        'outflow': 'Mean Outflow Strength  (what channel SENDS)',
        'inflow':  'Mean Inflow Strength   (what channel RECEIVES)',
        'total':   'Mean Total Strength    (outflow + inflow)',
    }

    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel(direction_labels.get(direction, 'Mean Connectivity Strength'),
                  fontsize=11, fontweight='bold')
    ax.set_title(
        f"{subject_results['subject']} — Channel-Specific Connectivity  "
        f"[{direction.upper()}]\n"
        f"Pre-ictal: {subject_results['n_pre_ictal']} epochs  |  "
        f"Ictal: {subject_results['n_ictal']} epochs  |  "
        f"Post-ictal excluded: {subject_results['n_excluded']} epochs",
        fontsize=13, fontweight='bold',
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def create_summary_table(all_results, output_dir, direction='total'):
    """Summary CSV and 4-panel overview figure across all subjects."""
    summary_data = []

    for result in all_results:
        if result is None:
            continue
        focal_info = identify_focal_channels(result)
        summary_data.append({
            'Subject':        result['subject'],
            'Direction':      direction,
            'Seizure Type':   focal_info['seizure_type'],
            'Focal Channels': ', '.join(focal_info['focal_channels']),
            'Brain Regions':  ', '.join(focal_info['regions_involved']),
            'N Increased':    focal_info['n_increased'],
            'N Decreased':    focal_info['n_decreased'],
            'N Pre-ictal':    result['n_pre_ictal'],
            'N Ictal':        result['n_ictal'],
            'N Excluded':     result['n_excluded'],
        })

    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / f'subject_summary_{direction}.csv', index=False)

    # 4-panel summary figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1 — seizure type distribution
    ax = axes[0, 0]
    type_counts = df['Seizure Type'].value_counts()
    ax.bar(range(len(type_counts)), type_counts.values,
           alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(type_counts)))
    ax.set_xticklabels(type_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Seizure Type Distribution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel 2 — brain region involvement
    ax = axes[0, 1]
    all_regions = []
    for regions in df['Brain Regions']:
        if isinstance(regions, str) and regions:
            all_regions.extend(regions.split(', '))
    region_counts = Counter(all_regions)
    ax.bar(range(len(region_counts)), region_counts.values(),
           alpha=0.7, color='crimson', edgecolor='black')
    ax.set_xticks(range(len(region_counts)))
    ax.set_xticklabels(region_counts.keys(), rotation=45, ha='right')
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Brain Region Involvement', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel 3 — increased channel count distribution
    ax = axes[1, 0]
    ax.hist(df['N Increased'], bins=range(0, 20),
            alpha=0.7, color='green', edgecolor='black')
    median_val = df['N Increased'].median()
    ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.0f}')
    ax.set_xlabel(f'Channels with Increased {direction.capitalize()}', fontsize=11)
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Focal Channel Count Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Panel 4 — channel involvement heatmap
    ax = axes[1, 1]
    channel_involvement = {ch: 0 for ch in CHANNELS}
    for focal_str in df['Focal Channels']:
        if isinstance(focal_str, str) and focal_str:
            for ch in focal_str.split(', '):
                if ch in channel_involvement:
                    channel_involvement[ch] += 1
    ch_counts = [channel_involvement[ch] for ch in CHANNELS]
    max_count = max(ch_counts) if max(ch_counts) > 0 else 1
    colors    = plt.cm.hot(np.array(ch_counts) / max_count)
    ax.bar(range(len(CHANNELS)), ch_counts,
           alpha=0.8, color=colors, edgecolor='black')
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right')
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Channel Involvement Across All Subjects', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    direction_labels = {
        'outflow': 'OUTFLOW — Channels that become DRIVERS during seizure',
        'inflow':  'INFLOW  — Channels that become FOLLOWERS during seizure',
        'total':   'TOTAL   — Overall channel involvement during seizure',
    }
    plt.suptitle(
        f'Per-Subject Focal Seizure Analysis Summary\n'
        f'{direction_labels.get(direction, direction.upper())}\n'
        f'(post-ictal epochs excluded via training_mask)',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_dir / f'summary_analysis_{direction}.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Per-subject channel-specific analysis with direction control'
    )
    parser.add_argument('--connectivity_dir', required=True,
                        help='Directory with subject_XX_graphs.npz files')
    parser.add_argument('--epochs_dir', required=True,
                        help='Directory with subject_XX_training_mask.npy files')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--band',      default='integrated')
    parser.add_argument('--metric',    default='pdc',
                        choices=['pdc', 'dtf'])
    parser.add_argument('--direction', default='total',
                        choices=['outflow', 'inflow', 'total'],
                        help=(
                            'outflow = what channels SEND (identifies drivers/focus)  |  '
                            'inflow  = what channels RECEIVE (identifies followers)   |  '
                            'total   = outflow + inflow (overall involvement)'
                        ))
    parser.add_argument('--top_subjects', type=int, default=None,
                        help='Only plot this many subjects (default: all)')
    args = parser.parse_args()

    conn_dir   = Path(args.connectivity_dir)
    epochs_dir = Path(args.epochs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('PER-SUBJECT CHANNEL-SPECIFIC ANALYSIS')
    print('=' * 80)
    print(f'Connectivity dir: {conn_dir}')
    print(f'Epochs dir:       {epochs_dir}')
    print(f'Output dir:       {output_dir}')
    print(f'Band:             {args.band}')
    print(f'Metric:           {args.metric.upper()}')
    print(f'Direction:        {args.direction.upper()}')
    print()

    direction_explanation = {
        'outflow': 'Measuring OUTFLOW — how much each channel SENDS to others.\n'
                   '  Rising outflow during seizure = channel becomes a DRIVER.\n'
                   '  Most relevant for identifying the epileptic focus.',
        'inflow':  'Measuring INFLOW — how much each channel RECEIVES from others.\n'
                   '  Rising inflow during seizure = channel becomes a FOLLOWER.\n'
                   '  Useful for identifying recruited/propagation regions.',
        'total':   'Measuring TOTAL (outflow + inflow) — overall involvement.\n'
                   '  Combines driver and follower effects.\n'
                   '  Use outflow and inflow separately for directional insight.',
    }
    print(direction_explanation[args.direction])
    print()
    print('Matrix convention: connectivity[i,j] = FROM j TO i')
    print('  outflow of j = sum over rows (axis=0) = column sum')
    print('  inflow  of i = sum over cols (axis=1) = row sum')
    print('=' * 80)

    files = sorted(conn_dir.glob('subject_*_graphs.npz'))
    print(f'\nFound {len(files)} subjects\n')

    all_results = []

    for file in tqdm(files, desc='Subjects'):
        result = analyze_single_subject(
            file, args.band, args.metric, epochs_dir,
            direction=args.direction)
        if result:
            result['focal_info'] = identify_focal_channels(result)
            all_results.append(result)

    print(f'\nSuccessfully analyzed: {len(all_results)} subjects')

    # Individual plots
    plot_subjects = all_results[:args.top_subjects] if args.top_subjects else all_results
    ind_dir       = output_dir / f'individual_subjects_{args.direction}'
    ind_dir.mkdir(exist_ok=True)

    print(f'\nCreating individual plots ({args.direction}) for {len(plot_subjects)} subjects...')
    for result in tqdm(plot_subjects):
        plot_path = ind_dir / f"{result['subject']}_{args.direction}_comparison.png"
        plot_subject_comparison(result, plot_path)

    # Summary
    print('\nCreating summary...')
    df_summary = create_summary_table(all_results, output_dir, direction=args.direction)

    # Save JSON
    with open(output_dir / f'detailed_results_{args.direction}.json', 'w') as f:
        json.dump([{
            'subject':     r['subject'],
            'direction':   r['direction'],
            'n_pre_ictal': r['n_pre_ictal'],
            'n_ictal':     r['n_ictal'],
            'n_excluded':  r['n_excluded'],
            'focal_info':  r['focal_info'],
            'channels':    r['channels'],
        } for r in all_results], f, indent=2)

    # Print summary
    print('\n' + '=' * 80)
    print(f'SUMMARY — {args.direction.upper()}')
    print('=' * 80)
    print('\nSeizure Type Distribution:')
    print(df_summary['Seizure Type'].value_counts())
    print(f'\nMost Common Channels with Increased {args.direction.capitalize()}:')
    all_focal = []
    for focal_str in df_summary['Focal Channels']:
        if isinstance(focal_str, str) and focal_str:
            all_focal.extend(focal_str.split(', '))
    focal_counts = Counter(all_focal)
    for ch, count in focal_counts.most_common(10):
        print(f'  {ch}: {count} subjects ({100*count/len(all_results):.1f}%)')

    print('\n' + '=' * 80)
    print('ANALYSIS COMPLETE')
    print('=' * 80)
    print(f'\nOutputs in {output_dir}:')
    print(f'  subject_summary_{args.direction}.csv')
    print(f'  summary_analysis_{args.direction}.png')
    print(f'  detailed_results_{args.direction}.json')
    print(f'  individual_subjects_{args.direction}/*.png')
    print('=' * 80)


if __name__ == '__main__':
    main()