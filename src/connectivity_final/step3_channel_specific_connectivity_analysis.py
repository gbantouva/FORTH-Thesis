"""
Per-Subject Channel-Specific Analysis
======================================
Analyzes EACH subject individually to identify their specific focal channels.

KEY CHANGE vs original:
    Loads training_mask from preprocessed_epochs directory and filters out
    post-ictal epochs BEFORE any analysis. Only epochs from recording start
    up to seizure end are used (pre-ictal + ictal, no post-ictal).
    This is consistent with the GNN training set.

Usage:
    python step3_channel_specific_connectivity_analysis.py \
        --connectivity_dir connectivity \
        --epochs_dir preprocessed_epochs \
        --output_dir figures/per_subject_analysis \
        --band integrated \
        --metric pdc
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
        Directory containing subject_XX_training_mask.npy
    subject_name : str
        e.g. 'subject_02'
    n_epochs_in_connectivity : int
        Number of epochs in the connectivity file (after bad-epoch removal)

    Returns
    -------
    mask : np.ndarray (n_epochs_in_connectivity,) bool
        True  = epoch is in training set (pre-ictal or ictal, no post-ictal)
        False = post-ictal epoch, exclude from analysis
    found : bool
        False if the training_mask file was not found (use all epochs instead)
    """
    mask_path = epochs_dir / f'{subject_name}_training_mask.npy'

    if not mask_path.exists():
        print(f'  ⚠  training_mask not found for {subject_name} — using all epochs')
        return np.ones(n_epochs_in_connectivity, dtype=bool), False

    full_mask = np.load(mask_path)   # shape = (n_raw_epochs,) bool

    # If connectivity has the same number of epochs as the mask, use directly
    if len(full_mask) == n_epochs_in_connectivity:
        return full_mask.astype(bool), True

    # Otherwise training_mask refers to raw epochs; we cannot align without
    # the indices array — caller must pass it separately (see analyze_single_subject)
    # Return None to signal that alignment is needed
    return full_mask.astype(bool), True


# ══════════════════════════════════════════════════════════════════════════════
# PER-CHANNEL STRENGTH
# ══════════════════════════════════════════════════════════════════════════════

def compute_channel_strength(connectivity_matrix):
    """
    Total connectivity strength per channel = outflow + inflow.

    Parameters
    ----------
    connectivity_matrix : np.ndarray (n_channels, n_channels)
        Diagonal should already be zero.

    Returns
    -------
    total_strength : np.ndarray (n_channels,)
    """
    out_strength = np.sum(connectivity_matrix, axis=1)  # row sum = inflow received
    in_strength  = np.sum(connectivity_matrix, axis=0)  # col sum = outflow sent
    return out_strength + in_strength


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE SUBJECT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_single_subject(file_path, band, metric, epochs_dir):
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
        # Mask refers to raw (unfiltered) epochs; align via 'indices' array
        if 'indices' in data:
            indices  = data['indices'].astype(int)   # which raw epochs survived
            if indices.max() < len(raw_mask):
                training_mask = raw_mask[indices]
            else:
                print(f'  ⚠  indices out of range for {subject_name} mask — using all')
                training_mask = np.ones(n_epochs, dtype=bool)
        else:
            print(f'  ⚠  no indices array in {file_path.name} — using all epochs')
            training_mask = np.ones(n_epochs, dtype=bool)
    else:
        training_mask = raw_mask

    # ------------------------------------------------------------------
    # Apply mask — drop post-ictal epochs
    # ------------------------------------------------------------------
    n_total    = n_epochs
    n_excluded = int((~training_mask).sum())

    connectivity = connectivity[training_mask]
    labels       = labels[training_mask]
    n_epochs     = len(labels)

    print(f'  {subject_name}: {n_total} epochs total, '
          f'{n_excluded} post-ictal excluded, '
          f'{n_epochs} kept  '
          f'(pre-ictal={int((labels==0).sum())}, ictal={int((labels==1).sum())})')

    if n_epochs == 0:
        print(f'  ⚠  No epochs left after masking for {subject_name}')
        return None

    # ------------------------------------------------------------------
    # Compute per-channel strength per epoch
    # ------------------------------------------------------------------
    channel_strength = np.zeros((n_epochs, 19))
    for i in range(n_epochs):
        channel_strength[i] = compute_channel_strength(connectivity[i])

    pre_ictal_strength = channel_strength[labels == 0]
    ictal_strength     = channel_strength[labels == 1]

    if len(ictal_strength) == 0:
        print(f'  ⚠  No ictal epochs for {subject_name} after masking')
        return None

    # ------------------------------------------------------------------
    # Per-channel statistics (Mann-Whitney U, pre vs ictal)
    # ------------------------------------------------------------------
    results = {
        'subject':      subject_name,
        'n_pre_ictal':  int(len(pre_ictal_strength)),
        'n_ictal':      int(len(ictal_strength)),
        'n_excluded':   n_excluded,
        'channels':     {},
    }

    for ch_idx, ch_name in enumerate(CHANNELS):
        pre = pre_ictal_strength[:, ch_idx]
        ict = ictal_strength[:, ch_idx]

        pre_mean = float(pre.mean())
        ict_mean = float(ict.mean())
        diff     = ict_mean - pre_mean

        _, p_val = mannwhitneyu(pre, ict, alternative='two-sided')

        results['channels'][ch_name] = {
            'pre_mean':   pre_mean,
            'ictal_mean': ict_mean,
            'difference': diff,
            'p_value':    float(p_val),
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
        seizure_type   = 'No focal pattern (all decrease)'
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
    channels = subject_results['channels']

    ch_names    = list(CHANNELS)
    pre_means   = [channels[ch]['pre_mean']   for ch in ch_names]
    ictal_means = [channels[ch]['ictal_mean'] for ch in ch_names]
    p_values    = [channels[ch]['p_value']    for ch in ch_names]

    fig, ax = plt.subplots(figsize=(16, 6))
    x     = np.arange(len(ch_names))
    width = 0.35

    ax.bar(x - width/2, pre_means,   width, label='Pre-ictal', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, ictal_means, width, label='Ictal',     alpha=0.8, color='crimson')

    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            y_max  = max(pre_means[i], ictal_means[i])
            marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            ax.text(i, y_max * 1.05, marker, ha='center',
                    fontsize=12, fontweight='bold', color='red')

    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Connectivity Strength', fontsize=12, fontweight='bold')
    ax.set_title(
        f"{subject_results['subject']} — Channel-Specific Connectivity\n"
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


def create_summary_table(all_results, output_dir):
    """Summary CSV and 4-panel overview figure across all subjects."""
    summary_data = []

    for result in all_results:
        if result is None:
            continue
        focal_info = identify_focal_channels(result)
        summary_data.append({
            'Subject':        result['subject'],
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
    df.to_csv(output_dir / 'subject_summary.csv', index=False)

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

    # Panel 3 — focal channel count distribution
    ax = axes[1, 0]
    ax.hist(df['N Increased'], bins=range(0, 20),
            alpha=0.7, color='green', edgecolor='black')
    median_val = df['N Increased'].median()
    ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.0f}')
    ax.set_xlabel('Channels with Increased Connectivity', fontsize=11)
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
    ch_counts  = [channel_involvement[ch] for ch in CHANNELS]
    max_count  = max(ch_counts) if max(ch_counts) > 0 else 1
    colors     = plt.cm.hot(np.array(ch_counts) / max_count)
    ax.bar(range(len(CHANNELS)), ch_counts,
           alpha=0.8, color=colors, edgecolor='black')
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right')
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Channel Involvement Across All Subjects', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Per-Subject Focal Seizure Analysis Summary\n'
        '(post-ictal epochs excluded via training_mask)',
        fontsize=16, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Per-subject channel-specific analysis (post-ictal excluded)'
    )
    parser.add_argument('--connectivity_dir', required=True,
                        help='Directory with subject_XX_graphs.npz files')
    parser.add_argument('--epochs_dir', required=True,
                        help='Directory with subject_XX_training_mask.npy files')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--band',   default='integrated')
    parser.add_argument('--metric', default='pdc')
    parser.add_argument('--top_subjects', type=int, default=None,
                        help='Only plot this many subjects (default: all)')
    args = parser.parse_args()

    conn_dir   = Path(args.connectivity_dir)
    epochs_dir = Path(args.epochs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('PER-SUBJECT CHANNEL-SPECIFIC ANALYSIS')
    print('(post-ictal epochs excluded via training_mask)')
    print('=' * 80)
    print(f'Connectivity dir: {conn_dir}')
    print(f'Epochs dir:       {epochs_dir}')
    print(f'Output dir:       {output_dir}')
    print(f'Band:   {args.band}')
    print(f'Metric: {args.metric}')
    print('=' * 80)

    files = sorted(conn_dir.glob('subject_*_graphs.npz'))
    print(f'\nFound {len(files)} subjects\n')

    all_results = []

    for file in tqdm(files, desc='Subjects'):
        result = analyze_single_subject(file, args.band, args.metric, epochs_dir)
        if result:
            result['focal_info'] = identify_focal_channels(result)
            all_results.append(result)

    print(f'\nSuccessfully analyzed: {len(all_results)} subjects')

    # Individual plots
    plot_subjects  = all_results[:args.top_subjects] if args.top_subjects else all_results
    ind_dir        = output_dir / 'individual_subjects'
    ind_dir.mkdir(exist_ok=True)

    print(f'\nCreating individual plots for {len(plot_subjects)} subjects...')
    for result in tqdm(plot_subjects):
        plot_path = ind_dir / f"{result['subject']}_channel_comparison.png"
        plot_subject_comparison(result, plot_path)

    # Summary
    print('\nCreating summary...')
    df_summary = create_summary_table(all_results, output_dir)

    # Save JSON
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump([{
            'subject':     r['subject'],
            'n_pre_ictal': r['n_pre_ictal'],
            'n_ictal':     r['n_ictal'],
            'n_excluded':  r['n_excluded'],
            'focal_info':  r['focal_info'],
            'channels':    r['channels'],
        } for r in all_results], f, indent=2)

    # Print summary
    print('\n' + '=' * 80)
    print('SUMMARY ACROSS ALL SUBJECTS')
    print('=' * 80)
    print('\nSeizure Type Distribution:')
    print(df_summary['Seizure Type'].value_counts())
    print('\nMost Common Focal Channels:')
    all_focal = []
    for focal_str in df_summary['Focal Channels']:
        if isinstance(focal_str, str) and focal_str:
            all_focal.extend(focal_str.split(', '))
    focal_counts = Counter(all_focal)
    for ch, count in focal_counts.most_common(10):
        print(f'  {ch}: {count} subjects ({100*count/len(all_results):.1f}%)')

    print('\n' + '=' * 80)
    print('✅ ANALYSIS COMPLETE')
    print('=' * 80)
    print(f'\nOutputs:')
    print(f'  subject_summary.csv       — per-subject table')
    print(f'  summary_analysis.png      — 4-panel overview')
    print(f'  detailed_results.json     — full statistics')
    print(f'  individual_subjects/*.png — {len(plot_subjects)} bar charts')
    print('=' * 80)


if __name__ == '__main__':
    main()