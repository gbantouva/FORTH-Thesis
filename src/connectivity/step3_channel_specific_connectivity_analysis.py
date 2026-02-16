"""
Per-Subject Channel-Specific Analysis
======================================
Analyzes EACH subject individually to identify their specific focal channels.

This is CRITICAL because different subjects have different seizure foci:
- Subject 1: Left temporal
- Subject 2: Right temporal  
- Subject 3: Frontal
etc.

Averaging across subjects cancels out individual focal patterns!

Usage:
    python per_subject_channel_analysis.py \
        --connectivity_dir connectivity \
        --output_dir figures/per_subject_analysis \
        --band integrated \
        --metric pdc \
        --top_subjects 10
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import json
import pandas as pd

# Channel names
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2']

# Channel groups
REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Temporal': ['T3', 'T4', 'T5', 'T6'],
    'Central': ['C3', 'Cz', 'C4'],
    'Parietal': ['P3', 'Pz', 'P4'],
    'Occipital': ['O1', 'O2']
}


def compute_channel_strength(connectivity_matrix):
    """
    Compute total connectivity strength per channel.
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray (n_channels, n_channels)
    
    Returns:
    --------
    total_strength : np.ndarray (n_channels,)
    """
    n_ch = connectivity_matrix.shape[0]
    
    # Out-strength (row sum) + In-strength (column sum)
    out_strength = np.sum(connectivity_matrix, axis=1)  # Already diagonal=0
    in_strength = np.sum(connectivity_matrix, axis=0)
    total_strength = out_strength + in_strength
    
    return total_strength


def analyze_single_subject(file_path, band, metric):
    """
    Analyze one subject's focal pattern.
    
    Returns:
    --------
    results : dict with per-channel statistics
    """
    data = np.load(file_path)
    subject_name = Path(file_path).stem.replace('_graphs', '')
    
    # Load connectivity matrices
    conn_key = f'{metric}_{band}'
    if conn_key not in data:
        return None
    
    connectivity = data[conn_key]  # (n_epochs, 19, 19)
    labels = data['labels']  # (n_epochs,)
    
    # Compute per-channel strength for each epoch
    n_epochs = len(labels)
    channel_strength = np.zeros((n_epochs, 19))
    
    for i in range(n_epochs):
        channel_strength[i] = compute_channel_strength(connectivity[i])
    
    # Separate pre-ictal vs ictal
    pre_ictal_strength = channel_strength[labels == 0]  # (n_pre, 19)
    ictal_strength = channel_strength[labels == 1]      # (n_ictal, 19)
    
    if len(ictal_strength) == 0:
        return None
    
    # Per-channel statistics
    results = {
        'subject': subject_name,
        'n_pre_ictal': len(pre_ictal_strength),
        'n_ictal': len(ictal_strength),
        'channels': {}
    }
    
    for ch_idx, ch_name in enumerate(CHANNELS):
        pre = pre_ictal_strength[:, ch_idx]
        ict = ictal_strength[:, ch_idx]
        
        # Statistics
        pre_mean = pre.mean()
        ict_mean = ict.mean()
        diff = ict_mean - pre_mean
        
        # Mann-Whitney U test
        _, p_val = mannwhitneyu(pre, ict, alternative='two-sided')
        
        results['channels'][ch_name] = {
            'pre_mean': float(pre_mean),
            'ictal_mean': float(ict_mean),
            'difference': float(diff),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05)  # Convert to Python bool
        }
    
    return results


def identify_focal_channels(subject_results, threshold=0.05):
    """
    Identify focal channels for a subject.
    
    Returns:
    --------
    focal_info : dict with focal channel identification
    """
    channels = subject_results['channels']
    
    # Find channels with significant INCREASE
    increases = []
    decreases = []
    
    for ch_name, stats in channels.items():
        if stats['significant']:
            if stats['difference'] > 0:
                increases.append((ch_name, stats['difference'], stats['p_value']))
            else:
                decreases.append((ch_name, stats['difference'], stats['p_value']))
    
    # Sort by difference magnitude
    increases.sort(key=lambda x: x[1], reverse=True)
    decreases.sort(key=lambda x: x[1])
    
    # Classify seizure focus
    if len(increases) == 0:
        seizure_type = "No focal pattern (all decrease)"
        focal_channels = []
    elif len(increases) <= 4:
        seizure_type = "Focal seizure"
        focal_channels = [ch for ch, _, _ in increases]
    else:
        seizure_type = "Widespread/Generalized"
        focal_channels = [ch for ch, _, _ in increases[:4]]
    
    # Determine brain region
    regions_involved = []
    for region, region_channels in REGIONS.items():
        if any(ch in focal_channels for ch in region_channels):
            regions_involved.append(region)
    
    return {
        'seizure_type': seizure_type,
        'focal_channels': focal_channels,
        'regions_involved': regions_involved,
        'n_increased': len(increases),
        'n_decreased': len(decreases),
        'top_increases': increases[:5],
        'top_decreases': decreases[:5]
    }


def plot_subject_comparison(subject_results, output_path):
    """
    Create bar chart for one subject.
    """
    channels = subject_results['channels']
    
    ch_names = []
    pre_means = []
    ictal_means = []
    p_values = []
    
    for ch_name in CHANNELS:
        stats = channels[ch_name]
        ch_names.append(ch_name)
        pre_means.append(stats['pre_mean'])
        ictal_means.append(stats['ictal_mean'])
        p_values.append(stats['p_value'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x = np.arange(len(ch_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pre_means, width, label='Pre-ictal', alpha=0.8, color='blue')
    bars2 = ax.bar(x + width/2, ictal_means, width, label='Ictal', alpha=0.8, color='red')
    
    # Mark significant channels
    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            y_max = max(pre_means[i], ictal_means[i])
            marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            ax.text(i, y_max * 1.05, marker, ha='center', fontsize=12, 
                   fontweight='bold', color='red')
    
    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Connectivity Strength', fontsize=12, fontweight='bold')
    ax.set_title(f"{subject_results['subject']} - Channel-Specific Connectivity\n"
                 f"Pre-ictal: {subject_results['n_pre_ictal']} epochs | "
                 f"Ictal: {subject_results['n_ictal']} epochs",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def create_summary_table(all_results, output_dir):
    """
    Create summary table across all subjects.
    """
    summary_data = []
    
    for result in all_results:
        if result is None:
            continue
        
        focal_info = identify_focal_channels(result)
        
        summary_data.append({
            'Subject': result['subject'],
            'Seizure Type': focal_info['seizure_type'],
            'Focal Channels': ', '.join(focal_info['focal_channels']),
            'Brain Regions': ', '.join(focal_info['regions_involved']),
            'N Increased': focal_info['n_increased'],
            'N Decreased': focal_info['n_decreased'],
            'N Pre-ictal': result['n_pre_ictal'],
            'N Ictal': result['n_ictal']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save CSV
    df.to_csv(output_dir / 'subject_summary.csv', index=False)
    
    # Create visual summary
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Seizure type distribution
    ax = axes[0, 0]
    type_counts = df['Seizure Type'].value_counts()
    ax.bar(range(len(type_counts)), type_counts.values, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(type_counts)))
    ax.set_xticklabels(type_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Seizure Type Distribution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 2: Brain region involvement
    ax = axes[0, 1]
    all_regions = []
    for regions in df['Brain Regions']:
        if isinstance(regions, str):
            all_regions.extend(regions.split(', '))
    
    from collections import Counter
    region_counts = Counter(all_regions)
    
    ax.bar(range(len(region_counts)), region_counts.values(), alpha=0.7, color='crimson', edgecolor='black')
    ax.set_xticks(range(len(region_counts)))
    ax.set_xticklabels(region_counts.keys(), rotation=45, ha='right')
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Brain Region Involvement', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 3: Number of focal channels
    ax = axes[1, 0]
    ax.hist(df['N Increased'], bins=range(0, 20), alpha=0.7, color='green', edgecolor='black')
    ax.axvline(df['N Increased'].median(), color='red', linestyle='--', linewidth=2,
              label=f'Median: {df["N Increased"].median():.0f}')
    ax.set_xlabel('Number of Channels with Increased Connectivity', fontsize=11)
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Focal Channel Count Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 4: Channel frequency heatmap
    ax = axes[1, 1]
    channel_involvement = {ch: 0 for ch in CHANNELS}
    
    for focal_str in df['Focal Channels']:
        if isinstance(focal_str, str) and focal_str:
            for ch in focal_str.split(', '):
                if ch in channel_involvement:
                    channel_involvement[ch] += 1
    
    ch_counts = [channel_involvement[ch] for ch in CHANNELS]
    colors = plt.cm.hot(np.array(ch_counts) / max(ch_counts) if max(ch_counts) > 0 else 0)
    
    ax.bar(range(len(CHANNELS)), ch_counts, alpha=0.8, color=colors, edgecolor='black')
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right')
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Channel Involvement Across All Subjects', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Per-Subject Focal Seizure Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Per-subject channel-specific analysis")
    parser.add_argument("--connectivity_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--band", default="integrated")
    parser.add_argument("--metric", default="pdc")
    parser.add_argument("--top_subjects", type=int, default=None,
                       help="Only plot this many subjects (default: all)")
    
    args = parser.parse_args()
    
    conn_dir = Path(args.connectivity_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PER-SUBJECT CHANNEL-SPECIFIC ANALYSIS")
    print("=" * 80)
    print(f"Input: {conn_dir}")
    print(f"Output: {output_dir}")
    print(f"Band: {args.band}")
    print(f"Metric: {args.metric}")
    print("=" * 80)
    
    # Find all subject files
    files = sorted(conn_dir.glob("subject_*_graphs.npz"))
    print(f"\nFound {len(files)} subjects")
    
    # Analyze each subject
    print("\nAnalyzing subjects...")
    all_results = []
    
    for file in tqdm(files):
        result = analyze_single_subject(file, args.band, args.metric)
        if result:
            all_results.append(result)
            
            # Identify focal pattern
            focal_info = identify_focal_channels(result)
            result['focal_info'] = focal_info
    
    print(f"\nSuccessfully analyzed: {len(all_results)} subjects")
    
    # Create individual plots for top subjects
    plot_subjects = all_results[:args.top_subjects] if args.top_subjects else all_results
    
    print(f"\nCreating individual plots for {len(plot_subjects)} subjects...")
    
    subject_plots_dir = output_dir / 'individual_subjects'
    subject_plots_dir.mkdir(exist_ok=True)
    
    for result in tqdm(plot_subjects):
        plot_path = subject_plots_dir / f"{result['subject']}_channel_comparison.png"
        plot_subject_comparison(result, plot_path)
    
    # Create summary
    print("\nCreating summary analysis...")
    df_summary = create_summary_table(all_results, output_dir)
    
    # Save detailed JSON
    detailed_results = []
    for result in all_results:
        detailed_results.append({
            'subject': result['subject'],
            'n_pre_ictal': result['n_pre_ictal'],
            'n_ictal': result['n_ictal'],
            'focal_info': result['focal_info'],
            'channels': result['channels']
        })
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL SUBJECTS")
    print("=" * 80)
    
    print("\nSeizure Type Distribution:")
    print(df_summary['Seizure Type'].value_counts())
    
    print("\nMost Common Focal Channels:")
    all_focal = []
    for focal_str in df_summary['Focal Channels']:
        if isinstance(focal_str, str) and focal_str:
            all_focal.extend(focal_str.split(', '))
    
    from collections import Counter
    focal_counts = Counter(all_focal)
    for ch, count in focal_counts.most_common(10):
        print(f"  {ch}: {count} subjects ({100*count/len(all_results):.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - subject_summary.csv")
    print(f"  - summary_analysis.png")
    print(f"  - detailed_results.json")
    print(f"  - individual_subjects/*.png ({len(plot_subjects)} plots)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()