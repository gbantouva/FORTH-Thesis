"""
TUC Dataset - Temporal Evolution of Per-Channel Connectivity
=============================================================
Visualizes how each channel's connectivity strength evolves over time.

This shows which channels are focal (increase during seizure) vs not focal.

Usage:
    # Single subject
    python temporal_channel_evolution.py \
        --file connectivity/subject_01_graphs.npz \
        --output_dir figures/temporal_evolution \
        --band integrated \
        --metric pdc
    
    # All subjects
    python temporal_channel_evolution.py \
        --connectivity_dir connectivity \
        --output_dir figures/temporal_evolution \
        --band integrated \
        --metric pdc \
        --plot_subjects 5
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# TUC channels
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2']

# Channel groups for interpretation
CHANNEL_GROUPS = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Temporal': ['T3', 'T4', 'T5', 'T6'],
    'Central': ['C3', 'Cz', 'C4'],
    'Parietal': ['P3', 'Pz', 'P4'],
    'Occipital': ['O1', 'O2']
}


def compute_channel_strengths(connectivity_matrix):
    """
    Compute per-channel connectivity strengths.
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Shape (n_epochs, n_channels, n_channels)
    
    Returns:
    --------
    dict with 'out_strength', 'in_strength', 'total_strength'
        Each shape (n_epochs, n_channels)
    """
    n_epochs, n_channels, _ = connectivity_matrix.shape
    
    # Out-strength: sum of outgoing connections (row sum)
    out_strength = np.sum(connectivity_matrix, axis=2)  # (n_epochs, n_channels)
    
    # In-strength: sum of incoming connections (column sum)
    in_strength = np.sum(connectivity_matrix, axis=1)  # (n_epochs, n_channels)
    
    # Total strength
    total_strength = out_strength + in_strength
    
    return {
        'out_strength': out_strength,
        'in_strength': in_strength,
        'total_strength': total_strength
    }


def plot_temporal_evolution_single_subject(file_path, output_dir, band, metric):
    """
    Create temporal evolution plots for one subject.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = np.load(file_path)
    subject_name = Path(file_path).stem.replace('_graphs', '')
    
    connectivity_key = f'{metric}_{band}'
    if connectivity_key not in data:
        print(f"❌ {connectivity_key} not found in {file_path}")
        return
    
    connectivity = data[connectivity_key]  # (n_epochs, 19, 19)
    labels = data['labels']  # (n_epochs,)
    
    # Get time_from_onset if available
    if 'time_from_onset' in data:
        time = data['time_from_onset']
    else:
        time = np.arange(len(labels)) * 4.0  # 4-second epochs
    
    # Compute per-channel strengths
    strengths = compute_channel_strengths(connectivity)
    total_strength = strengths['total_strength']  # (n_epochs, 19)
    
    # Find seizure period
    ictal_mask = labels == 1
    seizure_start_idx = np.where(ictal_mask)[0][0] if np.any(ictal_mask) else None
    seizure_end_idx = np.where(ictal_mask)[0][-1] if np.any(ictal_mask) else None
    
    # =========================================================================
    # PLOT 1: Heatmap (Channels × Time)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 8))
    
    im = ax.imshow(total_strength.T, aspect='auto', cmap='hot', interpolation='nearest')
    
    # Mark seizure period
    if seizure_start_idx is not None:
        ax.axvline(seizure_start_idx, color='cyan', linestyle='--', linewidth=2, label='Seizure Start')
        ax.axvline(seizure_end_idx, color='cyan', linestyle='--', linewidth=2, label='Seizure End')
        ax.axvspan(seizure_start_idx, seizure_end_idx, alpha=0.2, color='red', label='Ictal Period')
    
    ax.set_xlabel('Epoch Index (Time →)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Channel', fontsize=12, fontweight='bold')
    ax.set_yticks(range(19))
    ax.set_yticklabels(CHANNELS, fontsize=10)
    ax.set_title(f'{subject_name} - Temporal Evolution of Connectivity Strength\n'
                 f'{metric.upper()} {band} band',
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Total Connectivity Strength')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{subject_name}_heatmap_{metric}_{band}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # PLOT 2: Line Plots (Per-Channel Time Series)
    # =========================================================================
    fig, axes = plt.subplots(5, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    for ch_idx, ch_name in enumerate(CHANNELS):
        ax = axes[ch_idx]
        
        # Plot connectivity strength over time
        ax.plot(time, total_strength[:, ch_idx], linewidth=1.5, color='steelblue')
        
        # Mark seizure period
        if seizure_start_idx is not None:
            ax.axvspan(time[seizure_start_idx], time[seizure_end_idx], 
                      alpha=0.3, color='red', label='Ictal')
        
        # Statistics
        pre_ictal = total_strength[labels == 0, ch_idx]
        ictal = total_strength[labels == 1, ch_idx]
        
        if len(ictal) > 0:
            _, p_val = mannwhitneyu(pre_ictal, ictal)
            sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            
            # Add mean lines
            ax.axhline(pre_ictal.mean(), color='blue', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(ictal.mean(), color='red', linestyle='--', alpha=0.5, linewidth=1)
        else:
            sig_marker = ''
        
        ax.set_title(f'{ch_name} {sig_marker}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (sec)', fontsize=9)
        ax.set_ylabel('Strength', fontsize=9)
        ax.grid(alpha=0.3)
    
    # Remove extra subplots
    for idx in range(len(CHANNELS), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'{subject_name} - Per-Channel Temporal Evolution\n'
                 f'{metric.upper()} {band} band (*** p<0.001, ** p<0.01, * p<0.05)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{subject_name}_timeseries_{metric}_{band}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # PLOT 3: Channel Group Comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pre-ictal vs Ictal per channel
    ax = axes[0]
    pre_means = []
    ictal_means = []
    ch_names = []
    p_values = []
    
    for ch_idx, ch_name in enumerate(CHANNELS):
        pre = total_strength[labels == 0, ch_idx]
        ict = total_strength[labels == 1, ch_idx]
        
        if len(ict) > 0:
            pre_means.append(pre.mean())
            ictal_means.append(ict.mean())
            ch_names.append(ch_name)
            
            _, p_val = mannwhitneyu(pre, ict)
            p_values.append(p_val)
    
    x = np.arange(len(ch_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pre_means, width, label='Pre-ictal', alpha=0.8, color='blue')
    bars2 = ax.bar(x + width/2, ictal_means, width, label='Ictal', alpha=0.8, color='red')
    
    # Mark significant channels
    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            y_max = max(pre_means[i], ictal_means[i])
            marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            ax.text(i, y_max * 1.05, marker, ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Connectivity Strength', fontsize=12, fontweight='bold')
    ax.set_title('Pre-ictal vs Ictal Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Channel group averages
    ax = axes[1]
    
    group_pre = []
    group_ictal = []
    group_names = []
    
    for group_name, group_channels in CHANNEL_GROUPS.items():
        group_indices = [CHANNELS.index(ch) for ch in group_channels if ch in CHANNELS]
        
        pre = total_strength[labels == 0][:, group_indices].mean()
        ict = total_strength[labels == 1][:, group_indices].mean() if np.any(labels == 1) else 0
        
        group_pre.append(pre)
        group_ictal.append(ict)
        group_names.append(group_name)
    
    x = np.arange(len(group_names))
    bars1 = ax.bar(x - width/2, group_pre, width, label='Pre-ictal', alpha=0.8, color='blue')
    bars2 = ax.bar(x + width/2, group_ictal, width, label='Ictal', alpha=0.8, color='red')
    
    ax.set_xlabel('Channel Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Connectivity Strength', fontsize=12, fontweight='bold')
    ax.set_title('Connectivity by Brain Region', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'{subject_name} - Statistical Comparison\n{metric.upper()} {band} band',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{subject_name}_comparison_{metric}_{band}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created plots for {subject_name}")


def main():
    parser = argparse.ArgumentParser(description="Temporal evolution per-channel analysis")
    parser.add_argument("--file", type=str, help="Single connectivity file to analyze")
    parser.add_argument("--connectivity_dir", type=str, help="Directory with all connectivity files")
    parser.add_argument("--output_dir", required=True, help="Output directory for plots")
    parser.add_argument("--band", default="integrated", 
                       choices=['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1'])
    parser.add_argument("--metric", default="pdc", choices=['dtf', 'pdc'])
    parser.add_argument("--plot_subjects", type=int, default=5, 
                       help="Number of subjects to plot (if using --connectivity_dir)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TEMPORAL EVOLUTION - PER-CHANNEL CONNECTIVITY")
    print("=" * 80)
    print(f"Metric: {args.metric.upper()}")
    print(f"Band: {args.band}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    if args.file:
        # Single file
        print(f"\nProcessing: {args.file}")
        plot_temporal_evolution_single_subject(args.file, output_dir, args.band, args.metric)
    
    elif args.connectivity_dir:
        # Multiple files
        conn_dir = Path(args.connectivity_dir)
        files = sorted(conn_dir.glob("subject_*_graphs.npz"))
        
        print(f"\nFound {len(files)} files")
        print(f"Plotting first {args.plot_subjects} subjects...")
        
        for i, file in enumerate(files[:args.plot_subjects]):
            print(f"\n[{i+1}/{min(args.plot_subjects, len(files))}] {file.name}")
            plot_temporal_evolution_single_subject(file, output_dir, args.band, args.metric)
    
    else:
        print("\n❌ Provide either --file or --connectivity_dir")
        return
    
    print("\n" + "=" * 80)
    print("✅ TEMPORAL EVOLUTION PLOTS CREATED")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nPlots created:")
    print("  • *_heatmap_*.png       - Channels × Time heatmap")
    print("  • *_timeseries_*.png    - Per-channel line plots (19 subplots)")
    print("  • *_comparison_*.png    - Statistical comparison plots")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()