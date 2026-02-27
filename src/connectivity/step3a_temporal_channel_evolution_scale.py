"""
TUC Dataset - Temporal Evolution Per-Channel with FIXED Y-AXIS
===============================================================
FIXED VERSION addressing professor's feedback:

1. SHARED Y-AXIS: All 19 channel subplots use SAME y-axis range for direct comparison
2. CORRECTED TIME: Time 0 = first ictal epoch (not metadata, uses actual connectivity data)
3. Individual plots per channel with statistics

Usage:
------
python temporal_channel_evolution_FIXED.py \
    --connectivity_dir F:\FORTH_Final_Thesis\FORTH-Thesis\connectivity \
    --epochs_dir F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs \
    --output_dir F:\FORTH_Final_Thesis\FORTH-Thesis\figures\temporal_detailed \
    --band integrated \
    --metric pdc \
    --subject_ids 1 2 3
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
    
    # # column sum = outflow from each channel
    out_strength = np.sum(connectivity_matrix, axis=1)  # (n_epochs, n_channels)
    
     # row sum = inflow to each channel
    in_strength = np.sum(connectivity_matrix, axis=2)  # (n_epochs, n_channels)
    
    # Total strength
    total_strength = out_strength + in_strength
    
    return {
        'out_strength': out_strength,
        'in_strength': in_strength,
        'total_strength': total_strength
    }


def plot_temporal_evolution_single_subject(
    graphs_file, 
    epochs_dir,
    output_dir, 
    band, 
    metric
):
    """
    Create temporal evolution plots for one subject with FIXED Y-AXIS.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load connectivity data
    data = np.load(graphs_file)
    subject_name = Path(graphs_file).stem.replace('_graphs', '')
    
    connectivity_key = f'{metric}_{band}'
    if connectivity_key not in data:
        print(f"❌ {connectivity_key} not found in {graphs_file}")
        return
    
    connectivity = data[connectivity_key]  # (n_epochs, 19, 19)
    labels = data['labels']  # (n_epochs,) - already filtered
    
    # =========================================================================
    # CRITICAL: Reconstruct correct time axis based on ACTUAL ictal epochs
    # =========================================================================
    # Find first and last ictal epochs in the FILTERED data
    ictal_indices = np.where(labels == 1)[0]
    
    if len(ictal_indices) == 0:
        print(f"⚠️  No ictal epochs found for {subject_name}")
        return
    
    first_ictal_idx = ictal_indices[0]
    last_ictal_idx = ictal_indices[-1]
    
    # Create time array where time=0 at first ictal epoch
    # Each epoch is 4 seconds
    n_epochs = len(labels)
    time = np.arange(n_epochs) * 4.0  # Epoch indices × 4 seconds
    time = time - time[first_ictal_idx]  # Shift so first ictal = 0
    
    print(f"\n{'='*80}")
    print(f"TIMING CHECK - {subject_name}")
    print(f"{'='*80}")
    print(f"Total epochs with connectivity: {n_epochs}")
    print(f"First ictal epoch index: {first_ictal_idx} → time = {time[first_ictal_idx]:.1f}s")
    print(f"Last ictal epoch index:  {last_ictal_idx} → time = {time[last_ictal_idx]:.1f}s")
    print(f"Ictal duration (epochs): {last_ictal_idx - first_ictal_idx + 1}")
    print(f"Ictal duration (seconds): {(last_ictal_idx - first_ictal_idx + 1) * 4}s")
    print(f"{'='*80}\n")
    
    # Compute per-channel strengths
    strengths = compute_channel_strengths(connectivity)
    total_strength = strengths['total_strength']  # (n_epochs, 19)
    
    # =========================================================================
    # PLOT 1: Heatmap (Channels × Time)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 8))
    
    im = ax.imshow(total_strength.T, aspect='auto', cmap='hot', interpolation='nearest')
    
    # Mark seizure period (using actual indices)
    ax.axvline(first_ictal_idx, color='cyan', linestyle='--', linewidth=2, label='Seizure Start')
    ax.axvline(last_ictal_idx, color='cyan', linestyle='--', linewidth=2, label='Seizure End')
    ax.axvspan(first_ictal_idx, last_ictal_idx, alpha=0.2, color='red', label='Ictal Period')
    
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
    # PLOT 2: Line Plots (Per-Channel Time Series) - FIXED Y-AXIS
    # =========================================================================
    
    # ========================================================================
    # CRITICAL FIX: Compute GLOBAL y-axis limits across ALL channels
    # =========================================================================
    global_ymin = np.min(total_strength)
    global_ymax = np.max(total_strength)
    
    # Add 5% padding
    y_range = global_ymax - global_ymin
    global_ymin -= 0.05 * y_range
    global_ymax += 0.05 * y_range
    
    print(f"📊 Global y-axis range for grid plots: [{global_ymin:.2f}, {global_ymax:.2f}]")
    print(f"   This will be SHARED across all 19 channel subplots!\n")
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    for ch_idx, ch_name in enumerate(CHANNELS):
        ax = axes[ch_idx]
        
        # Plot connectivity strength over time
        ax.plot(time, total_strength[:, ch_idx], linewidth=1.5, color='steelblue')
        
        # Shade ictal period
        ax.axvspan(time[first_ictal_idx], time[last_ictal_idx], 
                  alpha=0.3, color='red', label='Ictal')
        
        # Mark time=0 (seizure onset)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
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
        
        # =====================================================================
        # CRITICAL FIX: Use GLOBAL y-limits
        # =====================================================================
        ax.set_ylim([global_ymin, global_ymax])
        
        ax.set_title(f'{ch_name} {sig_marker}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time from seizure onset (sec)', fontsize=9)
        ax.set_ylabel('Strength', fontsize=9)
        ax.grid(alpha=0.3)
    
    # Remove extra subplots
    for idx in range(len(CHANNELS), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'{subject_name} - Per-Channel Temporal Evolution\n'
                 f'{metric.upper()} {band} band (*** p<0.001, ** p<0.01, * p<0.05)\n'
                 f'Y-axis FIXED: [{global_ymin:.2f}, {global_ymax:.2f}] (shared across all channels)',
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
    parser = argparse.ArgumentParser(
        description="Temporal evolution per-channel analysis with FIXED Y-AXIS"
    )
    parser.add_argument("--connectivity_dir", required=True,
                       help="Directory with connectivity files")
    parser.add_argument("--epochs_dir", required=True,
                       help="Directory with epoch metadata")
    parser.add_argument("--output_dir", required=True, 
                       help="Output directory for plots")
    parser.add_argument("--band", default="integrated", 
                       choices=['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1'])
    parser.add_argument("--metric", default="pdc", choices=['dtf', 'pdc'])
    parser.add_argument("--subject_ids", nargs='+', type=int,
                       default=list(range(1, 35)),
                       help="Subject IDs to process")
    
    args = parser.parse_args()
    
    connectivity_dir = Path(args.connectivity_dir)
    epochs_dir = Path(args.epochs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TEMPORAL EVOLUTION - PER-CHANNEL WITH FIXED Y-AXIS")
    print("=" * 80)
    print(f"Metric: {args.metric.upper()}")
    print(f"Band: {args.band}")
    print(f"Output: {output_dir}")
    print(f"Subjects: {len(args.subject_ids)}")
    print("=" * 80)
    print("\nFIXES APPLIED:")
    print("  ✅ SHARED Y-AXIS across all 19 channel subplots")
    print("  ✅ Time 0 = first ictal epoch (from actual connectivity data)")
    print("  ✅ Corrected time alignment for skipped epochs")
    print("=" * 80)
    
    success = 0
    errors = 0
    
    for subj_id in args.subject_ids:
        subject_name = f"subject_{subj_id:02d}"
        graphs_file = connectivity_dir / f"{subject_name}_graphs.npz"
        
        if not graphs_file.exists():
            print(f"\n⚠️  Skip {subject_name}: file not found")
            errors += 1
            continue
        
        print(f"\nProcessing {subject_name}...")
        
        try:
            plot_temporal_evolution_single_subject(
                graphs_file, 
                epochs_dir,
                output_dir, 
                args.band, 
                args.metric
            )
            success += 1
        except Exception as e:
            print(f"❌ Error processing {subject_name}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Success: {success}/{len(args.subject_ids)}")
    print(f"❌ Errors:  {errors}/{len(args.subject_ids)}")
    print("=" * 80)
    print("\nPlots created per subject:")
    print("  • *_heatmap_*.png       - Channels × Time heatmap")
    print("  • *_timeseries_*.png    - 19 channel subplots (FIXED Y-AXIS)")
    print("  • *_comparison_*.png    - Statistical comparison plots")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()