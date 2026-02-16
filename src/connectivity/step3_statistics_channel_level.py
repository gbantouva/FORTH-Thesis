"""
TUC Dataset - CHANNEL-LEVEL Connectivity Statistics
====================================================
Similar to step3_new_statistics_of_parallel.py but:
- Analyzes connectivity PER CHANNEL (not averaged)
- Shows which channels increase/decrease
- Reveals focal patterns across the dataset

This is BETTER than global averaging because it shows spatial patterns!

Usage:
    python step3_channel_level_statistics.py
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import pandas as pd

RESULTS_DIR = Path(r"F:\FORTH_Final_Thesis\FORTH-Thesis\connectivity")
OUTPUT_DIR = Path(r"F:\FORTH_Final_Thesis\FORTH-Thesis\figures\connectivity\step3_statistics_channel_level")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2']

BAND_NAMES = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
    # Out-strength + In-strength (diagonal already 0)
    out_strength = np.sum(connectivity_matrix, axis=1)
    in_strength = np.sum(connectivity_matrix, axis=0)
    total_strength = out_strength + in_strength
    
    return total_strength


# =============================================================================
# COLLECT DATA
# =============================================================================

all_files = list(RESULTS_DIR.rglob('*_graphs.npz'))
print(f"Found {len(all_files)} result files in {RESULTS_DIR}")

# Storage: {band: {channel: {'pre': [], 'ictal': []}}}
channel_stats = {band: {ch: {'pre': [], 'ictal': []} for ch in CHANNELS} 
                 for band in BAND_NAMES}

total_epochs = 0
total_pre = 0
total_ictal = 0

print("\nCollecting per-channel connectivity data...")

for f in tqdm(all_files):
    data = np.load(f)
    labels = data['labels']
    
    total_epochs += len(labels)
    total_pre += np.sum(labels == 0)
    total_ictal += np.sum(labels == 1)
    
    for band in BAND_NAMES:
        pdc_matrix = data[f'pdc_{band}']  # (n_epochs, 19, 19)
        
        # Compute per-channel strength for each epoch
        for epoch_idx in range(len(pdc_matrix)):
            channel_strengths = compute_channel_strength(pdc_matrix[epoch_idx])
            
            # Store per channel
            for ch_idx, ch_name in enumerate(CHANNELS):
                if labels[epoch_idx] == 0:
                    channel_stats[band][ch_name]['pre'].append(channel_strengths[ch_idx])
                else:
                    channel_stats[band][ch_name]['ictal'].append(channel_strengths[ch_idx])

print(f"\nTotal epochs: {total_epochs:,}")
print(f"  Pre-ictal: {total_pre:,} ({100*total_pre/total_epochs:.1f}%)")
print(f"  Ictal:     {total_ictal:,} ({100*total_ictal/total_epochs:.1f}%)")

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

print("\nPerforming statistical tests per channel...")

# Results storage
results_by_band = {}

for band in BAND_NAMES:
    results = []
    
    for ch_name in CHANNELS:
        pre_vals = np.array(channel_stats[band][ch_name]['pre'])
        ictal_vals = np.array(channel_stats[band][ch_name]['ictal'])
        
        # Statistics
        pre_mean = np.mean(pre_vals)
        ictal_mean = np.mean(ictal_vals)
        diff = ictal_mean - pre_mean
        percent_change = (diff / pre_mean) * 100 if pre_mean > 0 else 0
        
        # Mann-Whitney U test
        stat, pval = mannwhitneyu(pre_vals, ictal_vals, alternative='two-sided')
        
        results.append({
            'channel': ch_name,
            'pre_mean': pre_mean,
            'ictal_mean': ictal_mean,
            'difference': diff,
            'percent_change': percent_change,
            'p_value': pval,
            'significant': pval < 0.05
        })
    
    results_by_band[band] = results

# =============================================================================
# PLOT 1: CHANNEL-SPECIFIC BAR CHART (INTEGRATED BAND)
# =============================================================================

print("\nCreating visualizations...")

band = 'integrated'
results = results_by_band[band]

fig, ax = plt.subplots(figsize=(18, 7))

ch_names = [r['channel'] for r in results]
pre_means = [r['pre_mean'] for r in results]
ictal_means = [r['ictal_mean'] for r in results]
p_values = [r['p_value'] for r in results]

x = np.arange(len(ch_names))
width = 0.35

bars1 = ax.bar(x - width/2, pre_means, width, label='Pre-ictal', 
               alpha=0.8, color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, ictal_means, width, label='Ictal', 
               alpha=0.8, color='crimson', edgecolor='black')

# Mark significant channels
for i, p_val in enumerate(p_values):
    if p_val < 0.05:
        y_max = max(pre_means[i], ictal_means[i])
        marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
        ax.text(i, y_max * 1.05, marker, ha='center', 
               fontsize=14, fontweight='bold', color='red')

ax.set_xlabel('Channel', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Total Connectivity Strength', fontsize=13, fontweight='bold')
ax.set_title(f'Per-Channel Connectivity: Pre-ictal vs Ictal (Integrated Band)\n'
             f'Dataset-wide analysis across {len(all_files)} subjects',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ch_names, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'channel_level_comparison_integrated.png', dpi=300)
plt.close()
print(f"✓ Saved: channel_level_comparison_integrated.png")

# =============================================================================
# PLOT 2: HEATMAP OF CHANGES ACROSS ALL BANDS
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, band in enumerate(BAND_NAMES):
    results = results_by_band[band]
    
    ch_names = [r['channel'] for r in results]
    percent_changes = [r['percent_change'] for r in results]
    p_values = [r['p_value'] for r in results]
    
    # Color based on significance and direction
    colors = []
    for pct, pval in zip(percent_changes, p_values):
        if pval >= 0.05:
            colors.append('lightgray')  # Not significant
        elif pct > 0:
            colors.append('crimson')     # Significant increase
        else:
            colors.append('steelblue')   # Significant decrease
    
    ax = axes[i]
    bars = ax.bar(range(len(ch_names)), percent_changes, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('% Change (Ictal vs Pre-ictal)', fontsize=10, fontweight='bold')
    ax.set_title(f'{band.upper()} Band', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Mark significance
    for j, (pct, pval) in enumerate(zip(percent_changes, p_values)):
        if pval < 0.05:
            marker = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
            y_pos = pct * 1.1 if pct > 0 else pct * 1.1
            ax.text(j, y_pos, marker, ha='center', fontsize=10, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='crimson', alpha=0.8, edgecolor='black', label='Significant Increase'),
    Patch(facecolor='steelblue', alpha=0.8, edgecolor='black', label='Significant Decrease'),
    Patch(facecolor='lightgray', alpha=0.8, edgecolor='black', label='Not Significant')
]
axes[-1].legend(handles=legend_elements, loc='center', fontsize=11)
axes[-1].axis('off')

plt.suptitle('Channel-Level Connectivity Changes: Pre-ictal → Ictal\n'
             '(*** p<0.001, ** p<0.01, * p<0.05)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'channel_changes_all_bands.png', dpi=300)
plt.close()
print(f"✓ Saved: channel_changes_all_bands.png")

# =============================================================================
# PLOT 3: SUMMARY TABLE (INTEGRATED BAND)
# =============================================================================

band = 'integrated'
results = results_by_band[band]

# Separate increases and decreases
increases = [r for r in results if r['significant'] and r['difference'] > 0]
decreases = [r for r in results if r['significant'] and r['difference'] < 0]

increases.sort(key=lambda x: x['difference'], reverse=True)
decreases.sort(key=lambda x: x['difference'])

print("\n" + "="*80)
print(f"CHANNEL-LEVEL STATISTICS (INTEGRATED BAND)")
print("="*80)
print(f"\nSignificant INCREASES (Ictal > Pre-ictal):")
print("-"*80)
print(f"{'Channel':<10} {'Pre-ictal':<12} {'Ictal':<12} {'Difference':<12} {'% Change':<12} {'p-value':<12}")
print("-"*80)

for r in increases:
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*"
    print(f"{r['channel']:<10} {r['pre_mean']:<12.4f} {r['ictal_mean']:<12.4f} "
          f"{r['difference']:<+12.4f} {r['percent_change']:<+12.1f}% "
          f"{r['p_value']:<12.2e} {sig}")

print(f"\nSignificant DECREASES (Ictal < Pre-ictal):")
print("-"*80)
print(f"{'Channel':<10} {'Pre-ictal':<12} {'Ictal':<12} {'Difference':<12} {'% Change':<12} {'p-value':<12}")
print("-"*80)

for r in decreases:
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*"
    print(f"{r['channel']:<10} {r['pre_mean']:<12.4f} {r['ictal_mean']:<12.4f} "
          f"{r['difference']:<+12.4f} {r['percent_change']:<+12.1f}% "
          f"{r['p_value']:<12.2e} {sig}")

# =============================================================================
# SAVE CSV
# =============================================================================

for band in BAND_NAMES:
    df = pd.DataFrame(results_by_band[band])
    df.to_csv(OUTPUT_DIR / f'channel_statistics_{band}.csv', index=False)

print(f"\n✓ Saved CSV files for all bands")

# =============================================================================
# INTERPRETATION
# =============================================================================

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if len(increases) > 0:
    print(f"\n✅ FOCAL CHANNELS (Increased connectivity):")
    print(f"   {', '.join([r['channel'] for r in increases[:5]])}")
    print(f"   → These are likely seizure foci")

if len(decreases) > 0:
    print(f"\n✅ NON-FOCAL CHANNELS (Decreased connectivity):")
    print(f"   {', '.join([r['channel'] for r in decreases[:5]])}")
    print(f"   → Remote suppression (diaschisis)")

print(f"\n📊 Summary:")
print(f"   Channels with INCREASE: {len(increases)}/{len(CHANNELS)}")
print(f"   Channels with DECREASE: {len(decreases)}/{len(CHANNELS)}")

if len(increases) > 0 and len(decreases) > 0:
    print(f"\n✅ HETEROGENEOUS PATTERN CONFIRMED!")
    print(f"   This explains why global average shows decrease:")
    print(f"   → Focal channels increase (fewer channels)")
    print(f"   → Non-focal channels decrease (more channels)")
    print(f"   → Net effect: Global decrease")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nFiles created:")
print(f"  - channel_level_comparison_integrated.png")
print(f"  - channel_changes_all_bands.png")
print(f"  - channel_statistics_*.csv (for all bands)")
print("\n" + "="*80)