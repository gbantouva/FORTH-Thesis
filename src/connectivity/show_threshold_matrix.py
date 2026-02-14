"""
Visualize Surrogate Thresholds as 19×19 Matrix
===============================================
Shows the threshold value for each channel pair in matrix format.

Three visualization styles:
1. Heatmap of threshold values
2. PDC matrix with threshold values as text annotations
3. Side-by-side comparison: PDC vs Thresholds

Usage:
    python show_threshold_matrix.py \
        --results_file thresholds/subject_01/subject_01_surrogate_results.json \
        --output_dir figures
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2']


def plot_threshold_heatmap(thresholds, output_path, subject_name):
    """
    Style 1: Simple heatmap showing threshold values.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(thresholds, cmap='plasma', vmin=0, vmax=np.max(thresholds))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Threshold Value (95th Percentile)', 
                   fontsize=12, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(range(19))
    ax.set_yticks(range(19))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(CHANNELS, fontsize=10)
    
    # Labels
    ax.set_xlabel('Target Channel (TO)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Source Channel (FROM)', fontsize=12, fontweight='bold')
    ax.set_title(f'{subject_name} - Surrogate-Based Thresholds\n'
                 f'(Each cell shows the 95th percentile of surrogate PDC)',
                 fontsize=13, fontweight='bold', pad=15)
    
    # Add grid
    ax.set_xticks(np.arange(19) - 0.5, minor=True)
    ax.set_yticks(np.arange(19) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Style 1: Threshold heatmap saved")


def plot_pdc_with_threshold_text(pdc_real, thresholds, output_path, subject_name):
    """
    Style 2: PDC as background, threshold values as text.
    Each cell shows: PDC value (color) + threshold (number).
    """
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # PDC as background
    im = ax.imshow(pdc_real, cmap='hot', vmin=0, vmax=np.max(pdc_real))
    
    # Annotate with threshold values
    for i in range(19):
        for j in range(19):
            if i == j:
                # Diagonal: just show channel name
                text = CHANNELS[i]
                color = 'black'
                fontsize = 10
                fontweight = 'bold'
            else:
                # Off-diagonal: show threshold
                thresh = thresholds[i, j]
                real = pdc_real[i, j]
                
                # Format: show threshold value
                text = f'{thresh:.3f}'
                
                # Color based on significance
                if real > thresh:
                    color = 'lime'  # Significant
                    fontweight = 'bold'
                    fontsize = 8
                else:
                    color = 'lightgray'  # Not significant
                    fontweight = 'normal'
                    fontsize = 7
            
            ax.text(j, i, text, ha='center', va='center',
                   color=color, fontsize=fontsize, fontweight=fontweight)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PDC Value (Background Color)', fontsize=11, fontweight='bold')
    
    # Ticks
    ax.set_xticks(range(19))
    ax.set_yticks(range(19))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CHANNELS, fontsize=9)
    
    # Labels
    ax.set_xlabel('Target Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Source Channel', fontsize=12, fontweight='bold')
    ax.set_title(f'{subject_name} - PDC Matrix with Threshold Values\n'
                 f'Background Color: PDC | Text: Threshold (Lime=Significant, Gray=Not)',
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Style 2: PDC with threshold text saved")


def plot_comparison_side_by_side(pdc_real, thresholds, pdc_thresholded, 
                                 output_path, subject_name):
    """
    Style 3: Three panels comparing PDC, Thresholds, and Result.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    # Common settings
    vmax = max(pdc_real.max(), thresholds.max())
    
    # Panel 1: Original PDC
    ax = axes[0]
    im1 = ax.imshow(pdc_real, cmap='hot', vmin=0, vmax=vmax)
    ax.set_title('(A) Original PDC Values', fontsize=13, fontweight='bold')
    ax.set_xticks(range(19))
    ax.set_yticks(range(19))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CHANNELS, fontsize=8)
    ax.set_xlabel('Target', fontsize=10)
    ax.set_ylabel('Source', fontsize=10)
    plt.colorbar(im1, ax=ax, fraction=0.046)
    
    # Panel 2: Thresholds
    ax = axes[1]
    im2 = ax.imshow(thresholds, cmap='plasma', vmin=0, vmax=vmax)
    ax.set_title('(B) Surrogate Thresholds (95%)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(19))
    ax.set_yticks(range(19))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CHANNELS, fontsize=8)
    ax.set_xlabel('Target', fontsize=10)
    ax.set_ylabel('Source', fontsize=10)
    plt.colorbar(im2, ax=ax, fraction=0.046)
    
    # Panel 3: Thresholded result
    ax = axes[2]
    im3 = ax.imshow(pdc_thresholded, cmap='hot', vmin=0, vmax=vmax)
    ax.set_title('(C) Significant Connections Only\n(PDC > Threshold)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(19))
    ax.set_yticks(range(19))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CHANNELS, fontsize=8)
    ax.set_xlabel('Target', fontsize=10)
    ax.set_ylabel('Source', fontsize=10)
    plt.colorbar(im3, ax=ax, fraction=0.046)
    
    # Overall title
    plt.suptitle(f'{subject_name} - Surrogate Thresholding Process',
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Style 3: Side-by-side comparison saved")


def create_threshold_table(thresholds, pdc_real, output_path):
    """
    Style 4: Create a text table showing threshold values.
    Useful for appendix or supplementary materials.
    """
    with open(output_path, 'w') as f:
        f.write("Surrogate-Based Thresholds (95th Percentile)\n")
        f.write("=" * 100 + "\n\n")
        
        # Header
        f.write("FROM \\ TO  ")
        for ch in CHANNELS:
            f.write(f"{ch:>7}")
        f.write("\n")
        f.write("-" * 100 + "\n")
        
        # Rows
        for i, ch_from in enumerate(CHANNELS):
            f.write(f"{ch_from:>10}  ")
            for j in range(19):
                if i == j:
                    f.write(f"{'---':>7}")
                else:
                    thresh = thresholds[i, j]
                    real = pdc_real[i, j]
                    
                    # Mark significant with asterisk
                    if real > thresh:
                        f.write(f"{thresh:>6.3f}*")
                    else:
                        f.write(f"{thresh:>7.3f}")
            f.write("\n")
        
        f.write("\n* = Significant (Real PDC > Threshold)\n")
    
    print(f"✅ Style 4: Threshold table (text) saved")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize threshold matrix"
    )
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--output_dir", required=True)
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    pdc_real = np.array(results['pdc_real'])
    thresholds = np.array(results['thresholds'])
    pdc_thresholded = np.array(results['pdc_thresholded'])
    subject_name = results['subject']
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"VISUALIZING THRESHOLDS FOR {subject_name}")
    print("="*80)
    print(f"\nThreshold range: {thresholds.min():.4f} - {thresholds.max():.4f}")
    print(f"PDC range: {pdc_real.min():.4f} - {pdc_real.max():.4f}")
    print(f"Significant connections: {results['n_significant_connections']}/{results['n_total_connections']}")
    print("\n" + "="*80)
    
    # Create all visualizations
    print("\nCreating visualizations...\n")
    
    plot_threshold_heatmap(
        thresholds,
        output_dir / f'{subject_name}_threshold_heatmap.png',
        subject_name
    )
    
    plot_pdc_with_threshold_text(
        pdc_real,
        thresholds,
        output_dir / f'{subject_name}_pdc_with_thresholds.png',
        subject_name
    )
    
    plot_comparison_side_by_side(
        pdc_real,
        thresholds,
        pdc_thresholded,
        output_dir / f'{subject_name}_comparison_panels.png',
        subject_name
    )
    
    create_threshold_table(
        thresholds,
        pdc_real,
        output_dir / f'{subject_name}_threshold_table.txt'
    )
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nCreated 4 files:")
    print(f"  1. {subject_name}_threshold_heatmap.png")
    print(f"     → Heatmap showing threshold values")
    print(f"  2. {subject_name}_pdc_with_thresholds.png")
    print(f"     → PDC matrix with threshold numbers as text")
    print(f"  3. {subject_name}_comparison_panels.png")
    print(f"     → Side-by-side: PDC | Thresholds | Result")
    print(f"  4. {subject_name}_threshold_table.txt")
    print(f"     → Text table of threshold values")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()