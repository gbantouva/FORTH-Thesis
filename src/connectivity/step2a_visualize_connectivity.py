"""
VISUALIZATION TOOL - FIXED COLOR SCALE VERSION
===============================================
- Reads your .npz connectivity files
- Plots 2x7 Heatmaps (DTF/PDC across all 7 bands)
- FIXED: Uses [0, 1] color scale for consistent comparison
- FIXED: Channel names (T3/T4 instead of A1/A2)
- Allows you to pick specific epochs

CRITICAL CHANGE:
----------------
PDC and DTF are theoretically bounded [0, 1]. 
Now using vmin=0, vmax=1.0 for ALL plots.

This ensures:
- Same colors = same values across ALL epochs and subjects
- Direct visual comparison is possible
- Addresses professor's feedback about inconsistent colors

Usage example:
--------------
python visualize_connectivity_FIXED.py \
    --file F:\FORTH-DATA\Thesis\connectivity\subject_01_graphs.npz \
    --output_dir F:\FORTH-DATA\Thesis\figures\visualize_connectivity_fixed \
    --epochs 0 10 20 30 40 50 60 70

PS F:\FORTH_Final_Thesis\FORTH-Thesis\src> & C:/Users/georg/AppData/Local/Programs/Python/Python311/python.exe f:/FORTH_Final_Thesis/FORTH-Thesis/src/connectivity/step2a_visualize_connectivity.py --file F:\FORTH_Final_Thesis\FORTH-Thesis\connectivity\subject_01_graphs.npz --epochs 0 1 2 3 4 5 10 15 20 25 30 35 50 70 72 73 74 75 76 77 78 79 80 85 86 87 88 89 90 91 92 93 94 95 100 110 111 112 113 114 115 120 --output_dir F:\FORTH_Final_Thesis\FORTH-Thesis\figures\connectivity\step2a_visualize_connectivity_scale
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T3', 'C3', 'Cz', 'C4', 'T4',
                 'T5', 'P3', 'Pz', 'P4', 'T6', 
                 'O1', 'O2']  # ← 19 channels (TUC)


# Band order for plotting
BAND_ORDER = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']

BAND_LABELS = {
    'integrated': 'Integrated (0.5-45 Hz)',
    'delta': 'Delta (δ, 0.5-4 Hz)',
    'theta': 'Theta (θ, 4-8 Hz)',
    'alpha': 'Alpha (α, 8-15 Hz)',
    'beta': 'Beta (β, 15-30 Hz)',
    'gamma1': 'Gamma1 (γ1, 30-45 Hz)'
}

# ============================================================================
# PLOTTING LOGIC
# ============================================================================

def plot_epoch(data, epoch_idx, patient_id, output_dir):
    """Generates the 2x7 grid for a single epoch with FIXED [0,1] color scale."""
    
    # 1. Check if epoch exists
    n_epochs = len(data['orders'])
    if epoch_idx >= n_epochs:
        print(f"Epoch {epoch_idx} out of bounds (File has {n_epochs} epochs). Skipping.")
        return

    # 2. Extract Data
    matrices = {}
    actual_max = 0  # Track actual max for diagnostic purposes
    
    for band in BAND_ORDER:
        # Load the specific epoch's matrix (19x19)
        d = data[f'dtf_{band}'][epoch_idx]
        p = data[f'pdc_{band}'][epoch_idx]
        
        matrices[f'dtf_{band}'] = d
        matrices[f'pdc_{band}'] = p
        
        # Track actual maximum (for diagnostic printing)
        actual_max = max(actual_max, d.max(), p.max())
    
    # =========================================================================
    # CRITICAL FIX: Use FIXED color scale [0, 1]
    # =========================================================================
    # PDC and DTF are theoretically bounded [0, 1]
    # Using fixed scale ensures colors are comparable across all plots
    FIXED_VMIN = 0.0
    FIXED_VMAX = 1.0
    
    # Print diagnostic if values exceed expected range
    if actual_max > 1.0:
        print(f"⚠️  Warning: Epoch {epoch_idx} has values > 1.0 (max={actual_max:.3f})")
        print(f"   This suggests normalization issue in connectivity computation.")

    # 3. Setup the Grid
    n_bands = len(BAND_ORDER)
    fig, axes = plt.subplots(2, n_bands, figsize=(4.5*n_bands, 8))

    # Get Label
    if 'labels' in data:
        label = "Ictal" if data['labels'][epoch_idx] == 1 else "Pre-ictal"
    else:
        label = "Unknown"
    p_order = data['orders'][epoch_idx]

    # 4. Fill the Grid
    for col, band in enumerate(BAND_ORDER):
        # ========================================================================
        # TOP ROW: DTF (no colorbar) with FIXED scale
        # ========================================================================
        sns.heatmap(matrices[f'dtf_{band}'], ax=axes[0, col], 
                    cmap='viridis', square=True, 
                    vmin=FIXED_VMIN, vmax=FIXED_VMAX,  # ← FIXED SCALE
                    cbar=False, 
                    xticklabels=[], 
                    yticklabels=CHANNEL_NAMES if col == 0 else [],
                    linewidths=0.5, linecolor='gray')
        axes[0, col].set_title(f'DTF\n{band.capitalize()}', fontsize=10, fontweight='bold')
        
        if col == 0:
            axes[0, col].set_ylabel('Sink (To)', fontsize=10, fontweight='bold')
        
        # ========================================================================
        # BOTTOM ROW: PDC with FIXED scale (colorbar on last column only)
        # ========================================================================
        if col == n_bands - 1:
            # Last column: heatmap + thin colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            
            sns.heatmap(matrices[f'pdc_{band}'], ax=axes[1, col], 
                        cmap='viridis', square=True, 
                        vmin=FIXED_VMIN, vmax=FIXED_VMAX,  # ← FIXED SCALE
                        cbar=False,
                        xticklabels=CHANNEL_NAMES, 
                        yticklabels=CHANNEL_NAMES if col == 0 else [],
                        linewidths=0.5, linecolor='gray')
            
            # Add colorbar with FIXED scale
            divider = make_axes_locatable(axes[1, col])
            cax = divider.append_axes("right", size="3%", pad=0.05)
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                       norm=plt.Normalize(vmin=FIXED_VMIN, vmax=FIXED_VMAX))
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Connectivity Strength', rotation=270, labelpad=15)
        else:
            # All other columns: no colorbar
            sns.heatmap(matrices[f'pdc_{band}'], ax=axes[1, col], 
                        cmap='viridis', square=True, 
                        vmin=FIXED_VMIN, vmax=FIXED_VMAX,  # ← FIXED SCALE
                        cbar=False,
                        xticklabels=CHANNEL_NAMES, 
                        yticklabels=CHANNEL_NAMES if col == 0 else [],
                        linewidths=0.5, linecolor='gray')
        
        # PDC labels (apply to all columns)
        axes[1, col].set_title(f'PDC\n{band.capitalize()}', fontsize=10, fontweight='bold')
        axes[1, col].tick_params(axis='x', rotation=90, labelsize=8)
        
        if col == 0:
            axes[1, col].set_ylabel('Sink (To)', fontsize=10, fontweight='bold')
        
        if col == n_bands // 2:
            axes[1, col].set_xlabel('Source (From)', fontsize=10, fontweight='bold')

    # ========================================================================
    # LAYOUT AND SAVING
    # ========================================================================
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])

    # Updated title to show actual max and fixed scale
    fig.suptitle(
        f'{patient_id} | Epoch {epoch_idx} ({label}) | Order p={p_order} | Color scale: [0.0, 1.0] | Actual max={actual_max:.3f}',
        fontsize=14, fontweight='bold'
    )

    save_name = f"{patient_id}_ep{epoch_idx:03d}_{label}.png"
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_name}")


def plot_single_band_comparison(data, epoch_idx, patient_id, output_dir, band='integrated'):
    """Plot a single band with FIXED [0,1] scale for thesis figures."""
    
    n_epochs = len(data['orders'])
    if epoch_idx >= n_epochs:
        print(f"Epoch {epoch_idx} out of bounds")
        return
    
    dtf = data[f'dtf_{band}'][epoch_idx]
    pdc = data[f'pdc_{band}'][epoch_idx]
    
    if 'labels' in data:
        label = "Ictal" if data['labels'][epoch_idx] == 1 else "Pre-ictal"
    else:
        label = "Unknown"

    p_order = data['orders'][epoch_idx]
    
    # =========================================================================
    # CRITICAL FIX: Use FIXED [0, 1] scale
    # =========================================================================
    FIXED_VMIN = 0.0
    FIXED_VMAX = 1.0
    
    actual_max = max(dtf.max(), pdc.max())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # DTF with FIXED scale
    sns.heatmap(dtf, ax=axes[0], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=FIXED_VMIN, vmax=FIXED_VMAX,  # ← FIXED
               cbar_kws={'label': 'Connectivity Strength'},
               linewidths=0.5, linecolor='white')
    axes[0].set_title(f'DTF - {BAND_LABELS[band]}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Source (From)', fontsize=12)
    axes[0].set_ylabel('Sink (To)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # PDC with FIXED scale
    sns.heatmap(pdc, ax=axes[1], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=FIXED_VMIN, vmax=FIXED_VMAX,  # ← FIXED
               cbar_kws={'label': 'Connectivity Strength'},
               linewidths=0.5, linecolor='white')
    axes[1].set_title(f'PDC - {BAND_LABELS[band]}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Source (From)', fontsize=12)
    axes[1].set_ylabel('Sink (To)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    fig.suptitle(
        f'{patient_id} | Epoch {epoch_idx} ({label}) | Order p={p_order} | Color scale: [0.0, 1.0] | Actual max={actual_max:.3f}', 
        fontsize=16, fontweight='bold'
    )
    
    plt.tight_layout()
    
    save_name = f"{patient_id}_ep{epoch_idx:03d}_{band}_detailed.png"
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved detailed plot: {save_name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Connectivity Matrices with FIXED [0,1] color scale"
    )
    parser.add_argument("--file", required=True, help="Path to a single .npz file")
    parser.add_argument("--output_dir", required=True, help="Where to save images")
    parser.add_argument("--epochs", nargs='+', type=int, default=[0, 10, 20], 
                       help="List of epoch indices to plot (e.g. 0 5 10)")
    parser.add_argument("--all_epilepsy", action="store_true", 
                       help="If set, plots ALL epilepsy epochs")
    parser.add_argument("--all_control", action="store_true",
                       help="If set, plots ALL control epochs")
    parser.add_argument("--detailed_band", type=str, default=None,
                       choices=['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1'],
                       help="Generate detailed plots for a specific band")
    parser.add_argument("--max_plots", type=int, default=50,
                       help="Maximum number of plots to generate (safety limit)")

    args = parser.parse_args()
    
    file_path = Path(args.file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return

    print(f"Loading: {file_path.name}")
    data = np.load(file_path)
    patient_id = file_path.stem.replace('_graphs', '')
    
    print(f"   Total epochs: {len(data['orders'])}")
    if 'labels' in data:
        print(f"   Ictal epochs: {np.sum(data['labels'] == 1)}")
        print(f"   Pre-ictal epochs: {np.sum(data['labels'] == 0)}")
    
    print(f"\n{'='*80}")
    print(f"USING FIXED COLOR SCALE: [0.0, 1.0]")
    print(f"This ensures colors are comparable across ALL epochs and subjects")
    print(f"{'='*80}\n")

    # Determine which epochs to plot
    epochs_to_plot = list(args.epochs)
    
    if args.all_epilepsy and 'labels' in data:
        epilepsy_indices = np.where(data['labels'] == 1)[0]
        if len(epilepsy_indices) > 0:
            print(f"Found {len(epilepsy_indices)} epilepsy epochs! Adding them.")
            epochs_to_plot.extend(list(epilepsy_indices))
        else:
            print("No epilepsy epochs found.")
    
    if args.all_control and 'labels' in data:
        control_indices = np.where(data['labels'] == 0)[0]
        if len(control_indices) > 0:
            print(f"Found {len(control_indices)} control epochs! Adding them.")
            epochs_to_plot.extend(list(control_indices))
        else:
            print("No control epochs found.")
    
    # Remove duplicates and sort
    epochs_to_plot = sorted(list(set(epochs_to_plot)))
    
    # Safety limit
    if len(epochs_to_plot) > args.max_plots:
        print(f"⚠️  Too many plots requested ({len(epochs_to_plot)}). Limiting to {args.max_plots}.")
        epochs_to_plot = epochs_to_plot[:args.max_plots]
    
    print(f"Generating plots for {len(epochs_to_plot)} epochs\n")
    
    # Generate full 2x7 grid plots
    for ep in epochs_to_plot:
        plot_epoch(data, ep, patient_id, output_dir)
    
    # Generate detailed single-band plots if requested
    if args.detailed_band:
        print(f"\nGenerating detailed {args.detailed_band} band plots...")
        for ep in epochs_to_plot:
            plot_single_band_comparison(data, ep, patient_id, output_dir, args.detailed_band)
    
    print(f"\n✅ Done! Images saved in: {output_dir}")
    print(f"   Generated {len(epochs_to_plot)} full plots")
    if args.detailed_band:
        print(f"   Generated {len(epochs_to_plot)} detailed {args.detailed_band} plots")
    print(f"\n{'='*80}")
    print(f"All plots use FIXED color scale [0.0, 1.0]")
    print(f"→ Same colors = same values across ALL figures")
    print(f"→ Direct visual comparison is now possible!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()