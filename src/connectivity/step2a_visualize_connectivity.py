"""
VISUALIZATION TOOL - CORRECTED VERSION
=======================================
- Reads your .npz connectivity files
- Plots 2x7 Heatmaps (DTF/PDC across all 7 bands)
- FIXED: Channel names (T1/T2 instead of A1/A2)
- Allows you to pick specific epochs

PS F:\FORTH-DATA> & C:/Users/georg/AppData/Local/Programs/Python/Python311/python.exe f:/FORTH-DATA/Thesis/src/final_connectivity/visualize_connectivity.py --file F:\FORTH-DATA\Thesis\connectivity\subject_01_graphs.npz --output_dir F:\FORTH-DATA\Thesis\figures\visualize_connectivity --epochs 0 10 20 30 40 50 60 61 62 63 64 65 70 75 100 110 111 112 113 114 115 120
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
    'delta': 'Delta (δ´, 0.5-4 Hz)',
    'theta': 'Theta (θ¸, 4-8 Hz)',
    'alpha': 'Alpha (α, 8-15 Hz)',
    'beta': 'Beta (β, 15-30 Hz)',
    'gamma1': 'Gamma1 (γ1‚, 30-45 Hz)'
}

# ============================================================================
# PLOTTING LOGIC
# ============================================================================

def plot_epoch(data, epoch_idx, patient_id, output_dir):
    """Generates the 2x7 grid for a single epoch."""
    
    # 1. Check if epoch exists
    n_epochs = len(data['orders'])
    if epoch_idx >= n_epochs:
        print(f"Epoch {epoch_idx} out of bounds (File has {n_epochs} epochs). Skipping.")
        return

    # 2. Extract Data & Find Max Value (for consistent colors)
    matrices = {}
    global_max = 0
    
    for band in BAND_ORDER:
        # Load the specific epoch's matrix (22x22)
        d = data[f'dtf_{band}'][epoch_idx]
        p = data[f'pdc_{band}'][epoch_idx]
        
        matrices[f'dtf_{band}'] = d
        matrices[f'pdc_{band}'] = p
        
        global_max = max(global_max, d.max(), p.max())

    # 3. Setup the Grid (2 Rows, 7 Columns)
    #fig, axes = plt.subplots(2, 7, figsize=(28, 8), constrained_layout=True)
    # NEW (dynamic column count based on your bands)
    n_bands = len(BAND_ORDER)
    # NEW
    fig, axes = plt.subplots(2, n_bands, figsize=(4.5*n_bands, 8))

    


    # Get Label (Epilepsy vs Control)
    # NEW (more accurate for temporal analysis)
    #label = "ICTAL" if data['labels'][epoch_idx] == 1 else "PRE-ICTAL"
    # Alternative: Generic temporal indicator
    #label = "Unknown"  # Until labels are fixed
    # Get label from file
    if 'labels' in data:
        label = "Ictal" if data['labels'][epoch_idx] == 1 else "Pre-ictal"
    else:
        label = "Unknown"
    p_order = data['orders'][epoch_idx]

        # 4. Fill the Grid
    for col, band in enumerate(BAND_ORDER):
        # ========================================================================
        # TOP ROW: DTF (no colorbar)
        # ========================================================================
        sns.heatmap(matrices[f'dtf_{band}'], ax=axes[0, col], 
                    cmap='viridis', square=True, vmin=0, vmax=global_max, 
                    cbar=False, 
                    xticklabels=[], 
                    yticklabels=CHANNEL_NAMES if col == 0 else [],
                    linewidths=0.5, linecolor='gray')
        axes[0, col].set_title(f'DTF\n{band.capitalize()}', fontsize=10, fontweight='bold')
        
        if col == 0:
            axes[0, col].set_ylabel('Sink (To)', fontsize=10, fontweight='bold')
        
        # ========================================================================
        # BOTTOM ROW: PDC (colorbar on last column only)
        # ========================================================================
        if col == n_bands - 1:
            # Last column: heatmap + thin colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            
            sns.heatmap(matrices[f'pdc_{band}'], ax=axes[1, col], 
                        cmap='viridis', square=True, vmin=0, vmax=global_max, 
                        cbar=False,
                        xticklabels=CHANNEL_NAMES, 
                        yticklabels=CHANNEL_NAMES if col == 0 else [],
                        linewidths=0.5, linecolor='gray')
            
            # Add colorbar
            divider = make_axes_locatable(axes[1, col])
            cax = divider.append_axes("right", size="3%", pad=0.05)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=global_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Connectivity Strength', rotation=270, labelpad=15)
        else:
            # All other columns: no colorbar
            sns.heatmap(matrices[f'pdc_{band}'], ax=axes[1, col], 
                        cmap='viridis', square=True, vmin=0, vmax=global_max, 
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
    # LAYOUT AND SAVING (outside loop)
    # ========================================================================
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])

    fig.suptitle(f'{patient_id} | Epoch {epoch_idx} ({label}) | Order p={p_order} | Max Connectivity={global_max:.3f}', 
                fontsize=14, fontweight='bold')

    save_name = f"{patient_id}_ep{epoch_idx:03d}_{label}.png"
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_name}")



def plot_single_band_comparison(data, epoch_idx, patient_id, output_dir, band='integrated'):
    """Plot a single band with better detail for thesis figures."""
    
    n_epochs = len(data['orders'])
    if epoch_idx >= n_epochs:
        print(f"Epoch {epoch_idx} out of bounds")
        return
    
    dtf = data[f'dtf_{band}'][epoch_idx]
    pdc = data[f'pdc_{band}'][epoch_idx]
    #label = "ICTAL" if data['labels'][epoch_idx] == 1 else "PRE-ICTAL"
    # Alternative: Generic temporal indicator
    label = "Unknown"  # Until labels are fixed

    p_order = data['orders'][epoch_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    global_max = max(dtf.max(), pdc.max())
    
    # DTF
    sns.heatmap(dtf, ax=axes[0], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=0, vmax=global_max,
               cbar_kws={'label': 'Connectivity Strength'},
               linewidths=0.5, linecolor='white')
    axes[0].set_title(f'DTF - {BAND_LABELS[band]}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Source (From)', fontsize=12)
    axes[0].set_ylabel('Sink (To)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # PDC
    sns.heatmap(pdc, ax=axes[1], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=0, vmax=global_max,
               cbar_kws={'label': 'Connectivity Strength'},
               linewidths=0.5, linecolor='white')
    axes[1].set_title(f'PDC - {BAND_LABELS[band]}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Source (From)', fontsize=12)
    axes[1].set_ylabel('Sink (To)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    fig.suptitle(f'{patient_id} | Epoch {epoch_idx} ({label}) | Order p={p_order}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    save_name = f"{patient_id}_ep{epoch_idx:03d}_{band}_detailed.png"
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed plot: {save_name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Connectivity Matrices")
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
        print(f"Error: File not found: {file_path}")
        return

    print(f"Loading: {file_path.name}")
    data = np.load(file_path)
    patient_id = file_path.stem.replace('_graphs', '')
    
    print(f"   Total epochs: {len(data['orders'])}")
    print(f"   Ictal epochs: {np.sum(data['labels'] == 1)}")
    print(f"   Pre-ictal epochs: {np.sum(data['labels'] == 0)}")

    # Determine which epochs to plot
    epochs_to_plot = list(args.epochs)
    
    if args.all_epilepsy:
        epilepsy_indices = np.where(data['labels'] == 1)[0]
        if len(epilepsy_indices) > 0:
            print(f"Found {len(epilepsy_indices)} epilepsy epochs! Adding them.")
            epochs_to_plot.extend(list(epilepsy_indices))
        else:
            print("No epilepsy epochs found.")
    
    if args.all_control:
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
        print(f"âš ï¸  Too many plots requested ({len(epochs_to_plot)}). Limiting to {args.max_plots}.")
        epochs_to_plot = epochs_to_plot[:args.max_plots]
    
    print(f"Generating plots for {len(epochs_to_plot)} epochs")
    
    # Generate full 2x7 grid plots
    for ep in epochs_to_plot:
        plot_epoch(data, ep, patient_id, output_dir)
    
    # Generate detailed single-band plots if requested
    if args.detailed_band:
        print(f"\nGenerating detailed {args.detailed_band} band plots...")
        for ep in epochs_to_plot:
            plot_single_band_comparison(data, ep, patient_id, output_dir, args.detailed_band)
    
    print(f"\Done! Images saved in: {output_dir}")
    print(f"   Generated {len(epochs_to_plot)} full plots")
    if args.detailed_band:
        print(f"   Generated {len(epochs_to_plot)} detailed {args.detailed_band} plots")


if __name__ == "__main__":
    main()