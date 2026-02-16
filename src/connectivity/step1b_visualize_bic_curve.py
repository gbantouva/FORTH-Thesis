"""
INDIVIDUAL EPOCH BIC CURVE VISUALIZATION
=========================================
Shows how BIC changes with model order for a specific epoch.
Similar to the example image: BIC curve with optimal order marked.

Usage:
    python visualize_epoch_bic_curve.py \
        --epochs_file path/to/patient_epochs.npy \
        --epoch_idx 0 \
        --output_dir bic_curves

    python visualize_epoch_bic_curve.py --epochs_file "F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001_epochs.npy" --epoch_idx 0 1 2 3 4 5 6 7 8 --output_dir "F:\October-Thesis\thesis-epilepsy-gnn\connectivity\january_final_fixed_order\bic_curves"
    F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced\00_epilepsy\aaaaaicb\s002_2011\01_tcp_ar\aaaaaicb_s002_t000_epochs.npy
    python visualize_epoch_bic_curve.py --epochs_file "F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced\00_epilepsy\aaaaaicb\s002_2011\01_tcp_ar\aaaaaicb_s002_t000_epochs.npy" --epoch_idx 0 5 10 15 20 25 30 50 75 90 100 --output_dir "F:\October-Thesis\thesis-epilepsy-gnn\connectivity\january_final_fixed_order\bic_curves"

    PS F:\FORTH-DATA> & C:/Users/georg/AppData/Local/Programs/Python/Python311/python.exe f:/FORTH-DATA/Thesis/src/final_connectivity/visualize_bic_curve.py --epochs_file F:\FORTH-DATA\Thesis\FORTH_preprocessed_epochs_label\subject_05_epochs.npy --labels_file "F:\FORTH-DATA\Thesis\FORTH_preprocessed_epochs_label\subject_05_labels.npy"  --epoch_idx 0 1 2 3 4 5 6 7 8 9 10 15 20 25 26 27 28 29 30 35 
40 50 55 60 61 62 63 64 65 70 75 100 125 150 --output_dir F:\FORTH-DATA\Thesis\figures\bic_curve_examples --min_order 5 --max_order 22
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# BIC COMPUTATION FOR SINGLE EPOCH
# ============================================================================

def compute_bic_curve(data, min_order=8, max_order=22):
    """
    Compute BIC values for different model orders.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Single epoch data
    min_order : int
        Minimum order to test
    max_order : int
        Maximum order to test
        
    Returns
    -------
    orders : list
        List of orders tested
    bic_values : list
        Corresponding BIC values (lower is better)
    best_order : int
        Optimal order (minimum BIC)
    best_bic : float
        BIC value at optimal order
    """
    # Scale data for numerical stability
    data_std = np.std(data)
    if data_std < 1e-10:
        return None, None, None, None
    
    data_scaled = data / data_std
    
    # Fit VAR for different orders
    try:
        model = VAR(data_scaled.T)  # Transpose: (n_times, n_channels)
        
        orders = []
        bic_values = []
        
        for p in range(min_order, max_order + 1):
            try:
                result = model.fit(maxlags=p, trend='c', verbose=False)
                
                # CRITICAL: Only append if both order AND bic are valid
                bic = result.bic
                if not np.isnan(bic) and not np.isinf(bic):
                    orders.append(p)
                    bic_values.append(bic)
                    
            except Exception as e:
                # Skip this order if fitting fails
                continue
        
        if len(orders) == 0:
            return None, None, None, None
        
        # Verify lengths match (defensive check)
        if len(orders) != len(bic_values):
            print(f"Warning: Length mismatch! orders={len(orders)}, bic_values={len(bic_values)}")
            return None, None, None, None
        
        # Find optimal order
        best_idx = np.argmin(bic_values)
        best_order = orders[best_idx]
        best_bic = bic_values[best_idx]
        
        return orders, bic_values, best_order, best_bic
        
    except Exception as e:
        print(f"Error computing BIC curve: {e}")
        return None, None, None, None


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_bic_curve(orders, bic_values, best_order, best_bic, 
                   epoch_idx, patient_id, output_dir, label=None):
    """
    Create BIC curve plot similar to the example image.
    
    Parameters
    ----------
    orders : list
        Model orders tested
    bic_values : list
        BIC values for each order
    best_order : int
        Optimal order (minimum BIC)
    best_bic : float
        BIC value at optimal order
    epoch_idx : int
        Epoch index
    patient_id : str
        Patient identifier
    output_dir : Path
        Where to save the plot
    label : str, optional
        "EPILEPSY" or "CONTROL"
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot BIC curve
    ax.plot(orders, bic_values, 'o-', linewidth=2, markersize=8, 
            color='#2E86AB', label='BIC Score')
    
    # Mark optimal order with red star
    ax.plot(best_order, best_bic, '*', markersize=20, 
            color='red', label=f'Optimal Order: p={best_order}', zorder=5)
    
    # Add vertical line at optimal order
    ax.axvline(best_order, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Shade regions
    # Underfitting region (left of optimal)
    if best_order > min(orders):
        ax.axvspan(min(orders), best_order, alpha=0.15, color='orange', 
                   label='Underfitting region')
    
    # Overfitting region (right of optimal)
    if best_order < max(orders):
        ax.axvspan(best_order, max(orders), alpha=0.15, color='red', 
                   label='Overfitting region')
    
    # Add statistics text box
    bic_range = max(bic_values) - min(bic_values)
    stats_text = f"Statistics:\n"
    stats_text += f"Optimal order: {best_order}\n"
    stats_text += f"Optimal BIC: {best_bic:.1f}\n"
    stats_text += f"BIC range: {min(bic_values):.1f} to {max(bic_values):.1f}\n"
    stats_text += f"Parameters: {best_order * 19 * 19 + 19} coefficients"
    
    # Position text box in upper left
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels and title
    ax.set_xlabel('Model Order (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BIC Score (lower is better)', fontsize=12, fontweight='bold')
    
    title = f'BIC-Based Model Order Selection for Epoch {epoch_idx}\n'
    title += f'{patient_id}'
    if label:
        title += f' ({label})'
    title += f' | Optimal Order: p={best_order}'
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # Set x-axis to show all integer orders
    ax.set_xticks(range(min(orders), max(orders) + 1))
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    save_name = f"{patient_id}_epoch{epoch_idx:03d}_bic_curve.png"
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_name}")
    print(f"   Optimal order: p={best_order}, BIC={best_bic:.2f}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize BIC curve for individual epochs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single epoch
  python visualize_epoch_bic_curve.py --epochs_file data_pp/patient_epochs.npy --epoch_idx 0 --output_dir bic_curves
  
  # Multiple epochs
  python visualize_epoch_bic_curve.py --epochs_file data_pp/patient_epochs.npy --epoch_idx 0 5 10 20 --output_dir bic_curves
  
  # With label file
  python visualize_epoch_bic_curve.py --epochs_file data_pp/patient_epochs.npy --labels_file data_pp/patient_labels.npy --epoch_idx 0 --output_dir bic_curves
        """
    )
    
    parser.add_argument("--epochs_file", required=True, 
                       help="Path to *_epochs.npy file")
    parser.add_argument("--labels_file", default=None,
                       help="Path to *_labels.npy file (optional, for labeling plots)")
    parser.add_argument("--epoch_idx", nargs='+', type=int, default=[0],
                       help="Epoch indices to plot (e.g., 0 5 10)")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for plots")
    parser.add_argument("--min_order", type=int, default=5,
                       help="Minimum MVAR order to test (default: 8)")
    parser.add_argument("--max_order", type=int, default=18,
                       help="Maximum MVAR order to test (default: 22)")
    
    args = parser.parse_args()
    
    # Validate paths
    epochs_file = Path(args.epochs_file)
    if not epochs_file.exists():
        print(f"âŒ Error: Epochs file not found: {epochs_file}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading: {epochs_file.name}")
    epochs = np.load(epochs_file)
    print(f"   Shape: {epochs.shape} (n_epochs, n_channels, n_times)")
    print(f"   Total epochs: {len(epochs)}")
    
    # Load labels if available
    labels = None
    if args.labels_file:
        labels_file = Path(args.labels_file)
        if labels_file.exists():
            labels = np.load(labels_file)
            print(f"   Labels loaded: {len(labels)}")
        else:
            print(f" Labels file not found, skipping")
    
    # Extract patient ID from filename
    patient_id = epochs_file.stem.replace('_epochs', '')
    
    # Process each requested epoch
    print(f"\nGenerating BIC curves for {len(args.epoch_idx)} epochs...")
    print(f"   Order range: {args.min_order}-{args.max_order}")
    print(f"   Output directory: {output_dir}\n")
    
    for idx in args.epoch_idx:
        if idx >= len(epochs):
            print(f" Epoch {idx} out of bounds (max: {len(epochs)-1}), skipping")
            continue
        
        print(f"Processing epoch {idx}...")
        
        # Get epoch data
        epoch_data = epochs[idx]  # Shape: (n_channels, n_times)
        
        # Get label if available
        # Get label if available
        label_str = None
        if labels is not None:
            label_str = "Ictal" if labels[idx] == 1 else "Pre-ictal"
        
        # Compute BIC curve
        orders, bic_vals, best_order, best_bic = compute_bic_curve(
            epoch_data, 
            min_order=args.min_order, 
            max_order=args.max_order
        )
        
        if orders is None:
            print(f"Failed to compute BIC curve for epoch {idx}")
            continue
        
        # Plot
        plot_bic_curve(
            orders, bic_vals, best_order, best_bic,
            idx, patient_id, output_dir, label_str
        )
    
    print(f"\nDone! Saved {len(args.epoch_idx)} BIC curve plots to: {output_dir}")


if __name__ == "__main__":
    main()