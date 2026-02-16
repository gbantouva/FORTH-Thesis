"""
Frequency-Domain PDC Visualization
===================================
Creates 19×19 grid showing PDC as a function of frequency for each channel pair.
Similar to the reference thesis visualization.

This shows HOW connectivity strength varies across the frequency spectrum,
rather than just averaging over frequency bands.

Usage:

    pythhon visualize_frequency_domain_threshold.py \
    --file preprocessed_epochs/subject_01_epochs.npy \
    --output_dir figures/freq_with_thresholds \
    --condition pre_ictal \
    --thresholds_file thresholds/subject_01_pre/subject_01_thresholds.npy
    
    python visualize_frequency_domain_pdc.py \
        --file connectivity/subject_01_graphs.npz \
        --output_dir figures/frequency_domain \
        --metric pdc \
        --condition pre_ictal
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal

# Channel names
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2']

# Frequency bands for highlighting
BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 15.0),
    'beta': (15.0, 30.0),
    'gamma': (30.0, 45.0)
}


def compute_frequency_domain_pdc(epochs, fs=256, nfft=512, fixed_order=12):
    """
    Compute PDC in frequency domain for given epochs.
    
    Parameters:
    -----------
    epochs : np.ndarray
        Shape (n_epochs, n_channels, n_timepoints)
    fs : float
        Sampling frequency
    nfft : int
        FFT length
    fixed_order : int
        MVAR model order
    
    Returns:
    --------
    pdc_spectrum : np.ndarray
        Shape (n_channels, n_channels, n_freqs)
        PDC values across frequency for each channel pair
    freqs : np.ndarray
        Frequency vector
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    from scipy import linalg
    
    n_epochs, n_channels, n_timepoints = epochs.shape
    n_freqs = nfft // 2 + 1
    freqs = np.linspace(0, fs/2, n_freqs)
    
    # Average PDC across epochs
    pdc_sum = np.zeros((n_channels, n_channels, n_freqs))
    valid_epochs = 0
    
    for epoch_idx in range(n_epochs):
        data = epochs[epoch_idx]  # (n_channels, n_timepoints)
        
        # Standardize
        data_std = np.std(data)
        if data_std < 1e-10:
            continue
        data_scaled = data / data_std
        
        try:
            # Fit VAR model
            model = VAR(data_scaled.T)  # Transpose to (n_timepoints, n_channels)
            results = model.fit(maxlags=fixed_order, trend='c', verbose=False)
            
            if results.k_ar == 0:
                continue
            
            # Get VAR coefficients
            coefs = results.coefs  # (p, n_channels, n_channels)
            p, K, _ = coefs.shape
            
            # Compute A(f) for each frequency
            A_f = np.zeros((n_freqs, K, K), dtype=complex)
            I = np.eye(K)
            
            for f_idx, f in enumerate(freqs):
                A_sum = np.zeros((K, K), dtype=complex)
                for k in range(p):
                    phase = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
                    A_sum += coefs[k] * phase
                
                A_f[f_idx] = I - A_sum
            
            # Compute PDC (column-wise normalization)
            pdc = np.zeros((K, K, n_freqs))
            for f_idx in range(n_freqs):
                Af = A_f[f_idx]
                col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))
                col_norms[col_norms == 0] = 1e-10
                pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]
            
            pdc_sum += pdc
            valid_epochs += 1
            
        except:
            continue
    
    # Average across epochs
    if valid_epochs > 0:
        pdc_spectrum = pdc_sum / valid_epochs
    else:
        pdc_spectrum = np.zeros((n_channels, n_channels, n_freqs))
    
    return pdc_spectrum, freqs


def plot_frequency_domain_grid(pdc_spectrum, freqs, output_path, 
                               title="PDC Frequency Domain", 
                               highlight_bands=True,
                               thresholds=None):
    """
    Create 19×19 grid of frequency plots.
    
    Parameters:
    -----------
    pdc_spectrum : np.ndarray (19, 19, n_freqs)
    freqs : np.ndarray (n_freqs,)
    output_path : Path
    thresholds : np.ndarray (19, 19), optional
        Threshold values to overlay as horizontal lines
    """
    n_channels = len(CHANNELS)
    
    # Create figure with tight spacing
    fig = plt.figure(figsize=(24, 24))
    gs = GridSpec(n_channels, n_channels, figure=fig, 
                  wspace=0.05, hspace=0.05,
                  left=0.05, right=0.98, top=0.96, bottom=0.04)
    
    # Global max for consistent y-axis
    global_max = pdc_spectrum.max()
    
    for i in range(n_channels):
        for j in range(n_channels):
            ax = fig.add_subplot(gs[i, j])
            
            if i == j:
                # Diagonal: show channel name
                ax.text(0.5, 0.5, CHANNELS[i], 
                       ha='center', va='center',
                       fontsize=10, fontweight='bold')
                ax.set_xlim(0, 45)
                ax.set_ylim(0, 1)
                ax.axis('off')
            else:
                # Off-diagonal: plot PDC(f)
                pdc_values = pdc_spectrum[i, j, :]  # From j to i
                
                # Plot frequency curve
                ax.plot(freqs, pdc_values, 'b-', linewidth=1)
                ax.fill_between(freqs, pdc_values, alpha=0.3)
                
                # ADD THRESHOLD LINE (if provided)
                if thresholds is not None:
                    threshold_val = thresholds[i, j]
                    ax.axhline(threshold_val, color='red', linestyle='--', 
                              linewidth=1.5, alpha=0.8, label='Threshold')
                    
                    # Check if significant (mean PDC > threshold)
                    mean_pdc = pdc_values.mean()
                    if mean_pdc > threshold_val:
                        # Mark as significant
                        ax.text(0.95, 0.95, '✓', transform=ax.transAxes,
                               fontsize=8, color='green', fontweight='bold',
                               ha='right', va='top')
                
                # Highlight frequency bands with background colors
                if highlight_bands:
                    ax.axvspan(0.5, 4.0, alpha=0.1, color='purple')   # delta
                    ax.axvspan(4.0, 8.0, alpha=0.1, color='blue')     # theta
                    ax.axvspan(8.0, 15.0, alpha=0.1, color='green')   # alpha
                    ax.axvspan(15.0, 30.0, alpha=0.1, color='yellow') # beta
                    ax.axvspan(30.0, 45.0, alpha=0.1, color='red')    # gamma
                
                # Set limits
                ax.set_xlim(0, 45)
                ax.set_ylim(0, global_max * 1.1)
                
                # Remove tick labels except edges
                if i < n_channels - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Hz', fontsize=6)
                    ax.tick_params(labelsize=6)
                
                if j > 0:
                    ax.set_yticklabels([])
                else:
                    ax.tick_params(labelsize=6)
                
                # Grid
                ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Add row/column labels
    fig.text(0.01, 0.5, 'Source Channel (FROM)', 
            va='center', rotation='vertical', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.01, 'Target Channel (TO)', 
            ha='center', fontsize=12, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def create_simplified_grid(pdc_spectrum, freqs, output_path, title="PDC Frequency Domain", thresholds=None):
    """
    Create a more readable version with larger plots (10×10 subset).
    """
    # Select subset of channels for clarity
    selected_indices = [0, 1, 2, 3, 4, 7, 8, 11, 12, 15]  # 10 channels
    selected_channels = [CHANNELS[i] for i in selected_indices]
    
    n_selected = len(selected_indices)
    
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(n_selected, n_selected, figure=fig,
                  wspace=0.1, hspace=0.1)
    
    global_max = pdc_spectrum.max()
    
    for row_idx, i in enumerate(selected_indices):
        for col_idx, j in enumerate(selected_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            if i == j:
                # Diagonal
                ax.text(0.5, 0.5, selected_channels[row_idx],
                       ha='center', va='center',
                       fontsize=12, fontweight='bold')
                ax.set_xlim(0, 45)
                ax.set_ylim(0, 1)
                ax.axis('off')
            else:
                # Plot PDC
                pdc_values = pdc_spectrum[i, j, :]
                ax.plot(freqs, pdc_values, 'b-', linewidth=1.5)
                ax.fill_between(freqs, pdc_values, alpha=0.3, color='steelblue')
                
                # ADD THRESHOLD LINE
                if thresholds is not None:
                    threshold_val = thresholds[i, j]
                    ax.axhline(threshold_val, color='red', linestyle='--',
                              linewidth=2, alpha=0.9)
                    
                    # Check significance
                    mean_pdc = pdc_values.mean()
                    if mean_pdc > threshold_val:
                        ax.text(0.95, 0.95, '✓', transform=ax.transAxes,
                               fontsize=10, color='green', fontweight='bold',
                               ha='right', va='top',
                               bbox=dict(boxstyle='round', facecolor='white', 
                                        alpha=0.8, edgecolor='green'))
                
                # Frequency bands
                ax.axvspan(0.5, 4.0, alpha=0.05, color='purple')
                ax.axvspan(8.0, 15.0, alpha=0.05, color='green')
                ax.axvspan(15.0, 30.0, alpha=0.05, color='yellow')
                
                ax.set_xlim(0, 45)
                ax.set_ylim(0, global_max * 1.1)
                
                if row_idx == n_selected - 1:
                    ax.set_xlabel('Freq (Hz)', fontsize=8)
                else:
                    ax.set_xticklabels([])
                
                if col_idx == 0:
                    ax.set_ylabel('PDC', fontsize=8)
                else:
                    ax.set_yticklabels([])
                
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
    
    title_text = title
    if thresholds is not None:
        title_text += "\n(Red dashed line = 95% threshold, ✓ = Significant)"
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved simplified grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Frequency-domain PDC visualization"
    )
    parser.add_argument("--file", required=True, help="Connectivity file path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metric", default="pdc", choices=['pdc', 'dtf'])
    parser.add_argument("--condition", default="pre_ictal", 
                       choices=['pre_ictal', 'ictal', 'both'])
    parser.add_argument("--fixed_order", type=int, default=12)
    parser.add_argument("--create_simplified", action="store_true",
                       help="Also create 10×10 simplified version")
    parser.add_argument("--thresholds_file", type=str, default=None,
                       help="Path to thresholds .npy file (optional, adds threshold lines)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FREQUENCY-DOMAIN PDC VISUALIZATION")
    print("=" * 80)
    print(f"File: {args.file}")
    print(f"Output: {output_dir}")
    print(f"Metric: {args.metric.upper()}")
    print(f"Condition: {args.condition}")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    file_path = Path(args.file)
    
    # Determine subject name and file type
    if '_epochs.npy' in file_path.name:
        # This is an epoch file
        subject_name = file_path.stem.replace('_epochs', '')
        epochs_file = file_path
        labels_file = file_path.parent / f"{subject_name}_labels.npy"
    elif '_graphs.npz' in file_path.name:
        print("❌ Error: This script needs raw EEG epochs, not connectivity matrices.")
        print("   Please provide the epoch file instead:")
        subject_name = file_path.stem.replace('_graphs', '')
        print(f"   Try: --file preprocessed_epochs/{subject_name}_epochs.npy")
        return
    else:
        print(f"❌ Error: Unrecognized file format: {file_path.name}")
        print("   Expected: subject_XX_epochs.npy")
        return
    
    # Load epochs and labels
    if not epochs_file.exists():
        print(f"❌ Error: Epochs file not found: {epochs_file}")
        return
    
    if not labels_file.exists():
        print(f"❌ Error: Labels file not found: {labels_file}")
        return
    
    print(f"  Loading epochs: {epochs_file.name}")
    print(f"  Loading labels: {labels_file.name}")
    
    epochs = np.load(epochs_file)
    labels = np.load(labels_file)
    
    # Separate conditions
    if args.condition == 'pre_ictal':
        selected_epochs = epochs[labels == 0]
        title_suffix = "Pre-ictal"
    elif args.condition == 'ictal':
        selected_epochs = epochs[labels == 1]
        title_suffix = "Ictal"
    else:
        selected_epochs = epochs
        title_suffix = "All"
    
    print(f"\nComputing frequency-domain {args.metric.upper()}...")
    print(f"  Epochs: {len(selected_epochs)}")
    print(f"  Condition: {args.condition}")
    
    # Compute PDC spectrum
    pdc_spectrum, freqs = compute_frequency_domain_pdc(
        selected_epochs, 
        fs=256, 
        nfft=512, 
        fixed_order=args.fixed_order
    )
    
    print(f"  PDC spectrum shape: {pdc_spectrum.shape}")
    print(f"  Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    
    # Load thresholds if provided
    thresholds = None
    if args.thresholds_file:
        print(f"\nLoading thresholds from: {args.thresholds_file}")
        thresholds = np.load(args.thresholds_file)
        print(f"  Thresholds shape: {thresholds.shape}")
        print(f"  Threshold range: {thresholds.min():.4f} - {thresholds.max():.4f}")
    
    # Create full 19×19 grid
    print("\nCreating full 19×19 grid...")
    title = f"{subject_name} - {args.metric.upper()} Frequency Domain ({title_suffix})"
    output_path = output_dir / f"{subject_name}_{args.metric}_freq_domain_{args.condition}.png"
    plot_frequency_domain_grid(pdc_spectrum, freqs, output_path, title, 
                              thresholds=thresholds)
    
    # Create simplified version if requested
    if args.create_simplified:
        print("\nCreating simplified 10×10 grid...")
        simplified_path = output_dir / f"{subject_name}_{args.metric}_freq_domain_{args.condition}_simplified.png"
        create_simplified_grid(pdc_spectrum, freqs, simplified_path, title,
                             thresholds=thresholds)
    
    print("\n" + "=" * 80)
    print("✅ FREQUENCY-DOMAIN VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {output_dir}")
    print("\nNote: Each cell shows PDC as a function of frequency (0-45 Hz)")
    print("      Colored backgrounds indicate frequency bands:")
    print("        Purple: Delta (0.5-4 Hz)")
    print("        Blue: Theta (4-8 Hz)")
    print("        Green: Alpha (8-15 Hz)")
    print("        Yellow: Beta (15-30 Hz)")
    print("        Red: Gamma (30-45 Hz)")
    print("=" * 80)


if __name__ == "__main__":
    main()