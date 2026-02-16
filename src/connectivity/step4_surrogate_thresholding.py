"""
Surrogate Data Thresholding for PDC Connectivity
=================================================
Implements statistical thresholding using surrogate data via iAAFT
(Iterative Amplitude Adjusted Fourier Transform).

This establishes significance thresholds for each channel pair independently,
identifying connections that exceed chance-level connectivity.

Usage:
    python surrogate_thresholding.py \
        --file preprocessed_epochs/subject_01_epochs.npy \
        --output_dir thresholds \
        --n_surrogates 100 \
        --percentile 95
        
Reference:
    Schreiber & Schmitz (2000). "Surrogate time series"
    Physica D: Nonlinear Phenomena
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm import tqdm
import json


CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2']


# ============================================================================
# IAAFT SURROGATE GENERATION
# ============================================================================

def iaaft_surrogate(signal_data, max_iterations=100, tolerance=1e-6):
    """
    Generate surrogate data using iterative AAFT (iAAFT).
    
    This preserves:
    - Amplitude distribution (histogram)
    - Power spectrum (autocorrelation)
    
    This destroys:
    - Phase relationships between channels
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Original signal (n_timepoints,)
    max_iterations : int
        Maximum iterations for convergence
    tolerance : float
        Convergence tolerance
        
    Returns:
    --------
    surrogate : np.ndarray
        Surrogate signal with same statistical properties
    """
    n = len(signal_data)
    
    # Store original properties
    original_sorted = np.sort(signal_data)
    original_fft = np.fft.fft(signal_data)
    original_power = np.abs(original_fft)
    
    # Initialize with phase-randomized version
    random_phases = np.random.uniform(0, 2*np.pi, n)
    surrogate_fft = original_power * np.exp(1j * random_phases)
    surrogate = np.fft.ifft(surrogate_fft).real
    
    # Iterative refinement
    for iteration in range(max_iterations):
        # Step 1: Match amplitude distribution
        surrogate_sorted_indices = np.argsort(surrogate)
        surrogate_rank_ordered = np.empty_like(surrogate)
        surrogate_rank_ordered[surrogate_sorted_indices] = original_sorted
        
        # Step 2: Match power spectrum
        surrogate_fft = np.fft.fft(surrogate_rank_ordered)
        phases = np.angle(surrogate_fft)
        new_fft = original_power * np.exp(1j * phases)
        new_surrogate = np.fft.ifft(new_fft).real
        
        # Check convergence
        diff = np.mean((new_surrogate - surrogate)**2)
        surrogate = new_surrogate
        
        if diff < tolerance:
            break
    
    return surrogate


def generate_multivariate_surrogates(epochs, n_surrogates=100):
    """
    Generate surrogate data for multivariate EEG.
    
    Parameters:
    -----------
    epochs : np.ndarray
        Shape (n_epochs, n_channels, n_timepoints)
    n_surrogates : int
        Number of surrogate datasets to generate
        
    Returns:
    --------
    surrogates : list of np.ndarray
        Each element: (n_epochs, n_channels, n_timepoints)
    """
    n_epochs, n_channels, n_timepoints = epochs.shape
    
    surrogates = []
    
    print(f"Generating {n_surrogates} surrogate datasets...")
    for _ in tqdm(range(n_surrogates)):
        surrogate_epochs = np.zeros_like(epochs)
        
        # Generate surrogates for each epoch and channel
        for ep_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                original_signal = epochs[ep_idx, ch_idx, :]
                surrogate_signal = iaaft_surrogate(original_signal)
                surrogate_epochs[ep_idx, ch_idx, :] = surrogate_signal
        
        surrogates.append(surrogate_epochs)
    
    return surrogates


# ============================================================================
# PDC COMPUTATION
# ============================================================================

def compute_pdc_single_epoch(epoch_data, model_order=12, fs=256, nfft=512):
    """
    Compute PDC for a single epoch.
    
    Returns:
    --------
    pdc_integrated : np.ndarray
        Shape (n_channels, n_channels) - integrated across frequency
    """
    n_channels, n_timepoints = epoch_data.shape
    
    # Standardize
    data_std = np.std(epoch_data)
    if data_std < 1e-10:
        return np.zeros((n_channels, n_channels))
    
    data_scaled = epoch_data / data_std
    
    try:
        # Fit VAR
        model = VAR(data_scaled.T)
        results = model.fit(maxlags=model_order, trend='c', verbose=False)
        
        if results.k_ar == 0:
            return np.zeros((n_channels, n_channels))
        
        # Get coefficients
        coefs = results.coefs  # (p, K, K)
        p, K, _ = coefs.shape
        
        # Compute A(f)
        freqs = np.linspace(0, fs/2, nfft//2 + 1)
        n_freqs = len(freqs)
        
        A_f = np.zeros((n_freqs, K, K), dtype=complex)
        I = np.eye(K)
        
        for f_idx, f in enumerate(freqs):
            A_sum = np.zeros((K, K), dtype=complex)
            for k in range(p):
                phase = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
                A_sum += coefs[k] * phase
            A_f[f_idx] = I - A_sum
        
        # Compute PDC
        pdc = np.zeros((K, K, n_freqs))
        for f_idx in range(n_freqs):
            Af = A_f[f_idx]
            col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))
            col_norms[col_norms == 0] = 1e-10
            pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]
        
        # Integrate across frequency (0.5-45 Hz)
        freq_mask = (freqs >= 0.5) & (freqs <= 45.0)
        pdc_integrated = np.mean(pdc[:, :, freq_mask], axis=2)
        
        # Set diagonal to 0
        np.fill_diagonal(pdc_integrated, 0)
        
        return pdc_integrated
    
    except:
        return np.zeros((n_channels, n_channels))


def compute_pdc_dataset(epochs, model_order=12):
    """
    Compute PDC for all epochs, return mean.
    
    Returns:
    --------
    pdc_mean : np.ndarray (n_channels, n_channels)
    """
    n_epochs, n_channels, n_timepoints = epochs.shape
    
    pdc_sum = np.zeros((n_channels, n_channels))
    valid_count = 0
    
    for ep_idx in range(n_epochs):
        pdc_ep = compute_pdc_single_epoch(epochs[ep_idx], model_order=model_order)
        if not np.all(pdc_ep == 0):
            pdc_sum += pdc_ep
            valid_count += 1
    
    if valid_count > 0:
        return pdc_sum / valid_count
    else:
        return np.zeros((n_channels, n_channels))


# ============================================================================
# THRESHOLDING
# ============================================================================

def compute_surrogate_thresholds(epochs, n_surrogates=100, percentile=95, model_order=12):
    """
    Compute thresholds using surrogate data.
    
    Returns:
    --------
    thresholds : np.ndarray (n_channels, n_channels)
        Threshold for each channel pair
    surrogate_distributions : np.ndarray (n_channels, n_channels, n_surrogates)
        Full surrogate distributions
    """
    print("\n" + "="*80)
    print("SURROGATE THRESHOLDING")
    print("="*80)
    
    # Generate surrogates
    surrogates = generate_multivariate_surrogates(epochs, n_surrogates=n_surrogates)
    
    # Compute PDC on each surrogate
    print(f"\nComputing PDC on {n_surrogates} surrogate datasets...")
    n_channels = epochs.shape[1]
    surrogate_pdcs = np.zeros((n_channels, n_channels, n_surrogates))
    
    for surr_idx, surrogate in enumerate(tqdm(surrogates)):
        surrogate_pdcs[:, :, surr_idx] = compute_pdc_dataset(surrogate, model_order=model_order)
    
    # Compute thresholds (percentile per channel pair)
    print(f"\nComputing {percentile}th percentile thresholds...")
    thresholds = np.percentile(surrogate_pdcs, percentile, axis=2)
    
    return thresholds, surrogate_pdcs


def apply_thresholds(pdc_matrix, thresholds):
    """
    Apply thresholds to PDC matrix.
    
    Values below threshold → 0
    Values above threshold → kept
    """
    thresholded = pdc_matrix.copy()
    thresholded[pdc_matrix < thresholds] = 0
    return thresholded


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_threshold_histograms(pdc_real, surrogate_distributions, thresholds, 
                              output_dir, subject_name, n_examples=9):
    """
    Create histogram plots showing surrogate distributions and thresholds.
    Similar to Figure 3.10 in reference thesis.
    """
    n_channels = pdc_real.shape[0]
    
    # Select representative channel pairs
    # Find pairs with highest real PDC values
    flat_indices = np.argsort(pdc_real.flatten())[::-1]
    selected_pairs = []
    
    for idx in flat_indices:
        i, j = np.unravel_index(idx, pdc_real.shape)
        if i != j and len(selected_pairs) < n_examples:
            selected_pairs.append((i, j))
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for plot_idx, (i, j) in enumerate(selected_pairs):
        ax = axes[plot_idx]
        
        # Surrogate distribution
        surr_values = surrogate_distributions[i, j, :]
        real_value = pdc_real[i, j]
        threshold = thresholds[i, j]
        
        # Histogram
        ax.hist(surr_values, bins=30, alpha=0.7, color='steelblue', 
               edgecolor='black', label='Surrogate PDC')
        
        # Threshold line
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold (95%)')
        
        # Real value
        ax.axvline(real_value, color='green', linestyle='-', linewidth=2,
                  label=f'Real PDC')
        
        # Labels
        ax.set_xlabel('PDC Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{CHANNELS[i]} → {CHANNELS[j]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Significance indicator
        if real_value > threshold:
            ax.text(0.95, 0.95, '✓ Significant', transform=ax.transAxes,
                   fontsize=10, color='green', fontweight='bold',
                   ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.95, 0.95, '✗ Not Significant', transform=ax.transAxes,
                   fontsize=10, color='red', fontweight='bold',
                   ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'{subject_name} - Surrogate Data Thresholding\n'
                 f'Pairwise PDC Distributions vs Thresholds',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{subject_name}_surrogate_histograms.png', 
               dpi=200, bbox_inches='tight')
    plt.close()


def plot_thresholded_matrices(pdc_real, pdc_thresholded, output_dir, subject_name):
    """
    Show original vs thresholded connectivity matrices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original PDC
    ax = axes[0]
    im = ax.imshow(pdc_real, cmap='hot', vmin=0, vmax=pdc_real.max())
    ax.set_title('Original PDC', fontsize=13, fontweight='bold')
    ax.set_xticks(range(19))
    ax.set_yticks(range(19))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right')
    ax.set_yticklabels(CHANNELS)
    plt.colorbar(im, ax=ax)
    
    # Panel 2: Thresholded PDC
    ax = axes[1]
    im = ax.imshow(pdc_thresholded, cmap='hot', vmin=0, vmax=pdc_real.max())
    ax.set_title('Thresholded PDC (Significant Only)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(19))
    ax.set_yticks(range(19))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right')
    ax.set_yticklabels(CHANNELS)
    plt.colorbar(im, ax=ax)
    
    # Panel 3: Binary significant connections
    ax = axes[2]
    binary = (pdc_thresholded > 0).astype(float)
    im = ax.imshow(binary, cmap='gray_r', vmin=0, vmax=1)
    ax.set_title('Significant Connections (Binary)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(19))
    ax.set_yticks(range(19))
    ax.set_xticklabels(CHANNELS, rotation=45, ha='right')
    ax.set_yticklabels(CHANNELS)
    
    # Count significant connections
    n_significant = np.sum(binary)
    n_total = 19 * 18  # Excluding diagonal
    ax.text(0.5, -0.15, f'Significant: {n_significant}/{n_total} ({100*n_significant/n_total:.1f}%)',
           transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'{subject_name} - PDC Thresholding Results',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{subject_name}_thresholded_matrices.png',
               dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Surrogate data thresholding for PDC connectivity"
    )
    parser.add_argument("--file", required=True, help="Epoch file (subject_XX_epochs.npy)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_surrogates", type=int, default=100,
                       help="Number of surrogate datasets (default: 100)")
    parser.add_argument("--percentile", type=float, default=95,
                       help="Percentile for threshold (default: 95)")
    parser.add_argument("--model_order", type=int, default=12)
    parser.add_argument("--condition", default="pre_ictal",
                       choices=['pre_ictal', 'ictal', 'both'])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SURROGATE DATA THRESHOLDING")
    print("="*80)
    print(f"File: {args.file}")
    print(f"N Surrogates: {args.n_surrogates}")
    print(f"Percentile: {args.percentile}%")
    print(f"Condition: {args.condition}")
    print("="*80)
    
    # Load data
    file_path = Path(args.file)
    subject_name = file_path.stem.replace('_epochs', '')
    labels_file = file_path.parent / f"{subject_name}_labels.npy"
    
    epochs = np.load(file_path)
    labels = np.load(labels_file)
    
    # Select condition
    if args.condition == 'pre_ictal':
        selected_epochs = epochs[labels == 0]
    elif args.condition == 'ictal':
        selected_epochs = epochs[labels == 1]
    else:
        selected_epochs = epochs
    
    print(f"\nEpochs selected: {len(selected_epochs)}")
    
    # Compute real PDC
    print("\nComputing real PDC...")
    pdc_real = compute_pdc_dataset(selected_epochs, model_order=args.model_order)
    
    # Compute thresholds
    thresholds, surrogate_distributions = compute_surrogate_thresholds(
        selected_epochs,
        n_surrogates=args.n_surrogates,
        percentile=args.percentile,
        model_order=args.model_order
    )
    
    # Apply thresholds
    pdc_thresholded = apply_thresholds(pdc_real, thresholds)
    
    # Visualizations
    print("\nCreating visualizations...")
    plot_threshold_histograms(pdc_real, surrogate_distributions, thresholds,
                             output_dir, subject_name)
    plot_thresholded_matrices(pdc_real, pdc_thresholded, output_dir, subject_name)
    
    # Save results
    print("\nSaving results...")
    results = {
        'subject': subject_name,
        'condition': args.condition,
        'n_epochs': len(selected_epochs),
        'n_surrogates': args.n_surrogates,
        'percentile': args.percentile,
        'model_order': args.model_order,
        'pdc_real': pdc_real.tolist(),
        'thresholds': thresholds.tolist(),
        'pdc_thresholded': pdc_thresholded.tolist(),
        'n_significant_connections': int(np.sum(pdc_thresholded > 0)),
        'n_total_connections': 19 * 18
    }
    
    with open(output_dir / f'{subject_name}_surrogate_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save matrices as numpy
    np.save(output_dir / f'{subject_name}_pdc_thresholded.npy', pdc_thresholded)
    np.save(output_dir / f'{subject_name}_thresholds.npy', thresholds)
    
    print("\n" + "="*80)
    print("✅ SURROGATE THRESHOLDING COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  Significant connections: {results['n_significant_connections']}/342")
    print(f"  Percentage: {100*results['n_significant_connections']/342:.1f}%")
    print(f"\nOutput files:")
    print(f"  - {subject_name}_surrogate_histograms.png")
    print(f"  - {subject_name}_thresholded_matrices.png")
    print(f"  - {subject_name}_surrogate_results.json")
    print("="*80)


if __name__ == "__main__":
    main()