"""
TUC Dataset - Compute Connectivity with Fixed Order
====================================================
Computes DTF and PDC connectivity matrices using a FIXED model order from BIC analysis.

Features:
- Fixed MVAR order (from BIC analysis)
- Multi-band support (6 frequency bands)
- Diagonal set to ZERO (inter-channel connectivity only)
- Parallel processing
- Checkpointing (resume capability)
- Handles training mask properly

Usage:
    python step2_compute_connectivity_tuc.py \
        --inputdir F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs \
        --outputdir F:\FORTH_Final_Thesis\FORTH-Thesis\connectivity \
        --fixedorder 12 \
        --workers 8
"""

import argparse
from pathlib import Path
import numpy as np
import warnings
from tqdm import tqdm
from scipy import linalg
import concurrent.futures
import multiprocessing
import json
from datetime import datetime

from statsmodels.tsa.vector_ar.var_model import VAR

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# TUC channel names
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T3', 'C3', 'Cz', 'C4', 'T4',
                 'T5', 'P3', 'Pz', 'P4', 'T6', 
                 'O1', 'O2']

# ==============================================================================
# CORE CONNECTIVITY FUNCTIONS
# ==============================================================================

def compute_dtf_pdc_from_var(coefs, fs=256.0, nfft=512):
    """
    Compute DTF and PDC from VAR coefficients.
    
    Parameters:
    -----------
    coefs : np.ndarray
        VAR coefficients, shape (p, n_channels, n_channels)
    fs : float
        Sampling frequency (Hz)
    nfft : int
        FFT length
    
    Returns:
    --------
    dtf : np.ndarray (n_channels, n_channels, n_freqs)
    pdc : np.ndarray (n_channels, n_channels, n_freqs)
    freqs : np.ndarray (n_freqs,)
    """
    p, K, _ = coefs.shape
    n_freqs = nfft // 2 + 1
    freqs = np.linspace(0, fs/2, n_freqs)
    
    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    H_f = np.zeros((n_freqs, K, K), dtype=complex)
    I = np.eye(K)
    
    # Compute A(f) and H(f) for each frequency
    for f_idx, f in enumerate(freqs):
        A_sum = np.zeros((K, K), dtype=complex)
        for k in range(p):
            phase = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
            A_sum += coefs[k] * phase
        
        A_f[f_idx] = I - A_sum
        
        try:
            H_f[f_idx] = linalg.inv(A_f[f_idx])
        except linalg.LinAlgError:
            H_f[f_idx] = linalg.pinv(A_f[f_idx])
    
    # PDC (column-wise normalization of A(f))
    pdc = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Af = A_f[f_idx]
        col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))
        col_norms[col_norms == 0] = 1e-10
        pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]
    
    # DTF (row-wise normalization of H(f))
    dtf = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Hf = H_f[f_idx]
        row_norms = np.sqrt(np.sum(np.abs(Hf)**2, axis=1))
        row_norms[row_norms == 0] = 1e-10
        dtf[:, :, f_idx] = np.abs(Hf) / row_norms[:, None]
    
    return dtf, pdc, freqs


def process_single_epoch(data, fs, fixed_order, nfft):
    """
    Process one epoch with FIXED model order.
    
    Parameters:
    -----------
    data : np.ndarray
        Single epoch (n_channels, n_timepoints) - e.g., (19, 1024)
    fs : float
        Sampling frequency
    fixed_order : int
        Fixed MVAR order (from BIC analysis)
    nfft : int
        FFT length
    
    Returns:
    --------
    dict with 'dtf_bands', 'pdc_bands', 'order' or None if failed
    """
    # Check data quality
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    
    # Standardize
    data_scaled = data / data_std
    
    try:
        # Transpose: statsmodels expects (n_timepoints, n_channels)
        model = VAR(data_scaled.T)
        
        # Fit with FIXED order
        try:
            results = model.fit(maxlags=fixed_order, trend='c', verbose=False)
        except:
            return None
        
        if results.k_ar == 0:
            return None
        
        # Stability check (optional)
        try:
            if not results.is_stable():
                return None
        except:
            pass
        
        # Compute full spectrum DTF/PDC
        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(
            results.coefs, fs, nfft
        )
        
        # Define frequency bands (TUC-specific: 0.5-45 Hz)
        bands = {
            'integrated': (0.5, 45.0),
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 15.0),
            'beta': (15.0, 30.0),
            'gamma1': (30.0, 45.0)
        }
        
        # Integrate over each frequency band
        dtf_bands = {}
        pdc_bands = {}
        
        for band_name, (f_low, f_high) in bands.items():
            idx_band = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            
            if len(idx_band) == 0:
                return None
            
            # Average over frequency
            dtf_band = np.mean(dtf_spectrum[:, :, idx_band], axis=2)
            pdc_band = np.mean(pdc_spectrum[:, :, idx_band], axis=2)
            
            # SET DIAGONAL TO ZERO (inter-channel connectivity only)
            np.fill_diagonal(dtf_band, 0.0)
            np.fill_diagonal(pdc_band, 0.0)
            
            dtf_bands[band_name] = dtf_band
            pdc_bands[band_name] = pdc_band
        
        return {
            'dtf_bands': dtf_bands,
            'pdc_bands': pdc_bands,
            'order': fixed_order
        }
        
    except:
        return None


# ==============================================================================
# FILE WORKER (PARALLEL PROCESSING)
# ==============================================================================

def process_subject_file(args_bundle):
    """
    Process one subject file.
    
    Parameters:
    -----------
    args_bundle : tuple
        (epochs_file, output_dir, fs, fixed_order, nfft)
    
    Returns:
    --------
    tuple : (status, n_epochs, additional_info)
    """
    epochs_file, output_dir, fs, fixed_order, nfft = args_bundle
    
    try:
        # Setup paths
        subject_name = epochs_file.stem.replace('_epochs', '')
        out_file = output_dir / f"{subject_name}_graphs.npz"
        
        # Skip if already exists
        if out_file.exists():
            return ('skipped', 0, subject_name)
        
        # Load epochs
        epochs = np.load(epochs_file)  # (n_epochs, 19, 1024)
        
        # Load labels
        labels_file = epochs_file.parent / f"{subject_name}_labels.npy"
        if not labels_file.exists():
            return ('error', 0, f"Labels not found: {subject_name}")
        labels = np.load(labels_file)
        
        # Load time_from_onset if available
        time_file = epochs_file.parent / f"{subject_name}_time_from_onset.npy"
        if time_file.exists():
            time_from_onset = np.load(time_file)
        else:
            time_from_onset = None
        
        # Initialize storage for each band
        band_names = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']
        
        dtf_data = {band: [] for band in band_names}
        pdc_data = {band: [] for band in band_names}
        
        valid_indices = []
        orders = []
        
        # Process each epoch
        for i in range(len(epochs)):
            result = process_single_epoch(epochs[i], fs, fixed_order, nfft)
            
            if result is not None:
                # Store all bands
                for band in band_names:
                    dtf_data[band].append(result['dtf_bands'][band])
                    pdc_data[band].append(result['pdc_bands'][band])
                
                valid_indices.append(i)
                orders.append(result['order'])
        
        # Save if we have valid epochs
        if len(valid_indices) > 0:
            # Prepare save dict
            save_dict = {}
            
            # Add connectivity matrices for each band
            for band in band_names:
                save_dict[f'dtf_{band}'] = np.array(dtf_data[band])
                save_dict[f'pdc_{band}'] = np.array(pdc_data[band])
            
            # Add metadata
            save_dict['labels'] = labels[valid_indices]
            save_dict['indices'] = np.array(valid_indices)
            save_dict['orders'] = np.array(orders)
            save_dict['fixed_order'] = fixed_order
            
            # Add time_from_onset if available
            if time_from_onset is not None:
                save_dict['time_from_onset'] = time_from_onset[valid_indices]
            
            # Save
            np.savez_compressed(out_file, **save_dict)
            
            # Return success with first epoch data for plotting
            return ('success', len(valid_indices), subject_name,
                   dtf_data['integrated'][0], pdc_data['integrated'][0])
        else:
            return ('failed', 0, subject_name)
            
    except Exception as e:
        return ('error', 0, f"{epochs_file.name}: {str(e)}")


# ==============================================================================
# PLOTTING
# ==============================================================================

def save_diagnostic_plot(dtf, pdc, fixed_order, subject_name, output_dir):
    """Save diagnostic heatmap for one subject."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    vmax = max(dtf.max(), pdc.max())
    
    # DTF
    sns.heatmap(dtf, ax=axes[0], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=0, vmax=vmax,
               cbar_kws={'label': 'Connectivity Strength'})
    axes[0].set_title(f'DTF (Integrated 0.5-45 Hz) - {subject_name}\nOrder p={fixed_order}', 
                     fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Source (From)', fontsize=11)
    axes[0].set_ylabel('Sink (To)', fontsize=11)
    
    # PDC
    sns.heatmap(pdc, ax=axes[1], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=0, vmax=vmax,
               cbar_kws={'label': 'Connectivity Strength'})
    axes[1].set_title(f'PDC (Integrated 0.5-45 Hz) - {subject_name}\nOrder p={fixed_order}', 
                     fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Source (From)', fontsize=11)
    axes[1].set_ylabel('Sink (To)', fontsize=11)
    
    # Note about diagonal
    fig.text(0.5, 0.02, '✓ Diagonal set to 0 (inter-channel connectivity only)',
             ha='center', fontsize=10, style='italic', color='green')
    
    plt.tight_layout()
    
    # Save
    plot_dir = output_dir / 'diagnostic_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'{subject_name}_connectivity.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute connectivity for TUC dataset with fixed order",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--inputdir", required=True,
                       help="Input directory with epoch files")
    parser.add_argument("--outputdir", required=True,
                       help="Output directory for connectivity results")
    parser.add_argument("--fixedorder", type=int, required=True,
                       help="Fixed MVAR order (from BIC analysis)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of workers (default: all CPU cores)")
    parser.add_argument("--save_plots", type=int, default=5,
                       help="Number of diagnostic plots to save")
    
    args = parser.parse_args()
    
    input_dir = Path(args.inputdir)
    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find epoch files
    epoch_files = sorted(input_dir.glob("subject_*_epochs.npy"))
    
    print("=" * 80)
    print("TUC DATASET - CONNECTIVITY COMPUTATION")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Subjects found:   {len(epoch_files)}")
    print(f"Fixed order:      p = {args.fixedorder}")
    print(f"Frequency bands:  integrated, delta, theta, alpha, beta, gamma1")
    print(f"Diagonal:         SET TO ZERO ✓")
    
    print(f"Workers:          {args.workers or multiprocessing.cpu_count()}")
    print("=" * 80)
    
    if len(epoch_files) == 0:
        print("\n❌ No epoch files found!")
        return
    
    # Prepare tasks
    tasks = [
        (f, output_dir, 256.0, args.fixedorder, 512)
        for f in epoch_files
    ]
    
    # Run parallel processing
    stats = {'success': 0, 'skipped': 0, 'failed': 0, 'error': 0}
    total_epochs = 0
    plots_saved = 0
    
    print(f"\nProcessing...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_subject_file, task): task[0] for task in tasks}
        
        pbar = tqdm(total=len(epoch_files), desc="Computing connectivity", unit="subject")
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            status = result[0]
            
            if status == 'success':
                stats['success'] += 1
                total_epochs += result[1]
                
                # Save diagnostic plot
                if plots_saved < args.save_plots:
                    _, _, subject_name, dtf, pdc = result
                    save_diagnostic_plot(dtf, pdc, args.fixedorder, subject_name, output_dir)
                    plots_saved += 1
                
                pbar.set_postfix({
                    'epochs': f"{total_epochs:,}",
                    'success': stats['success']
                })
                
            elif status == 'skipped':
                stats['skipped'] += 1
            elif status == 'failed':
                stats['failed'] += 1
            elif status == 'error':
                stats['error'] += 1
                print(f"\n⚠️  {result[2]}")
            
            pbar.update(1)
        
        pbar.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("CONNECTIVITY COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"  Success:  {stats['success']} subjects")
    print(f"  Skipped:  {stats['skipped']} subjects (already done)")
    print(f"  Failed:   {stats['failed']} subjects")
    print(f"  Errors:   {stats['error']} subjects")
    print(f"  Total epochs processed: {total_epochs:,}")
    
    print(f"\n📁 Output:")
    print(f"  Connectivity files: {output_dir}")
    print(f"  Diagnostic plots:   {output_dir / 'diagnostic_plots'}")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Review diagnostic plots to verify connectivity")
    print(f"  2. Run channel-specific analysis:")
    print(f"     python channel_specific_connectivity_analysis.py \\")
    print(f"       --connectivity_dir \"{output_dir}\" \\")
    print(f"       --output_dir \"figures/channel_specific\" \\")
    print(f"       --band integrated --metric pdc")
    
    print("\n" + "=" * 80)
    print("✅ DONE")
    print("=" * 80)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()