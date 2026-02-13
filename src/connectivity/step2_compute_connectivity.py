"""
STEP 2: COMPUTE CONNECTIVITY WITH FIXED ORDER (PARALLEL + CHECKPOINTING)
========================================================================
Computes DTF and PDC connectivity matrices using a FIXED model order from Step 1.

FEATURES:
  - Uses fixed MVAR order (no BIC selection per epoch)
  - Multi-band support (7 frequency bands)
  - Diagonal set to ZERO (inter-channel connectivity only)
  - Parallel processing (all CPU cores)
  - Checkpointing (stop/resume anytime)

Usage:
    # After Step 1 recommended p=15
    python step2_compute_connectivity.py \
        --input_dir F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced \
        --output_dirac F:\October-Thesis\thesis-epilepsy-gnn\connectivity_fixed_p15 \
        --fixed_order 15 \
        --workers 8
    PS F:\FORTH-DATA> & C:/Users/georg/AppData/Local/Programs/Python/Python311/python.exe f:/FORTH-DATA/Thesis/src/final_connectivity/step2_compute_connectivity.py --input_dir F:\FORTH-DATA\Thesis\FORTH_preprocessed_epochs_label --output_dir F:\FORTH-DATA\Thesis\connectivity --fixed_order 12 --save_plots 20                                                                                            
                                                                  
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

from statsmodels.tsa.vector_ar.var_model import VAR

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def compute_dtf_pdc_from_var(coefs, fs=256.0, nfft=512):
    """
    Compute DTF and PDC from VAR coefficients.
    
    Returns DTF and PDC with diagonal INCLUDED (will be zeroed later per band).
    """
    p, K, _ = coefs.shape
    n_freqs = nfft // 2 + 1
    freqs = np.linspace(0, fs/2, n_freqs)
    
    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    H_f = np.zeros((n_freqs, K, K), dtype=complex)
    I = np.eye(K)
    
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
    
    # PDC (column-wise normalization)
    pdc = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Af = A_f[f_idx]
        col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))
        col_norms[col_norms == 0] = 1e-10
        pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]
    
    # DTF (row-wise normalization)
    dtf = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Hf = H_f[f_idx]
        row_norms = np.sqrt(np.sum(np.abs(Hf)**2, axis=1))
        row_norms[row_norms == 0] = 1e-10
        dtf[:, :, f_idx] = np.abs(Hf) / row_norms[:, None]
    
    return dtf, pdc, freqs


def process_single_epoch(data, fs, fixed_order, nfft):
    """
    Process one epoch with FIXED model order (NO BIC SELECTION).
    
    Parameters
    ----------
    data : ndarray
        Single epoch (n_channels, n_times)
    fs : float
        Sampling frequency
    fixed_order : int
        Fixed MVAR order (from Step 1)
    nfft : int
        FFT length
    
    Returns
    -------
    dict with 'dtf_bands' and 'pdc_bands' (diagonal already set to 0)
    """
    # Scale data
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    data_scaled = data / data_std
    
    try:
        model = VAR(data_scaled.T)
        
        # ✅ FIT WITH FIXED ORDER (NO BIC LOOP!)
        try:
            results = model.fit(maxlags=fixed_order, trend='c', verbose=False)
        except:
            return None
        
        if results.k_ar == 0:
            return None
        
        # Stability check
        try:
            if not results.is_stable():
                from statsmodels.tsa.vector_ar.util import comp_matrix
                max_eig = np.max(np.abs(np.linalg.eigvals(comp_matrix(results.coefs))))
                if max_eig > 1.0:
                    return None
        except:
            pass
        
        # Compute full spectrum
        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(
            results.coefs, fs, nfft
        )
        
        # Define frequency bands
        bands = {
            'integrated': (0.5, 45.0),
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 15.0),
            'beta': (15.0, 30.0),
            'gamma1': (30.0, 45.0)
        }
        
        # Integrate each band
        dtf_bands = {}
        pdc_bands = {}
        
        for band_name, (f_low, f_high) in bands.items():
            idx_band = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            
            if len(idx_band) == 0:
                return None
            
            # Average over frequency
            dtf_band = np.mean(dtf_spectrum[:, :, idx_band], axis=2)
            pdc_band = np.mean(pdc_spectrum[:, :, idx_band], axis=2)
            
            # ✨ SET DIAGONAL TO ZERO ✨
            np.fill_diagonal(dtf_band, 0.0)
            np.fill_diagonal(pdc_band, 0.0)
            
            dtf_bands[band_name] = dtf_band
            pdc_bands[band_name] = pdc_band
        
        return {
            'dtf_bands': dtf_bands,
            'pdc_bands': pdc_bands,
            'order': fixed_order  # ← Always the same!
        }
        
    except:
        return None


# ==============================================================================
# FILE WORKER (PARALLEL PROCESSING)
# ==============================================================================

def process_patient_file(args_bundle):
    """
    Process one patient file.
    
    Parameters
    ----------
    args_bundle : tuple
        (file_path, input_dir, output_dir, fs, fixed_order, nfft)
    """
    f, input_dir, output_dir, fs, fixed_order, nfft = args_bundle
    
    try:
        # Setup paths
        rel_path = f.relative_to(input_dir)
        out_file = output_dir / rel_path.parent / f"{f.stem.replace('_epochs', '_graphs')}.npz"
        
        # Skip if exists
        if out_file.exists():
            return ('skipped', 0, None)
        
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load data
        epochs = np.load(f)
        labels_file = f.parent / f"{f.stem.replace('_epochs', '_labels')}.npy"
        
        if not labels_file.exists():
            return ('error', 0, f"Labels file not found: {labels_file}")
        
        labels = np.load(labels_file)
        
        # Initialize lists for each band
        dtf_integrated, pdc_integrated = [], []
        dtf_delta, pdc_delta = [], []
        dtf_theta, pdc_theta = [], []
        dtf_alpha, pdc_alpha = [], []
        dtf_beta, pdc_beta = [], []
        dtf_gamma1, pdc_gamma1 = [], []
        
        valid_idx, orders = [], []
        
        # Process epochs
        for i in range(len(epochs)):
            res = process_single_epoch(epochs[i], fs, fixed_order, nfft)
            
            if res:
                # Extract all bands
                dtf_integrated.append(res['dtf_bands']['integrated'])
                pdc_integrated.append(res['pdc_bands']['integrated'])
                
                dtf_delta.append(res['dtf_bands']['delta'])
                pdc_delta.append(res['pdc_bands']['delta'])
                
                dtf_theta.append(res['dtf_bands']['theta'])
                pdc_theta.append(res['pdc_bands']['theta'])
                
                dtf_alpha.append(res['dtf_bands']['alpha'])
                pdc_alpha.append(res['pdc_bands']['alpha'])
                
                dtf_beta.append(res['dtf_bands']['beta'])
                pdc_beta.append(res['pdc_bands']['beta'])
                
                dtf_gamma1.append(res['dtf_bands']['gamma1'])
                pdc_gamma1.append(res['pdc_bands']['gamma1'])
                
                
                valid_idx.append(i)
                orders.append(res['order'])
        
        # Save if valid
        if dtf_integrated:
            np.savez_compressed(out_file,
                # Integrated
                dtf_integrated=np.array(dtf_integrated),
                pdc_integrated=np.array(pdc_integrated),
                
                # Per-band
                dtf_delta=np.array(dtf_delta),
                pdc_delta=np.array(pdc_delta),
                
                dtf_theta=np.array(dtf_theta),
                pdc_theta=np.array(pdc_theta),
                
                dtf_alpha=np.array(dtf_alpha),
                pdc_alpha=np.array(pdc_alpha),
                
                dtf_beta=np.array(dtf_beta),
                pdc_beta=np.array(pdc_beta),
                
                dtf_gamma1=np.array(dtf_gamma1),
                pdc_gamma1=np.array(pdc_gamma1),
                
                
                # Metadata
                labels=labels[valid_idx],
                indices=np.array(valid_idx),
                orders=np.array(orders),  # All same value (fixed_order)
                fixed_order=fixed_order    # Store the fixed order used
            )
            
            return ('success', len(dtf_integrated), f,
                   dtf_integrated[0], pdc_integrated[0], fixed_order)
        else:
            return ('failed', 0, None)
            
    except Exception as e:
        return ('error', 0, str(e))


# ==============================================================================
# PLOTTING
# ==============================================================================

def save_diagnostic_plot(dtf, pdc, order, patient_id, output_dir):
    """Save diagnostic heatmap."""
    CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T3', 'C3', 'Cz', 'C4', 'T4',
                 'T5', 'P3', 'Pz', 'P4', 'T6',
                 'O1', 'O2']
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    vmax = max(dtf.max(), pdc.max())
    
    sns.heatmap(dtf, ax=ax[0], cmap='viridis', square=True, vmin=0, vmax=vmax,
                xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES)
    ax[0].set_title(f'DTF (Integrated) - {patient_id} (p={order})')
    
    sns.heatmap(pdc, ax=ax[1], cmap='viridis', square=True, vmin=0, vmax=vmax,
                xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES)
    ax[1].set_title(f'PDC (Integrated) - {patient_id} (p={order})')
    
    # Add note about diagonal
    fig.text(0.5, 0.02, 'Note: Diagonal set to 0 (inter-channel connectivity only)',
             ha='center', fontsize=10, style='italic')
    
    plot_dir = output_dir / 'diagnostic_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'{patient_id}_connectivity.png', dpi=100, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN WITH CHECKPOINTING
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Compute connectivity with fixed order (PARALLEL + CHECKPOINTING)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After Step 1 recommended p=15
  python step2_compute_connectivity.py \
      --input_dir data_pp_balanced \
      --output_dir connectivity_fixed_p15 \
      --fixed_order 15 \
      --workers 8
  
  # Resume interrupted run (same command)
  python step2_compute_connectivity.py \
      --input_dir data_pp_balanced \
      --output_dir connectivity_fixed_p15 \
      --fixed_order 15 \
      --workers 8
        """
    )
    
    parser.add_argument("--input_dir", required=True,
                       help="Input directory with epoch files")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for connectivity results")
    parser.add_argument("--fixed_order", type=int, required=True,
                       help="Fixed MVAR order (from Step 1)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of workers (default: all CPU cores)")
    parser.add_argument("--save_plots", type=int, default=5,
                       help="Number of diagnostic plots to save")
    parser.add_argument("--checkpoint_dir", type=str, default='connectivity_checkpoints',
                       help="Checkpoint directory")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_file = checkpoint_dir / 'processed_files.json'
    
    epoch_files = list(input_dir.rglob("*_epochs.npy"))
    
    print(f"\n{'='*80}")
    print(f"STEP 2: CONNECTIVITY COMPUTATION (FIXED ORDER + CHECKPOINTING)")
    print(f"{'='*80}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(epoch_files)}")
    print(f"FIXED ORDER:      p = {args.fixed_order}")
    print(f"Diagonal:         SET TO ZERO")
    print(f"Bands:            integrated, delta, theta, alpha, beta, gamma1")
    print(f"Workers:          {args.workers or multiprocessing.cpu_count()}")
    print(f"\n💾 CHECKPOINTING ENABLED:")
    print(f"    Press Ctrl+C to stop safely at any time")
    print(f"    Re-run same command to resume from checkpoint")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    processed_files = set()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_files = set(checkpoint_data.get('processed_files', []))
            print(f"✅ RESUMING FROM CHECKPOINT:")
            print(f"    Already processed: {len(processed_files)} files")
            print(f"    Remaining: {len(epoch_files) - len(processed_files)} files\n")
        except:
            processed_files = set()
    
    # Filter files
    files_to_process = [f for f in epoch_files if str(f) not in processed_files]
    files_already_done = len(epoch_files) - len(files_to_process)
    
    if len(files_to_process) == 0:
        print("✅ All files already processed!\n")
        return
    
    # Prepare tasks
    tasks = [
        (f, input_dir, output_dir, 256.0, args.fixed_order, 512)
        for f in files_to_process
    ]
    
    plots_saved = 0
    stats = {'success': 0, 'skipped': 0, 'failed': 0, 'error': 0}
    total_epochs = 0
    
    # Run parallel with checkpointing
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_patient_file, task): task[0] for task in tasks}
            
            pbar = tqdm(
                total=len(files_to_process),
                desc=f"Processing (already done: {files_already_done})",
                unit="file"
            )
            
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                result = future.result()
                status = result[0]
                
                if status == 'success':
                    stats['success'] += 1
                    total_epochs += result[1]
                    processed_files.add(str(file_path))
                    
                    # Save plot
                    if plots_saved < args.save_plots:
                        _, _, f_path, dtf, pdc, order = result
                        save_diagnostic_plot(dtf, pdc, order,
                                           f_path.stem.replace('_epochs', ''),
                                           output_dir)
                        plots_saved += 1
                    
                    # Save checkpoint every 10 files
                    if len(processed_files) % 10 == 0:
                        with open(checkpoint_file, 'w') as f:
                            json.dump({
                                'processed_files': list(processed_files),
                                'total_epochs': total_epochs,
                                'timestamp': pd.Timestamp.now().isoformat() if 'pd' in dir() else 'unknown'
                            }, f)
                    
                    pbar.set_postfix({
                        'epochs': f"{total_epochs:,}",
                        'success': stats['success'],
                        'total_done': files_already_done + stats['success']
                    })
                    
                elif status == 'skipped':
                    stats['skipped'] += 1
                elif status == 'failed':
                    stats['failed'] += 1
                elif status == 'error':
                    stats['error'] += 1
                    if result[2]:
                        print(f"\n⚠️  {result[2]}")
                
                pbar.update(1)
            
            pbar.close()
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print(f"⚠️  INTERRUPTED BY USER (Ctrl+C)")
        print(f"{'='*80}")
        print(f"✅ Progress saved to checkpoint!")
        print(f"📊 Statistics:")
        print(f"    Files processed: {stats['success']}")
        print(f"    Epochs processed: {total_epochs:,}")
        print(f"    Checkpoint location: {checkpoint_dir}")
        print(f"\n🔄 To resume from this point:")
        print(f"    Run the exact same command again")
        print(f"{'='*80}\n")
        return
    
    # Final save and cleanup
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'processed_files': list(processed_files),
            'total_epochs': total_epochs,
            'completed': True
        }, f)
    
    print(f"\n{'='*80}")
    print(f"DONE!")
    print(f"{'='*80}")
    print(f"Success:  {stats['success']}")
    print(f"Skipped:  {stats['skipped']}")
    print(f"Failed:   {stats['failed']}")
    print(f"Errors:   {stats['error']}")
    print(f"Total epochs processed: {total_epochs:,}")
    print(f"{'='*80}\n")
    
    # Cleanup checkpoint
    try:
        checkpoint_file.unlink()
        checkpoint_dir.rmdir()
        print("✅ Checkpoints removed (analysis complete)\n")
    except:
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()