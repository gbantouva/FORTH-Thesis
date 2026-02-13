"""
TUC Dataset - BIC Analysis for Connectivity
============================================
Determines optimal MVAR model order using Bayesian Information Criterion (BIC).

Features:
- Parallel processing for speed
- Optional training mask filtering
- Pre-ictal vs Ictal analysis
- Comprehensive visualization

Usage:
    # Analyze all epochs
    python step1_bic_analysis.py --inputdir preprocessed_epochs
    
    # Sample 100 epochs per subject for faster analysis
    python step1_bic_analysis.py --inputdir preprocessed_epochs --sample_size 100
    
    # Only analyze training epochs (pre-ictal + ictal, excluding post-ictal)
    python step1_bic_analysis.py --inputdir preprocessed_epochs --use_training_mask
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing

warnings.filterwarnings("ignore")

# Default BIC order range
MIN_ORDER = 8
MAX_ORDER = 22

# ============================================================================
# BIC COMPUTATION
# ============================================================================

def compute_bic_for_epoch(data, min_order=MIN_ORDER, max_order=MAX_ORDER):
    """
    Compute optimal BIC order for a single epoch.
    
    Parameters:
    -----------
    data : np.ndarray
        Shape (n_channels, n_timepoints) - e.g., (19, 1024)
    
    Returns:
    --------
    int or None : Optimal order, or None if computation failed
    """
    # Check data quality
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    
    # Standardize
    data_scaled = data / data_std
    
    try:
        # Transpose: statsmodels VAR expects (n_timepoints, n_channels)
        model = VAR(data_scaled.T)
        
        bic_values = []
        for p in range(min_order, max_order + 1):
            try:
                result = model.fit(maxlags=p, trend='c', verbose=False)
                bic = result.bic
                
                # Only keep valid BIC values
                if not np.isnan(bic) and not np.isinf(bic):
                    bic_values.append((p, bic))
            except:
                continue
        
        if len(bic_values) == 0:
            return None
        
        # Return order with minimum BIC
        best_order, _ = min(bic_values, key=lambda x: x[1])
        return best_order
        
    except:
        return None


# ============================================================================
# FILE PROCESSING
# ============================================================================

def process_file_with_epochs(args):
    """
    Process one subject file and return epoch-level BIC results.
    
    Parameters:
    -----------
    args : tuple
        (filepath, sample_size, use_training_mask, min_order, max_order)
    
    Returns:
    --------
    list : Results with subject, epoch_idx, optimal_order, label
    """
    f, sample_size, use_training_mask, min_order, max_order = args
    
    # Extract subject name
    subject_name = f.stem.replace('_epochs', '')
    
    # Load epochs and labels
    try:
        epochs = np.load(f)  # (n_epochs, 19, 1024)
        n_epochs = len(epochs)
        
        # Load labels (0=pre-ictal, 1=ictal)
        labels_file = f.parent / f"{subject_name}_labels.npy"
        labels = np.load(labels_file) if labels_file.exists() else np.zeros(n_epochs)
        
        # Load training mask if requested
        if use_training_mask:
            mask_file = f.parent / f"{subject_name}_training_mask.npy"
            if mask_file.exists():
                training_mask = np.load(mask_file)
            else:
                print(f"⚠️  Warning: training_mask not found for {subject_name}, using all epochs")
                training_mask = np.ones(n_epochs, dtype=bool)
        else:
            training_mask = np.ones(n_epochs, dtype=bool)
        
        # Get valid indices (within training mask)
        valid_indices = np.where(training_mask)[0]
        
        # Sample if requested
        if sample_size is not None and sample_size < len(valid_indices):
            indices = np.random.choice(valid_indices, sample_size, replace=False)
        else:
            indices = valid_indices
        
    except Exception as e:
        print(f"Error loading {f.name}: {e}")
        return []
    
    # Process each epoch
    epoch_results = []
    
    for idx in indices:
        order = compute_bic_for_epoch(epochs[idx], min_order, max_order)
        
        if order is not None:
            epoch_results.append({
                'subject': subject_name,
                'epoch_idx': int(idx),
                'optimal_order': int(order),
                'label': int(labels[idx])  # 0=pre-ictal, 1=ictal
            })
    
    return epoch_results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BIC analysis for optimal MVAR order selection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--inputdir", required=True, help="Directory with epoch files")
    parser.add_argument("--sample_size", type=int, default=0, 
                       help="Epochs per subject to analyze (0=all)")
    parser.add_argument("--use_training_mask", action="store_true",
                       help="Only analyze epochs in training set (exclude post-ictal)")
    parser.add_argument("--minorder", type=int, default=MIN_ORDER,
                       help=f"Minimum MVAR order (default: {MIN_ORDER})")
    parser.add_argument("--maxorder", type=int, default=MAX_ORDER,
                       help=f"Maximum MVAR order (default: {MAX_ORDER})")
    parser.add_argument("--output_prefix", type=str, default="tuc",
                       help="Prefix for output files (default: tuc)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.inputdir)
    sample_size = None if args.sample_size == 0 else args.sample_size
    min_order = args.minorder
    max_order = args.maxorder
    
    print("=" * 80)
    print("TUC DATASET - BIC ANALYSIS")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Order range: {min_order}-{max_order}")
    print(f"Sample size: {sample_size if sample_size else 'ALL EPOCHS'}")
    print(f"Training mask: {'ENABLED' if args.use_training_mask else 'DISABLED'}")
    print("=" * 80)
    
    # Find epoch files
    epoch_files = sorted(input_dir.glob("subject_*_epochs.npy"))
    print(f"\nFound {len(epoch_files)} subjects")
    
    if len(epoch_files) == 0:
        print("❌ No epoch files found!")
        return
    
    # Count total epochs
    total_epochs = sum(np.load(f).shape[0] for f in epoch_files)
    print(f"Total epochs: {total_epochs:,}")
    
    if sample_size:
        estimated = len(epoch_files) * sample_size
        print(f"Will analyze ~{estimated:,} epochs (sample_size={sample_size})")
    
    # Prepare tasks
    tasks = [(f, sample_size, args.use_training_mask, min_order, max_order) 
             for f in epoch_files]
    
    # Process with parallel execution
    print(f"\nProcessing with {multiprocessing.cpu_count()} workers...")
    all_results = []
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(process_file_with_epochs, task): task[0] 
                  for task in tasks}
        
        pbar = tqdm(total=len(epoch_files), desc="Processing subjects")
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
                pbar.set_postfix({'total_epochs': f"{len(all_results):,}"})
            except Exception as e:
                print(f"\nError: {e}")
            
            pbar.update(1)
        
        pbar.close()
    
    if len(all_results) == 0:
        print("\n❌ No valid results!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # ========================================================================
    # GLOBAL STATISTICS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("GLOBAL STATISTICS")
    print("=" * 80)
    
    mean_order = df['optimal_order'].mean()
    median_order = df['optimal_order'].median()
    mode_order = df['optimal_order'].mode().iloc[0]
    std_order = df['optimal_order'].std()
    
    print(f"\nTotal epochs analyzed: {len(df):,}")
    print(f"  Pre-ictal: {(df['label'] == 0).sum():,}")
    print(f"  Ictal:     {(df['label'] == 1).sum():,}")
    
    print(f"\nGlobal order statistics:")
    print(f"  Mean:   {mean_order:.2f}")
    print(f"  Median: {median_order:.0f}")
    print(f"  Mode:   {mode_order}")
    print(f"  Std:    {std_order:.2f}")
    
    # ========================================================================
    # PRE-ICTAL VS ICTAL COMPARISON
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PRE-ICTAL VS ICTAL COMPARISON")
    print("=" * 80)
    
    pre_ictal_orders = df[df['label'] == 0]['optimal_order']
    ictal_orders = df[df['label'] == 1]['optimal_order']
    
    if len(ictal_orders) > 0:
        print(f"\nPre-ictal epochs (n={len(pre_ictal_orders):,}):")
        print(f"  Mean:   {pre_ictal_orders.mean():.2f}")
        print(f"  Median: {pre_ictal_orders.median():.0f}")
        print(f"  Mode:   {pre_ictal_orders.mode().iloc[0] if len(pre_ictal_orders.mode()) > 0 else 'N/A'}")
        
        print(f"\nIctal epochs (n={len(ictal_orders):,}):")
        print(f"  Mean:   {ictal_orders.mean():.2f}")
        print(f"  Median: {ictal_orders.median():.0f}")
        print(f"  Mode:   {ictal_orders.mode().iloc[0] if len(ictal_orders.mode()) > 0 else 'N/A'}")
        
        # Statistical test
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(pre_ictal_orders, ictal_orders)
        print(f"\nMann-Whitney U test:")
        print(f"  p-value: {pval:.4f}")
        print(f"  Significant difference: {'YES' if pval < 0.05 else 'NO'}")
    
    # ========================================================================
    # RECOMMENDED ORDER
    # ========================================================================
    
    recommended_order = mode_order
    mode_pct = 100 * (df['optimal_order'] == recommended_order).sum() / len(df)
    
    print("\n" + "=" * 80)
    print("RECOMMENDED ORDER")
    print("=" * 80)
    print(f"✅ p = {recommended_order} (mode)")
    print(f"   Selected by {mode_pct:.1f}% of epochs")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Epoch-level results
    df.to_csv(f'{args.output_prefix}_epoch_level_orders.csv', index=False)
    print(f"  ✅ {args.output_prefix}_epoch_level_orders.csv")
    
    # Subject-level summary
    subject_summary = df.groupby('subject')['optimal_order'].agg([
        'mean', 'median', 'std', 'count'
    ])
    subject_summary.to_csv(f'{args.output_prefix}_subject_level_orders.csv')
    print(f"  ✅ {args.output_prefix}_subject_level_orders.csv")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Create 2x2 figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Global distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['optimal_order'], bins=range(min_order, max_order + 2),
             edgecolor='black', alpha=0.7, color='steelblue', align='left')
    ax1.axvline(mode_order, color='red', linestyle='-', linewidth=2.5,
                label=f'Mode: p={mode_order} ({mode_pct:.1f}%)', zorder=10)
    if median_order != mode_order:
        ax1.axvline(median_order, color='orange', linestyle='--', linewidth=2,
                    label=f'Median: p={median_order:.0f}', zorder=9)
    ax1.set_xlabel('Optimal Order', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count (epochs)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Global BIC Distribution (n={len(df):,})', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Pre-ictal vs Ictal
    ax2 = fig.add_subplot(gs[0, 1])
    if len(ictal_orders) > 0:
        ax2.hist(pre_ictal_orders, bins=range(min_order, max_order + 2),
                alpha=0.6, label=f'Pre-ictal (n={len(pre_ictal_orders):,})', 
                color='blue', edgecolor='black', align='left')
        ax2.hist(ictal_orders, bins=range(min_order, max_order + 2),
                alpha=0.6, label=f'Ictal (n={len(ictal_orders):,})', 
                color='red', edgecolor='black', align='left')
        ax2.axvline(mode_order, color='green', linestyle='-', linewidth=2.5,
                   label=f'Global mode: p={mode_order}', zorder=10)
        ax2.set_xlabel('Optimal Order', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count (epochs)', fontsize=12, fontweight='bold')
        ax2.set_title('Pre-ictal vs Ictal Comparison', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No ictal epochs analyzed', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
    
    # Panel 3: Per-subject boxplot
    ax3 = fig.add_subplot(gs[1, :])
    subjects = sorted(df['subject'].unique())
    data_by_subject = [df[df['subject'] == s]['optimal_order'].values for s in subjects]
    
    bp = ax3.boxplot(data_by_subject, labels=subjects, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax3.axhline(mode_order, color='red', linestyle='-', linewidth=2.5,
                label=f'Mode: p={mode_order}', zorder=10)
    ax3.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Optimal Order', fontsize=12, fontweight='bold')
    ax3.set_title('Order Distribution per Subject', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=90, labelsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=10)
    ax3.set_ylim(min_order - 1, max_order + 1)
    
    plt.suptitle('TUC Dataset - BIC Analysis Results', fontsize=16, fontweight='bold')
    plt.savefig(f'{args.output_prefix}_bic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ {args.output_prefix}_bic_analysis.png")
    
    # ========================================================================
    # ORDER DISTRIBUTION TABLE
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("ORDER DISTRIBUTION")
    print("=" * 80)
    
    order_counts = df['optimal_order'].value_counts().sort_index()
    for order in range(min_order, max_order + 1):
        if order in order_counts.index:
            count = order_counts[order]
            pct = 100 * count / len(df)
            marker = " ← MODE" if order == mode_order else ""
            if order == median_order and order != mode_order:
                marker += " (MEDIAN)"
            print(f"  p={order:2d}: {count:5d} epochs ({pct:5.2f}%){marker}")
    
    # ========================================================================
    # NEXT STEPS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"\n✅ Recommended order: p = {recommended_order}")
    print(f"\nRun connectivity analysis:")
    print(f"  python step2_compute_connectivity.py \\")
    print(f"    --inputdir \"{input_dir}\" \\")
    print(f"    --outputdir \"connectivity_results\" \\")
    print(f"    --fixedorder {recommended_order}")
    
    print("\n" + "=" * 80)
    print("✅ BIC ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()