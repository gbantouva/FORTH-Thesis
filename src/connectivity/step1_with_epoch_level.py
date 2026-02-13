"""
TUC Dataset - BIC Analysis (Modified for flat structure)
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
import json

warnings.filterwarnings("ignore")

MIN_ORDER = 8
MAX_ORDER = 22

def compute_bic_for_epoch(data, min_order=MIN_ORDER, max_order=MAX_ORDER):
    """Compute optimal BIC order for a single epoch."""
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    
    data_scaled = data / data_std
    
    try:
        # Transpose: statsmodels expects (time, channels)
        model = VAR(data_scaled.T)
        
        bic_values = []
        for p in range(min_order, max_order + 1):
            try:
                result = model.fit(maxlags=p, trend='c', verbose=False)
                bic = result.bic
                if not np.isnan(bic) and not np.isinf(bic):
                    bic_values.append((p, bic))
            except:
                continue
        
        if len(bic_values) == 0:
            return None
        
        best_order, _ = min(bic_values, key=lambda x: x[1])
        return best_order
        
    except:
        return None

def process_file_with_epochs(args):
    """Process one TUC file and return epoch results."""
    f, sample_size, min_order, max_order = args
    
    # Extract subject name
    subject_name = f.stem.replace('_epochs', '')
    
    # Load epochs
    try:
        epochs = np.load(f)  # (n_epochs, 19, 1024)
        n_epochs = len(epochs)
        
        # Sample if requested
        if sample_size is not None and sample_size < n_epochs:
            indices = np.random.choice(n_epochs, sample_size, replace=False)
        else:
            indices = np.arange(n_epochs)
        
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
                'optimal_order': int(order)
            })
    
    return epoch_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", required=True, help="Directory with TUC epochs")
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--minorder", type=int, default=MIN_ORDER)
    parser.add_argument("--maxorder", type=int, default=MAX_ORDER)
    
    args = parser.parse_args()
    
    input_dir = Path(args.inputdir)
    sample_size = None if args.sample_size == 0 else args.sample_size
    min_order = args.minorder
    max_order = args.maxorder
    
    print("=" * 80)
    print("TUC DATASET - BIC ANALYSIS")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Order range: {min_order}-{max_order}")
    print(f"Sample size: {sample_size if sample_size else 'ALL EPOCHS'}")
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
    tasks = [(f, sample_size, min_order, max_order) for f in epoch_files]
    
    # Process with progress bar
    all_results = []
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(process_file_with_epochs, task): task[0] for task in tasks}
        
        pbar = tqdm(total=len(epoch_files), desc="Processing subjects")
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
                pbar.set_postfix({'epochs': f"{len(all_results):,}"})
            except Exception as e:
                print(f"\nError: {e}")
            
            pbar.update(1)
        
        pbar.close()
    
    if len(all_results) == 0:
        print("\n❌ No valid results!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Statistics
    mean_order = df['optimal_order'].mean()
    median_order = df['optimal_order'].median()
    mode_order = df['optimal_order'].mode().iloc[0]
    std_order = df['optimal_order'].std()
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total epochs analyzed: {len(df):,}")
    print(f"\nGlobal statistics:")
    print(f"  Mean optimal order:   {mean_order:.2f}")
    print(f"  Median optimal order: {median_order:.0f}")
    print(f"  Mode optimal order:   {mode_order}")
    print(f"  Std optimal order:    {std_order:.2f}")
    
    # Use MODE as recommended order (most frequent)
    recommended_order = mode_order
    print(f"\n✅ RECOMMENDED ORDER: p = {recommended_order} (mode)")
    print(f"   Selected by {100 * (df['optimal_order'] == recommended_order).sum() / len(df):.1f}% of epochs")
    
    # Save
    df.to_csv('tuc_epoch_level_orders.csv', index=False)
    print(f"\n✅ Saved: tuc_epoch_level_orders.csv")
    
    # Subject-level summary
    subject_summary = df.groupby('subject')['optimal_order'].agg(['mean', 'median', 'std', 'count'])
    subject_summary.to_csv('tuc_subject_level_orders.csv')
    print(f"✅ Saved: tuc_subject_level_orders.csv")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========================================================================
    # LEFT PANEL: Histogram with MODE and MEDIAN lines
    # ========================================================================
    axes[0].hist(df['optimal_order'], bins=range(min_order, max_order + 2),
                 edgecolor='black', alpha=0.7, color='steelblue', align='left')
    
    # Add MODE line (solid red, thicker)
    axes[0].axvline(mode_order, color='red', linestyle='-', linewidth=2.5,
                    label=f'Mode: p={mode_order} (53.2% of epochs)', zorder=10)
    
    # Add MEDIAN line (dashed orange) - only if different from mode
    if median_order != mode_order:
        axes[0].axvline(median_order, color='orange', linestyle='--', linewidth=2,
                        label=f'Median: p={median_order:.0f}', zorder=9)
    
    axes[0].set_xlabel('Optimal Order', fontsize=12)
    axes[0].set_ylabel('Count (epochs)', fontsize=12)
    axes[0].set_title(f'TUC Dataset - BIC Distribution (n={len(df):,})', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(min_order - 0.5, max_order + 0.5)
    
    # Add text box with statistics
    stats_text = f'Mean: {mean_order:.2f}\nMedian: {median_order:.0f}\nMode: {mode_order}\nStd: {std_order:.2f}'
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================================================
    # RIGHT PANEL: Per-subject boxplot with MODE line
    # ========================================================================
    subjects = sorted(df['subject'].unique())
    data_by_subject = [df[df['subject'] == s]['optimal_order'].values for s in subjects]
    
    bp = axes[1].boxplot(data_by_subject, labels=subjects, patch_artist=True)
    
    # Color boxplots
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Add MODE line (solid red)
    axes[1].axhline(mode_order, color='red', linestyle='-', linewidth=2.5,
                    label=f'Mode: p={mode_order}', zorder=10)
    
    # Add MEDIAN line (dashed orange) - only if different
    if median_order != mode_order:
        axes[1].axhline(median_order, color='orange', linestyle='--', linewidth=2,
                        label=f'Median: p={median_order:.0f}', zorder=9)
    
    axes[1].set_xlabel('Subject', fontsize=12)
    axes[1].set_ylabel('Optimal Order', fontsize=12)
    axes[1].set_title('Order Distribution per Subject', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=90, labelsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].set_ylim(min_order - 1, max_order + 1)
    
    plt.tight_layout()
    plt.savefig('tuc_bic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: tuc_bic_analysis.png")
    
    # ========================================================================
    # Print detailed order distribution
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
            marker = marker + " (MEDIAN)" if order == median_order else marker
            print(f"  Order {order:2d}: {count:4d} epochs ({pct:5.2f}%){marker}")
    
    print("\n" + "=" * 80)
    print("NEXT STEP:")
    print("=" * 80)
    print(f"python step2_compute_connectivity.py \\")
    print(f"  --inputdir \"{input_dir}\" \\")
    print(f"  --outputdir \"TUC_connectivity_p{recommended_order}\" \\")
    print(f"  --fixedorder {recommended_order}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
