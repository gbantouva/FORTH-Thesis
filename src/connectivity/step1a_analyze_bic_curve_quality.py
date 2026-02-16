"""
TUC Dataset - BIC Curve Quality Analysis
=========================================
Analyzes BIC curve quality for TUC focal seizures dataset.

Usage:
    python tuc_analyze_bic_quality.py --inputdir "F:\FORTH-DATA\Thesis\FORTH_preprocessed_epochs" --n_samples 200
    PS F:\Forth_Final_Thesis\FORTH-Thesis\figures> & C:/Users/georg/AppData/Local/Programs/Python/Python311/python.exe f:/FORTH_Final_Thesis/FORTH-Thesis/src/connectivity/step1a_analyze_bic_curve_quality.py --inputdir F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs --n_samples 200 --output_dir F:\FORTH_Final_Thesis\FORTH-Thesis\figures\bic_curve_quality
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# TUC-specific constants
MIN_ORDER = 8
MAX_ORDER = 22
N_CHANNELS = 19  # TUC standard montage
SAMPLES_PER_EPOCH = 1024  # 4 seconds at 256 Hz

def analyze_bic_curve_quality(orders, bic_values):
    """
    Classify BIC curve quality.
    
    Categories:
    - clear: Smooth U-shape, obvious minimum
    - acceptable: U-shape visible despite some noise
    - noisy: General trend visible but irregular
    - edge: Optimal at boundary (p=8 or p=22)
    - chaotic: No interpretable pattern
    - flat: No variation
    """
    if len(orders) < 5:
        return 'insufficient_data', {}
    
    # Normalize BIC values
    bic_range = np.max(bic_values) - np.min(bic_values)
    
    if bic_range < 1e-6:
        return 'flat', {'range': bic_range}
    
    bic_norm = (bic_values - np.min(bic_values)) / bic_range
    
    # Find optimal order
    optimal_idx = np.argmin(bic_values)
    optimal_order = orders[optimal_idx]
    
    # Edge detection
    is_edge = (optimal_order == MIN_ORDER) or (optimal_order == MAX_ORDER)
    
    # Smoothness (2nd derivative variance)
    if len(bic_norm) >= 3:
        second_deriv = np.diff(np.diff(bic_norm))
        smoothness = np.std(second_deriv)
    else:
        smoothness = np.inf
    
    # U-shape detection
    left_slope = 0
    right_slope = 0
    
    if optimal_idx > 0:
        left_slope = np.mean(np.diff(bic_norm[:optimal_idx+1]))
    
    if optimal_idx < len(bic_norm) - 1:
        right_slope = np.mean(np.diff(bic_norm[optimal_idx:]))
    
    has_left_decrease = left_slope < 0
    has_right_increase = right_slope > 0
    u_shape_present = has_left_decrease and has_right_increase
    strong_u_shape = (left_slope < -0.005) and (right_slope > 0.005)
    
    # Depth of minimum
    edge_avg = (bic_norm[0] + bic_norm[-1]) / 2
    depth = edge_avg - bic_norm[optimal_idx]
    
    # Relative range
    relative_range = bic_range / np.abs(np.mean(bic_values))
    
    metrics = {
        'optimal_order': optimal_order,
        'is_edge': is_edge,
        'smoothness': smoothness,
        'left_slope': left_slope,
        'right_slope': right_slope,
        'u_shape_present': u_shape_present,
        'strong_u_shape': strong_u_shape,
        'depth': depth,
        'relative_range': relative_range
    }
    
    # Classification
    if is_edge:
        category = 'edge' if relative_range >= 0.01 else 'flat'
    elif smoothness > 1.0:
        category = 'chaotic'
    elif strong_u_shape and smoothness < 0.4 and depth > 0.2:
        category = 'clear'
    elif u_shape_present and smoothness < 0.8:
        category = 'acceptable'
    else:
        category = 'noisy'
    
    return category, metrics

def compute_bic_curve(epoch_data, min_order=MIN_ORDER, max_order=MAX_ORDER):
    """
    Compute BIC curve for one epoch.
    
    Parameters:
    -----------
    epoch_data : np.ndarray
        Shape (19, 1024) - channels × time
    
    Returns:
    --------
    orders, bic_values, optimal_idx or (None, None, None)
    """
    try:
        # Check for bad data
        if np.any(np.isnan(epoch_data)) or np.any(np.isinf(epoch_data)):
            return None, None, None
        
        # Standardize
        data_std = np.std(epoch_data)
        if data_std < 1e-10:
            return None, None, None
        
        data_scaled = epoch_data / data_std
        
        # Transpose for statsmodels (time, channels)
        model = VAR(data_scaled.T)
        
        orders = []
        bic_values = []
        
        for p in range(min_order, max_order + 1):
            try:
                result = model.fit(maxlags=p, trend='c', verbose=False)
                bic = result.bic
                
                if not np.isnan(bic) and not np.isinf(bic) and np.abs(bic) < 1e10:
                    orders.append(p)
                    bic_values.append(bic)
            except:
                continue
        
        if len(orders) < 5:
            return None, None, None
        
        return orders, bic_values, np.argmin(bic_values)
        
    except:
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="TUC BIC Quality Analysis")
    parser.add_argument("--inputdir", required=True, help="TUC epochs directory")
    parser.add_argument("--n_samples", type=int, default=200, help="Epochs to sample")
    parser.add_argument("--output_dir", default="tuc_bic_quality", help="Output directory")
    
    args = parser.parse_args()
    
    input_dir = Path(args.inputdir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("TUC DATASET - BIC CURVE QUALITY ANALYSIS")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Samples: {args.n_samples}")
    print(f"Order range: {MIN_ORDER}-{MAX_ORDER}")
    print("=" * 80)
    
    # Find epoch files
    epoch_files = sorted(input_dir.glob("subject_*_epochs.npy"))
    print(f"\nFound {len(epoch_files)} subjects")
    
    if len(epoch_files) == 0:
        print("❌ No epoch files found!")
        return
    
    # Sample random epochs
    np.random.seed(42)
    sampled_data = []
    failures = {'load_error': 0, 'bic_error': 0}
    
    print(f"\nSampling {args.n_samples} random epochs...")
    
    attempts = 0
    max_attempts = args.n_samples * 10
    
    with tqdm(total=args.n_samples) as pbar:
        while len(sampled_data) < args.n_samples and attempts < max_attempts:
            # Random file
            subj_file = epoch_files[np.random.randint(0, len(epoch_files))]
            
            # Load and select random epoch
            try:
                epochs = np.load(subj_file)
                epoch_idx = np.random.randint(0, len(epochs))
                epoch_data = epochs[epoch_idx]  # (19, 1024)
            except:
                failures['load_error'] += 1
                attempts += 1
                continue
            
            # Compute BIC curve
            orders, bic_vals, _ = compute_bic_curve(epoch_data)
            
            if orders is None:
                failures['bic_error'] += 1
                attempts += 1
                continue
            
            # Analyze quality
            category, metrics = analyze_bic_curve_quality(orders, bic_vals)
            
            sampled_data.append({
                'file': subj_file.name,
                'epoch_idx': epoch_idx,
                'category': category,
                'optimal_order': metrics.get('optimal_order', np.nan),
                'is_edge': metrics.get('is_edge', False),
                'smoothness': metrics.get('smoothness', np.nan),
                'u_shape_present': metrics.get('u_shape_present', False),
                'depth': metrics.get('depth', np.nan)
            })
            
            pbar.update(1)
            attempts += 1
    
    df = pd.DataFrame(sampled_data)
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    category_counts = df['category'].value_counts()
    print("\nBIC Curve Categories:")
    print("-" * 50)
    
    category_order = ['clear', 'acceptable', 'noisy', 'edge', 'chaotic', 'flat']
    for cat in category_order:
        if cat in category_counts.index:
            count = category_counts[cat]
            pct = 100 * count / len(df)
            print(f"  {cat:15s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nTotal analyzed: {len(df)}")
    print(f"Load errors: {failures['load_error']}")
    print(f"BIC errors: {failures['bic_error']}")
    
    # Quality summary
    good = df['category'].isin(['clear', 'acceptable']).sum()
    okay = df['category'].isin(['noisy']).sum()
    poor = df['category'].isin(['edge', 'chaotic', 'flat']).sum()
    
    good_pct = 100 * good / len(df)
    okay_pct = 100 * okay / len(df)
    usable_pct = good_pct + okay_pct
    
    print("\n" + "-" * 50)
    print("Quality Summary:")
    print("-" * 50)
    print(f"  Good (clear + acceptable): {good:3d} ({good_pct:5.1f}%)")
    print(f"  Okay (noisy):              {okay:3d} ({okay_pct:5.1f}%)")
    print(f"  Poor (edge + chaotic):     {poor:3d} ({100*poor/len(df):5.1f}%)")
    print(f"  TOTAL USABLE:              {good+okay:3d} ({usable_pct:5.1f}%)")
    
    # Assessment
    print("\n" + "-" * 50)
    if usable_pct > 75:
        print(f"✅ EXCELLENT ({usable_pct:.1f}% usable)")
        print("   Your p=12 choice is strongly supported!")
    elif usable_pct > 60:
        print(f"✓ GOOD ({usable_pct:.1f}% usable)")
        print("   p=12 should work well.")
    else:
        print(f"~ ACCEPTABLE ({usable_pct:.1f}% usable)")
    
    # Optimal order distribution
    print("\n" + "-" * 50)
    print("Optimal Order Distribution:")
    print("-" * 50)
    order_counts = df['optimal_order'].value_counts().sort_index()
    for order, count in order_counts.items():
        if not np.isnan(order):
            pct = 100 * count / len(df)
            marker = " ← RECOMMENDED" if int(order) == 12 else ""
            print(f"  p={int(order):2d}: {count:3d} ({pct:5.1f}%){marker}")
    
    # Save
    df.to_csv(output_dir / 'tuc_bic_quality.csv', index=False)
    print(f"\n✅ Saved: {output_dir / 'tuc_bic_quality.csv'}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {
        'clear': '#27ae60',
        'acceptable': '#3498db',
        'noisy': '#f39c12',
        'edge': '#e67e22',
        'chaotic': '#e74c3c',
        'flat': '#95a5a6'
    }
    
    # Plot 1: Category distribution
    categories_present = [c for c in category_order if c in category_counts.index]
    counts = [category_counts.get(c, 0) for c in categories_present]
    bar_colors = [colors.get(c, 'gray') for c in categories_present]
    
    axes[0].bar(range(len(categories_present)), counts, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(categories_present)))
    axes[0].set_xticklabels(categories_present, rotation=45, ha='right')
    axes[0].set_title('BIC Curve Quality Distribution', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(counts):
        pct = 100 * count / len(df)
        axes[0].text(i, count + max(counts)*0.02, f'{pct:.1f}%', ha='center', fontweight='bold')
    
    # Plot 2: Quality summary
    quality_groups = ['Good\n(clear+acceptable)', 'Okay\n(noisy)', 'Poor\n(edge+chaotic)']
    quality_counts = [good, okay, poor]
    quality_colors = ['#27ae60', '#f39c12', '#e74c3c']
    
    axes[1].bar(quality_groups, quality_counts, color=quality_colors, alpha=0.8, edgecolor='black')
    axes[1].set_title('Overall Quality', fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(quality_counts):
        pct = 100 * count / len(df)
        axes[1].text(i, count + max(quality_counts)*0.02, f'{pct:.1f}%', ha='center', fontweight='bold')
    
    # Plot 3: Optimal order by category
    for cat in ['clear', 'acceptable', 'noisy']:
        if cat in df['category'].values:
            subset = df[df['category'] == cat]['optimal_order'].dropna()
            if len(subset) > 0:
                axes[2].hist(subset, bins=range(MIN_ORDER, MAX_ORDER+2),
                           alpha=0.6, label=cat, edgecolor='black', color=colors.get(cat))
    
    axes[2].axvline(12, color='red', linestyle='-', linewidth=2.5, label='Recommended: p=12', zorder=10)
    axes[2].set_title('Optimal Order by Quality', fontweight='bold')
    axes[2].set_xlabel('Optimal Order (p)')
    axes[2].set_ylabel('Count')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tuc_bic_quality_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_dir / 'tuc_bic_quality_summary.png'}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
