import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

RESULTS_DIR = Path(r"F:\FORTH_Final_Thesis\FORTH-Thesis\connectivity")
OUTPUT_DIR = Path(r"F:\FORTH_Final_Thesis\FORTH-Thesis\figures\connectivity\step3_statistics_globally_averaged_connectivity")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

BAND_NAMES = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']

all_files = list(RESULTS_DIR.rglob('*_graphs.npz'))
print(f"Found {len(all_files)} result files in {RESULTS_DIR}")

# Separate stats by label
stats = {band: {
    'pdc_all': [], 'dtf_all': [],
    'pdc_pre': [], 'dtf_pre': [],  # Pre-ictal (label=0)
    'pdc_ict': [], 'dtf_ict': []   # Ictal (label=1)
} for band in BAND_NAMES}

total_epochs = 0
total_pre = 0
total_ict = 0

for f in tqdm(all_files):
    data = np.load(f)
    labels = data['labels']  # 0=pre-ictal, 1=ictal
    n_epochs = len(labels)
    total_epochs += n_epochs
    total_pre += np.sum(labels == 0)
    total_ict += np.sum(labels == 1)

    for band in BAND_NAMES:
        pdc_matrix = data[f'pdc_{band}']  # (n_epochs, 19, 19)
        dtf_matrix = data[f'dtf_{band}']

        pdc_mean_vals = np.mean(pdc_matrix, axis=(1, 2))
        dtf_mean_vals = np.mean(dtf_matrix, axis=(1, 2))

        # All epochs
        stats[band]['pdc_all'].extend(pdc_mean_vals)
        stats[band]['dtf_all'].extend(dtf_mean_vals)
        
        # Separate by label
        stats[band]['pdc_pre'].extend(pdc_mean_vals[labels == 0])
        stats[band]['dtf_pre'].extend(dtf_mean_vals[labels == 0])
        stats[band]['pdc_ict'].extend(pdc_mean_vals[labels == 1])
        stats[band]['dtf_ict'].extend(dtf_mean_vals[labels == 1])

print(f"\nTotal epochs: {total_epochs}")
print(f"  Pre-ictal (0): {total_pre} ({100*total_pre/total_epochs:.1f}%)")
print(f"  Ictal (1):     {total_ict} ({100*total_ict/total_epochs:.1f}%)")

# =============================================================================
# PLOT 1: Global distribution (integrated band)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

pdc_vals = stats['integrated']['pdc_all']
axes[0].hist(pdc_vals, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[0].set_title(f"Global PDC Distribution (Integrated)\nMean: {np.mean(pdc_vals):.4f}", fontweight='bold')
axes[0].set_xlabel("Mean connectivity strength")
axes[0].set_ylabel("Count (epochs)")

dtf_vals = stats['integrated']['dtf_all']
axes[1].hist(dtf_vals, bins=50, color='teal', alpha=0.7, edgecolor='black')
axes[1].set_title(f"Global DTF Distribution (Integrated)\nMean: {np.mean(dtf_vals):.4f}", fontweight='bold')
axes[1].set_xlabel("Mean connectivity strength")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tuc_global_distribution_histograms.png', dpi=300)
plt.close()
print(f"✓ Saved: tuc_global_distribution_histograms.png")

# =============================================================================
# PLOT 2: All bands distribution
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, band in enumerate(BAND_NAMES):
    pdc_vals = stats[band]['pdc_all']
    dtf_vals = stats[band]['dtf_all']
    
    axes[i].hist(pdc_vals, bins=40, alpha=0.6, label=f'PDC (μ={np.mean(pdc_vals):.3f})', color='purple')
    axes[i].hist(dtf_vals, bins=40, alpha=0.6, label=f'DTF (μ={np.mean(dtf_vals):.3f})', color='teal')
    axes[i].set_title(f'{band.upper()} Band', fontweight='bold')
    axes[i].set_xlabel('Mean connectivity')
    axes[i].set_ylabel('Count')
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.suptitle('TUC Dataset - Connectivity Distribution by Frequency Band', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tuc_all_bands_distribution.png', dpi=300)
plt.close()
print(f"✓ Saved: tuc_all_bands_distribution.png")

# =============================================================================
# PLOT 3: PRE-ICTAL vs ICTAL COMPARISON (KEY FINDING!)
# =============================================================================
from scipy.stats import mannwhitneyu

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

print("\n" + "="*60)
print("PRE-ICTAL vs ICTAL PDC COMPARISON")
print("="*60)
print(f"{'Band':<12} {'Pre-ictal':<12} {'Ictal':<12} {'Diff':<12} {'p-value':<12}")
print("-"*60)

results = []

for i, band in enumerate(BAND_NAMES):
    pdc_pre = np.array(stats[band]['pdc_pre'])
    pdc_ict = np.array(stats[band]['pdc_ict'])
    
    mean_pre = np.mean(pdc_pre)
    mean_ict = np.mean(pdc_ict)
    diff = mean_ict - mean_pre
    
    # Statistical test
    stat, pval = mannwhitneyu(pdc_pre, pdc_ict, alternative='two-sided')
    
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"{band:<12} {mean_pre:<12.4f} {mean_ict:<12.4f} {diff:<+12.4f} {pval:<12.2e} {sig}")
    
    results.append({
        'band': band, 'pre': mean_pre, 'ict': mean_ict, 'diff': diff, 'pval': pval
    })
    
    # Plot
    axes[i].hist(pdc_pre, bins=30, alpha=0.6, label=f'Pre-ictal (μ={mean_pre:.3f})', color='blue')
    axes[i].hist(pdc_ict, bins=30, alpha=0.6, label=f'Ictal (μ={mean_ict:.3f})', color='red')
    axes[i].set_title(f'{band.upper()}\nΔ={diff:+.4f}, p={pval:.2e} {sig}', fontweight='bold')
    axes[i].set_xlabel('Mean PDC')
    axes[i].set_ylabel('Count')
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.suptitle('TUC Dataset - Pre-ictal vs Ictal PDC Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tuc_pre_vs_ictal_pdc.png', dpi=300)
plt.close()
print(f"\n✓ Saved: tuc_pre_vs_ictal_pdc.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("KEY FINDING")
print("="*60)

# Check direction
all_negative = all(r['diff'] < 0 for r in results)
all_significant = all(r['pval'] < 0.05 for r in results)

if all_negative and all_significant:
    print("✅ PRE-ICTAL HYPERCONNECTIVITY CONFIRMED!")
    print("   All bands show HIGHER connectivity during pre-ictal")
    print("   All differences are statistically significant (p < 0.05)")
elif all_negative:
    print("✅ Pre-ictal shows higher connectivity in all bands")
else:
    print("Mixed results - check individual bands")

# Find most significant band
most_sig = min(results, key=lambda x: x['pval'])
print(f"\n📊 Most significant band: {most_sig['band'].upper()}")
print(f"   Δ = {most_sig['diff']:+.4f}, p = {most_sig['pval']:.2e}")
