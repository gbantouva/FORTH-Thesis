"""
FINAL VALIDATION: Test YOUR Pipeline with Professor's EXACT MATLAB Code
========================================================================
Uses exact Python translation of signals_test.m to validate your pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# IMPORT EXACT MATLAB TRANSLATION
# =============================================================================

from signals_test import signals_test

print("=" * 80)
print("USING EXACT TRANSLATION OF PROFESSOR'S signals_test.m")
print("=" * 80)

# =============================================================================
# IMPORT YOUR ACTUAL PIPELINE FUNCTIONS  
# =============================================================================

from statsmodels.tsa.vector_ar.var_model import VAR
from scipy import linalg
import warnings
warnings.filterwarnings("ignore")

def compute_bic_for_epoch(data, min_order=8, max_order=22):
    """YOUR ACTUAL BIC function - copy from your step1_bic_analysis.py"""
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    
    data_scaled = data / data_std
    
    try:
        # YOUR CODE - check if you have .T here
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


def compute_dtf_pdc_from_var(coefs, fs=256.0, nfft=512):
    """YOUR ACTUAL PDC/DTF computation - copy from step2_compute_connectivity.py"""
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
    
    # PDC
    pdc = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Af = A_f[f_idx]
        col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))
        col_norms[col_norms == 0] = 1e-10
        pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]
    
    # DTF
    dtf = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Hf = H_f[f_idx]
        row_norms = np.sqrt(np.sum(np.abs(Hf)**2, axis=1))
        row_norms[row_norms == 0] = 1e-10
        dtf[:, :, f_idx] = np.abs(Hf) / row_norms[:, None]
    
    return dtf, pdc, freqs


def process_single_epoch(data, fs, fixed_order, nfft):
    """YOUR ACTUAL connectivity function - copy from step2_compute_connectivity.py"""
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    data_scaled = data / data_std
    
    try:
        # YOUR CODE - check if you have .T here
        model = VAR(data_scaled.T)
        
        try:
            results = model.fit(maxlags=fixed_order, trend='c', verbose=False)
        except:
            return None
        
        if results.k_ar == 0:
            return None
        
        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(
            results.coefs, fs, nfft
        )
        
        freq_mask = (freqs >= 0.5) & (freqs <= 45.0)
        dtf_integrated = np.mean(dtf_spectrum[:, :, freq_mask], axis=2)
        pdc_integrated = np.mean(pdc_spectrum[:, :, freq_mask], axis=2)
        
        np.fill_diagonal(dtf_integrated, 0.0)
        np.fill_diagonal(pdc_integrated, 0.0)
        
        return {
            'dtf': dtf_integrated,
            'pdc': pdc_integrated,
            'order': fixed_order
        }
        
    except:
        return None


# =============================================================================
# VALIDATION TEST
# =============================================================================

def test_pipeline(snr_db=20, n_trials=10):
    """Test YOUR pipeline with professor's EXACT test signals."""
    
    print(f"\n{'='*80}")
    print(f"VALIDATION TEST - SNR = {snr_db} dB")
    print(f"{'='*80}")
    print(f"Trials: {n_trials}")
    print(f"Using: Professor's EXACT signals_test.m (Python translation)")
    print(f"{'='*80}\n")
    
    pdc_matrices = []
    dtf_matrices = []
    bic_orders = []
    
    for trial in range(n_trials):
        # Generate using EXACT MATLAB translation
        x = signals_test(db_noise=snr_db)  # (1901, 5)
        
        # Format as epoch (n_channels, n_samples)
        epoch = x[100:1124, :].T  # (5, 1024)
        
        # Test BIC
        order = compute_bic_for_epoch(epoch, min_order=8, max_order=22)
        if order is not None:
            bic_orders.append(order)
        
        # Test connectivity
        result = process_single_epoch(epoch, fs=1000, fixed_order=12, nfft=512)
        if result is not None:
            pdc_matrices.append(result['pdc'])
            dtf_matrices.append(result['dtf'])
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    
    print("STEP 1: BIC ANALYSIS")
    print("=" * 80)
    if len(bic_orders) > 0:
        print(f"✅ BIC succeeded: {len(bic_orders)}/{n_trials} trials")
        print(f"   Mean order: {np.mean(bic_orders):.1f}")
        print(f"   Median order: {np.median(bic_orders):.0f}")
    else:
        print("❌ BIC failed on all trials!")
    
    print(f"\nSTEP 2: CONNECTIVITY COMPUTATION")
    print("=" * 80)
    
    if len(pdc_matrices) == 0:
        print("❌ Connectivity failed on all trials!")
        return None, None, False
    
    pdc_avg = np.mean(pdc_matrices, axis=0)
    dtf_avg = np.mean(dtf_matrices, axis=0)
    
    print(f"✅ Connectivity succeeded: {len(pdc_matrices)}/{n_trials} trials\n")
    
    print("PDC Results - Ground Truth (should be HIGH):")
    print(f"  Ch1 → Ch2: {pdc_avg[1, 0]:.3f}")
    print(f"  Ch2 → Ch3: {pdc_avg[2, 1]:.3f}")
    print(f"  Ch3 → Ch4: {pdc_avg[3, 2]:.3f}")
    
    print("\nDTF Results - Ground Truth (should be HIGH):")
    print(f"  Ch1 → Ch2: {dtf_avg[1, 0]:.3f}")
    print(f"  Ch2 → Ch3: {dtf_avg[2, 1]:.3f}")
    print(f"  Ch3 → Ch4: {dtf_avg[3, 2]:.3f}")
    
    print("\nPDC Indirect (should be LOW):")
    print(f"  Ch1 → Ch3: {pdc_avg[2, 0]:.3f}")
    print(f"  Ch1 → Ch4: {pdc_avg[3, 0]:.3f}")
    
    print("\nDTF Indirect (should be LOW):")
    print(f"  Ch1 → Ch3: {dtf_avg[2, 0]:.3f}")
    print(f"  Ch1 → Ch4: {dtf_avg[3, 0]:.3f}")
    
    print("\nPDC Ch5 Isolation (should be ~0):")
    print(f"  Ch5 → others: {np.mean(pdc_avg[0:4, 4]):.3f}")
    print(f"  Others → Ch5: {np.mean(pdc_avg[4, 0:4]):.3f}")
    
    print("\nDTF Ch5 Isolation (should be ~0):")
    print(f"  Ch5 → others: {np.mean(dtf_avg[0:4, 4]):.3f}")
    print(f"  Others → Ch5: {np.mean(dtf_avg[4, 0:4]):.3f}")
    
    # Validation
    print(f"\n{'='*80}")
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    # PDC checks
    pdc_direct = np.mean([pdc_avg[1, 0], pdc_avg[2, 1], pdc_avg[3, 2]])
    pdc_indirect = np.mean([pdc_avg[2, 0], pdc_avg[3, 0]])
    pdc_isolation = np.mean([np.mean(pdc_avg[0:4, 4]), np.mean(pdc_avg[4, 0:4])])
    
    # DTF checks
    dtf_direct = np.mean([dtf_avg[1, 0], dtf_avg[2, 1], dtf_avg[3, 2]])
    dtf_indirect = np.mean([dtf_avg[2, 0], dtf_avg[3, 0]])
    dtf_isolation = np.mean([np.mean(dtf_avg[0:4, 4]), np.mean(dtf_avg[4, 0:4])])
    
    passed = 0
    
    # PDC validation
    print("\nPDC Validation:")
    if pdc_direct > 0.3:
        print(f"  ✅ PDC Direct connections: {pdc_direct:.3f} > 0.3")
        passed += 1
    else:
        print(f"  ❌ PDC Direct too weak: {pdc_direct:.3f} < 0.3")
    
    if pdc_direct > pdc_indirect * 2:
        print(f"  ✅ PDC Direct > 2×Indirect: {pdc_direct:.3f} > {2*pdc_indirect:.3f}")
        passed += 1
    else:
        print(f"  ❌ PDC Not strong enough")
    
    if pdc_isolation < 0.15:
        print(f"  ✅ PDC Ch5 isolated: {pdc_isolation:.3f} < 0.15")
        passed += 1
    else:
        print(f"  ❌ PDC Ch5 contaminated: {pdc_isolation:.3f}")
    
    # DTF validation
    print("\nDTF Validation:")
    if dtf_direct > 0.3:
        print(f"  ✅ DTF Direct connections: {dtf_direct:.3f} > 0.3")
        passed += 1
    else:
        print(f"  ❌ DTF Direct too weak: {dtf_direct:.3f} < 0.3")
    
    if dtf_direct > dtf_indirect * 2:
        print(f"  ✅ DTF Direct > 2×Indirect: {dtf_direct:.3f} > {2*dtf_indirect:.3f}")
        passed += 1
    else:
        print(f"  ❌ DTF Not strong enough")
    
    if dtf_isolation < 0.15:
        print(f"  ✅ DTF Ch5 isolated: {dtf_isolation:.3f} < 0.15")
        passed += 1
    else:
        print(f"  ❌ DTF Ch5 contaminated: {dtf_isolation:.3f}")
    
    success = (passed == 6)  # All 6 checks must pass (3 for PDC, 3 for DTF)
    
    print(f"\n{'='*80}")
    if success:
        print("✅✅✅ YOUR PIPELINE IS CORRECT! ✅✅✅")
    else:
        print("❌ YOUR PIPELINE HAS BUGS")
        print("\nMost likely issue: .T transpose in VAR() calls")
        print("Check: compute_bic_for_epoch() and process_single_epoch()")
    print("=" * 80)
    
    # Visualization - show both PDC and DTF
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PDC heatmap
    sns.heatmap(pdc_avg, ax=axes[0], cmap='YlOrRd', annot=True, fmt='.3f',
                xticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                yticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'PDC'})
    
    for i, j in [(1, 0), (2, 1), (3, 2)]:
        axes[0].add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                       edgecolor='blue', lw=3))
    
    axes[0].set_title(f'PDC - Professor\'s signals_test.m\nSNR={snr_db}dB',
                     fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Source')
    axes[0].set_ylabel('Target')
    
    # DTF heatmap
    sns.heatmap(dtf_avg, ax=axes[1], cmap='YlOrRd', annot=True, fmt='.3f',
                xticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                yticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'DTF'})
    
    for i, j in [(1, 0), (2, 1), (3, 2)]:
        axes[1].add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                       edgecolor='blue', lw=3))
    
    axes[1].set_title(f'DTF - Professor\'s signals_test.m\nSNR={snr_db}dB',
                     fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Source')
    axes[1].set_ylabel('Target')
    
    plt.tight_layout()
    
    output_dir = Path("final_validation")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'validation_snr{snr_db}db.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved: final_validation/validation_snr{snr_db}db.png\n")
    
    return pdc_avg, dtf_avg, success


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    results = {}
    
    for snr in [20, 10]:
        pdc, dtf, passed = test_pipeline(snr_db=snr, n_trials=10)
        results[snr] = {'pdc': pdc, 'dtf': dtf, 'passed': passed}
    
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for snr in [20, 10]:
        status = "✅ PASS" if results[snr]['passed'] else "❌ FAIL"
        print(f"SNR {snr:2d} dB: {status}")
    
    if all(results[snr]['passed'] for snr in [20, 10]):
        print("\n🎉 PDC AND DTF BOTH VALIDATED AND CORRECT! 🎉")
    else:
        print("\n⚠️  Fix the .T transpose and re-run")
    print("=" * 80)