"""
CONNECTIVITY PIPELINE VALIDATION
=================================
Tests YOUR actual step2_compute_connectivity.py functions using synthetic signals.

This validates:
1. signals_test.m  → Cascade network (Ch1→Ch2→Ch3→Ch4)
2. signals_test2.m → Branching network (Ch1→{Ch2,Ch3}, Ch2→Ch4)

If validation passes → Your pipeline is CORRECT ✅
If validation fails  → Check .T transpose in VAR() calls ❌
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import linalg
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings("ignore")

# Import test signal generators
from signals_test import signals_test
from signals_test2 import signals_test2


# =============================================================================
# YOUR ACTUAL CONNECTIVITY FUNCTIONS (from step2_compute_connectivity.py)
# =============================================================================

def compute_dtf_pdc_from_var(coefs, fs=1000.0, nfft=512):
    """
    YOUR ACTUAL function - copied from step2_compute_connectivity.py
    """
    p, K, _ = coefs.shape
    n_freqs = nfft // 2 + 1
    freqs = np.linspace(0, fs/2, n_freqs)
    
    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    H_f = np.zeros((n_freqs, K, K), dtype=complex)
    I = np.eye(K)
    
    # Compute A(f) and H(f)
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
    """
    YOUR ACTUAL function - copied from step2_compute_connectivity.py
    """
    # Check data quality
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    
    # Standardize
    data_scaled = data / data_std
    
    try:
        # CRITICAL: Check YOUR actual code for .T transpose!
        # From your step2_compute_connectivity.py line 140:
        model = VAR(data_scaled.T)  # ← YOUR ACTUAL CODE
        
        # Fit with FIXED order
        try:
            results = model.fit(maxlags=fixed_order, trend='c', verbose=False)
        except:
            return None
        
        if results.k_ar == 0:
            return None
        
        # Compute full spectrum DTF/PDC
        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(
            results.coefs, fs, nfft
        )
        
        # Integrate over 0.5-45 Hz (or full band for validation)
        freq_mask = (freqs >= 0.5) & (freqs <= 45.0)
        dtf_integrated = np.mean(dtf_spectrum[:, :, freq_mask], axis=2)
        pdc_integrated = np.mean(pdc_spectrum[:, :, freq_mask], axis=2)
        
        # Set diagonal to zero
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
# VALIDATION TESTS
# =============================================================================

def validate_cascade(snr_db=20, n_trials=10):
    """
    Test 1: Cascade Network (signals_test.m)
    
    Ground Truth: Ch1→Ch2→Ch3→Ch4, Ch5=noise
    """
    print("\n" + "=" * 80)
    print(f"TEST 1: CASCADE NETWORK (signals_test.m) - SNR = {snr_db} dB")
    print("=" * 80)
    print("Ground Truth: Ch1 → Ch2 → Ch3 → Ch4")
    print("              Ch5: Noise (isolated)")
    print("=" * 80)
    
    pdc_matrices = []
    dtf_matrices = []
    
    for trial in range(n_trials):
        # Generate test signals
        x = signals_test(db_noise=snr_db)  # (1900, 5)
        
        # Format as epoch (n_channels, n_samples)
        # Use middle 1024 samples
        epoch = x[400:1424, :].T  # (5, 1024)
        
        # Process with YOUR pipeline
        result = process_single_epoch(epoch, fs=1000, fixed_order=12, nfft=512)
        
        if result is not None:
            pdc_matrices.append(result['pdc'])
            dtf_matrices.append(result['dtf'])
    
    # Average across trials
    if len(pdc_matrices) == 0:
        print("❌ Pipeline FAILED on all trials!")
        return None, None, False
    
    pdc_avg = np.mean(pdc_matrices, axis=0)
    dtf_avg = np.mean(dtf_matrices, axis=0)
    
    # =========================================================================
    # VALIDATION CHECKS
    # =========================================================================
    
    print(f"\n✅ Succeeded on {len(pdc_matrices)}/{n_trials} trials\n")
    
    print("PDC Results:")
    print(f"  Direct connections (should be HIGH):")
    print(f"    Ch1→Ch2: {pdc_avg[1, 0]:.3f}")
    print(f"    Ch2→Ch3: {pdc_avg[2, 1]:.3f}")
    print(f"    Ch3→Ch4: {pdc_avg[3, 2]:.3f}")
    
    print(f"\n  Indirect connections (should be LOW):")
    print(f"    Ch1→Ch3: {pdc_avg[2, 0]:.3f}")
    print(f"    Ch1→Ch4: {pdc_avg[3, 0]:.3f}")
    
    print(f"\n  Ch5 isolation (should be ~0):")
    print(f"    Ch5→others: {np.mean(pdc_avg[0:4, 4]):.3f}")
    print(f"    Others→Ch5: {np.mean(pdc_avg[4, 0:4]):.3f}")
    
    print("\nDTF Results:")
    print(f"  Direct connections: {dtf_avg[1,0]:.3f}, {dtf_avg[2,1]:.3f}, {dtf_avg[3,2]:.3f}")
    print(f"  Indirect:          {dtf_avg[2,0]:.3f}, {dtf_avg[3,0]:.3f}")
    print(f"  Ch5 isolation:     {np.mean(dtf_avg[0:4, 4]):.3f}, {np.mean(dtf_avg[4, 0:4]):.3f}")
    
    # Validation criteria
    pdc_direct = np.mean([pdc_avg[1, 0], pdc_avg[2, 1], pdc_avg[3, 2]])
    pdc_indirect = np.mean([pdc_avg[2, 0], pdc_avg[3, 0]])
    pdc_isolation = np.mean([np.mean(pdc_avg[0:4, 4]), np.mean(pdc_avg[4, 0:4])])
    
    dtf_direct = np.mean([dtf_avg[1, 0], dtf_avg[2, 1], dtf_avg[3, 2]])
    dtf_indirect = np.mean([dtf_avg[2, 0], dtf_avg[3, 0]])
    
    print("\n" + "-" * 80)
    print("VALIDATION CHECKS:")
    print("-" * 80)
    
    passed = 0
    total = 0
    
    # PDC checks
    total += 1
    if pdc_direct > 0.3:
        print(f"✅ PDC direct > 0.3:    {pdc_direct:.3f}")
        passed += 1
    else:
        print(f"❌ PDC direct too low:  {pdc_direct:.3f}")
    
    total += 1
    if pdc_direct > pdc_indirect * 2:
        print(f"✅ PDC direct > 2×indirect: {pdc_direct:.3f} > {2*pdc_indirect:.3f}")
        passed += 1
    else:
        print(f"❌ PDC ratio too low")
    
    total += 1
    if pdc_isolation < 0.15:
        print(f"✅ PDC Ch5 isolated:    {pdc_isolation:.3f}")
        passed += 1
    else:
        print(f"❌ PDC Ch5 contaminated: {pdc_isolation:.3f}")
    
    # DTF checks
    total += 1
    if dtf_direct > 0.3:
        print(f"✅ DTF direct > 0.3:    {dtf_direct:.3f}")
        passed += 1
    else:
        print(f"❌ DTF direct too low:  {dtf_direct:.3f}")
    
    total += 1
    if dtf_direct > dtf_indirect * 2:
        print(f"✅ DTF direct > 2×indirect: {dtf_direct:.3f} > {2*dtf_indirect:.3f}")
        passed += 1
    else:
        print(f"❌ DTF ratio too low")
    
    success = (passed == total)
    
    print("-" * 80)
    if success:
        print(f"✅ CASCADE TEST PASSED ({passed}/{total})")
    else:
        print(f"❌ CASCADE TEST FAILED ({passed}/{total})")
    print("=" * 80)
    
    return pdc_avg, dtf_avg, success


def validate_branching(snr_db=20, n_trials=10):
    """
    Test 2: Branching Network (signals_test2.m)
    
    Ground Truth: Ch1→{Ch2,Ch3}, Ch2→Ch4, Ch5=noise
    """
    print("\n" + "=" * 80)
    print(f"TEST 2: BRANCHING NETWORK (signals_test2.m) - SNR = {snr_db} dB")
    print("=" * 80)
    print("Ground Truth:    Ch1")
    print("                /   \\")
    print("              Ch2   Ch3")
    print("               |")
    print("              Ch4")
    print("              Ch5: Noise")
    print("=" * 80)
    
    pdc_matrices = []
    dtf_matrices = []
    
    for trial in range(n_trials):
        x = signals_test2(db_noise=snr_db)
        epoch = x[400:1424, :].T  # (5, 1024)
        
        result = process_single_epoch(epoch, fs=1000, fixed_order=12, nfft=512)
        
        if result is not None:
            pdc_matrices.append(result['pdc'])
            dtf_matrices.append(result['dtf'])
    
    if len(pdc_matrices) == 0:
        print("❌ Pipeline FAILED on all trials!")
        return None, None, False
    
    pdc_avg = np.mean(pdc_matrices, axis=0)
    dtf_avg = np.mean(dtf_matrices, axis=0)
    
    print(f"\n✅ Succeeded on {len(pdc_matrices)}/{n_trials} trials\n")
    
    print("PDC Results:")
    print(f"  Common source (should be HIGH):")
    print(f"    Ch1→Ch2: {pdc_avg[1, 0]:.3f}")
    print(f"    Ch1→Ch3: {pdc_avg[2, 0]:.3f}")
    
    print(f"\n  Downstream (should be HIGH):")
    print(f"    Ch2→Ch4: {pdc_avg[3, 1]:.3f}")
    
    print(f"\n  Independent branches (should be LOW):")
    print(f"    Ch2→Ch3: {pdc_avg[2, 1]:.3f}")
    print(f"    Ch3→Ch2: {pdc_avg[1, 2]:.3f}")
    
    print("\nDTF Results:")
    print(f"  Common source: {dtf_avg[1,0]:.3f}, {dtf_avg[2,0]:.3f}")
    print(f"  Downstream:    {dtf_avg[3,1]:.3f}")
    print(f"  Independent:   {dtf_avg[2,1]:.3f}, {dtf_avg[1,2]:.3f}")
    
    # Validation
    pdc_fork = np.mean([pdc_avg[1, 0], pdc_avg[2, 0]])
    pdc_independent = np.mean([pdc_avg[2, 1], pdc_avg[1, 2]])
    
    print("\n" + "-" * 80)
    print("VALIDATION CHECKS:")
    print("-" * 80)
    
    passed = 0
    total = 0
    
    total += 1
    if pdc_fork > 0.3:
        print(f"✅ PDC fork detected:   {pdc_fork:.3f}")
        passed += 1
    else:
        print(f"❌ PDC fork too weak:   {pdc_fork:.3f}")
    
    total += 1
    if pdc_fork > pdc_independent * 2:
        print(f"✅ PDC branches independent: {pdc_fork:.3f} > {2*pdc_independent:.3f}")
        passed += 1
    else:
        print(f"❌ PDC branches contaminated")
    
    total += 1
    if pdc_avg[3, 1] > 0.3:
        print(f"✅ PDC Ch2→Ch4 detected: {pdc_avg[3, 1]:.3f}")
        passed += 1
    else:
        print(f"❌ PDC Ch2→Ch4 too weak: {pdc_avg[3, 1]:.3f}")
    
    success = (passed == total)
    
    print("-" * 80)
    if success:
        print(f"✅ BRANCHING TEST PASSED ({passed}/{total})")
    else:
        print(f"❌ BRANCHING TEST FAILED ({passed}/{total})")
    print("=" * 80)
    
    return pdc_avg, dtf_avg, success


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(pdc_cascade, pdc_branch, dtf_cascade, dtf_branch, snr_db, output_dir):
    """Create validation plots - separate for PDC and DTF."""
    
    # =========================================================================
    # PDC PLOTS
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Cascade
    sns.heatmap(pdc_cascade, ax=axes[0], cmap='YlOrRd', annot=True, fmt='.3f',
                xticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                yticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'PDC'})
    
    for i, j in [(1, 0), (2, 1), (3, 2)]:
        axes[0].add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                       edgecolor='blue', lw=3))
    
    axes[0].set_title(f'PDC - Cascade (Ch1→Ch2→Ch3→Ch4)\nSNR={snr_db}dB',
                     fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Source')
    axes[0].set_ylabel('Target')
    
    # Branching
    sns.heatmap(pdc_branch, ax=axes[1], cmap='YlOrRd', annot=True, fmt='.3f',
                xticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                yticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'PDC'})
    
    for i, j in [(1, 0), (2, 0), (3, 1)]:
        axes[1].add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                       edgecolor='blue', lw=3))
    
    axes[1].set_title(f'PDC - Branching (Ch1→{{Ch2,Ch3}}, Ch2→Ch4)\nSNR={snr_db}dB',
                     fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Source')
    axes[1].set_ylabel('Target')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'PDC_validation_snr{snr_db}db.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # DTF PLOTS
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Cascade
    sns.heatmap(dtf_cascade, ax=axes[0], cmap='viridis', annot=True, fmt='.3f',
                xticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                yticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'DTF'})
    
    for i, j in [(1, 0), (2, 1), (3, 2)]:
        axes[0].add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                       edgecolor='lime', lw=3))
    
    axes[0].set_title(f'DTF - Cascade (Ch1→Ch2→Ch3→Ch4)\nSNR={snr_db}dB',
                     fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Source')
    axes[0].set_ylabel('Target')
    
    # Branching
    sns.heatmap(dtf_branch, ax=axes[1], cmap='viridis', annot=True, fmt='.3f',
                xticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                yticklabels=['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5'],
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'DTF'})
    
    for i, j in [(1, 0), (2, 0), (3, 1)]:
        axes[1].add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                       edgecolor='lime', lw=3))
    
    axes[1].set_title(f'DTF - Branching (Ch1→{{Ch2,Ch3}}, Ch2→Ch4)\nSNR={snr_db}dB',
                     fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Source')
    axes[1].set_ylabel('Target')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'DTF_validation_snr{snr_db}db.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_numerical_results(results, output_dir):
    """Save all numerical results to files."""
    import json
    
    for snr in results.keys():
        # Save PDC matrices
        np.save(output_dir / f'PDC_cascade_snr{snr}db.npy', 
                results[snr]['pdc_cascade'])
        np.save(output_dir / f'PDC_branch_snr{snr}db.npy', 
                results[snr]['pdc_branch'])
        
        # Save DTF matrices
        np.save(output_dir / f'DTF_cascade_snr{snr}db.npy', 
                results[snr]['dtf_cascade'])
        np.save(output_dir / f'DTF_branch_snr{snr}db.npy', 
                results[snr]['dtf_branch'])
        
        # Save summary statistics
        summary = {
            'snr_db': snr,
            'cascade_test': {
                'passed': results[snr]['cascade_passed'],
                'pdc': {
                    'Ch1→Ch2': float(results[snr]['pdc_cascade'][1, 0]),
                    'Ch2→Ch3': float(results[snr]['pdc_cascade'][2, 1]),
                    'Ch3→Ch4': float(results[snr]['pdc_cascade'][3, 2]),
                    'Ch1→Ch3_indirect': float(results[snr]['pdc_cascade'][2, 0]),
                    'Ch1→Ch4_indirect': float(results[snr]['pdc_cascade'][3, 0]),
                },
                'dtf': {
                    'Ch1→Ch2': float(results[snr]['dtf_cascade'][1, 0]),
                    'Ch2→Ch3': float(results[snr]['dtf_cascade'][2, 1]),
                    'Ch3→Ch4': float(results[snr]['dtf_cascade'][3, 2]),
                    'Ch1→Ch3_indirect': float(results[snr]['dtf_cascade'][2, 0]),
                    'Ch1→Ch4_indirect': float(results[snr]['dtf_cascade'][3, 0]),
                }
            },
            'branching_test': {
                'passed': results[snr]['branch_passed'],
                'pdc': {
                    'Ch1→Ch2': float(results[snr]['pdc_branch'][1, 0]),
                    'Ch1→Ch3': float(results[snr]['pdc_branch'][2, 0]),
                    'Ch2→Ch4': float(results[snr]['pdc_branch'][3, 1]),
                    'Ch2→Ch3_should_be_low': float(results[snr]['pdc_branch'][2, 1]),
                },
                'dtf': {
                    'Ch1→Ch2': float(results[snr]['dtf_branch'][1, 0]),
                    'Ch1→Ch3': float(results[snr]['dtf_branch'][2, 0]),
                    'Ch2→Ch4': float(results[snr]['dtf_branch'][3, 1]),
                    'Ch2→Ch3_should_be_low': float(results[snr]['dtf_branch'][2, 1]),
                }
            }
        }
        
        with open(output_dir / f'summary_snr{snr}db.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"\n✅ Saved numerical results:")
    print(f"   - PDC matrices (.npy)")
    print(f"   - DTF matrices (.npy)")
    print(f"   - Summary statistics (.json)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CONNECTIVITY PIPELINE VALIDATION")
    print("=" * 80)
    print("Testing YOUR step2_compute_connectivity.py functions")
    print("Using professor's test signals (Python translations)")
    print("=" * 80)
    
    results = {}
    
    for snr in [20, 10]:
        print(f"\n\n{'#' * 80}")
        print(f"# SNR = {snr} dB")
        print(f"{'#' * 80}")
        
        pdc1, dtf1, pass1 = validate_cascade(snr_db=snr, n_trials=10)
        pdc2, dtf2, pass2 = validate_branching(snr_db=snr, n_trials=10)
        
        results[snr] = {
            'cascade_passed': pass1,
            'branch_passed': pass2,
            'pdc_cascade': pdc1,
            'pdc_branch': pdc2,
            'dtf_cascade': dtf1,
            'dtf_branch': dtf2
        }
        
        # Plot - separate PDC and DTF
        if pdc1 is not None and pdc2 is not None:
            plot_results(pdc1, pdc2, dtf1, dtf2, snr, output_dir)
            print(f"\n✅ Saved: validation_results/PDC_validation_snr{snr}db.png")
            print(f"✅ Saved: validation_results/DTF_validation_snr{snr}db.png")
    
    # Save all numerical results
    save_numerical_results(results, output_dir)
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for snr in [20, 10]:
        cascade = "✅ PASS" if results[snr]['cascade_passed'] else "❌ FAIL"
        branch = "✅ PASS" if results[snr]['branch_passed'] else "❌ FAIL"
        print(f"\nSNR {snr}dB:")
        print(f"  Cascade test:   {cascade}")
        print(f"  Branching test: {branch}")
        
        if not (results[snr]['cascade_passed'] and results[snr]['branch_passed']):
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉🎉🎉 YOUR PIPELINE IS CORRECT! 🎉🎉🎉")
        print("\nYour connectivity computation is validated and ready for use!")
    else:
        print("❌ YOUR PIPELINE HAS ISSUES")
        print("\nMost likely bug: .T transpose in VAR() call")
        print("Check line 140 in process_single_epoch()")
        print("\nIf using .T → Remove it")
        print("If NOT using .T → Add it")
    print("=" * 80)


if __name__ == "__main__":
    main()