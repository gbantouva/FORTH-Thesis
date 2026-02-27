"""
Comprehensive Seizure Timing Verification
==========================================
This script checks if seizure times are correctly calculated by comparing:
1. Metadata (ground truth from annotations)
2. Epoch labels (created by step0)
3. Connectivity data (filtered by step2)
4. Time arrays (time_from_onset)

Usage:
------
python verify_seizure_timing_comprehensive.py \
    --epochs_dir F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs \
    --connectivity_dir F:\FORTH_Final_Thesis\FORTH-Thesis\connectivity \
    --subject_ids 1 2 3 4 5
"""

import argparse
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2']


def verify_subject_timing(subject_id, epochs_dir, connectivity_dir):
    """
    Comprehensive timing verification for one subject.
    
    Returns:
    --------
    dict with verification results
    """
    subject_name = f"subject_{subject_id:02d}"
    
    # =========================================================================
    # LOAD ALL FILES
    # =========================================================================
    
    # Metadata (ground truth)
    metadata_file = epochs_dir / f"{subject_name}_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Epoch files (created by step0)
    labels_file = epochs_dir / f"{subject_name}_labels.npy"
    time_from_onset_file = epochs_dir / f"{subject_name}_time_from_onset.npy"
    epoch_present_mask_file = epochs_dir / f"{subject_name}_epoch_present_mask.npy"
    
    labels_all = np.load(labels_file)
    time_from_onset_all = np.load(time_from_onset_file)
    
    # Check if epoch_present_mask exists
    if epoch_present_mask_file.exists():
        epoch_present_mask = np.load(epoch_present_mask_file)
    else:
        print(f"⚠️  {subject_name}: epoch_present_mask.npy not found, assuming all epochs present")
        epoch_present_mask = np.ones(len(labels_all), dtype=bool)
    
    # Connectivity data (filtered by step2)
    graphs_file = connectivity_dir / f"{subject_name}_graphs.npz"
    data = np.load(graphs_file)
    labels_filtered = data['labels']
    
    # =========================================================================
    # EXTRACT KEY PARAMETERS
    # =========================================================================
    
    fs = metadata.get('fs', 256)
    seizure_start_sample = metadata['seizure_start_sample']
    seizure_end_sample = metadata['seizure_end_sample']
    
    seizure_start_sec = seizure_start_sample / fs
    seizure_end_sec = seizure_end_sample / fs
    seizure_duration_sec = seizure_end_sec - seizure_start_sec
    
    # =========================================================================
    # CHECK 1: Metadata Consistency
    # =========================================================================
    
    checks = {}
    
    checks['metadata_start_sec'] = seizure_start_sec
    checks['metadata_end_sec'] = seizure_end_sec
    checks['metadata_duration_sec'] = seizure_duration_sec
    checks['metadata_duration_samples'] = seizure_end_sample - seizure_start_sample
    
    # =========================================================================
    # CHECK 2: Epoch Labels in Original Data (step0 output)
    # =========================================================================
    
    # Find ictal epochs
    ictal_indices_all = np.where(labels_all == 1)[0]
    
    if len(ictal_indices_all) == 0:
        print(f"❌ {subject_name}: No ictal epochs found!")
        return None
    
    first_ictal_all = ictal_indices_all[0]
    last_ictal_all = ictal_indices_all[-1]
    
    # Expected first ictal epoch (from metadata)
    # Epoch length = 4 seconds, sample rate = 256 Hz
    epoch_length_samples = 4 * fs  # 1024 samples
    expected_first_ictal = seizure_start_sample // epoch_length_samples
    expected_last_ictal = seizure_end_sample // epoch_length_samples
    
    checks['n_epochs_all'] = len(labels_all)
    checks['n_ictal_all'] = len(ictal_indices_all)
    checks['first_ictal_all'] = first_ictal_all
    checks['last_ictal_all'] = last_ictal_all
    checks['expected_first_ictal'] = expected_first_ictal
    checks['expected_last_ictal'] = expected_last_ictal
    
    # Verify first ictal epoch
    checks['first_ictal_match'] = abs(first_ictal_all - expected_first_ictal) <= 1
    checks['first_ictal_error'] = first_ictal_all - expected_first_ictal
    
    # =========================================================================
    # CHECK 3: time_from_onset Alignment
    # =========================================================================
    
    # time_from_onset should be 0 at first ictal epoch
    time_at_first_ictal = time_from_onset_all[first_ictal_all]
    time_at_last_ictal = time_from_onset_all[last_ictal_all]
    
    checks['time_at_first_ictal'] = time_at_first_ictal
    checks['time_at_last_ictal'] = time_at_last_ictal
    
    # Should be close to 0 (within ±4 seconds for one epoch tolerance)
    checks['time_first_ictal_ok'] = abs(time_at_first_ictal) < 4.0
    
    # Last ictal time should match seizure duration
    checks['time_last_ictal_vs_duration'] = time_at_last_ictal - seizure_duration_sec
    checks['time_last_ictal_ok'] = abs(time_at_last_ictal - seizure_duration_sec) < 8.0
    
    # =========================================================================
    # CHECK 4: Epoch Present Mask (connectivity filtering)
    # =========================================================================
    
    n_failed = len(epoch_present_mask) - epoch_present_mask.sum()
    
    checks['epoch_present_mask_length'] = len(epoch_present_mask)
    checks['n_epochs_with_connectivity'] = epoch_present_mask.sum()
    checks['n_epochs_failed'] = n_failed
    checks['success_rate'] = 100 * epoch_present_mask.sum() / len(epoch_present_mask)
    
    # =========================================================================
    # CHECK 5: Filtered Data (connectivity graphs)
    # =========================================================================
    
    # Apply mask
    labels_masked = labels_all[epoch_present_mask]
    time_masked = time_from_onset_all[epoch_present_mask]
    
    # Find ictal in masked data
    ictal_indices_masked = np.where(labels_masked == 1)[0]
    
    if len(ictal_indices_masked) == 0:
        print(f"❌ {subject_name}: No ictal epochs after filtering!")
        return None
    
    first_ictal_masked = ictal_indices_masked[0]
    last_ictal_masked = ictal_indices_masked[-1]
    
    checks['n_ictal_filtered'] = len(ictal_indices_masked)
    checks['first_ictal_filtered'] = first_ictal_masked
    checks['last_ictal_filtered'] = last_ictal_masked
    
    # Time at filtered ictal epochs
    time_first_filtered = time_masked[first_ictal_masked]
    time_last_filtered = time_masked[last_ictal_masked]
    
    checks['time_first_filtered'] = time_first_filtered
    checks['time_last_filtered'] = time_last_filtered
    
    # =========================================================================
    # CHECK 6: Connectivity Data Match
    # =========================================================================
    
    # Check if filtered labels match connectivity labels
    checks['labels_match'] = np.array_equal(labels_masked, labels_filtered)
    checks['n_epochs_connectivity'] = len(labels_filtered)
    
    # =========================================================================
    # OVERALL VERIFICATION
    # =========================================================================
    
    all_passed = (
        checks['first_ictal_match'] and
        checks['time_first_ictal_ok'] and
        checks['time_last_ictal_ok'] and
        checks['labels_match']
    )
    
    checks['all_checks_passed'] = all_passed
    checks['subject_id'] = subject_id
    
    return checks


def print_verification_report(checks):
    """Print detailed verification report."""
    
    subject_id = checks['subject_id']
    
    print("\n" + "="*80)
    print(f"SEIZURE TIMING VERIFICATION - Subject {subject_id:02d}")
    print("="*80)
    
    print("\n📋 METADATA (Ground Truth)")
    print(f"  Seizure start:    {checks['metadata_start_sec']:.2f}s (sample {checks['metadata_start_sec']*256:.0f})")
    print(f"  Seizure end:      {checks['metadata_end_sec']:.2f}s (sample {checks['metadata_end_sec']*256:.0f})")
    print(f"  Duration:         {checks['metadata_duration_sec']:.2f}s ({checks['metadata_duration_samples']} samples)")
    
    print("\n📊 EPOCH LABELS (step0 output)")
    print(f"  Total epochs:     {checks['n_epochs_all']}")
    print(f"  Ictal epochs:     {checks['n_ictal_all']}")
    print(f"  First ictal:      Epoch {checks['first_ictal_all']} (expected: {checks['expected_first_ictal']})")
    print(f"  Last ictal:       Epoch {checks['last_ictal_all']} (expected: ~{checks['expected_last_ictal']})")
    
    if checks['first_ictal_match']:
        print(f"  ✅ First ictal epoch matches metadata (error: {checks['first_ictal_error']} epochs)")
    else:
        print(f"  ❌ First ictal epoch OFF by {checks['first_ictal_error']} epochs!")
    
    print("\n⏱️  TIME_FROM_ONSET Array")
    print(f"  At first ictal:   {checks['time_at_first_ictal']:.2f}s (should be ~0)")
    print(f"  At last ictal:    {checks['time_at_last_ictal']:.2f}s (should be ~{checks['metadata_duration_sec']:.2f}s)")
    
    if checks['time_first_ictal_ok']:
        print(f"  ✅ First ictal time is correct")
    else:
        print(f"  ❌ First ictal time is OFF by {abs(checks['time_at_first_ictal']):.2f}s!")
    
    if checks['time_last_ictal_ok']:
        print(f"  ✅ Last ictal time matches duration (diff: {checks['time_last_ictal_vs_duration']:.2f}s)")
    else:
        print(f"  ❌ Last ictal time is OFF by {abs(checks['time_last_ictal_vs_duration']):.2f}s!")
    
    print("\n🔍 CONNECTIVITY FILTERING (step2 output)")
    print(f"  Epochs with connectivity: {checks['n_epochs_with_connectivity']}/{checks['epoch_present_mask_length']}")
    print(f"  Failed VAR fitting:       {checks['n_epochs_failed']} epochs ({100-checks['success_rate']:.1f}%)")
    print(f"  Ictal after filtering:    {checks['n_ictal_filtered']} epochs")
    print(f"  Time at first filtered:   {checks['time_first_filtered']:.2f}s")
    print(f"  Time at last filtered:    {checks['time_last_filtered']:.2f}s")
    
    if checks['labels_match']:
        print(f"  ✅ Filtered labels match connectivity data")
    else:
        print(f"  ❌ Filtered labels DO NOT match connectivity data!")
    
    print("\n" + "="*80)
    if checks['all_checks_passed']:
        print("✅ ALL CHECKS PASSED - Seizure timing is CORRECT")
    else:
        print("❌ SOME CHECKS FAILED - Review issues above")
    print("="*80)


def plot_timing_visualization(subject_id, epochs_dir, connectivity_dir, output_dir):
    """
    Create visual verification plot showing:
    - Epoch labels
    - time_from_onset
    - Seizure boundaries from metadata
    """
    subject_name = f"subject_{subject_id:02d}"
    
    # Load data
    metadata_file = epochs_dir / f"{subject_name}_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    labels = np.load(epochs_dir / f"{subject_name}_labels.npy")
    time_from_onset = np.load(epochs_dir / f"{subject_name}_time_from_onset.npy")
    
    epoch_mask_file = epochs_dir / f"{subject_name}_epoch_present_mask.npy"
    if epoch_mask_file.exists():
        epoch_mask = np.load(epoch_mask_file)
    else:
        epoch_mask = np.ones(len(labels), dtype=bool)
    
    # Get metadata
    fs = metadata.get('fs', 256)
    seizure_start_sec = metadata['seizure_start_sample'] / fs
    seizure_end_sec = metadata['seizure_end_sample'] / fs
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    epochs_indices = np.arange(len(labels))
    
    # =========================================================================
    # Plot 1: Epoch Labels (0 = pre-ictal, 1 = ictal)
    # =========================================================================
    ax = axes[0]
    
    # Color by label and mask
    colors = ['blue' if epoch_mask[i] else 'gray' for i in range(len(labels))]
    colors = ['red' if labels[i] == 1 and epoch_mask[i] else c for i, c in enumerate(colors)]
    
    ax.scatter(epochs_indices, labels, c=colors, s=20, alpha=0.6)
    ax.set_ylabel('Label\n(0=pre, 1=ictal)', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.2, 1.2])
    ax.set_title(f'{subject_name} - Seizure Timing Verification', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(['Pre-ictal (good)', 'Failed VAR', 'Ictal (good)'], loc='upper right')
    
    # =========================================================================
    # Plot 2: time_from_onset
    # =========================================================================
    ax = axes[1]
    
    ax.plot(epochs_indices, time_from_onset, color='steelblue', linewidth=2)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Time = 0 (expected ictal start)')
    ax.axhline(seizure_end_sec - seizure_start_sec, color='orange', linestyle='--', 
               linewidth=2, label=f'Expected ictal end ({seizure_end_sec - seizure_start_sec:.1f}s)')
    
    # Mark actual ictal epochs
    ictal_indices = np.where(labels == 1)[0]
    if len(ictal_indices) > 0:
        ax.axvspan(ictal_indices[0], ictal_indices[-1], alpha=0.2, color='red', label='Actual ictal period')
    
    ax.set_ylabel('Time from onset (s)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch Index', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    
    # =========================================================================
    # Plot 3: Epoch Present Mask (which epochs have connectivity)
    # =========================================================================
    ax = axes[2]
    
    ax.plot(epochs_indices, epoch_mask.astype(int), color='green', linewidth=2, marker='o', markersize=3)
    ax.fill_between(epochs_indices, 0, epoch_mask.astype(int), alpha=0.3, color='green')
    ax.set_ylabel('Has Connectivity\n(1=yes, 0=failed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch Index', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.2, 1.2])
    ax.grid(alpha=0.3)
    
    # Add text annotation
    n_failed = len(epoch_mask) - epoch_mask.sum()
    ax.text(0.02, 0.95, f'Failed VAR: {n_failed}/{len(epoch_mask)} epochs ({100*n_failed/len(epoch_mask):.1f}%)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{subject_name}_timing_verification.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Saved visualization: {subject_name}_timing_verification.png")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive seizure timing verification"
    )
    parser.add_argument("--epochs_dir", required=True,
                       help="Directory with epoch files")
    parser.add_argument("--connectivity_dir", required=True,
                       help="Directory with connectivity files")
    parser.add_argument("--output_dir", default="figures/timing_verification",
                       help="Output directory for plots")
    parser.add_argument("--subject_ids", nargs='+', type=int,
                       default=list(range(1, 35)),
                       help="Subject IDs to verify")
    parser.add_argument("--plot", action='store_true',
                       help="Generate visual verification plots")
    
    args = parser.parse_args()
    
    epochs_dir = Path(args.epochs_dir)
    connectivity_dir = Path(args.connectivity_dir)
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("COMPREHENSIVE SEIZURE TIMING VERIFICATION")
    print("="*80)
    print(f"Epochs dir:       {epochs_dir}")
    print(f"Connectivity dir: {connectivity_dir}")
    print(f"Subjects:         {len(args.subject_ids)}")
    print("="*80)
    
    all_results = []
    passed = 0
    failed = 0
    
    for subj_id in args.subject_ids:
        checks = verify_subject_timing(subj_id, epochs_dir, connectivity_dir)
        
        if checks is None:
            failed += 1
            continue
        
        print_verification_report(checks)
        
        if args.plot:
            plot_timing_visualization(subj_id, epochs_dir, connectivity_dir, output_dir)
        
        all_results.append(checks)
        
        if checks['all_checks_passed']:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Passed:  {passed}/{len(args.subject_ids)}")
    print(f"❌ Failed:  {failed}/{len(args.subject_ids)}")
    
    if failed > 0:
        print("\n⚠️  Subjects with issues:")
        for result in all_results:
            if not result['all_checks_passed']:
                print(f"  - Subject {result['subject_id']:02d}")
    
    print("="*80)


if __name__ == "__main__":
    main()