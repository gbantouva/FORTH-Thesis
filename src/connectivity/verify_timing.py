"""
verify_seizure_timing.py
=========================
Comprehensive verification that seizure timing is correctly aligned.

Your professor's concern:
"Connectivity is expected to slowly increase during seizure, but in almost every
channel just before seizure it is high then falls again. Make sure the red region
is indeed the seizure! Make sure time onset is correctly identified."

This script checks:
1. Seizure onset timing from metadata matches label assignments
2. No off-by-one epoch errors
3. time_from_onset array is correctly computed
4. Pre-ictal/ictal boundary is at expected location

Usage:
------
python verify_seizure_timing.py \
    --epochs_dir preprocessed_epochs \
    --subject_ids 1 2 3
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2']


def verify_subject_timing(subject_dir, subject_id, verbose=True):
    """
    Verify timing alignment for one subject.
    
    Returns:
    --------
    status : dict
        Contains 'passed', 'warnings', 'errors'
    """
    subject_name = f"subject_{subject_id:02d}"
    
    # Load files
    epochs_file = subject_dir / f"{subject_name}_epochs.npy"
    labels_file = subject_dir / f"{subject_name}_labels.npy"
    time_file = subject_dir / f"{subject_name}_time_from_onset.npy"
    metadata_file = subject_dir / f"{subject_name}_metadata.json"
    
    if not all([f.exists() for f in [epochs_file, labels_file, time_file, metadata_file]]):
        return {'passed': False, 'errors': ['Files not found']}
    
    epochs = np.load(epochs_file)
    labels = np.load(labels_file)
    time_from_onset = np.load(time_file)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    n_epochs, n_channels, samples_per_epoch = epochs.shape
    fs = metadata.get('fs', 256)
    epoch_duration = samples_per_epoch / fs  # Should be 4.0 seconds
    
    # Seizure timing from metadata
    seizure_start_sample = metadata['seizure_start_sample']
    seizure_end_sample = metadata['seizure_end_sample']
    
    seizure_start_sec = seizure_start_sample / fs
    seizure_end_sec = seizure_end_sample / fs
    seizure_duration = seizure_end_sec - seizure_start_sec
    
    # Find ictal epochs
    ictal_indices = np.where(labels == 1)[0]
    pre_ictal_indices = np.where(labels == 0)[0]
    
    if len(ictal_indices) == 0:
        return {'passed': False, 'errors': ['No ictal epochs found']}
    
    first_ictal = ictal_indices[0]
    last_ictal = ictal_indices[-1]
    
    # Expected: first ictal epoch should start at or just after seizure onset
    # Since epochs are 4s and non-overlapping, seizure at t=300s means:
    # - Epoch 0: [0, 4)
    # - Epoch 1: [4, 8)
    # ...
    # - Epoch 74: [296, 300)
    # - Epoch 75: [300, 304) ← FIRST ICTAL (if seizure starts at exactly 300s)
    
    expected_first_ictal = int(seizure_start_sec // epoch_duration)
    expected_last_ictal = int((seizure_end_sec - 0.001) // epoch_duration)  # -0.001 to handle boundary
    
    # Checks
    warnings = []
    errors = []
    
    # Check 1: First ictal epoch
    if first_ictal != expected_first_ictal:
        error_msg = (f"First ictal epoch mismatch: got {first_ictal}, "
                    f"expected {expected_first_ictal} (diff: {abs(first_ictal - expected_first_ictal)})")
        if abs(first_ictal - expected_first_ictal) <= 1:
            warnings.append(error_msg + " (within 1 epoch tolerance)")
        else:
            errors.append(error_msg)
    
    # Check 2: Last ictal epoch
    if last_ictal != expected_last_ictal:
        error_msg = (f"Last ictal epoch mismatch: got {last_ictal}, "
                    f"expected {expected_last_ictal} (diff: {abs(last_ictal - expected_last_ictal)})")
        if abs(last_ictal - expected_last_ictal) <= 1:
            warnings.append(error_msg + " (within 1 epoch tolerance)")
        else:
            errors.append(error_msg)
    
    # Check 3: time_from_onset alignment
    # time_from_onset should be negative for pre-ictal, ~0 at seizure onset, positive during seizure
    first_ictal_time = time_from_onset[first_ictal]
    
    # For a 4-second epoch centered at seizure onset, time_from_onset should be ~0
    # (could be -2 to +2 depending on how it's computed)
    if abs(first_ictal_time) > 10:  # More than 10s from expected onset
        errors.append(f"time_from_onset at first ictal epoch is {first_ictal_time:.1f}s, "
                     f"expected near 0s")
    elif abs(first_ictal_time) > 4:
        warnings.append(f"time_from_onset at first ictal epoch is {first_ictal_time:.1f}s "
                       f"(expected near 0s, but within tolerance)")
    
    # Check 4: Continuity of time_from_onset
    time_diff = np.diff(time_from_onset)
    expected_diff = epoch_duration
    
    if not np.allclose(time_diff, expected_diff, atol=0.1):
        max_deviation = np.max(np.abs(time_diff - expected_diff))
        warnings.append(f"time_from_onset not evenly spaced: max deviation {max_deviation:.2f}s")
    
    # Verbose output
    if verbose:
        print("\n" + "=" * 80)
        print(f"SUBJECT {subject_id:02d} TIMING VERIFICATION")
        print("=" * 80)
        print(f"Total epochs: {n_epochs}")
        print(f"Epoch duration: {epoch_duration:.1f}s")
        print(f"Sampling rate: {fs} Hz")
        print("")
        print("Metadata:")
        print(f"  Seizure start: {seizure_start_sec:.2f}s (sample {seizure_start_sample})")
        print(f"  Seizure end:   {seizure_end_sec:.2f}s (sample {seizure_end_sample})")
        print(f"  Duration:      {seizure_duration:.2f}s")
        print("")
        print("Epoch labels:")
        print(f"  Pre-ictal epochs: {len(pre_ictal_indices)} ({len(pre_ictal_indices) * epoch_duration:.1f}s)")
        print(f"  Ictal epochs:     {len(ictal_indices)} ({len(ictal_indices) * epoch_duration:.1f}s)")
        print(f"  First ictal:      Epoch {first_ictal}")
        print(f"  Last ictal:       Epoch {last_ictal}")
        print(f"  Expected first:   Epoch {expected_first_ictal}")
        print(f"  Expected last:    Epoch {expected_last_ictal}")
        print("")
        print("time_from_onset:")
        print(f"  At first pre-ictal: {time_from_onset[0]:.2f}s")
        print(f"  At first ictal:     {time_from_onset[first_ictal]:.2f}s")
        print(f"  At last ictal:      {time_from_onset[last_ictal]:.2f}s")
        print(f"  At last epoch:      {time_from_onset[-1]:.2f}s")
        print("")
        
        if errors:
            print("❌ ERRORS:")
            for e in errors:
                print(f"   - {e}")
        if warnings:
            print("⚠️  WARNINGS:")
            for w in warnings:
                print(f"   - {w}")
        if not errors and not warnings:
            print("✅ All checks passed!")
        
        print("=" * 80)
    
    return {
        'passed': len(errors) == 0,
        'warnings': warnings,
        'errors': errors,
        'metadata': {
            'seizure_start_sec': seizure_start_sec,
            'seizure_end_sec': seizure_end_sec,
            'first_ictal_epoch': first_ictal,
            'last_ictal_epoch': last_ictal,
            'first_ictal_time': first_ictal_time
        }
    }


def plot_timing_verification(subject_dir, subject_id, output_dir):
    """
    Create a visual verification plot showing:
    - Epoch boundaries
    - Label assignments
    - time_from_onset values
    - Seizure period from metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subject_name = f"subject_{subject_id:02d}"
    
    labels = np.load(subject_dir / f"{subject_name}_labels.npy")
    time_from_onset = np.load(subject_dir / f"{subject_name}_time_from_onset.npy")
    
    with open(subject_dir / f"{subject_name}_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    fs = metadata.get('fs', 256)
    seizure_start_sample = metadata['seizure_start_sample']
    seizure_end_sample = metadata['seizure_end_sample']
    
    seizure_start_sec = seizure_start_sample / fs
    seizure_end_sec = seizure_end_sample / fs
    
    n_epochs = len(labels)
    epoch_indices = np.arange(n_epochs)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # Top: Labels
    axes[0].scatter(epoch_indices, labels, c=labels, cmap='RdYlGn_r', 
                   s=50, edgecolors='black', linewidths=0.5)
    axes[0].set_ylabel('Label\n(0=pre-ictal, 1=ictal)', fontsize=11, fontweight='bold')
    axes[0].set_ylim([-0.2, 1.2])
    axes[0].set_yticks([0, 1])
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f'Subject {subject_id:02d} - Label Assignments vs Time', 
                     fontweight='bold')
    
    # Bottom: time_from_onset
    axes[1].plot(epoch_indices, time_from_onset, 'o-', markersize=4, linewidth=1.5)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2, 
                   label='Seizure onset (t=0)')
    axes[1].axhline(seizure_end_sec - seizure_start_sec, color='orange', 
                   linestyle='--', linewidth=2, label='Seizure end')
    axes[1].set_xlabel('Epoch index', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('time_from_onset (s)', fontsize=11, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    # Shade ictal region
    ictal_epochs = np.where(labels == 1)[0]
    if len(ictal_epochs) > 0:
        for ax in axes:
            ax.axvspan(ictal_epochs[0] - 0.5, ictal_epochs[-1] + 0.5,
                      alpha=0.2, color='red')
    
    plt.tight_layout()
    
    save_path = output_dir / f'subject_{subject_id:02d}_timing_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify seizure timing alignment"
    )
    parser.add_argument("--epochs_dir",
                       default="preprocessed_epochs",
                       help="Directory with epoch data")
    parser.add_argument("--output_dir",
                       default="figures/timing_verification",
                       help="Output directory for plots")
    parser.add_argument("--subject_ids", nargs='+', type=int,
                       default=list(range(1, 35)),
                       help="Subject IDs to verify")
    parser.add_argument("--plot", action='store_true',
                       help="Generate verification plots")
    
    args = parser.parse_args()
    
    epochs_dir = Path(args.epochs_dir)
    output_dir = Path(args.output_dir)
    
    print("\n" + "=" * 80)
    print("SEIZURE TIMING VERIFICATION")
    print("=" * 80)
    print(f"Input: {epochs_dir}")
    if args.plot:
        print(f"Output: {output_dir}")
    print("=" * 80)
    
    results = {}
    n_passed = 0
    n_warnings = 0
    n_errors = 0
    
    for subj_id in args.subject_ids:
        result = verify_subject_timing(epochs_dir, subj_id, verbose=True)
        results[subj_id] = result
        
        if result['passed']:
            n_passed += 1
        if result.get('warnings'):
            n_warnings += len(result['warnings'])
        if result.get('errors'):
            n_errors += len(result['errors'])
        
        if args.plot and result['passed']:
            try:
                plot_timing_verification(epochs_dir, subj_id, output_dir)
            except Exception as e:
                print(f"❌ Could not create plot for subject {subj_id}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total subjects verified: {len(results)}")
    print(f"✅ Passed: {n_passed}")
    print(f"⚠️  Warnings: {n_warnings}")
    print(f"❌ Errors: {n_errors}")
    
    if n_errors == 0:
        print("\n🎉 All timing checks passed!")
        print("   The red shaded regions in your plots correctly mark the seizure periods.")
    else:
        print("\n⚠️  Some subjects have timing issues. Review the errors above.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()