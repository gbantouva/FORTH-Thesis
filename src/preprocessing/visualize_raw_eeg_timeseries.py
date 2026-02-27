"""
Visualize Raw EEG Time Series from MATLAB File
===============================================
Reads the continuous raw signal directly from the MATLAB file
(no epoch reconstruction needed!) and creates stacked plots
with SHARED Y-AXIS per subject for direct channel comparison.

This is the CORRECT approach - use the original continuous signal!

Usage:
------
python visualize_raw_eeg_from_matlab.py \
    --matlab_file "F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat" \
    --output_dir "F:\FORTH_Final_Thesis\FORTH-Thesis\figures\raw_eeg_timeseries" \
    --subjects 1 2 3 4 5
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# TUC channel names (standard 19-channel montage)
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2']

FS = 256  # Sampling frequency (Hz)


def plot_subject_raw_eeg(subject_idx, x, annotations, output_dir, zoom_window=60):
    """
    Plot raw EEG for one subject with shared y-axis across all 19 channels.
    
    Parameters:
    -----------
    subject_idx : int
        Subject index (0-indexed)
    x : array
        Raw EEG signal (n_samples, 19)
    annotations : array
        Seizure [start, end] samples
    output_dir : Path
        Output directory
    zoom_window : int
        Window in seconds around seizure for zoomed plot
    """
    subject_id = subject_idx + 1  # Convert to 1-indexed
    
    # Get signal and annotations
    signal = x[subject_idx]  # (n_samples, 19)
    n_samples, n_channels = signal.shape
    
    seizure_start_sample = int(annotations[subject_idx][0])
    seizure_end_sample = int(annotations[subject_idx][1])
    
    # Convert to time
    time_axis = np.arange(n_samples) / FS
    seizure_start_sec = seizure_start_sample / FS
    seizure_end_sec = seizure_end_sample / FS
    seizure_duration_sec = seizure_end_sec - seizure_start_sec
    
    print(f"\n  Subject {subject_id:02d}:")
    print(f"    Signal shape: {signal.shape}")
    print(f"    Duration: {time_axis[-1]:.1f}s ({time_axis[-1]/60:.1f}min)")
    print(f"    Seizure: {seizure_start_sec:.1f}s - {seizure_end_sec:.1f}s ({seizure_duration_sec:.1f}s)")
    
    # =========================================================================
    # CRITICAL: Calculate SHARED y-axis limits for THIS SUBJECT
    # =========================================================================
    # Use ACTUAL min/max - show EVERYTHING (artifacts included!)
    global_min = np.min(signal)  # Absolute minimum
    global_max = np.max(signal)  # Absolute maximum
    y_margin = (global_max - global_min) * 0.05
    y_limits = [global_min - y_margin, global_max + y_margin]
    
    print(f"    Shared y-axis: [{y_limits[0]:.1f}, {y_limits[1]:.1f}] µV (ABSOLUTE min/max)")
    
    # =========================================================================
    # PLOT 1: Full recording with stacked channels
    # =========================================================================
    fig, axes = plt.subplots(19, 1, figsize=(18, 20), sharex=True, sharey=True)
    
    for ch_idx, ch_name in enumerate(CHANNELS):
        ax = axes[ch_idx]
        
        # Get signal for this channel
        ch_signal = signal[:, ch_idx]
        
        # Plot
        ax.plot(time_axis, ch_signal, linewidth=0.5, color='steelblue', alpha=0.8)
        
        # Mark seizure period
        ax.axvspan(seizure_start_sec, seizure_end_sec, 
                   alpha=0.3, color='red', zorder=0)
        
        # Seizure onset line
        ax.axvline(seizure_start_sec, color='red', 
                   linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
        
        # Channel label
        ax.set_ylabel(ch_name, fontsize=11, rotation=0, 
                     labelpad=30, fontweight='bold', va='center')
        
        # Grid
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
    
    # Apply SHARED y-axis to all subplots
    for ax in axes:
        ax.set_ylim(y_limits)
    
    # X-axis only on bottom
    axes[-1].set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    axes[-1].set_xlim([0, time_axis[-1]])
    
    # Title
    fig.suptitle(
        f'Subject {subject_id:02d} - Raw EEG Time Series (19 Channels)\n'
        f'Total: {time_axis[-1]:.1f}s ({time_axis[-1]/60:.1f}min)  |  '
        f'Seizure: {seizure_start_sec:.1f}s - {seizure_end_sec:.1f}s ({seizure_duration_sec:.1f}s)\n'
        f'SHARED Y-AXIS: [{y_limits[0]:.1f}, {y_limits[1]:.1f}] µV (ABSOLUTE min/max - all signal shown)',
        fontsize=15, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / f'subject_{subject_id:02d}_raw_eeg_full.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ Saved: {save_path.name}")
    
    # =========================================================================
    # PLOT 2: Zoomed view around seizure (±zoom_window seconds)
    # =========================================================================
    zoom_start_sec = max(0, seizure_start_sec - zoom_window)
    zoom_end_sec = min(time_axis[-1], seizure_end_sec + zoom_window)
    
    zoom_start_idx = int(zoom_start_sec * FS)
    zoom_end_idx = int(zoom_end_sec * FS)
    
    zoom_signal = signal[zoom_start_idx:zoom_end_idx, :]
    zoom_time = time_axis[zoom_start_idx:zoom_end_idx]
    
    # Calculate y-limits for zoomed region - ACTUAL min/max
    zoom_min = np.min(zoom_signal)
    zoom_max = np.max(zoom_signal)
    zoom_margin = (zoom_max - zoom_min) * 0.05
    zoom_limits = [zoom_min - zoom_margin, zoom_max + zoom_margin]
    
    fig2, axes2 = plt.subplots(19, 1, figsize=(18, 20), sharex=True, sharey=True)
    
    for ch_idx, ch_name in enumerate(CHANNELS):
        ax = axes2[ch_idx]
        
        # Get signal for this channel (zoomed)
        ch_signal = zoom_signal[:, ch_idx]
        
        # Plot
        ax.plot(zoom_time, ch_signal, linewidth=0.8, color='steelblue', alpha=0.9)
        
        # Mark seizure period
        ax.axvspan(seizure_start_sec, seizure_end_sec,
                   alpha=0.3, color='red', zorder=0)
        
        # Seizure onset line
        ax.axvline(seizure_start_sec, color='red',
                   linestyle='--', linewidth=2, alpha=0.8, zorder=1)
        
        # Channel label
        ax.set_ylabel(ch_name, fontsize=11, rotation=0,
                     labelpad=30, fontweight='bold', va='center')
        
        # Grid
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
    
    # Apply SHARED y-axis to all subplots (zoomed scale)
    for ax in axes2:
        ax.set_ylim(zoom_limits)
    
    axes2[-1].set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    axes2[-1].set_xlim([zoom_start_sec, zoom_end_sec])
    
    fig2.suptitle(
        f'Subject {subject_id:02d} - Zoomed View (±{zoom_window}s around seizure)\n'
        f'Window: {zoom_start_sec:.1f}s - {zoom_end_sec:.1f}s  |  '
        f'SHARED Y-AXIS: [{zoom_limits[0]:.1f}, {zoom_limits[1]:.1f}] µV (ABSOLUTE min/max)',
        fontsize=15, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    
    save_path_zoom = output_dir / f'subject_{subject_id:02d}_raw_eeg_zoom.png'
    plt.savefig(save_path_zoom, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ Saved: {save_path_zoom.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize raw EEG directly from MATLAB file with shared y-axis per subject"
    )
    parser.add_argument("--matlab_file", required=True,
                       help="Path to MATLAB .mat file")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for plots")
    parser.add_argument("--subjects", nargs='+', type=int,
                       default=list(range(1, 35)),
                       help="Subject IDs to process (1-34)")
    parser.add_argument("--zoom_window", type=int, default=60,
                       help="Zoom window around seizure in seconds (default: 60)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RAW EEG VISUALIZATION FROM MATLAB FILE")
    print("="*80)
    print(f"MATLAB file: {args.matlab_file}")
    print(f"Output dir:  {output_dir}")
    print(f"Subjects:    {len(args.subjects)}")
    print(f"Zoom window: ±{args.zoom_window}s around seizure")
    print("="*80)
    print("\nKEY FEATURES:")
    print("  ✅ Uses raw continuous signal directly (no epoch reconstruction)")
    print("  ✅ SHARED Y-AXIS per subject (compare channels directly)")
    print("  ✅ ABSOLUTE min/max scaling (ALL signal shown - artifacts included!)")
    print("  ✅ Creates both full and zoomed views")
    print("="*80)
    
    # Load MATLAB file
    print("\nLoading MATLAB file...")
    data = loadmat(args.matlab_file, squeeze_me=True, struct_as_record=False)
    seizure = data['seizure']
    
    x = seizure.x  # Raw EEG signals
    annotations = seizure.annotation  # Seizure start/stop samples
    
    n_subjects = len(x)
    print(f"✓ Loaded {n_subjects} subjects")
    
    # Process requested subjects
    success = 0
    errors = 0
    
    for subj_id in args.subjects:
        subj_idx = subj_id - 1  # Convert to 0-indexed
        
        if subj_idx < 0 or subj_idx >= n_subjects:
            print(f"\n⚠️  Subject {subj_id} out of range (1-{n_subjects})")
            errors += 1
            continue
        
        try:
            plot_subject_raw_eeg(
                subj_idx, 
                x, 
                annotations, 
                output_dir, 
                zoom_window=args.zoom_window
            )
            success += 1
        except Exception as e:
            print(f"\n❌ Error processing subject {subj_id}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Success: {success}/{len(args.subjects)}")
    print(f"❌ Errors:  {errors}/{len(args.subjects)}")
    print("\nCreated 2 plots per subject:")
    print("  • subject_XX_raw_eeg_full.png - Full recording")
    print("  • subject_XX_raw_eeg_zoom.png - Zoomed around seizure")
    print(f"\nOutput directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()