"""
Extract Patient Information and Seizure Details from TUC Dataset
=================================================================
This script reads the MATLAB file and creates a comprehensive JSON file with:
- Subject ID → Patient ID mapping
- Seizure duration for each subject
- Channel information
- Segment information

Usage:
------
python extract_patient_info.py \
    --matlab_file F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat \
    --output_file F:\FORTH_Final_Thesis\FORTH-Thesis\patient_info.json
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import json
from datetime import datetime

def extract_patient_info(matlab_file, output_file):
    """
    Extract patient information from MATLAB file.
    
    Parameters:
    -----------
    matlab_file : str or Path
        Path to the MATLAB .mat file
    output_file : str or Path
        Path to save the JSON output
    """
    
    print("="*80)
    print("EXTRACTING PATIENT INFORMATION FROM TUC DATASET")
    print("="*80)
    print(f"\nLoading: {matlab_file}")
    
    # Load MATLAB file
    data = loadmat(matlab_file, squeeze_me=True, struct_as_record=False)
    seizure = data['seizure']
    
    # Get data
    x = seizure.x  # EEG signals
    patient_ids = seizure.info  # Patient codes
    annotations = seizure.annotation  # Seizure start/stop samples
    
    n_subjects = len(x)
    fs = 256  # Sampling frequency
    
    print(f"✓ Loaded {n_subjects} subjects")
    
    # =========================================================================
    # Extract information for each subject
    # =========================================================================
    
    patient_info = {
        'dataset_name': 'TUC Focal Seizures',
        'n_subjects': int(n_subjects),
        'sampling_rate': int(fs),
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'subjects': []
    }
    
    print("\nExtracting information...")
    
    for subj_idx in range(n_subjects):
        subject_id = subj_idx + 1  # 1-indexed
        
        # Get patient ID (convert to string to handle different types)
        try:
            if isinstance(patient_ids, np.ndarray):
                patient_id = str(patient_ids[subj_idx])
            else:
                patient_id = str(patient_ids)
        except:
            patient_id = f"Unknown_{subject_id}"
        
        # Get EEG data
        eeg_data = x[subj_idx]
        n_samples, n_channels = eeg_data.shape
        
        # Get seizure annotation
        seizure_start_sample = int(annotations[subj_idx][0])
        seizure_end_sample = int(annotations[subj_idx][1])
        
        # Calculate durations
        seizure_duration_samples = seizure_end_sample - seizure_start_sample
        seizure_duration_sec = seizure_duration_samples / fs
        total_duration_sec = n_samples / fs
        
        # Calculate pre-ictal and post-ictal durations
        pre_ictal_duration_sec = seizure_start_sample / fs
        post_ictal_duration_sec = (n_samples - seizure_end_sample) / fs
        
        # Get segment information if available
        try:
            segment = seizure.segment[subj_idx]
            segment_start = int(segment[0])
            segment_end = int(segment[1])
        except:
            segment_start = 0
            segment_end = n_samples
        
        # Get selected seizure number if available
        try:
            selected_seizure = int(seizure.selected_seizures[subj_idx])
        except:
            selected_seizure = 1
        
        # Create subject entry
        subject_entry = {
            'subject_id': int(subject_id),
            'patient_id': patient_id,
            'selected_seizure_number': selected_seizure,
            
            'signal_info': {
                'n_samples': int(n_samples),
                'n_channels': int(n_channels),
                'total_duration_sec': float(total_duration_sec),
                'total_duration_min': float(total_duration_sec / 60)
            },
            
            'seizure_timing': {
                'start_sample': int(seizure_start_sample),
                'end_sample': int(seizure_end_sample),
                'start_sec': float(seizure_start_sample / fs),
                'end_sec': float(seizure_end_sample / fs),
                'duration_samples': int(seizure_duration_samples),
                'duration_sec': float(seizure_duration_sec),
                'duration_min': float(seizure_duration_sec / 60)
            },
            
            'period_durations': {
                'pre_ictal_sec': float(pre_ictal_duration_sec),
                'pre_ictal_min': float(pre_ictal_duration_sec / 60),
                'ictal_sec': float(seizure_duration_sec),
                'ictal_min': float(seizure_duration_sec / 60),
                'post_ictal_sec': float(post_ictal_duration_sec),
                'post_ictal_min': float(post_ictal_duration_sec / 60)
            },
            
            'segment_info': {
                'segment_start_sample': int(segment_start),
                'segment_end_sample': int(segment_end),
                'segment_duration_sec': float((segment_end - segment_start) / fs)
            }
        }
        
        patient_info['subjects'].append(subject_entry)
        
        print(f"  Subject {subject_id:02d}: Patient {patient_id:15s} | "
              f"Seizure: {seizure_duration_sec:6.2f}s | "
              f"Total: {total_duration_sec/60:5.2f}min")
    
    # =========================================================================
    # Calculate summary statistics
    # =========================================================================
    
    seizure_durations = [s['seizure_timing']['duration_sec'] for s in patient_info['subjects']]
    
    patient_info['summary'] = {
        'seizure_duration_stats': {
            'min_sec': float(np.min(seizure_durations)),
            'max_sec': float(np.max(seizure_durations)),
            'mean_sec': float(np.mean(seizure_durations)),
            'median_sec': float(np.median(seizure_durations)),
            'std_sec': float(np.std(seizure_durations))
        },
        'unique_patients': int(len(set(s['patient_id'] for s in patient_info['subjects']))),
        'total_subjects': int(n_subjects)
    }
    
    # =========================================================================
    # Create patient → subjects mapping
    # =========================================================================
    
    patient_to_subjects = {}
    for subject in patient_info['subjects']:
        patient_id = subject['patient_id']
        subject_id = subject['subject_id']
        
        if patient_id not in patient_to_subjects:
            patient_to_subjects[patient_id] = []
        patient_to_subjects[patient_id].append(subject_id)
    
    patient_info['patient_to_subjects_mapping'] = patient_to_subjects
    
    # =========================================================================
    # Save to JSON
    # =========================================================================
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(patient_info, f, indent=2)
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"  Total subjects:    {n_subjects}")
    print(f"  Unique patients:   {patient_info['summary']['unique_patients']}")
    print(f"\n⏱️  Seizure Duration Statistics:")
    print(f"  Min:     {patient_info['summary']['seizure_duration_stats']['min_sec']:.2f}s")
    print(f"  Max:     {patient_info['summary']['seizure_duration_stats']['max_sec']:.2f}s")
    print(f"  Mean:    {patient_info['summary']['seizure_duration_stats']['mean_sec']:.2f}s")
    print(f"  Median:  {patient_info['summary']['seizure_duration_stats']['median_sec']:.2f}s")
    print(f"  Std:     {patient_info['summary']['seizure_duration_stats']['std_sec']:.2f}s")
    
    print(f"\n📁 Saved to: {output_file}")
    
    # Print patients with multiple seizures
    multiple_seizures = {p: s for p, s in patient_to_subjects.items() if len(s) > 1}
    if multiple_seizures:
        print(f"\n👥 Patients with multiple seizures recorded:")
        for patient_id, subject_ids in sorted(multiple_seizures.items()):
            print(f"  Patient {patient_id}: {len(subject_ids)} seizures (subjects {subject_ids})")
    else:
        print(f"\n✓ Each patient has exactly 1 seizure recorded")
    
    print("\n" + "="*80)
    
    return patient_info


def main():
    parser = argparse.ArgumentParser(
        description="Extract patient information from TUC MATLAB file"
    )
    parser.add_argument("--matlab_file", required=True,
                       help="Path to MATLAB .mat file")
    parser.add_argument("--output_file", required=True,
                       help="Path to output JSON file")
    
    args = parser.parse_args()
    
    extract_patient_info(args.matlab_file, args.output_file)


if __name__ == "__main__":
    main()