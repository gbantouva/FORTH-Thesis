"""
TUC Dataset - Create Epochs for Connectivity Analysis
======================================================
Correct labeling based on professor's requirements:
- Pre-ictal (0): ALL epochs OUTSIDE the seizure period (before AND after)
- Ictal (1): ONLY epochs that overlap with the seizure period

For GNN training: Use only epochs from start → seizure_end
For visualization: Use all epochs

Usage:
    python create_epochs_tuc.py
"""
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import json
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

MATLAB_FILE = r"F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat"
OUTPUT_DIR = Path(r"F:\FORTH_Final_Thesis\FORTH-Thesis\preprocessed_epochs")

FS = 256
EPOCH_LEN = 4.0
SAMPLES_PER_EPOCH = int(FS * EPOCH_LEN)  # 1024 samples

# Channel names (from exploration)
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 
            'O1', 'O2']

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("TUC DATASET - EPOCH CREATION")
print("="*80)
print(f"\nLoading: {MATLAB_FILE}")

data = loadmat(MATLAB_FILE, squeeze_me=True, struct_as_record=False)
seizure = data['seizure']

x = seizure.x
annotations = seizure.annotation
patient_ids = seizure.info

n_subjects = len(x)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ Loaded {n_subjects} subjects")
print(f"✓ Output directory: {OUTPUT_DIR}")

# ============================================================================
# EPOCH EACH SUBJECT
# ============================================================================

print(f"\n{'='*80}")
print("CREATING EPOCHS")
print("="*80)
print(f"\nEpoch length: {EPOCH_LEN} seconds ({SAMPLES_PER_EPOCH} samples)")
print(f"Sampling rate: {FS} Hz")
print(f"Channels: {len(CHANNELS)}")

total_epochs = 0
total_pre_ictal = 0
total_ictal = 0
all_metadata = []

for subj_idx in tqdm(range(n_subjects), desc="Processing subjects"):
    subj_data = x[subj_idx]  # (n_samples, 19)
    patient_id = patient_ids[subj_idx]
    
    # Get seizure annotation
    seizure_start_sample = int(annotations[subj_idx][0])
    seizure_end_sample = int(annotations[subj_idx][1])
    
    n_samples, n_channels = subj_data.shape
    assert n_channels == 19, f"Expected 19 channels, got {n_channels}"
    
    # Create non-overlapping epochs
    n_epochs = n_samples // SAMPLES_PER_EPOCH
    truncated_samples = n_epochs * SAMPLES_PER_EPOCH
    subj_data_truncated = subj_data[:truncated_samples, :]
    
    # Reshape: (n_epochs, samples_per_epoch, n_channels) → (n_epochs, n_channels, samples_per_epoch)
    epochs = subj_data_truncated.reshape(n_epochs, SAMPLES_PER_EPOCH, n_channels)
    epochs = np.transpose(epochs, (0, 2, 1))  # (n_epochs, 19, 1024)
    
    # =========================================================================
    # LABEL EPOCHS: Pre-ictal (0) vs Ictal (1)
    # =========================================================================
    # CRITICAL: An epoch is "ictal" if it OVERLAPS with the seizure period
    # [seizure_start_sample, seizure_end_sample]
    
    labels = np.zeros(n_epochs, dtype=np.int64)
    
    for ep_idx in range(n_epochs):
        epoch_start_sample = ep_idx * SAMPLES_PER_EPOCH
        epoch_end_sample = (ep_idx + 1) * SAMPLES_PER_EPOCH
        
        # Check if epoch overlaps with seizure period
        # Overlap condition: epoch_end > seizure_start AND epoch_start < seizure_end
        if (epoch_end_sample > seizure_start_sample and 
            epoch_start_sample < seizure_end_sample):
            labels[ep_idx] = 1  # Ictal
        else:
            labels[ep_idx] = 0  # Pre-ictal (includes both pre- and post-seizure)
    
    n_pre = np.sum(labels == 0)
    n_ict = np.sum(labels == 1)
    total_pre_ictal += n_pre
    total_ictal += n_ict
    
    # =========================================================================
    # TIME FROM ONSET (for analysis and visualization)
    # =========================================================================
    time_from_onset = np.zeros(n_epochs, dtype=np.float32)
    
    for ep_idx in range(n_epochs):
        epoch_start_sample = ep_idx * SAMPLES_PER_EPOCH
        # Negative = before onset, Positive = after onset
        time_from_onset[ep_idx] = (epoch_start_sample - seizure_start_sample) / FS
    
    # =========================================================================
    # TRAINING MASK (for GNN - only use start → seizure_end)
    # =========================================================================
    # Your professor said: "For training, only use start → seizure_end"
    # For visualization: use all epochs
    
    training_mask = np.zeros(n_epochs, dtype=bool)
    
    for ep_idx in range(n_epochs):
        epoch_start_sample = ep_idx * SAMPLES_PER_EPOCH
        # Include if epoch starts BEFORE seizure ends
        if epoch_start_sample < seizure_end_sample:
            training_mask[ep_idx] = True
    
    n_training = np.sum(training_mask)
    
    # Channel presence mask (all channels present for TUC)
    present_mask = np.ones(n_channels, dtype=bool)
    
    # =========================================================================
    # SAVE FILES
    # =========================================================================
    subject_name = f"subject_{subj_idx+1:02d}"
    
    np.save(OUTPUT_DIR / f"{subject_name}_epochs.npy", epochs)
    np.save(OUTPUT_DIR / f"{subject_name}_labels.npy", labels)
    np.save(OUTPUT_DIR / f"{subject_name}_time_from_onset.npy", time_from_onset)
    np.save(OUTPUT_DIR / f"{subject_name}_training_mask.npy", training_mask)
    np.save(OUTPUT_DIR / f"{subject_name}_present_mask.npy", present_mask)
    
    # Metadata
    seizure_duration_sec = (seizure_end_sample - seizure_start_sample) / FS
    
    metadata = {
        'subject_id': int(subj_idx + 1),
        'patient_id': str(patient_id),
        'n_epochs_total': int(n_epochs),
        'n_epochs_training': int(n_training),
        'n_pre_ictal': int(n_pre),
        'n_ictal': int(n_ict),
        'n_channels': 19,
        'sampling_rate': 256,
        'epoch_duration_sec': 4.0,
        'seizure_start_sample': seizure_start_sample,
        'seizure_start_sec': float(seizure_start_sample / FS),
        'seizure_end_sample': seizure_end_sample,
        'seizure_end_sec': float(seizure_end_sample / FS),
        'seizure_duration_sec': float(seizure_duration_sec),
        'channels': CHANNELS
    }
    
    with open(OUTPUT_DIR / f"{subject_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    all_metadata.append(metadata)
    total_epochs += n_epochs

# ============================================================================
# SAVE GLOBAL FILES
# ============================================================================

print(f"\n{'='*80}")
print("SAVING GLOBAL FILES")
print("="*80)

# Global time_from_onset (concatenated across all subjects)
all_time_from_onset = []
for subj_idx in range(n_subjects):
    subject_name = f"subject_{subj_idx+1:02d}"
    tfo = np.load(OUTPUT_DIR / f"{subject_name}_time_from_onset.npy")
    all_time_from_onset.append(tfo)

all_time_from_onset = np.concatenate(all_time_from_onset)
np.save(OUTPUT_DIR / 'all_time_from_onset.npy', all_time_from_onset)
print(f"  ✓ Saved: all_time_from_onset.npy")

# Global metadata
global_metadata = {
    'dataset': 'TUC Focal Seizures',
    'description': 'Pre-processed EEG epochs for connectivity analysis',
    'created': np.datetime64('now').astype(str),
    'n_subjects': int(n_subjects),
    'total_epochs': int(total_epochs),
    'total_pre_ictal': int(total_pre_ictal),
    'total_ictal': int(total_ictal),
    'channels': CHANNELS,
    'n_channels': len(CHANNELS),
    'sampling_rate': 256,
    'epoch_duration_sec': 4.0,
    'samples_per_epoch': SAMPLES_PER_EPOCH,
    'preprocessing': {
        'bandpass_filter_hz': [0.5, 45.0],
        'scaling_factor': 1.8,
        'source': 'TUC preprocessed MATLAB file'
    },
    'label_mapping': {
        '0': 'pre-ictal (outside seizure period)',
        '1': 'ictal (during seizure period)'
    },
    'usage_notes': {
        'training': 'Use epochs where training_mask=True (start → seizure_end)',
        'visualization': 'Use all epochs',
        'time_from_onset': 'Negative = before seizure, Positive = after seizure'
    },
    'subjects': all_metadata
}

with open(OUTPUT_DIR / 'dataset_metadata.json', 'w') as f:
    json.dump(global_metadata, f, indent=2)

print(f"  ✓ Saved: dataset_metadata.json")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("EPOCHING COMPLETE")
print("="*80)

print(f"\n📊 Dataset Summary:")
print(f"  Total subjects: {n_subjects}")
print(f"  Total epochs: {total_epochs:,}")
print(f"    Pre-ictal (label=0): {total_pre_ictal:,} ({100*total_pre_ictal/total_epochs:.1f}%)")
print(f"    Ictal (label=1):     {total_ictal:,} ({100*total_ictal/total_epochs:.1f}%)")
print(f"  Average epochs/subject: {total_epochs/n_subjects:.1f}")

print(f"\n📁 Output files per subject:")
print(f"  • subject_XX_epochs.npy         - EEG data (n_epochs, 19, 1024)")
print(f"  • subject_XX_labels.npy         - Labels (0=pre-ictal, 1=ictal)")
print(f"  • subject_XX_time_from_onset.npy - Time relative to seizure (seconds)")
print(f"  • subject_XX_training_mask.npy  - Which epochs to use for GNN training")
print(f"  • subject_XX_present_mask.npy   - Which channels are present")
print(f"  • subject_XX_metadata.json      - Subject-specific metadata")

print(f"\n📁 Global files:")
print(f"  • all_time_from_onset.npy  - Concatenated time arrays")
print(f"  • dataset_metadata.json    - Complete dataset information")

print(f"\n💾 All files saved to: {OUTPUT_DIR}")

print(f"\n🎯 Next steps:")
print(f"  1. Verify labels: Check a few subjects' metadata.json files")
print(f"  2. Run BIC analysis: python step1_with_epoch_level.py")
print(f"  3. Compute connectivity: python step2_compute_connectivity.py")
print(f"  4. Channel-specific analysis: python channel_specific_connectivity_analysis.py")

print(f"\n{'='*80}")
print("✅ SUCCESS")
print("="*80)