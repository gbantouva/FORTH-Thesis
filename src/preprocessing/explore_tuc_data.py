"""
TUC Dataset - Complete Exploratory Data Analysis
=================================================
A comprehensive exploration script that discovers the dataset structure from scratch.
We pretend to know NOTHING and explore everything systematically.

This script will:
1. Load the MATLAB file and examine its structure
2. Discover what fields are available
3. Extract channel names (if possible)
4. Analyze all subjects' data
5. Find seizure annotations
6. Verify preprocessing quality
7. Create comprehensive visualizations
8. Save a detailed report

Usage:
    python explore_tuc_dataset.py
"""

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to MATLAB file - UPDATE THIS!
MATLAB_FILE = r"F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat"

# Output directory for plots and reports
OUTPUT_DIR = Path(r"F:\FORTH_Final_Thesis\FORTH-Thesis\figures\exploration_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title, char='='):
    """Print a formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def explore_struct(obj, name="struct", indent=0):
    """Recursively explore a MATLAB struct and print its contents."""
    prefix = "  " * indent
    
    if hasattr(obj, '_fieldnames'):
        print(f"{prefix}{name} (MATLAB struct):")
        for field in obj._fieldnames:
            val = getattr(obj, field)
            print(f"{prefix}  • {field}: {type(val).__name__}", end='')
            
            if isinstance(val, np.ndarray):
                print(f" {val.shape} {val.dtype}", end='')
                if val.size > 0 and hasattr(val.flat[0], '_fieldnames'):
                    print(" [array of structs]")
                    if indent < 2:  # Avoid too deep recursion
                        explore_struct(val.flat[0], f"{field}[0]", indent + 1)
                else:
                    print()
                    if val.size <= 5 and val.dtype.kind in ['U', 'S', 'O']:
                        for i, item in enumerate(val.flat):
                            print(f"{prefix}    [{i}] {item}")
            else:
                print()
    else:
        print(f"{prefix}{name}: {type(obj).__name__}")


def try_extract_channel_names(seizure):
    """
    Attempt to extract channel names from various possible locations.
    Returns (channel_names, source) or (None, None) if extraction fails.
    """
    print_section("ATTEMPTING TO EXTRACT CHANNEL NAMES", '-')
    
    # Strategy 1: Check 'chans' struct array FIRST (TUC-specific)
    # This is where the actual channel names are stored
    if hasattr(seizure, 'chans'):
        print("\n✓ Found 'chans' field")
        chans = seizure.chans
        print(f"  Type: {type(chans)}")
        
        if isinstance(chans, np.ndarray):
            print(f"  Shape: {chans.shape}")
            print(f"  Number of elements: {chans.size}")
            
            if chans.size > 0:
                first_chan = chans.flat[0] if chans.ndim > 0 else chans
                
                if hasattr(first_chan, '_fieldnames'):
                    print(f"  Fields in each channel struct: {first_chan._fieldnames}")
                    
                    # TUC-specific: Try 'selected' field first (this is where the 19 channels are)
                    if hasattr(first_chan, 'selected'):
                        selected = getattr(first_chan, 'selected')
                        if isinstance(selected, np.ndarray) and selected.size > 0:
                            try:
                                names = [str(item).strip() for item in selected.flat]
                                print(f"  ✅ Extracted {len(names)} channel names from 'selected' field!")
                                print(f"  Channels: {names}")
                                return names, "chans[0].selected field"
                            except Exception as e:
                                print(f"  ❌ Failed to extract from 'selected': {e}")
                    
                    # Try different possible field names for labels
                    for field in ['labels', 'Label', 'label', 'name', 'names', 'initial', 'type']:
                        if hasattr(first_chan, field):
                            val = getattr(first_chan, field)
                            print(f"  Found field '{field}': type={type(val)}")
                            
                            # Try to extract
                            if isinstance(val, np.ndarray) and val.size > 0:
                                try:
                                    # Check if it's string array
                                    if val.dtype.kind in ['U', 'S', 'O']:
                                        names = [str(item).strip() for item in val.flat]
                                        if len(names) > 0 and not names[0].isdigit():  # Avoid electrode IDs
                                            print(f"  ✅ Extracted {len(names)} channel names from '{field}' field!")
                                            print(f"  Channels: {names}")
                                            return names, f"chans[0].{field} field"
                                except Exception as e:
                                    print(f"  ❌ Failed to extract from '{field}': {e}")
    
    # Strategy 2: Check for 'loc_electrode' field (fallback - may contain electrode IDs)
    
    # Strategy 2: Check for 'loc_electrode' field (fallback - may contain electrode IDs)
    if hasattr(seizure, 'loc_electrode'):
        print("\n⚠️  Checking 'loc_electrode' field (may contain IDs instead of names)")
        loc = seizure.loc_electrode
        print(f"  Type: {type(loc)}")
        
        if isinstance(loc, np.ndarray):
            print(f"  Shape: {loc.shape}")
            print(f"  Dtype: {loc.dtype}")
            
            # Only use if it looks like string names, not numeric IDs
            if loc.dtype.kind in ['U', 'S', 'O']:
                try:
                    names = [str(item).strip() for item in loc.flat]
                    # Check if these look like channel names (contain letters)
                    if names and any(c.isalpha() for c in str(names[0])):
                        print(f"  ✅ Extracted {len(names)} channel names!")
                        print(f"  Channels: {names}")
                        return names, "loc_electrode field"
                    else:
                        print(f"  ⚠️  loc_electrode contains IDs/numbers, not channel names")
                        print(f"  Values: {names[:5]}...")
                except Exception as e:
                    print(f"  ❌ Extraction failed: {e}")
    
    # Strategy 3: Check for other common EEG field names
    for field_name in ['channel_labels', 'channel_names', 'electrodes', 'electrode_names']:
        if hasattr(seizure, field_name):
            print(f"\n✓ Found '{field_name}' field")
            val = getattr(seizure, field_name)
            print(f"  Type: {type(val)}")
            
            if isinstance(val, np.ndarray):
                try:
                    names = [str(item).strip() for item in val.flat]
                    print(f"  ✅ Extracted {len(names)} channel names!")
                    print(f"  Channels: {names}")
                    return names, field_name
                except Exception as e:
                    print(f"  ❌ Extraction failed: {e}")
    
    print("\n❌ Could not automatically extract channel names.")
    print("   Will use generic labels (Ch1, Ch2, ...) for now.")
    return None, None


# ============================================================================
# MAIN EXPLORATION
# ============================================================================

def main():
    print_section("TUC EPILEPSY DATASET - EXPLORATORY DATA ANALYSIS")
    print(f"\nStarting exploration at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    exploration_log = {
        'exploration_date': datetime.now().isoformat(),
        'matlab_file': str(MATLAB_FILE),
        'discoveries': []
    }
    
    # ========================================================================
    # STEP 1: LOAD AND EXAMINE FILE STRUCTURE
    # ========================================================================
    print_section("STEP 1: LOADING MATLAB FILE")
    
    print(f"\nAttempting to load: {MATLAB_FILE}")
    
    try:
        data = loadmat(MATLAB_FILE, squeeze_me=True, struct_as_record=False)
        print("✅ File loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Examine top-level structure
    print("\n" + "-" * 80)
    print("TOP-LEVEL VARIABLES IN FILE:")
    print("-" * 80)
    
    for key in data.keys():
        if not key.startswith('__'):
            val = data[key]
            print(f"  • {key}: {type(val).__name__}", end='')
            if isinstance(val, np.ndarray):
                print(f" {val.shape}")
            else:
                print()
    
    # Find the main data structure
    main_var = None
    for key in data.keys():
        if not key.startswith('__'):
            main_var = key
            break
    
    if main_var is None:
        print("\n❌ No data variable found!")
        return
    
    seizure = data[main_var]
    print(f"\n✓ Main data variable: '{main_var}'")
    
    exploration_log['discoveries'].append({
        'step': 1,
        'finding': 'Main data structure',
        'details': {'variable_name': main_var, 'type': type(seizure).__name__}
    })
    
    # ========================================================================
    # STEP 2: EXPLORE MAIN STRUCTURE
    # ========================================================================
    print_section("STEP 2: EXPLORING MAIN DATA STRUCTURE")
    
    if hasattr(seizure, '_fieldnames'):
        print(f"\n'{main_var}' is a MATLAB struct with the following fields:")
        print("-" * 80)
        
        for field in seizure._fieldnames:
            val = getattr(seizure, field)
            print(f"\n  {field}:")
            print(f"    Type: {type(val).__name__}")
            
            if isinstance(val, np.ndarray):
                print(f"    Shape: {val.shape}")
                print(f"    Dtype: {val.dtype}")
                
                # Show sample content for small arrays
                if val.size <= 10 and val.dtype.kind in ['U', 'S', 'O', 'i', 'f']:
                    print(f"    Content: {val}")
                
                # Check if it's an array of structs
                if val.size > 0 and hasattr(val.flat[0], '_fieldnames'):
                    print(f"    → Array of {val.size} MATLAB structs")
                    print(f"    → Each struct has fields: {val.flat[0]._fieldnames}")
        
        exploration_log['discoveries'].append({
            'step': 2,
            'finding': 'Main structure fields',
            'details': {'fields': seizure._fieldnames}
        })
    
    # ========================================================================
    # STEP 3: DISCOVER CHANNEL NAMES
    # ========================================================================
    print_section("STEP 3: DISCOVERING CHANNEL NAMES")
    
    channel_names, channel_source = try_extract_channel_names(seizure)
    
    # If extraction failed, we'll use the actual data to determine number of channels
    # and create generic names
    if channel_names is None and hasattr(seizure, 'x'):
        x = seizure.x
        if isinstance(x, np.ndarray) and x.size > 0:
            first_subject = x[0] if x.ndim == 1 else x.flat[0]
            if isinstance(first_subject, np.ndarray):
                n_channels = first_subject.shape[1] if first_subject.ndim == 2 else 1
                channel_names = [f"Ch{i+1}" for i in range(n_channels)]
                channel_source = "Generated from data dimensions"
                print(f"\n  → Created {n_channels} generic channel names: {channel_names}")
    
    if channel_names:
        exploration_log['discoveries'].append({
            'step': 3,
            'finding': 'Channel names',
            'details': {
                'source': channel_source,
                'n_channels': len(channel_names),
                'names': channel_names
            }
        })
    
    # ========================================================================
    # STEP 4: ANALYZE SUBJECT DATA
    # ========================================================================
    print_section("STEP 4: ANALYZING SUBJECT DATA")
    
    if not hasattr(seizure, 'x'):
        print("❌ No 'x' field found (expected to contain EEG data)")
        return
    
    x = seizure.x
    print(f"\nFound data field 'x':")
    print(f"  Type: {type(x).__name__}")
    print(f"  Shape: {x.shape}")
    print(f"  Number of subjects: {len(x)}")
    
    # Try to infer sampling rate from filename or data
    fs = None
    
    # From filename pattern
    if '256' in str(MATLAB_FILE) or 'fs256' in str(MATLAB_FILE).lower():
        fs = 256
        print(f"\n✓ Inferred sampling rate from filename: {fs} Hz")
    
    # If not in filename, ask user or use default
    if fs is None:
        print("\n⚠️  Could not infer sampling rate from filename.")
        print("   Common EEG sampling rates: 128, 200, 250, 256, 500, 512 Hz")
        fs = 256  # Most common for clinical EEG
        print(f"   Using default: {fs} Hz (verify this is correct!)")
    
    # Analyze each subject
    print(f"\n{'#':<5} {'Samples':<12} {'Channels':<10} {'Duration':<15} {'Data Range'}")
    print("-" * 80)
    
    subject_stats = []
    
    for i in range(len(x)):
        subj_data = x[i]
        
        if isinstance(subj_data, np.ndarray) and subj_data.ndim == 2:
            n_samples, n_channels = subj_data.shape
            duration_sec = n_samples / fs
            duration_min = duration_sec / 60
            data_min = subj_data.min()
            data_max = subj_data.max()
            
            subject_stats.append({
                'subject_id': i + 1,
                'n_samples': int(n_samples),
                'n_channels': int(n_channels),
                'duration_sec': float(duration_sec),
                'data_min': float(data_min),
                'data_max': float(data_max),
                'data_mean': float(subj_data.mean()),
                'data_std': float(subj_data.std())
            })
            
            print(f"{i+1:<5} {n_samples:<12} {n_channels:<10} "
                  f"{duration_min:>6.2f} min     [{data_min:>8.2f}, {data_max:>8.2f}]")
    
    # Summary statistics
    if subject_stats:
        durations = [s['duration_sec'] / 60 for s in subject_stats]
        channels = [s['n_channels'] for s in subject_stats]
        
        print("\n" + "-" * 80)
        print("SUMMARY:")
        print("-" * 80)
        print(f"  Total subjects: {len(subject_stats)}")
        print(f"  Channels per subject: {set(channels)}")
        print(f"  Duration range: {min(durations):.2f} - {max(durations):.2f} minutes")
        print(f"  Mean duration: {np.mean(durations):.2f} minutes")
        print(f"  Total recording time: {sum(durations):.2f} min ({sum(durations)/60:.2f} hours)")
        
        exploration_log['discoveries'].append({
            'step': 4,
            'finding': 'Subject data analysis',
            'details': {
                'n_subjects': len(subject_stats),
                'sampling_rate_hz': fs,
                'n_channels': list(set(channels))[0] if len(set(channels)) == 1 else channels,
                'duration_stats': {
                    'min_min': float(min(durations)),
                    'max_min': float(max(durations)),
                    'mean_min': float(np.mean(durations)),
                    'total_min': float(sum(durations))
                }
            }
        })
    
    # ========================================================================
    # STEP 5: DISCOVER ANNOTATIONS/LABELS
    # ========================================================================
    print_section("STEP 5: SEARCHING FOR ANNOTATIONS")
    
    annotation_field = None
    
    # Common field names for annotations
    possible_annotation_fields = ['annotation', 'annotations', 'labels', 'markers', 
                                   'events', 'seizure_times', 'onset']
    
    for field_name in possible_annotation_fields:
        if hasattr(seizure, field_name):
            print(f"\n✓ Found annotation field: '{field_name}'")
            annotation_field = field_name
            break
    
    if annotation_field:
        annotations = getattr(seizure, annotation_field)
        print(f"  Type: {type(annotations).__name__}")
        
        if isinstance(annotations, np.ndarray):
            print(f"  Shape: {annotations.shape}")
            print(f"  Number of annotations: {len(annotations)}")
            
            # Try to understand annotation format
            print("\n  Examining first few annotations:")
            for i in range(min(3, len(annotations))):
                ann = annotations[i]
                print(f"    Subject {i+1}: {ann} (type: {type(ann).__name__})")
            
            # Assume annotations are [start, end] samples
            print("\n  Assuming annotations are [start_sample, end_sample]:")
            print(f"  {'Subject':<10} {'Start':<12} {'End':<12} {'Duration (sec)':<15}")
            print("  " + "-" * 50)
            
            seizure_stats = []
            
            for i in range(len(annotations)):
                ann = annotations[i]
                if hasattr(ann, '__len__') and len(ann) >= 2:
                    start_sample = int(ann[0])
                    end_sample = int(ann[1])
                    duration_sec = (end_sample - start_sample) / fs
                    
                    seizure_stats.append({
                        'subject_id': i + 1,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'duration_sec': float(duration_sec)
                    })
                    
                    print(f"  {i+1:<10} {start_sample:<12} {end_sample:<12} {duration_sec:<15.1f}")
            
            if seizure_stats:
                seizure_durations = [s['duration_sec'] for s in seizure_stats]
                
                print("\n  Annotation Summary:")
                print(f"    Min duration: {min(seizure_durations):.1f} sec")
                print(f"    Max duration: {max(seizure_durations):.1f} sec")
                print(f"    Mean duration: {np.mean(seizure_durations):.1f} sec")
                
                exploration_log['discoveries'].append({
                    'step': 5,
                    'finding': 'Seizure annotations',
                    'details': {
                        'annotation_field': annotation_field,
                        'n_annotations': len(seizure_stats),
                        'duration_stats': {
                            'min_sec': float(min(seizure_durations)),
                            'max_sec': float(max(seizure_durations)),
                            'mean_sec': float(np.mean(seizure_durations))
                        }
                    }
                })
    else:
        print("\n❌ No annotation field found")
        print("   Searched for: " + ", ".join(possible_annotation_fields))
    
    # ========================================================================
    # STEP 6: CHECK FOR ADDITIONAL METADATA
    # ========================================================================
    print_section("STEP 6: SEARCHING FOR METADATA")
    
    metadata_fields = ['info', 'patient_id', 'subject_info', 'demographics']
    
    for field_name in metadata_fields:
        if hasattr(seizure, field_name):
            print(f"\n✓ Found metadata field: '{field_name}'")
            val = getattr(seizure, field_name)
            print(f"  Type: {type(val).__name__}")
            
            if isinstance(val, np.ndarray):
                print(f"  Shape: {val.shape}")
                if val.size <= 10:
                    print(f"  Content: {val}")
                else:
                    print(f"  First 5 entries: {val[:5]}")
    
    # ========================================================================
    # STEP 7: DATA QUALITY CHECK
    # ========================================================================
    print_section("STEP 7: DATA QUALITY CHECK")
    
    # Check first subject in detail
    if len(x) > 0:
        subj_data = x[0]
        
        print(f"\nAnalyzing first subject in detail:")
        print(f"  Shape: {subj_data.shape}")
        print(f"  Dtype: {subj_data.dtype}")
        print(f"  Min: {subj_data.min():.4f}")
        print(f"  Max: {subj_data.max():.4f}")
        print(f"  Mean: {subj_data.mean():.4f}")
        print(f"  Std: {subj_data.std():.4f}")
        
        # Check for bad values
        has_nan = np.isnan(subj_data).any()
        has_inf = np.isinf(subj_data).any()
        
        print(f"\n  Data quality:")
        print(f"    NaN values: {'❌ FOUND' if has_nan else '✅ None'}")
        print(f"    Inf values: {'❌ FOUND' if has_inf else '✅ None'}")
        
        # Check channel variance
        channel_vars = np.var(subj_data, axis=0)
        low_var_channels = np.where(channel_vars < 0.01)[0]
        
        print(f"\n  Channel variance:")
        for i, var in enumerate(channel_vars):
            ch_name = channel_names[i] if channel_names and i < len(channel_names) else f"Ch{i+1}"
            status = "⚠️ LOW" if var < 0.01 else "✓"
            print(f"    {ch_name}: {var:.4f} {status}")
        
        if len(low_var_channels) > 0:
            print(f"\n  ⚠️  Warning: {len(low_var_channels)} channels with very low variance")
        else:
            print(f"\n  ✅ All channels have reasonable variance")
    
    # ========================================================================
    # STEP 8: FREQUENCY CONTENT ANALYSIS
    # ========================================================================
    print_section("STEP 8: FREQUENCY CONTENT ANALYSIS")
    
    # Try to infer filter settings from filename
    filter_info = {}
    filename_lower = str(MATLAB_FILE).lower()
    
    # Look for bandwidth pattern like "bw_0.5_45"
    import re
    bw_match = re.search(r'bw[_\s]*([\d.]+)[_\s]*([\d.]+)', filename_lower)
    if bw_match:
        filter_info['low'] = float(bw_match.group(1))
        filter_info['high'] = float(bw_match.group(2))
        print(f"\n✓ Inferred filter band from filename: {filter_info['low']}-{filter_info['high']} Hz")
    
    # Look for scaling factor like "sc_1.8"
    sc_match = re.search(r'sc[_\s]*([\d.]+)', filename_lower)
    if sc_match:
        filter_info['scaling'] = float(sc_match.group(1))
        print(f"✓ Inferred scaling factor from filename: {filter_info['scaling']}")
    
    # Compute PSD to verify
    if len(x) > 0:
        subj_data = x[0]
        channel_idx = 0  # Analyze first channel
        
        print(f"\nComputing Power Spectral Density (first channel)...")
        freqs, psd = welch(subj_data[:, channel_idx], fs=fs, nperseg=1024)
        
        # Calculate power in different frequency bands
        total_power = np.sum(psd)
        
        bands = {
            'DC-0.5 Hz': (0, 0.5),
            '0.5-4 Hz (Delta)': (0.5, 4),
            '4-8 Hz (Theta)': (4, 8),
            '8-13 Hz (Alpha)': (8, 13),
            '13-30 Hz (Beta)': (13, 30),
            '30-50 Hz (Gamma)': (30, 50),
            '50+ Hz': (50, fs/2)
        }
        
        print(f"\n  Power distribution:")
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            power = np.sum(psd[mask])
            pct = 100 * power / total_power
            print(f"    {band_name:20s}: {pct:5.1f}%")
        
        # Check if filtering was applied
        if 'low' in filter_info and 'high' in filter_info:
            below_cutoff = (freqs < filter_info['low'])
            above_cutoff = (freqs > filter_info['high'])
            
            power_below = 100 * np.sum(psd[below_cutoff]) / total_power
            power_above = 100 * np.sum(psd[above_cutoff]) / total_power
            
            print(f"\n  Verification of {filter_info['low']}-{filter_info['high']} Hz filter:")
            print(f"    Power < {filter_info['low']} Hz: {power_below:.2f}%")
            print(f"    Power > {filter_info['high']} Hz: {power_above:.2f}%")
            
            if power_below < 1.0 and power_above < 1.0:
                print(f"    ✅ Filtering confirmed!")
            else:
                print(f"    ⚠️  Filter may not be applied correctly")
        
        exploration_log['discoveries'].append({
            'step': 8,
            'finding': 'Frequency content',
            'details': filter_info
        })
    
    # ========================================================================
    # STEP 9: CREATE VISUALIZATIONS
    # ========================================================================
    print_section("STEP 9: CREATING VISUALIZATIONS")
    
    # Determine how many channels to plot
    n_channels_to_plot = min(5, subj_data.shape[1])
    
    # Plot 1: EEG Overview
    print("\n  Creating EEG overview plot...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    subj_idx = 0
    subj_data = x[subj_idx]
    n_samples = subj_data.shape[0]
    time = np.arange(n_samples) / fs
    
    # Get seizure times if available
    if annotation_field and len(annotations) > subj_idx:
        ann = annotations[subj_idx]
        if hasattr(ann, '__len__') and len(ann) >= 2:
            seizure_start = ann[0] / fs
            seizure_end = ann[1] / fs
        else:
            seizure_start = None
            seizure_end = None
    else:
        seizure_start = None
        seizure_end = None
    
    # Panel 1: Full recording
    ax = axes[0]
    for ch in range(n_channels_to_plot):
        signal = subj_data[:, ch]
        signal_norm = (signal - signal.mean()) / (signal.std() + 1e-10) + ch * 5
        ch_label = channel_names[ch] if channel_names and ch < len(channel_names) else f"Ch{ch+1}"
        ax.plot(time, signal_norm, linewidth=0.3, alpha=0.8, label=ch_label)
    
    if seizure_start and seizure_end:
        ax.axvspan(seizure_start, seizure_end, alpha=0.3, color='red', label='Seizure')
    
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Channels (normalized)', fontsize=10)
    ax.set_title(f'Full Recording - Subject 1', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel 2: Zoomed view (first 60 seconds or around seizure if available)
    ax = axes[1]
    
    if seizure_start and seizure_end:
        # Zoom on seizure
        margin = 10  # seconds
        zoom_start_sec = max(0, seizure_start - margin)
        zoom_end_sec = min(time[-1], seizure_end + margin)
        zoom_start = int(zoom_start_sec * fs)
        zoom_end = int(zoom_end_sec * fs)
        zoom_title = f'Zoomed on Seizure (±{margin}s)'
    else:
        # First 60 seconds
        zoom_start = 0
        zoom_end = min(n_samples, int(60 * fs))
        zoom_title = 'First 60 Seconds'
    
    time_zoom = np.arange(zoom_start, zoom_end) / fs
    
    for ch in range(n_channels_to_plot):
        signal = subj_data[zoom_start:zoom_end, ch]
        signal_norm = (signal - signal.mean()) / (signal.std() + 1e-10) + ch * 5
        ch_label = channel_names[ch] if channel_names and ch < len(channel_names) else f"Ch{ch+1}"
        ax.plot(time_zoom, signal_norm, linewidth=0.5, alpha=0.8, label=ch_label)
    
    if seizure_start and seizure_end:
        ax.axvspan(seizure_start, seizure_end, alpha=0.3, color='red')
        ax.axvline(seizure_start, color='red', linestyle='--', linewidth=2)
        ax.axvline(seizure_end, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Channels (normalized)', fontsize=10)
    ax.set_title(zoom_title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel 3: Power Spectral Density
    ax = axes[2]
    for ch in range(n_channels_to_plot):
        freqs, psd = welch(subj_data[:, ch], fs=fs, nperseg=1024)
        ch_label = channel_names[ch] if channel_names and ch < len(channel_names) else f"Ch{ch+1}"
        ax.plot(freqs, 10*np.log10(psd + 1e-20), alpha=0.7, linewidth=1, label=ch_label)
    
    if 'low' in filter_info and 'high' in filter_info:
        ax.axvspan(filter_info['low'], filter_info['high'], alpha=0.2, color='green',
                  label=f"Filter Band ({filter_info['low']}-{filter_info['high']} Hz)")
        ax.axvline(filter_info['low'], color='green', linestyle='--', alpha=0.7)
        ax.axvline(filter_info['high'], color='green', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power (dB)', fontsize=10)
    ax.set_title('Power Spectral Density', fontsize=12, fontweight='bold')
    ax.set_xlim([0, min(60, fs/2)])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eeg_overview.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: eeg_overview.png")
    
    # Plot 2: Dataset Statistics
    print("  Creating dataset statistics plot...")
    
    if seizure_stats:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Recording durations
        ax = axes[0]
        ax.bar(range(len(durations)), durations, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axhline(np.mean(durations), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(durations):.1f} min')
        ax.set_xlabel('Subject Index', fontsize=11)
        ax.set_ylabel('Duration (minutes)', fontsize=11)
        ax.set_title('Recording Duration per Subject', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # Seizure durations
        ax = axes[1]
        seizure_durations = [s['duration_sec'] for s in seizure_stats]
        ax.bar(range(len(seizure_durations)), seizure_durations, alpha=0.7, 
              color='crimson', edgecolor='black')
        ax.axhline(np.mean(seizure_durations), color='blue', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(seizure_durations):.1f} sec')
        ax.set_xlabel('Subject Index', fontsize=11)
        ax.set_ylabel('Seizure Duration (seconds)', fontsize=11)
        ax.set_title('Seizure Duration per Subject', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'dataset_statistics.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Saved: dataset_statistics.png")
    
    # Plot 3: Comprehensive Quality Report
    print("  Creating comprehensive quality report...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Panel 1: Time series
    ax1 = fig.add_subplot(gs[0, :])
    for ch in range(n_channels_to_plot):
        signal = subj_data[:, ch]
        signal_norm = (signal - signal.mean()) / (signal.std() + 1e-10) + ch * 5
        ch_label = channel_names[ch] if channel_names and ch < len(channel_names) else f"Ch{ch+1}"
        ax1.plot(time, signal_norm, linewidth=0.4, alpha=0.8, label=ch_label)
    
    if seizure_start and seizure_end:
        ax1.axvspan(seizure_start, seizure_end, alpha=0.2, color='red')
    
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Normalized Amplitude', fontsize=11)
    ax1.set_title('Time Series - First Subject', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Panel 2: PSD - Full spectrum
    ax2 = fig.add_subplot(gs[1, 0])
    for ch in range(subj_data.shape[1]):
        freqs, psd = welch(subj_data[:, ch], fs=fs, nperseg=1024)
        ax2.plot(freqs, 10*np.log10(psd + 1e-20), alpha=0.3, linewidth=0.5, color='steelblue')
    
    if 'low' in filter_info and 'high' in filter_info:
        ax2.axvspan(filter_info['low'], filter_info['high'], alpha=0.2, color='green',
                   label=f"Passband ({filter_info['low']}-{filter_info['high']} Hz)")
    
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Power (dB)', fontsize=11)
    ax2.set_title('Power Spectral Density - All Channels', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, fs/2])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: PSD - Passband zoom
    ax3 = fig.add_subplot(gs[1, 1])
    for ch in range(subj_data.shape[1]):
        freqs, psd = welch(subj_data[:, ch], fs=fs, nperseg=1024)
        ax3.plot(freqs, 10*np.log10(psd + 1e-20), alpha=0.3, linewidth=0.8, color='steelblue')
    
    if 'low' in filter_info and 'high' in filter_info:
        ax3.set_xlim([filter_info['low'], filter_info['high']])
        ax3.set_title(f"PSD - Passband Detail ({filter_info['low']}-{filter_info['high']} Hz)",
                     fontsize=13, fontweight='bold')
    else:
        ax3.set_xlim([0, 50])
        ax3.set_title('PSD - Low Frequency Detail (0-50 Hz)', fontsize=13, fontweight='bold')
    
    ax3.set_xlabel('Frequency (Hz)', fontsize=11)
    ax3.set_ylabel('Power (dB)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Channel variance
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(range(len(channel_vars)), channel_vars, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.set_xlabel('Channel', fontsize=11)
    ax4.set_ylabel('Variance', fontsize=11)
    ax4.set_title('Channel-wise Variance', fontsize=13, fontweight='bold')
    
    if channel_names and len(channel_names) == len(channel_vars):
        ax4.set_xticks(range(len(channel_vars)))
        ax4.set_xticklabels([ch[:4] for ch in channel_names], rotation=45, ha='right', fontsize=8)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: Amplitude distribution
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(subj_data.flatten(), bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax5.axvline(subj_data.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {subj_data.mean():.2f}')
    ax5.axvline(subj_data.mean() + subj_data.std(), color='orange', linestyle='--', linewidth=2,
               label=f'±1 SD: {subj_data.std():.2f}')
    ax5.axvline(subj_data.mean() - subj_data.std(), color='orange', linestyle='--', linewidth=2)
    ax5.set_xlabel('Amplitude', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Amplitude Distribution', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3, axis='y')
    
    # Panel 6: Annotated segment (if available)
    ax6 = fig.add_subplot(gs[3, :])
    
    if seizure_start and seizure_end:
        margin = 30
        seg_start = max(0, int((seizure_start - margin) * fs))
        seg_end = min(n_samples, int((seizure_end + margin) * fs))
        time_seg = np.arange(seg_start, seg_end) / fs
        
        for ch in range(n_channels_to_plot):
            signal = subj_data[seg_start:seg_end, ch]
            signal_norm = (signal - signal.mean()) / (signal.std() + 1e-10) + ch * 5
            ch_label = channel_names[ch] if channel_names and ch < len(channel_names) else f"Ch{ch+1}"
            ax6.plot(time_seg, signal_norm, linewidth=0.5, alpha=0.8, label=ch_label)
        
        ax6.axvspan(seizure_start, seizure_end, alpha=0.3, color='red', label='Annotated Period')
        ax6.axvline(seizure_start, color='red', linestyle='--', linewidth=2)
        ax6.axvline(seizure_end, color='red', linestyle='--', linewidth=2)
        ax6.set_title(f'Annotated Event Period (±{margin}s)', fontsize=13, fontweight='bold')
    else:
        # Just show first minute if no annotations
        seg_end = min(n_samples, int(60 * fs))
        time_seg = np.arange(seg_end) / fs
        
        for ch in range(n_channels_to_plot):
            signal = subj_data[:seg_end, ch]
            signal_norm = (signal - signal.mean()) / (signal.std() + 1e-10) + ch * 5
            ch_label = channel_names[ch] if channel_names and ch < len(channel_names) else f"Ch{ch+1}"
            ax6.plot(time_seg, signal_norm, linewidth=0.5, alpha=0.8, label=ch_label)
        
        ax6.set_title('First 60 Seconds', fontsize=13, fontweight='bold')
    
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Channels (normalized)', fontsize=11)
    ax6.legend(loc='upper right', fontsize=9, ncol=2)
    ax6.grid(alpha=0.3)
    
    plt.suptitle('Dataset Quality Report', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(OUTPUT_DIR / 'quality_report.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: quality_report.png")
    
    # ========================================================================
    # STEP 10: SAVE EXPLORATION REPORT
    # ========================================================================
    print_section("STEP 10: SAVING EXPLORATION REPORT")
    
    # Create comprehensive report
    report = {
        'exploration_metadata': {
            'date': datetime.now().isoformat(),
            'matlab_file': str(MATLAB_FILE),
            'output_directory': str(OUTPUT_DIR)
        },
        'dataset_info': {
            'n_subjects': len(subject_stats),
            'sampling_rate_hz': fs,
            'n_channels': list(set(channels))[0] if len(set(channels)) == 1 else 'variable',
            'channel_names': channel_names if channel_names else None,
            'channel_source': channel_source if channel_names else None
        },
        'preprocessing': filter_info,
        'recording_statistics': {
            'duration_min': {
                'min': float(min(durations)),
                'max': float(max(durations)),
                'mean': float(np.mean(durations)),
                'total': float(sum(durations))
            }
        },
        'subjects': subject_stats,
        'discoveries': exploration_log['discoveries']
    }
    
    # Add seizure info if available
    if seizure_stats:
        report['seizure_annotations'] = {
            'annotation_field': annotation_field,
            'n_annotations': len(seizure_stats),
            'duration_sec': {
                'min': float(min(seizure_durations)),
                'max': float(max(seizure_durations)),
                'mean': float(np.mean(seizure_durations))
            },
            'annotations': seizure_stats
        }
    
    # Save JSON
    with open(OUTPUT_DIR / 'exploration_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  ✅ Saved: exploration_report.json")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_section("EXPLORATION COMPLETE")
    
    print(f"""
📊 DATASET DISCOVERED
{'='*80}
File: {Path(MATLAB_FILE).name}

📁 Structure:
  • Main variable: '{main_var}'
  • Fields found: {', '.join(seizure._fieldnames) if hasattr(seizure, '_fieldnames') else 'N/A'}

🧠 EEG Data:
  • Subjects: {len(subject_stats)}
  • Channels: {list(set(channels))[0] if len(set(channels)) == 1 else channels}
  • Channel names: {'✅ Extracted' if channel_names and channel_source != 'Generated from data dimensions' else '⚠️  Generated (not found in file)'}
  • Sampling rate: {fs} Hz

📏 Recording Statistics:
  • Duration range: {min(durations):.2f} - {max(durations):.2f} minutes
  • Mean duration: {np.mean(durations):.2f} minutes
  • Total time: {sum(durations):.2f} minutes ({sum(durations)/60:.2f} hours)

""")
    
    if seizure_stats:
        print(f"""🔴 Annotations Found:
  • Field: '{annotation_field}'
  • Number: {len(seizure_stats)}
  • Duration range: {min(seizure_durations):.1f} - {max(seizure_durations):.1f} seconds
  • Mean duration: {np.mean(seizure_durations):.1f} seconds
""")
    
    if filter_info:
        print(f"""⚙️  Preprocessing (from filename):
  • Filter band: {filter_info.get('low', '?')}-{filter_info.get('high', '?')} Hz
  • Scaling: {filter_info.get('scaling', '?')}
  • Verified: {'✅ Yes' if 'low' in filter_info else '⚠️  Check plots'}
""")
    
    print(f"""
📁 Output Files:
{'='*80}
All files saved to: {OUTPUT_DIR}

  • exploration_report.json    - Complete metadata and discoveries
  • eeg_overview.png            - EEG time series visualization
  • dataset_statistics.png      - Duration and annotation statistics
  • quality_report.png          - Comprehensive quality control

🎯 Next Steps:
{'='*80}
1. Review the quality_report.png to verify preprocessing
2. Check exploration_report.json for complete details
3. If everything looks good, proceed with your analysis pipeline:
   - Create epochs (4-second windows)
   - Compute connectivity (DTF/PDC)
   - Run GNN models

{'='*80}
✅ Exploration completed successfully!
{'='*80}
""")


if __name__ == "__main__":
    main()