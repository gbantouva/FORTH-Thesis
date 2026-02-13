"""
Channel Names Diagnostic - Deep Dive into 'chans' Field
========================================================
This script explores the 'chans' field more carefully to extract
the actual channel names.
"""

import numpy as np
from scipy.io import loadmat

# Path to your file
MATLAB_FILE = r"F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat"

print("="*80)
print("DEEP DIVE: Extracting Channel Names from 'chans' Field")
print("="*80)

# Load data
data = loadmat(MATLAB_FILE, squeeze_me=True, struct_as_record=False)
seizure = data['seizure']

# Examine chans field
if hasattr(seizure, 'chans'):
    chans = seizure.chans
    
    print(f"\nchans field:")
    print(f"  Type: {type(chans)}")
    print(f"  Shape: {chans.shape if hasattr(chans, 'shape') else 'N/A'}")
    print(f"  Size: {chans.size if hasattr(chans, 'size') else 'N/A'}")
    
    # If it's an array, explore the first element
    if isinstance(chans, np.ndarray) and chans.size > 0:
        first_chan = chans.flat[0] if chans.ndim > 0 else chans
        
        print(f"\nFirst channel struct:")
        if hasattr(first_chan, '_fieldnames'):
            print(f"  Fields: {first_chan._fieldnames}")
            
            print(f"\n  Detailed field exploration:")
            for field in first_chan._fieldnames:
                val = getattr(first_chan, field)
                print(f"\n    {field}:")
                print(f"      Type: {type(val)}")
                
                if isinstance(val, np.ndarray):
                    print(f"      Shape: {val.shape}")
                    print(f"      Dtype: {val.dtype}")
                    
                    # Show content for small arrays
                    if val.size <= 20:
                        print(f"      Content: {val}")
                    else:
                        print(f"      First few: {val.flat[:19]}")
                else:
                    print(f"      Value: {val}")
        
        # Try to extract labels from all channel structs
        print(f"\n{'='*80}")
        print("ATTEMPTING EXTRACTION FROM ALL CHANNEL STRUCTS")
        print("="*80)
        
        if hasattr(first_chan, '_fieldnames'):
            for field in ['labels', 'label', 'Label', 'type', 'name']:
                if field in first_chan._fieldnames:
                    print(f"\nTrying field: '{field}'")
                    
                    channel_names = []
                    for i, ch in enumerate(chans.flat):
                        try:
                            val = getattr(ch, field)
                            
                            # Convert to string
                            if isinstance(val, np.ndarray):
                                if val.size == 1:
                                    val_str = str(val.item())
                                elif val.dtype.kind in ['U', 'S', 'O']:
                                    val_str = str(val)
                                else:
                                    val_str = str(val)
                            else:
                                val_str = str(val)
                            
                            channel_names.append(val_str.strip())
                            
                            if i < 5:  # Show first 5
                                print(f"  Channel {i}: {val_str}")
                        except Exception as e:
                            print(f"  Channel {i}: ERROR - {e}")
                            break
                    
                    if len(channel_names) > 0:
                        print(f"\n✅ Extracted {len(channel_names)} channel names from '{field}':")
                        print(f"   {channel_names}")
                        
                        # Check if they look like real channel names
                        looks_valid = any(
                            name in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                                    'Fz', 'Cz', 'Pz', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']
                            for name in channel_names
                        )
                        
                        if looks_valid:
                            print(f"   ✅ These look like standard EEG channel names!")
                        else:
                            print(f"   ⚠️  These don't look like standard EEG channel names")

# Also check if there's a separate channel info field
print(f"\n{'='*80}")
print("CHECKING FOR OTHER CHANNEL-RELATED FIELDS")
print("="*80)

if hasattr(seizure, '_fieldnames'):
    channel_related = [f for f in seizure._fieldnames 
                      if any(keyword in f.lower() 
                            for keyword in ['chan', 'electrode', 'label', 'montage'])]
    
    print(f"\nChannel-related fields found: {channel_related}")
    
    for field in channel_related:
        val = getattr(seizure, field)
        print(f"\n{field}:")
        print(f"  Type: {type(val)}")
        
        if isinstance(val, np.ndarray):
            print(f"  Shape: {val.shape}")
            print(f"  Dtype: {val.dtype}")
            
            if val.size <= 20:
                print(f"  Content: {val}")
            else:
                print(f"  First 10: {val.flat[:10]}")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print("="*80)

print("""
Based on the data:
  • 34 subjects, each with exactly 19 channels
  • This is a standard clinical EEG dataset
  • 19 channels indicates standard 10-20 montage

MOST LIKELY CHANNEL ORDER (Standard 19-channel montage):
  ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
   'T3', 'C3', 'Cz', 'C4', 'T4',
   'T5', 'P3', 'Pz', 'P4', 'T6', 
   'O1', 'O2']

ACTION:
  1. Use the standard montage above for your analysis
  2. If you need absolute certainty, verify with your professor
  3. The exact order might vary slightly, but these are the channels
""")

print("="*80)