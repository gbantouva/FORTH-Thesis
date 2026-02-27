"""
Step 5 - Build Graph Dataset for GCN
======================================
For each subject, for each valid epoch:
  - Node features: Hjorth parameters (3) + band power (5) = 8 features per node
  - Edges: PDC integrated band (averaged to make undirected, fully connected weighted)
  - Label: 0 = control (first 2 min, far from seizure), 1 = ictal (during seizure)
  - DROPPED: pre-ictal epochs between 2min mark and seizure onset (ambiguous zone)
  - DROPPED: all post-ictal epochs (after seizure end)

Class balance strategy (per subject):
  - Ictal epochs: ALL ictal epochs for that subject
  - Control epochs: match the number of ictal epochs exactly
    → taken from the START of the recording (furthest from seizure)
  - This gives perfect 50/50 balance within each subject
  - Prevents accuracy inflation from majority-class dominance

Why match per subject rather than globally?
  - Seizure lengths vary hugely (4 to 43 ictal epochs)
  - A fixed 30-control-epoch window would create 88% majority class
    for short seizures, making accuracy a misleading metric
  - Matching ensures F1, sensitivity, specificity are meaningful

Output:
  - data/graphs/subject_XX_graphs.pt   (list of PyG Data objects)
  - data/graphs/dataset_info.json      (summary)

Usage:
  python step5_build_graphs.py \
      --epochs_dir preprocessed_epochs \
      --connectivity_dir connectivity \
      --output_dir data/graphs \
      --control_minutes 2.0
"""

import argparse
import json
import numpy as np
from pathlib import Path
import torch
from torch_geometric.data import Data
from scipy.signal import welch
from tqdm import tqdm

# ============================================================================
# CONSTANTS
# ============================================================================

FS = 256
EPOCH_SAMPLES = 1024   # 4s * 256 Hz
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2']
N_CHANNELS = 19

# Frequency bands for band power (Hz)
BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 15.0),
    'beta':  (15.0, 30.0),
    'gamma': (30.0, 45.0),
}

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def hjorth_parameters(signal_1d):
    """
    Compute Hjorth Activity, Mobility, Complexity for a 1D signal.

    Activity   = variance of signal
    Mobility   = sqrt(var(diff) / var(signal))
    Complexity = mobility(diff) / mobility(signal)
    """
    diff1 = np.diff(signal_1d)
    diff2 = np.diff(diff1)

    var0  = np.var(signal_1d)
    var1  = np.var(diff1)
    var2  = np.var(diff2)

    activity   = var0
    mobility   = np.sqrt(var1 / (var0 + 1e-12))
    complexity = np.sqrt(var2 / (var1 + 1e-12)) / (mobility + 1e-12)

    return activity, mobility, complexity


def band_power(signal_1d, fs=FS):
    """
    Compute average power in each of the 5 frequency bands using Welch's method.
    Returns a list of 5 values (one per band).
    """
    freqs, psd = welch(signal_1d, fs=fs, nperseg=min(256, len(signal_1d)))
    powers = []
    for band_name, (lo, hi) in BANDS.items():
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        powers.append(np.mean(psd[idx]) if idx.any() else 0.0)
    return powers


def extract_node_features(epoch):
    """
    Extract 8 node features per channel from a single epoch.

    Parameters:
    -----------
    epoch : np.ndarray  shape (19, 1024)

    Returns:
    --------
    features : np.ndarray  shape (19, 8)
        [activity, mobility, complexity, delta_power, theta_power,
         alpha_power, beta_power, gamma_power]
    """
    features = np.zeros((N_CHANNELS, 8), dtype=np.float32)

    for ch in range(N_CHANNELS):
        sig = epoch[ch]  # (1024,)

        act, mob, comp = hjorth_parameters(sig)
        bp = band_power(sig)

        features[ch, 0] = act
        features[ch, 1] = mob
        features[ch, 2] = comp
        features[ch, 3:8] = bp

    return features


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_edge_index_and_weights(pdc_matrix):
    """
    Build fully connected graph edge list from PDC matrix.

    PDC is directed (i→j), but GCN needs undirected edges.
    We symmetrise by averaging: w(i,j) = (PDC[i,j] + PDC[j,i]) / 2

    Parameters:
    -----------
    pdc_matrix : np.ndarray  shape (19, 19)

    Returns:
    --------
    edge_index : torch.LongTensor  shape (2, 19*18)  — all non-self edges
    edge_attr  : torch.FloatTensor shape (19*18,)
    """
    # Symmetrise
    sym = (pdc_matrix + pdc_matrix.T) / 2.0

    src, dst, weights = [], [], []
    for i in range(N_CHANNELS):
        for j in range(N_CHANNELS):
            if i != j:
                src.append(i)
                dst.append(j)
                weights.append(sym[i, j])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float)

    return edge_index, edge_attr


def normalize_features(features):
    """
    Standardise node features across the 19 nodes (z-score per feature column).
    Prevents scale domination (activity is in µV², powers are much smaller).
    """
    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0, keepdims=True) + 1e-12
    return ((features - mean) / std).astype(np.float32)


# ============================================================================
# EPOCH SELECTION LOGIC
# ============================================================================

def select_valid_epochs(labels, metadata, control_minutes=2.0):
    """
    Select which epochs to keep and assign final binary labels.

    Rules:
      - ICTAL   (label=1): ALL epochs that overlap seizure period
      - CONTROL (label=0): first K epochs from start of recording,
                           where K = number of ictal epochs (perfect balance)
                           but never more than control_minutes worth of epochs
      - DROP everything else (ambiguous pre-ictal zone + all post-ictal)

    This guarantees 50/50 balance within every subject regardless of
    how long the seizure is, making metrics meaningful across all folds.

    Parameters:
    -----------
    labels : np.ndarray  shape (n_epochs,)  0=pre-ictal, 1=ictal from step0
    metadata : dict  from subject_XX_metadata.json
    control_minutes : float  — hard upper cap on control window

    Returns:
    --------
    selected_indices : list[int]
    selected_labels  : list[int]
    """
    # Hard cap: never use more than control_minutes worth of control epochs
    max_control_epochs = int((control_minutes * 60) / 4.0)  # 4s per epoch

    # Step 1: collect ALL ictal epoch indices
    ictal_indices = [ep for ep, lbl in enumerate(labels) if lbl == 1]
    n_ictal = len(ictal_indices)

    if n_ictal == 0:
        return [], []

    # Step 2: how many control epochs to take (match ictal count, capped)
    n_control_to_take = min(n_ictal, max_control_epochs)

    # Step 3: collect control epochs from the START of the recording only
    # (furthest from the seizure → cleanest baseline)
    control_indices = []
    for ep_idx, lbl in enumerate(labels):
        if lbl == 0 and len(control_indices) < n_control_to_take:
            control_indices.append(ep_idx)
        # Stop once we've collected enough
        if len(control_indices) == n_control_to_take:
            break

    # Step 4: combine and sort by epoch index (preserves temporal order)
    selected_indices = sorted(control_indices + ictal_indices)
    selected_labels  = [
        0 if idx in set(control_indices) else 1
        for idx in selected_indices
    ]

    return selected_indices, selected_labels


# ============================================================================
# PER-SUBJECT PROCESSING
# ============================================================================

def process_subject(subj_id, epochs_dir, connectivity_dir,
                    control_minutes=2.0):
    """
    Build a list of PyG Data objects for one subject.

    Returns:
    --------
    graphs : list[Data]
    stats  : dict  (counts for logging)
    """
    subject_name = f"subject_{subj_id:02d}"
    epochs_dir      = Path(epochs_dir)
    connectivity_dir = Path(connectivity_dir)

    # ── Load epoch data ────────────────────────────────────────────────────
    epochs_file   = epochs_dir / f"{subject_name}_epochs.npy"
    labels_file   = epochs_dir / f"{subject_name}_labels.npy"
    metadata_file = epochs_dir / f"{subject_name}_metadata.json"

    if not epochs_file.exists():
        print(f"  ⚠ Epochs not found for {subject_name}, skipping")
        return [], {}

    epochs   = np.load(epochs_file)   # (n_epochs, 19, 1024)
    labels   = np.load(labels_file)   # (n_epochs,)

    with open(metadata_file) as f:
        metadata = json.load(f)

    # ── Load connectivity ──────────────────────────────────────────────────
    conn_file = connectivity_dir / f"{subject_name}_graphs.npz"
    if not conn_file.exists():
        print(f"  ⚠ Connectivity not found for {subject_name}, skipping")
        return [], {}

    conn_data = np.load(conn_file)
    pdc_integrated = conn_data['pdc_integrated']   # (n_epochs, 19, 19)

    # Make sure dimensions align
    n_epochs = min(len(epochs), len(labels), len(pdc_integrated))
    epochs   = epochs[:n_epochs]
    labels   = labels[:n_epochs]
    pdc_integrated = pdc_integrated[:n_epochs]

    # ── Select valid epochs ────────────────────────────────────────────────
    sel_indices, sel_labels = select_valid_epochs(
        labels, metadata, control_minutes
    )

    # ── Build one graph per selected epoch ────────────────────────────────
    graphs = []
    for ep_idx, y in zip(sel_indices, sel_labels):
        epoch = epochs[ep_idx]          # (19, 1024)
        pdc   = pdc_integrated[ep_idx]  # (19, 19)

        # Node features
        raw_features = extract_node_features(epoch)          # (19, 8)
        norm_features = normalize_features(raw_features)     # (19, 8)

        x = torch.tensor(norm_features, dtype=torch.float)  # (19, 8)

        # Edge index + weights
        edge_index, edge_attr = build_edge_index_and_weights(pdc)

        # Label
        y_tensor = torch.tensor(y, dtype=torch.long)

        # PyG Data object
        graph = Data(
            x          = x,
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = y_tensor,
            subject_id = torch.tensor(subj_id, dtype=torch.long),
            epoch_idx  = torch.tensor(ep_idx, dtype=torch.long),
        )
        graphs.append(graph)

    # Stats
    n_ctrl = sum(1 for lbl in sel_labels if lbl == 0)
    n_ict  = sum(1 for lbl in sel_labels if lbl == 1)

    stats = {
        'subject_id':  subj_id,
        'n_control':   n_ctrl,
        'n_ictal':     n_ict,
        'n_total':     len(graphs),
        'n_dropped':   n_epochs - len(graphs),
    }

    return graphs, stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build PyG graph dataset")
    parser.add_argument("--epochs_dir",       required=True)
    parser.add_argument("--connectivity_dir", required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--control_minutes",  type=float, default=2.0,
                        help="Minutes from start to use as control class")
    parser.add_argument("--n_subjects",       type=int,   default=34)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 5 — BUILD GRAPH DATASET")
    print("=" * 70)
    print(f"Control window : first {args.control_minutes} min")
    print(f"Node features  : 8  (3 Hjorth + 5 band power)")
    print(f"Edge definition: PDC integrated (symmetrised)")
    print(f"Subjects       : {args.n_subjects}")
    print("=" * 70)

    all_graphs = []
    all_stats  = []

    for subj_id in tqdm(range(1, args.n_subjects + 1), desc="Subjects"):
        graphs, stats = process_subject(
            subj_id,
            args.epochs_dir,
            args.connectivity_dir,
            args.control_minutes,
        )
        if graphs:
            all_graphs.extend(graphs)
            all_stats.append(stats)

            # Save per-subject graph list
            subj_path = output_dir / f"subject_{subj_id:02d}_graphs.pt"
            torch.save(graphs, subj_path)

    # ── Summary ───────────────────────────────────────────────────────────
    total_ctrl = sum(s['n_control'] for s in all_stats)
    total_ict  = sum(s['n_ictal']   for s in all_stats)
    total      = total_ctrl + total_ict

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"  Subjects processed : {len(all_stats)}")
    print(f"  Total graphs       : {total}")
    print(f"    Control (0)      : {total_ctrl}  ({100*total_ctrl/total:.1f}%)")
    print(f"    Ictal   (1)      : {total_ict}  ({100*total_ict/total:.1f}%)")
    print(f"  Graphs per subject : {total/len(all_stats):.1f} avg")

    # Save full dataset (all subjects combined)
    full_path = output_dir / "all_graphs.pt"
    torch.save(all_graphs, full_path)
    print(f"\n  Saved full dataset : {full_path}")

    # Save info JSON
    info = {
        'n_subjects':     len(all_stats),
        'total_graphs':   total,
        'total_control':  total_ctrl,
        'total_ictal':    total_ict,
        'control_minutes': args.control_minutes,
        'node_features':  8,
        'node_feature_names': [
            'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
            'delta_power', 'theta_power', 'alpha_power',
            'beta_power', 'gamma_power'
        ],
        'edge_definition': 'PDC integrated band (symmetrised)',
        'n_nodes':        N_CHANNELS,
        'channels':       CHANNELS,
        'subjects':       all_stats,
    }
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print("  Saved dataset_info.json")
    print("\n✅ Step 5 complete. Run step6_train_baseline_gcn.py next.")
    print("=" * 70)


if __name__ == "__main__":
    main()