"""
Step 3c — Build PyTorch Geometric Graphs
==========================================
Converts per-epoch connectivity matrices + node features into PyTorch
Geometric (PyG) Data objects ready for GNN training.

Graph structure per epoch
──────────────────────────
  Nodes  : 19 EEG channels
  Node features (x)  : (19, 12)  — band power + Hjorth + stats  (Step 3a)
  Edge index : COO format sparse adjacency from DTF or PDC matrix
  Edge weights (edge_attr) : connectivity values after thresholding
  Label (y)  : 0 = pre-ictal,  1 = ictal   (graph-level)

Thresholding strategy
──────────────────────
  Keeping all 19×18 = 342 directed edges per epoch creates a very dense
  graph. We threshold by keeping only edges above a percentile of the
  connectivity matrix. Default: top 30% of edges (70th percentile).
  This is a common choice in EEG graph papers.

  The threshold is computed GLOBALLY across all subjects so that edge
  density is comparable across subjects and epochs.

Dataset splits
───────────────
  LOSO-CV (Leave-One-Subject-Out) — same as Step 3b.
  We save the full dataset as a single .pt file and split at training time.
  Subject IDs are saved per graph so you can reconstruct any split.

Output
───────
  graphs/
    dataset.pt          — list of PyG Data objects (all subjects)
    dataset_info.json   — metadata, split indices, class counts
    subject_index.npy   — subject ID per graph (for LOSO splitting)
    label_index.npy     — label per graph

Usage:
    pip install torch torch_geometric

    python step3c_build_graphs.py \\
        --conndir   path/to/connectivity \\
        --featdir   path/to/node_features \\
        --epochdir  path/to/preprocessed_epochs \\
        --outputdir path/to/graphs \\
        --band      integrated \\
        --threshold 0.70
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2',
]
N_CHANNELS  = 19
N_FEATURES  = 12   # from Step 3a

VALID_BANDS = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']


# ══════════════════════════════════════════════════════════════════════════════
# 1. IMPORT TORCH  (with helpful error message)
# ══════════════════════════════════════════════════════════════════════════════

def import_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "\n❌  PyTorch not found.\n"
            "    Install with:  pip install torch torchvision torchaudio\n"
            "    Then:          pip install torch_geometric\n"
        )

def import_pyg():
    try:
        from torch_geometric.data import Data
        return Data
    except ImportError:
        raise ImportError(
            "\n❌  PyTorch Geometric not found.\n"
            "    Install with:  pip install torch_geometric\n"
            "    See:  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD SUBJECT DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_subject(subject_name, conn_dir, feat_dir, epoch_dir, band):
    """
    Load connectivity matrix + node features + labels for one subject.

    Returns
    -------
    adj_matrices  : ndarray (n_epochs, 19, 19)  — DTF or PDC, chosen band
    node_features : ndarray (n_epochs, 19, 12)
    labels        : ndarray (n_epochs,)
    or None if any required file is missing
    """
    # ── Connectivity ──────────────────────────────────────────────────────
    npz_path = conn_dir / f"{subject_name}_graphs.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    key  = f"dtf_{band}"
    if key not in data:
        print(f"  ⚠️  Band '{band}' not found in {npz_path.name}")
        return None

    adj_matrices = data[key].astype(np.float32)   # (n_epochs, 19, 19)
    labels_conn  = data['labels']                  # (n_epochs,)

    # ── Node features ─────────────────────────────────────────────────────
    norm_path = feat_dir / f"{subject_name}_node_features_normalized.npy"
    raw_path  = feat_dir / f"{subject_name}_node_features.npy"
    feat_path = norm_path if norm_path.exists() else (
                raw_path  if raw_path.exists()  else None)

    if feat_path is None:
        print(f"  ⚠️  Node features not found for {subject_name}")
        return None

    node_features = np.load(feat_path).astype(np.float32)  # (n_total, 19, 12)

    # ── Align epoch counts ────────────────────────────────────────────────
    # Connectivity may have fewer epochs (VAR failures filtered out)
    # Node features cover all original epochs
    # We need to use the 'indices' array from the connectivity file to align

    if 'indices' in data:
        valid_indices = data['indices']   # which original epochs survived VAR
        if len(valid_indices) <= len(node_features):
            node_features = node_features[valid_indices]

    n = min(len(adj_matrices), len(node_features), len(labels_conn))
    adj_matrices  = adj_matrices[:n]
    node_features = node_features[:n]
    labels        = labels_conn[:n]

    # ── Training mask ─────────────────────────────────────────────────────
    mask_path = epoch_dir / f"{subject_name}_training_mask.npy"
    if mask_path.exists():
        full_mask = np.load(mask_path)
        # Align mask to valid epochs
        if 'indices' in data:
            valid_indices = data['indices'][:n]
            mask = full_mask[valid_indices] if len(valid_indices) <= len(full_mask) else np.ones(n, dtype=bool)
        else:
            mask = full_mask[:n]

        adj_matrices  = adj_matrices[mask]
        node_features = node_features[mask]
        labels        = labels[mask]

    return adj_matrices, node_features, labels


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE GLOBAL THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════

def compute_global_threshold(all_adj, percentile=70.0):
    """
    Compute a connectivity threshold globally across all subjects and epochs.

    We exclude diagonal values (self-connections = 0) before computing.

    Parameters
    ----------
    all_adj     : list of ndarray (n_epochs_i, 19, 19)
    percentile  : float  e.g. 70.0 means keep top 30% of edges

    Returns
    -------
    threshold : float
    """
    off_diag_values = []
    mask = ~np.eye(N_CHANNELS, dtype=bool)

    for adj in all_adj:
        # adj shape: (n_epochs, 19, 19)
        for ep in range(len(adj)):
            off_diag_values.append(adj[ep][mask])

    all_vals  = np.concatenate(off_diag_values)
    threshold = float(np.percentile(all_vals, percentile))
    return threshold


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD SINGLE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(adj, node_feat, label, threshold, torch, Data):
    """
    Convert one epoch into a PyG Data object.

    Parameters
    ----------
    adj       : ndarray (19, 19) — connectivity matrix (diagonal = 0)
    node_feat : ndarray (19, 12) — node features
    label     : int  (0 or 1)
    threshold : float
    torch     : torch module
    Data      : torch_geometric.data.Data class

    Returns
    -------
    PyG Data object
    """
    # ── Threshold edges ───────────────────────────────────────────────────
    # Keep edges where connectivity > threshold (directed graph)
    edge_mask = (adj > threshold) & (~np.eye(N_CHANNELS, dtype=bool))
    src, dst  = np.where(edge_mask)

    edge_index = torch.tensor(
        np.stack([src, dst], axis=0), dtype=torch.long
    )                                                # (2, n_edges)

    edge_attr = torch.tensor(
        adj[src, dst], dtype=torch.float32
    ).unsqueeze(1)                                   # (n_edges, 1)

    x = torch.tensor(node_feat, dtype=torch.float32)   # (19, 12)
    y = torch.tensor([label],   dtype=torch.long)       # (1,)

    # If no edges survive threshold, add self-loops to avoid isolated nodes
    if edge_index.shape[1] == 0:
        self_loops = torch.arange(N_CHANNELS, dtype=torch.long)
        edge_index = torch.stack([self_loops, self_loops], dim=0)
        edge_attr  = torch.ones(N_CHANNELS, 1, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build PyTorch Geometric graphs from EEG connectivity + node features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--conndir',    required=True)
    parser.add_argument('--featdir',    required=True)
    parser.add_argument('--epochdir',   required=True)
    parser.add_argument('--outputdir',  required=True)
    parser.add_argument('--band',       default='integrated',
                        choices=VALID_BANDS,
                        help='Connectivity band to use as edges (default: integrated)')
    parser.add_argument('--threshold',  type=float, default=0.70,
                        help='Percentile threshold for edge pruning (default: 0.70 = keep top 30%%)')
    parser.add_argument('--use_pdc',    action='store_true',
                        help='Use PDC instead of DTF for edges')
    args = parser.parse_args()

    # ── Import PyTorch / PyG ──────────────────────────────────────────────
    torch = import_torch()
    Data  = import_pyg()

    conn_dir   = Path(args.conndir)
    feat_dir   = Path(args.featdir)
    epoch_dir  = Path(args.epochdir)
    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn_type = 'pdc' if args.use_pdc else 'dtf'
    band_key  = f"{conn_type}_{args.band}"

    subject_names = sorted([
        f.stem.replace('_epochs', '')
        for f in epoch_dir.glob('subject_*_epochs.npy')
    ])

    print("=" * 72)
    print("STEP 3c — BUILD PyTorch Geometric GRAPHS")
    print("=" * 72)
    print(f"  Subjects:       {len(subject_names)}")
    print(f"  Connectivity:   {conn_type.upper()} — {args.band} band")
    print(f"  Edge threshold: {args.threshold:.0%} percentile "
          f"(keep top {1-args.threshold:.0%} of edges)")
    print(f"  Node features:  (19, {N_FEATURES})")
    print(f"  Output:         {output_dir}")
    print("=" * 72)

    # ── Step 1: Load all subjects ─────────────────────────────────────────
    print("\nLoading subject data...")
    all_adj     = []   # (n_epochs, 19, 19) per subject
    all_feats   = []   # (n_epochs, 19, 12) per subject
    all_labels  = []   # (n_epochs,) per subject
    valid_subjs = []

    for subj in tqdm(subject_names, desc="Loading"):
        result = load_subject(subj, conn_dir, feat_dir, epoch_dir, args.band)
        if result is None:
            print(f"  ⚠️  Skipping {subj}")
            continue
        adj, feats, labels = result

        # Skip subjects with only one class
        if len(np.unique(labels)) < 2:
            print(f"  ⚠️  Skipping {subj} — only one class")
            continue

        all_adj.append(adj)
        all_feats.append(feats)
        all_labels.append(labels)
        valid_subjs.append(subj)

    print(f"\n  Loaded {len(valid_subjs)} subjects")
    total_epochs = sum(len(l) for l in all_labels)
    total_ictal  = sum((l == 1).sum() for l in all_labels)
    print(f"  Total epochs:   {total_epochs:,}  "
          f"(ictal={total_ictal:,}, pre-ictal={total_epochs-total_ictal:,})")

    # ── Step 2: Compute global threshold ──────────────────────────────────
    print(f"\nComputing global edge threshold ({args.threshold:.0%} percentile)...")
    threshold = compute_global_threshold(all_adj, percentile=args.threshold * 100)
    print(f"  Threshold value: {threshold:.4f}")

    # Estimate edge density after thresholding
    sample_adj  = all_adj[0][0]
    mask        = ~np.eye(N_CHANNELS, dtype=bool)
    kept_edges  = int((sample_adj[mask] > threshold).sum())
    total_edges = N_CHANNELS * (N_CHANNELS - 1)
    print(f"  Edge density (sample epoch): {kept_edges}/{total_edges} = "
          f"{kept_edges/total_edges:.1%}")

    # ── Step 3: Build PyG graphs ──────────────────────────────────────────
    print("\nBuilding graphs...")
    graph_list     = []   # all PyG Data objects
    subject_index  = []   # subject ID per graph (int)
    label_index    = []   # label per graph

    for s_idx, (subj, adj_all, feat_all, lbl_all) in enumerate(
            tqdm(zip(valid_subjs, all_adj, all_feats, all_labels),
                 total=len(valid_subjs), desc="Building")):

        n_bad = 0
        for ep_idx in range(len(adj_all)):
            adj      = np.nan_to_num(adj_all[ep_idx],  nan=0.0, posinf=0.0, neginf=0.0)
            feats    = np.nan_to_num(feat_all[ep_idx], nan=0.0, posinf=0.0, neginf=0.0)
            label    = int(lbl_all[ep_idx])

            graph = build_graph(adj, feats, label, threshold, torch, Data)

            # Tag graph with metadata (useful for debugging and LOSO splits)
            graph.subject_id  = torch.tensor([s_idx], dtype=torch.long)
            graph.subject_name = subj
            graph.epoch_idx   = torch.tensor([ep_idx], dtype=torch.long)

            graph_list.append(graph)
            subject_index.append(s_idx)
            label_index.append(label)

        tqdm.write(f"  {subj}: {len(adj_all)} graphs")

    print(f"\n  Total graphs built: {len(graph_list):,}")

    # ── Step 4: Normalize node features globally ──────────────────────────
    # z-score across ALL subjects (not per-subject as in Step 3a)
    print("\nApplying global z-score normalization to node features...")

    # Stack all node features: (total_graphs, 19, 12)
    all_x = np.stack([g.x.numpy() for g in graph_list], axis=0)
    global_mean = all_x.mean(axis=0, keepdims=True)   # (1, 19, 12)
    global_std  = all_x.std(axis=0,  keepdims=True)   # (1, 19, 12)
    global_std[global_std < 1e-8] = 1.0

    for g in graph_list:
        x_norm = (g.x.numpy() - global_mean[0]) / global_std[0]
        g.x    = torch.tensor(x_norm, dtype=torch.float32)

    print("  ✅ Global normalization applied")

    # ── Step 5: Save dataset ──────────────────────────────────────────────
    print("\nSaving dataset...")

    dataset_path = output_dir / 'dataset.pt'
    torch.save(graph_list, dataset_path)
    print(f"  ✅ {dataset_path}  ({len(graph_list):,} graphs)")

    subject_index_arr = np.array(subject_index)
    label_index_arr   = np.array(label_index)
    np.save(output_dir / 'subject_index.npy', subject_index_arr)
    np.save(output_dir / 'label_index.npy',   label_index_arr)

    # ── Step 6: Save LOSO split indices ──────────────────────────────────
    loso_splits = {}
    for s_idx, subj in enumerate(valid_subjs):
        test_idx  = np.where(subject_index_arr == s_idx)[0].tolist()
        train_idx = np.where(subject_index_arr != s_idx)[0].tolist()
        loso_splits[subj] = {
            'test':  test_idx,
            'train': train_idx,
        }

    splits_path = output_dir / 'loso_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(loso_splits, f, indent=2)
    print(f"  ✅ {splits_path}  ({len(loso_splits)} LOSO folds)")

    # ── Step 7: Save metadata ─────────────────────────────────────────────
    info = {
        'description':    'PyTorch Geometric EEG graph dataset',
        'n_graphs':        len(graph_list),
        'n_subjects':      len(valid_subjs),
        'subjects':        valid_subjs,
        'connectivity':    conn_type.upper(),
        'band':            args.band,
        'edge_threshold':  float(threshold),
        'threshold_pct':   args.threshold,
        'n_nodes':         N_CHANNELS,
        'node_feature_dim': N_FEATURES,
        'edge_feature_dim': 1,
        'channel_names':   CHANNEL_NAMES,
        'normalization':   'global z-score across all subjects',
        'label_mapping':   {'0': 'pre-ictal', '1': 'ictal'},
        'class_counts': {
            'pre_ictal': int((label_index_arr == 0).sum()),
            'ictal':     int((label_index_arr == 1).sum()),
        },
        'epochs_per_subject': {
            subj: int((subject_index_arr == i).sum())
            for i, subj in enumerate(valid_subjs)
        },
        'sample_graph': {
            'x_shape':         '[19, 12]',
            'edge_index_shape': '[2, n_edges]',
            'edge_attr_shape':  '[n_edges, 1]',
            'y_shape':          '[1]',
        },
        'files': {
            'dataset.pt':       'list of PyG Data objects',
            'subject_index.npy': 'subject ID per graph',
            'label_index.npy':   'label per graph (0/1)',
            'loso_splits.json':  'train/test indices per LOSO fold',
        },
    }

    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  ✅ {info_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    sample = graph_list[0]
    print("\n" + "=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(f"  Total graphs:        {len(graph_list):,}")
    print(f"    Pre-ictal:         {(label_index_arr==0).sum():,}")
    print(f"    Ictal:             {(label_index_arr==1).sum():,}")
    print(f"\n  Sample graph:")
    print(f"    x (node features): {list(sample.x.shape)}")
    print(f"    edge_index:        {list(sample.edge_index.shape)}")
    print(f"    edge_attr:         {list(sample.edge_attr.shape)}")
    print(f"    y (label):         {list(sample.y.shape)}")
    print(f"\n  LOSO folds:        {len(loso_splits)}")
    print(f"\n  How to load in your GNN script:")
    print(f"    import torch")
    print(f"    graphs = torch.load('{dataset_path}')")
    print(f"    # graphs[i] is a PyG Data object")
    print()
    print("  Next step:")
    print("    Step 3d: Train supervised GCN")
    print("      python step3d_train_gcn.py \\")
    print(f"          --datadir {output_dir}")
    print("=" * 72)


if __name__ == '__main__':
    main()
