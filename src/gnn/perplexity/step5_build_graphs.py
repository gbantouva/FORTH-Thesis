"""
Step 5 — Build PyTorch Geometric Graphs from DTF/PDC Connectivity
=================================================================
Input : connectivity/<subject>_graphs.npz  (from step2)
Output: graphs/all_graphs.pt
         -> list of torch_geometric.data.Data objects
         -> each Data has:
              x         : (19, node_feat_dim)  node features
              edge_index: (2, n_edges)          COO format
              edge_attr : (n_edges, 2)          [DTF, PDC] weights
              y         : scalar label (0=pre-ictal, 1=ictal)
              patient   : scalar patient ID

Node features per node (19 total per feature):
  - DTF out-strength per band (6) = how much this channel drives others
  - DTF in-strength  per band (6) = how much this channel receives
  - PDC out-strength per band (6)
  - PDC in-strength  per band (6)
  => 24 features per node

Edge construction:
  - Use integrated band DTF as primary weight
  - Keep only edges above a threshold (top-k or percentile)
    to avoid a fully connected graph with noisy weak edges
  - Edge attr = [dtf_integrated, pdc_integrated] for each kept edge

Usage:
  python step5_build_graphs.py \
    --conndir path/to/connectivity \
    --outdir  path/to/graphs \
    --threshold 0.1 \
    --ratio 2
"""

import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path

BANDS = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']
K     = 19  # channels

# ── Subject → Patient mapping ──────────────────────────────────
SUBJECT_TO_PATIENT = {}
for s in [1]:            SUBJECT_TO_PATIENT[s] = 1
for s in [2]:            SUBJECT_TO_PATIENT[s] = 13
for s in range(3, 11):   SUBJECT_TO_PATIENT[s] = 14
for s in [11]:           SUBJECT_TO_PATIENT[s] = 15
for s in range(12, 26):  SUBJECT_TO_PATIENT[s] = 24
for s in range(26, 33):  SUBJECT_TO_PATIENT[s] = 27
for s in [33]:           SUBJECT_TO_PATIENT[s] = 29
for s in [34]:           SUBJECT_TO_PATIENT[s] = 35


# ═══════════════════════════════════════════════════════════════
# 1. NODE FEATURES
# ═══════════════════════════════════════════════════════════════

def node_features(data_npz, epoch_idx):
    """
    Build node feature matrix (19, 24) for one epoch.

    Features per node:
      DTF out-strength x6 bands, DTF in-strength x6 bands
      PDC out-strength x6 bands, PDC in-strength x6 bands
    """
    feats = []
    for measure in ['dtf', 'pdc']:
        out_bands, in_bands = [], []
        for band in BANDS:
            mat = data_npz[f'{measure}_{band}'][epoch_idx]  # (19,19)
            out_bands.append(mat.sum(axis=1))   # (19,) out-strength
            in_bands.append(mat.sum(axis=0))    # (19,) in-strength
        # stack: (19, 6) each
        feats.append(np.stack(out_bands, axis=1))
        feats.append(np.stack(in_bands,  axis=1))

    # (19, 24)
    return np.concatenate(feats, axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# 2. EDGES
# ═══════════════════════════════════════════════════════════════

def build_edges(dtf_mat, pdc_mat, threshold=0.1):
    """
    Build edge_index and edge_attr from (19,19) DTF and PDC matrices.

    Keeps directed edges where dtf_integrated > threshold.
    edge_attr columns: [dtf_weight, pdc_weight]

    Returns
    -------
    edge_index : (2, n_edges)  torch.long
    edge_attr  : (n_edges, 2)  torch.float
    """
    rows, cols, dtf_vals, pdc_vals = [], [], [], []

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            if dtf_mat[i, j] > threshold:
                rows.append(i)
                cols.append(j)
                dtf_vals.append(dtf_mat[i, j])
                pdc_vals.append(pdc_mat[i, j])

    if len(rows) == 0:
        # fallback: keep top-10% edges regardless of threshold
        flat    = dtf_mat.copy()
        np.fill_diagonal(flat, 0)
        cutoff  = np.percentile(flat[flat > 0], 90)
        return build_edges(dtf_mat, pdc_mat, threshold=cutoff * 0.5)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr  = torch.tensor(
        list(zip(dtf_vals, pdc_vals)), dtype=torch.float
    )
    return edge_index, edge_attr


# ═══════════════════════════════════════════════════════════════
# 3. EPOCH SELECTION (same logic as step3)
# ═══════════════════════════════════════════════════════════════

def select_epochs(labels_orig, ratio=2):
    ictal_idx     = np.where(labels_orig == 1)[0]
    non_ictal_idx = np.sort(np.where(labels_orig != 1)[0])
    n_ictal       = len(ictal_idx)
    n_pre         = min(ratio * n_ictal, len(non_ictal_idx))
    preictal_idx  = non_ictal_idx[:n_pre]
    keep_idx      = np.sort(np.concatenate([ictal_idx, preictal_idx]))
    ictal_set     = set(ictal_idx.tolist())
    y_binary      = np.array(
        [1 if i in ictal_set else 0 for i in keep_idx],
        dtype=np.int64
    )
    return keep_idx, y_binary


# ═══════════════════════════════════════════════════════════════
# 4. PROCESS ONE SUBJECT
# ═══════════════════════════════════════════════════════════════

def process_subject(npz_path, patient_id, threshold=0.1, ratio=2):
    data        = np.load(npz_path)
    labels_orig = data['labels'].copy()
    n_ictal     = int((labels_orig == 1).sum())

    if n_ictal == 0:
        print(f"  [SKIP] {npz_path.stem} — no ictal epochs")
        return []

    keep_idx, y_binary = select_epochs(labels_orig, ratio=ratio)
    graphs = []

    for pos, orig_i in enumerate(keep_idx):
        # Node features (19, 24)
        x = node_features(data, orig_i)

        # Edge index + attr from integrated band
        dtf_int = data['dtf_integrated'][orig_i]   # (19,19)
        pdc_int = data['pdc_integrated'][orig_i]
        edge_index, edge_attr = build_edges(dtf_int, pdc_int, threshold)

        graph = Data(
            x          = torch.tensor(x, dtype=torch.float),
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor(y_binary[pos], dtype=torch.long),
            patient    = torch.tensor(patient_id,    dtype=torch.long),
        )
        graphs.append(graph)

    print(f"  {npz_path.stem.split('_')[1]:>3s}  "
          f"(patient {patient_id:2d})  "
          f"graphs={len(graphs):4d}  "
          f"ictal={y_binary.sum():3d}  "
          f"pre={( y_binary==0).sum():3d}  "
          f"avg_edges={np.mean([g.edge_index.shape[1] for g in graphs]):.0f}")
    return graphs


# ═══════════════════════════════════════════════════════════════
# 5. BUILD FULL DATASET
# ═══════════════════════════════════════════════════════════════

def build_dataset(conn_dir, out_dir, threshold=0.1, ratio=2):
    conn_dir = Path(conn_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(conn_dir.glob('subject_*_graphs.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No subject_*_graphs.npz in {conn_dir}")

    print(f"\nFound {len(npz_files)} subject files")
    print(f"Edge threshold : DTF > {threshold}")
    print(f"Pre-ictal ratio: {ratio}x ictal")
    print(f"{'─'*55}")

    all_graphs = []
    for npz in npz_files:
        try:
            subject_num = int(npz.stem.split('_')[1])
        except (IndexError, ValueError):
            continue

        patient_id = SUBJECT_TO_PATIENT.get(subject_num)
        if patient_id is None:
            continue

        graphs = process_subject(npz, patient_id, threshold, ratio)
        all_graphs.extend(graphs)

    out_path = out_dir / 'all_graphs.pt'
    torch.save(all_graphs, out_path)

    labels   = torch.tensor([g.y.item() for g in all_graphs])
    patients = torch.tensor([g.patient.item() for g in all_graphs])
    avg_edges = np.mean([g.edge_index.shape[1] for g in all_graphs])
    node_dim  = all_graphs[0].x.shape[1]

    print(f"{'─'*55}")
    print(f"\nDataset summary:")
    print(f"  Total graphs  : {len(all_graphs)}")
    print(f"  Ictal         : {(labels==1).sum().item()}")
    print(f"  Pre-ictal     : {(labels==0).sum().item()}")
    print(f"  Patients      : {sorted(patients.unique().tolist())}")
    print(f"  Node feat dim : {node_dim}")
    print(f"  Avg edges/graph: {avg_edges:.1f}")
    print(f"\nSaved -> {out_path}")


# ═══════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 5 — Graph construction")
    parser.add_argument('--conndir',   required=True)
    parser.add_argument('--outdir',    required=True)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Min DTF edge weight to keep (default: 0.1)')
    parser.add_argument('--ratio',     type=int,   default=2,
                        help='Pre-ictal per ictal (default: 2)')
    args = parser.parse_args()
    build_dataset(args.conndir, args.outdir, args.threshold, args.ratio)
