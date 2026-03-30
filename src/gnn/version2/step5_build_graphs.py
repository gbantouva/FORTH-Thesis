"""
Step 5 — Build PyTorch Geometric Graphs from DTF/PDC Connectivity
=================================================================
Input : connectivity/<subject>_graphs.npz  (from step2)
Output: graphs/all_graphs_thresh<threshold>.pt
         -> list of torch_geometric.data.Data objects
         -> each Data has:
              x         : (19, 24)      node features
              edge_index: (2, n_edges)  COO format  source→sink
              edge_attr : (n_edges, 2)  [DTF, PDC] weights
              y         : scalar label (0=pre-ictal, 1=ictal)
              patient   : scalar patient ID

Node features per node (19 channels):
  Convention: mat[i,j] = influence of source j on sink i
  - DTF out-strength per band (6): mat.sum(axis=0) — col sum
  - DTF in-strength  per band (6): mat.sum(axis=1) — row sum
  - PDC out-strength per band (6): mat.sum(axis=0)
  - PDC in-strength  per band (6): mat.sum(axis=1)
  => 24 features per node

Edge construction:
  Convention: mat[i,j] = j→i, so edge stored as (j, i)
  - Keep directed edges where dtf_integrated[i,j] > threshold
  - Edge attr = [dtf_integrated, pdc_integrated]
  - Fallback if no edges survive: top-10% by DTF value

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
K     = 19


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

    Convention: mat[i,j] = influence of source j on sink i
      out-strength of node j = mat.sum(axis=0)[j]  (col sum)
      in-strength  of node i = mat.sum(axis=1)[i]  (row sum)

    Feature order per node:
      [DTF_out x6, DTF_in x6, PDC_out x6, PDC_in x6]
    """
    feats = []
    for measure in ['dtf', 'pdc']:
        out_bands, in_bands = [], []
        for band in BANDS:
            mat = data_npz[f'{measure}_{band}'][epoch_idx]  # (19,19)
            out_bands.append(mat.sum(axis=0))   # (19,) col sum = out-strength
            in_bands.append(mat.sum(axis=1))    # (19,) row sum = in-strength
        feats.append(np.stack(out_bands, axis=1))  # (19,6)
        feats.append(np.stack(in_bands,  axis=1))  # (19,6)

    return np.concatenate(feats, axis=1).astype(np.float32)  # (19,24)


# ═══════════════════════════════════════════════════════════════
# 2. EDGES
# ═══════════════════════════════════════════════════════════════

def build_edges(dtf_mat, pdc_mat, threshold=0.1):
    """
    Build edge_index and edge_attr from (19,19) DTF and PDC matrices.

    Convention: dtf_mat[i,j] = influence of source j on sink i = edge j→i
    So we store edge as (source=j, sink=i).

    Keeps edges where dtf_mat[i,j] > threshold.
    Fallback if no edges survive: lower threshold to top-10% percentile.

    Returns
    -------
    edge_index : (2, n_edges)  torch.long   [source, sink]
    edge_attr  : (n_edges, 2)  torch.float  [dtf_weight, pdc_weight]
    """
    rows, cols, dtf_vals, pdc_vals = [], [], [], []

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            if dtf_mat[i, j] > threshold:
                rows.append(j)        # source = j
                cols.append(i)        # sink   = i
                dtf_vals.append(dtf_mat[i, j])
                pdc_vals.append(pdc_mat[i, j])

    if len(rows) == 0:
        # fallback: use 90th percentile of non-zero off-diagonal values
        flat = dtf_mat.copy()
        np.fill_diagonal(flat, 0)
        nonzero = flat[flat > 0]
        if len(nonzero) == 0:
            # absolute fallback: top-K edges by value
            flat_idx = np.argsort(flat.ravel())[::-1][:K]
            for idx in flat_idx:
                i, j = divmod(int(idx), K)
                if i != j:
                    rows.append(j)
                    cols.append(i)
                    dtf_vals.append(float(dtf_mat[i, j]))
                    pdc_vals.append(float(pdc_mat[i, j]))
        else:
            cutoff = np.percentile(nonzero, 90)
            return build_edges(dtf_mat, pdc_mat, threshold=cutoff * 0.5)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr  = torch.tensor(
        list(zip(dtf_vals, pdc_vals)), dtype=torch.float
    )
    return edge_index, edge_attr


# ═══════════════════════════════════════════════════════════════
# 3. EPOCH SELECTION
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
        x       = node_features(data, orig_i)
        dtf_int = data['dtf_integrated'][orig_i]  # (19,19)
        pdc_int = data['pdc_integrated'][orig_i]  # (19,19)

        edge_index, edge_attr = build_edges(dtf_int, pdc_int, threshold)

        graph = Data(
            x          = torch.tensor(x, dtype=torch.float),
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor(y_binary[pos], dtype=torch.long),
            patient    = torch.tensor(patient_id,    dtype=torch.long),
        )
        graphs.append(graph)

    avg_e = np.mean([g.edge_index.shape[1] for g in graphs])
    print(f"  {npz_path.stem.split('_')[1]:>3s}  "
          f"(patient {patient_id:2d})  "
          f"graphs={len(graphs):4d}  "
          f"ictal={y_binary.sum():3d}  "
          f"pre={(y_binary==0).sum():3d}  "
          f"avg_edges={avg_e:.0f}")
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
    print(f"Edge threshold : DTF > {threshold}  (variable edges per graph)")
    print(f"Edge direction : source→sink  (j→i when mat[i,j] = j drives i)")
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

    # filename encodes threshold for traceability
    out_path = out_dir / f'all_graphs_thresh{threshold}.pt'
    torch.save(all_graphs, out_path)

    labels    = torch.tensor([g.y.item()       for g in all_graphs])
    patients  = torch.tensor([g.patient.item() for g in all_graphs])
    avg_edges = np.mean([g.edge_index.shape[1] for g in all_graphs])
    min_edges = min(g.edge_index.shape[1] for g in all_graphs)
    max_edges = max(g.edge_index.shape[1] for g in all_graphs)

    print(f"{'─'*55}")
    print(f"\nDataset summary:")
    print(f"  Total graphs    : {len(all_graphs)}")
    print(f"  Ictal           : {(labels==1).sum().item()}")
    print(f"  Pre-ictal       : {(labels==0).sum().item()}")
    print(f"  Patients        : {sorted(patients.unique().tolist())}")
    print(f"  Node feat dim   : {all_graphs[0].x.shape[1]}")
    print(f"  Edge attr dim   : {all_graphs[0].edge_attr.shape[1]}")
    print(f"  Avg edges/graph : {avg_edges:.1f}")
    print(f"  Min edges/graph : {min_edges}")
    print(f"  Max edges/graph : {max_edges}")
    print(f"\nSaved -> {out_path}")


# ═══════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 5 v1 — Graph construction (threshold-based edges)")
    parser.add_argument('--conndir',   required=True,
                        help='Path to connectivity/ folder (step2 output)')
    parser.add_argument('--outdir',    required=True,
                        help='Output folder for .pt graph file')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Min DTF value to keep an edge (default: 0.1)')
    parser.add_argument('--ratio',     type=int,   default=2,
                        help='Pre-ictal epochs per ictal (default: 2)')
    args = parser.parse_args()
    build_dataset(args.conndir, args.outdir, args.threshold, args.ratio)
