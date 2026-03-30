"""
Step 5 v2 — Build PyTorch Geometric Graphs with Edge Attributes
===============================================================
Changes from v1:
  - Uniform edge count: exactly top-k outgoing edges per node
  - NNConv/GATv2-compatible edge attributes: [dtf, pdc] (2-dim)
  - --mode dtf | pdc controls which measure ranks edges
    (both always stored in edge_attr regardless)
  - edge_attr shape: (n_edges, 2) = [dtf_integrated, pdc_integrated]

Node features (24 per node):
  Convention: mat[i,j] = influence of source j on sink i
  - DTF out-strength x6 bands: mat.sum(axis=0) — col sum
  - DTF in-strength  x6 bands: mat.sum(axis=1) — row sum
  - PDC out-strength x6 bands: mat.sum(axis=0)
  - PDC in-strength  x6 bands: mat.sum(axis=1)

Edge construction:
  Convention: mat[i,j] = j→i, so edge stored as (source=j, sink=i)
  For each sink i, keep top-k strongest incoming sources j,
  ranked by rank_mat[i,j] (= influence of j on i).
  edge_attr = [dtf_integrated[i,j], pdc_integrated[i,j]]

Usage:
  python step5_build_graphs_v2.py \
    --conndir path/to/connectivity \
    --outdir  path/to/graphs_v2 \
    --topk    6 \
    --mode    dtf \
    --ratio   2
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
    Build (19, 24) node feature matrix.

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
# 2. EDGES WITH ATTRIBUTES
# ═══════════════════════════════════════════════════════════════

def build_edges_topk(dtf_mat, pdc_mat, top_k=6, mode='dtf'):
    """
    For each sink node i, keep top_k strongest incoming sources j,
    ranked by rank_mat[i,j] (influence of j on i).

    Convention: mat[i,j] = j→i  →  stored as (source=j, sink=i)

    Returns
    -------
    edge_index : (2, K*top_k)  torch.long   row0=source, row1=sink
    edge_attr  : (K*top_k, 2)  torch.float  [dtf_weight, pdc_weight]
    """
    rank_mat = dtf_mat if mode == 'dtf' else pdc_mat

    rows, cols, dtf_vals, pdc_vals = [], [], [], []

    for i in range(K):
        # rank incoming sources to sink i by rank_mat[i,j]
        scores = [(rank_mat[i, j], j) for j in range(K) if i != j]
        scores.sort(reverse=True)
        for _, j in scores[:top_k]:
            rows.append(j)   # source
            cols.append(i)   # sink
            dtf_vals.append(float(dtf_mat[i, j]))
            pdc_vals.append(float(pdc_mat[i, j]))

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr  = torch.tensor(
        list(zip(dtf_vals, pdc_vals)), dtype=torch.float  # (E, 2)
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
        [1 if i in ictal_set else 0 for i in keep_idx], dtype=np.int64
    )
    return keep_idx, y_binary


# ═══════════════════════════════════════════════════════════════
# 4. PROCESS ONE SUBJECT
# ═══════════════════════════════════════════════════════════════

def process_subject(npz_path, patient_id, top_k=6, mode='dtf', ratio=2):
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

        edge_index, edge_attr = build_edges_topk(
            dtf_int, pdc_int, top_k=top_k, mode=mode
        )

        graph = Data(
            x          = torch.tensor(x, dtype=torch.float),
            edge_index = edge_index,
            edge_attr  = edge_attr,              # (E, 2): [dtf, pdc]
            y          = torch.tensor(y_binary[pos], dtype=torch.long),
            patient    = torch.tensor(patient_id,    dtype=torch.long),
        )
        graphs.append(graph)

    avg_e = np.mean([g.edge_index.shape[1] for g in graphs])
    print(f"  subj {npz_path.stem.split('_')[1]:>3s}  "
          f"(pat {patient_id:2d})  "
          f"graphs={len(graphs):4d}  "
          f"ictal={y_binary.sum():3d}  "
          f"pre={(y_binary==0).sum():3d}  "
          f"avg_edges={avg_e:.0f}")
    return graphs


# ═══════════════════════════════════════════════════════════════
# 5. BUILD DATASET
# ═══════════════════════════════════════════════════════════════

def build_dataset(conn_dir, out_dir, top_k=6, mode='dtf', ratio=2):
    conn_dir = Path(conn_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(conn_dir.glob('subject_*_graphs.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No subject_*_graphs.npz in {conn_dir}")

    print(f"\nFound {len(npz_files)} subject files")
    print(f"Edge mode      : {mode.upper()} top-{top_k} per sink node")
    print(f"Edge direction : source→sink  (j→i when mat[i,j] = j drives i)")
    print(f"Edge attr      : [dtf_integrated, pdc_integrated]  (2-dim)")
    print(f"Edges per graph: exactly {K * top_k}  ({K} nodes × {top_k})")
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
        graphs = process_subject(npz, patient_id,
                                 top_k=top_k, mode=mode, ratio=ratio)
        all_graphs.extend(graphs)

    out_path = out_dir / f'all_graphs_{mode}_topk{top_k}.pt'
    torch.save(all_graphs, out_path)

    labels   = torch.tensor([g.y.item()       for g in all_graphs])
    patients = torch.tensor([g.patient.item() for g in all_graphs])
    avg_edges = np.mean([g.edge_index.shape[1] for g in all_graphs])

    print(f"{'─'*55}")
    print(f"\nDataset summary:")
    print(f"  Total graphs    : {len(all_graphs)}")
    print(f"  Ictal           : {(labels==1).sum().item()}")
    print(f"  Pre-ictal       : {(labels==0).sum().item()}")
    print(f"  Patients        : {sorted(patients.unique().tolist())}")
    print(f"  Node feat dim   : {all_graphs[0].x.shape[1]}")
    print(f"  Edge attr dim   : {all_graphs[0].edge_attr.shape[1]}")
    print(f"  Edges per graph : {int(avg_edges)}  (uniform)")
    print(f"\nSaved -> {out_path}")


# ═══════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Step 5 v2 — Graph construction with top-k edges")
    parser.add_argument('--conndir', required=True,
                        help='Path to connectivity/ folder (step2 output)')
    parser.add_argument('--outdir',  required=True,
                        help='Output folder for .pt graph file')
    parser.add_argument('--topk',    type=int, default=6,
                        help='Top-k incoming edges per node (default: 6)')
    parser.add_argument('--mode',    default='dtf',
                        choices=['dtf', 'pdc'],
                        help='Measure used to rank edges (default: dtf)')
    parser.add_argument('--ratio',   type=int, default=2,
                        help='Pre-ictal epochs per ictal (default: 2)')
    args = parser.parse_args()
    build_dataset(args.conndir, args.outdir,
                  args.topk, args.mode, args.ratio)
