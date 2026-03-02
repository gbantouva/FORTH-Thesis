"""
Step 5 — Build PyTorch Geometric Dataset
=========================================
Reads per-subject connectivity matrices and node features, and assembles
a list of torch_geometric.data.Data objects ready for GNN training.

Graph structure (per epoch):
    Nodes  : 19 EEG channels
    Edges  : fully connected (all i≠j pairs), weighted by DTF integrated band
    Node features (x)    : (19, 9)  — from step4_extract_node_features.py
    Edge index           : (2, 19*18)
    Edge weights (edge_attr) : (19*18, 1)  — DTF[i,j] for each directed edge j→i
    Label (y)            : 0 (pre-ictal/non-ictal) or 1 (ictal)

Patient-independent split:
    Train : subjects 3-10 (PAT14), 11 (PAT15), 12-25 (PAT24), 34 (PAT35)
            ~201 ictal epochs
    Val   : subject 33 (PAT29)
            60 ictal epochs  — used for early stopping and hyperparameter tuning
    Test  : subjects 1 (PAT11), 2 (PAT13)
            46 ictal epochs  — touched ONCE at the very end

Usage:
    python step5_build_dataset.py \
        --connectivity_dir F:/FORTH_Final_Thesis/FORTH-Thesis/final_connectivity \
        --features_dir     F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/node_features \
        --output_dir       F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/gnn_dataset
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    raise ImportError(
        'Install dependencies first:\n'
        '  pip install torch torch_geometric'
    )

# ── patient-independent split ─────────────────────────────────────────────────
PATIENT_SPLITS = {
    'train': list(range(3, 11)) + [11] + list(range(12, 26)) + [34],
    'val'  : [33],
    'test' : [1, 2],
}

SUBJECT_TO_PATIENT = {
    1: 'PAT11', 2: 'PAT13',
    **{i: 'PAT14' for i in range(3, 11)},
    11: 'PAT15',
    **{i: 'PAT24' for i in range(12, 26)},
    **{i: 'PAT27' for i in range(26, 33)},
    33: 'PAT29', 34: 'PAT35',
}


# ── helpers ───────────────────────────────────────────────────────────────────

def build_edge_index_and_weights(dtf_matrix: np.ndarray):
    """
    dtf_matrix : (19, 19)  — directed, diagonal=0, matrix[i,j] = j→i

    Returns:
        edge_index : LongTensor  (2, 342)
        edge_attr  : FloatTensor (342, 1)
    """
    K = dtf_matrix.shape[0]
    sources, sinks, weights = [], [], []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            sources.append(j)
            sinks.append(i)
            weights.append(dtf_matrix[i, j])

    edge_index = torch.tensor([sources, sinks], dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


def load_subject_graphs(subj_id, connectivity_dir, features_dir):
    subject_name  = f'subject_{subj_id:02d}'
    graphs_file   = connectivity_dir / f'{subject_name}_graphs.npz'
    features_file = features_dir     / f'{subject_name}_node_features.npy'

    if not graphs_file.exists():
        print(f'  [SKIP] {subject_name}: graphs.npz not found')
        return []
    if not features_file.exists():
        print(f'  [SKIP] {subject_name}: node_features.npy not found')
        return []

    graphs        = np.load(graphs_file)
    node_features = np.load(features_file)   # (n_valid, 19, 9)
    dtf_all       = graphs['dtf_integrated'] # (n_valid, 19, 19)
    labels_all    = graphs['labels']         # (n_valid,)
    n_valid       = len(labels_all)

    # align training_mask to post-VAR epoch indices
    if 'training_mask' in graphs:
        raw_mask = graphs['training_mask']
        if 'indices' in graphs:
            mask = raw_mask[graphs['indices'].astype(int)]
        else:
            mask = raw_mask[:n_valid]
    else:
        mask = np.ones(n_valid, dtype=bool)

    assert node_features.shape[0] == n_valid, (
        f'{subject_name}: feature/connectivity epoch count mismatch '
        f'({node_features.shape[0]} vs {n_valid})'
    )

    data_list = []
    for ep in range(n_valid):
        if not mask[ep]:
            continue

        edge_index, edge_attr = build_edge_index_and_weights(dtf_all[ep])
        graph = Data(
            x          = torch.tensor(node_features[ep], dtype=torch.float32),
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor([int(labels_all[ep])], dtype=torch.long),
        )
        graph.subject_id = subj_id
        graph.epoch_idx  = ep
        data_list.append(graph)

    n_ict = sum(1 for g in data_list if g.y.item() == 1)
    print(f'  subject_{subj_id:02d} ({SUBJECT_TO_PATIENT.get(subj_id,"?"):6s}): '
          f'{len(data_list):4d} epochs  '
          f'(ictal={n_ict}, pre-ictal={len(data_list)-n_ict})')
    return data_list


def print_split_stats(name, graphs):
    n_ict = sum(1 for g in graphs if g.y.item() == 1)
    n_pre = len(graphs) - n_ict
    print(f'  {name:8s}: {len(graphs):5d} graphs  '
          f'ictal={n_ict:4d}  pre-ictal={n_pre:5d}  '
          f'imbalance={n_pre/max(n_ict,1):.1f}:1')


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', required=True)
    parser.add_argument('--features_dir',     required=True)
    parser.add_argument('--output_dir',       required=True)
    args = parser.parse_args()

    connectivity_dir = Path(args.connectivity_dir)
    features_dir     = Path(args.features_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('STEP 5 — BUILD PYTORCH GEOMETRIC DATASET')
    print('=' * 70)
    print('Split: PATIENT-INDEPENDENT')
    print(f"  Train : subjects {PATIENT_SPLITS['train']}")
    print(f"  Val   : subjects {PATIENT_SPLITS['val']}  (early stopping only)")
    print(f"  Test  : subjects {PATIENT_SPLITS['test']}  (reported once at end)")
    print('=' * 70)

    split_graphs = defaultdict(list)
    all_subjects = sorted(
        PATIENT_SPLITS['train'] + PATIENT_SPLITS['val'] + PATIENT_SPLITS['test']
    )

    for subj_id in all_subjects:
        if   subj_id in PATIENT_SPLITS['train']: split = 'train'
        elif subj_id in PATIENT_SPLITS['val']:   split = 'val'
        else:                                     split = 'test'
        split_graphs[split].extend(
            load_subject_graphs(subj_id, connectivity_dir, features_dir)
        )

    print('\n── Split Summary ─────────────────────────────────────────────────')
    for split in ['train', 'val', 'test']:
        print_split_stats(split, split_graphs[split])

    print()
    for split in ['train', 'val', 'test']:
        out_path = output_dir / f'{split}_graphs.pt'
        torch.save(split_graphs[split], out_path)
        print(f'  Saved: {out_path}')

    info = {
        'n_node_features'   : 9,
        'n_nodes'           : 19,
        'node_feature_names': [
            'bp_delta', 'bp_theta', 'bp_alpha', 'bp_beta', 'bp_gamma',
            'dtf_outflow', 'dtf_inflow', 'pdc_outflow', 'pdc_inflow'
        ],
        'edge_convention'   : 'matrix[i,j] = j→i, directed, fully connected',
        'label_mapping'     : {'0': 'pre-ictal/non-ictal', '1': 'ictal'},
        'split_strategy'    : 'patient-independent',
        'patient_splits'    : PATIENT_SPLITS,
        'counts': {
            s: {
                'total'    : len(split_graphs[s]),
                'ictal'    : sum(1 for g in split_graphs[s] if g.y.item() == 1),
                'pre_ictal': sum(1 for g in split_graphs[s] if g.y.item() == 0),
            }
            for s in ['train', 'val', 'test']
        }
    }
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f'  Saved: {output_dir}/dataset_info.json')

    print('\n' + '=' * 70)
    print('✓  Dataset ready.')
    print('Next: python step6_train_gcn.py --dataset_dir <output_dir>')
    print('=' * 70)


if __name__ == '__main__':
    main()