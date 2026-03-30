"""
Step 5 — Build PyTorch Geometric Dataset
=========================================
Reads per-subject connectivity matrices and node features, and assembles
a list of torch_geometric.data.Data objects ready for GNN training.

Pre-ictal sampling strategy (clinically motivated):
    For each subject, keep ALL ictal epochs and select pre-ictal epochs
    from the START of the recording (most negative time_from_onset first).
    Limit pre-ictal to ratio × n_ictal epochs.

Patient-independent split:
    Train : subjects 3-10 (PAT14), 11 (PAT15), 12-25 (PAT24), 34 (PAT35)
    Val   : subject 33 (PAT29)   — early stopping only
    Test  : subjects 1, 2        — reported ONCE at the end

Usage:
    # DTF edges (default)
    python step5_build_dataset.py \
        --connectivity_dir F:/FORTH_Final_Thesis/FORTH-Thesis/connectivity \
        --features_dir     F:/FORTH_Final_Thesis/FORTH-Thesis/node_features \
        --output_dir       F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/dataset_dtf \
        --edge_type dtf

    # PDC edges
    python step5_build_dataset.py \
        --connectivity_dir F:/FORTH_Final_Thesis/FORTH-Thesis/connectivity \
        --features_dir     F:/FORTH_Final_Thesis/FORTH-Thesis/node_features \
        --output_dir       F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/dataset_pdc \
        --edge_type pdc
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
    raise ImportError('pip install torch torch_geometric')


# ── config ────────────────────────────────────────────────────────────────────

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

def build_edge_index_and_weights(weight_matrix: np.ndarray):
    """
    weight_matrix : (19, 19)  diagonal=0, convention matrix[i,j] = j→i
    Returns edge_index (2, 342) and edge_attr (342, 1).
    """
    K = weight_matrix.shape[0]
    sources, sinks, weights = [], [], []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            sources.append(j)
            sinks.append(i)
            weights.append(weight_matrix[i, j])
    return (
        torch.tensor([sources, sinks], dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32).unsqueeze(1),
    )


def load_subject_graphs(subj_id, connectivity_dir, features_dir,
                        ratio=2, edge_type='dtf'):
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
    node_features = np.load(features_file)        # (n_valid, 19, 9)
    labels_all    = graphs['labels']              # (n_valid,)
    n_valid       = len(labels_all)

    # ── select edge weight matrix ─────────────────────────────────────────────
    key = 'dtf_integrated' if edge_type == 'dtf' else 'pdc_integrated'
    if key not in graphs:
        print(f'  [SKIP] {subject_name}: {key} not found in graphs.npz')
        return []
    connectivity_all = graphs[key]                # (n_valid, 19, 19)

    # ── time_from_onset ───────────────────────────────────────────────────────
    if 'time_from_onset' in graphs:
        time_from_onset = graphs['time_from_onset']
    else:
        first_ictal = np.where(labels_all == 1)[0]
        offset = first_ictal[0] if len(first_ictal) > 0 else 0
        time_from_onset = (np.arange(n_valid) - offset) * 4.0
        print(f'  [{subject_name}] time_from_onset reconstructed from labels')

    # ── training_mask ─────────────────────────────────────────────────────────
    if 'training_mask' in graphs:
        raw_mask = graphs['training_mask']
        mask = (raw_mask[graphs['indices'].astype(int)]
                if 'indices' in graphs else raw_mask[:n_valid])
    else:
        mask = np.ones(n_valid, dtype=bool)

    assert node_features.shape[0] == n_valid, (
        f'{subject_name}: feature/connectivity mismatch '
        f'({node_features.shape[0]} vs {n_valid})'
    )

    # ── sampling ──────────────────────────────────────────────────────────────
    ictal_indices    = [i for i in range(n_valid) if mask[i] and labels_all[i] == 1]
    preictal_indices = [i for i in range(n_valid) if mask[i] and labels_all[i] == 0]

    n_ictal   = len(ictal_indices)
    n_pre_max = ratio * n_ictal

    preictal_sorted   = sorted(preictal_indices, key=lambda i: time_from_onset[i])
    selected_preictal = preictal_sorted[:n_pre_max]
    selected_indices  = ictal_indices + selected_preictal

    # ── build Data objects ────────────────────────────────────────────────────
    data_list = []
    for ep in selected_indices:
        edge_index, edge_attr = build_edge_index_and_weights(connectivity_all[ep])
        graph = Data(
            x          = torch.tensor(node_features[ep], dtype=torch.float32),
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor([int(labels_all[ep])], dtype=torch.long),
        )
        graph.subject_id      = subj_id
        graph.epoch_idx       = ep
        graph.time_from_onset = float(time_from_onset[ep])
        data_list.append(graph)

    n_kept_pre = len(selected_preictal)
    n_dropped  = len(preictal_indices) - n_kept_pre
    print(f'  subject_{subj_id:02d} ({SUBJECT_TO_PATIENT.get(subj_id,"?"):6s}): '
          f'{len(data_list):4d} graphs  '
          f'(ictal={n_ictal}, pre-ictal kept={n_kept_pre} '
          f'[from start], dropped={n_dropped})')
    return data_list


def print_split_stats(name, graphs):
    n_ict = sum(1 for g in graphs if g.y.item() == 1)
    n_pre = len(graphs) - n_ict
    print(f'  {name:8s}: {len(graphs):5d} graphs  '
          f'ictal={n_ict:4d}  pre-ictal={n_pre:5d}  '
          f'ratio={n_pre/max(n_ict,1):.1f}:1')


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build PyTorch Geometric dataset for epilepsy GNN')
    parser.add_argument('--connectivity_dir', required=True)
    parser.add_argument('--features_dir',     required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--ratio',     type=int,  default=2,
                        help='Pre-ictal:ictal ratio (default=2)')
    parser.add_argument('--edge_type', default='dtf', choices=['dtf', 'pdc'],
                        help='Which connectivity matrix to use as edge weights (default=dtf)')
    args = parser.parse_args()

    connectivity_dir = Path(args.connectivity_dir)
    features_dir     = Path(args.features_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('STEP 5 — BUILD PYTORCH GEOMETRIC DATASET')
    print('=' * 70)
    print(f'Edge weights     : {args.edge_type.upper()}')
    print(f'Pre-ictal ratio  : {args.ratio}:1  (from START of recording)')
    print('Split            : PATIENT-INDEPENDENT')
    print(f"  Train : {PATIENT_SPLITS['train']}")
    print(f"  Val   : {PATIENT_SPLITS['val']}  (early stopping only)")
    print(f"  Test  : {PATIENT_SPLITS['test']}  (reported once at end)")
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
            load_subject_graphs(
                subj_id, connectivity_dir, features_dir,
                ratio=args.ratio, edge_type=args.edge_type,
            )
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
            'dtf_outflow', 'dtf_inflow', 'pdc_outflow', 'pdc_inflow',
        ],
        'edge_type'         : args.edge_type,
        'edge_convention'   : 'matrix[i,j] = j→i, directed, fully connected',
        'label_mapping'     : {'0': 'pre-ictal/non-ictal', '1': 'ictal'},
        'split_strategy'    : 'patient-independent',
        'preictal_sampling' : {
            'strategy' : 'from_start_of_recording',
            'ratio'    : args.ratio,
            'rationale': (
                'Pre-ictal epochs taken from the start of recording '
                '(most negative time_from_onset) to maximise distance '
                'from seizure onset.'
            ),
        },
        'patient_splits': PATIENT_SPLITS,
        'counts': {
            s: {
                'total'    : len(split_graphs[s]),
                'ictal'    : sum(1 for g in split_graphs[s] if g.y.item() == 1),
                'pre_ictal': sum(1 for g in split_graphs[s] if g.y.item() == 0),
            }
            for s in ['train', 'val', 'test']
        },
    }
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f'  Saved: {output_dir}/dataset_info.json')

    print('\n' + '=' * 70)
    print('✓  Dataset ready.')
    print(f'Next: python step6_train_gcn.py --dataset_dir {output_dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
