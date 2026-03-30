"""
Step 3 - Feature Extraction
============================
Reads:
  - preprocessed_epochs/subjectXX_epochs.npy        (N, 19, 1024)
  - preprocessed_epochs/subjectXX_labels.npy         (N,)
  - preprocessed_epochs/subjectXX_timefromonset.npy  (N,)
  - preprocessed_epochs/subjectXX_metadata.json
  - connectivity/subjectXX_graphs.npz                (dtf_<band>, pdc_<band> each E,19,19)

Outputs (saved to --outputdir):
  - features_all.npz          -> X, y, subject_ids, patient_ids, feature_names
  - features_all.csv          -> human-readable version
  - feature_summary.txt       -> shapes, class counts, feature groups

Usage:
  python step3_extract_features.py \
      --epochdir preprocessed_epochs \
      --conndir  connectivity \
      --outputdir features \
      --ratio 2          # pre-ictal = ratio * n_ictal (from start of recording)

Patient→Subject mapping (from your figure):
  PAT11->1, PAT13->2, PAT14->3-10, PAT15->11,
  PAT24->12-25, PAT27->26-32, PAT29->33, PAT35->34
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from tqdm import tqdm

warnings.filterwarnings("ignore")

FS          = 256
EPOCH_LEN   = 4.0
N_CHANNELS  = 19
N_SAMPLES   = 1024   # 4s × 256 Hz

CHANNEL_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T3','C3','Cz','C4','T4',
    'T5','P3','Pz','P4','T6',
    'O1','O2'
]

# Left / right hemisphere channel indices for asymmetry index
LEFT_CH  = [0, 3, 7, 8, 12, 13, 17]   # Fp1,F3,T3,C3,T5,P3,O1
RIGHT_CH = [1, 5, 9, 10, 14, 15, 18]  # Fp2,F4,T4,C4,T6,P4,O2

BANDS = {
    'integrated': (0.5, 45.0),
    'delta':      (0.5,  4.0),
    'theta':      (4.0,  8.0),
    'alpha':      (8.0, 15.0),
    'beta':       (15.0, 30.0),
    'gamma1':     (30.0, 45.0),
}

# Patient ID map: subject index (1-based) -> patient string
PATIENT_MAP = {}
for s in [1]:            PATIENT_MAP[s] = 'PAT11'
for s in [2]:            PATIENT_MAP[s] = 'PAT13'
for s in range(3, 11):   PATIENT_MAP[s] = 'PAT14'
for s in [11]:           PATIENT_MAP[s] = 'PAT15'
for s in range(12, 26):  PATIENT_MAP[s] = 'PAT24'
for s in range(26, 33):  PATIENT_MAP[s] = 'PAT27'
for s in [33]:           PATIENT_MAP[s] = 'PAT29'
for s in [34]:           PATIENT_MAP[s] = 'PAT35'


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def time_domain_features(epoch):
    """
    epoch: (19, 1024)
    Returns flat vector + names, per channel:
      mean, std, skewness, kurtosis, line_length, zero_crossing_rate
    """
    feats, names = [], []
    for ci, ch in enumerate(CHANNEL_NAMES):
        x = epoch[ci]
        feats.append(np.mean(x))
        feats.append(np.std(x))
        feats.append(float(skew(x)))
        feats.append(float(kurtosis(x)))
        feats.append(float(np.sum(np.abs(np.diff(x)))))           # line length
        feats.append(float(np.sum(np.diff(np.sign(x)) != 0)))     # ZCR
        for stat in ['mean','std','skew','kurt','linelen','zcr']:
            names.append(f'td_{stat}_{ch}')
    return np.array(feats, dtype=np.float32), names


def hjorth_features(epoch):
    """
    epoch: (19, 1024)
    Returns activity, mobility, complexity per channel.
    """
    feats, names = [], []
    for ci, ch in enumerate(CHANNEL_NAMES):
        x  = epoch[ci]
        dx = np.diff(x)
        d2x = np.diff(dx)

        activity   = np.var(x)
        mob_x      = np.std(dx) / (np.std(x) + 1e-12)
        mob_dx     = np.std(d2x) / (np.std(dx) + 1e-12)
        complexity = mob_dx / (mob_x + 1e-12)

        feats.extend([activity, mob_x, complexity])
        for stat in ['activity','mobility','complexity']:
            names.append(f'hjorth_{stat}_{ch}')
    return np.array(feats, dtype=np.float32), names


def spectral_features(epoch):
    """
    epoch: (19, 1024)
    Returns relative band power per channel per band.
    """
    feats, names = [], []
    for ci, ch in enumerate(CHANNEL_NAMES):
        freqs, psd = welch(epoch[ci], fs=FS, nperseg=256)
        total_power = np.trapz(psd, freqs) + 1e-12
        for band, (flo, fhi) in BANDS.items():
            idx = np.where((freqs >= flo) & (freqs < fhi))[0]
            bp  = np.trapz(psd[idx], freqs[idx]) / total_power
            feats.append(float(bp))
            names.append(f'spec_{band}_{ch}')
    return np.array(feats, dtype=np.float32), names


def graph_features(dtf_bands, pdc_bands):
    """
    dtf_bands, pdc_bands: dict {band_name: (19,19)}
    Returns graph-level connectivity features + names.
    Uses: integrated, alpha, beta bands only (most informative for focal seizures).
    """
    feats, names = [], []
    selected_bands = ['integrated', 'alpha', 'beta']

    for band in selected_bands:
        for metric_name, mat in [('dtf', dtf_bands[band]),
                                  ('pdc', pdc_bands[band])]:
            # Per-node out-degree (source strength): sum over sinks (axis=0, col sum for DTF)
            # Convention: mat[i,j] = influence of SOURCE j on SINK i
            # out-degree of j = sum over i of mat[:,j]  → axis=0 sum
            out_deg = mat.sum(axis=0)   # (19,)  source strength
            in_deg  = mat.sum(axis=1)   # (19,)  sink strength

            feats.extend(out_deg.tolist())
            feats.extend(in_deg.tolist())
            for ch in CHANNEL_NAMES:
                names.append(f'graph_{metric_name}_{band}_outdeg_{ch}')
            for ch in CHANNEL_NAMES:
                names.append(f'graph_{metric_name}_{band}_indeg_{ch}')

            # Global mean connectivity (off-diagonal)
            mask = ~np.eye(N_CHANNELS, dtype=bool)
            global_mean = mat[mask].mean()
            feats.append(float(global_mean))
            names.append(f'graph_{metric_name}_{band}_global_mean')

            # Asymmetry index: (left_out - right_out) / (left_out + right_out)
            left_out  = out_deg[LEFT_CH].mean()
            right_out = out_deg[RIGHT_CH].mean()
            asym = (left_out - right_out) / (left_out + right_out + 1e-12)
            feats.append(float(asym))
            names.append(f'graph_{metric_name}_{band}_asymmetry')

    return np.array(feats, dtype=np.float32), names


def node_features_for_gnn(epoch, dtf_integrated, pdc_integrated):
    """
    Returns per-node feature matrix of shape (19, n_node_feats) for GNN use.
    Node features: [rel_band_powers x6, hjorth x3, td_stats x5, dtf_outdeg, pdc_outdeg]
    = 16 features per node
    """
    node_feats = np.zeros((N_CHANNELS, 16), dtype=np.float32)

    for ci in range(N_CHANNELS):
        x = epoch[ci]
        # Spectral (6 relative band powers)
        freqs, psd = welch(x, fs=FS, nperseg=256)
        total_power = np.trapz(psd, freqs) + 1e-12
        for bi, (band, (flo, fhi)) in enumerate(BANDS.items()):
            idx = np.where((freqs >= flo) & (freqs < fhi))[0]
            node_feats[ci, bi] = np.trapz(psd[idx], freqs[idx]) / total_power

        # Hjorth (3)
        dx, d2x = np.diff(x), np.diff(np.diff(x))
        node_feats[ci, 6] = np.var(x)
        node_feats[ci, 7] = np.std(dx) / (np.std(x) + 1e-12)
        mob_dx = np.std(d2x) / (np.std(dx) + 1e-12)
        node_feats[ci, 8] = mob_dx / (node_feats[ci, 7] + 1e-12)

        # Time-domain (5)
        node_feats[ci, 9]  = float(np.mean(x))
        node_feats[ci, 10] = float(np.std(x))
        node_feats[ci, 11] = float(skew(x))
        node_feats[ci, 12] = float(kurtosis(x))
        node_feats[ci, 13] = float(np.sum(np.abs(np.diff(x))))

        # DTF out-degree & PDC out-degree from integrated band (2)
        node_feats[ci, 14] = dtf_integrated[:, ci].sum()   # out-degree of node ci
        node_feats[ci, 15] = pdc_integrated[:, ci].sum()

    return node_feats   # (19, 16)


# ---------------------------------------------------------------------------
# Sampling: balance classes using temporal distance strategy
# ---------------------------------------------------------------------------

def select_balanced_indices(labels, time_from_onset, ratio=2):
    """
    Select all ictal epochs + `ratio` × n_ictal pre-ictal epochs
    taken from the BEGINNING of the recording (most negative time_from_onset).
    """
    ictal_idx    = np.where(labels == 1)[0]
    preictal_idx = np.where(labels == 0)[0]

    n_ictal  = len(ictal_idx)
    n_select = min(ratio * n_ictal, len(preictal_idx))

    # Sort pre-ictal by time_from_onset ascending (most negative first = farthest from seizure)
    sorted_preictal = preictal_idx[np.argsort(time_from_onset[preictal_idx])]
    selected_pre    = sorted_preictal[:n_select]

    return np.concatenate([ictal_idx, selected_pre])


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_subject(subj_idx, epoch_dir, conn_dir, ratio):
    """
    Returns list of dicts, one per selected epoch:
      flat_features, node_features, label, subject_id, patient_id
    """
    #subj_name = f'subject{subj_idx:02d}'
    subj_name = f'subject_{subj_idx:02d}'

    patient_id = PATIENT_MAP.get(subj_idx, f'PAT_UNKNOWN_{subj_idx}')

    epoch_file = epoch_dir / f'{subj_name}_epochs.npy'
    label_file = epoch_dir / f'{subj_name}_labels.npy'
    tfo_file   = epoch_dir / f'{subj_name}_timefromonset.npy'
    conn_file  = conn_dir  / f'{subj_name}_graphs.npz'

    if not epoch_file.exists():
        print(f'  [SKIP] {subj_name}: epoch file not found')
        return []
    if not conn_file.exists():
        print(f'  [SKIP] {subj_name}: connectivity file not found')
        return []

    epochs         = np.load(epoch_file)           # (N, 19, 1024)
    labels         = np.load(label_file)           # (N,)
    time_from_onset = np.load(tfo_file) if tfo_file.exists() else np.zeros(len(labels))

    conn = np.load(conn_file, allow_pickle=False)
    # conn keys: dtf_<band>, pdc_<band>, labels, indices, ...
    # conn indices maps valid connectivity epochs back to original epoch indices
    conn_indices = conn['indices']                  # (E,)  original epoch indices
    conn_labels  = conn['labels']                   # (E,)

    # Build lookup: original_epoch_idx -> connectivity row idx
    orig_to_conn = {int(orig): ci for ci, orig in enumerate(conn_indices)}

    # Select balanced subset of original epochs
    selected = select_balanced_indices(labels, time_from_onset, ratio=ratio)

    # Only keep selected epochs that also have connectivity computed
    selected = np.array([idx for idx in selected if idx in orig_to_conn])

    if len(selected) == 0:
        print(f'  [WARN] {subj_name}: no valid epochs after intersection with connectivity')
        return []

    # Pre-compute feature names once (they are constant across epochs)
    dummy_epoch = np.zeros((N_CHANNELS, N_SAMPLES), dtype=np.float32)
    dummy_mat   = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float32)
    _, td_names  = time_domain_features(dummy_epoch)
    _, hj_names  = hjorth_features(dummy_epoch)
    _, sp_names  = spectral_features(dummy_epoch)
    dummy_bands  = {b: dummy_mat for b in BANDS}
    _, gr_names  = graph_features(dummy_bands, dummy_bands)
    feature_names = td_names + hj_names + sp_names + gr_names

    records = []
    for orig_idx in selected:
        ci = orig_to_conn[orig_idx]
        epoch = epochs[orig_idx]          # (19, 1024)

        # Load connectivity matrices for this epoch
        dtf_bands = {b: conn[f'dtf_{b}'][ci] for b in BANDS}
        pdc_bands = {b: conn[f'pdc_{b}'][ci] for b in BANDS}

        # --- Flat features for baseline ML ---
        td_f, _  = time_domain_features(epoch)
        hj_f, _  = hjorth_features(epoch)
        sp_f, _  = spectral_features(epoch)
        gr_f, _  = graph_features(dtf_bands, pdc_bands)
        flat_features = np.concatenate([td_f, hj_f, sp_f, gr_f])

        # --- Per-node features for GNN ---
        node_feats = node_features_for_gnn(epoch, dtf_bands['integrated'], pdc_bands['integrated'])

        records.append({
            'flat_features':  flat_features,
            'node_features':  node_feats,          # (19, 16)
            'adj_dtf':        dtf_bands['integrated'],  # (19,19) edge weights
            'adj_pdc':        pdc_bands['integrated'],
            'label':          int(labels[orig_idx]),
            'subject_id':     subj_idx,
            'patient_id':     patient_id,
            'orig_epoch_idx': int(orig_idx),
            'time_from_onset': float(time_from_onset[orig_idx]),
        })

    return records, feature_names


def main():
    parser = argparse.ArgumentParser(description='Step 3 - Feature Extraction')
    parser.add_argument('--epochdir',   required=True, help='preprocessed_epochs dir')
    parser.add_argument('--conndir',    required=True, help='connectivity dir (npz files)')
    parser.add_argument('--outputdir',  required=True, help='output dir for features')
    parser.add_argument('--ratio',      type=int, default=2,
                        help='pre-ictal = ratio × n_ictal (default 2)')
    args = parser.parse_args()

    epoch_dir  = Path(args.epochdir)
    conn_dir   = Path(args.conndir)
    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('STEP 3 — FEATURE EXTRACTION')
    print('=' * 70)
    print(f'Epoch dir   : {epoch_dir}')
    print(f'Conn dir    : {conn_dir}')
    print(f'Output dir  : {output_dir}')
    print(f'Balance ratio: 1 ictal : {args.ratio} pre-ictal (from start of recording)')
    print('=' * 70)

    all_flat, all_node, all_adj_dtf, all_adj_pdc = [], [], [], []
    all_labels, all_subject_ids, all_patient_ids = [], [], []
    all_tfo, feature_names = [], None

    # Find subject files
    epoch_files = sorted(epoch_dir.glob('subject*_epochs.npy'))
    n_subjects  = len(epoch_files)
    print(f'Found {n_subjects} subjects')

    for ep_file in tqdm(epoch_files, desc='Subjects'):
        subj_name = ep_file.stem.replace('_epochs', '')
        #subj_idx  = int(subj_name.replace('subject', ''))
        subj_idx  = int(subj_name.replace('subject_', ''))


        result = process_subject(subj_idx, epoch_dir, conn_dir, args.ratio)
        if not result:
            continue
        records, feat_names = result
        if feature_names is None:
            feature_names = feat_names

        for rec in records:
            all_flat.append(rec['flat_features'])
            all_node.append(rec['node_features'])
            all_adj_dtf.append(rec['adj_dtf'])
            all_adj_pdc.append(rec['adj_pdc'])
            all_labels.append(rec['label'])
            all_subject_ids.append(rec['subject_id'])
            all_patient_ids.append(rec['patient_id'])
            all_tfo.append(rec['time_from_onset'])

    if len(all_flat) == 0:
        print('No features extracted. Check your epoch and connectivity directories.')
        return

    X           = np.stack(all_flat)       # (total_epochs, n_features)
    node_feats  = np.stack(all_node)       # (total_epochs, 19, 16)
    adj_dtf     = np.stack(all_adj_dtf)   # (total_epochs, 19, 19)
    adj_pdc     = np.stack(all_adj_pdc)
    y           = np.array(all_labels,     dtype=np.int64)
    subject_ids = np.array(all_subject_ids, dtype=np.int32)
    patient_ids = np.array(all_patient_ids)
    tfo         = np.array(all_tfo,        dtype=np.float32)

    # --- Save ---
    np.savez_compressed(
        output_dir / 'features_all.npz',
        X=X,
        node_features=node_feats,
        adj_dtf=adj_dtf,
        adj_pdc=adj_pdc,
        y=y,
        subject_ids=subject_ids,
        patient_ids=patient_ids,
        time_from_onset=tfo,
        feature_names=np.array(feature_names, dtype=object),
    )

    # Human-readable CSV (flat features only)
    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, 'label',      y)
    df.insert(1, 'subject_id', subject_ids)
    df.insert(2, 'patient_id', patient_ids)
    df.insert(3, 'time_from_onset', tfo)
    df.to_csv(output_dir / 'features_all.csv', index=False)

    # --- Summary ---
    n_ictal    = int((y == 1).sum())
    n_preictal = int((y == 0).sum())
    n_patients = len(np.unique(patient_ids))

    summary = f"""
FEATURE EXTRACTION SUMMARY
===========================
Total epochs    : {len(y)}
  Ictal (1)     : {n_ictal}  ({100*n_ictal/len(y):.1f}%)
  Pre-ictal (0) : {n_preictal}  ({100*n_preictal/len(y):.1f}%)
Subjects        : {len(np.unique(subject_ids))}
Patients (LOPO) : {n_patients}  → {np.unique(patient_ids).tolist()}

Feature vector  : {X.shape[1]} flat features
  Time-domain   : {len([f for f in feature_names if f.startswith('td_')])}
  Hjorth        : {len([f for f in feature_names if f.startswith('hjorth_')])}
  Spectral      : {len([f for f in feature_names if f.startswith('spec_')])}
  Graph (DTF/PDC): {len([f for f in feature_names if f.startswith('graph_')])}

Node features   : {node_feats.shape}  (epochs, channels, per-node-feats)
Adjacency (DTF) : {adj_dtf.shape}
Adjacency (PDC) : {adj_pdc.shape}

Saved:
  features_all.npz  (X, node_features, adj_dtf, adj_pdc, y, subject_ids, patient_ids)
  features_all.csv  (flat features, human-readable)

Next steps:
  python step4_baseline_ml.py  --featfile features/features_all.npz
  python step5_gnn_supervised.py --featfile features/features_all.npz
"""
    print(summary)
    with open(output_dir / 'feature_summary.txt', 'w') as f:
        f.write(summary)


if __name__ == '__main__':
    main()
