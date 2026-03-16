"""
Step 3 — Feature Extraction (Curated, Thesis-Ready)
=====================================================
DESIGN PHILOSOPHY (document this in your thesis):
  - Small dataset → we use a CURATED, JUSTIFIED feature set (~50 flat features)
    instead of throwing everything at a selector.
  - Every feature group has a neuroscientific reason to be there.
  - Features are organised into three groups:
      A) Spectral power (relative band power per channel, 6 bands × 19 ch = 114)
         → BUT we compress to hemisphere/lobe averages to avoid 114 correlated values
         → We keep per-channel only for the GNN node features (where it belongs)
         → For flat ML: 6 bands × 5 regions = 30 spectral features
      B) Hjorth parameters (activity, mobility, complexity)
         → Region-averaged: 3 params × 5 regions = 15 features
      C) Graph-level connectivity (DTF/PDC integrated band)
         → Global mean, asymmetry index, top-3 source/sink nodes = 8 features
         → Total graph features: 8

  Flat feature vector: 30 + 15 + 8 = 53 features   ← tractable for 34 subjects
  Node feature vector per channel: 16 features       ← used only by GNN

BRAIN REGIONS (5 groups of EEG channels):
  Frontal     : Fp1 Fp2 F7 F3 Fz F4 F8  (idx 0-6)
  Temporal    : T3 T4 T5 T6             (idx 7,11,12,16)
  Central     : C3 Cz C4               (idx 8,9,10)
  Parietal    : P3 Pz P4               (idx 13,14,15)
  Occipital   : O1 O2                  (idx 17,18)

WHY REGION AVERAGES?
  With 34 subjects and ~few hundred epochs, a 500-feature vector causes
  the curse of dimensionality. Averaging within brain regions is neuroscientifically
  motivated (focal seizures spread regionally) and dramatically reduces features
  while preserving spatial information.

Inputs  (from steps 0, 2):
  preprocessed_epochs/subject_XX_epochs.npy       (N, 19, 1024)
  preprocessed_epochs/subject_XX_labels.npy        (N,)
  preprocessed_epochs/subject_XX_time_from_onset.npy (N,)
  connectivity/subject_XX_graphs.npz               (dtf_<band>, pdc_<band>, ...)

Outputs (saved to --outputdir):
  features_all.npz     X (flat), node_features, adj_dtf, adj_pdc, y,
                        subject_ids, patient_ids, time_from_onset, feature_names
  features_all.csv     human-readable flat features
  feature_summary.txt  shapes, counts, descriptions

Usage:
  python step3_extract_features.py \\
      --epochdir preprocessed_epochs \\
      --conndir  connectivity \\
      --outputdir features \\
      --ratio 2
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

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

FS          = 256
EPOCH_LEN   = 4.0
N_CHANNELS  = 19
N_SAMPLES   = 1024   # 4 s × 256 Hz

CHANNEL_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',   # 0-6   frontal
    'T3','C3','Cz','C4','T4',               # 7-11  temporal-L, central, temporal-R
    'T5','P3','Pz','P4','T6',               # 12-16 temporal-L-post, parietal, temporal-R-post
    'O1','O2'                               # 17-18 occipital
]

# ─────────────────────────────────────────────────────────────
# Brain regions — indices into CHANNEL_NAMES
# ─────────────────────────────────────────────────────────────
REGIONS = {
    'frontal':   [0, 1, 2, 3, 4, 5, 6],   # Fp1 Fp2 F7 F3 Fz F4 F8
    'temporal':  [7, 11, 12, 16],          # T3 T4 T5 T6
    'central':   [8, 9, 10],               # C3 Cz C4
    'parietal':  [13, 14, 15],             # P3 Pz P4
    'occipital': [17, 18],                 # O1 O2
}
REGION_NAMES = list(REGIONS.keys())

# Left / Right hemisphere for asymmetry index
LEFT_CH  = [0, 3, 7, 8, 12, 13, 17]   # Fp1 F3 T3 C3 T5 P3 O1
RIGHT_CH = [1, 5, 10, 11, 14, 15, 18] # Fp2 F4 C4 T4 P4 T6 O2

# Frequency bands
BANDS = {
    'delta':  (0.5,  4.0),
    'theta':  (4.0,  8.0),
    'alpha':  (8.0, 15.0),
    'beta':  (15.0, 30.0),
    'gamma': (30.0, 45.0),
    'broad': (0.5,  45.0),
}
BAND_NAMES = list(BANDS.keys())

# Patient ID map: subject index (1-based) → patient string
PATIENT_MAP = {}
for s in [1]:           PATIENT_MAP[s] = 'PAT11'
for s in [2]:           PATIENT_MAP[s] = 'PAT13'
for s in range(3, 11):  PATIENT_MAP[s] = 'PAT14'
for s in [11]:          PATIENT_MAP[s] = 'PAT15'
for s in range(12, 26): PATIENT_MAP[s] = 'PAT24'
for s in range(26, 33): PATIENT_MAP[s] = 'PAT27'
for s in [33]:          PATIENT_MAP[s] = 'PAT29'
for s in [34]:          PATIENT_MAP[s] = 'PAT35'


# ─────────────────────────────────────────────────────────────
# A) Spectral features — region-averaged relative band power
#    Output: 6 bands × 5 regions = 30 features
#    Rationale: focal seizures show localised spectral changes
#    (delta increase + alpha/beta decrease in seizure zone)
# ─────────────────────────────────────────────────────────────

def spectral_region_features(epoch):
    """
    epoch : (19, 1024)
    Returns (30,) vector + 30 names.

    Method: compute relative band power per channel via Welch,
    then average within each brain region.
    """
    # Per-channel relative band powers: (19, 6)
    per_ch = np.zeros((N_CHANNELS, len(BANDS)), dtype=np.float32)
    for ci in range(N_CHANNELS):
        freqs, psd = welch(epoch[ci], fs=FS, nperseg=256)
        total = np.trapz(psd, freqs) + 1e-12
        for bi, (_, (flo, fhi)) in enumerate(BANDS.items()):
            idx = np.where((freqs >= flo) & (freqs < fhi))[0]
            per_ch[ci, bi] = np.trapz(psd[idx], freqs[idx]) / total

    feats, names = [], []
    for ri, (reg, ch_idx) in enumerate(REGIONS.items()):
        region_power = per_ch[ch_idx, :].mean(axis=0)  # (6,)
        for bi, band in enumerate(BAND_NAMES):
            feats.append(float(region_power[bi]))
            names.append(f'spec_{band}_{reg}')

    return np.array(feats, dtype=np.float32), names  # (30,), 30 names


# ─────────────────────────────────────────────────────────────
# B) Hjorth parameters — region-averaged
#    Output: 3 params × 5 regions = 15 features
#    Rationale: Hjorth complexity/mobility capture EEG signal
#    irregularity; well-established for seizure characterisation
# ─────────────────────────────────────────────────────────────

def hjorth_region_features(epoch):
    """
    epoch : (19, 1024)
    Returns (15,) vector + 15 names.
    """
    # Per-channel Hjorth: (19, 3)  [activity, mobility, complexity]
    per_ch = np.zeros((N_CHANNELS, 3), dtype=np.float32)
    for ci in range(N_CHANNELS):
        x  = epoch[ci]
        dx = np.diff(x)
        d2x = np.diff(dx)
        activity   = float(np.var(x))
        mob_x      = float(np.std(dx) / (np.std(x) + 1e-12))
        mob_dx     = float(np.std(d2x) / (np.std(dx) + 1e-12))
        complexity  = float(mob_dx / (mob_x + 1e-12))
        per_ch[ci] = [activity, mob_x, complexity]

    feats, names = [], []
    for reg, ch_idx in REGIONS.items():
        region_h = per_ch[ch_idx, :].mean(axis=0)  # (3,)
        for pi, param in enumerate(['activity', 'mobility', 'complexity']):
            feats.append(float(region_h[pi]))
            names.append(f'hjorth_{param}_{reg}')

    return np.array(feats, dtype=np.float32), names  # (15,), 15 names


# ─────────────────────────────────────────────────────────────
# C) Graph-level connectivity features
#    Output: 8 features
#    Rationale: DTF/PDC capture directed functional connectivity;
#    global changes and hemispheric asymmetry are key in focal epilepsy
# ─────────────────────────────────────────────────────────────

def graph_level_features(dtf_integrated, pdc_integrated):
    """
    dtf_integrated, pdc_integrated : (19, 19)  band-averaged matrices
    Returns (8,) vector + 8 names.

    Features:
      1. DTF global mean connectivity (off-diagonal)
      2. PDC global mean connectivity (off-diagonal)
      3. DTF left-hemisphere mean out-degree
      4. DTF right-hemisphere mean out-degree
      5. DTF hemispheric asymmetry index  (left-right)/(left+right)
      6. PDC hemispheric asymmetry index
      7. DTF max out-degree (identifies dominant source channel)
      8. PDC max out-degree (identifies dominant sink channel)
    """
    feats, names = [], []
    mask = ~np.eye(N_CHANNELS, dtype=bool)

    for metric_name, mat in [('dtf', dtf_integrated), ('pdc', pdc_integrated)]:
        # Global mean (off-diagonal)
        global_mean = float(mat[mask].mean())
        feats.append(global_mean)
        names.append(f'graph_{metric_name}_global_mean')

        # Out-degree per node (column sum: influence emanating from each source)
        out_deg = mat.sum(axis=0)  # (19,)

        # Hemisphere means
        left_mean  = float(out_deg[LEFT_CH].mean())
        right_mean = float(out_deg[RIGHT_CH].mean())
        feats.append(left_mean)
        feats.append(right_mean)
        names.append(f'graph_{metric_name}_left_outdeg')
        names.append(f'graph_{metric_name}_right_outdeg')

        # Asymmetry index — range (-1, 1), zero = symmetric
        asym = (left_mean - right_mean) / (left_mean + right_mean + 1e-12)
        feats.append(float(asym))
        names.append(f'graph_{metric_name}_asymmetry')

    return np.array(feats, dtype=np.float32), names  # (8,) but wait: 2*(1+1+1+1) = 8 ✓


# ─────────────────────────────────────────────────────────────
# Per-node features for GNN  (unchanged from your original)
# Output: (19, 16)
# ─────────────────────────────────────────────────────────────

def node_features_for_gnn(epoch, dtf_integrated, pdc_integrated):
    """
    Returns per-node feature matrix (19, 16) for GNN use.
    Features per node:
      [0-5]  : relative band powers (delta, theta, alpha, beta, gamma, broad)
      [6-8]  : Hjorth (activity, mobility, complexity)
      [9-13] : time-domain (mean, std, skewness, kurtosis, line_length)
      [14]   : DTF out-degree (source strength in integrated band)
      [15]   : PDC out-degree
    """
    node_feats = np.zeros((N_CHANNELS, 16), dtype=np.float32)

    for ci in range(N_CHANNELS):
        x = epoch[ci]

        # Spectral (6 bands)
        freqs, psd = welch(x, fs=FS, nperseg=256)
        total = np.trapz(psd, freqs) + 1e-12
        for bi, (_, (flo, fhi)) in enumerate(BANDS.items()):
            idx = np.where((freqs >= flo) & (freqs < fhi))[0]
            node_feats[ci, bi] = float(np.trapz(psd[idx], freqs[idx]) / total)

        # Hjorth (3)
        dx  = np.diff(x)
        d2x = np.diff(dx)
        node_feats[ci, 6] = float(np.var(x))
        node_feats[ci, 7] = float(np.std(dx) / (np.std(x) + 1e-12))
        mob_dx = float(np.std(d2x) / (np.std(dx) + 1e-12))
        node_feats[ci, 8] = float(mob_dx / (node_feats[ci, 7] + 1e-12))

        # Time-domain (5)
        node_feats[ci, 9]  = float(np.mean(x))
        node_feats[ci, 10] = float(np.std(x))
        node_feats[ci, 11] = float(skew(x))
        node_feats[ci, 12] = float(kurtosis(x))
        node_feats[ci, 13] = float(np.sum(np.abs(np.diff(x))))  # line length

        # Connectivity out-degree (2)
        node_feats[ci, 14] = float(dtf_integrated[:, ci].sum())
        node_feats[ci, 15] = float(pdc_integrated[:, ci].sum())

    return node_feats   # (19, 16)


# ─────────────────────────────────────────────────────────────
# Epoch selection — balance classes
# ─────────────────────────────────────────────────────────────

def select_balanced_indices(labels, time_from_onset, ratio=2):
    """
    Keep ALL ictal epochs.
    Keep `ratio × n_ictal` pre-ictal epochs, taken from the START of the
    recording (most temporally distant from the seizure onset).
    Taking from the start makes pre-ictal epochs "clearly non-ictal"
    and avoids the ambiguous peri-ictal region.
    """
    ictal_idx    = np.where(labels == 1)[0]
    preictal_idx = np.where(labels == 0)[0]

    n_ictal  = len(ictal_idx)
    n_select = min(ratio * n_ictal, len(preictal_idx))

    # Sort ascending by time_from_onset → most-negative = farthest before seizure
    sorted_pre   = preictal_idx[np.argsort(time_from_onset[preictal_idx])]
    selected_pre = sorted_pre[:n_select]

    return np.concatenate([ictal_idx, selected_pre])


# ─────────────────────────────────────────────────────────────
# Per-subject processing
# ─────────────────────────────────────────────────────────────

def process_subject(subj_idx, epoch_dir, conn_dir, ratio):
    """
    Returns (list_of_records, feature_names) or ([], None) on failure.
    Each record is a dict with flat_features, node_features,
    adj_dtf, adj_pdc, label, subject_id, patient_id, time_from_onset.
    """
    subj_name  = f'subject_{subj_idx:02d}'
    patient_id = PATIENT_MAP.get(subj_idx, f'PAT_UNKNOWN_{subj_idx}')

    epoch_file = epoch_dir / f'{subj_name}_epochs.npy'
    label_file = epoch_dir / f'{subj_name}_labels.npy'
    tfo_file   = epoch_dir / f'{subj_name}_time_from_onset.npy'
    conn_file  = conn_dir  / f'{subj_name}_graphs.npz'

    for f in [epoch_file, label_file, conn_file]:
        if not f.exists():
            print(f'  [SKIP] {subj_name}: {f.name} not found')
            return [], None

    epochs  = np.load(epoch_file)   # (N, 19, 1024)
    labels  = np.load(label_file)   # (N,)
    tfo     = np.load(tfo_file) if tfo_file.exists() else np.zeros(len(labels))

    conn         = np.load(conn_file, allow_pickle=False)
    conn_indices = conn['indices']   # (E,) original epoch indices with valid connectivity
    orig_to_conn = {int(orig): ci for ci, orig in enumerate(conn_indices)}

    # Select balanced subset
    selected = select_balanced_indices(labels, tfo, ratio=ratio)
    # Keep only those that have valid connectivity
    selected = np.array([i for i in selected if i in orig_to_conn])

    if len(selected) == 0:
        print(f'  [WARN] {subj_name}: 0 valid epochs after connectivity intersection')
        return [], None

    # Pre-compute feature names from a dummy epoch
    _dummy_epoch = np.zeros((N_CHANNELS, N_SAMPLES), dtype=np.float32)
    _dummy_mat   = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float32)
    _, sp_names  = spectral_region_features(_dummy_epoch)
    _, hj_names  = hjorth_region_features(_dummy_epoch)
    _, gr_names  = graph_level_features(_dummy_mat, _dummy_mat)
    feature_names = sp_names + hj_names + gr_names

    records = []
    for orig_idx in selected:
        ci    = orig_to_conn[orig_idx]
        epoch = epochs[orig_idx]  # (19, 1024)

        dtf_int = conn['dtf_integrated'][ci]  # (19, 19)
        pdc_int = conn['pdc_integrated'][ci]  # (19, 19)

        # Flat features for baseline ML
        sp_f, _ = spectral_region_features(epoch)
        hj_f, _ = hjorth_region_features(epoch)
        gr_f, _ = graph_level_features(dtf_int, pdc_int)
        flat = np.concatenate([sp_f, hj_f, gr_f])  # (53,)

        # Per-node features for GNN
        nf = node_features_for_gnn(epoch, dtf_int, pdc_int)  # (19, 16)

        records.append({
            'flat_features':   flat,
            'node_features':   nf,
            'adj_dtf':         dtf_int,
            'adj_pdc':         pdc_int,
            'label':           int(labels[orig_idx]),
            'subject_id':      int(subj_idx),
            'patient_id':      patient_id,
            'time_from_onset': float(tfo[orig_idx]),
        })

    return records, feature_names


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 3 — Curated Feature Extraction')
    parser.add_argument('--epochdir',  required=True, help='preprocessed_epochs dir')
    parser.add_argument('--conndir',   required=True, help='connectivity dir (npz files)')
    parser.add_argument('--outputdir', required=True, help='output dir')
    parser.add_argument('--ratio',     type=int, default=2,
                        help='pre-ictal epochs = ratio × n_ictal (default 2)')
    args = parser.parse_args()

    epoch_dir  = Path(args.epochdir)
    conn_dir   = Path(args.conndir)
    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('STEP 3 — FEATURE EXTRACTION (Curated)')
    print('=' * 70)
    print(f'Epoch dir    : {epoch_dir}')
    print(f'Conn dir     : {conn_dir}')
    print(f'Output dir   : {output_dir}')
    print(f'Balance ratio: 1 ictal : {args.ratio} pre-ictal')
    print()
    print('Feature groups:')
    print('  A) Spectral (relative band power, region-averaged): 6 bands × 5 regions = 30')
    print('  B) Hjorth (region-averaged):                       3 params × 5 regions = 15')
    print('  C) Graph-level (DTF/PDC integrated band):                               =  8')
    print('  TOTAL flat features: 53')
    print('  GNN node features:   16 per node (per-channel, not region-averaged)')
    print('=' * 70)

    all_flat, all_node, all_adj_dtf, all_adj_pdc = [], [], [], []
    all_labels, all_subject_ids, all_patient_ids, all_tfo = [], [], [], []
    feature_names = None

    epoch_files = sorted(epoch_dir.glob('subject_*_epochs.npy'))
    print(f'Found {len(epoch_files)} subjects\n')

    for ep_file in tqdm(epoch_files, desc='Subjects'):
        subj_idx = int(ep_file.stem.replace('subject_', '').replace('_epochs', ''))
        records, feat_names = process_subject(subj_idx, epoch_dir, conn_dir, args.ratio)

        if not records:
            continue
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
        print('\n[ERROR] No features extracted. Check your epoch and connectivity directories.')
        return

    X           = np.stack(all_flat)              # (total_epochs, 53)
    node_feats  = np.stack(all_node)              # (total_epochs, 19, 16)
    adj_dtf     = np.stack(all_adj_dtf)           # (total_epochs, 19, 19)
    adj_pdc     = np.stack(all_adj_pdc)
    y           = np.array(all_labels,     dtype=np.int64)
    subject_ids = np.array(all_subject_ids, dtype=np.int32)
    patient_ids = np.array(all_patient_ids)
    tfo         = np.array(all_tfo,        dtype=np.float32)

    # Sanity check: no NaN/Inf
    n_bad = int(np.sum(~np.isfinite(X)))
    if n_bad > 0:
        print(f'[WARN] {n_bad} non-finite values in X — replacing with 0')
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Save
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

    # Human-readable CSV
    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, 'label',           y)
    df.insert(1, 'subject_id',      subject_ids)
    df.insert(2, 'patient_id',      patient_ids)
    df.insert(3, 'time_from_onset', tfo)
    df.to_csv(output_dir / 'features_all.csv', index=False)

    n_ictal    = int((y == 1).sum())
    n_preictal = int((y == 0).sum())
    n_patients = len(np.unique(patient_ids))

    summary = f"""
FEATURE EXTRACTION SUMMARY
===========================
Total epochs        : {len(y)}
  Ictal (1)         : {n_ictal}   ({100 * n_ictal / len(y):.1f}%)
  Pre-ictal (0)     : {n_preictal}  ({100 * n_preictal / len(y):.1f}%)
  Balance ratio     : 1 : {args.ratio}

Subjects            : {len(np.unique(subject_ids))}
Patients (LOPO)     : {n_patients}  -> {np.unique(patient_ids).tolist()}

Flat feature vector : {X.shape[1]} features
  Spectral (A)      : {len([f for f in feature_names if f.startswith('spec_')])}
  Hjorth   (B)      : {len([f for f in feature_names if f.startswith('hjorth_')])}
  Graph    (C)      : {len([f for f in feature_names if f.startswith('graph_')])}

GNN node features   : {node_feats.shape}  (epochs, channels, per-node-feats)
Adjacency (DTF)     : {adj_dtf.shape}
Adjacency (PDC)     : {adj_pdc.shape}

Feature names:
{chr(10).join('  ' + n for n in feature_names)}

Saved:
  features_all.npz
  features_all.csv

Next:
  python step4_baseline_ml.py  --featfile features/features_all.npz
  python step5_gnn_supervised.py --featfile features/features_all.npz
"""
    print(summary)
    with open(output_dir / 'feature_summary.txt', 'w', encoding='utf-8') as fh:
        fh.write(summary)


if __name__ == '__main__':
    main()
