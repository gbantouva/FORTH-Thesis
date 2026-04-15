"""
Step 3 — Feature Extraction (Curated, Thesis-Ready)
=====================================================
DESIGN PHILOSOPHY:
  - Small dataset : use a CURATED, JUSTIFIED feature set (~50 flat features)
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
         → Global mean, asymmetry index, hemisphere out-degrees = 8 features

  Flat feature vector : 30 + 15 + 8 = 53 features  ← tractable for 34 subjects
  Node feature vector : 16 features per channel     ← used only by GNN

BRAIN REGIONS (5 groups of EEG channels):
  Frontal   : Fp1 Fp2 F7 F3 Fz F4 F8  (idx 0-6)
  Temporal  : T3 T4 T5 T6             (idx 7,11,12,16)
  Central   : C3 Cz C4               (idx 8,9,10)
  Parietal  : P3 Pz P4               (idx 13,14,15)
  Occipital : O1 O2                  (idx 17,18)

PATIENT MAP (loaded from dataset_metadata.json — NOT hardcoded):
  Patient IDs are read directly from the metadata file produced by step 0.
  This avoids silent errors if subject ordering changes.

  Actual patients in this dataset (from metadata):
    PAT_11  → subject 1
    PAT_13  → subject 2
    PAT_14  → subjects 3-10   (8 recordings, same patient)
    PAT_15  → subject 11
    PAT_24  → subjects 12-25  (14 recordings)
    PAT_27  → subjects 26-32  (7 recordings)
    PAT_29  → subject 33
    PAT_35  → subject 34
  Total: 8 unique patients → 8-fold LOPO cross-validation

  IMPORTANT: Patient ID is extracted as the PATIENT part only (e.g. "PAT_14")
  from the full filename (e.g. "PAT_14_EEG_160.mat") so all recordings from
  the same patient get the same group label.

Inputs  (from steps 0, 2):
  final_preprocessed_epochs/subject_XX_epochs.npy          (N, 19, 1024)
  final_preprocessed_epochs/subject_XX_labels.npy           (N,)
  final_preprocessed_epochs/subject_XX_time_from_onset.npy  (N,)
  final_preprocessed_epochs/dataset_metadata.json           ← patient map
  final_connectivity/subject_XX_graphs.npz                  (dtf_<band>, pdc_<band>, ...)

Outputs (saved to --outputdir):
  features_all.npz     X (flat), node_features, raw_epochs, adj_dtf, adj_pdc,
                        y, subject_ids, patient_ids, time_from_onset, feature_names
  features_all.csv     human-readable flat features
  feature_summary.txt  shapes, counts, descriptions

Usage:
  python step3_extract_features.py \\
      --epochdir  F:\\FORTH_Final_Thesis\\FORTH-Thesis\\final_preprocessed_epochs \\
      --conndir   F:\\FORTH_Final_Thesis\\FORTH-Thesis\\final_connectivity \\
      --outputdir F:\\FORTH_Final_Thesis\\FORTH-Thesis\\features \\
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

FS         = 256
EPOCH_LEN  = 4.0
N_CHANNELS = 19
N_SAMPLES  = 1024   # 4 s × 256 Hz

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',   # 0-6   frontal
    'T3',  'C3',  'Cz',  'C4',  'T4',                  # 7-11  temporal-L, central, temporal-R
    'T5',  'P3',  'Pz',  'P4',  'T6',                  # 12-16 temporal-L-post, parietal, temporal-R-post
    'O1',  'O2',                                        # 17-18 occipital
]

# Brain regions — indices into CHANNEL_NAMES
REGIONS = {
    'frontal':   [0, 1, 2, 3, 4, 5, 6],   # Fp1 Fp2 F7 F3 Fz F4 F8
    'temporal':  [7, 11, 12, 16],           # T3 T4 T5 T6
    'central':   [8, 9, 10],               # C3 Cz C4
    'parietal':  [13, 14, 15],             # P3 Pz P4
    'occipital': [17, 18],                 # O1 O2
}
REGION_NAMES = list(REGIONS.keys())

# Left / Right hemisphere for asymmetry index
LEFT_CH  = [0, 3, 7, 8, 12, 13, 17]    # Fp1 F3 T3 C3 T5 P3 O1
RIGHT_CH = [1, 5, 10, 11, 14, 15, 18]  # Fp2 F4 C4 T4 P4 T6 O2

# Frequency bands
BANDS = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 15.0),
    'beta':  (15.0, 30.0),
    'gamma': (30.0, 45.0),
    'broad': (0.5,  45.0),
}
BAND_NAMES = list(BANDS.keys())


# ─────────────────────────────────────────────────────────────
# Patient map — loaded from metadata, NOT hardcoded
# ─────────────────────────────────────────────────────────────

def build_patient_map(epoch_dir: Path) -> dict:
    """
    Load patient IDs from dataset_metadata.json produced by step 0.

    The patient_id field in the JSON looks like "PAT_14_EEG_160.mat".
    We extract only the patient part ("PAT_14") so that all recordings
    from the same patient share the same group label for LOPO.

    Returns: dict  { subject_id (int) → patient_label (str) }
    e.g. { 3: 'PAT_14', 4: 'PAT_14', ..., 11: 'PAT_15', ... }
    """
    meta_file = epoch_dir / 'dataset_metadata.json'
    if not meta_file.exists():
        raise FileNotFoundError(
            f"dataset_metadata.json not found in {epoch_dir}.\n"
            f"Expected: {meta_file}"
        )

    with open(meta_file, encoding='utf-8') as f:
        meta = json.load(f)

    patient_map = {}
    for subj in meta['subjects']:
        subj_id    = int(subj['subject_id'])
        raw_pid    = str(subj['patient_id'])   # e.g. "PAT_14_EEG_160.mat"

        # Extract "PAT_XX" — take first two underscore-separated tokens
        # "PAT_14_EEG_160.mat" → ["PAT", "14", "EEG", "160.mat"] → "PAT_14"
        parts      = raw_pid.split('_')
        patient_id = f"{parts[0]}_{parts[1]}"  # "PAT_14"
        patient_map[subj_id] = patient_id

    return patient_map


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
    Relative power = band_power / total_power  → sum across bands ≈ 1.
    """
    per_ch = np.zeros((N_CHANNELS, len(BANDS)), dtype=np.float32)
    for ci in range(N_CHANNELS):
        freqs, psd = welch(epoch[ci], fs=FS, nperseg=256)
        total = np.trapz(psd, freqs) + 1e-12
        for bi, (_, (flo, fhi)) in enumerate(BANDS.items()):
            idx = np.where((freqs >= flo) & (freqs < fhi))[0]
            per_ch[ci, bi] = np.trapz(psd[idx], freqs[idx]) / total

    feats, names = [], []
    for reg, ch_idx in REGIONS.items():
        region_power = per_ch[ch_idx, :].mean(axis=0)   # (6,)
        for bi, band in enumerate(BAND_NAMES):
            feats.append(float(region_power[bi]))
            names.append(f'spec_{band}_{reg}')

    return np.array(feats, dtype=np.float32), names   # (30,)


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

    Activity   = variance of signal          (amplitude proxy)
    Mobility   = std(dx) / std(x)            (mean frequency proxy)
    Complexity = mobility(dx) / mobility(x)  (bandwidth proxy)
    """
    per_ch = np.zeros((N_CHANNELS, 3), dtype=np.float32)
    for ci in range(N_CHANNELS):
        x   = epoch[ci]
        dx  = np.diff(x)
        d2x = np.diff(dx)
        activity   = float(np.var(x))
        mob_x      = float(np.std(dx)  / (np.std(x)  + 1e-12))
        mob_dx     = float(np.std(d2x) / (np.std(dx) + 1e-12))
        complexity = float(mob_dx / (mob_x + 1e-12))
        per_ch[ci] = [activity, mob_x, complexity]

    feats, names = [], []
    for reg, ch_idx in REGIONS.items():
        region_h = per_ch[ch_idx, :].mean(axis=0)   # (3,)
        for pi, param in enumerate(['activity', 'mobility', 'complexity']):
            feats.append(float(region_h[pi]))
            names.append(f'hjorth_{param}_{reg}')

    return np.array(feats, dtype=np.float32), names   # (15,)


# ─────────────────────────────────────────────────────────────
# C) Graph-level connectivity features
#    Output: 8 features
#    Rationale: DTF/PDC capture directed functional connectivity;
#    global changes and hemispheric asymmetry are key in focal epilepsy
# ─────────────────────────────────────────────────────────────

def graph_level_features(dtf_integrated, pdc_integrated):
    """
    dtf_integrated, pdc_integrated : (19, 19)  band-averaged, diagonal=0
    Returns (8,) vector + 8 names.

    Per metric (DTF, PDC) — 4 features each = 8 total:
      1. Global mean connectivity (off-diagonal)
      2. Left-hemisphere mean out-degree
      3. Right-hemisphere mean out-degree
      4. Hemispheric asymmetry index  (L-R)/(L+R), range (-1,1)

    Out-degree = column sum = influence EMANATING from that source node.
    DTF bright columns = strong source channels  (correct by convention).
    Asymmetry index: positive = left dominant, negative = right dominant.
    """
    feats, names = [], []
    mask = ~np.eye(N_CHANNELS, dtype=bool)   # off-diagonal mask

    for metric_name, mat in [('dtf', dtf_integrated), ('pdc', pdc_integrated)]:
        # 1. Global mean (off-diagonal only — diagonal is 0 anyway)
        global_mean = float(mat[mask].mean())
        feats.append(global_mean)
        names.append(f'graph_{metric_name}_global_mean')

        # Out-degree per node (sum of column = all inputs from that source)
        out_deg = mat.sum(axis=0)   # (19,)

        # 2 & 3. Hemisphere means
        left_mean  = float(out_deg[LEFT_CH].mean())
        right_mean = float(out_deg[RIGHT_CH].mean())
        feats.append(left_mean)
        feats.append(right_mean)
        names.append(f'graph_{metric_name}_left_outdeg')
        names.append(f'graph_{metric_name}_right_outdeg')

        # 4. Asymmetry index
        asym = (left_mean - right_mean) / (left_mean + right_mean + 1e-12)
        feats.append(float(asym))
        names.append(f'graph_{metric_name}_asymmetry')

    # Verify: 2 metrics × 4 features = 8
    assert len(feats) == 8, f"Expected 8 graph features, got {len(feats)}"
    return np.array(feats, dtype=np.float32), names   # (8,)


# ─────────────────────────────────────────────────────────────
# Per-node features for GNN
# Output: (19, 16)
# ─────────────────────────────────────────────────────────────

def node_features_for_gnn(epoch, dtf_integrated, pdc_integrated):
    """
    Returns per-node feature matrix (19, 16) for GNN use.
    These are per-CHANNEL (not region-averaged) because the GNN
    operates on individual nodes.

    Features per node (channel):
      [0-5]  : relative band powers (delta, theta, alpha, beta, gamma, broad)
      [6-8]  : Hjorth (activity, mobility, complexity)
      [9-13] : time-domain (mean, std, skewness, kurtosis, line_length)
      [14]   : DTF out-degree   (source strength in integrated band)
      [15]   : PDC out-degree   (sink   strength in integrated band)
    """
    node_feats = np.zeros((N_CHANNELS, 16), dtype=np.float32)

    for ci in range(N_CHANNELS):
        x = epoch[ci]

        # [0-5] Spectral — relative band powers
        freqs, psd = welch(x, fs=FS, nperseg=256)
        total = np.trapz(psd, freqs) + 1e-12
        for bi, (_, (flo, fhi)) in enumerate(BANDS.items()):
            idx = np.where((freqs >= flo) & (freqs < fhi))[0]
            node_feats[ci, bi] = float(np.trapz(psd[idx], freqs[idx]) / total)

        # [6-8] Hjorth
        dx  = np.diff(x)
        d2x = np.diff(dx)
        node_feats[ci, 6] = float(np.var(x))
        node_feats[ci, 7] = float(np.std(dx) / (np.std(x) + 1e-12))
        mob_dx = float(np.std(d2x) / (np.std(dx) + 1e-12))
        node_feats[ci, 8] = float(mob_dx / (node_feats[ci, 7] + 1e-12))

        # [9-13] Time-domain
        node_feats[ci, 9]  = float(np.mean(x))
        node_feats[ci, 10] = float(np.std(x))
        node_feats[ci, 11] = float(skew(x))
        node_feats[ci, 12] = float(kurtosis(x))
        node_feats[ci, 13] = float(np.sum(np.abs(np.diff(x))))   # line length

        # [14-15] Connectivity out-degree
        node_feats[ci, 14] = float(dtf_integrated[:, ci].sum())
        node_feats[ci, 15] = float(pdc_integrated[:, ci].sum())

    return node_feats   # (19, 16)


# ─────────────────────────────────────────────────────────────
# Epoch selection — balance classes
# ─────────────────────────────────────────────────────────────

def select_balanced_indices(labels, time_from_onset, ratio=2):
    """
    Keep ALL ictal epochs.
    Keep `ratio × n_ictal` pre-ictal epochs taken from the START of the
    recording (most temporally distant from seizure onset).

    Rationale: epochs farthest from seizure onset are the most unambiguously
    pre-ictal. Epochs just before seizure onset may show pre-ictal changes
    that are hard to label cleanly, so we avoid the peri-ictal region.

    Returns empty array (not an error) if subject has 0 ictal epochs.
    """
    ictal_idx    = np.where(labels == 1)[0]
    preictal_idx = np.where(labels == 0)[0]

    n_ictal = len(ictal_idx)

    # Guard: subject has no ictal epochs → skip gracefully
    if n_ictal == 0:
        return np.array([], dtype=np.int64)

    n_select   = min(ratio * n_ictal, len(preictal_idx))
    sorted_pre = preictal_idx[np.argsort(time_from_onset[preictal_idx])]
    selected   = sorted_pre[:n_select]

    return np.concatenate([ictal_idx, selected])


# ─────────────────────────────────────────────────────────────
# Per-subject processing
# ─────────────────────────────────────────────────────────────

def process_subject(subj_idx, epoch_dir, conn_dir, patient_map, ratio):
    """
    Returns (list_of_records, feature_names) or ([], None) on failure.
    Each record contains flat features, node features, raw epoch,
    adjacency matrices, label, subject_id, patient_id, time_from_onset.
    """
    subj_name  = f'subject_{subj_idx:02d}'
    patient_id = patient_map.get(subj_idx, f'PAT_UNKNOWN_{subj_idx}')

    epoch_file = epoch_dir / f'{subj_name}_epochs.npy'
    label_file = epoch_dir / f'{subj_name}_labels.npy'
    tfo_file   = epoch_dir / f'{subj_name}_time_from_onset.npy'
    conn_file  = conn_dir  / f'{subj_name}_graphs.npz'

    # Check required files exist
    for f in [epoch_file, label_file, conn_file]:
        if not f.exists():
            print(f'  [SKIP] {subj_name}: {f.name} not found')
            return [], None

    epochs = np.load(epoch_file)   # (N, 19, 1024)
    labels = np.load(label_file)   # (N,)
    tfo    = np.load(tfo_file) if tfo_file.exists() else np.zeros(len(labels))

    # Load connectivity — only epochs with valid VAR fits are stored
    conn         = np.load(conn_file, allow_pickle=False)
    conn_indices = conn['indices']   # (E,) original epoch indices with valid connectivity
    orig_to_conn = {int(orig): ci for ci, orig in enumerate(conn_indices)}

    # Select balanced subset
    selected = select_balanced_indices(labels, tfo, ratio=ratio)

    if len(selected) == 0:
        n_ictal = int((labels == 1).sum())
        print(f'  [SKIP] {subj_name}: 0 ictal epochs (n_ictal={n_ictal})')
        return [], None

    # Intersect with epochs that have valid connectivity
    selected = np.array([i for i in selected if i in orig_to_conn])

    if len(selected) == 0:
        print(f'  [WARN] {subj_name}: 0 valid epochs after connectivity intersection')
        return [], None

    # Pre-compute feature names once (use dummy data — names don't depend on values)
    _dummy_epoch = np.zeros((N_CHANNELS, N_SAMPLES), dtype=np.float32)
    _dummy_mat   = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float32)
    _, sp_names  = spectral_region_features(_dummy_epoch)
    _, hj_names  = hjorth_region_features(_dummy_epoch)
    _, gr_names  = graph_level_features(_dummy_mat, _dummy_mat)
    feature_names = sp_names + hj_names + gr_names   # 30 + 15 + 8 = 53

    records = []
    for orig_idx in selected:
        ci    = orig_to_conn[orig_idx]
        epoch = epochs[orig_idx]   # (19, 1024)

        dtf_int = conn['dtf_integrated'][ci]   # (19, 19)
        pdc_int = conn['pdc_integrated'][ci]   # (19, 19)

        # Flat features for baseline ML (RF, SVM)
        sp_f, _ = spectral_region_features(epoch)
        hj_f, _ = hjorth_region_features(epoch)
        gr_f, _ = graph_level_features(dtf_int, pdc_int)
        flat = np.concatenate([sp_f, hj_f, gr_f])   # (53,)

        # Per-node features for GNN
        nf = node_features_for_gnn(epoch, dtf_int, pdc_int)   # (19, 16)

        records.append({
            'flat_features':   flat,
            'node_features':   nf,
            'raw_epoch':       epoch,     # (19, 1024) — needed for PureGCN
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
    parser = argparse.ArgumentParser(
        description='Step 3 — Curated Feature Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--epochdir', required=True,
        help='Directory with preprocessed epoch files + dataset_metadata.json'
    )
    parser.add_argument(
        '--conndir', required=True,
        help='Directory with connectivity npz files'
    )
    parser.add_argument(
        '--outputdir', required=True,
        help='Output directory for features'
    )
    parser.add_argument(
        '--ratio', type=int, default=2,
        help='Pre-ictal epochs = ratio × n_ictal per subject (default 2)'
    )
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
    print('  A) Spectral (relative band power, region-averaged): 6 × 5 = 30')
    print('  B) Hjorth   (region-averaged):                     3 × 5 = 15')
    print('  C) Graph-level (DTF/PDC integrated band):                =  8')
    print('  TOTAL flat features                                      = 53')
    print('  GNN node features: 16 per channel (not region-averaged)')
    print('  PureGCN: raw epoch (19, 1024) also saved')
    print('=' * 70)

    # ── Load patient map from metadata JSON ──────────────────────────────────
    print(f'\nLoading patient map from dataset_metadata.json ...')
    patient_map = build_patient_map(epoch_dir)

    unique_patients = sorted(set(patient_map.values()))
    print(f'  Subjects loaded : {len(patient_map)}')
    print(f'  Unique patients : {len(unique_patients)} → {unique_patients}')
    print()

    # Per-patient subject counts — useful for understanding LOPO folds
    from collections import Counter
    pat_counts = Counter(patient_map.values())
    for pat in sorted(pat_counts):
        subj_list = [s for s, p in patient_map.items() if p == pat]
        print(f'    {pat}: {pat_counts[pat]} recording(s) — subjects {subj_list}')
    print()

    # ── Find epoch files ──────────────────────────────────────────────────────
    epoch_files = sorted(epoch_dir.glob('subject_*_epochs.npy'))
    print(f'Found {len(epoch_files)} subject epoch files\n')

    if len(epoch_files) == 0:
        print('[ERROR] No epoch files found. Check --epochdir.')
        return

    # ── Process each subject ──────────────────────────────────────────────────
    all_flat       = []
    all_node       = []
    all_raw        = []
    all_adj_dtf    = []
    all_adj_pdc    = []
    all_labels     = []
    all_subject_ids = []
    all_patient_ids = []
    all_tfo        = []
    feature_names  = None

    skipped_subjects = []

    for ep_file in tqdm(epoch_files, desc='Processing subjects'):
        subj_idx = int(ep_file.stem.replace('subject_', '').replace('_epochs', ''))
        records, feat_names = process_subject(
            subj_idx, epoch_dir, conn_dir, patient_map, args.ratio
        )

        if not records:
            skipped_subjects.append(subj_idx)
            continue

        if feature_names is None:
            feature_names = feat_names

        for rec in records:
            all_flat.append(rec['flat_features'])
            all_node.append(rec['node_features'])
            all_raw.append(rec['raw_epoch'])
            all_adj_dtf.append(rec['adj_dtf'])
            all_adj_pdc.append(rec['adj_pdc'])
            all_labels.append(rec['label'])
            all_subject_ids.append(rec['subject_id'])
            all_patient_ids.append(rec['patient_id'])
            all_tfo.append(rec['time_from_onset'])

    if len(all_flat) == 0:
        print('\n[ERROR] No features extracted. Check epoch and connectivity directories.')
        return

    # ── Stack arrays ──────────────────────────────────────────────────────────
    X           = np.stack(all_flat)                        # (N, 53)
    node_feats  = np.stack(all_node)                        # (N, 19, 16)
    raw_epochs  = np.stack(all_raw)                         # (N, 19, 1024)
    adj_dtf     = np.stack(all_adj_dtf)                     # (N, 19, 19)
    adj_pdc     = np.stack(all_adj_pdc)                     # (N, 19, 19)
    y           = np.array(all_labels,      dtype=np.int64)
    subject_ids = np.array(all_subject_ids, dtype=np.int32)
    patient_ids = np.array(all_patient_ids)                 # string array
    tfo         = np.array(all_tfo,         dtype=np.float32)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    n_bad = int(np.sum(~np.isfinite(X)))
    if n_bad > 0:
        print(f'\n[WARN] {n_bad} non-finite values in X — replacing with 0')
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    assert len(feature_names) == 53, (
        f"Expected 53 feature names, got {len(feature_names)}"
    )
    assert X.shape[1] == 53, (
        f"Expected X shape (N, 53), got {X.shape}"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_npz = output_dir / 'features_all.npz'
    np.savez_compressed(
        out_npz,
        X=X,                                          # (N, 53)  flat features
        node_features=node_feats,                     # (N, 19, 16)
        raw_epochs=raw_epochs,                        # (N, 19, 1024) for PureGCN
        adj_dtf=adj_dtf,                              # (N, 19, 19)
        adj_pdc=adj_pdc,                              # (N, 19, 19)
        y=y,                                          # (N,) labels
        subject_ids=subject_ids,                      # (N,)
        patient_ids=patient_ids,                      # (N,) string
        time_from_onset=tfo,                          # (N,)
        feature_names=np.array(feature_names, dtype=object),
    )
    print(f'\n  ✓ Saved: {out_npz}')

    # Human-readable CSV (flat features only — npz has everything)
    out_csv = output_dir / 'features_all.csv'
    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, 'label',           y)
    df.insert(1, 'subject_id',      subject_ids)
    df.insert(2, 'patient_id',      patient_ids)
    df.insert(3, 'time_from_onset', tfo)
    df.to_csv(out_csv, index=False)
    print(f'  ✓ Saved: {out_csv}')

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ictal    = int((y == 1).sum())
    n_preictal = int((y == 0).sum())
    n_total    = len(y)
    n_patients = len(np.unique(patient_ids))

    majority_acc = max(n_ictal, n_preictal) / n_total * 100
    balance_actual = f'{n_preictal / n_ictal:.1f}' if n_ictal > 0 else 'N/A'

    # Per-patient breakdown
    pat_breakdown = []
    for pat in sorted(np.unique(patient_ids)):
        mask    = patient_ids == pat
        n_ict   = int((y[mask] == 1).sum())
        n_pre   = int((y[mask] == 0).sum())
        n_subj  = len(np.unique(subject_ids[mask]))
        pat_breakdown.append(
            f'    {pat}: {n_subj} subj, {n_ict} ictal, {n_pre} pre-ictal'
        )

    summary = f"""
FEATURE EXTRACTION SUMMARY
===========================
Total epochs        : {n_total}
  Ictal (1)         : {n_ictal}   ({100 * n_ictal / n_total:.1f}%)
  Pre-ictal (0)     : {n_preictal}  ({100 * n_preictal / n_total:.1f}%)
  Actual ratio      : 1 : {balance_actual}  (requested 1 : {args.ratio})

Majority-class acc  : {majority_acc:.1f}%  ← accuracy baseline (dummy classifier)
NOTE: Always report accuracy alongside this baseline — with imbalanced
      data, a model predicting all pre-ictal gets {majority_acc:.1f}% accuracy for free.

Subjects processed  : {len(epoch_files) - len(skipped_subjects)} / {len(epoch_files)}
Subjects skipped    : {skipped_subjects if skipped_subjects else 'none'}
Patients (LOPO)     : {n_patients} → {sorted(np.unique(patient_ids).tolist())}

Per-patient breakdown:
{chr(10).join(pat_breakdown)}

Flat feature vector : {X.shape[1]} features
  Spectral (A)      : {len([f for f in feature_names if f.startswith('spec_')])}  (6 bands × 5 regions)
  Hjorth   (B)      : {len([f for f in feature_names if f.startswith('hjorth_')])}  (3 params × 5 regions)
  Graph    (C)      : {len([f for f in feature_names if f.startswith('graph_')])}  (DTF+PDC: mean, L/R outdeg, asymmetry)

GNN node features   : {node_feats.shape}  (epochs × channels × per-node-feats)
Raw epochs          : {raw_epochs.shape}   (epochs × channels × samples) ← for PureGCN
Adjacency (DTF)     : {adj_dtf.shape}
Adjacency (PDC)     : {adj_pdc.shape}

Feature names (53):
{chr(10).join('  ' + n for n in feature_names)}

Saved files:
  features_all.npz   ← main file (X, node_features, raw_epochs, adj_dtf,
                        adj_pdc, y, subject_ids, patient_ids, time_from_onset,
                        feature_names)
  features_all.csv   ← human-readable flat features

Next steps:
  python step4_baseline_ml.py     --featfile features/features_all.npz
  python step5_gnn_supervised.py  --featfile features/features_all.npz
  python step5b_pure_gcn.py       --featfile features/features_all.npz
  python step6_ssl_gnn.py         --featfile features/features_all.npz
"""
    print(summary)

    out_txt = output_dir / 'feature_summary.txt'
    with open(out_txt, 'w', encoding='utf-8') as fh:
        fh.write(summary)
    print(f'  ✓ Saved: {out_txt}')

    print('\n' + '=' * 70)
    print('STEP 3 COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()