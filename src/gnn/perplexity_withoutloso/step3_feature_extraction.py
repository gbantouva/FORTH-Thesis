"""
Step 3 — Feature Extraction from DTF/PDC Connectivity Matrices
==============================================================
Input : connectivity/<subject>_graphs.npz  (from step2)
Output: features/all_features.npz
         -> X      : (n_epochs_total, n_features)
         -> y      : (n_epochs_total,)  binary  0=pre-ictal  1=ictal
         -> groups : (n_epochs_total,)  patient ID  (for LOPO CV)

Feature vector per epoch:
  Per band (6) x measure (DTF + PDC):
    - Upper-triangle edges : 19*18/2 = 171
    - Out-strength (row sum): 19
    - In-strength  (col sum): 19
    - Global stats (mean, std, max): 3
    => 212 per band per measure
    => 6 x 2 x 212 = 2544 features total

Pre-ictal selection:
  Take the FIRST (ratio * n_ictal) non-ictal epochs in each recording.
  "First" = lowest epoch indices = furthest in time from seizure onset.

Usage:
  python step3_feature_extraction.py \
    --conndir path/to/connectivity \
    --outdir  path/to/features \
    --ratio   2
"""

import argparse
import numpy as np
from pathlib import Path

BANDS = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']
K = 19  # EEG channels


# ═══════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION FROM ONE MATRIX
# ═══════════════════════════════════════════════════════════════

def matrix_features(mat):
    """
    Extract a 212-d feature vector from one (K, K) connectivity matrix.
    Diagonal is already 0 from step2 (self-edges removed).
    """
    triu_idx = np.triu_indices(K, k=1)          # upper triangle, no diag
    edges    = mat[triu_idx]                     # (171,)
    out_str  = mat.sum(axis=1)                   # row sums  (19,)  out-strength
    in_str   = mat.sum(axis=0)                   # col sums  (19,)  in-strength
    stats    = np.array([mat.mean(), mat.std(), mat.max()])  # (3,)
    return np.concatenate([edges, out_str, in_str, stats])   # (212,)


# ═══════════════════════════════════════════════════════════════
# 2. BINARY LABELING: ICTAL vs PRE-ICTAL
# ═══════════════════════════════════════════════════════════════

def select_epochs(labels_orig, ratio=2):
    """
    Select which epochs to keep and assign binary labels.

    Strategy:
      - Ictal     (y=1): all epochs where labels_orig == 1
      - Pre-ictal (y=0): first (ratio * n_ictal) non-ictal epochs
                         sorted by index (= earliest in recording)

    Returns
    -------
    keep_idx : np.ndarray  indices into the original epoch array to keep
    y_binary : np.ndarray  binary labels (0=pre-ictal, 1=ictal)
    """
    ictal_idx     = np.where(labels_orig == 1)[0]
    non_ictal_idx = np.sort(np.where(labels_orig != 1)[0])  # time order
    n_ictal       = len(ictal_idx)

    n_preictal  = min(ratio * n_ictal, len(non_ictal_idx))
    preictal_idx = non_ictal_idx[:n_preictal]   # earliest epochs

    keep_idx = np.sort(np.concatenate([ictal_idx, preictal_idx]))

    ictal_set = set(ictal_idx.tolist())
    y_binary  = np.array(
        [1 if i in ictal_set else 0 for i in keep_idx],
        dtype=np.int64
    )
    return keep_idx, y_binary


# ═══════════════════════════════════════════════════════════════
# 3. PROCESS ONE SUBJECT FILE
# ═══════════════════════════════════════════════════════════════

def extract_subject(npz_path, patient_id, ratio=2):
    """
    Load one subject npz, select epochs, extract features.

    Returns (X, y, groups) or (None, None, None) if no ictal epochs.
    """
    data        = np.load(npz_path)
    labels_orig = data['labels'].copy()
    n_ictal     = int((labels_orig == 1).sum())

    if n_ictal == 0:
        print(f"  [SKIP] {npz_path.stem} — no ictal epochs")
        return None, None, None

    keep_idx, y_binary = select_epochs(labels_orig, ratio=ratio)

    rows = []
    for i in keep_idx:
        feat = []
        for band in BANDS:
            dtf = data[f'dtf_{band}'][i]    # (19, 19)
            pdc = data[f'pdc_{band}'][i]    # (19, 19)
            feat.append(matrix_features(dtf))
            feat.append(matrix_features(pdc))
        rows.append(np.concatenate(feat))   # (2544,)

    X      = np.array(rows, dtype=np.float32)                  # (n_kept, 2544)
    groups = np.full(len(y_binary), patient_id, dtype=np.int64)

    print(f"  {npz_path.stem:30s}  total={len(labels_orig):4d} "
          f"| kept={len(y_binary):4d} "
          f"| ictal={y_binary.sum():3d} "
          f"| pre-ictal={(y_binary == 0).sum():3d}")
    return X, y_binary, groups


# ═══════════════════════════════════════════════════════════════
# 4. BUILD FULL DATASET
# ═══════════════════════════════════════════════════════════════

def build_dataset(conn_dir, out_dir, ratio=2):
    conn_dir = Path(conn_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(conn_dir.glob('subject_*_graphs.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No subject_*_graphs.npz found in {conn_dir}")

    print(f"\nFound {len(npz_files)} subject file(s)")
    print(f"Pre-ictal ratio: {ratio}x ictal\n")
    print(f"{'─'*60}")

    all_X, all_y, all_groups = [], [], []

    for pid, npz in enumerate(npz_files):
        X, y, g = extract_subject(npz, patient_id=pid, ratio=ratio)
        if X is None:
            continue
        all_X.append(X)
        all_y.append(y)
        all_groups.append(g)

    if not all_X:
        raise RuntimeError("No valid subjects found — check your connectivity files.")

    X      = np.concatenate(all_X,      axis=0)
    y      = np.concatenate(all_y,      axis=0)
    groups = np.concatenate(all_groups, axis=0)

    out_path = out_dir / 'all_features.npz'
    np.savez_compressed(out_path, X=X, y=y, groups=groups)

    print(f"{'─'*60}")
    print(f"\nDataset summary:")
    print(f"  Total epochs  : {len(y)}")
    print(f"  Ictal         : {(y == 1).sum()}")
    print(f"  Pre-ictal     : {(y == 0).sum()}")
    print(f"  Patients      : {len(np.unique(groups))}")
    print(f"  Feature dim   : {X.shape[1]}")
    print(f"\nSaved → {out_path}")


# ═══════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 3 — Feature extraction")
    parser.add_argument('--conndir', required=True,
                        help='Path to connectivity/ folder (step2 output)')
    parser.add_argument('--outdir',  required=True,
                        help='Path to features/ output folder')
    parser.add_argument('--ratio',   type=int, default=2,
                        help='Pre-ictal epochs per ictal epoch (default: 2)')
    args = parser.parse_args()
    build_dataset(args.conndir, args.outdir, ratio=args.ratio)
