"""
Step 5 — Node Feature Extraction
==================================
Computes per-channel, per-epoch features from the raw EEG epochs.
Uses selected_epochs.json to know WHICH epoch indices to load per subject.

Features computed per channel (19 channels = 19 nodes in the GNN graph):
  Band power (5 features):
    - Delta  : 0.5 –  4.0 Hz
    - Theta  :  4.0 –  8.0 Hz
    - Alpha  :  8.0 – 15.0 Hz
    - Beta   : 15.0 – 30.0 Hz
    - Gamma  : 30.0 – 45.0 Hz

  Hjorth parameters (3 features):
    - Activity   : variance of the signal (power proxy)
    - Mobility   : sqrt(var(1st derivative) / var(signal))
    - Complexity : mobility(1st deriv) / mobility(signal)

  Connectivity summary (4 features, from your existing DTF/PDC matrices):
    - DTF outflow  (integrated band)
    - DTF inflow   (integrated band)
    - PDC outflow  (integrated band)
    - PDC inflow   (integrated band)

  Total: 5 + 3 + 4 = 12 features per channel per epoch
  (or 8 if --connectivity_dir is not provided)

Output files (saved to output_dir):
  subject_XX_node_features.npy   shape: (n_epochs, 19, 12)
  subject_XX_node_labels.npy     shape: (n_epochs,)
  subject_XX_epoch_indices.npy   shape: (n_epochs,)
  node_features_metadata.json    feature names + per-fold normalization stats

Usage
-----
  # With connectivity (recommended):
  python step5_extract_node_features.py \
      --epochs_dir       F:\\...\\final_preprocessed_epochs \
      --connectivity_dir F:\\...\\connectivity \
      --selected_epochs  F:\\...\\splits\\selected_epochs.json \
      --splits           F:\\...\\splits\\splits.json \
      --output_dir       F:\\...\\node_features

  # Without connectivity (EEG features only, 8 features):
  python step5_extract_node_features.py \
      --epochs_dir      F:\\...\\final_preprocessed_epochs \
      --selected_epochs F:\\...\\splits\\selected_epochs.json \
      --splits          F:\\...\\splits\\splits.json \
      --output_dir      F:\\...\\node_features
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FS            = 256
EPOCH_SAMPLES = 1024
N_CHANNELS    = 19

# numpy >= 2.0 renamed trapz -> trapezoid; support both
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

FREQ_BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  15.0),
    "beta":  (15.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]

CHANNELS = [
    "Fp1", "Fp2", "F7",  "F3",  "Fz",  "F4",  "F8",
    "T3",  "C3",  "Cz",  "C4",  "T4",
    "T5",  "P3",  "Pz",  "P4",  "T6",
    "O1",  "O2",
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def bandpower(signal_1d, fs, fmin, fmax):
    """
    Absolute band power via Welch PSD, integrated with trapz.
    """
    nperseg = min(256, len(signal_1d))
    freqs, psd = welch(signal_1d, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapezoid(psd[mask], freqs[mask]))


def hjorth_parameters(signal_1d):
    """
    Activity  = var(x)
    Mobility  = sqrt(var(dx) / var(x))
    Complexity = Mobility(dx) / Mobility(x)
    """
    x   = signal_1d
    dx  = np.diff(x)
    ddx = np.diff(dx)

    var_x   = np.var(x)
    var_dx  = np.var(dx)
    var_ddx = np.var(ddx)

    if var_x < 1e-12:
        return 0.0, 0.0, 0.0

    activity   = float(var_x)
    mobility   = float(np.sqrt(var_dx / var_x))
    mob_dx     = float(np.sqrt(var_ddx / var_dx)) if var_dx > 1e-12 else 0.0
    complexity = float(mob_dx / mobility) if mobility > 1e-12 else 0.0

    return activity, mobility, complexity


def extract_eeg_features(epoch_data):
    """
    epoch_data : (19, 1024)
    Returns    : (19, 8)  — 5 band powers + 3 Hjorth per channel
    """
    features = np.zeros((N_CHANNELS, 8), dtype=np.float32)
    for ch in range(N_CHANNELS):
        sig = epoch_data[ch].astype(np.float64)
        for b_idx, band in enumerate(BAND_ORDER):
            fmin, fmax = FREQ_BANDS[band]
            features[ch, b_idx] = bandpower(sig, FS, fmin, fmax)
        act, mob, comp = hjorth_parameters(sig)
        features[ch, 5] = act
        features[ch, 6] = mob
        features[ch, 7] = comp
    return features


def extract_connectivity_features(conn_file, epoch_indices):
    """
    Load DTF/PDC matrices and compute per-channel outflow/inflow.

    Returns (n_epochs, 19, 4) or None if file not found / keys missing.
    Matrix convention: [i,j] = j -> i
      outflow of j = column sum = axis=0
      inflow  of i = row    sum = axis=1
    """
    if not conn_file.exists():
        return None

    data = np.load(conn_file)

    for key in ["dtf_integrated", "pdc_integrated", "indices"]:
        if key not in data:
            print(f"    WARNING: key '{key}' missing in {conn_file.name} "
                  "— skipping connectivity features")
            return None

    dtf         = data["dtf_integrated"]   # (n_conn_epochs, 19, 19)
    pdc         = data["pdc_integrated"]   # (n_conn_epochs, 19, 19)
    conn_idx    = data["indices"]
    raw_to_conn = {int(r): k for k, r in enumerate(conn_idx)}

    n_epochs = len(epoch_indices)
    features = np.full((n_epochs, N_CHANNELS, 4), np.nan, dtype=np.float32)

    for i, raw_idx in enumerate(epoch_indices):
        if raw_idx not in raw_to_conn:
            continue
        k = raw_to_conn[raw_idx]
        features[i, :, 0] = dtf[k].sum(axis=0)   # dtf outflow
        features[i, :, 1] = dtf[k].sum(axis=1)   # dtf inflow
        features[i, :, 2] = pdc[k].sum(axis=0)   # pdc outflow
        features[i, :, 3] = pdc[k].sum(axis=1)   # pdc inflow

    missing = int(np.isnan(features[:, 0, 0]).sum())
    if missing > 0:
        print(f"    INFO: {missing}/{n_epochs} epochs missing connectivity "
              "(dropped VAR fit) — filled with 0")

    return np.nan_to_num(features, nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# PER-SUBJECT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_subject(subj_info, epochs_dir, connectivity_dir, output_dir):
    sid      = subj_info["subject_id"]
    pat      = subj_info["patient_code"]
    excluded = subj_info["excluded"]
    pre_idx  = subj_info["pre_ictal_indices"]
    ict_idx  = subj_info["ictal_indices"]

    if excluded:
        print(f"  Subject {sid:02d} (PAT_{pat}): SKIPPED (excluded)")
        return None

    # Combine pre-ictal and ictal indices, sort by time
    all_pairs    = sorted([(i, 0) for i in pre_idx] + [(i, 1) for i in ict_idx])
    epoch_indices = [x[0] for x in all_pairs]
    labels        = np.array([x[1] for x in all_pairs], dtype=np.int64)
    n_epochs      = len(epoch_indices)

    # Load raw EEG
    epochs_file = epochs_dir / f"subject_{sid:02d}_epochs.npy"
    if not epochs_file.exists():
        print(f"  Subject {sid:02d}: MISSING file {epochs_file.name}")
        return None

    all_epochs = np.load(epochs_file)   # (total_epochs, 19, 1024)

    # EEG features
    eeg_features = np.zeros((n_epochs, N_CHANNELS, 8), dtype=np.float32)
    for i, raw_idx in enumerate(epoch_indices):
        eeg_features[i] = extract_eeg_features(all_epochs[raw_idx])

    # Connectivity features (optional)
    conn_features = None
    if connectivity_dir is not None:
        conn_file     = connectivity_dir / f"subject_{sid:02d}_graphs.npz"
        conn_features = extract_connectivity_features(conn_file, epoch_indices)

    # Concatenate
    if conn_features is not None:
        node_features = np.concatenate([eeg_features, conn_features], axis=2)
        feat_dim = 12
    else:
        node_features = eeg_features
        feat_dim = 8

    # Save
    prefix = output_dir / f"subject_{sid:02d}"
    np.save(f"{prefix}_node_features.npy",  node_features)
    np.save(f"{prefix}_node_labels.npy",    labels)
    np.save(f"{prefix}_epoch_indices.npy",  np.array(epoch_indices, dtype=np.int64))

    n_pre = int((labels == 0).sum())
    n_ict = int((labels == 1).sum())
    print(f"  Subject {sid:02d} (PAT_{pat}): "
          f"{n_pre} pre + {n_ict} ictal = {n_epochs} epochs | "
          f"shape {node_features.shape} | "
          f"{'EEG+Conn' if conn_features is not None else 'EEG only'}")

    return {
        "subject_id":        sid,
        "patient_code":      pat,
        "n_epochs":          n_epochs,
        "n_pre_ictal":       n_pre,
        "n_ictal":           n_ict,
        "feature_dim":       feat_dim,
        "has_connectivity":  conn_features is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION STATS  (per LOPO fold, fit on train only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_normalization_stats(output_dir, train_subjects, feature_dim):
    """
    Mean + std per feature per channel, computed over all training epochs.
    Returns {mean: (19, F) as list, std: (19, F) as list}.
    """
    all_feats = []
    for sid in train_subjects:
        p = output_dir / f"subject_{sid:02d}_node_features.npy"
        if p.exists():
            all_feats.append(np.load(p))   # (n_epochs, 19, F)

    if not all_feats:
        return {}

    combined = np.concatenate(all_feats, axis=0)   # (N_total, 19, F)
    mean = combined.mean(axis=0)                    # (19, F)
    std  = combined.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)

    return {"mean": mean.tolist(), "std": std.tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract node features: band power + Hjorth + connectivity"
    )
    parser.add_argument("--epochs_dir",       required=True)
    parser.add_argument("--selected_epochs",  required=True)
    parser.add_argument("--splits",           required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--connectivity_dir", default=None)
    args = parser.parse_args()

    epochs_dir       = Path(args.epochs_dir)
    output_dir       = Path(args.output_dir)
    connectivity_dir = Path(args.connectivity_dir) if args.connectivity_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 5 - NODE FEATURE EXTRACTION")
    print("=" * 70)
    print(f"  Epochs dir       : {epochs_dir}")
    print(f"  Connectivity dir : {connectivity_dir or 'NOT PROVIDED'}")
    print(f"  Output dir       : {output_dir}")
    print()
    print("  Features per channel:")
    print("    Band power  (5) : delta, theta, alpha, beta, gamma")
    print("    Hjorth      (3) : activity, mobility, complexity")
    if connectivity_dir:
        print("    Connectivity(4) : DTF outflow/inflow, PDC outflow/inflow")
    print()

    with open(args.selected_epochs) as f:
        selected = json.load(f)
    with open(args.splits) as f:
        splits = json.load(f)

    print("Processing subjects...")
    print("-" * 70)

    summaries   = []
    feature_dim = None

    for sid_str, subj_info in selected.items():
        result = process_subject(subj_info, epochs_dir, connectivity_dir, output_dir)
        if result:
            summaries.append(result)
            feature_dim = result["feature_dim"]

    # Per-fold normalization stats
    print()
    print("Computing per-fold normalization statistics...")
    print("-" * 70)

    fold_stats = []
    for fold in splits["folds"]:
        train_subs = fold["train_subjects"]
        stats      = compute_normalization_stats(output_dir, train_subs, feature_dim)
        fold_stats.append({
            "fold":           fold["fold"],
            "test_patient":   fold["test_patient"],
            "train_subjects": train_subs,
            "normalization":  stats,
        })
        print(f"  Fold {fold['fold']}: test=PAT_{fold['test_patient']} | "
              f"norm from {len(train_subs)} train subjects")

    # Save metadata
    feature_names = (
        ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
         "hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
        + (["dtf_outflow", "dtf_inflow", "pdc_outflow", "pdc_inflow"]
           if connectivity_dir else [])
    )

    metadata = {
        "feature_names":       feature_names,
        "n_features":          feature_dim,
        "n_channels":          N_CHANNELS,
        "channels":            CHANNELS,
        "freq_bands":          {b: list(r) for b, r in FREQ_BANDS.items()},
        "sampling_rate":       FS,
        "has_connectivity":    connectivity_dir is not None,
        "subjects":            summaries,
        "fold_normalization":  fold_stats,
        "usage": {
            "load":      "np.load('subject_01_node_features.npy')  -> (n_epochs, 19, F)",
            "normalize": "Use fold_normalization[fold]['normalization'] mean/std",
        },
    }

    meta_path = output_dir / "node_features_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Saved: {meta_path}")

    # Summary
    total_epochs = sum(s["n_epochs"]    for s in summaries)
    total_pre    = sum(s["n_pre_ictal"] for s in summaries)
    total_ict    = sum(s["n_ictal"]     for s in summaries)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Subjects processed : {len(summaries)}")
    print(f"  Total epochs       : {total_epochs}  "
          f"({total_pre} pre + {total_ict} ictal)")
    print(f"  Features per node  : {feature_dim}")
    print(f"  Output shape       : (n_epochs, 19, {feature_dim})")
    print()
    print("  Output files per subject:")
    print("    subject_XX_node_features.npy  -> (n_epochs, 19, F)")
    print("    subject_XX_node_labels.npy    -> (n_epochs,)  0=pre / 1=ictal")
    print("    subject_XX_epoch_indices.npy  -> (n_epochs,)  original indices")
    print("    node_features_metadata.json   -> names + fold normalization")
    print()
    print("  Next step: step6_baseline_ml.py  (SVM + Random Forest baseline)")
    print("=" * 70)


if __name__ == "__main__":
    main()
