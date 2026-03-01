"""
Step 3c_v2 — Filter Dataset + Patient-Level LOSO Splits
=========================================================
Reads the existing dataset.pt from Step 3c and produces a filtered
version with:

  1. All ictal epochs kept (label=1)
  2. Only the FIRST X pre-ictal epochs per recording kept (label=0)
     These are the earliest epochs in the recording — maximally far
     from the seizure, unambiguously non-ictal.
  3. Patient-level LOSO splits (8 folds, one per patient)
     Fixes the data leakage issue where recordings from the same
     patient appeared in both train and test sets.

Patient mapping (from diagram):
  PAT11 → subject_01
  PAT13 → subject_02
  PAT14 → subject_03 to subject_10  (8 recordings)
  PAT15 → subject_11
  PAT24 → subject_12 to subject_25  (14 recordings)
  PAT27 → subject_26 to subject_32  (7 recordings)
  PAT29 → subject_33
  PAT35 → subject_34

Why first X pre-ictal epochs?
  The recording starts well before the seizure. The first epochs
  represent the brain in a clearly non-ictal state — the maximum
  distance from the seizure. Using these as the negative class gives
  a clean ictal vs. non-ictal distinction (seizure detection task).

Usage:
    python step3c_v2_filter_dataset.py \\
        --datadir   path/to/graphs \\
        --outdir    path/to/graphs_filtered \\
        --n_preictal 20

    # n_preictal: how many pre-ictal epochs to keep per recording
    # Recommended: 2× the typical number of ictal epochs (~10)
    # so n_preictal=20 gives roughly 2:1 ratio pre:ictal

Output:
    graphs_filtered/
        dataset_filtered.pt      ← filtered PyG graph list
        dataset_filtered_info.json
        loso_splits_patient.json ← 8 patient-level folds
        subject_index.npy
        label_index.npy
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Patient mapping ────────────────────────────────────────────────────────
PATIENT_MAP = {
    'PAT11': ['subject_01'],
    'PAT13': ['subject_02'],
    'PAT14': ['subject_03', 'subject_04', 'subject_05', 'subject_06',
              'subject_07', 'subject_08', 'subject_09', 'subject_10'],
    'PAT15': ['subject_11'],
    'PAT24': ['subject_12', 'subject_13', 'subject_14', 'subject_15',
              'subject_16', 'subject_17', 'subject_18', 'subject_19',
              'subject_20', 'subject_21', 'subject_22', 'subject_23',
              'subject_24', 'subject_25'],
    'PAT27': ['subject_26', 'subject_27', 'subject_28', 'subject_29',
              'subject_30', 'subject_31', 'subject_32'],
    'PAT29': ['subject_33'],
    'PAT35': ['subject_34'],
}

# Reverse map: subject → patient
SUBJECT_TO_PATIENT = {
    subj: pat
    for pat, subjects in PATIENT_MAP.items()
    for subj in subjects
}


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset + build patient-level LOSO splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--datadir',    required=True,
                        help='Directory with dataset.pt from Step 3c')
    parser.add_argument('--outdir',     required=True,
                        help='Output directory for filtered dataset')
    parser.add_argument('--n_preictal', type=int, default=20,
                        help='Max pre-ictal epochs to keep per recording (default: 20)')
    args = parser.parse_args()

    import torch
    data_dir   = Path(args.datadir)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load existing dataset ──────────────────────────────────────────────
    print("=" * 70)
    print("STEP 3c_v2 — FILTER DATASET + PATIENT-LEVEL LOSO")
    print("=" * 70)
    print(f"\n  Loading dataset from {data_dir / 'dataset.pt'} ...")
    graphs = torch.load(data_dir / 'dataset.pt', map_location='cpu')
    print(f"  ✅ Loaded {len(graphs):,} graphs")

    # ── Group graphs by subject ────────────────────────────────────────────
    # Each graph has: g.subject_name, g.y (label), g.epoch_idx
    by_subject = defaultdict(list)
    for i, g in enumerate(graphs):
        subj = g.subject_name
        by_subject[subj].append((i, g))

    print(f"\n  Subjects found: {sorted(by_subject.keys())}")

    # ── Filter: keep all ictal + first N pre-ictal per subject ────────────
    print(f"\n  Filtering: keep ALL ictal + first {args.n_preictal} pre-ictal per subject")

    kept_graphs    = []
    subject_index  = []
    label_index    = []

    # Map subject name → integer ID (same as original)
    all_subjects = sorted(by_subject.keys())
    subj_to_idx  = {s: i for i, s in enumerate(all_subjects)}

    stats = []

    for subj in all_subjects:
        subject_graphs = by_subject[subj]  # list of (original_idx, graph)

        # Sort by epoch_idx to ensure temporal order
        subject_graphs.sort(key=lambda x: int(x[1].epoch_idx.item()))

        ictal_graphs   = [(i, g) for i, g in subject_graphs if g.y.item() == 1]
        preictal_graphs = [(i, g) for i, g in subject_graphs if g.y.item() == 0]

        # Keep ALL ictal
        kept_ictal = ictal_graphs

        # Keep only FIRST n_preictal pre-ictal (earliest epoch_idx)
        kept_pre = preictal_graphs[:args.n_preictal]

        n_discarded = len(preictal_graphs) - len(kept_pre)

        stats.append({
            'subject':      subj,
            'patient':      SUBJECT_TO_PATIENT.get(subj, 'unknown'),
            'ictal_kept':   len(kept_ictal),
            'pre_kept':     len(kept_pre),
            'pre_discarded':n_discarded,
            'pre_total':    len(preictal_graphs),
        })

        for _, g in kept_ictal + kept_pre:
            kept_graphs.append(g)
            subject_index.append(subj_to_idx[subj])
            label_index.append(int(g.y.item()))

    label_arr   = np.array(label_index)
    subject_arr = np.array(subject_index)

    # ── Print filtering summary ────────────────────────────────────────────
    print(f"\n  {'Subject':<14} {'Patient':<8} {'Ictal':>7} {'Pre kept':>9} "
          f"{'Pre disc':>9} {'Ratio':>7}")
    print("  " + "-" * 58)
    for s in stats:
        ratio = s['pre_kept'] / max(s['ictal_kept'], 1)
        print(f"  {s['subject']:<14} {s['patient']:<8} {s['ictal_kept']:>7} "
              f"{s['pre_kept']:>9} {s['pre_discarded']:>9} {ratio:>7.1f}:1")

    print("  " + "-" * 58)
    total_ict = sum(s['ictal_kept'] for s in stats)
    total_pre = sum(s['pre_kept'] for s in stats)
    total_disc = sum(s['pre_discarded'] for s in stats)
    print(f"  {'TOTAL':<14} {'':>8} {total_ict:>7} {total_pre:>9} "
          f"{total_disc:>9} {total_pre/max(total_ict,1):>7.1f}:1")
    print(f"\n  Original dataset: {len(graphs):,} graphs")
    print(f"  Filtered dataset: {len(kept_graphs):,} graphs  "
          f"(removed {len(graphs)-len(kept_graphs):,} pre-ictal epochs)")
    print(f"  Class balance: ictal={total_ict}  pre-ictal={total_pre}  "
          f"ratio={total_pre/max(total_ict,1):.1f}:1")

    # ── Build patient-level LOSO splits ───────────────────────────────────
    print(f"\n  Building patient-level LOSO splits ({len(PATIENT_MAP)} folds)...")

    # Map each graph in kept_graphs to its patient
    graph_patients = []
    for g in kept_graphs:
        subj = g.subject_name
        pat  = SUBJECT_TO_PATIENT.get(subj, 'unknown')
        graph_patients.append(pat)

    graph_patients = np.array(graph_patients)

    loso_patient_splits = {}
    for pat in sorted(PATIENT_MAP.keys()):
        test_idx  = np.where(graph_patients == pat)[0].tolist()
        train_idx = np.where(graph_patients != pat)[0].tolist()

        # Count classes in test
        test_labels = [int(kept_graphs[i].y.item()) for i in test_idx]
        n_test_ict  = sum(1 for l in test_labels if l == 1)
        n_test_pre  = sum(1 for l in test_labels if l == 0)

        train_labels = [int(kept_graphs[i].y.item()) for i in train_idx]
        n_train_ict  = sum(1 for l in train_labels if l == 1)

        print(f"    {pat}: test={len(test_idx):4d} (ictal={n_test_ict}, "
              f"pre={n_test_pre})  |  train={len(train_idx):4d} "
              f"(ictal={n_train_ict})")

        loso_patient_splits[pat] = {
            'test':     test_idx,
            'train':    train_idx,
            'subjects': PATIENT_MAP[pat],
        }

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n  Saving filtered dataset...")

    torch.save(kept_graphs, output_dir / 'dataset_filtered.pt')
    print(f"  ✅ dataset_filtered.pt  ({len(kept_graphs):,} graphs)")

    np.save(output_dir / 'subject_index.npy', subject_arr)
    np.save(output_dir / 'label_index.npy',   label_arr)

    with open(output_dir / 'loso_splits_patient.json', 'w') as f:
        json.dump(loso_patient_splits, f, indent=2)
    print(f"  ✅ loso_splits_patient.json  (8 patient-level folds)")

    # Also copy the recording-level splits for reference
    orig_splits_path = Path(args.datadir) / 'loso_splits.json'
    if orig_splits_path.exists():
        with open(orig_splits_path) as f:
            orig_splits = json.load(f)
        # Rebuild recording-level splits for the filtered dataset
        recording_splits = {}
        for subj in all_subjects:
            s_idx = subj_to_idx[subj]
            test_idx  = np.where(subject_arr == s_idx)[0].tolist()
            train_idx = np.where(subject_arr != s_idx)[0].tolist()
            recording_splits[subj] = {'test': test_idx, 'train': train_idx}
        with open(output_dir / 'loso_splits_recording.json', 'w') as f:
            json.dump(recording_splits, f, indent=2)
        print(f"  ✅ loso_splits_recording.json  (34 recording-level folds, for reference)")

    # Save info
    info = {
        'description':       'Filtered EEG graph dataset — patient-level LOSO',
        'n_graphs_original': len(graphs),
        'n_graphs_filtered': len(kept_graphs),
        'n_preictal_kept':   args.n_preictal,
        'n_subjects':        len(all_subjects),
        'n_patients':        len(PATIENT_MAP),
        'patient_map':       PATIENT_MAP,
        'class_counts': {
            'pre_ictal': int((label_arr == 0).sum()),
            'ictal':     int((label_arr == 1).sum()),
        },
        'class_ratio':       float(total_pre / max(total_ict, 1)),
        'loso_type':         'patient-level (8 folds)',
        'pre_ictal_selection': f'first {args.n_preictal} epochs per recording (earliest, max distance from seizure)',
        'ictal_selection':   'all ictal epochs per recording',
        'files': {
            'dataset_filtered.pt':       'filtered PyG graph list',
            'loso_splits_patient.json':  '8 patient-level LOSO folds',
            'loso_splits_recording.json':'34 recording-level folds (reference only)',
            'label_index.npy':           'label per graph',
            'subject_index.npy':         'subject ID per graph',
        }
    }
    with open(output_dir / 'dataset_filtered_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  ✅ dataset_filtered_info.json")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\n  Filtered dataset: {len(kept_graphs):,} graphs")
    print(f"    Pre-ictal: {total_pre}  (first {args.n_preictal} per recording)")
    print(f"    Ictal:     {total_ict}  (all kept)")
    print(f"    Ratio:     {total_pre/max(total_ict,1):.1f}:1")
    print(f"\n  LOSO: 8 patient-level folds")
    for pat, subjects in PATIENT_MAP.items():
        print(f"    {pat}: {len(subjects)} recording(s) → {subjects}")
    print(f"\n  Next steps — rerun with filtered dataset:")
    print(f"    Step 3b (baseline ML):")
    print(f"      --conndir ... --featdir ... --outdir ...")
    print(f"      (update to use patient-level splits)")
    print(f"    Step 3d (supervised GCN):")
    print(f"      --datadir {output_dir} \\")
    print(f"      --splits  {output_dir}/loso_splits_patient.json")
    print(f"    Step 4a (SSL pretrain):")
    print(f"      --datadir {output_dir}")
    print(f"    Step 4b (SSL evaluate):")
    print(f"      --datadir {output_dir} \\")
    print(f"      --splits  {output_dir}/loso_splits_patient.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
