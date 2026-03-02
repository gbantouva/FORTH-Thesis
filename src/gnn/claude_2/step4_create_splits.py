"""
Step 4 — Epoch Sampling & LOPO Split Definition
================================================
Reads dataset_metadata.json and produces:

  1. selected_epochs.json
     Per-subject list of which epoch INDICES to use (pre-ictal + ictal),
     applying the 2:1 pre-ictal ratio from the START of the recording,
     with a 60-second exclusion zone before seizure onset.

  2. splits.json
     8-fold Leave-One-Patient-Out (LOPO) split.
     Each fold defines: test_patient, val_patient, train_patients,
     and the exact subject IDs in each set.

  3. dataset_summary.txt
     Human-readable summary printed + saved for your thesis appendix.

Rules applied
-------------
- Pre-ictal epochs: taken from epoch index 0 upward, STOPPING at
  (seizure_start_sec - 60s) to exclude the transitional zone.
- Pre-ictal count: min(2 × n_ictal, available_pre_ictal_before_cutoff)
- Post-ictal epochs: NEVER used (not selected as pre-ictal).
- Subject 34 (PAT_35, 2 ictal epochs, 4.5s seizure): excluded from
  training/validation. Kept in a separate 'excluded' list.
- Subject 1 (PAT_11, 'not in paper'): included but flagged.

LOPO scheme
-----------
  Test  : 1 patient (rotated through all 8)
  Val   : PAT_13 when not test, else PAT_11 (small single-recording patients)
  Train : remaining patients

Usage
-----
  python step4_create_splits.py \
      --metadata  path/to/dataset_metadata.json \
      --output_dir path/to/output_folder

  # Example (Windows paths):
  python step4_create_splits.py \
      --metadata "F:\\FORTH_Final_Thesis\\FORTH-Thesis\\final_preprocessed_epochs\\dataset_metadata.json" \
      --output_dir "F:\\FORTH_Final_Thesis\\FORTH-Thesis\\splits"
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
EPOCH_DURATION_SEC  = 4.0
EXCLUSION_ZONE_SEC  = 60.0       # seconds before seizure to exclude from pre-ictal
PRE_ICTAL_RATIO     = 2          # pre-ictal epochs = PRE_ICTAL_RATIO × n_ictal
EXCLUDED_SUBJECTS   = [34]       # PAT_35: only 2 ictal epochs (4.5s seizure)
FLAGGED_SUBJECTS    = [1]        # PAT_11: not in original paper

# Fixed validation patient (used when it's not the test patient)
PRIMARY_VAL_PAT   = "13"   # PAT_13 — 1 recording, 25 ictal
SECONDARY_VAL_PAT = "11"   # PAT_11 — used as val only when PAT_13 is test

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_patient_code(patient_id_str: str) -> str:
    """'PAT_24_EEG_001.mat' → '24'"""
    parts = patient_id_str.split("_")
    return parts[1] if len(parts) >= 2 else patient_id_str


def select_epochs_for_subject(subj: dict) -> dict:
    """
    Returns a dict describing which epoch indices to use for one subject.

    Returns
    -------
    {
        'subject_id'      : int,
        'patient_code'    : str,
        'n_ictal'         : int,
        'ictal_indices'   : [int, ...],   # ALL ictal epoch indices
        'pre_ictal_indices': [int, ...],  # selected pre-ictal indices
        'n_selected_pre'  : int,
        'n_available_pre_before_cutoff': int,
        'cutoff_epoch'    : int,          # last safe pre-ictal epoch index
        'excluded'        : bool,
        'flagged'         : bool,
        'notes'           : str,
    }
    """
    sid        = subj["subject_id"]
    n_total    = subj["n_epochs_total"]
    n_ictal    = subj["n_ictal"]
    n_pre      = subj["n_pre_ictal"]
    sz_start_s = subj["seizure_start_sec"]
    sz_end_s   = subj["seizure_end_sec"]
    pat_code   = extract_patient_code(subj["patient_id"])

    # ── Compute ictal epoch indices ──────────────────────────────────────────
    # An epoch at index i spans [i*4, (i+1)*4) seconds.
    # It is ictal if it overlaps [sz_start_s, sz_end_s].
    ictal_indices = []
    for i in range(n_total):
        ep_start = i * EPOCH_DURATION_SEC
        ep_end   = ep_start + EPOCH_DURATION_SEC
        if ep_end > sz_start_s and ep_start < sz_end_s:
            ictal_indices.append(i)

    # Sanity check against stored n_ictal
    assert len(ictal_indices) == n_ictal, (
        f"Subject {sid}: computed {len(ictal_indices)} ictal indices "
        f"but metadata says {n_ictal}"
    )

    # ── Compute cutoff for pre-ictal selection ───────────────────────────────
    # Last pre-ictal epoch must END at least EXCLUSION_ZONE_SEC before seizure.
    # ep_end = (i+1)*4 ≤ sz_start_s - EXCLUSION_ZONE_SEC
    # → i ≤ (sz_start_s - EXCLUSION_ZONE_SEC) / 4 - 1
    cutoff_epoch = int((sz_start_s - EXCLUSION_ZONE_SEC) / EPOCH_DURATION_SEC) - 1

    # Collect available pre-ictal indices before the cutoff
    available_pre = [i for i in range(cutoff_epoch + 1)
                     if i not in ictal_indices]

    n_want = PRE_ICTAL_RATIO * n_ictal
    selected_pre = available_pre[:n_want]   # earliest epochs first

    notes = []
    if sid in EXCLUDED_SUBJECTS:
        notes.append("EXCLUDED: too few ictal epochs for reliable training")
    if sid in FLAGGED_SUBJECTS:
        notes.append("FLAGGED: subject not present in original paper")
    if len(available_pre) < n_want:
        notes.append(
            f"WARNING: wanted {n_want} pre-ictal but only "
            f"{len(available_pre)} available before cutoff"
        )
    if n_ictal <= 3:
        notes.append("WARNING: very few ictal epochs (<=3)")

    return {
        "subject_id"                  : sid,
        "patient_code"                : pat_code,
        "n_ictal"                     : n_ictal,
        "ictal_indices"               : ictal_indices,
        "pre_ictal_indices"           : selected_pre,
        "n_selected_pre"              : len(selected_pre),
        "n_available_pre_before_cutoff": len(available_pre),
        "cutoff_epoch"                : cutoff_epoch,
        "seizure_start_sec"           : sz_start_s,
        "seizure_end_sec"             : sz_end_s,
        "seizure_duration_sec"        : subj["seizure_duration_sec"],
        "excluded"                    : sid in EXCLUDED_SUBJECTS,
        "flagged"                     : sid in FLAGGED_SUBJECTS,
        "notes"                       : "; ".join(notes) if notes else "OK",
    }


def build_lopo_splits(patients: dict, excluded_subjects: list) -> list:
    """
    Build 8-fold LOPO splits.

    patients : {patient_code: [subject_id, ...]}
    Returns list of fold dicts.
    """
    # Only patients that have at least one non-excluded subject
    valid_patients = {
        p: [s for s in sids if s not in excluded_subjects]
        for p, sids in patients.items()
    }
    valid_patients = {p: sids for p, sids in valid_patients.items() if sids}
    all_patient_codes = sorted(valid_patients.keys(), key=int)

    folds = []
    for test_pat in all_patient_codes:
        remaining = [p for p in all_patient_codes if p != test_pat]

        # Choose validation patient
        if PRIMARY_VAL_PAT in remaining:
            val_pat = PRIMARY_VAL_PAT
        elif SECONDARY_VAL_PAT in remaining:
            val_pat = SECONDARY_VAL_PAT
        else:
            # Fallback: smallest patient by ictal count
            val_pat = min(remaining, key=lambda p: len(valid_patients[p]))

        train_pats = [p for p in remaining if p != val_pat]

        fold = {
            "fold"              : len(folds) + 1,
            "test_patient"      : test_pat,
            "val_patient"       : val_pat,
            "train_patients"    : train_pats,
            "test_subjects"     : valid_patients[test_pat],
            "val_subjects"      : valid_patients[val_pat],
            "train_subjects"    : [s for p in train_pats
                                   for s in valid_patients[p]],
        }
        folds.append(fold)

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Epoch sampling + LOPO split definition for TUC dataset"
    )
    parser.add_argument("--metadata",   required=True,
                        help="Path to dataset_metadata.json")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write output files")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────────
    with open(metadata_path) as f:
        meta = json.load(f)

    subjects = meta["subjects"]
    print("=" * 70)
    print("STEP 4 — EPOCH SAMPLING & LOPO SPLITS")
    print("=" * 70)
    print(f"  Subjects loaded : {len(subjects)}")
    print(f"  Epoch duration  : {EPOCH_DURATION_SEC}s")
    print(f"  Exclusion zone  : {EXCLUSION_ZONE_SEC}s before seizure")
    print(f"  Pre/ictal ratio : {PRE_ICTAL_RATIO}:1")
    print(f"  Excluded subjs  : {EXCLUDED_SUBJECTS}")
    print()

    # ── Select epochs ────────────────────────────────────────────────────────
    selected = {}
    patients = defaultdict(list)   # patient_code → [subject_ids]

    for subj in subjects:
        result = select_epochs_for_subject(subj)
        selected[subj["subject_id"]] = result
        patients[result["patient_code"]].append(subj["subject_id"])

    # ── Print per-subject table ──────────────────────────────────────────────
    print(f"{'Subj':<6} {'PAT':<6} {'Ictal':<7} {'Pre sel':<9} "
          f"{'Pre avail':<11} {'Cutoff ep':<11} {'Status'}")
    print("-" * 75)

    total_ictal_used = 0
    total_pre_used   = 0

    for sid, r in selected.items():
        status = "EXCLUDED" if r["excluded"] else ("FLAGGED" if r["flagged"] else "OK")
        if "WARNING" in r["notes"] and status == "OK":
            status = "WARN"
        print(f"{sid:<6} {r['patient_code']:<6} {r['n_ictal']:<7} "
              f"{r['n_selected_pre']:<9} {r['n_available_pre_before_cutoff']:<11} "
              f"{r['cutoff_epoch']:<11} {status}")
        if not r["excluded"]:
            total_ictal_used += r["n_ictal"]
            total_pre_used   += r["n_selected_pre"]

    print("-" * 75)
    print(f"  Total ictal epochs used  : {total_ictal_used}")
    print(f"  Total pre-ictal selected : {total_pre_used}")
    print(f"  Effective ratio          : {total_pre_used/total_ictal_used:.2f}:1")
    print(f"  Total usable epochs      : {total_ictal_used + total_pre_used}")

    # ── Build LOPO splits ────────────────────────────────────────────────────
    folds = build_lopo_splits(patients, EXCLUDED_SUBJECTS)

    print()
    print("=" * 70)
    print("LOPO SPLITS")
    print("=" * 70)
    print(f"{'Fold':<6} {'Test PAT':<10} {'Val PAT':<9} "
          f"{'Train PATs':<25} {'#Test subj':<12} {'#Val subj':<10} {'#Train subj'}")
    print("-" * 85)
    for fold in folds:
        print(f"{fold['fold']:<6} PAT_{fold['test_patient']:<6} "
              f"PAT_{fold['val_patient']:<5} "
              f"{str(['PAT_'+p for p in fold['train_patients']]):<25} "
              f"{len(fold['test_subjects']):<12} "
              f"{len(fold['val_subjects']):<10} "
              f"{len(fold['train_subjects'])}")

    # ── Warnings ─────────────────────────────────────────────────────────────
    # Deduplicate warnings (each subject appears once)
    seen_warnings = set()
    warning_lines = []
    for sid, r in selected.items():
        if r["notes"] != "OK" and sid not in seen_warnings:
            seen_warnings.add(sid)
            warning_lines.append(f"  Subject {sid:02d} (PAT_{r['patient_code']}): {r['notes']}")

    print()
    print("NOTES & WARNINGS:")
    for line in warning_lines:
        print(line)

    # ── Save selected_epochs.json ────────────────────────────────────────────
    # Convert lists to JSON-serializable form
    epochs_out = {}
    for sid, r in selected.items():
        epochs_out[str(sid)] = {
            "subject_id"          : r["subject_id"],
            "patient_code"        : r["patient_code"],
            "excluded"            : r["excluded"],
            "flagged"             : r["flagged"],
            "n_ictal"             : r["n_ictal"],
            "n_pre_ictal_selected": r["n_selected_pre"],
            "n_pre_ictal_available": r["n_available_pre_before_cutoff"],
            "cutoff_epoch_index"  : r["cutoff_epoch"],
            "seizure_start_sec"   : r["seizure_start_sec"],
            "seizure_end_sec"     : r["seizure_end_sec"],
            "seizure_duration_sec": r["seizure_duration_sec"],
            "ictal_indices"       : r["ictal_indices"],
            "pre_ictal_indices"   : r["pre_ictal_indices"],
            "notes"               : r["notes"],
        }

    epochs_path = output_dir / "selected_epochs.json"
    with open(epochs_path, "w") as f:
        json.dump(epochs_out, f, indent=2)
    print(f"\n✅ Saved: {epochs_path}")

    # ── Save splits.json ─────────────────────────────────────────────────────
    splits_out = {
        "n_folds"         : len(folds),
        "scheme"          : "Leave-One-Patient-Out (LOPO)",
        "excluded_subjects": EXCLUDED_SUBJECTS,
        "exclusion_reason": {
            str(s): selected[s]["notes"] for s in EXCLUDED_SUBJECTS
        },
        "pre_ictal_ratio" : PRE_ICTAL_RATIO,
        "exclusion_zone_sec": EXCLUSION_ZONE_SEC,
        "folds"           : folds,
    }

    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits_out, f, indent=2)
    print(f"✅ Saved: {splits_path}")

    # ── Save human-readable summary ──────────────────────────────────────────
    summary_lines = [
        "DATASET SPLIT SUMMARY",
        "=" * 70,
        f"Total subjects          : {len(subjects)}",
        f"Excluded subjects       : {EXCLUDED_SUBJECTS} (too few ictal epochs)",
        f"Flagged subjects        : {FLAGGED_SUBJECTS} (not in original paper)",
        f"Usable subjects         : {len(subjects) - len(EXCLUDED_SUBJECTS)}",
        "",
        "EPOCH SELECTION",
        "-" * 40,
        f"Pre-ictal ratio         : {PRE_ICTAL_RATIO}:1",
        f"Exclusion zone          : {EXCLUSION_ZONE_SEC}s before seizure onset",
        f"Pre-ictal source        : earliest epochs from recording start",
        f"Post-ictal              : never used",
        f"Total ictal epochs      : {total_ictal_used}",
        f"Total pre-ictal selected: {total_pre_used}",
        f"Total usable epochs     : {total_ictal_used + total_pre_used}",
        "",
        "LOPO FOLDS",
        "-" * 40,
    ]
    for fold in folds:
        summary_lines.append(
            f"Fold {fold['fold']}: Test=PAT_{fold['test_patient']} "
            f"({len(fold['test_subjects'])} subj) | "
            f"Val=PAT_{fold['val_patient']} "
            f"({len(fold['val_subjects'])} subj) | "
            f"Train={len(fold['train_subjects'])} subj"
        )

    summary_lines += [
        "",
        "NOTES & WARNINGS",
        "-" * 70,
    ]
    for line in warning_lines:
        summary_lines.append(line)

    summary_lines += [
        "",
        "PER-SUBJECT DETAIL",
        "-" * 70,
        f"{'Subj':<6} {'PAT':<6} {'Ictal':<7} {'Pre sel':<9} {'Cutoff ep':<11} Notes",
    ]
    for sid, r in selected.items():
        summary_lines.append(
            f"{sid:<6} {r['patient_code']:<6} {r['n_ictal']:<7} "
            f"{r['n_selected_pre']:<9} {r['cutoff_epoch']:<11} {r['notes']}"
        )

    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / "dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"✅ Saved: {summary_path}")

    print()
    print("=" * 70)
    print("DONE — next step: compute node features (band power + Hjorth)")
    print("=" * 70)


if __name__ == "__main__":
    main()