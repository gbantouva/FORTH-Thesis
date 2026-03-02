"""
Step 6 — Baseline ML: SVM + Random Forest (LOPO Cross-Validation)
==================================================================
Establishes a strong baseline BEFORE any GNN work.
If your GNN does not beat these numbers, something is wrong with the GNN.

Pipeline per fold
-----------------
  1. Load node features  (n_epochs, 19, 8)  for each subject
  2. Flatten per epoch   → (n_epochs, 19*8=152)  feature vector
  3. Fit StandardScaler  on TRAIN subjects only   (no leakage)
  4. Transform train, val, test with train scaler
  5. Train SVM  (RBF kernel) and Random Forest
  6. Evaluate on TEST subjects → AUROC, F1, Accuracy, Precision, Recall

LOPO scheme
-----------
  7 folds (from splits.json).
  Each fold: one patient held out as test, one as validation, rest as train.
  Final results: mean ± std across all 7 folds.

Output files
------------
  results_per_fold.json      — per-fold metrics for every model
  results_summary.json       — mean ± std across folds
  confusion_matrices.png     — confusion matrix grid (SVM + RF, all folds)
  roc_curves.png             — ROC curves per fold + mean AUC
  feature_importance.png     — RF feature importances (averaged across folds)
  baseline_results.txt       — copy-paste ready table for your thesis

Usage
-----
  python step6_baseline_ml.py \
      --features_dir  F:\\...\\node_features \
      --splits        F:\\...\\splits\\splits.json \
      --output_dir    F:\\...\\baseline_results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve
)
from sklearn.dummy           import DummyClassifier

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
N_CHANNELS  = 19
N_FEATURES  = None   # detected at runtime from first file
FLAT_DIM    = None   # detected at runtime from first file

FEATURE_NAMES = [
    f"{ch}_{feat}"
    for ch in [
        "Fp1","Fp2","F7","F3","Fz","F4","F8",
        "T3","C3","Cz","C4","T4",
        "T5","P3","Pz","P4","T6","O1","O2"
    ]
    for feat in ["delta","theta","alpha","beta","gamma",
                 "activity","mobility","complexity"]
]

BAND_FEATURE_NAMES = ["delta","theta","alpha","beta","gamma",
                      "activity","mobility","complexity"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_subjects(subject_ids: list, features_dir: Path):
    """
    Load and flatten node features for a list of subject IDs.

    Returns
    -------
    X : (n_epochs_total, 152)
    y : (n_epochs_total,)
    subject_ids_per_epoch : (n_epochs_total,)  — which subject each epoch came from
    """
    X_list, y_list, sid_list = [], [], []

    for sid in subject_ids:
        feat_file  = features_dir / f"subject_{sid:02d}_node_features.npy"
        label_file = features_dir / f"subject_{sid:02d}_node_labels.npy"

        if not feat_file.exists():
            print(f"    [SKIP] subject_{sid:02d}: feature file not found")
            continue

        feat   = np.load(feat_file)    # (n_epochs, 19, 8)
        labels = np.load(label_file)   # (n_epochs,)

        # Flatten: (n_epochs, 19, 8) → (n_epochs, 152)
        flat_dim = feat.shape[1] * feat.shape[2]
        flat = feat.reshape(len(feat), flat_dim)

        X_list.append(flat)
        y_list.append(labels)
        sid_list.append(np.full(len(labels), sid, dtype=np.int32))

    if not X_list:
        return None, None, None

    return (
        np.concatenate(X_list,   axis=0),
        np.concatenate(y_list,   axis=0),
        np.concatenate(sid_list, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    """Return dict of all evaluation metrics."""
    return {
        "auroc"    : float(roc_auc_score(y_true, y_prob)),
        "f1_macro" : float(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0)),
        "f1_ictal" : float(f1_score(y_true, y_pred, pos_label=1,
                                    zero_division=0)),
        "accuracy" : float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1,
                                           zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred, pos_label=1,
                                        zero_division=0)),
        "n_test"   : int(len(y_true)),
        "n_ictal"  : int(np.sum(y_true == 1)),
        "n_pre"    : int(np.sum(y_true == 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(all_cms, model_names, fold_labels, output_dir):
    """Plot confusion matrix for each fold × model."""
    n_models = len(model_names)
    n_folds  = len(fold_labels)

    fig, axes = plt.subplots(
        n_folds, n_models,
        figsize=(4 * n_models, 3.5 * n_folds)
    )
    if n_folds == 1:
        axes = axes[np.newaxis, :]

    for fi, fold_label in enumerate(fold_labels):
        for mi, model_name in enumerate(model_names):
            ax  = axes[fi, mi]
            cm  = all_cms[model_name][fi]
            im  = ax.imshow(cm, cmap="Blues")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pre-ictal", "Ictal"], fontsize=9)
            ax.set_yticklabels(["Pre-ictal", "Ictal"], fontsize=9)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True",      fontsize=9)
            ax.set_title(f"{model_name} | {fold_label}", fontsize=10,
                         fontweight="bold")

            for r in range(2):
                for c in range(2):
                    ax.text(c, r, str(cm[r, c]),
                            ha="center", va="center",
                            fontsize=13, fontweight="bold",
                            color="white" if cm[r, c] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = output_dir / "confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_roc_curves(all_roc, model_names, output_dir):
    """Plot ROC curves per fold with mean AUC."""
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for mi, model_name in enumerate(model_names):
        ax = axes[mi]
        aucs = []

        for fi, (fpr, tpr, auc_val) in enumerate(all_roc[model_name]):
            ax.plot(fpr, tpr, color=colors[fi % 10], alpha=0.7, linewidth=1.5,
                    label=f"Fold {fi+1} (AUC={auc_val:.3f})")
            aucs.append(auc_val)

        mean_auc = np.mean(aucs)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate",  fontsize=11)
        ax.set_title(f"{model_name}\nMean AUC = {mean_auc:.3f} ± {np.std(aucs):.3f}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    path = output_dir / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_feature_importance(importances_list, output_dir):
    """Plot mean RF feature importances averaged across folds, per feature type."""
    mean_imp = np.mean(importances_list, axis=0)  # (152,)
    std_imp  = np.std(importances_list,  axis=0)

    # Aggregate by feature type (average over 19 channels)
    imp_by_feat = np.zeros((N_FEATURES,))
    std_by_feat = np.zeros((N_FEATURES,))
    for fi in range(N_FEATURES):
        channel_vals = mean_imp[fi::N_FEATURES]   # every 8th element
        imp_by_feat[fi] = channel_vals.mean()
        std_by_feat[fi] = channel_vals.std()

    # Aggregate by channel (average over 8 features)
    imp_by_ch = np.zeros((N_CHANNELS,))
    for ci in range(N_CHANNELS):
        imp_by_ch[ci] = mean_imp[ci * N_FEATURES:(ci + 1) * N_FEATURES].mean()

    channel_names = [
        "Fp1","Fp2","F7","F3","Fz","F4","F8",
        "T3","C3","Cz","C4","T4",
        "T5","P3","Pz","P4","T6","O1","O2"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: importance by feature type
    ax = axes[0]
    x  = np.arange(N_FEATURES)
    bars = ax.bar(x, imp_by_feat, yerr=std_by_feat,
                  capsize=4, alpha=0.8, edgecolor="black",
                  color=["#3498db","#2ecc71","#e74c3c",
                         "#f39c12","#9b59b6","#1abc9c","#e67e22","#95a5a6"])
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_FEATURE_NAMES, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean Importance (averaged over channels)", fontsize=10)
    ax.set_title("RF Feature Importance by Feature Type\n(mean over 19 channels, all folds)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: importance by channel
    ax = axes[1]
    x  = np.arange(N_CHANNELS)
    ax.bar(x, imp_by_ch, alpha=0.8, edgecolor="black", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Importance (averaged over features)", fontsize=10)
    ax.set_title("RF Feature Importance by Channel\n(mean over 8 features, all folds)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def save_thesis_table(results_summary, output_dir):
    """Save a plain-text table ready for copy-paste into your thesis."""
    lines = [
        "BASELINE ML RESULTS — LOPO CROSS-VALIDATION",
        "=" * 72,
        "Dataset: TUC Focal Seizures (34 recordings, 7 patients used)",
        f"Features: {N_CHANNELS} channels x {N_FEATURES} = {FLAT_DIM} features per epoch",
        "         (band-power + Hjorth parameters per channel)",
        "Split: Leave-One-Patient-Out (7 folds)",
        "=" * 72,
        "",
        f"{'Model':<22} {'AUROC':>8} {'F1-macro':>10} {'F1-ictal':>10} "
        f"{'Accuracy':>10} {'Precision':>11} {'Recall':>8}",
        "-" * 72,
    ]

    for model_name, stats in results_summary.items():
        def fmt(key):
            m = stats[key]["mean"]
            s = stats[key]["std"]
            return f"{m:.3f}±{s:.3f}"

        lines.append(
            f"{model_name:<22} {fmt('auroc'):>8} {fmt('f1_macro'):>10} "
            f"{fmt('f1_ictal'):>10} {fmt('accuracy'):>10} "
            f"{fmt('precision'):>11} {fmt('recall'):>8}"
        )

    lines += [
        "-" * 72,
        "",
        "Values reported as mean ± std across 7 LOPO folds.",
        "Primary metric: AUROC (class-imbalance robust).",
        "F1-ictal: F1 score for the positive (ictal) class only.",
        "",
        "Notes:",
        "  - Pre-ictal epochs sampled from recording start (2:1 ratio)",
        "  - 60s exclusion zone before seizure onset",
        "  - Subject 34 (PAT_35, 2 ictal epochs) excluded from all folds",
        "  - Normalisation: StandardScaler fit on train subjects only",
    ]

    text = "\n".join(lines)
    path = output_dir / "baseline_results.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path.name}")
    print()
    print(text)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline ML (SVM + RF) with LOPO cross-validation"
    )
    parser.add_argument("--features_dir", required=True,
                        help="Directory with subject_XX_node_features.npy files")
    parser.add_argument("--splits",       required=True,
                        help="Path to splits.json (from step4)")
    parser.add_argument("--output_dir",   required=True,
                        help="Directory to save results and plots")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load splits ──────────────────────────────────────────────────────────
    with open(args.splits) as f:
        splits = json.load(f)

    folds = splits["folds"]

    # ── Detect actual feature shape from first available file ──────────────
    global N_FEATURES, FLAT_DIM, BAND_FEATURE_NAMES
    first_feat_file = sorted(features_dir.glob("subject_*_node_features.npy"))
    if not first_feat_file:
        print("ERROR: No node feature files found in", features_dir)
        return
    probe = np.load(first_feat_file[0])   # (n_epochs, n_channels, n_features)
    N_FEATURES = probe.shape[2]
    FLAT_DIM   = N_CHANNELS * N_FEATURES

    # Rebuild feature name list for the actual feature count
    if N_FEATURES == 8:
        BAND_FEATURE_NAMES = ["delta","theta","alpha","beta","gamma",
                              "activity","mobility","complexity"]
    elif N_FEATURES == 12:
        BAND_FEATURE_NAMES = ["delta","theta","alpha","beta","gamma",
                              "activity","mobility","complexity",
                              "feat_8","feat_9","feat_10","feat_11"]
    else:
        BAND_FEATURE_NAMES = [f"feat_{i}" for i in range(N_FEATURES)]

    print("=" * 70)
    print("STEP 6 — BASELINE ML: SVM + RANDOM FOREST")
    print("=" * 70)
    print(f"  Features dir : {features_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  LOPO folds   : {len(folds)}")
    print(f"  Detected feature shape: {probe.shape}  →  n_features={N_FEATURES}")
    print(f"  Flat dim     : {FLAT_DIM}  ({N_CHANNELS} channels x {N_FEATURES} features)")
    print()

    # ── Model definitions ────────────────────────────────────────────────────
    models = {
        "SVM (RBF)"     : SVC(kernel="rbf", C=1.0, gamma="scale",
                               probability=True, random_state=42,
                               class_weight="balanced"),
        "Random Forest" : RandomForestClassifier(n_estimators=200,
                                                  max_depth=None,
                                                  class_weight="balanced",
                                                  random_state=42,
                                                  n_jobs=-1),
        "Majority Vote" : DummyClassifier(strategy="most_frequent",
                                           random_state=42),
    }

    # Storage
    all_fold_results = []
    all_cms          = {m: [] for m in models}
    all_roc          = {m: [] for m in models}
    rf_importances   = []

    # ── LOPO loop ────────────────────────────────────────────────────────────
    for fold in folds:
        fold_num    = fold["fold"]
        test_pat    = fold["test_patient"]
        val_pat     = fold["val_patient"]
        train_subs  = fold["train_subjects"]
        val_subs    = fold["val_subjects"]
        test_subs   = fold["test_subjects"]

        fold_label = f"Fold{fold_num}_test=PAT{test_pat}"
        print(f"{'='*70}")
        print(f"FOLD {fold_num}/7  |  Test: PAT_{test_pat}  "
              f"Val: PAT_{val_pat}  Train: {len(train_subs)} subjects")
        print(f"{'='*70}")

        # Load data
        X_train, y_train, _ = load_subjects(train_subs, features_dir)
        X_val,   y_val,   _ = load_subjects(val_subs,   features_dir)
        X_test,  y_test,  _ = load_subjects(test_subs,  features_dir)

        if X_train is None or X_test is None:
            print(f"  [SKIP] Not enough data for fold {fold_num}")
            continue

        print(f"  Train: {X_train.shape[0]} epochs "
              f"({int(np.sum(y_train==1))} ictal / {int(np.sum(y_train==0))} pre)")
        print(f"  Val:   {X_val.shape[0]}   epochs "
              f"({int(np.sum(y_val==1))} ictal / {int(np.sum(y_val==0))} pre)")
        print(f"  Test:  {X_test.shape[0]}  epochs "
              f"({int(np.sum(y_test==1))} ictal / {int(np.sum(y_test==0))} pre)")

        # ── Normalise using TRAIN statistics only ────────────────────────────
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        fold_result = {
            "fold"        : fold_num,
            "test_patient": test_pat,
            "val_patient" : val_pat,
            "n_train"     : int(X_train.shape[0]),
            "n_val"       : int(X_val.shape[0]),
            "n_test"      : int(X_test.shape[0]),
            "models"      : {},
        }

        # ── Train & evaluate each model ──────────────────────────────────────
        for model_name, model in models.items():
            print(f"\n  [{model_name}]")

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)

            print(f"    AUROC={metrics['auroc']:.3f}  "
                  f"F1-ictal={metrics['f1_ictal']:.3f}  "
                  f"Accuracy={metrics['accuracy']:.3f}")

            fold_result["models"][model_name] = metrics

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            all_cms[model_name].append(cm)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            all_roc[model_name].append((fpr, tpr, metrics["auroc"]))

            # RF feature importances
            if model_name == "Random Forest":
                rf_importances.append(model.feature_importances_)

        all_fold_results.append(fold_result)
        print()

    # ── Aggregate results across folds ──────────────────────────────────────
    print("=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    metric_keys = ["auroc", "f1_macro", "f1_ictal",
                   "accuracy", "precision", "recall"]

    results_summary = {}
    for model_name in models:
        results_summary[model_name] = {}
        for key in metric_keys:
            values = [
                fold["models"][model_name][key]
                for fold in all_fold_results
                if model_name in fold["models"]
            ]
            results_summary[model_name][key] = {
                "mean"   : float(np.mean(values)),
                "std"    : float(np.std(values)),
                "values" : [float(v) for v in values],
            }

    # ── Save JSON results ────────────────────────────────────────────────────
    with open(output_dir / "results_per_fold.json", "w", encoding="utf-8") as f:
        json.dump(all_fold_results, f, indent=2)
    print(f"\n  Saved: results_per_fold.json")

    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Saved: results_summary.json")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    fold_labels = [f"Fold{f['fold']} PAT{f['test_patient']}"
                   for f in all_fold_results]

    plot_confusion_matrices(all_cms, list(models.keys()), fold_labels, output_dir)
    plot_roc_curves(all_roc, list(models.keys()), output_dir)

    if rf_importances:
        plot_feature_importance(rf_importances, output_dir)

    # ── Thesis table ─────────────────────────────────────────────────────────
    print("\nResults table:")
    save_thesis_table(results_summary, output_dir)

    print()
    print("=" * 70)
    print("DONE — baseline complete")
    print("=" * 70)
    print()
    print("  Next step: step7_gcn.py")
    print("    Build graph objects from DTF/PDC + node features,")
    print("    train a 2-layer GCN with the same LOPO folds,")
    print("    compare against these baseline numbers.")
    print("=" * 70)


if __name__ == "__main__":
    main()