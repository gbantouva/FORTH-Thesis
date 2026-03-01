"""
Step 3b — Baseline ML Classifiers
===================================
Trains and evaluates classical ML models (Logistic Regression, SVM, Random
Forest) on flattened EEG features BEFORE using a GNN.

This step serves two purposes:
  1. Sanity check: if connectivity + node features carry no signal, the GNN
     won't help either.
  2. Thesis baseline: GNN must beat these numbers to be worth the complexity.

Three feature sets are evaluated independently and combined:
  A) Connectivity only  — DTF + PDC (upper triangle, 6 bands)
  B) Node features only — band power + Hjorth + stats (19 × 12)
  C) Combined           — A + B concatenated

Evaluation strategy
────────────────────
Leave-One-Subject-Out Cross Validation (LOSO-CV):
  - Train on 33 subjects, test on 1, repeat 34 times.
  - This is the correct strategy for your dataset (8 patients, 34 recordings).
  - It tests generalization to unseen patients — which is what matters clinically.

Metrics reported per fold and averaged:
  Accuracy, Sensitivity (Recall for ictal), Specificity, F1, AUC-ROC

Usage:
    python step3b_baseline_ml.py \\
        --conndir    path/to/connectivity \\
        --featdir    path/to/node_features \\
        --epochdir   path/to/preprocessed_epochs \\
        --outputdir  path/to/results
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
BANDS      = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']
N_CHANNELS = 19

MODELS = {
    'LogReg': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000, class_weight='balanced',
                                      solver='lbfgs', C=1.0)),
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    SVC(kernel='rbf', C=1.0, gamma='scale',
                       class_weight='balanced', probability=True)),
    ]),
    'RandomForest': Pipeline([
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=10,
                                        class_weight='balanced',
                                        n_jobs=-1, random_state=42)),
    ]),
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_connectivity_features(conn_dir, subject_name):
    """
    Load DTF + PDC matrices and flatten upper triangle (no diagonal).
    Shape per epoch: (2 × 6 × n_upper,) where n_upper = 19×18//2 = 171
    Total: 2 × 6 × 171 = 2052 features

    Returns
    -------
    features : ndarray (n_epochs, 2052)  or None
    """
    npz_path = conn_dir / f"{subject_name}_graphs.npz"
    if not npz_path.exists():
        return None

    data    = np.load(npz_path)
    n_epochs = len(data['labels'])

    # Upper triangle indices (no diagonal)
    triu_idx = np.triu_indices(N_CHANNELS, k=1)   # 171 pairs

    feat_list = []
    for band in BANDS:
        dtf = data[f'dtf_{band}']   # (n_epochs, 19, 19)
        pdc = data[f'pdc_{band}']   # (n_epochs, 19, 19)
        feat_list.append(dtf[:, triu_idx[0], triu_idx[1]])   # (n_epochs, 171)
        feat_list.append(pdc[:, triu_idx[0], triu_idx[1]])

    # Stack: (n_epochs, 2052)
    features = np.concatenate(feat_list, axis=1).astype(np.float32)
    return features


def load_node_features(feat_dir, subject_name):
    """
    Load node features and flatten channels × features.
    Shape per epoch: (19 × 12,) = 228 features

    Returns
    -------
    features : ndarray (n_epochs, 228)  or None
    """
    # Prefer normalized version
    norm_path = feat_dir / f"{subject_name}_node_features_normalized.npy"
    raw_path  = feat_dir / f"{subject_name}_node_features.npy"

    path = norm_path if norm_path.exists() else (raw_path if raw_path.exists() else None)
    if path is None:
        return None

    feats = np.load(path)   # (n_epochs, 19, 12)
    return feats.reshape(len(feats), -1).astype(np.float32)


def load_labels(conn_dir, epoch_dir, subject_name):
    """Load labels, preferring connectivity file (already aligned with valid epochs)."""
    npz_path = conn_dir / f"{subject_name}_graphs.npz"
    if npz_path.exists():
        return np.load(npz_path)['labels']
    lbl_path = epoch_dir / f"{subject_name}_labels.npy"
    if lbl_path.exists():
        return np.load(lbl_path)
    return None


def load_training_mask(epoch_dir, subject_name, n_epochs):
    """Load training mask (exclude post-ictal epochs)."""
    mask_path = epoch_dir / f"{subject_name}_training_mask.npy"
    if mask_path.exists():
        full_mask = np.load(mask_path)
        return full_mask
    return np.ones(n_epochs, dtype=bool)


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD DATASET PER FEATURE SET
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(subject_names, conn_dir, feat_dir, epoch_dir, feature_set):
    """
    Load and align features + labels for all subjects.

    feature_set : 'connectivity' | 'node' | 'combined'

    Returns
    -------
    X_all      : list of ndarray (n_epochs_i, n_features)  one per subject
    y_all      : list of ndarray (n_epochs_i,)
    subj_names : list of str (same order)
    """
    X_all      = []
    y_all      = []
    valid_subj = []

    for subj in subject_names:
        labels = load_labels(conn_dir, epoch_dir, subj)
        if labels is None:
            continue

        n_epochs = len(labels)

        # Load requested features
        if feature_set == 'connectivity':
            X = load_connectivity_features(conn_dir, subj)
        elif feature_set == 'node':
            X = load_node_features(feat_dir, subj)
        else:  # combined
            Xc = load_connectivity_features(conn_dir, subj)
            Xn = load_node_features(feat_dir, subj)
            if Xc is not None and Xn is not None:
                # Align lengths (connectivity may have fewer epochs after VAR filtering)
                n = min(len(Xc), len(Xn), n_epochs)
                X = np.concatenate([Xc[:n], Xn[:n]], axis=1)
                labels = labels[:n]
            else:
                X = Xc if Xc is not None else Xn

        if X is None:
            print(f"  ⚠️  Skipping {subj} — features not found")
            continue

        # Align epoch counts
        n = min(len(X), len(labels))
        X, labels = X[:n], labels[:n]

        # Apply training mask (exclude post-ictal)
        mask = load_training_mask(epoch_dir, subj, n)[:n]
        X, labels = X[mask], labels[mask]

        # Skip subjects with only one class
        if len(np.unique(labels)) < 2:
            print(f"  ⚠️  Skipping {subj} — only one class present")
            continue

        # Replace any NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_all.append(X)
        y_all.append(labels)
        valid_subj.append(subj)

    return X_all, y_all, valid_subj


# ══════════════════════════════════════════════════════════════════════════════
# 3. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute classification metrics.

    Returns dict with accuracy, sensitivity, specificity, f1, auc
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = specificity = 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    return {
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'sensitivity': float(sensitivity),   # recall for ictal class
        'specificity': float(specificity),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'auc':         float(auc),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOSO-CV
# ══════════════════════════════════════════════════════════════════════════════

def run_loso(X_all, y_all, subj_names, model_name, model, feature_set):
    """
    Leave-One-Subject-Out cross validation.

    For each fold:
      - Test  : one subject
      - Train : all other subjects concatenated

    Returns
    -------
    results : list of dicts, one per fold
    """
    n_subjects = len(X_all)
    results    = []

    pbar = tqdm(range(n_subjects),
                desc=f"  LOSO {model_name} [{feature_set}]",
                leave=False)

    for test_idx in pbar:
        # Split
        train_X = np.concatenate([X_all[i] for i in range(n_subjects) if i != test_idx])
        train_y = np.concatenate([y_all[i] for i in range(n_subjects) if i != test_idx])
        test_X  = X_all[test_idx]
        test_y  = y_all[test_idx]

        # Skip degenerate folds
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            continue

        # Fit
        try:
            model.fit(train_X, train_y)
            y_pred = model.predict(test_X)
            y_prob = model.predict_proba(test_X)[:, 1]
        except Exception as e:
            print(f"\n    ⚠️  {subj_names[test_idx]} failed: {e}")
            continue

        metrics = compute_metrics(test_y, y_pred, y_prob)
        metrics.update({
            'subject':     subj_names[test_idx],
            'model':       model_name,
            'feature_set': feature_set,
            'n_train':     len(train_y),
            'n_test':      len(test_y),
            'n_ictal_test':   int((test_y == 1).sum()),
            'n_preictal_test': int((test_y == 0).sum()),
        })
        results.append(metrics)

        pbar.set_postfix({
            'acc':  f"{metrics['accuracy']:.2f}",
            'auc':  f"{metrics['auc']:.2f}",
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(df, output_dir):
    """
    Summary plots:
      A) Boxplot of AUC per model per feature set
      B) Per-subject AUC heatmap (model × subject)
      C) Metric comparison bar chart
    """
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── A: AUC boxplot ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    feat_sets  = df['feature_set'].unique()
    model_names = df['model'].unique()
    colors     = ['#2980b9', '#27ae60', '#e74c3c']

    x_positions = []
    x_labels    = []
    box_data    = []
    box_colors  = []

    pos = 0
    gap = 0.5
    for feat in feat_sets:
        for i, mdl in enumerate(model_names):
            sub = df[(df['feature_set'] == feat) & (df['model'] == mdl)]['auc'].dropna()
            box_data.append(sub.values)
            box_colors.append(colors[i % len(colors)])
            x_positions.append(pos)
            x_labels.append(f"{feat}\n{mdl}")
            pos += 1
        pos += gap

    bp = ax1.boxplot(box_data, positions=x_positions, patch_artist=True,
                     widths=0.6, showfliers=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, fontsize=8, rotation=20, ha='right')
    ax1.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
    ax1.set_title('LOSO-CV AUC per Model per Feature Set',
                  fontsize=13, fontweight='bold')
    ax1.axhline(0.5, color='gray', linestyle='--', lw=1.5, label='Random chance')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=9)

    # ── B: Mean metrics bar chart ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    summary = df.groupby(['feature_set', 'model'])[metrics].mean().reset_index()
    summary['label'] = summary['feature_set'] + '\n' + summary['model']

    x = np.arange(len(summary))
    width = 0.15
    for k, metric in enumerate(metrics):
        ax2.bar(x + k * width, summary[metric], width,
                label=metric, alpha=0.8)

    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(summary['label'], fontsize=7, rotation=30, ha='right')
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Metrics (LOSO-CV)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    # ── C: AUC per subject (best model per feature set) ───────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    best_model = (
        df.groupby(['feature_set', 'model'])['auc']
        .mean()
        .idxmax()
    )
    best_feat, best_mdl = best_model
    sub_df = df[(df['feature_set'] == best_feat) & (df['model'] == best_mdl)]
    subjects = sorted(sub_df['subject'].unique())
    aucs     = [sub_df[sub_df['subject'] == s]['auc'].values[0] for s in subjects]

    bar_colors = ['#e74c3c' if a < 0.6 else '#f39c12' if a < 0.8 else '#27ae60'
                  for a in aucs]
    ax3.bar(range(len(subjects)), aucs, color=bar_colors, alpha=0.8, edgecolor='black')
    ax3.axhline(0.5, color='gray', linestyle='--', lw=1.5)
    ax3.axhline(0.8, color='green', linestyle='--', lw=1, alpha=0.5, label='AUC=0.8')
    ax3.set_xticks(range(len(subjects)))
    ax3.set_xticklabels(subjects, rotation=90, fontsize=7)
    ax3.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
    ax3.set_title(f'Per-Subject AUC\n(best: {best_mdl} / {best_feat})',
                  fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=9)

    plt.suptitle('Step 3b — Baseline ML Results (LOSO-CV)\n'
                 'Ictal (1) vs Pre-ictal (0)',
                 fontsize=14, fontweight='bold')

    plot_path = output_dir / 'baseline_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {plot_path}")


def print_summary_table(df):
    """Print a clean summary table to console."""
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    summary = df.groupby(['feature_set', 'model'])[metrics].agg(['mean', 'std'])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()

    print("\n" + "=" * 90)
    print("RESULTS SUMMARY (LOSO-CV mean ± std)")
    print("=" * 90)
    header = f"{'Feature Set':<15} {'Model':<14} " + \
             "  ".join([f"{m.upper()[:4]:>10}" for m in metrics])
    print(header)
    print("-" * 90)

    for _, row in summary.iterrows():
        line = f"{row['feature_set']:<15} {row['model']:<14} "
        for m in metrics:
            line += f"  {row[f'{m}_mean']:>4.3f}±{row[f'{m}_std']:>4.3f}"
        print(line)
    print("=" * 90)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Baseline ML classifiers with LOSO-CV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--conndir',   required=True,
                        help='Directory with subject_XX_graphs.npz (Step 2)')
    parser.add_argument('--featdir',   required=True,
                        help='Directory with subject_XX_node_features.npy (Step 3a)')
    parser.add_argument('--epochdir',  required=True,
                        help='Directory with subject_XX_epochs.npy + labels + masks')
    parser.add_argument('--outputdir', required=True,
                        help='Output directory for results')
    parser.add_argument('--feature_sets', nargs='+',
                        default=['connectivity', 'node', 'combined'],
                        choices=['connectivity', 'node', 'combined'],
                        help='Which feature sets to evaluate (default: all)')
    parser.add_argument('--models', nargs='+',
                        default=['LogReg', 'SVM', 'RandomForest'],
                        choices=['LogReg', 'SVM', 'RandomForest'],
                        help='Which models to run (default: all)')
    args = parser.parse_args()

    conn_dir   = Path(args.conndir)
    feat_dir   = Path(args.featdir)
    epoch_dir  = Path(args.epochdir)
    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover subjects ─────────────────────────────────────────────────
    subject_names = sorted([
        f.stem.replace('_epochs', '')
        for f in epoch_dir.glob('subject_*_epochs.npy')
    ])

    print("=" * 72)
    print("STEP 3b — BASELINE ML CLASSIFIERS")
    print("=" * 72)
    print(f"  Subjects:     {len(subject_names)}")
    print(f"  Feature sets: {args.feature_sets}")
    print(f"  Models:       {args.models}")
    print(f"  Strategy:     Leave-One-Subject-Out CV (LOSO)")
    print(f"  Output:       {output_dir}")
    print("=" * 72)

    all_results = []

    for feature_set in args.feature_sets:
        print(f"\n{'─'*72}")
        print(f"  Feature set: {feature_set.upper()}")
        print(f"{'─'*72}")

        X_all, y_all, valid_subj = build_dataset(
            subject_names, conn_dir, feat_dir, epoch_dir, feature_set
        )

        print(f"  Valid subjects: {len(valid_subj)}")
        print(f"  Feature dim:    {X_all[0].shape[1] if X_all else 'N/A'}")
        total_ictal   = sum((y == 1).sum() for y in y_all)
        total_preictal = sum((y == 0).sum() for y in y_all)
        print(f"  Total epochs:   {sum(len(y) for y in y_all):,}  "
              f"(ictal={total_ictal:,}, pre-ictal={total_preictal:,})")

        for model_name in args.models:
            if model_name not in MODELS:
                print(f"  ⚠️  Unknown model: {model_name}")
                continue

            # Re-instantiate model to avoid state leakage between feature sets
            from sklearn.base import clone
            model = clone(MODELS[model_name])

            fold_results = run_loso(
                X_all, y_all, valid_subj,
                model_name, model, feature_set
            )
            all_results.extend(fold_results)

            if fold_results:
                aucs = [r['auc'] for r in fold_results]
                accs = [r['accuracy'] for r in fold_results]
                print(f"    {model_name:<15} — "
                      f"AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
                      f"Acc {np.mean(accs):.3f}±{np.std(accs):.3f}")

    if not all_results:
        print("\n❌ No results — check your input directories.")
        return

    # ── Save results ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = output_dir / 'baseline_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Saved: {csv_path}")

    # ── Summary table ──────────────────────────────────────────────────────
    print_summary_table(df)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_results(df, output_dir)

    # ── JSON summary ──────────────────────────────────────────────────────
    metrics     = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    summary_dict = {}
    for feat in df['feature_set'].unique():
        summary_dict[feat] = {}
        for mdl in df['model'].unique():
            sub = df[(df['feature_set'] == feat) & (df['model'] == mdl)]
            summary_dict[feat][mdl] = {
                m: {'mean': float(sub[m].mean()), 'std': float(sub[m].std())}
                for m in metrics
            }

    json_path = output_dir / 'baseline_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"  ✅ Saved: {json_path}")

    # ── Final recommendation ───────────────────────────────────────────────
    best_row = df.groupby(['feature_set', 'model'])['auc'].mean().idxmax()
    best_auc = df.groupby(['feature_set', 'model'])['auc'].mean().max()

    print("\n" + "=" * 72)
    print("BASELINE COMPLETE")
    print("=" * 72)
    print(f"  Best combination:  {best_row[1]} on {best_row[0]}")
    print(f"  Best mean AUC:     {best_auc:.3f}")
    print()
    print("  → These numbers are your thesis baseline.")
    print("    Your GCN (Step 3d) must beat them to justify the complexity.")
    print()
    print("  Next steps:")
    print("    Step 3c: Build PyTorch Geometric graphs")
    print("    Step 3d: Train supervised GCN")
    print("=" * 72)


if __name__ == '__main__':
    main()
