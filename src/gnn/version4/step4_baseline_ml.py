"""
Step 4 — Baseline ML: Random Forest + SVM  (LOPO Cross-Validation)
===================================================================
DESIGN DECISIONS (document in thesis):

1. EVALUATION: Leave-One-Patient-Out (LOPO) CV.
   Correct for clinical datasets — tests generalisation to an unseen patient.
   We split by PATIENT (not subject), so PAT14's 8 recordings are either
   all in train or all in test.

2. FEATURE SELECTION: None beyond what step3 already decided.
   With only 53 curated features, we do NOT need a secondary selector.
   Adding a selector fit on training data would be fine in principle, but
   with 53 features and a small dataset, it adds variance without benefit.
   We use ALL 53 features, scaled per fold (StandardScaler fit on train only).

3. IMBALANCE:
   - class_weight='balanced' in both RF and SVM.
   - Pre-processing already set ratio=2 (2 pre-ictal per ictal).
   - Primary metrics are AUC, F1 (ictal class), Sensitivity, MCC.
     We report accuracy last — it is misleading with imbalanced data.

4. HYPERPARAMETERS:
   RF : n_estimators=300, max_depth=6, min_samples_leaf=5
        → Deliberately shallow to prevent memorisation of 34-subject data
   SVM: C=1.0, gamma='scale', RBF kernel
        → Conservative regularisation, no grid search (too few samples to
          tune without leaking val-set info)

Outputs:
  results_all.json          all fold metrics + summary stats
  summary_table.csv         mean ± std per model
  cm_{model}_{patient}.png  per-fold confusion matrices
  roc_{model}.png           LOPO ROC curves
  per_fold_{model}.png      per-patient bar chart
  model_comparison.png      final side-by-side comparison

Usage:
  python step4_baseline_ml.py \\
      --featfile  features/features_all.npz \\
      --outputdir results/baseline_ml
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    """
    Returns a dict of all classification metrics.
    Sensitivity = recall for the ictal (positive) class.
    Specificity = recall for the pre-ictal (negative) class.
    MCC is balanced and reliable for imbalanced datasets.
    """
    cm           = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity  = tp / (tp + fn + 1e-12)
    specificity  = tn / (tn + fp + 1e-12)
    return {
        'auc':         float(roc_auc_score(y_true, y_prob)),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision':   float(precision_score(y_true, y_pred, zero_division=0)),
        'mcc':         float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name, patient_id, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Pre-ictal', 'Ictal'],
        yticklabels=['Pre-ictal', 'Ictal']
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'{model_name} | Test: {patient_id}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / f'cm_{model_name.lower().replace(" ","_")}_{patient_id}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_all_folds(fold_roc_data, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for fpr, tpr, auc, pat in fold_roc_data:
        ax.plot(fpr, tpr, alpha=0.55, lw=1.5, label=f'{pat} (AUC={auc:.2f})')
        aucs.append(auc)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'{model_name} — LOPO ROC\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f'roc_{model_name.lower().replace(" ","_")}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_per_fold_metrics(fold_metrics, model_name, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange']
    x        = np.arange(len(patients))
    width    = 0.2
    fig, ax  = plt.subplots(figsize=(12, 5))
    for i, (met, col) in enumerate(zip(met_keys, colors)):
        vals = [m[met] for m in fold_metrics]
        ax.bar(x + i * width, vals, width, label=met.upper(),
               color=col, alpha=0.8, edgecolor='black')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title(f'{model_name} — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(
        output_dir / f'per_fold_{model_name.lower().replace(" ","_")}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_aggregate_confusion(all_y_true, all_y_pred, model_name, output_dir):
    cm      = confusion_matrix(all_y_true, all_y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',   'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalised'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=['Pre-ictal', 'Ictal'],
                    yticklabels=['Pre-ictal', 'Ictal'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'{model_name} — Aggregate CM ({title})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        output_dir / f'cm_aggregate_{model_name.lower().replace(" ","_")}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_feature_importance(importances_per_fold, feature_names, model_name, output_dir, top_n=30):
    """
    importances_per_fold : list of (n_features,) arrays — one per fold.
    We average across folds for a stable importance estimate.
    """
    mean_imp = np.mean(importances_per_fold, axis=0)
    idx      = np.argsort(mean_imp)[::-1][:top_n]
    fig, ax  = plt.subplots(figsize=(12, 5))
    ax.bar(range(top_n), mean_imp[idx], color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(np.array(feature_names)[idx], rotation=90, fontsize=7)
    ax.set_ylabel('Mean importance (LOPO folds)', fontsize=11)
    ax.set_title(f'{model_name} — Top {top_n} Feature Importances', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(
        output_dir / f'feature_importance_{model_name.lower().replace(" ","_")}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()


# ─────────────────────────────────────────────────────────────
# LOPO evaluation for one model
# ─────────────────────────────────────────────────────────────

def run_lopo(model_name, model, X, y, patient_ids, feature_names, output_dir):
    """
    Runs Leave-One-Patient-Out CV.
    Scaling is fit on the training set only — no leakage.
    Returns fold_metrics list and summary_stats dict.
    """
    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []
    importances   = []   # RF only

    print(f'\n{"=" * 60}')
    print(f'  {model_name} — LOPO CV ({len(patients)} patient folds)')
    print(f'{"=" * 60}')

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: test set has only one class — cannot evaluate')
            continue

        # ── Scale: fit on train ONLY ───────────────────────
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_train)
        X_te_sc  = scaler.transform(X_test)

        # ── Fit ────────────────────────────────────────────
        import sklearn.base as skbase
        clf = skbase.clone(model)   # fresh clone per fold
        clf.fit(X_tr_sc, y_train)

        # ── Predict ────────────────────────────────────────
        y_pred = clf.predict(X_te_sc)
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_te_sc)[:, 1]
        else:
            # SVC with probability=True uses predict_proba; fallback just in case
            raw = clf.decision_function(X_te_sc)
            y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['patient'] = pat
        metrics['n_train'] = int(train_mask.sum())
        metrics['n_test']  = int(test_mask.sum())
        fold_metrics.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        plot_confusion_matrix(confusion_matrix(y_test, y_pred), model_name, pat, output_dir)

        # RF feature importance
        if hasattr(clf, 'feature_importances_'):
            importances.append(clf.feature_importances_)

        print(f'  {pat:8s} | AUC={metrics["auc"]:.3f}  F1={metrics["f1"]:.3f}'
              f'  Sens={metrics["sensitivity"]:.3f}  Spec={metrics["specificity"]:.3f}'
              f'  MCC={metrics["mcc"]:.3f}')

    if len(fold_metrics) == 0:
        print(f'  [ERROR] No valid folds for {model_name}')
        return [], {}

    # ── Aggregate plots ──────────────────────────────────
    plot_roc_all_folds(fold_roc_data, model_name, output_dir)
    plot_per_fold_metrics(fold_metrics, model_name, output_dir)
    plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred), model_name, output_dir)

    if importances:
        plot_feature_importance(importances, feature_names, model_name, output_dir)

    # ── Summary stats ────────────────────────────────────
    met_keys      = ['auc', 'f1', 'sensitivity', 'specificity', 'precision', 'mcc']
    summary_stats = {}
    print(f'\n  {"─" * 50}')
    print(f'  {model_name} — Mean ± Std across {len(fold_metrics)} folds')
    print(f'  {"─" * 50}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    return fold_metrics, summary_stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 4 — Baseline ML with LOPO CV')
    parser.add_argument('--featfile',  required=True, help='features/features_all.npz')
    parser.add_argument('--outputdir', default='results/baseline_ml')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────
    print('=' * 60)
    print('STEP 4 — BASELINE ML')
    print('=' * 60)
    data = np.load(args.featfile, allow_pickle=True)
    X             = data['X'].astype(np.float32)
    y             = data['y'].astype(np.int64)
    patient_ids   = data['patient_ids']
    feature_names = data['feature_names'].tolist()

    # Safety: replace any remaining NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f'Loaded  : X={X.shape}  y={y.shape}')
    print(f'Ictal   : {(y == 1).sum()}  |  Pre-ictal: {(y == 0).sum()}')
    print(f'Patients: {np.unique(patient_ids).tolist()}')
    print(f'Features: {X.shape[1]} (no secondary selection — all features from step 3)')

    # ── Model definitions ──────────────────────────────────
    # Deliberately simple / regularised to avoid overfitting on small data.
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=6,           # shallow = less overfit
            min_samples_leaf=5,    # require at least 5 samples per leaf
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        'SVM RBF': SVC(
            kernel='rbf',
            C=1.0,                 # moderate regularisation
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
        ),
    }

    all_results = {}

    for model_name, model in models.items():
        fold_metrics, summary_stats = run_lopo(
            model_name, model,
            X, y, patient_ids, feature_names, output_dir
        )
        all_results[model_name] = {
            'fold_metrics':  fold_metrics,
            'summary_stats': summary_stats,
        }

    # ── Save JSON ──────────────────────────────────────────
    results_path = output_dir / 'results_all.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nAll results saved → {results_path}')

    # ── Final comparison table ─────────────────────────────
    print('\n' + '=' * 60)
    print('FINAL COMPARISON')
    print('=' * 60)
    rows = []
    for model_name, res in all_results.items():
        if not res['summary_stats']:
            continue
        row = {'Model': model_name}
        for k, v in res['summary_stats'].items():
            row[k] = f"{v['mean']:.3f} ± {v['std']:.3f}"
        rows.append(row)

    if rows:
        df_summary = pd.DataFrame(rows).set_index('Model')
        print(df_summary.to_string())
        df_summary.to_csv(output_dir / 'summary_table.csv', encoding='utf-8')
        print(f'\nSummary table → {output_dir / "summary_table.csv"}')

    # ── Side-by-side bar chart ─────────────────────────────
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato']
    x        = np.arange(len(met_keys))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (model_name, res) in enumerate(all_results.items()):
        if not res['summary_stats']:
            continue
        means  = [res['summary_stats'][k]['mean'] for k in met_keys]
        stds   = [res['summary_stats'][k]['std']  for k in met_keys]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=model_name, color=colors[i % len(colors)],
               capsize=4, edgecolor='black', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('Baseline ML — Model Comparison (LOPO CV)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Model comparison chart → {output_dir / "model_comparison.png"}')

    print('\n' + '=' * 60)
    print('STEP 4 COMPLETE')
    print('=' * 60)
    print('\nNext: python step5_gnn_supervised.py --featfile features/features_all.npz')


if __name__ == '__main__':
    main()
