"""
Step 4 - Baseline ML (Random Forest + SVM)
===========================================
- Leave-One-Patient-Out (LOPO) cross-validation
- Variance thresholding + RF feature selection (top-N)
- Models: Random Forest, SVM (RBF kernel)
- Metrics: AUC-ROC, F1, Sensitivity, Specificity, Precision, MCC
- Outputs: confusion matrices, ROC curves, learning curves (overfit check),
           feature importance, per-fold results, final summary

Usage:
  python step4_baseline_ml.py \
      --featfile features/features_all.npz \
      --outputdir results/baseline_ml \
      --topfeatures 100
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, precision_score, recall_score,
    matthews_corrcoef, classification_report
)
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-12)   # recall for ictal class
    specificity = tn / (tn + fp + 1e-12)
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
# Feature selection (fit on train, apply to test)
# ─────────────────────────────────────────────────────────────

def select_features(X_train, y_train, X_test, feature_names, top_n=100):
    """
    1. Remove near-zero variance features
    2. Fit RF on train, keep top_n by importance
    Returns X_train_sel, X_test_sel, selected_names, importances
    """
    # Step 1: variance threshold
    vt = VarianceThreshold(threshold=1e-6)
    X_tr_vt = vt.fit_transform(X_train)
    X_te_vt = vt.transform(X_test)
    names_vt = np.array(feature_names)[vt.get_support()]

    # Step 2: RF importance ranking
    rf_sel = RandomForestClassifier(
        n_estimators=100, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_sel.fit(X_tr_vt, y_train)
    importances = rf_sel.feature_importances_

    top_idx = np.argsort(importances)[::-1][:top_n]
    top_idx_sorted = np.sort(top_idx)

    return (
        X_tr_vt[:, top_idx_sorted],
        X_te_vt[:, top_idx_sorted],
        names_vt[top_idx_sorted],
        importances[top_idx_sorted],
    )


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
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'{model_name} | Test: {patient_id}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / f'cm_{model_name.lower().replace(" ","_")}_{patient_id}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_all_folds(fold_roc_data, model_name, output_dir):
    """
    fold_roc_data: list of (fpr, tpr, auc, patient_id)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for fpr, tpr, auc, pat in fold_roc_data:
        ax.plot(fpr, tpr, alpha=0.5, lw=1.5, label=f'{pat} (AUC={auc:.2f})')
        aucs.append(auc)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'{model_name} — LOPO ROC Curves\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{model_name.lower().replace(" ","_")}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curve(model, X, y, model_name, output_dir):
    """
    Learning curve: training vs validation score as a function of training size.
    Used to diagnose overfitting / underfitting.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=5,
        scoring='roc_auc',
        train_sizes=np.linspace(0.2, 1.0, 8),
        n_jobs=-1,
        random_state=42,
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, 'o-', color='royalblue', label='Train AUC')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='royalblue')
    ax.plot(train_sizes, val_mean, 's-', color='tomato', label='Validation AUC (CV-5)')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='tomato')
    ax.set_xlabel('Training samples', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(f'{model_name} — Learning Curve (Overfit Check)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    gap = float(np.mean(train_mean - val_mean))
    ax.text(0.05, 0.05, f'Mean train-val gap: {gap:.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_dir / f'learning_curve_{model_name.lower().replace(" ","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances, names, model_name, output_dir, top_n=30):
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), importances[idx], color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(names[idx], rotation=90, fontsize=7)
    ax.set_ylabel('Mean Importance (across LOPO folds)', fontsize=11)
    ax.set_title(f'{model_name} — Top {top_n} Feature Importances', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_importance_{model_name.lower().replace(" ","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, model_name, output_dir):
    """Bar chart of AUC, F1, Sensitivity, Specificity per patient fold."""
    patients = [m['patient'] for m in fold_metrics]
    metrics  = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange']

    x = np.arange(len(patients))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (met, col) in enumerate(zip(metrics, colors)):
        vals = [m[met] for m in fold_metrics]
        ax.bar(x + i * width, vals, width, label=met.upper(), color=col, alpha=0.8, edgecolor='black')

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'{model_name} — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'per_fold_{model_name.lower().replace(" ","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion(all_y_true, all_y_pred, model_name, output_dir):
    """Aggregate confusion matrix across all LOPO folds."""
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',    'Counts'),
        (axes[1], cm_norm, '.2f',  'Normalized'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=['Pre-ictal', 'Ictal'],
                    yticklabels=['Pre-ictal', 'Ictal'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'{model_name} — Aggregate CM ({title})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_aggregate_{model_name.lower().replace(" ","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────
# LOPO evaluation for one model
# ─────────────────────────────────────────────────────────────

def run_lopo(model_name, model, X, y, patient_ids, subject_ids,
             feature_names, top_n, output_dir):

    patients = np.unique(patient_ids)
    fold_metrics     = []
    fold_roc_data    = []
    all_importances  = []
    all_sel_names    = []
    all_y_true       = []
    all_y_pred       = []

    print(f'\n{"="*60}')
    print(f'  {model_name} — LOPO Cross-Validation ({len(patients)} folds)')
    print(f'{"="*60}')

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: test set has only one class')
            continue

        # Feature selection (fit on train only — no leakage)
        X_tr_sel, X_te_sel, sel_names, importances = select_features(
            X_train, y_train, X_test, feature_names, top_n=top_n
        )
        all_importances.append(importances)
        all_sel_names.append(sel_names)

        # Scale (fit on train only)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_sel)
        X_te_sc = scaler.transform(X_te_sel)

        # Fit
        model.fit(X_tr_sc, y_train)

        # Predict
        y_pred = model.predict(X_te_sc)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_te_sc)[:, 1]
        else:
            y_prob = model.decision_function(X_te_sc)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['patient'] = pat
        metrics['n_train'] = int(train_mask.sum())
        metrics['n_test']  = int(test_mask.sum())
        metrics['n_features_selected'] = int(X_tr_sel.shape[1])
        fold_metrics.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        # ROC per fold
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        # Per-fold confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, model_name, pat, output_dir)

        print(f'  {pat:8s} | AUC={metrics["auc"]:.3f}  F1={metrics["f1"]:.3f}'
              f'  Sens={metrics["sensitivity"]:.3f}  Spec={metrics["specificity"]:.3f}'
              f'  MCC={metrics["mcc"]:.3f}')

    # ── Aggregate plots ──────────────────────────────────────
    plot_roc_all_folds(fold_roc_data, model_name, output_dir)
    plot_per_fold_metrics(fold_metrics, model_name, output_dir)
    plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred), model_name, output_dir)

    # Mean feature importance across folds (align by position — same top_n selection)
    mean_imp  = np.mean(all_importances, axis=0)
    # Use names from first fold as representative
    rep_names = all_sel_names[0]
    plot_feature_importance(mean_imp, rep_names, model_name, output_dir)

    # ── Summary statistics ──────────────────────────────────
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'precision', 'mcc']
    print(f'\n  {"─"*50}')
    print(f'  {model_name} — Mean ± Std across {len(fold_metrics)} folds')
    print(f'  {"─"*50}')
    summary_stats = {}
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = np.mean(vals), np.std(vals)
        summary_stats[k] = {'mean': float(mean_), 'std': float(std_)}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    return fold_metrics, summary_stats, fold_roc_data


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 4 - Baseline ML with LOPO CV')
    parser.add_argument('--featfile',    required=True, help='features/features_all.npz')
    parser.add_argument('--outputdir',   default='results/baseline_ml')
    parser.add_argument('--topfeatures', type=int, default=100,
                        help='Top-N features to keep after RF selection (default 100)')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────
    print('=' * 60)
    print('STEP 4 — BASELINE ML')
    print('=' * 60)
    data = np.load(args.featfile, allow_pickle=True)
    X            = data['X'].astype(np.float32)
    y            = data['y'].astype(np.int64)
    patient_ids  = data['patient_ids']
    subject_ids  = data['subject_ids']
    feature_names = data['feature_names'].tolist()

    print(f'Loaded: X={X.shape}, y={y.shape}')
    print(f'Ictal: {(y==1).sum()}  Pre-ictal: {(y==0).sum()}')
    print(f'Patients: {np.unique(patient_ids).tolist()}')
    print(f'Top features to select: {args.topfeatures}')

    # Replace any NaN/Inf with 0 (safety)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Define models ────────────────────────────────────────
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=8,          # shallow = less overfit
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        'SVM RBF': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
        ),
    }

    all_results = {}

    for model_name, model in models.items():
        fold_metrics, summary_stats, fold_roc_data = run_lopo(
            model_name, model,
            X, y, patient_ids, subject_ids,
            feature_names, args.topfeatures, output_dir
        )
        all_results[model_name] = {
            'fold_metrics':  fold_metrics,
            'summary_stats': summary_stats,
        }

        # Learning curve (overfit diagnostic) — fit on full dataset for visualization
        print(f'\n  Generating learning curve for {model_name}...')
        plot_learning_curve(model, X[:, :args.topfeatures], y, model_name, output_dir)

    # ── Save all results to JSON ─────────────────────────────
    results_path = output_dir / 'results_all.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nAll results saved to {results_path}')

    # ── Final comparison table ───────────────────────────────
    print('\n' + '=' * 60)
    print('FINAL COMPARISON')
    print('=' * 60)
    rows = []
    for model_name, res in all_results.items():
        row = {'Model': model_name}
        for k, v in res['summary_stats'].items():
            row[k] = f"{v['mean']:.3f} ± {v['std']:.3f}"
        rows.append(row)
    df_summary = pd.DataFrame(rows).set_index('Model')
    print(df_summary.to_string())
    df_summary.to_csv(output_dir / 'summary_table.csv', encoding='utf-8')
    print(f'\nSummary table saved to {output_dir / "summary_table.csv"}')

    # ── Final comparison bar chart ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange']
    x = np.arange(len(met_keys))
    width = 0.35
    for i, (model_name, res) in enumerate(all_results.items()):
        means = [res['summary_stats'][k]['mean'] for k in met_keys]
        stds  = [res['summary_stats'][k]['std']  for k in met_keys]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=model_name,
               capsize=4, edgecolor='black', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_title('Baseline ML — Model Comparison (LOPO CV)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Model comparison chart saved.')

    print('\n' + '=' * 60)
    print('STEP 4 COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
