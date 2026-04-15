"""
Step 4 — Baseline ML: Random Forest + SVM + SVM-RFE  (LOPO + Inner CV)
=======================================================================
DESIGN DECISIONS (document in thesis):

1. EVALUATION: Nested Leave-One-Patient-Out (LOPO) CV.
   ┌─ Outer loop (8 folds) ─────────────────────────────────────────┐
   │  Leave 1 patient out → test fold                               │
   │  ┌─ Inner loop (7 folds per outer fold) ───────────────────────┐│
   │  │  Of the 7 remaining training patients, rotate 1 as val     ││
   │  │  → 7 inner fits, average val AUC                           ││
   │  │  → This val score is reported alongside test score         ││
   │  └─────────────────────────────────────────────────────────────┘│
   └────────────────────────────────────────────────────────────────┘
   The inner CV gives an unbiased estimate of how well the model
   generalises to an unseen patient during training — it is reported
   for completeness (not for hyperparameter tuning, since we use fixed
   hyperparameters justified in the thesis).

2. MODELS:
   (a) Random Forest — shallow (max_depth=6) to prevent memorisation
   (b) SVM RBF       — moderate regularisation (C=1.0)
   (c) SVM-RFE       — SVM with linear kernel + Recursive Feature
       Elimination (10 features retained). Neuroscientific motivation:
       RFE with a linear SVM is well-established in EEG/BCI literature
       (e.g. Guyon et al. 2002). The selected features identify which
       spectral/Hjorth/connectivity features are most discriminative.
       RFE selector is fit on training fold ONLY — no leakage.

3. METRICS (added accuracy per professor's request):
   Primary: AUC, F1, Sensitivity, Specificity, MCC, Accuracy
   Accuracy is reported last and interpreted alongside class balance
   (misleading in isolation with imbalanced data).

4. FEATURE SELECTION (SVM-RFE only):
   n_features_to_select=10 chosen as a reasonable compression of 53
   features. Can be adjusted via --rfe_n_features argument.
   For RF and SVM, all 53 curated features are used.

5. IMBALANCE:
   class_weight='balanced' in all models.
   Pre-processing already provides ratio=2 (2 pre-ictal per ictal).

Outputs:
  results_all.json            all fold metrics + summary stats
  summary_table.csv           mean ± std per model
  cm_{model}_{patient}.png    per-fold confusion matrices
  roc_{model}.png             LOPO ROC curves
  per_fold_{model}.png        per-patient bar chart
  model_comparison.png        final side-by-side comparison
  rfe_selected_features.csv   features selected by SVM-RFE per fold
  feature_importance_*.png    RF feature importance

Usage:
  python step4_baseline_ml.py \\
      --featfile  features/features_all.npz \\
      --outputdir results/baseline_ml \\
      --rfe_n_features 10
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
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
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
    Returns a dict of all classification metrics including accuracy.
    Sensitivity = recall for the ictal (positive) class.
    Specificity = recall for the pre-ictal (negative) class.
    MCC is balanced and reliable for imbalanced datasets.
    Accuracy is included per professor's request; interpret alongside
    class balance since it can be misleading when classes are unequal.
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
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'mcc':         float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ─────────────────────────────────────────────────────────────
# Inner CV — validation fold loop
# ─────────────────────────────────────────────────────────────

def run_inner_cv(model_fn, X_train, y_train, train_patient_ids, feature_names,
                 rfe_n_features=None):
    """
    Leave-one-patient-out over the TRAINING patients only (inner CV).
    Used to estimate generalisation to an unseen patient during training,
    without ever touching the outer test patient.

    model_fn : callable that returns a fresh unfitted sklearn estimator
    rfe_n_features : if not None, wrap model in RFE with this many features
    Returns (mean_val_auc, mean_val_accuracy, rfe_support or None)
    """
    inner_patients = np.unique(train_patient_ids)
    inner_aucs     = []
    inner_accs     = []
    rfe_supports   = []

    for val_pat in inner_patients:
        inner_val_mask   = (train_patient_ids == val_pat)
        inner_train_mask = ~inner_val_mask

        Xi_tr, yi_tr = X_train[inner_train_mask], y_train[inner_train_mask]
        Xi_va, yi_va = X_train[inner_val_mask],   y_train[inner_val_mask]

        if len(np.unique(yi_va)) < 2:
            continue

        scaler  = StandardScaler()
        Xi_tr_s = scaler.fit_transform(Xi_tr)
        Xi_va_s = scaler.transform(Xi_va)

        clf = model_fn()

        if rfe_n_features is not None:
            # Linear SVM for RFE (non-linear kernels don't have coef_)
            base_estimator = SVC(
                kernel='linear', C=0.1,
                class_weight='balanced', max_iter=2000
            )
            selector = RFE(base_estimator, n_features_to_select=rfe_n_features, step=1)
            selector.fit(Xi_tr_s, yi_tr)
            Xi_tr_sel = selector.transform(Xi_tr_s)
            Xi_va_sel = selector.transform(Xi_va_s)
            clf.fit(Xi_tr_sel, yi_tr)
            rfe_supports.append(selector.support_)
            y_prob = clf.predict_proba(Xi_va_sel)[:, 1] if hasattr(clf, 'predict_proba') \
                     else _decision_to_prob(clf.decision_function(Xi_va_sel))
            y_pred = clf.predict(Xi_va_sel)
        else:
            clf.fit(Xi_tr_s, yi_tr)
            y_prob = clf.predict_proba(Xi_va_s)[:, 1] if hasattr(clf, 'predict_proba') \
                     else _decision_to_prob(clf.decision_function(Xi_va_s))
            y_pred = clf.predict(Xi_va_s)

        try:
            inner_aucs.append(float(roc_auc_score(yi_va, y_prob)))
        except Exception:
            pass
        inner_accs.append(float(accuracy_score(yi_va, y_pred)))

    mean_auc = float(np.mean(inner_aucs)) if inner_aucs else float('nan')
    mean_acc = float(np.mean(inner_accs)) if inner_accs else float('nan')

    # Majority vote on RFE support across inner folds
    rfe_support_consensus = None
    if rfe_supports:
        rfe_support_consensus = np.stack(rfe_supports).mean(axis=0) >= 0.5

    return mean_auc, mean_acc, rfe_support_consensus


def _decision_to_prob(raw):
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)


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
    fname = output_dir / f'cm_{model_name.lower().replace(" ", "_")}_{patient_id}.png'
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
        output_dir / f'roc_{model_name.lower().replace(" ", "_")}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_per_fold_metrics(fold_metrics, model_name, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    x        = np.arange(len(patients))
    width    = 0.16
    fig, ax  = plt.subplots(figsize=(14, 5))
    for i, (met, col) in enumerate(zip(met_keys, colors)):
        vals = [m[met] for m in fold_metrics]
        ax.bar(x + i * width, vals, width, label=met.upper(),
               color=col, alpha=0.8, edgecolor='black')
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title(f'{model_name} — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(
        output_dir / f'per_fold_{model_name.lower().replace(" ", "_")}.png',
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
        output_dir / f'cm_aggregate_{model_name.lower().replace(" ", "_")}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_feature_importance(importances_per_fold, feature_names, model_name, output_dir, top_n=30):
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
        output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()


def plot_rfe_feature_frequency(rfe_support_per_fold, feature_names, output_dir):
    """
    Bar chart showing how often each feature was selected by SVM-RFE
    across LOPO outer folds. Stable features (selected in all/most folds)
    are the most neuroscientifically meaningful.
    """
    freq = np.stack(rfe_support_per_fold).mean(axis=0)
    idx  = np.argsort(freq)[::-1]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(feature_names)), freq[idx],
           color='darkorange', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(np.array(feature_names)[idx], rotation=90, fontsize=7)
    ax.set_ylabel('Fraction of outer folds selected', fontsize=11)
    ax.set_title('SVM-RFE — Feature Selection Stability (across LOPO outer folds)',
                 fontsize=12, fontweight='bold')
    ax.axhline(0.5, color='red', linestyle='--', lw=1, label='50% threshold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'rfe_feature_stability.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_inner_vs_outer_auc(fold_metrics, model_name, output_dir):
    """
    Scatter plot: inner CV val AUC vs outer test AUC.
    Points near the diagonal = well-calibrated generalisation.
    Points above the diagonal = inner CV over-estimates test performance.
    Useful for thesis discussion.
    """
    inner = [m.get('inner_val_auc', float('nan')) for m in fold_metrics]
    outer = [m['auc'] for m in fold_metrics]
    valid = [(i, o) for i, o in zip(inner, outer) if not np.isnan(i)]
    if not valid:
        return
    xi, xo = zip(*valid)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xi, xo, s=80, zorder=5, color='steelblue', edgecolor='black')
    lims = [min(min(xi), min(xo)) - 0.05, max(max(xi), max(xo)) + 0.05]
    ax.plot(lims, lims, 'k--', lw=1, label='Diagonal (inner=outer)')
    for m, xi_, xo_ in zip(fold_metrics, xi, xo):
        ax.annotate(m['patient'], (xi_, xo_), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Inner CV val AUC', fontsize=12)
    ax.set_ylabel('Outer test AUC', fontsize=12)
    ax.set_title(f'{model_name} — Inner vs Outer AUC', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f'inner_vs_outer_{model_name.lower().replace(" ", "_")}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()


# ─────────────────────────────────────────────────────────────
# LOPO evaluation for one model
# ─────────────────────────────────────────────────────────────

def run_lopo(model_name, model_fn, X, y, patient_ids, feature_names, output_dir,
             rfe_n_features=None):
    """
    Outer LOPO CV with inner LOPO CV for validation.

    model_fn       : callable that returns a fresh sklearn estimator
    rfe_n_features : if not None, apply RFE with this many features
    Returns fold_metrics list and summary_stats dict.
    """
    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []
    importances   = []
    rfe_supports  = []

    print(f'\n{"=" * 65}')
    print(f'  {model_name} — Nested LOPO CV ({len(patients)} outer folds)')
    if rfe_n_features:
        print(f'  RFE: retaining top {rfe_n_features} features per fold (fit on train only)')
    print(f'{"=" * 65}')

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask],  y[train_mask]
        X_test,  y_test  = X[test_mask],   y[test_mask]
        train_pats       = patient_ids[train_mask]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: test set has only one class')
            continue

        # ── Inner CV ──────────────────────────────────────────────
        inner_val_auc, inner_val_acc, rfe_support_inner = run_inner_cv(
            model_fn, X_train, y_train, train_pats, feature_names,
            rfe_n_features=rfe_n_features
        )

        # ── Outer fold: scale on train only ───────────────────────
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_train)
        X_te_sc  = scaler.transform(X_test)

        clf = model_fn()

        if rfe_n_features is not None:
            # Use RFE consensus from inner CV; if unavailable, re-fit RFE on full train
            base_estimator = SVC(
                kernel='linear', C=0.1,
                class_weight='balanced', max_iter=2000
            )
            selector = RFE(base_estimator, n_features_to_select=rfe_n_features, step=1)
            selector.fit(X_tr_sc, y_train)
            X_tr_sel = selector.transform(X_tr_sc)
            X_te_sel = selector.transform(X_te_sc)
            rfe_supports.append(selector.support_)
            # The actual classifier (not the linear SVM used for RFE selection)
            clf.fit(X_tr_sel, y_train)
            y_pred = clf.predict(X_te_sel)
            y_prob = clf.predict_proba(X_te_sel)[:, 1] if hasattr(clf, 'predict_proba') \
                     else _decision_to_prob(clf.decision_function(X_te_sel))
        else:
            clf.fit(X_tr_sc, y_train)
            y_pred = clf.predict(X_te_sc)
            y_prob = clf.predict_proba(X_te_sc)[:, 1] if hasattr(clf, 'predict_proba') \
                     else _decision_to_prob(clf.decision_function(X_te_sc))

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['patient']       = pat
        metrics['n_train']       = int(train_mask.sum())
        metrics['n_test']        = int(test_mask.sum())
        metrics['inner_val_auc'] = inner_val_auc
        metrics['inner_val_acc'] = inner_val_acc
        fold_metrics.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        plot_confusion_matrix(confusion_matrix(y_test, y_pred), model_name, pat, output_dir)

        if hasattr(clf, 'feature_importances_'):
            importances.append(clf.feature_importances_)

        print(f'  {pat:8s} | Test  AUC={metrics["auc"]:.3f}  F1={metrics["f1"]:.3f}'
              f'  Acc={metrics["accuracy"]:.3f}'
              f'  Sens={metrics["sensitivity"]:.3f}  Spec={metrics["specificity"]:.3f}'
              f'  MCC={metrics["mcc"]:.3f}')
        print(f'           | Inner AUC={inner_val_auc:.3f}  Acc={inner_val_acc:.3f}')

    if not fold_metrics:
        return [], {}

    # ── Aggregate plots ────────────────────────────────────────
    plot_roc_all_folds(fold_roc_data, model_name, output_dir)
    plot_per_fold_metrics(fold_metrics, model_name, output_dir)
    plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred), model_name, output_dir)
    plot_inner_vs_outer_auc(fold_metrics, model_name, output_dir)

    if importances:
        plot_feature_importance(importances, feature_names, model_name, output_dir)

    if rfe_supports:
        plot_rfe_feature_frequency(rfe_supports, feature_names, output_dir)

    # ── Summary stats ──────────────────────────────────────────
    met_keys      = ['auc', 'f1', 'sensitivity', 'specificity', 'precision', 'accuracy', 'mcc']
    summary_stats = {}
    print(f'\n  {"─" * 55}')
    print(f'  {model_name} — Mean ± Std across {len(fold_metrics)} outer folds')
    print(f'  {"─" * 55}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    # Inner CV summary
    inner_aucs = [m['inner_val_auc'] for m in fold_metrics if not np.isnan(m.get('inner_val_auc', float('nan')))]
    if inner_aucs:
        print(f'\n  Inner CV val AUC (mean): {np.mean(inner_aucs):.3f} ± {np.std(inner_aucs):.3f}')
        summary_stats['inner_val_auc'] = {'mean': float(np.mean(inner_aucs)), 'std': float(np.std(inner_aucs))}

    return fold_metrics, summary_stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 4 — Baseline ML (Nested LOPO CV)')
    parser.add_argument('--featfile',       required=True, help='features/features_all.npz')
    parser.add_argument('--outputdir',      default='results/baseline_ml')
    parser.add_argument('--rfe_n_features', type=int, default=10,
                        help='Number of features to retain with SVM-RFE (default 10)')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('STEP 4 — BASELINE ML (Nested LOPO CV)')
    print('=' * 65)
    data = np.load(args.featfile, allow_pickle=True)
    X             = data['X'].astype(np.float32)
    y             = data['y'].astype(np.int64)
    patient_ids   = data['patient_ids']
    feature_names = data['feature_names'].tolist()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f'Loaded  : X={X.shape}  y={y.shape}')
    print(f'Ictal   : {(y == 1).sum()}  |  Pre-ictal: {(y == 0).sum()}')
    print(f'Patients: {np.unique(patient_ids).tolist()}')
    print(f'Features: {X.shape[1]}')
    print(f'RFE n_features: {args.rfe_n_features}')
    print()
    print('Nested LOPO protocol:')
    print('  Outer: leave 1 patient out (test)')
    print('  Inner: of remaining 7 patients, rotate 1 as val (7 inner folds)')
    print('  No hyperparameter tuning — inner CV is for reporting val score only')

    # ── Model definitions ──────────────────────────────────────
    # Use callables (lambdas) so each fold gets a fresh estimator.

    def make_rf():
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )

    def make_svm():
        return SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=42,
        )

    def make_svm_rfe():
        # The classifier used AFTER RFE selection — RBF SVM on the reduced feature set.
        # RFE itself uses a linear SVM internally (required for coef_-based ranking).
        return SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=42,
        )

    # (model_name, model_fn, rfe_n_features or None)
    model_configs = [
        ('Random Forest', make_rf,      None),
        ('SVM RBF',       make_svm,     None),
        ('SVM RFE',       make_svm_rfe, args.rfe_n_features),
    ]

    all_results = {}
    for model_name, model_fn, rfe_n in model_configs:
        fold_metrics, summary_stats = run_lopo(
            model_name, model_fn,
            X, y, patient_ids, feature_names, output_dir,
            rfe_n_features=rfe_n,
        )
        all_results[model_name] = {
            'fold_metrics':  fold_metrics,
            'summary_stats': summary_stats,
        }

    # ── Save JSON ───────────────────────────────────────────────
    results_path = output_dir / 'results_all.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nAll results → {results_path}')

    # ── Final comparison table ──────────────────────────────────
    print('\n' + '=' * 65)
    print('FINAL COMPARISON')
    print('=' * 65)
    rows = []
    for model_name, res in all_results.items():
        if not res['summary_stats']:
            continue
        row = {'Model': model_name}
        for k in ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy', 'mcc']:
            v = res['summary_stats'].get(k, {})
            row[k] = f"{v.get('mean', float('nan')):.3f} ± {v.get('std', float('nan')):.3f}"
        rows.append(row)

    if rows:
        df_summary = pd.DataFrame(rows).set_index('Model')
        print(df_summary.to_string())
        df_summary.to_csv(output_dir / 'summary_table.csv', encoding='utf-8')
        print(f'\nSummary table → {output_dir / "summary_table.csv"}')

    # ── Side-by-side comparison chart ──────────────────────────
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    colors   = ['steelblue', 'tomato', 'darkorange']
    x        = np.arange(len(met_keys))
    width    = 0.28

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (model_name, res) in enumerate(all_results.items()):
        if not res['summary_stats']:
            continue
        means  = [res['summary_stats'].get(k, {}).get('mean', 0) for k in met_keys]
        stds   = [res['summary_stats'].get(k, {}).get('std', 0)  for k in met_keys]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=model_name, color=colors[i % len(colors)],
               capsize=4, edgecolor='black', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('Baseline ML — Model Comparison (Nested LOPO CV)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Model comparison chart → {output_dir / "model_comparison.png"}')

    print('\n' + '=' * 65)
    print('STEP 4 COMPLETE')
    print('=' * 65)
    print('\nNext: python step5_gnn_supervised.py --featfile features/features_all.npz'
          ' --baseline_json results/baseline_ml/results_all.json')


if __name__ == '__main__':
    main()
