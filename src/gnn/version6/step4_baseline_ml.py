"""
Step 4 — Baseline ML: Random Forest + SVM  (LOPO Cross-Validation)
===================================================================
DESIGN DECISIONS (document in thesis):

1. EVALUATION: Leave-One-Patient-Out (LOPO) CV.
   Correct for clinical datasets — tests generalisation to an unseen patient.
   We split by PATIENT, so PAT_14's 8 recordings are either
   all in train or all in test.

2. FEATURE SELECTION: None beyond what step3 already decided.
   With only 53 curated features, we do NOT need a secondary selector.
   We use ALL 53 features, scaled per fold (StandardScaler fit on train only).
   No leakage: scaler is fit on training split and applied to test split.

3. IMBALANCE:
   - class_weight='balanced' in both RF and SVM.
   - Pre-processing already set ratio=2 (2 pre-ictal per ictal).
   - PRIMARY metrics: AUC, F1 (ictal class), Sensitivity, Specificity, MCC.
   - Accuracy IS reported (professor's requirement) but always shown
     alongside the majority-class baseline so it is not misleading.

4. HYPERPARAMETERS:
   RF : n_estimators=300, max_depth=6, min_samples_leaf=5
        → Deliberately shallow to prevent memorisation of 34-subject data.
   SVM: C=1.0, gamma='scale', RBF kernel
        → Conservative regularisation. No grid search — too few samples
          to tune without leaking val-set information.

5. OVERFITTING ANALYSIS:
   For each fold we compute both TRAIN and TEST scores.
   A large train-test AUC gap signals overfitting.
   Outputs: per-fold gap bar chart + summary overfitting figure.

Outputs:
  results_all.json              all fold metrics + summary stats
  summary_table.csv             mean ± std per model (all metrics)
  cm_{model}_{patient}.png      per-fold confusion matrices
  roc_{model}.png               LOPO ROC curves
  per_fold_{model}.png          per-patient metric bar chart
  cm_aggregate_{model}.png      aggregate confusion matrix
  feature_importance_{model}.png RF feature importances (averaged across folds)
  overfitting_{model}.png       train vs test AUC per fold
  overfitting_summary.png       all models side-by-side train vs test AUC
  model_comparison.png          final AUC/F1/Sens/Spec bar chart

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
import sklearn.base as skbase

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 1. METRIC HELPERS
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob):
    """
    Returns a dict of all classification metrics.

    Accuracy    — included per professor's request; always interpret
                  alongside majority-class baseline.
    Sensitivity — recall for ictal (positive) class.
    Specificity — recall for pre-ictal (negative) class.
    MCC         — balanced metric, reliable with imbalanced classes.
    AUC         — primary metric; threshold-independent.
    """
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_total        = len(y_true)
    majority_n     = max((y_true == 0).sum(), (y_true == 1).sum())

    return {
        'accuracy':          float(accuracy_score(y_true, y_pred)),
        'majority_baseline': float(majority_n / n_total),   # dummy classifier score
        'auc':               float(roc_auc_score(y_true, y_prob)),
        'f1':                float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity':       float(tp / (tp + fn + 1e-12)),
        'specificity':       float(tn / (tn + fp + 1e-12)),
        'precision':         float(precision_score(y_true, y_pred, zero_division=0)),
        'mcc':               float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ══════════════════════════════════════════════════════════════
# 2. PLOT HELPERS
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, model_name, patient_id, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Pre-ictal', 'Ictal'],
        yticklabels=['Pre-ictal', 'Ictal'],
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
        ax.plot(fpr, tpr, alpha=0.6, lw=1.5, label=f'{pat} (AUC={auc:.2f})')
        aucs.append(auc)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'{model_name} — LOPO ROC\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
        fontsize=11, fontweight='bold',
    )
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f'roc_{model_name.lower().replace(" ", "_")}.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close()


def plot_per_fold_metrics(fold_metrics, model_name, output_dir):
    """
    Per-patient bar chart for AUC, F1, Sensitivity, Specificity.
    Accuracy shown as a separate line overlay so it is contextualised
    by the majority-class baseline.
    """
    patients  = [m['patient'] for m in fold_metrics]
    met_keys  = ['auc', 'f1', 'sensitivity', 'specificity']
    colors    = ['steelblue', 'tomato', 'seagreen', 'darkorange']
    x         = np.arange(len(patients))
    width     = 0.2

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (met, col) in enumerate(zip(met_keys, colors)):
        vals = [m[met] for m in fold_metrics]
        ax.bar(x + i * width, vals, width, label=met.upper(),
               color=col, alpha=0.8, edgecolor='black')

    # Accuracy as scatter + line
    acc_vals  = [m['accuracy'] for m in fold_metrics]
    base_vals = [m['majority_baseline'] for m in fold_metrics]
    ax.plot(x + 1.5 * width, acc_vals,  'k^-', ms=7, lw=1.5, label='Accuracy')
    ax.plot(x + 1.5 * width, base_vals, 'r--', ms=5, lw=1,   label='Majority baseline')

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.axhline(0.5, color='gray', linestyle=':', lw=1)
    ax.set_title(f'{model_name} — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(
        output_dir / f'per_fold_{model_name.lower().replace(" ", "_")}.png',
        dpi=150, bbox_inches='tight',
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
        ax.set_title(f'{model_name} — Aggregate CM ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        output_dir / f'cm_aggregate_{model_name.lower().replace(" ", "_")}.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close()


def plot_feature_importance(importances_per_fold, feature_names,
                             model_name, output_dir, top_n=30):
    """
    Average RF importances across LOPO folds for a stable estimate.
    Fold-level std shown as error bars.
    """
    imp_arr  = np.array(importances_per_fold)   # (n_folds, n_features)
    mean_imp = imp_arr.mean(axis=0)
    std_imp  = imp_arr.std(axis=0)
    idx      = np.argsort(mean_imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(top_n), mean_imp[idx],
           yerr=std_imp[idx], capsize=3,
           color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(np.array(feature_names)[idx], rotation=90, fontsize=7)
    ax.set_ylabel('Mean importance ± std (LOPO folds)', fontsize=11)
    ax.set_title(f'{model_name} — Top {top_n} Feature Importances',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(
        output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close()


def plot_overfitting_per_model(fold_train_aucs, fold_test_aucs,
                                patients, model_name, output_dir):
    """
    Per-fold train vs test AUC bar chart for one model.
    A large gap = overfitting on that fold.
    """
    x     = np.arange(len(patients))
    width = 0.35
    gap   = [tr - te for tr, te in zip(fold_train_aucs, fold_test_aucs)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: grouped bars
    axes[0].bar(x - width / 2, fold_train_aucs, width,
                label='Train AUC', color='steelblue', alpha=0.85, edgecolor='black')
    axes[0].bar(x + width / 2, fold_test_aucs,  width,
                label='Test AUC',  color='tomato',    alpha=0.85, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(patients, rotation=30, fontsize=9)
    axes[0].set_ylabel('AUC', fontsize=12)
    axes[0].set_ylim(0, 1.15)
    axes[0].axhline(0.5, color='gray', linestyle='--', lw=1)
    axes[0].set_title(f'{model_name} — Train vs Test AUC (LOPO)',
                      fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: gap bar
    colors = ['tomato' if g > 0.10 else 'steelblue' for g in gap]
    axes[1].bar(x, gap, color=colors, edgecolor='black', alpha=0.85)
    axes[1].axhline(0.10, color='red', linestyle='--', lw=1.5,
                    label='Gap = 0.10 (overfitting warning)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(patients, rotation=30, fontsize=9)
    axes[1].set_ylabel('Train AUC − Test AUC', fontsize=12)
    axes[1].set_title(f'{model_name} — Overfitting Gap per Fold',
                      fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    mean_gap = np.mean(gap)
    fig.suptitle(
        f'{model_name} | Mean gap = {mean_gap:.3f} '
        f'{"⚠ Overfitting" if mean_gap > 0.10 else "✓ OK"}',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f'overfitting_{model_name.lower().replace(" ", "_")}.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close()


def plot_overfitting_summary(all_model_train_aucs, all_model_test_aucs,
                              model_names, output_dir):
    """
    Summary bar chart comparing train vs test AUC across ALL models.
    """
    x     = np.arange(len(model_names))
    width = 0.35

    train_means = [np.mean(all_model_train_aucs[m]) for m in model_names]
    test_means  = [np.mean(all_model_test_aucs[m])  for m in model_names]
    train_stds  = [np.std(all_model_train_aucs[m])  for m in model_names]
    test_stds   = [np.std(all_model_test_aucs[m])   for m in model_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, train_means, width, yerr=train_stds, capsize=4,
           label='Train AUC', color='steelblue', alpha=0.85, edgecolor='black')
    ax.bar(x + width / 2, test_means,  width, yerr=test_stds,  capsize=4,
           label='Test AUC',  color='tomato',    alpha=0.85, edgecolor='black')

    # Annotate gaps
    for i, (tr, te) in enumerate(zip(train_means, test_means)):
        ax.text(i, max(tr, te) + 0.03, f'Δ={tr - te:.2f}',
                ha='center', fontsize=10, fontweight='bold',
                color='tomato' if tr - te > 0.10 else 'black')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel('Mean AUC ± Std (LOPO)', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('Overfitting Analysis — Train vs Test AUC (All Models)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ overfitting_summary.png')


# ══════════════════════════════════════════════════════════════
# 3. LOPO EVALUATION
# ══════════════════════════════════════════════════════════════

def run_lopo(model_name, model, X, y, patient_ids, feature_names, output_dir):
    """
    Leave-One-Patient-Out CV for one model.

    Key guarantees:
      - StandardScaler fit on TRAIN split only (no leakage).
      - Fresh model clone per fold (no state bleed between folds).
      - Both train AND test scores recorded for overfitting analysis.

    Returns:
      fold_metrics  : list of per-fold metric dicts
      summary_stats : mean ± std across folds
      train_aucs    : list of per-fold training AUC (for overfitting plot)
      test_aucs     : list of per-fold test AUC
      fold_patients : list of patient IDs in fold order
    """
    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []
    importances   = []
    train_aucs    = []
    test_aucs     = []
    fold_patients = []

    print(f'\n{"=" * 65}')
    print(f'  {model_name} — LOPO CV ({len(patients)} patient folds)')
    print(f'{"=" * 65}')
    print(f'  {"Patient":10s} | {"AUC":>6} {"F1":>6} {"Sens":>6} '
          f'{"Spec":>6} {"Acc":>6} {"MCC":>6} | {"TrainAUC":>9} {"Gap":>6}')
    print(f'  {"-" * 65}')

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # Skip folds where the test set has only one class
        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: test set has only one class')
            continue

        # ── Scale: fit on train ONLY ─────────────────────────────
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        # ── Fit fresh clone per fold ─────────────────────────────
        clf = skbase.clone(model)
        clf.fit(X_tr_sc, y_train)

        # ── Test predictions ─────────────────────────────────────
        y_pred = clf.predict(X_te_sc)
        if hasattr(clf, 'predict_proba'):
            y_prob_test = clf.predict_proba(X_te_sc)[:, 1]
        else:
            raw = clf.decision_function(X_te_sc)
            y_prob_test = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

        # ── Train predictions (for overfitting analysis) ─────────
        if hasattr(clf, 'predict_proba'):
            y_prob_train = clf.predict_proba(X_tr_sc)[:, 1]
        else:
            raw_tr = clf.decision_function(X_tr_sc)
            y_prob_train = (raw_tr - raw_tr.min()) / (raw_tr.max() - raw_tr.min() + 1e-12)

        train_auc = float(roc_auc_score(y_train, y_prob_train))
        test_auc  = float(roc_auc_score(y_test,  y_prob_test))
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
        fold_patients.append(pat)

        # ── Metrics ──────────────────────────────────────────────
        metrics            = compute_metrics(y_test, y_pred, y_prob_test)
        metrics['patient'] = pat
        metrics['n_train'] = int(train_mask.sum())
        metrics['n_test']  = int(test_mask.sum())
        metrics['train_auc'] = train_auc
        metrics['overfit_gap'] = round(train_auc - test_auc, 4)
        fold_metrics.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        fold_roc_data.append((fpr, tpr, test_auc, pat))

        plot_confusion_matrix(
            confusion_matrix(y_test, y_pred), model_name, pat, output_dir
        )

        if hasattr(clf, 'feature_importances_'):
            importances.append(clf.feature_importances_)

        gap_flag = ' ⚠' if train_auc - test_auc > 0.10 else ''
        print(f'  {pat:10s} | {test_auc:6.3f} {metrics["f1"]:6.3f} '
              f'{metrics["sensitivity"]:6.3f} {metrics["specificity"]:6.3f} '
              f'{metrics["accuracy"]:6.3f} {metrics["mcc"]:6.3f} | '
              f'{train_auc:9.3f} {train_auc - test_auc:6.3f}{gap_flag}')

    if len(fold_metrics) == 0:
        print(f'  [ERROR] No valid folds for {model_name}')
        return [], {}, [], [], []

    # ── Aggregate plots ───────────────────────────────────────────
    plot_roc_all_folds(fold_roc_data, model_name, output_dir)
    plot_per_fold_metrics(fold_metrics, model_name, output_dir)
    plot_aggregate_confusion(
        np.array(all_y_true), np.array(all_y_pred), model_name, output_dir
    )
    plot_overfitting_per_model(
        train_aucs, test_aucs, fold_patients, model_name, output_dir
    )

    if importances:
        plot_feature_importance(importances, feature_names, model_name, output_dir)

    # ── Summary stats ─────────────────────────────────────────────
    met_keys = [
        'accuracy', 'majority_baseline',
        'auc', 'f1', 'sensitivity', 'specificity', 'precision', 'mcc',
        'train_auc', 'overfit_gap',
    ]
    summary_stats = {}
    print(f'\n  {"─" * 55}')
    print(f'  {model_name} — Mean ± Std across {len(fold_metrics)} folds')
    print(f'  {"─" * 55}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        extra = ''
        if k == 'accuracy':
            extra = f'   ← dummy baseline ≈ {np.mean([m["majority_baseline"] for m in fold_metrics]):.3f}'
        if k == 'overfit_gap':
            extra = '  ← train AUC − test AUC (>0.10 = overfitting)'
        print(f'  {k:20s}: {mean_:.3f} ± {std_:.3f}{extra}')

    return fold_metrics, summary_stats, train_aucs, test_aucs, fold_patients


# ══════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 4 — Baseline ML with LOPO CV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--featfile', required=True,
        help='Path to features/features_all.npz from step 3',
    )
    parser.add_argument(
        '--outputdir', default='results/baseline_ml',
        help='Output directory for results and plots',
    )
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────
    print('=' * 65)
    print('STEP 4 — BASELINE ML (Random Forest + SVM, LOPO CV)')
    print('=' * 65)

    data          = np.load(args.featfile, allow_pickle=True)
    X             = data['X'].astype(np.float32)
    y             = data['y'].astype(np.int64)
    patient_ids   = data['patient_ids']
    feature_names = data['feature_names'].tolist()

    # Replace any remaining NaN/Inf (should not happen after step 3 fix)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_ictal    = int((y == 1).sum())
    n_preictal = int((y == 0).sum())
    majority_b = max(n_ictal, n_preictal) / len(y)

    print(f'Loaded   : X={X.shape}  y={y.shape}')
    print(f'Ictal    : {n_ictal}   Pre-ictal: {n_preictal}')
    print(f'Patients : {np.unique(patient_ids).tolist()}')
    print(f'Features : {X.shape[1]}')
    print(f'\nNOTE: Majority-class accuracy baseline = {majority_b * 100:.1f}%')
    print(f'      A model predicting all pre-ictal would score {majority_b * 100:.1f}% accuracy.')
    print(f'      Always interpret accuracy relative to this baseline.\n')

    # ── Model definitions ─────────────────────────────────────────
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=6,           # shallow → less overfitting
            min_samples_leaf=5,    # need ≥5 samples per leaf
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

    all_results        = {}
    all_model_tr_aucs  = {}
    all_model_te_aucs  = {}

    for model_name, model in models.items():
        fold_metrics, summary_stats, tr_aucs, te_aucs, fold_pats = run_lopo(
            model_name, model,
            X, y, patient_ids, feature_names, output_dir,
        )
        all_results[model_name] = {
            'fold_metrics':  fold_metrics,
            'summary_stats': summary_stats,
        }
        all_model_tr_aucs[model_name] = tr_aucs
        all_model_te_aucs[model_name] = te_aucs

    # ── Cross-model overfitting summary ───────────────────────────
    if len(all_model_tr_aucs) > 0:
        plot_overfitting_summary(
            all_model_tr_aucs, all_model_te_aucs,
            list(models.keys()), output_dir,
        )

    # ── Save JSON ─────────────────────────────────────────────────
    results_path = output_dir / 'results_all.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n✓ All results → {results_path}')

    # ── Summary table ─────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('FINAL SUMMARY TABLE')
    print('=' * 65)

    # Primary metrics for display
    display_keys = ['accuracy', 'auc', 'f1', 'sensitivity', 'specificity', 'mcc', 'overfit_gap']
    rows = []
    for model_name, res in all_results.items():
        if not res['summary_stats']:
            continue
        row = {'Model': model_name}
        for k in display_keys:
            if k in res['summary_stats']:
                v = res['summary_stats'][k]
                row[k] = f"{v['mean']:.3f} ± {v['std']:.3f}"
        rows.append(row)

    if rows:
        df_summary = pd.DataFrame(rows).set_index('Model')
        print(df_summary.to_string())
        df_summary.to_csv(output_dir / 'summary_table.csv', encoding='utf-8')
        print(f'\n✓ Summary table → {output_dir / "summary_table.csv"}')

    print(f'\nNOTE: Majority-class accuracy baseline = {majority_b * 100:.1f}%')
    print(f'      Accuracy above this threshold represents genuine learning.')

    # ── Final model comparison bar chart ──────────────────────────
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
    ax.set_ylabel('Mean Score ± Std (LOPO)', fontsize=12)
    ax.set_ylim(0, 1.30)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('Baseline ML — Model Comparison (LOPO CV)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Model comparison → {output_dir / "model_comparison.png"}')

    print('\n' + '=' * 65)
    print('STEP 4 COMPLETE')
    print('=' * 65)
    print('\nNext: python step5_gnn_supervised.py --featfile features/features_all.npz'
          ' --baseline_json results/baseline_ml/results_all.json')


if __name__ == '__main__':
    main()