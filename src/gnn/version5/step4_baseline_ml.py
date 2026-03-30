"""
Step 4 — Baseline ML: Random Forest + SVM + SVM-RFE  (Nested LOPO CV)
=======================================================================
DESIGN DECISIONS (document in thesis):

1. EVALUATION: Nested Leave-One-Patient-Out (LOPO) CV.
   Outer loop: test on one patient, train on remaining 7.
   Inner loop: within the 7 training patients, do another LOPO
   to select the best hyperparameters — no leakage into the test patient.

2. MODELS:
   - Random Forest  : nested CV over max_depth ∈ {4, 6, 8}
   - SVM RBF        : nested CV over C ∈ {0.1, 1.0, 10.0}
   - SVM RFE        : RFE with LinearSVC to select top-20 features,
                      then RBF-SVM on selected subset (fixed config)

3. IMBALANCE:
   - class_weight='balanced' in all models.
   - Primary metrics: AUC, F1, Sensitivity, MCC.
   - Accuracy also reported (but noted as misleading for imbalanced data).

4. SCALING:
   StandardScaler fit on training fold only — no leakage.

5. OVERFITTING DIAGNOSTICS:
   - Inner CV AUC vs Outer Test AUC scatter plot
   - Train AUC vs Test AUC per fold (direct overfit signal)
   - Calibration plot (reliability diagram)

Outputs:
  results_all.json
  summary_table.csv
  cm_{model}_{patient}.png
  cm_aggregate_{model}.png
  roc_{model}.png
  per_fold_{model}.png
  feature_importance_{model}.png   (RF only)
  model_comparison.png
  overfit_diagnostic.png
  train_vs_test_auc.png
  calibration_{model}.png

Usage:
  python step4_baseline_ml.py \
      --featfile  features/features_all.npz \
      --outputdir results/baseline_ml
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# SVM-RFE wrapper
# ─────────────────────────────────────────────────────────────

class SVMWithRFE:
    """
    Two-stage model:
      1. RFE with LinearSVC ranks features by |weight|^2 and keeps top-n.
      2. RBF-SVM trained on the selected features.
    RFE is fit on training data only — no leakage.
    n_features_to_select=20 keeps ~38% of the 53 features.
    """
    def __init__(self, n_features=20, C=1.0):
        self.n_features = n_features
        self.C          = C
        self.rfe_       = None
        self.clf_       = None

    def fit(self, X, y):
        self.rfe_ = RFE(
            LinearSVC(C=self.C, class_weight='balanced',
                      max_iter=2000, random_state=42),
            n_features_to_select=self.n_features
        )
        self.rfe_.fit(X, y)
        X_sel = self.rfe_.transform(X)
        self.clf_ = SVC(
            kernel='rbf', C=self.C, gamma='scale',
            class_weight='balanced', probability=True, random_state=42
        )
        self.clf_.fit(X_sel, y)
        return self

    def predict(self, X):
        return self.clf_.predict(self.rfe_.transform(X))

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.rfe_.transform(X))

    def get_params(self, deep=True):
        return {'n_features': self.n_features, 'C': self.C}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        return f'SVMWithRFE(n_features={self.n_features}, C={self.C})'


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    """
    Returns a dict of all classification metrics.
    Accuracy is reported but noted as misleading for imbalanced data.
    Sensitivity = recall for ictal (positive) class.
    Specificity = recall for pre-ictal (negative) class.
    MCC is the most reliable single metric for imbalanced binary classification.
    """
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity    = tp / (tp + fn + 1e-12)
    specificity    = tn / (tn + fp + 1e-12)
    accuracy       = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {
        'auc':         float(roc_auc_score(y_true, y_prob)),
        'accuracy':    float(accuracy),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision':   float(precision_score(y_true, y_pred, zero_division=0)),
        'mcc':         float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ─────────────────────────────────────────────────────────────
# Inner LOPO — hyperparameter selection
# ─────────────────────────────────────────────────────────────

def inner_lopo_score(model, X_train, y_train, patient_ids_train):
    """
    Inner LOPO over the 7 training patients.
    Returns mean AUC across inner folds.
    Called once per hyperparameter candidate per outer fold.
    The outer test patient is never passed here.
    """
    inner_patients = np.unique(patient_ids_train)
    if len(inner_patients) < 2:
        return 0.0

    aucs = []
    for inner_pat in inner_patients:
        inner_test_mask  = (patient_ids_train == inner_pat)
        inner_train_mask = ~inner_test_mask

        y_inner_test = y_train[inner_test_mask]
        if len(np.unique(y_inner_test)) < 2:
            continue

        scaler = StandardScaler()
        X_itr  = scaler.fit_transform(X_train[inner_train_mask])
        X_ite  = scaler.transform(X_train[inner_test_mask])

        clf = copy.deepcopy(model)
        clf.fit(X_itr, y_train[inner_train_mask])

        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(X_ite)[:, 1]
        else:
            raw  = clf.decision_function(X_ite)
            prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

        aucs.append(roc_auc_score(y_inner_test, prob))

    return float(np.mean(aucs)) if aucs else 0.0


# ─────────────────────────────────────────────────────────────
# Hyperparameter grids
# ─────────────────────────────────────────────────────────────

def get_param_grid(model_name):
    """
    Small, justified grids — 3 candidates per model.
    Kept small deliberately: with 7 training patients and inner LOPO,
    a large grid would overfit the hyperparameter search itself.
    """
    if model_name == 'Random Forest':
        return [
            RandomForestClassifier(n_estimators=300, max_depth=4,
                min_samples_leaf=5, class_weight='balanced',
                random_state=42, n_jobs=-1),
            RandomForestClassifier(n_estimators=300, max_depth=6,
                min_samples_leaf=5, class_weight='balanced',
                random_state=42, n_jobs=-1),
            RandomForestClassifier(n_estimators=300, max_depth=8,
                min_samples_leaf=3, class_weight='balanced',
                random_state=42, n_jobs=-1),
        ]
    elif model_name == 'SVM RBF':
        return [
            SVC(kernel='rbf', C=0.1,  gamma='scale', class_weight='balanced',
                probability=True, random_state=42),
            SVC(kernel='rbf', C=1.0,  gamma='scale', class_weight='balanced',
                probability=True, random_state=42),
            SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced',
                probability=True, random_state=42),
        ]
    else:
        return [SVMWithRFE(n_features=20, C=1.0)]


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name, patient_id, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'])
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
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / f'roc_{model_name.lower().replace(" ","_")}.png',
        dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, model_name, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'accuracy', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'purple', 'tomato', 'seagreen', 'darkorange']
    x        = np.arange(len(patients))
    width    = 0.15
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
        output_dir / f'per_fold_{model_name.lower().replace(" ","_")}.png',
        dpi=150, bbox_inches='tight')
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
        output_dir / f'cm_aggregate_{model_name.lower().replace(" ","_")}.png',
        dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances_per_fold, feature_names, model_name, output_dir, top_n=30):
    mean_imp = np.mean(importances_per_fold, axis=0)
    idx      = np.argsort(mean_imp)[::-1][:top_n]
    fig, ax  = plt.subplots(figsize=(12, 5))
    ax.bar(range(top_n), mean_imp[idx], color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(np.array(feature_names)[idx], rotation=90, fontsize=7)
    ax.set_ylabel('Mean importance (LOPO folds)', fontsize=11)
    ax.set_title(f'{model_name} — Top {top_n} Feature Importances',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(
        output_dir / f'feature_importance_{model_name.lower().replace(" ","_")}.png',
        dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfit_diagnostic(all_results, output_dir):
    """
    Inner CV AUC vs Outer Test AUC per patient per model.
    Points below the diagonal = generalisation drop.
    Mean gap shown in title — negative = no overfitting.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['steelblue', 'tomato', 'seagreen']

    for ax, (model_name, res), color in zip(axes, all_results.items(), colors):
        folds = res['fold_metrics']
        if not folds:
            continue
        inner_aucs = [f['inner_cv_auc'] for f in folds]
        outer_aucs = [f['auc']          for f in folds]
        patients   = [f['patient']       for f in folds]

        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='No gap (ideal)', alpha=0.6)
        for x, y, pat in zip(inner_aucs, outer_aucs, patients):
            ax.scatter(x, y, color=color, s=80, zorder=5)
            ax.annotate(pat, (x, y), textcoords='offset points',
                        xytext=(5, 3), fontsize=7.5)

        gap = float(np.mean(np.array(inner_aucs) - np.array(outer_aucs)))
        ax.set_title(f'{model_name}\nMean gap = {gap:+.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Inner CV AUC (train)', fontsize=10)
        ax.set_ylabel('Outer Test AUC (generalisation)', fontsize=10)
        ax.set_xlim(0.3, 1.05)
        ax.set_ylim(0.1, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle('Overfitting Diagnostic — Inner CV vs Outer Test AUC',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'overfit_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Overfitting diagnostic  → {output_dir / "overfit_diagnostic.png"}')


def plot_train_vs_test_auc(all_results, output_dir):
    """
    Direct overfitting check: Train AUC vs Test AUC per fold per model.
    Train AUC is evaluated on the same data the model was fit on.
    A large train-test gap = the model memorised training data.
    Perfect model: both high and close together.
    """
    model_names = list(all_results.keys())
    n_models    = len(model_names)
    colors      = ['steelblue', 'tomato', 'seagreen']

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, model_name, color in zip(axes, model_names, colors):
        folds = all_results[model_name]['fold_metrics']
        if not folds:
            continue

        train_aucs = [f.get('train_auc', np.nan) for f in folds]
        test_aucs  = [f['auc']                    for f in folds]
        patients   = [f['patient']                for f in folds]

        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='No gap')
        for tr, te, pat in zip(train_aucs, test_aucs, patients):
            if np.isnan(tr):
                continue
            ax.scatter(tr, te, color=color, s=80, zorder=5)
            ax.annotate(pat, (tr, te), textcoords='offset points',
                        xytext=(5, 3), fontsize=7.5)

        valid = [(tr, te) for tr, te in zip(train_aucs, test_aucs)
                 if not np.isnan(tr)]
        if valid:
            gaps = [tr - te for tr, te in valid]
            mean_gap = float(np.mean(gaps))
            ax.set_title(f'{model_name}\nTrain–Test gap = {mean_gap:+.3f}',
                         fontsize=11, fontweight='bold')

        ax.set_xlabel('Train AUC', fontsize=10)
        ax.set_ylabel('Test AUC', fontsize=10)
        ax.set_xlim(0.3, 1.05)
        ax.set_ylim(0.1, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle('Train vs Test AUC — Direct Overfitting Check',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'train_vs_test_auc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Train vs test AUC       → {output_dir / "train_vs_test_auc.png"}')


def plot_calibration(all_results_probs, output_dir):
    """
    Reliability diagram: does the model's predicted probability
    match its actual accuracy?
    A well-calibrated model lies on the diagonal.
    RF tends to be overconfident; SVM probabilities are less reliable.
    """
    fig, axes = plt.subplots(1, len(all_results_probs), figsize=(5 * len(all_results_probs), 5))
    if len(all_results_probs) == 1:
        axes = [axes]
    colors = ['steelblue', 'tomato', 'seagreen']

    for ax, (model_name, (y_true_all, y_prob_all)), color in \
            zip(axes, all_results_probs.items(), colors):

        if len(y_true_all) == 0:
            continue

        try:
            fraction_pos, mean_pred = calibration_curve(
                y_true_all, y_prob_all, n_bins=8, strategy='quantile'
            )
        except ValueError:
            continue

        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Perfect calibration')
        ax.plot(mean_pred, fraction_pos, 'o-', color=color, lw=2,
                markersize=6, label=model_name)
        ax.fill_between(mean_pred, fraction_pos, mean_pred,
                        alpha=0.15, color=color)

        ax.set_xlabel('Mean predicted probability', fontsize=10)
        ax.set_ylabel('Fraction of positives (ictal)', fontsize=10)
        ax.set_title(f'{model_name}\nCalibration (Reliability Diagram)',
                     fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle('Model Calibration — Predicted Probability vs Actual Fraction',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Calibration plot        → {output_dir / "calibration.png"}')


# ─────────────────────────────────────────────────────────────
# LOPO evaluation — one model
# ─────────────────────────────────────────────────────────────

def run_lopo(model_name, model, X, y, patient_ids, feature_names, output_dir):
    """
    Nested Leave-One-Patient-Out CV.
    Outer fold : test on PAT_i, train on remaining 7 patients.
    Inner fold : within the 7 training patients, select best hyperparameter
                 via another LOPO — test patient never seen during tuning.
    Also records train AUC per fold for overfitting diagnostics.
    """
    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []
    all_y_prob    = []   # for calibration plot
    importances   = []

    print(f'\n{"=" * 60}')
    print(f'  {model_name} — Nested LOPO CV ({len(patients)} outer folds)')
    print(f'{"=" * 60}')

    param_grid = get_param_grid(model_name)

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        pat_ids_train    = patient_ids[train_mask]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: test set has only one class')
            continue

        # ── Inner LOPO: pick best hyperparameter ──────────
        best_model = param_grid[0]
        best_score = -1.0
        for candidate in param_grid:
            score = inner_lopo_score(candidate, X_train, y_train, pat_ids_train)
            if score > best_score:
                best_score = score
                best_model = candidate
        print(f'  {pat}: inner CV → best={best_model}  (AUC={best_score:.3f})')

        # ── Retrain on ALL 7 train patients ───────────────
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        clf = copy.deepcopy(best_model)
        clf.fit(X_tr_sc, y_train)

        # ── Train AUC (overfitting diagnostic) ────────────
        if hasattr(clf, 'predict_proba'):
            y_train_prob = clf.predict_proba(X_tr_sc)[:, 1]
        else:
            raw          = clf.decision_function(X_tr_sc)
            y_train_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        train_auc = float(roc_auc_score(y_train, y_train_prob))

        # ── Test predict ──────────────────────────────────
        y_pred = clf.predict(X_te_sc)
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_te_sc)[:, 1]
        else:
            raw    = clf.decision_function(X_te_sc)
            y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

        metrics                    = compute_metrics(y_test, y_pred, y_prob)
        metrics['patient']         = pat
        metrics['n_train']         = int(train_mask.sum())
        metrics['n_test']          = int(test_mask.sum())
        metrics['best_hyperparameter'] = str(best_model)
        metrics['inner_cv_auc']    = float(best_score)
        metrics['train_auc']       = train_auc          # ← direct overfit signal
        fold_metrics.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        plot_confusion_matrix(confusion_matrix(y_test, y_pred),
                              model_name, pat, output_dir)

        if hasattr(clf, 'feature_importances_'):
            importances.append(clf.feature_importances_)

        print(f'  {pat:8s} | TrainAUC={train_auc:.3f}  TestAUC={metrics["auc"]:.3f}  '
              f'Gap={train_auc - metrics["auc"]:+.3f}  '
              f'F1={metrics["f1"]:.3f}  Sens={metrics["sensitivity"]:.3f}  '
              f'Spec={metrics["specificity"]:.3f}  MCC={metrics["mcc"]:.3f}')

    if len(fold_metrics) == 0:
        print(f'  [ERROR] No valid folds for {model_name}')
        return [], {}, [], []

    # ── Aggregate plots ───────────────────────────────────
    plot_roc_all_folds(fold_roc_data, model_name, output_dir)
    plot_per_fold_metrics(fold_metrics, model_name, output_dir)
    plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred),
                             model_name, output_dir)
    if importances:
        plot_feature_importance(importances, feature_names, model_name, output_dir)

    # ── Summary stats ─────────────────────────────────────
    met_keys      = ['auc', 'accuracy', 'f1', 'sensitivity',
                     'specificity', 'precision', 'mcc']
    summary_stats = {}
    print(f'\n  {"─" * 50}')
    print(f'  {model_name} — Mean ± Std across {len(fold_metrics)} folds')
    print(f'  {"─" * 50}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    train_aucs = [m['train_auc'] for m in fold_metrics]
    test_aucs  = [m['auc']       for m in fold_metrics]
    mean_gap   = float(np.mean(np.array(train_aucs) - np.array(test_aucs)))
    print(f'  {"train_auc":15s}: {np.mean(train_aucs):.3f} ± {np.std(train_aucs):.3f}')
    print(f'  {"overfit gap":15s}: {mean_gap:+.3f}  '
          f'(train_auc − test_auc; <0.10 = healthy)')

    return fold_metrics, summary_stats, all_y_true, all_y_prob


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 4 — Baseline ML with Nested LOPO CV')
    parser.add_argument('--featfile',  required=True, help='features/features_all.npz')
    parser.add_argument('--outputdir', default='results/baseline_ml')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('STEP 4 — BASELINE ML (Nested LOPO CV)')
    print('=' * 60)

    data          = np.load(args.featfile, allow_pickle=True)
    X             = data['X'].astype(np.float32)
    y             = data['y'].astype(np.int64)
    patient_ids   = data['patient_ids']
    feature_names = data['feature_names'].tolist()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f'Loaded  : X={X.shape}  y={y.shape}')
    print(f'Ictal   : {(y==1).sum()}  |  Pre-ictal: {(y==0).sum()}')
    print(f'Patients: {np.unique(patient_ids).tolist()}')
    print(f'Features: {X.shape[1]}')
    print(f'Models  : Random Forest (nested CV), SVM RBF (nested CV), SVM RFE (fixed)')

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),  # placeholder
        'SVM RBF':       SVC(random_state=42),                     # placeholder
        'SVM RFE':       SVMWithRFE(n_features=20, C=1.0),
    }

    all_results       = {}
    all_results_probs = {}   # for calibration

    for model_name, model in models.items():
        fold_metrics, summary_stats, y_true_all, y_prob_all = run_lopo(
            model_name, model, X, y, patient_ids, feature_names, output_dir
        )
        all_results[model_name] = {
            'fold_metrics':  fold_metrics,
            'summary_stats': summary_stats,
        }
        all_results_probs[model_name] = (y_true_all, y_prob_all)

    # ── Save JSON ─────────────────────────────────────────
    results_path = output_dir / 'results_all.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults → {results_path}')

    # ── Diagnostic plots ──────────────────────────────────
    plot_overfit_diagnostic(all_results, output_dir)
    plot_train_vs_test_auc(all_results, output_dir)
    plot_calibration(all_results_probs, output_dir)

    # ── Summary table ─────────────────────────────────────
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
        print(f'Summary table           → {output_dir / "summary_table.csv"}')

    # ── Comparison bar chart ──────────────────────────────
    met_keys = ['auc', 'accuracy', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'seagreen']
    x        = np.arange(len(met_keys))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (model_name, res) in enumerate(all_results.items()):
        if not res['summary_stats']:
            continue
        means  = [res['summary_stats'][k]['mean'] for k in met_keys]
        stds   = [res['summary_stats'][k]['std']  for k in met_keys]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=model_name, color=colors[i % len(colors)],
               capsize=4, edgecolor='black', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('Baseline ML — Model Comparison (Nested LOPO CV)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Model comparison chart  → {output_dir / "model_comparison.png"}')

    print('\n' + '=' * 60)
    print('STEP 4 COMPLETE')
    print('=' * 60)
    print('Next: python step5_gnn_supervised.py --featfile features/features_all.npz')


if __name__ == '__main__':
    main()