"""
Step 4 — Baseline ML: Random Forest + SVM + SVM-RFE  (Nested LOPO CV)
=======================================================================
DESIGN DECISIONS (document in thesis):

1. EVALUATION: Nested Leave-One-Patient-Out (LOPO) CV.
   ┌─ Outer loop (8 folds) ─────────────────────────────────────────┐
   │  Leave 1 patient out → test fold                               │
   │  ┌─ Inner loop (7 folds per outer fold) ───────────────────────┐│
   │  │  Of the 7 remaining patients, rotate 1 as val (7 fits)     ││
   │  │  → mean inner val AUC reported alongside outer test AUC    ││
   │  └─────────────────────────────────────────────────────────────┘│
   └────────────────────────────────────────────────────────────────┘
   Inner CV is for reporting only — not hyperparameter search.
   Hyperparameters are fixed a priori and justified below.

2. MODELS:
   (a) Random Forest  — see OVERFITTING section for justification
   (b) SVM RBF        — see OVERFITTING section
   (c) SVM-RFE        — RFE with linear SVM for ranking (Guyon et al. 2002),
       followed by RBF SVM on the reduced feature set. Selector fit on
       training fold ONLY — no leakage.

3. METRICS:
   Primary: AUC, F1, Sensitivity, Specificity, MCC
   Also reported: Accuracy (per professor's request — interpreted alongside
   class balance because it is misleading in isolation).

4. FEATURE SELECTION (SVM-RFE):
   n_features_to_select=10 out of 53. Controlled via --rfe_n_features.
   For RF and plain SVM, all 53 features are used.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERFITTING PREVENTION — METHODS (document each in thesis):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
With only 34 subjects / 8 patients, overfitting risk is HIGH.
Every design choice below directly addresses this.

A) Random Forest regularisation:
   - max_depth=6          : shallow trees — limits complexity per tree.
                            A deeper tree (e.g. 20) would fit noise in
                            individual patients perfectly.
   - min_samples_leaf=5   : leaf nodes must contain ≥5 samples.
                            Prevents isolated-sample leaves that memorise
                            single training examples.
   - max_features='sqrt'  : each split considers only √53 ≈ 7 features.
                            Introduces diversity and reduces variance.
   - n_estimators=300     : ensemble averaging further reduces variance.
   - class_weight='balanced': reweights loss — prevents majority-class
                            (pre-ictal) dominance from causing the model
                            to trivially predict all zeros.

B) SVM regularisation:
   - C=1.0                : moderate margin penalty — not too soft (under-
                            fit) nor too hard (overfit to training points).
   - gamma='scale'        : 1/(n_features × X.var()) — scale-invariant,
                            avoids the rbf kernel collapsing to dot-product
                            or becoming a nearest-neighbour classifier.
   - class_weight='balanced': same reason as RF above.

C) Feature scaling:
   - StandardScaler fit on TRAINING FOLD ONLY, then applied to val/test.
   - If fit on all data: test statistics contaminate the scaler → leakage.

D) Curated, compressed feature set (from step 3):
   - 53 features (not 114 raw per-channel spectral features).
   - Region averaging compresses 19-channel spectral data into 5 regions.
   - With 34 subjects a 114-feature input causes the curse of dimensionality;
     53 features keep the ratio of samples to features tractable.

E) SVM-RFE additional regularisation:
   - Reducing to 10 features adds a strong implicit regulariser.
   - Linear SVM base estimator (C=0.1, conservative) for ranking step.

F) No grid search / hyperparameter tuning:
   - With only 8 LOPO folds, any tuning loop inflates variance enormously.
   - Hyperparameters are set from published BCI/EEG literature defaults
     and justified here rather than selected by optimisation.
   - The inner CV reports val performance but does NOT feed back into
     hyperparameter selection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERFITTING DETECTION — PLOTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  train_vs_test_gap_{model}.png
      Bar chart: train AUC vs inner-val AUC vs outer-test AUC per fold.
      A large train–val gap signals fold-level overfitting.
      A consistent train–test gap across folds confirms generalisation.

  learning_curve_{model}.png
      Train and cross-val AUC vs number of training epochs / trees.
      For RF: score vs n_estimators (1→300).
      For SVM: score vs C (0.01→100, log scale).
      Diverging curves = overfitting; converging = good regularisation.

  overfitting_summary.png
      Per-fold (train AUC − test AUC) gap for all three models side by side.
      Values near 0 = no overfitting; values >> 0 = memorisation.

  inner_vs_outer_{model}.png
      Scatter of inner val AUC vs outer test AUC.
      Points near diagonal = inner CV is a reliable generalisation proxy.

Outputs (full list):
  results_all.json
  summary_table.csv
  cm_{model}_{patient}.png
  roc_{model}.png
  per_fold_{model}.png
  model_comparison.png
  train_vs_test_gap_{model}.png   ← overfitting plot
  learning_curve_{model}.png      ← overfitting plot
  overfitting_summary.png         ← overfitting plot
  inner_vs_outer_{model}.png      ← overfitting plot
  rfe_feature_stability.png
  feature_importance_*.png

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
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'auc':         float(roc_auc_score(y_true, y_prob)),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(tp / (tp + fn + 1e-12)),
        'specificity': float(tn / (tn + fp + 1e-12)),
        'precision':   float(precision_score(y_true, y_pred, zero_division=0)),
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'mcc':         float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def _decision_to_prob(raw):
    rng = raw.max() - raw.min()
    return (raw - raw.min()) / (rng + 1e-12)


def train_auc(clf, X_tr, y_tr):
    """AUC on the training set itself — used to measure the train/test gap."""
    if hasattr(clf, 'predict_proba'):
        prob = clf.predict_proba(X_tr)[:, 1]
    else:
        prob = _decision_to_prob(clf.decision_function(X_tr))
    try:
        return float(roc_auc_score(y_tr, prob))
    except Exception:
        return float('nan')


# ─────────────────────────────────────────────────────────────
# Inner CV
# ─────────────────────────────────────────────────────────────

def run_inner_cv(model_fn, X_train, y_train, train_patient_ids,
                 rfe_n_features=None):
    """
    Leave-one-patient-out over TRAINING patients only.
    Returns (mean_val_auc, mean_val_acc, rfe_support_consensus or None).
    Scaling is fit on inner-train only inside each inner fold.
    """
    inner_patients = np.unique(train_patient_ids)
    inner_aucs, inner_accs, rfe_supports = [], [], []

    for val_pat in inner_patients:
        iv_mask = (train_patient_ids == val_pat)
        it_mask = ~iv_mask

        Xi_tr, yi_tr = X_train[it_mask], y_train[it_mask]
        Xi_va, yi_va = X_train[iv_mask], y_train[iv_mask]

        if len(np.unique(yi_va)) < 2:
            continue

        sc       = StandardScaler()
        Xi_tr_s  = sc.fit_transform(Xi_tr)
        Xi_va_s  = sc.transform(Xi_va)
        clf      = model_fn()

        if rfe_n_features is not None:
            base = SVC(kernel='linear', C=0.1, class_weight='balanced', max_iter=2000)
            sel  = RFE(base, n_features_to_select=rfe_n_features, step=1)
            sel.fit(Xi_tr_s, yi_tr)
            Xi_tr_s = sel.transform(Xi_tr_s)
            Xi_va_s = sel.transform(Xi_va_s)
            rfe_supports.append(sel.support_)

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
    rfe_consensus = np.stack(rfe_supports).mean(axis=0) >= 0.5 if rfe_supports else None
    return mean_auc, mean_acc, rfe_consensus


# ─────────────────────────────────────────────────────────────
# Overfitting plots
# ─────────────────────────────────────────────────────────────

def plot_train_vs_test_gap(fold_metrics, model_name, output_dir):
    """
    Per-fold bar chart comparing train AUC, inner-val AUC, and outer-test AUC.

    Interpretation for thesis:
      - Train AUC >> test AUC across all folds → model is memorising training data.
      - Train AUC ≈ inner-val AUC ≈ outer-test AUC → good generalisation.
      - Inner-val AUC ≈ outer-test AUC → inner CV is a reliable proxy,
        validating the use of nested LOPO for model selection.

    Note: for sklearn models there is no epoch-wise training trajectory,
    so train AUC is computed on the full training fold after fitting.
    A gap here reflects the inherent difficulty of the held-out patient,
    not necessarily overfitting per se — which is why we report this per
    fold and discuss patient-level variability.
    """
    patients  = [m['patient'] for m in fold_metrics]
    tr_aucs   = [m.get('train_auc',     float('nan')) for m in fold_metrics]
    val_aucs  = [m.get('inner_val_auc', float('nan')) for m in fold_metrics]
    test_aucs = [m['auc'] for m in fold_metrics]

    x     = np.arange(len(patients))
    w     = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, tr_aucs,   w, label='Train AUC',      color='steelblue',  alpha=0.85, edgecolor='black')
    ax.bar(x,     val_aucs,  w, label='Inner val AUC',   color='darkorange', alpha=0.85, edgecolor='black')
    ax.bar(x + w, test_aucs, w, label='Outer test AUC',  color='tomato',     alpha=0.85, edgecolor='black')

    # Annotate gap (train − test) above each group
    for i, (tr, te) in enumerate(zip(tr_aucs, test_aucs)):
        if not (np.isnan(tr) or np.isnan(te)):
            gap = tr - te
            ax.text(x[i], max(tr, te) + 0.02, f'Δ{gap:+.2f}',
                    ha='center', fontsize=8, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.22)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title(f'{model_name} — Train / Inner-val / Test AUC per fold\n'
                 f'(Δ = train−test gap; values near 0 indicate no overfitting)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fname = f'train_vs_test_gap_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curve(model_name, model_fn, X_train, y_train,
                        scaler, output_dir, rfe_n_features=None):
    """
    Sklearn learning_curve: train and CV AUC vs training-set size.

    Interpretation for thesis:
      - If train AUC is high but CV AUC is low for small training sizes,
        the model overfits when data is scarce — expected for complex models.
      - Converging curves as size increases = model can generalise with more data.
      - Persistent gap even at full size = bias or fundamental generalisation limit.
      - This is especially important to discuss given the small dataset (34 subjects).

    We use a 5-fold stratified CV inside this function (not LOPO) for
    computational tractability. This is standard for learning curves.
    """
    X_sc = scaler.fit_transform(X_train)

    if rfe_n_features is not None:
        base_rfe = SVC(kernel='linear', C=0.1, class_weight='balanced', max_iter=2000)
        sel = RFE(base_rfe, n_features_to_select=rfe_n_features, step=1)
        sel.fit(X_sc, y_train)
        X_sc = sel.transform(X_sc)

    clf = model_fn()
    train_sizes = np.linspace(0.2, 1.0, 6)

    try:
        sizes, tr_scores, cv_scores = learning_curve(
            clf, X_sc, y_train,
            train_sizes=train_sizes,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            shuffle=True,
            random_state=42,
        )
    except Exception as e:
        print(f'  [WARN] Learning curve failed for {model_name}: {e}')
        return

    tr_mean, tr_std = tr_scores.mean(axis=1), tr_scores.std(axis=1)
    cv_mean, cv_std = cv_scores.mean(axis=1), cv_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, tr_mean, 'o-', color='steelblue', lw=2, label='Training AUC')
    ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color='steelblue')
    ax.plot(sizes, cv_mean, 's--', color='tomato', lw=2, label='CV AUC (5-fold stratified)')
    ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std,
                    alpha=0.15, color='tomato')
    ax.set_xlabel('Training set size (samples)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.set_title(
        f'{model_name} — Learning Curve\n'
        f'Converging curves = good regularisation; diverging = overfitting',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'learning_curve_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting_summary(all_fold_metrics_dict, output_dir):
    """
    Side-by-side (train AUC − test AUC) gap per fold, for all models.

    Interpretation for thesis:
      - Positive gap = model scores higher on its own training data than test.
      - Gap ≈ 0 across folds → model is not memorising; regularisation is effective.
      - Comparing models: a simpler model (SVM) may have a smaller gap than RF
        even if its test AUC is lower — this is a bias-variance discussion point.
    """
    models   = list(all_fold_metrics_dict.keys())
    patients = sorted({m['patient']
                       for mlist in all_fold_metrics_dict.values()
                       for m in mlist})

    x      = np.arange(len(patients))
    width  = 0.28
    colors = ['steelblue', 'tomato', 'darkorange']
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (model_name, fold_metrics) in enumerate(all_fold_metrics_dict.items()):
        pat_map = {m['patient']: m for m in fold_metrics}
        gaps = []
        for pat in patients:
            if pat in pat_map:
                tr  = pat_map[pat].get('train_auc', float('nan'))
                te  = pat_map[pat]['auc']
                gaps.append(tr - te if not (np.isnan(tr) or np.isnan(te)) else 0.0)
            else:
                gaps.append(0.0)
        offset = (i - 1) * width
        ax.bar(x + offset, gaps, width,
               label=model_name, color=colors[i % len(colors)],
               alpha=0.85, edgecolor='black')

    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Train AUC − Test AUC', fontsize=12)
    ax.set_title('Overfitting Summary — Train/Test AUC Gap per Patient Fold\n'
                 '(Near-zero bars = no overfitting; positive bars = memorisation)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Overfitting summary → {output_dir / "overfitting_summary.png"}')


def plot_inner_vs_outer_auc(fold_metrics, model_name, output_dir):
    """
    Scatter: inner-val AUC vs outer-test AUC per fold.
    Points near the diagonal confirm that the inner CV is a reliable
    proxy for generalisation — important for the nested LOPO justification.
    """
    inner = [m.get('inner_val_auc', float('nan')) for m in fold_metrics]
    outer = [m['auc'] for m in fold_metrics]
    valid = [(i, o, m['patient'])
             for i, o, m in zip(inner, outer, fold_metrics) if not np.isnan(i)]
    if not valid:
        return
    xi, xo, labels = zip(*valid)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xi, xo, s=80, zorder=5, color='steelblue', edgecolor='black')
    lim = [min(min(xi), min(xo)) - 0.05, max(max(xi), max(xo)) + 0.05]
    ax.plot(lim, lim, 'k--', lw=1, label='Diagonal (inner = outer)')
    for xi_, xo_, lab in zip(xi, xo, labels):
        ax.annotate(lab, (xi_, xo_), fontsize=8, xytext=(4, 4),
                    textcoords='offset points')
    ax.set_xlabel('Inner CV val AUC', fontsize=12)
    ax.set_ylabel('Outer test AUC', fontsize=12)
    ax.set_title(f'{model_name} — Inner vs Outer AUC\n'
                 f'(Points near diagonal = reliable nested CV)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'inner_vs_outer_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────
# Standard plots
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
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{model_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, model_name, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    x = np.arange(len(patients))
    w = 0.16
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (met, col) in enumerate(zip(met_keys, colors)):
        ax.bar(x + i * w, [m[met] for m in fold_metrics], w,
               label=met.upper(), color=col, alpha=0.8, edgecolor='black')
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title(f'{model_name} — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'per_fold_{model_name.lower().replace(" ", "_")}.png',
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
        ax.set_title(f'{model_name} — Aggregate CM ({title})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_aggregate_{model_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances_per_fold, feature_names, model_name, output_dir,
                            top_n=30):
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
    plt.savefig(output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_rfe_feature_stability(rfe_support_per_fold, feature_names, output_dir):
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


# ─────────────────────────────────────────────────────────────
# LOPO evaluation for one model
# ─────────────────────────────────────────────────────────────

def run_lopo(model_name, model_fn, X, y, patient_ids, feature_names,
             output_dir, rfe_n_features=None):
    """
    Outer LOPO CV with inner LOPO CV.
    For each outer fold:
      1. Run inner CV over training patients → record inner_val_auc.
      2. Fit final model on all training data (scaled on train only).
      3. Record train AUC (on training set) for overfitting detection.
      4. Evaluate on held-out test patient.
      5. After all folds: plot train/val/test gap and learning curve.
    """
    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []
    importances   = []
    rfe_supports  = []

    # Collect one training fold for learning curve (use last valid fold)
    lc_X_train, lc_y_train, lc_scaler = None, None, None

    print(f'\n{"=" * 65}')
    print(f'  {model_name} — Nested LOPO CV ({len(patients)} outer folds)')
    if rfe_n_features:
        print(f'  RFE: retaining top {rfe_n_features} features (linear SVM ranking)')
    print(f'{"=" * 65}')

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        train_pats       = patient_ids[train_mask]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: test set has only one class')
            continue

        # ── Inner CV ───────────────────────────────────────────
        inner_val_auc, inner_val_acc, _ = run_inner_cv(
            model_fn, X_train, y_train, train_pats,
            rfe_n_features=rfe_n_features
        )

        # ── Outer fold ─────────────────────────────────────────
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_train)
        X_te_sc  = scaler.transform(X_test)

        # Save for learning curve
        lc_X_train, lc_y_train, lc_scaler = X_train.copy(), y_train.copy(), scaler

        clf = model_fn()

        if rfe_n_features is not None:
            base = SVC(kernel='linear', C=0.1, class_weight='balanced', max_iter=2000)
            sel  = RFE(base, n_features_to_select=rfe_n_features, step=1)
            sel.fit(X_tr_sc, y_train)
            rfe_supports.append(sel.support_)
            X_tr_sel = sel.transform(X_tr_sc)
            X_te_sel = sel.transform(X_te_sc)
            clf.fit(X_tr_sel, y_train)
            y_pred = clf.predict(X_te_sel)
            y_prob = clf.predict_proba(X_te_sel)[:, 1] if hasattr(clf, 'predict_proba') \
                     else _decision_to_prob(clf.decision_function(X_te_sel))
            tr_auc = train_auc(clf, X_tr_sel, y_train)
        else:
            clf.fit(X_tr_sc, y_train)
            y_pred = clf.predict(X_te_sc)
            y_prob = clf.predict_proba(X_te_sc)[:, 1] if hasattr(clf, 'predict_proba') \
                     else _decision_to_prob(clf.decision_function(X_te_sc))
            tr_auc = train_auc(clf, X_tr_sc, y_train)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['patient']       = pat
        metrics['n_train']       = int(train_mask.sum())
        metrics['n_test']        = int(test_mask.sum())
        metrics['inner_val_auc'] = inner_val_auc
        metrics['inner_val_acc'] = inner_val_acc
        metrics['train_auc']     = tr_auc
        fold_metrics.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        plot_confusion_matrix(confusion_matrix(y_test, y_pred), model_name, pat, output_dir)

        if hasattr(clf, 'feature_importances_'):
            importances.append(clf.feature_importances_)

        print(f'  {pat:8s} | Test  AUC={metrics["auc"]:.3f}  Acc={metrics["accuracy"]:.3f}'
              f'  F1={metrics["f1"]:.3f}  Sens={metrics["sensitivity"]:.3f}'
              f'  Spec={metrics["specificity"]:.3f}  MCC={metrics["mcc"]:.3f}')
        print(f'           | Train AUC={tr_auc:.3f}  '
              f'Gap={tr_auc - metrics["auc"]:+.3f}  '
              f'Inner val AUC={inner_val_auc:.3f}')

    if not fold_metrics:
        return [], {}

    # ── Overfitting plots ──────────────────────────────────────
    plot_train_vs_test_gap(fold_metrics, model_name, output_dir)
    plot_inner_vs_outer_auc(fold_metrics, model_name, output_dir)

    if lc_X_train is not None:
        plot_learning_curve(model_name, model_fn, lc_X_train, lc_y_train,
                            StandardScaler(), output_dir,
                            rfe_n_features=rfe_n_features)

    # ── Standard plots ─────────────────────────────────────────
    plot_roc_all_folds(fold_roc_data, model_name, output_dir)
    plot_per_fold_metrics(fold_metrics, model_name, output_dir)
    plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred), model_name, output_dir)

    if importances:
        plot_feature_importance(importances, feature_names, model_name, output_dir)
    if rfe_supports:
        plot_rfe_feature_stability(rfe_supports, feature_names, output_dir)

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

    tr_aucs = [m['train_auc'] for m in fold_metrics if not np.isnan(m.get('train_auc', float('nan')))]
    if tr_aucs:
        mean_gap = float(np.mean(tr_aucs)) - summary_stats['auc']['mean']
        print(f'\n  Mean train AUC     : {np.mean(tr_aucs):.3f}')
        print(f'  Mean test  AUC     : {summary_stats["auc"]["mean"]:.3f}')
        print(f'  Mean overfit gap   : {mean_gap:+.3f}  (positive = model scores higher on train)')
        summary_stats['train_auc']    = {'mean': float(np.mean(tr_aucs)), 'std': float(np.std(tr_aucs))}
        summary_stats['overfit_gap']  = {'mean': mean_gap}

    inner_aucs = [m['inner_val_auc'] for m in fold_metrics
                  if not np.isnan(m.get('inner_val_auc', float('nan')))]
    if inner_aucs:
        print(f'  Mean inner val AUC : {np.mean(inner_aucs):.3f} ± {np.std(inner_aucs):.3f}')
        summary_stats['inner_val_auc'] = {'mean': float(np.mean(inner_aucs)),
                                          'std':  float(np.std(inner_aucs))}

    return fold_metrics, summary_stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 4 — Baseline ML (Nested LOPO + overfitting analysis)')
    parser.add_argument('--featfile',       required=True)
    parser.add_argument('--outputdir',      default='results/baseline_ml')
    parser.add_argument('--rfe_n_features', type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('STEP 4 — BASELINE ML (Nested LOPO + Overfitting Analysis)')
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
    print(f'RFE n_features: {args.rfe_n_features}')
    print()
    print('Overfitting prevention methods:')
    print('  RF  : max_depth=6, min_samples_leaf=5, max_features=sqrt, n_estimators=300')
    print('  SVM : C=1.0, gamma=scale, class_weight=balanced')
    print('  All : StandardScaler fit on train only, no test-set hyperparameter search')
    print()
    print('Overfitting detection plots (generated per model):')
    print('  train_vs_test_gap_*.png  — train/val/test AUC bar chart per fold')
    print('  learning_curve_*.png     — AUC vs training size')
    print('  overfitting_summary.png  — gap comparison across all models')
    print('  inner_vs_outer_*.png     — inner-val AUC vs outer-test AUC scatter')

    def make_rf():
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            max_features='sqrt',          # ← regularisation: sqrt(53) ≈ 7 features per split
            class_weight='balanced', random_state=42, n_jobs=-1,
        )

    def make_svm():
        return SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=42,
        )

    def make_svm_rfe():
        return SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=42,
        )

    model_configs = [
        ('Random Forest', make_rf,      None),
        ('SVM RBF',       make_svm,     None),
        ('SVM RFE',       make_svm_rfe, args.rfe_n_features),
    ]

    all_results      = {}
    all_fold_metrics = {}   # for cross-model overfitting summary

    for model_name, model_fn, rfe_n in model_configs:
        fold_metrics, summary_stats = run_lopo(
            model_name, model_fn, X, y, patient_ids, feature_names,
            output_dir, rfe_n_features=rfe_n,
        )
        all_results[model_name]      = {'fold_metrics': fold_metrics, 'summary_stats': summary_stats}
        all_fold_metrics[model_name] = fold_metrics

    # ── Cross-model overfitting summary ───────────────────────
    valid = {k: v for k, v in all_fold_metrics.items() if v}
    if valid:
        plot_overfitting_summary(valid, output_dir)

    # ── Save JSON ──────────────────────────────────────────────
    results_path = output_dir / 'results_all.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults → {results_path}')

    # ── Summary table ──────────────────────────────────────────
    print('\n' + '=' * 65)
    print('FINAL COMPARISON')
    print('=' * 65)
    rows = []
    for model_name, res in all_results.items():
        if not res['summary_stats']:
            continue
        row = {'Model': model_name}
        for k in ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy', 'mcc', 'overfit_gap']:
            v = res['summary_stats'].get(k, {})
            if isinstance(v, dict):
                m, s = v.get('mean', float('nan')), v.get('std', float('nan'))
                row[k] = f'{m:.3f} ± {s:.3f}' if not np.isnan(s) else f'{m:.3f}'
            else:
                row[k] = '-'
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows).set_index('Model')
        print(df.to_string())
        df.to_csv(output_dir / 'summary_table.csv', encoding='utf-8')

    # ── Comparison bar chart ───────────────────────────────────
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    colors   = ['steelblue', 'tomato', 'darkorange']
    x        = np.arange(len(met_keys))
    fig, ax  = plt.subplots(figsize=(11, 5))
    for i, (model_name, res) in enumerate(all_results.items()):
        if not res['summary_stats']:
            continue
        means  = [res['summary_stats'].get(k, {}).get('mean', 0) for k in met_keys]
        stds   = [res['summary_stats'].get(k, {}).get('std',  0) for k in met_keys]
        offset = (i - 1) * 0.28
        ax.bar(x + offset, means, 0.28, yerr=stds, label=model_name,
               color=colors[i % len(colors)], capsize=4, edgecolor='black', alpha=0.85)
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

    print('\n' + '=' * 65)
    print('STEP 4 COMPLETE')
    print('=' * 65)
    print('\nNext: python step5_gnn_supervised.py --featfile features/features_all.npz'
          ' --baseline_json results/baseline_ml/results_all.json')


if __name__ == '__main__':
    main()
