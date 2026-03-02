"""
Step 4 — Baseline ML with Leave-One-Patient-Out Cross-Validation
================================================================
Input : features/all_features.npz  (from step3)
Output: results/
          metrics_summary.csv
          fold_metrics.csv
          confusion_matrix_<model>.png
          metrics_per_fold_<model>.png
          roc_curve_<model>.png
          final_comparison.png

Labels : 0 = pre-ictal,  1 = ictal

Usage:
  python step4_baseline_ml.py \
    --features path/to/features/all_features.npz \
    --outdir   path/to/results
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings('ignore')

LABEL_NAMES = {0: 'Pre-ictal', 1: 'Ictal'}
PALETTE     = {'Pre-ictal': '#4C72B0', 'Ictal': '#DD8452'}


# ═══════════════════════════════════════════════════════════════
# 1. METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_ictal    = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)
    bal_acc     = balanced_accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    except Exception:
        auc = float('nan')

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision':   precision,
        'f1_ictal':    f1_ictal,
        'bal_acc':     bal_acc,
        'auc':         auc,
        'cm':          cm,
    }


# ═══════════════════════════════════════════════════════════════
# 2. PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm_total, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_total, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Pre-ictal', 'Ictal'],
        yticklabels=['Pre-ictal', 'Ictal'],
        linewidths=0.5, linecolor='gray',
        annot_kws={'size': 14, 'weight': 'bold'},
    )
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label',      fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}\n(all LOPO folds pooled)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = out_dir / f'confusion_matrix_{model_name.replace(" ", "_")}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_metrics_per_fold(fold_records, model_name, out_dir):
    df      = pd.DataFrame(fold_records)
    metrics = ['sensitivity', 'specificity', 'f1_ictal', 'bal_acc', 'auc']
    labels  = ['Sensitivity', 'Specificity', 'F1 (Ictal)', 'Bal. Accuracy', 'AUC']
    colors  = ['#4C72B0', '#55A868', '#DD8452', '#C44E52', '#8172B2']

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4), sharey=True)
    fig.suptitle(f'Per-Fold Metrics — {model_name}  (LOPO)',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, m, label, color in zip(axes, metrics, labels, colors):
        vals = df[m].values
        folds = np.arange(1, len(vals) + 1)
        ax.bar(folds, vals, color=color, alpha=0.75, edgecolor='black', linewidth=0.6)
        ax.axhline(np.nanmean(vals), color='red', linestyle='--',
                   linewidth=1.5, label=f'Mean={np.nanmean(vals):.2f}')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Fold (patient)', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(folds)
        ax.set_xticklabels([f"P{int(p)}" for p in df['patient'].values],
                           fontsize=8, rotation=45)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    if hasattr(axes[0], 'set_ylabel'):
        axes[0].set_ylabel('Score', fontsize=10)

    plt.tight_layout()
    path = out_dir / f'metrics_per_fold_{model_name.replace(" ", "_")}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_roc_curve(all_y_true, all_y_proba, fold_aucs, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))

    # Overall ROC (pooled across folds)
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    overall_auc = roc_auc_score(all_y_true, all_y_proba)
    ax.plot(fpr, tpr, color='#C44E52', lw=2.5,
            label=f'Overall ROC (AUC = {overall_auc:.3f})')

    # Chance line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')

    # Shaded AUC std band
    mean_auc = np.nanmean(fold_aucs)
    std_auc  = np.nanstd(fold_aucs)
    ax.set_title(f'ROC Curve — {model_name}\n'
                 f'Mean fold AUC = {mean_auc:.3f} ± {std_auc:.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / f'roc_curve_{model_name.replace(" ", "_")}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_final_comparison(summary_df, out_dir):
    metrics = ['sensitivity', 'specificity', 'f1_ictal', 'bal_acc', 'auc']
    labels  = ['Sensitivity', 'Specificity', 'F1 (Ictal)', 'Bal. Accuracy', 'AUC']
    models  = summary_df['model'].tolist()
    x       = np.arange(len(metrics))
    width   = 0.35
    colors  = ['#4C72B0', '#DD8452']

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (model, color) in enumerate(zip(models, colors)):
        row    = summary_df[summary_df['model'] == model].iloc[0]
        means  = [row[f'{m}_mean'] for m in metrics]
        stds   = [row[f'{m}_std']  for m in metrics]
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, means, width,
                        label=model, color=color, alpha=0.8,
                        edgecolor='black', linewidth=0.6)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt='none', color='black', capsize=4, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Score (mean ± std across LOPO folds)', fontsize=11)
    ax.set_title('Baseline Model Comparison — LOPO Cross-Validation',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'final_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════
# 3. LOPO EVALUATION
# ═══════════════════════════════════════════════════════════════

def run_lopo(X, y, groups, clf_name, clf, out_dir):
    logo         = LeaveOneGroupOut()
    fold_records = []
    cm_total     = np.zeros((2, 2), dtype=int)
    all_y_true   = []
    all_y_proba  = []

    print(f"\n{'='*65}")
    print(f"  Model: {clf_name}")
    print(f"{'='*65}")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_patient = int(groups[test_idx[0]])
        X_tr, y_tr   = X[train_idx], y[train_idx]
        X_te, y_te   = X[test_idx],  y[test_idx]

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_te_sc  = scaler.transform(X_te)

        model = type(clf)(**clf.get_params())
        model.fit(X_tr_sc, y_tr)

        y_pred  = model.predict(X_te_sc)
        y_proba = model.predict_proba(X_te_sc)

        m = compute_metrics(y_te, y_pred, y_proba)
        cm_total += m['cm']

        all_y_true.extend(y_te.tolist())
        all_y_proba.extend(y_proba[:, 1].tolist())

        fold_records.append({
            'fold':        fold + 1,
            'patient':     test_patient,
            'n_test':      len(y_te),
            'n_ictal':     int(y_te.sum()),
            'n_preictal':  int((y_te == 0).sum()),
            'sensitivity': m['sensitivity'],
            'specificity': m['specificity'],
            'precision':   m['precision'],
            'f1_ictal':    m['f1_ictal'],
            'bal_acc':     m['bal_acc'],
            'auc':         m['auc'],
        })

        print(f"  Fold {fold+1:2d} | patient={test_patient:2d} "
              f"| n={len(y_te):3d} "
              f"(ict={y_te.sum():2d} pre={( y_te==0).sum():2d}) "
              f"| Sens={m['sensitivity']:.3f} "
              f"Spec={m['specificity']:.3f} "
              f"AUC={m['auc']:.3f}")

    # ── Aggregate ──────────────────────────────────────────────
    metrics_keys = ['sensitivity', 'specificity', 'precision',
                    'f1_ictal', 'bal_acc', 'auc']
    summary_row  = {'model': clf_name}
    print(f"\n  {'─'*55}")
    print(f"  Summary ({len(fold_records)} folds):")
    for k in metrics_keys:
        vals = [r[k] for r in fold_records if not np.isnan(r[k])]
        mu, sd = np.mean(vals), np.std(vals)
        summary_row[f'{k}_mean'] = mu
        summary_row[f'{k}_std']  = sd
        print(f"    {k:15s}: {mu:.3f} ± {sd:.3f}")

    print(f"\n  Pooled confusion matrix:")
    print(f"  {cm_total}")
    print(f"  rows=true (pre-ictal, ictal)  cols=predicted")

    # ── Plots ──────────────────────────────────────────────────
    print(f"\n  Saving plots...")
    plot_confusion_matrix(cm_total, clf_name, out_dir)
    plot_metrics_per_fold(fold_records, clf_name, out_dir)
    plot_roc_curve(
        np.array(all_y_true),
        np.array(all_y_proba),
        [r['auc'] for r in fold_records],
        clf_name, out_dir,
    )

    return summary_row, fold_records


# ═══════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════

def main(features_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data   = np.load(features_path)
    X      = data['X']
    y      = data['y']
    groups = data['groups']

    print(f"\n{'='*65}")
    print(f"  STEP 4 — BASELINE ML  (LOPO cross-validation)")
    print(f"{'='*65}")
    print(f"  Features : {X.shape}")
    print(f"  Labels   : {Counter(y)}  (0=pre-ictal, 1=ictal)")
    print(f"  Patients : {sorted(np.unique(groups).tolist())}")

    if len(np.unique(groups)) < 2:
        raise ValueError("Need at least 2 patients for LOPO CV.")

    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=500,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        'SVM-RBF': SVC(
            kernel='rbf',
            C=1.0,
            class_weight='balanced',
            probability=True,
            random_state=42,
        ),
    }

    all_summaries  = []
    all_fold_rows  = []

    for name, clf in models.items():
        summary_row, fold_records = run_lopo(X, y, groups, name, clf, out_dir)
        all_summaries.append(summary_row)
        for r in fold_records:
            r['model'] = name
        all_fold_rows.extend(fold_records)

    # ── Save CSVs ──────────────────────────────────────────────
    summary_df  = pd.DataFrame(all_summaries)
    fold_df     = pd.DataFrame(all_fold_rows)

    summary_csv = out_dir / 'metrics_summary.csv'
    fold_csv    = out_dir / 'fold_metrics.csv'
    summary_df.to_csv(summary_csv, index=False, float_format='%.4f')
    fold_df.to_csv(fold_csv,    index=False, float_format='%.4f')
    print(f"\n  Saved: {summary_csv.name}")
    print(f"  Saved: {fold_csv.name}")

    # ── Final comparison plot ──────────────────────────────────
    plot_final_comparison(summary_df, out_dir)

    # ── Print final table ──────────────────────────────────────
    metrics_keys = ['sensitivity', 'specificity', 'f1_ictal', 'bal_acc', 'auc']
    print(f"\n{'='*65}")
    print(f"  FINAL COMPARISON (mean ± std across LOPO folds)")
    print(f"{'='*65}")
    header = f"  {'Model':15s}  {'Sens':>10}  {'Spec':>10}  {'F1':>10}  {'BalAcc':>10}  {'AUC':>10}"
    print(header)
    print(f"  {'─'*63}")
    for row in all_summaries:
        print(f"  {row['model']:15s}  "
              + "  ".join(
                  f"{row[f'{k}_mean']:.3f}±{row[f'{k}_std']:.2f}"
                  for k in metrics_keys
              ))

    print(f"\n  All outputs saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 4 — Baseline ML")
    parser.add_argument('--features', required=True,
                        help='Path to features/all_features.npz')
    parser.add_argument('--outdir', default='results',
                        help='Output folder for plots and CSVs (default: results/)')
    args = parser.parse_args()
    main(args.features, args.outdir)
