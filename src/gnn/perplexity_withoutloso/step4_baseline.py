"""
Step 4 — Baseline ML with Leave-One-Patient-Out Cross-Validation
================================================================
Input : features/all_features.npz  (from step3)

Models : Random Forest, SVM-RBF
CV     : Leave-One-Patient-Out (LOPO) — patients never split across folds
Metrics: Sensitivity, Specificity, F1, Balanced Accuracy, AUC-ROC

Labels : 0 = pre-ictal,  1 = ictal

Usage:
  python step4_baseline_ml.py --features path/to/features/all_features.npz
"""

import argparse
import warnings
import numpy as np
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
)

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# 1. METRICS HELPER
# ═══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_score):
    """
    Returns dict with sensitivity, specificity, F1 (ictal),
    balanced accuracy, and AUC.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm[0,0]=TN  cm[0,1]=FP  cm[1,0]=FN  cm[1,1]=TP
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall ictal
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # recall pre-ictal
    bal_acc     = balanced_accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_score[:, 1])
    except Exception:
        auc = float('nan')

    # F1 for ictal class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_ictal  = (2 * precision * sensitivity / (precision + sensitivity)
                 if (precision + sensitivity) > 0 else 0.0)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_ictal':    f1_ictal,
        'bal_acc':     bal_acc,
        'auc':         auc,
        'confusion':   cm,
    }


# ═══════════════════════════════════════════════════════════════
# 2. LOPO EVALUATION
# ═══════════════════════════════════════════════════════════════

def run_lopo(X, y, groups, clf_name, clf):
    logo         = LeaveOneGroupOut()
    fold_metrics = []
    all_y_true   = []
    all_y_pred   = []

    print(f"\n{'='*65}")
    print(f"  Model: {clf_name}")
    print(f"{'='*65}")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_patient = groups[test_idx[0]]
        X_tr, y_tr  = X[train_idx], y[train_idx]
        X_te, y_te  = X[test_idx],  y[test_idx]

        # Scale: fit on train only — no leakage
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # Fresh model per fold
        model = type(clf)(**clf.get_params())
        model.fit(X_tr_sc, y_tr)

        y_pred  = model.predict(X_te_sc)
        y_score = model.predict_proba(X_te_sc)

        m = compute_metrics(y_te, y_pred, y_score)
        fold_metrics.append(m)
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(f"  Fold {fold+1:2d} | patient={test_patient} "
              f"| n_test={len(y_te):3d} "
              f"(ictal={y_te.sum():2d} pre-ictal={(y_te==0).sum():2d}) "
              f"| Sens={m['sensitivity']:.2f} "
              f"Spec={m['specificity']:.2f} "
              f"AUC={m['auc']:.3f}")

    # Aggregate
    keys    = ['sensitivity', 'specificity', 'f1_ictal', 'bal_acc', 'auc']
    summary = {}
    print(f"\n  {'─'*55}")
    print(f"  LOPO Summary ({len(fold_metrics)} folds):")
    for k in keys:
        vals = [m[k] for m in fold_metrics if not np.isnan(m[k])]
        mu, sd = np.mean(vals), np.std(vals)
        summary[k] = (mu, sd)
        print(f"    {k:15s}: {mu:.3f} ± {sd:.3f}")

    # Overall confusion matrix across all folds
    print(f"\n  Overall confusion matrix (all folds pooled):")
    print(f"  {confusion_matrix(all_y_true, all_y_pred, labels=[0,1])}")
    print(f"  rows=true (pre-ictal, ictal)  cols=predicted")

    return summary, fold_metrics


# ═══════════════════════════════════════════════════════════════
# 3. MAIN
# ═══════════════════════════════════════════════════════════════

def main(features_path):
    data   = np.load(features_path)
    X      = data['X']
    y      = data['y']
    groups = data['groups']

    print(f"\n{'='*65}")
    print(f"  STEP 4 — BASELINE ML  (LOPO cross-validation)")
    print(f"{'='*65}")
    print(f"  Features  : {X.shape}")
    print(f"  Labels    : {Counter(y)}  (0=pre-ictal, 1=ictal)")
    print(f"  Patients  : {sorted(np.unique(groups).tolist())}")

    n_patients = len(np.unique(groups))
    if n_patients < 2:
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
            probability=True,       # needed for AUC
            random_state=42,
        ),
    }

    all_summaries = {}
    for name, clf in models.items():
        summary, _ = run_lopo(X, y, groups, name, clf)
        all_summaries[name] = summary

    # Final comparison table
    print(f"\n{'='*65}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*65}")
    header = f"  {'Model':15s}  {'Sens':>8}  {'Spec':>8}  {'F1-ictal':>10}  {'BalAcc':>8}  {'AUC':>8}"
    print(header)
    print(f"  {'─'*60}")
    for name, s in all_summaries.items():
        print(f"  {name:15s}  "
              f"{s['sensitivity'][0]:.3f}±{s['sensitivity'][1]:.2f}  "
              f"{s['specificity'][0]:.3f}±{s['specificity'][1]:.2f}  "
              f"{s['f1_ictal'][0]:.3f}±{s['f1_ictal'][1]:.2f}      "
              f"{s['bal_acc'][0]:.3f}±{s['bal_acc'][1]:.2f}  "
              f"{s['auc'][0]:.3f}±{s['auc'][1]:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 4 — Baseline ML")
    parser.add_argument('--features', required=True,
                        help='Path to features/all_features.npz')
    args = parser.parse_args()
    main(args.features)
