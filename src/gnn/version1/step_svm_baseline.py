"""
SVM Baseline — Flattened Connectivity Features
================================================
Outputs:
  svm_results.json              ← detailed experiment JSON
  svm_confusion_matrix.png
  svm_roc_curve.png
  svm_feature_importance.png    (linear kernel only)

Usage:
  python step_svm_baseline.py \
    --connectivity_dir F:/FORTH_Final_Thesis/FORTH-Thesis/connectivity \
    --features_dir     F:/FORTH_Final_Thesis/FORTH-Thesis/node_features \
    --output_dir       F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/results/svm \
    --features         dtf_pdc
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, balanced_accuracy_score,
)

# ── patient split ─────────────────────────────────────────────
PATIENT_SPLITS = {
    'train': list(range(3, 11)) + [11] + list(range(12, 26)) + [34],
    'val'  : [33],
    'test' : [1, 2],
}

SUBJECT_TO_PATIENT = {
    1: 'PAT11', 2: 'PAT13',
    **{i: 'PAT14' for i in range(3, 11)},
    11: 'PAT15',
    **{i: 'PAT24' for i in range(12, 26)},
    **{i: 'PAT27' for i in range(26, 33)},
    33: 'PAT29', 34: 'PAT35',
}

CHANNEL_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T3','C3','Cz','C4','T4',
    'T5','P3','Pz','P4','T6','O1','O2',
]


# ── feature extraction ────────────────────────────────────────

def load_subject_features(subj_id, connectivity_dir, features_dir,
                           feature_type, ratio=2):
    subject_name  = f'subject_{subj_id:02d}'
    graphs_file   = connectivity_dir / f'{subject_name}_graphs.npz'
    features_file = features_dir     / f'{subject_name}_node_features.npy'

    if not graphs_file.exists() or not features_file.exists():
        print(f'  [SKIP] {subject_name}')
        return None, None

    graphs        = np.load(graphs_file)
    node_features = np.load(features_file)       # (n_valid, 19, 9)
    dtf_all       = graphs['dtf_integrated']     # (n_valid, 19, 19)
    pdc_all       = graphs['pdc_integrated']     # (n_valid, 19, 19)
    labels_all    = graphs['labels']             # (n_valid,)
    n_valid       = len(labels_all)

    if 'time_from_onset' in graphs:
        time_from_onset = graphs['time_from_onset']
    else:
        first_ictal     = np.where(labels_all == 1)[0]
        offset          = first_ictal[0] if len(first_ictal) > 0 else 0
        time_from_onset = (np.arange(n_valid) - offset) * 4.0

    if 'training_mask' in graphs:
        raw_mask = graphs['training_mask']
        mask     = (raw_mask[graphs['indices'].astype(int)]
                    if 'indices' in graphs else raw_mask[:n_valid])
    else:
        mask = np.ones(n_valid, dtype=bool)

    ictal_idx    = [i for i in range(n_valid) if mask[i] and labels_all[i] == 1]
    preictal_idx = [i for i in range(n_valid) if mask[i] and labels_all[i] == 0]

    n_pre_max        = ratio * len(ictal_idx)
    preictal_sorted  = sorted(preictal_idx, key=lambda i: time_from_onset[i])
    selected_preictal = preictal_sorted[:n_pre_max]
    selected         = ictal_idx + selected_preictal

    feats = []
    for ep in selected:
        parts = []
        if feature_type in ('dtf', 'dtf_pdc', 'all'):
            parts.append(dtf_all[ep].flatten())
        if feature_type in ('pdc', 'dtf_pdc', 'all'):
            parts.append(pdc_all[ep].flatten())
        if feature_type in ('node', 'all'):
            parts.append(node_features[ep].flatten())
        feats.append(np.concatenate(parts))

    X = np.array(feats, dtype=np.float32)
    y = np.array([int(labels_all[ep]) for ep in selected], dtype=np.int64)
    return X, y


def build_split(split_subjects, connectivity_dir, features_dir,
                feature_type, ratio=2):
    X_list, y_list = [], []
    for subj_id in split_subjects:
        X, y = load_subject_features(
            subj_id, connectivity_dir, features_dir, feature_type, ratio)
        if X is not None:
            X_list.append(X)
            y_list.append(y)
            print(f'  subject_{subj_id:02d} '
                  f'({SUBJECT_TO_PATIENT.get(subj_id,"?"):6s}): '
                  f'{len(y):4d} epochs (ictal={int(y.sum())})')
    return np.vstack(X_list), np.concatenate(y_list)


# ── metrics helper ────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba):
    auroc   = float(roc_auc_score(y_true, y_proba))
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens    = tp / max(tp + fn, 1)
    spec    = tn / max(tn + fp, 1)
    prec    = tp / max(tp + fp, 1)
    f1      = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    return {
        'auroc'            : round(auroc, 4),
        'sensitivity'      : round(float(sens),    4),
        'specificity'      : round(float(spec),    4),
        'precision'        : round(float(prec),    4),
        'f1_ictal'         : round(float(f1),      4),
        'balanced_accuracy': round(bal_acc,         4),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
        'n_ictal'    : int(tp + fn),
        'n_preictal' : int(tn + fp),
        'n_total'    : int(len(y_true)),
    }


# ── plots ─────────────────────────────────────────────────────

def save_confusion_matrix(metrics, out_path, model_name):
    cm  = np.array([[metrics['tn'], metrics['fp']],
                    [metrics['fn'], metrics['tp']]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im  = ax.imshow(cm, cmap='Purples', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    classes = ['Pre-ictal (0)', 'Ictal (1)']
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / max(cm[i].sum(), 1)
            ax.text(j, i, f'{cm[i,j]}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_title(
        f'{model_name}\n'
        f'AUROC={metrics["auroc"]:.4f}  '
        f'Sens={metrics["sensitivity"]:.4f}  '
        f'Spec={metrics["specificity"]:.4f}  '
        f'F1={metrics["f1_ictal"]:.4f}',
        fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_roc_curve(y_true, y_scores, auroc, out_path, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', linewidth=2,
            label=f'AUROC = {auroc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(f'{model_name} — ROC Curve (Test Set)', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_top_features(coef, feature_type, out_path, top_k=20):
    names = []
    if feature_type in ('dtf', 'dtf_pdc', 'all'):
        for i in range(19):
            for j in range(19):
                names.append(f'DTF {CHANNEL_NAMES[j]}→{CHANNEL_NAMES[i]}')
    if feature_type in ('pdc', 'dtf_pdc', 'all'):
        for i in range(19):
            for j in range(19):
                names.append(f'PDC {CHANNEL_NAMES[j]}→{CHANNEL_NAMES[i]}')
    if feature_type in ('node', 'all'):
        for ch in CHANNEL_NAMES:
            for fn in ['bp_delta','bp_theta','bp_alpha','bp_beta','bp_gamma',
                       'dtf_out','dtf_in','pdc_out','pdc_in']:
                names.append(f'{ch}_{fn}')

    w = coef.flatten()
    if len(w) != len(names):
        print(f'  [SKIP feature plot] weight dim mismatch')
        return

    idx_sorted = np.argsort(w)
    top_pos    = idx_sorted[-top_k:][::-1]
    top_neg    = idx_sorted[:top_k]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('SVM Feature Weights  |  Left: ictal  |  Right: pre-ictal',
                 fontsize=12, fontweight='bold')
    for ax, indices, title, color in [
        (axes[0], top_pos, f'Top {top_k} ictal features',     'firebrick'),
        (axes[1], top_neg, f'Top {top_k} pre-ictal features', 'steelblue'),
    ]:
        vals = w[indices]
        lbls = [names[i] for i in indices]
        ypos = range(len(lbls))
        ax.barh(ypos, np.abs(vals), color=color, alpha=0.8)
        ax.set_yticks(ypos); ax.set_yticklabels(lbls, fontsize=8)
        ax.set_xlabel('|Weight|')
        ax.set_title(title, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ── JSON builder ──────────────────────────────────────────────

def build_experiment_json(args, best_C, best_val_auroc,
                           X_train, y_train,
                           X_val,   y_val,
                           X_test,  y_test,
                           val_metrics, test_metrics,
                           val_search_log, output_dir):
    """
    Build and save a fully self-contained experiment JSON.
    Every field needed to reproduce or understand the result
    is included.
    """
    feature_dims = {
        'dtf'    : 361,
        'pdc'    : 361,
        'dtf_pdc': 722,
        'node'   : 171,
        'all'    : 893,
    }

    doc = {
        # ── identity ──────────────────────────────────────────
        "experiment": {
            "script"         : "step_svm_baseline.py",
            "timestamp"      : datetime.now().isoformat(timespec='seconds'),
            "model_family"   : "Classical ML",
            "model"          : f"SVM-{args.kernel.upper()}",
            "description"    : (
                "SVM baseline trained on flattened DTF/PDC connectivity "
                "matrices. Patient-independent train/val/test split."
            ),
        },

        # ── hyperparameters ───────────────────────────────────
        "hyperparameters": {
            "kernel"         : args.kernel,
            "C"              : best_C,
            "C_search_space" : [0.01, 0.1, 1.0, 10.0, 100.0],
            "C_selected_by"  : "validation AUROC",
            "class_weight"   : "balanced",
            "probability"    : True,
            "random_state"   : 42,
            "preprocessing"  : "StandardScaler (fit on train only)",
        },

        # ── data ──────────────────────────────────────────────
        "data": {
            "connectivity_dir"  : str(args.connectivity_dir),
            "features_dir"      : str(args.features_dir),
            "feature_type"      : args.features,
            "feature_dim"       : feature_dims.get(args.features, -1),
            "preictal_ratio"    : args.ratio,
            "preictal_strategy" : "earliest epochs from start of recording",
            "cv_strategy"       : "fixed patient-independent split (train/val/test)",
            "split": {
                "train_subjects": PATIENT_SPLITS['train'],
                "val_subjects"  : PATIENT_SPLITS['val'],
                "test_subjects" : PATIENT_SPLITS['test'],
                "train_patients": sorted(set(
                    SUBJECT_TO_PATIENT[s]
                    for s in PATIENT_SPLITS['train']
                    if s in SUBJECT_TO_PATIENT)),
                "val_patients"  : [SUBJECT_TO_PATIENT.get(s)
                                   for s in PATIENT_SPLITS['val']],
                "test_patients" : [SUBJECT_TO_PATIENT.get(s)
                                   for s in PATIENT_SPLITS['test']],
            },
            "split_sizes": {
                "train": {
                    "n_total"  : int(len(y_train)),
                    "n_ictal"  : int(y_train.sum()),
                    "n_preictal": int((y_train == 0).sum()),
                },
                "val": {
                    "n_total"  : int(len(y_val)),
                    "n_ictal"  : int(y_val.sum()),
                    "n_preictal": int((y_val == 0).sum()),
                },
                "test": {
                    "n_total"  : int(len(y_test)),
                    "n_ictal"  : int(y_test.sum()),
                    "n_preictal": int((y_test == 0).sum()),
                },
            },
        },

        # ── hyperparameter search log ─────────────────────────
        "val_search": {
            "best_C"        : best_C,
            "best_val_auroc": round(best_val_auroc, 4),
            "all_trials"    : val_search_log,
        },

        # ── results ───────────────────────────────────────────
        "results": {
            "primary_metric": "auroc",
            "val" : val_metrics,
            "test": test_metrics,
            "confusion_matrix_test": {
                "layout"     : "[[TN, FP], [FN, TP]]",
                "matrix"     : [
                    [test_metrics['tn'], test_metrics['fp']],
                    [test_metrics['fn'], test_metrics['tp']],
                ],
            },
        },

        # ── outputs ───────────────────────────────────────────
        "outputs": {
            "output_dir"  : str(output_dir),
            "json_file"   : "svm_results.json",
            "plots"       : [
                f"svm_{args.features}_confusion_matrix.png",
                f"svm_{args.features}_roc_curve.png",
            ] + ([f"svm_{args.features}_feature_weights.png"]
                 if args.kernel == 'linear' else []),
        },
    }
    return doc


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SVM baseline for epilepsy detection')
    parser.add_argument('--connectivity_dir', required=True)
    parser.add_argument('--features_dir',     required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--features', default='dtf_pdc',
                        choices=['dtf', 'pdc', 'dtf_pdc', 'node', 'all'])
    parser.add_argument('--kernel',   default='rbf',
                        choices=['rbf', 'linear'])
    parser.add_argument('--ratio',    type=int, default=2)
    args = parser.parse_args()

    connectivity_dir = Path(args.connectivity_dir)
    features_dir     = Path(args.features_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('SVM BASELINE')
    print('=' * 70)
    print(f'  Features : {args.features}')
    print(f'  Kernel   : {args.kernel}')
    print(f'  Ratio    : {args.ratio}:1')
    print('=' * 70)

    # ── load data ─────────────────────────────────────────────
    print('\nBuilding TRAIN set...')
    X_train, y_train = build_split(
        PATIENT_SPLITS['train'], connectivity_dir,
        features_dir, args.features, args.ratio)

    print('\nBuilding VAL set...')
    X_val, y_val = build_split(
        PATIENT_SPLITS['val'], connectivity_dir,
        features_dir, args.features, args.ratio)

    print('\nBuilding TEST set...')
    X_test, y_test = build_split(
        PATIENT_SPLITS['test'], connectivity_dir,
        features_dir, args.features, args.ratio)

    # ── scale ─────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_te_sc  = scaler.transform(X_test)

    # ── C search on val ───────────────────────────────────────
    print('\n── C search (validation AUROC) ───────────────────────')
    print(f'{"C":>10} {"ValAUROC":>10} {"ValSens":>8} {"ValSpec":>8}')
    print('-' * 45)

    best_C, best_val_auroc = 1.0, -1.0
    val_search_log = []

    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        svm = SVC(kernel=args.kernel, C=C, probability=True,
                  class_weight='balanced', random_state=42)
        svm.fit(X_tr_sc, y_train)
        proba_v = svm.predict_proba(X_val_sc)[:, 1]
        pred_v  = svm.predict(X_val_sc)
        m       = compute_metrics(y_val, pred_v, proba_v)

        print(f'{C:>10.2f} {m["auroc"]:>10.4f} '
              f'{m["sensitivity"]:>8.4f} {m["specificity"]:>8.4f}')
        val_search_log.append({
            'C': C, 'auroc': m['auroc'],
            'sensitivity': m['sensitivity'],
            'specificity': m['specificity'],
        })
        if m['auroc'] > best_val_auroc:
            best_val_auroc = m['auroc']
            best_C         = C
            best_val_metrics = m

    print(f'\nBest C = {best_C}  (val AUROC = {best_val_auroc:.4f})')

    # ── final model: train+val → test ─────────────────────────
    X_tv    = np.vstack([X_train, X_val])
    y_tv    = np.concatenate([y_train, y_val])
    sc_fin  = StandardScaler()
    X_tv_sc = sc_fin.fit_transform(X_tv)
    X_te_sc = sc_fin.transform(X_test)

    final_svm = SVC(kernel=args.kernel, C=best_C, probability=True,
                    class_weight='balanced', random_state=42)
    final_svm.fit(X_tv_sc, y_tv)

    proba_test = final_svm.predict_proba(X_te_sc)[:, 1]
    pred_test  = final_svm.predict(X_te_sc)
    test_metrics = compute_metrics(y_test, pred_test, proba_test)

    print(f'\nTEST RESULTS:')
    for k, v in test_metrics.items():
        print(f'  {k:20s}: {v}')

    model_name = f'SVM-{args.kernel.upper()} ({args.features})'

    # ── save plots ────────────────────────────────────────────
    save_confusion_matrix(
        test_metrics,
        output_dir / f'svm_{args.features}_confusion_matrix.png',
        model_name)
    save_roc_curve(
        y_test, proba_test, test_metrics['auroc'],
        output_dir / f'svm_{args.features}_roc_curve.png',
        model_name)
    if args.kernel == 'linear':
        save_top_features(
            final_svm.coef_, args.features,
            output_dir / f'svm_{args.features}_feature_weights.png')

    # ── save JSON ─────────────────────────────────────────────
    experiment_doc = build_experiment_json(
        args, best_C, best_val_auroc,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        best_val_metrics, test_metrics,
        val_search_log,
        output_dir,
    )
    json_path = output_dir / 'svm_results.json'
    with open(json_path, 'w') as f:
        json.dump(experiment_doc, f, indent=2)
    print(f'  Saved: {json_path.name}')

    print(f'\nAll outputs saved to: {output_dir}')


if __name__ == '__main__':
    main()
