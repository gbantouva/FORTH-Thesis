"""
SVM Baseline — Flattened Connectivity Features
================================================
Classical ML baseline before any GNN. Fits an SVM directly on the
flattened DTF and/or PDC matrices per epoch.

Why this matters:
    If SVM ≈ GCN  → the graph structure is not helping, the raw
                     connectivity values contain the signal
    If GCN > SVM  → the message-passing over the graph topology
                     adds genuine value beyond raw feature vectors
    If SVM > GCN  → the GNN is overfitting; the simpler model generalises
                     better to unseen patients

Feature options (--features argument):
    dtf     : flatten 19×19 DTF matrix → 361 features
    pdc     : flatten 19×19 PDC matrix → 361 features
    dtf_pdc : concatenate both         → 722 features
    node    : 9 node features per channel, flattened → 171 features
    all     : dtf + pdc + node features → 893 features

Same patient-independent split as GNN experiments:
    Train : subjects 3-10, 11, 12-25, 34
    Val   : subject 33   (used for C/gamma search)
    Test  : subjects 1, 2  (reported once)

Outputs:
    svm_results.json
    svm_confusion_matrix.png
    svm_roc_curve.png
    svm_feature_importance.png  (mean |feature| per class, top-k)

Usage:
    python step_svm_baseline.py \
        --connectivity_dir F:/FORTH_Final_Thesis/FORTH-Thesis/connectivity \
        --features_dir     F:/FORTH_Final_Thesis/FORTH-Thesis/node_features \
        --output_dir       F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/results/svm \
        --features         dtf_pdc
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, classification_report
)
from sklearn.pipeline import Pipeline

# ── patient split (same as GNN) ───────────────────────────────────────────────

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


# ── feature extraction ────────────────────────────────────────────────────────

def load_subject_features(subj_id, connectivity_dir, features_dir,
                           feature_type, ratio=2):
    """
    Returns X (n_epochs, n_features) and y (n_epochs,) for one subject.
    Applies the same 2:1 pre-ictal sampling from start of recording.
    """
    subject_name  = f'subject_{subj_id:02d}'
    graphs_file   = connectivity_dir / f'{subject_name}_graphs.npz'
    features_file = features_dir     / f'{subject_name}_node_features.npy'

    if not graphs_file.exists() or not features_file.exists():
        print(f'  [SKIP] {subject_name}')
        return None, None

    graphs        = np.load(graphs_file)
    node_features = np.load(features_file)   # (n_valid, 19, 9)
    dtf_all       = graphs['dtf_integrated'] # (n_valid, 19, 19)
    pdc_all       = graphs['pdc_integrated'] # (n_valid, 19, 19)
    labels_all    = graphs['labels']         # (n_valid,)
    n_valid       = len(labels_all)

    # time_from_onset for sampling
    if 'time_from_onset' in graphs:
        time_from_onset = graphs['time_from_onset']
    else:
        first_ictal = np.where(labels_all == 1)[0]
        offset = first_ictal[0] if len(first_ictal) > 0 else 0
        time_from_onset = (np.arange(n_valid) - offset) * 4.0

    # training mask
    if 'training_mask' in graphs:
        raw_mask = graphs['training_mask']
        mask = raw_mask[graphs['indices'].astype(int)] \
               if 'indices' in graphs else raw_mask[:n_valid]
    else:
        mask = np.ones(n_valid, dtype=bool)

    # separate and sample pre-ictal
    ictal_idx    = [i for i in range(n_valid) if mask[i] and labels_all[i] == 1]
    preictal_idx = [i for i in range(n_valid) if mask[i] and labels_all[i] == 0]

    n_pre_max = ratio * len(ictal_idx)
    preictal_sorted  = sorted(preictal_idx, key=lambda i: time_from_onset[i])
    selected_preictal = preictal_sorted[:n_pre_max]
    selected = ictal_idx + selected_preictal

    # build feature matrix
    feats = []
    for ep in selected:
        parts = []

        if feature_type in ('dtf', 'dtf_pdc', 'all'):
            # flatten 19×19 DTF, zero diagonal already
            parts.append(dtf_all[ep].flatten())        # 361

        if feature_type in ('pdc', 'dtf_pdc', 'all'):
            parts.append(pdc_all[ep].flatten())        # 361

        if feature_type in ('node', 'all'):
            parts.append(node_features[ep].flatten())  # 19*9 = 171

        feats.append(np.concatenate(parts))

    X = np.array(feats, dtype=np.float32)
    y = np.array([int(labels_all[ep]) for ep in selected], dtype=np.int64)
    return X, y


def build_split(split_subjects, connectivity_dir, features_dir,
                feature_type, ratio=2):
    X_list, y_list = [], []
    for subj_id in split_subjects:
        X, y = load_subject_features(
            subj_id, connectivity_dir, features_dir, feature_type, ratio
        )
        if X is not None:
            X_list.append(X)
            y_list.append(y)
            n_ict = int(y.sum())
            print(f'  subject_{subj_id:02d} ({SUBJECT_TO_PATIENT.get(subj_id,"?"):6s}): '
                  f'{len(y):4d} epochs  (ictal={n_ict})')
    return np.vstack(X_list), np.concatenate(y_list)


# ── plotting ──────────────────────────────────────────────────────────────────

def save_confusion_matrix(tn, fp, fn, tp, metrics, out_path, model_name):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Purples', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    classes = ['Pre-ictal (0)', 'Ictal (1)']
    ax.set_xticks([0,1]); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([0,1]); ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i,j] / max(cm[i].sum(), 1)
            ax.text(j, i, f'{cm[i,j]}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if cm[i,j] > thresh else 'black')
    ax.set_title(
        f'{model_name} — Test Confusion Matrix\n'
        f'AUROC={metrics["auroc"]:.4f}  Sens={metrics["sensitivity"]:.4f}  '
        f'Spec={metrics["specificity"]:.4f}  F1={metrics["f1_ictal"]:.4f}',
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_roc_curve(y_true, y_scores, auroc, out_path, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', linewidth=2,
            label=f'AUROC = {auroc:.4f}')
    ax.plot([0,1], [0,1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)',      fontsize=12)
    ax.set_title(f'{model_name} — ROC Curve (Test Set)', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_top_features(coef, feature_type, out_path, top_k=20):
    """
    For linear SVM: show the top_k most positive (ictal) and negative
    (pre-ictal) feature weights with channel-pair labels.
    """
    # build feature names
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
        feat_names = ['bp_delta','bp_theta','bp_alpha','bp_beta','bp_gamma',
                      'dtf_out','dtf_in','pdc_out','pdc_in']
        for ch in CHANNEL_NAMES:
            for fn in feat_names:
                names.append(f'{ch}_{fn}')

    w = coef.flatten()
    if len(w) != len(names):
        print(f'  [SKIP feature plot] weight dim {len(w)} != names {len(names)}')
        return

    idx_sorted = np.argsort(w)
    top_pos = idx_sorted[-top_k:][::-1]  # most ictal
    top_neg = idx_sorted[:top_k]          # most pre-ictal

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('SVM Feature Weights\n'
                 'Left: most ictal  |  Right: most pre-ictal',
                 fontsize=12, fontweight='bold')

    for ax, indices, title, color in [
        (axes[0], top_pos, f'Top {top_k} ictal features',     'firebrick'),
        (axes[1], top_neg, f'Top {top_k} pre-ictal features', 'steelblue'),
    ]:
        vals  = w[indices]
        lbls  = [names[i] for i in indices]
        ypos  = range(len(lbls))
        ax.barh(ypos, np.abs(vals), color=color, alpha=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(lbls, fontsize=8)
        ax.set_xlabel('|Weight|')
        ax.set_title(title, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SVM baseline for epilepsy detection')
    parser.add_argument('--connectivity_dir', required=True)
    parser.add_argument('--features_dir',     required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--features', default='dtf_pdc',
                        choices=['dtf', 'pdc', 'dtf_pdc', 'node', 'all'],
                        help='Which features to use (default: dtf_pdc)')
    parser.add_argument('--kernel', default='rbf',
                        choices=['rbf', 'linear'],
                        help='SVM kernel (default: rbf)')
    parser.add_argument('--ratio', type=int, default=2,
                        help='Pre-ictal to ictal ratio (default: 2)')
    args = parser.parse_args()

    connectivity_dir = Path(args.connectivity_dir)
    features_dir     = Path(args.features_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('SVM BASELINE')
    print('=' * 70)
    print(f'Features : {args.features}')
    print(f'Kernel   : {args.kernel}')
    print(f'Ratio    : {args.ratio}:1 (pre-ictal:ictal, from start of recording)')
    print(f'Split    : patient-independent (train/val/test)')
    print('=' * 70)

    # ── load data ─────────────────────────────────────────────────────────────
    print('\nBuilding TRAIN set...')
    X_train, y_train = build_split(
        PATIENT_SPLITS['train'], connectivity_dir, features_dir,
        args.features, args.ratio
    )

    print('\nBuilding VAL set...')
    X_val, y_val = build_split(
        PATIENT_SPLITS['val'], connectivity_dir, features_dir,
        args.features, args.ratio
    )

    print('\nBuilding TEST set...')
    X_test, y_test = build_split(
        PATIENT_SPLITS['test'], connectivity_dir, features_dir,
        args.features, args.ratio
    )

    print(f'\nFeature dimensions: {X_train.shape[1]}')
    print(f'Train: {X_train.shape[0]} epochs  '
          f'(ictal={y_train.sum()}, pre={len(y_train)-y_train.sum()})')
    print(f'Val  : {X_val.shape[0]} epochs  '
          f'(ictal={y_val.sum()}, pre={len(y_val)-y_val.sum()})')
    print(f'Test : {X_test.shape[0]} epochs  '
          f'(ictal={y_test.sum()}, pre={len(y_test)-y_test.sum()})')

    # ── class weights ─────────────────────────────────────────────────────────
    # SVM class_weight='balanced' applies the same inverse-frequency
    # weighting as we used for the GNN
    class_weight = 'balanced'

    # ── grid search over C using val set ──────────────────────────────────────
    # Combine train+val for fitting, use val AUROC to pick C
    print('\n── Hyperparameter search (C) on validation AUROC ─────────────────')
    print(f'{"C":>10}  {"ValAUROC":>10}  {"ValSens":>8}  {"ValSpec":>8}')
    print('-' * 45)

    # Fit scaler on train only, transform both
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train)
    X_val_sc  = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    best_C      = 1.0
    best_auroc  = -1.0
    candidates  = [0.01, 0.1, 1.0, 10.0, 100.0]

    for C in candidates:
        svm = SVC(kernel=args.kernel, C=C, probability=True,
                  class_weight=class_weight, random_state=42)
        svm.fit(X_tr_sc, y_train)
        probs_val = svm.predict_proba(X_val_sc)[:, 1]
        preds_val = svm.predict(X_val_sc)

        auroc = roc_auc_score(y_val, probs_val)
        cm    = confusion_matrix(y_val, preds_val, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / max(tp+fn, 1)
        spec = tn / max(tn+fp, 1)
        print(f'{C:>10.2f}  {auroc:>10.4f}  {sens:>8.4f}  {spec:>8.4f}')

        if auroc > best_auroc:
            best_auroc = auroc
            best_C     = C

    print(f'\nBest C = {best_C}  (val AUROC = {best_auroc:.4f})')

    # ── final model: train on train+val, evaluate on test ─────────────────────
    print('\n── Final model: train+val → test ─────────────────────────────────')
    X_trainval   = np.vstack([X_train, X_val])
    y_trainval   = np.concatenate([y_train, y_val])
    scaler_final = StandardScaler()
    X_tv_sc      = scaler_final.fit_transform(X_trainval)
    X_test_sc2   = scaler_final.transform(X_test)

    final_svm = SVC(kernel=args.kernel, C=best_C, probability=True,
                    class_weight=class_weight, random_state=42)
    final_svm.fit(X_tv_sc, y_trainval)

    probs_test = final_svm.predict_proba(X_test_sc2)[:, 1]
    preds_test = final_svm.predict(X_test_sc2)

    auroc = roc_auc_score(y_test, probs_test)
    cm    = confusion_matrix(y_test, preds_test, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    sens    = tp / max(tp+fn, 1)
    spec    = tn / max(tn+fp, 1)
    f1_ict  = f1_score(y_test, preds_test, pos_label=1, zero_division=0)

    metrics = {
        'auroc'      : round(float(auroc), 4),
        'sensitivity': round(float(sens),  4),
        'specificity': round(float(spec),  4),
        'f1_ictal'   : round(float(f1_ict),4),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
    }

    print(f'\nTEST RESULTS (C={best_C}, features={args.features}):')
    print(f'  AUROC       : {metrics["auroc"]:.4f}')
    print(f'  Sensitivity : {metrics["sensitivity"]:.4f}  '
          f'— caught {tp} / {tp+fn} ictal epochs')
    print(f'  Specificity : {metrics["specificity"]:.4f}  '
          f'— correct on {tn} / {tn+fp} pre-ictal epochs')
    print(f'  F1-ictal    : {metrics["f1_ictal"]:.4f}')
    print(f'\n  Confusion matrix:')
    print(f'                  Predicted')
    print(f'                  Pre-ictal   Ictal')
    print(f'  Actual Pre-ictal  [ {tn:4d}      {fp:4d} ]')
    print(f'  Actual Ictal      [ {fn:4d}      {tp:4d} ]')

    # ── save ──────────────────────────────────────────────────────────────────
    print('\nSaving outputs...')
    model_name = f'SVM-{args.kernel.upper()} ({args.features})'

    results = {
        'model'        : model_name,
        'features'     : args.features,
        'kernel'       : args.kernel,
        'best_C'       : best_C,
        'best_val_auroc': best_auroc,
        'n_features'   : int(X_train.shape[1]),
        'test'         : metrics,
        'hparams'      : vars(args),
    }
    with open(output_dir / 'svm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('  Saved: svm_results.json')

    save_confusion_matrix(
        tn, fp, fn, tp, metrics,
        output_dir / f'svm_{args.features}_confusion_matrix.png',
        model_name
    )

    save_roc_curve(
        y_test, probs_test, auroc,
        output_dir / f'svm_{args.features}_roc_curve.png',
        model_name
    )

    # feature importance only for linear kernel
    if args.kernel == 'linear':
        save_top_features(
            final_svm.coef_,
            args.features,
            output_dir / f'svm_{args.features}_feature_weights.png',
        )

    print(f'\nAll outputs saved to: {output_dir}')
    print('\n── Summary for thesis comparison table ───────────────────────────')
    print(f'  SVM ({args.features}, C={best_C}):')
    print(f'    AUROC={metrics["auroc"]:.4f}  '
          f'Sens={metrics["sensitivity"]:.4f}  '
          f'Spec={metrics["specificity"]:.4f}  '
          f'F1={metrics["f1_ictal"]:.4f}')


if __name__ == '__main__':
    main()
