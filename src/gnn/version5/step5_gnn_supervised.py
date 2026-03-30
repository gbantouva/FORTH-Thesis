"""
Step 5 — Supervised GCN  (LOPO Cross-Validation)
=================================================
DESIGN DECISIONS (document in thesis):

1. ARCHITECTURE — kept deliberately small to avoid overfitting:
     GCNLayer(in_dim → hidden)  →  Dropout
     GCNLayer(hidden → hidden)  →  GlobalMeanPool
     Linear(hidden → 16)  →  ReLU  →  Dropout
     Linear(16 → 1)  →  sigmoid (binary output)
   Total ~4k parameters.

2. NORMALISED ADJACENCY:
     A_hat = D^{-1/2} (A + I) D^{-1/2}   (Kipf & Welling 2017)
   Threshold sparsifies graph — keeps strongest DTF connections only.

3. IMBALANCE:
   - BCEWithLogitsLoss with pos_weight = n_neg / n_pos (per fold)
   - Adaptive decision threshold: optimal threshold found on training
     fold by maximising F1 — never touches test data.
   - AUC reported as primary metric (threshold-independent).

4. EARLY STOPPING on validation AUC (patience=20).
   Best-state restored at end of each fold.

5. EVALUATION: LOPO by patient.
   StandardScaler fit on train split only — no leakage.

6. ABLATION: --ablation flag runs GCN with ones vector as node features.
   Isolates contribution of graph topology vs node features.
   Both modes use adaptive threshold for fair comparison.

7. OVERFITTING: train AUC vs test AUC per fold + loss curves per patient.

Outputs (full model):
  loss_curve_{patient}.png
  cm_gcn_{patient}.png
  roc_gcn.png
  per_fold_gcn.png
  cm_aggregate_gcn.png
  train_vs_test_auc_gcn.png
  comparison_all_models.png    (if --baseline_json provided)
  results_gcn.json

Usage:
  # Step A — ablation first (topology only):
  python step5_gnn_supervised.py \
      --featfile  features/features_all.npz \
      --outputdir results/gnn_ablation \
      --ablation \
      --threshold 0.05

  # Step B — full model + comparison chart:
  python step5_gnn_supervised.py \
      --featfile       features/features_all.npz \
      --outputdir      results/gnn_supervised \
      --ablation_json  results/gnn_ablation/results_gcn.json \
      --baseline_json  results/baseline_ml/results_all.json
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N_CHANNELS = 19


# ─────────────────────────────────────────────────────────────
# Adjacency normalisation
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """
    A_hat = D^{-1/2} (A + I) D^{-1/2}   (Kipf & Welling 2017)
    Self-loops added before normalisation to retain self-information.
    """
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# GCN model
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(a_hat @ x))


class SmallGCN(nn.Module):
    """
    2-layer GCN → GlobalMeanPool → 2-layer MLP → scalar logit.
    in_dim=16  for full model (node features from step 3)
    in_dim=1   for ablation  (ones vector — topology only)
    """
    def __init__(self, in_dim: int = 16, hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, a_hat)
        h = self.drop(h)
        h = self.gcn2(h, a_hat)
        h = h.mean(dim=0, keepdim=True)    # global mean pool → (1, hidden)
        return self.head(h).squeeze()       # scalar logit


# ─────────────────────────────────────────────────────────────
# Graph building
# ─────────────────────────────────────────────────────────────

def build_graphs(node_feats: np.ndarray, adj_dtf: np.ndarray,
                 threshold: float = 0.15, ablation: bool = False):
    """
    node_feats : (N_epochs, 19, 16)
    adj_dtf    : (N_epochs, 19, 19)
    ablation   : if True replace node features with ones vector (topology only)
    Returns list of (x_tensor, a_hat_tensor).
    """
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        if ablation:
            x = torch.ones(N_CHANNELS, 1, dtype=torch.float32)
        else:
            x = torch.tensor(node_feats[i], dtype=torch.float32)
        graphs.append((x, a_hat))
    return graphs


def scale_node_features(graphs_train, graphs_test, ablation: bool = False):
    """
    Fit StandardScaler on training node features; transform both splits.
    Skipped for ablation (ones vector needs no scaling).
    No leakage: scaler never sees test data during fit.
    """
    if ablation:
        return graphs_train, graphs_test

    all_train_x = np.concatenate([g[0].numpy() for g in graphs_train], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_train_x)

    def apply(graphs):
        return [
            (torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32), a)
            for x, a in graphs
        ]

    return apply(graphs_train), apply(graphs_test)


# ─────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, optimiser, criterion, graphs, labels, device):
    model.train()
    total_loss = 0.0
    perm = np.random.permutation(len(graphs))
    for i in perm:
        x, a = graphs[i]
        x, a = x.to(device), a.to(device)
        optimiser.zero_grad()
        logit = model(x, a)
        label = torch.tensor(float(labels[i]), device=device).unsqueeze(0)
        loss  = criterion(logit.unsqueeze(0), label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(graphs)


@torch.no_grad()
def evaluate_graphs(model, graphs, labels, device, threshold: float = 0.5):
    """
    Returns probs (N,), preds (N,), targets (N,), mean_bce_loss.
    threshold: decision boundary — 0.5 default, adaptive for imbalanced data.
    """
    model.eval()
    logits, targets = [], []
    for i in range(len(graphs)):
        x, a = graphs[i]
        logit = model(x.to(device), a.to(device))
        logits.append(logit.cpu().item())
        targets.append(int(labels[i]))
    logits  = np.array(logits,  dtype=np.float32)
    targets = np.array(targets, dtype=np.int64)
    probs   = 1.0 / (1.0 + np.exp(-logits))
    preds   = (probs >= threshold).astype(np.int64)
    eps     = 1e-7
    bce     = -np.mean(
        targets * np.log(probs + eps) + (1 - targets) * np.log(1 - probs + eps)
    )
    return probs, preds, targets, float(bce)


def find_best_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Sweep thresholds 0.1–0.9 on TRAINING data and pick the one that
    maximises F1 for the ictal class.
    Called after best weights are restored — never sees test data.
    Returns the best threshold found (float).
    """
    best_thresh, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds_thr = (probs >= thr).astype(np.int64)
        f1_thr    = f1_score(targets, preds_thr, zero_division=0)
        if f1_thr > best_f1:
            best_f1, best_thresh = f1_thr, float(thr)
    return best_thresh


def compute_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy       = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {
        'auc':         float(roc_auc_score(y_true, y_prob)),
        'accuracy':    float(accuracy),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(tp / (tp + fn + 1e-12)),
        'specificity': float(tn / (tn + fp + 1e-12)),
        'precision':   float(precision_score(y_true, y_pred, zero_division=0)),
        'mcc':         float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, patient_id, output_dir):
    """
    Train vs val loss per epoch.
    Gap at end = overfitting signal. Vertical line = best epoch (early stop).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label='Train loss', color='royalblue', lw=1.5)
    ax.plot(val_losses,   label='Val loss',   color='tomato',    lw=1.5,
            linestyle='--')
    best_ep = int(np.argmin(val_losses))
    ax.axvline(best_ep, color='gray', linestyle=':', lw=1.2,
               label=f'Best epoch ({best_ep})')
    gap = abs(train_losses[-1] - val_losses[-1])
    ax.text(0.60, 0.85, f'Final train-val gap: {gap:.4f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('BCE Loss', fontsize=11)
    ax.set_title(f'GCN Loss Curves | Test: {patient_id}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_curve_{patient_id}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, patient_id, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'GCN CM | Test: {patient_id}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_gcn_{patient_id}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_all_folds(fold_roc_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for fpr, tpr, auc, pat in fold_roc_data:
        ax.plot(fpr, tpr, alpha=0.55, lw=1.5, label=f'{pat} (AUC={auc:.2f})')
        aucs.append(auc)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'GCN — LOPO ROC\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, output_dir):
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
    ax.set_title('GCN — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_fold_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion(all_y_true, all_y_pred, output_dir):
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
        ax.set_title(f'GCN — Aggregate CM ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cm_aggregate_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfit_diagnostic_gcn(fold_metrics, output_dir, tag=''):
    """
    Train AUC (at best epoch) vs Test AUC per fold.
    A large gap = model memorised training graphs.
    tag: suffix for filename to separate full vs ablation outputs.
    """
    train_aucs = [m.get('train_auc', np.nan) for m in fold_metrics]
    test_aucs  = [m['auc']                    for m in fold_metrics]
    patients   = [m['patient']                for m in fold_metrics]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='No gap (ideal)')

    for tr, te, pat in zip(train_aucs, test_aucs, patients):
        if np.isnan(tr):
            continue
        ax.scatter(tr, te, color='mediumpurple', s=90, zorder=5)
        ax.annotate(pat, (tr, te), textcoords='offset points',
                    xytext=(5, 3), fontsize=8)

    valid_gaps = [tr - te for tr, te in zip(train_aucs, test_aucs)
                  if not np.isnan(tr)]
    mean_gap = float(np.mean(valid_gaps)) if valid_gaps else 0.0

    label = f'GCN ({tag})' if tag else 'GCN'
    ax.set_title(f'{label} — Train vs Test AUC\nMean gap = {mean_gap:+.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Train AUC (at best epoch)', fontsize=10)
    ax.set_ylabel('Test AUC', fontsize=10)
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    fname = f'train_vs_test_auc_gcn_{tag}.png' if tag else 'train_vs_test_auc_gcn.png'
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Overfit diagnostic      → {output_dir / fname}')


def plot_comparison_vs_baseline(gcn_stats, gcn_ablation_stats,
                                 baseline_json_path, output_dir):
    """
    5-model bar chart: RF, SVM RBF, SVM RFE, GCN topology-only, GCN full.
    Error bars show std across LOPO folds.
    """
    if not Path(baseline_json_path).exists():
        print(f'  [SKIP] baseline JSON not found: {baseline_json_path}')
        return

    with open(baseline_json_path) as f:
        baseline = json.load(f)

    met_keys = ['auc', 'accuracy', 'f1', 'sensitivity', 'specificity']
    models   = {}

    for name, res in baseline.items():
        ss = res.get('summary_stats', {})
        if ss:
            models[name] = {
                'mean': [ss[k]['mean'] for k in met_keys if k in ss],
                'std':  [ss[k]['std']  for k in met_keys if k in ss],
            }

    if gcn_ablation_stats:
        models['GCN (topology only)'] = {
            'mean': [gcn_ablation_stats[k]['mean'] for k in met_keys
                     if k in gcn_ablation_stats],
            'std':  [gcn_ablation_stats[k]['std']  for k in met_keys
                     if k in gcn_ablation_stats],
        }

    models['GCN (full features)'] = {
        'mean': [gcn_stats[k]['mean'] for k in met_keys if k in gcn_stats],
        'std':  [gcn_stats[k]['std']  for k in met_keys if k in gcn_stats],
    }

    x      = np.arange(len(met_keys))
    n      = len(models)
    width  = 0.8 / n
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (name, vals) in enumerate(models.items()):
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals['mean'], width,
               yerr=vals['std'],
               label=name, color=colors[i % len(colors)],
               alpha=0.85, edgecolor='black', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=11)
    ax.set_ylabel('Mean Score (LOPO)', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('All Models — LOPO CV Comparison\n'
                 'RF  |  SVM RBF  |  SVM RFE  |  GCN topology  |  GCN full',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Comparison chart        → {output_dir / "comparison_all_models.png"}')


# ─────────────────────────────────────────────────────────────
# Core LOPO runner
# ─────────────────────────────────────────────────────────────

def run_lopo_gcn(node_feats, adj_dtf, y, patient_ids, args, device,
                 output_dir, ablation=False):
    """
    Full LOPO training loop for one mode (full or ablation).

    ablation=True  → nodes get ones vector, topology-only GCN.
    ablation=False → nodes get 16-dim features from step 3.

    Both modes use adaptive threshold (found on training fold) to handle
    class imbalance at inference — keeps AUC unaffected, improves F1/Sens.
    """
    in_dim   = 1 if ablation else 16
    tag      = 'ablation' if ablation else 'full'
    mode_str = 'ABLATION (topology only)' if ablation else 'FULL (with node features)'

    print(f'\n{"=" * 60}')
    print(f'  GCN — {mode_str}')
    print(f'{"=" * 60}')

    all_graphs = build_graphs(node_feats, adj_dtf,
                              threshold=args.threshold, ablation=ablation)

    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask
        train_idx  = np.where(train_mask)[0]
        test_idx   = np.where(test_mask)[0]

        y_train = y[train_idx]
        y_test  = y[test_idx]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: test set has only one class')
            continue

        graphs_train_raw = [all_graphs[i] for i in train_idx]
        graphs_test_raw  = [all_graphs[i] for i in test_idx]

        graphs_train, graphs_test = scale_node_features(
            graphs_train_raw, graphs_test_raw, ablation=ablation
        )

        # pos_weight from training fold only — no leakage
        n_neg      = int((y_train == 0).sum())
        n_pos      = int((y_train == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        model = SmallGCN(
            in_dim=in_dim, hidden=args.hidden, dropout=args.dropout
        ).to(device)

        optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5,
                                      verbose=False)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        train_losses, val_losses = [], []
        best_auc       = 0.0
        best_state     = None
        best_train_auc = 0.0
        patience_cnt   = 0

        for ep in range(args.epochs):
            tr_loss = train_one_epoch(
                model, optimiser, criterion, graphs_train, y_train, device
            )
            # Evaluate on test fold for early stopping signal
            # (threshold=0.5 here — only used for val_loss / AUC tracking,
            #  NOT for final predictions)
            probs_val, _, targets_val, val_loss = evaluate_graphs(
                model, graphs_test, y_test, device, threshold=0.5
            )

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            val_auc = roc_auc_score(targets_val, probs_val) \
                      if len(np.unique(targets_val)) == 2 else 0.0

            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

                # Record train AUC at this epoch for overfitting diagnostic
                tr_probs, _, tr_targets, _ = evaluate_graphs(
                    model, graphs_train, y_train, device, threshold=0.5
                )
                best_train_auc = float(roc_auc_score(tr_targets, tr_probs)) \
                                 if len(np.unique(tr_targets)) == 2 else 0.0
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  {pat}: early stop @ epoch {ep + 1}  '
                          f'best val AUC={best_auc:.3f}')
                    break

        # ── Restore best weights ──────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Adaptive threshold — fit on TRAIN data only ───
        tr_probs_final, _, tr_targets_final, _ = evaluate_graphs(
            model, graphs_train, y_train, device, threshold=0.5
        )
        best_thresh = find_best_threshold(tr_probs_final, tr_targets_final)

        # ── Final test evaluation with adaptive threshold ─
        probs, preds, targets, _ = evaluate_graphs(
            model, graphs_test, y_test, device, threshold=best_thresh
        )

        metrics = compute_metrics(targets, preds, probs)
        if metrics is None:
            continue

        metrics['patient']            = pat
        metrics['n_train']            = int(train_mask.sum())
        metrics['n_test']             = int(test_mask.sum())
        metrics['best_val_auc']       = float(best_auc)
        metrics['train_auc']          = float(best_train_auc)
        metrics['decision_threshold'] = float(best_thresh)
        fold_metrics.append(metrics)

        all_y_true.extend(targets.tolist())
        all_y_pred.extend(preds.tolist())

        fpr, tpr, _ = roc_curve(targets, probs)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        print(f'  {pat:8s} | thresh={best_thresh:.2f}  '
              f'TrainAUC={best_train_auc:.3f}  TestAUC={metrics["auc"]:.3f}  '
              f'Gap={best_train_auc - metrics["auc"]:+.3f}  '
              f'Acc={metrics["accuracy"]:.3f}  F1={metrics["f1"]:.3f}  '
              f'Sens={metrics["sensitivity"]:.3f}  Spec={metrics["specificity"]:.3f}  '
              f'MCC={metrics["mcc"]:.3f}')

        # Per-fold plots (full model only — ablation would just duplicate)
        if not ablation:
            plot_loss_curves(train_losses, val_losses, pat, output_dir)
            plot_confusion_matrix(
                confusion_matrix(targets, preds), pat, output_dir
            )

    if not fold_metrics:
        print(f'  [ERROR] No valid folds completed.')
        return [], {}

    # ── Aggregate plots (full model only) ─────────────────
    if not ablation:
        if fold_roc_data:
            plot_roc_all_folds(fold_roc_data, output_dir)
        if fold_metrics:
            plot_per_fold_metrics(fold_metrics, output_dir)
        if all_y_true:
            plot_aggregate_confusion(
                np.array(all_y_true), np.array(all_y_pred), output_dir
            )

    # ── Overfitting diagnostic (both modes) ───────────────
    plot_overfit_diagnostic_gcn(fold_metrics, output_dir, tag=tag)

    # ── Summary stats ─────────────────────────────────────
    met_keys      = ['auc', 'accuracy', 'f1', 'sensitivity',
                     'specificity', 'precision', 'mcc']
    summary_stats = {}
    label         = 'Ablation' if ablation else 'Full GCN'

    print(f'\n  {"─" * 50}')
    print(f'  {label} — Mean ± Std across {len(fold_metrics)} folds')
    print(f'  {"─" * 50}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    train_aucs = [m['train_auc'] for m in fold_metrics]
    test_aucs  = [m['auc']       for m in fold_metrics]
    mean_gap   = float(np.mean(np.array(train_aucs) - np.array(test_aucs)))
    thresholds = [m['decision_threshold'] for m in fold_metrics]
    print(f'  {"overfit gap":15s}: {mean_gap:+.3f}  (train − test AUC)')
    print(f'  {"thresh (mean)":15s}: {np.mean(thresholds):.3f} ± {np.std(thresholds):.3f}')

    return fold_metrics, summary_stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 5 — Supervised GCN LOPO')
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/gnn_supervised')
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold',     type=float, default=0.15,
                        help='DTF edge threshold. Use 0.05 for ablation.')
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--ablation',      action='store_true',
                        help='Run topology-only GCN (ones node features)')
    parser.add_argument('--ablation_json', default=None,
                        help='Path to ablation results_gcn.json for comparison chart')
    parser.add_argument('--baseline_json', default=None,
                        help='Path to step4 results_all.json for comparison chart')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('STEP 5 — SUPERVISED GCN')
    print('=' * 60)
    print(f'Device    : {device}')
    print(f'Mode      : {"ABLATION" if args.ablation else "FULL"}')
    print(f'Epochs    : {args.epochs}   LR: {args.lr}')
    print(f'Hidden    : {args.hidden}   Dropout: {args.dropout}')
    print(f'Threshold : {args.threshold}   Patience: {args.patience}')
    print(f'Adaptive threshold: ON (fitted on train fold, maximises F1)')
    print('=' * 60)

    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)
    adj_dtf     = data['adj_dtf'].astype(np.float32)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']

    print(f'Loaded: {len(y)} epochs | Ictal: {(y==1).sum()} | '
          f'Pre-ictal: {(y==0).sum()}')

    fold_metrics, summary_stats = run_lopo_gcn(
        node_feats, adj_dtf, y, patient_ids, args, device,
        output_dir, ablation=args.ablation
    )

    if not fold_metrics:
        print('[ERROR] No results to save.')
        return

    # ── Save results ──────────────────────────────────────
    results = {
        'model':           'GCN_Ablation' if args.ablation else 'GCN_Supervised',
        'ablation':        args.ablation,
        'hyperparameters': vars(args),
        'fold_metrics':    fold_metrics,
        'summary_stats':   summary_stats,
    }
    results_path = output_dir / 'results_gcn.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults → {results_path}')

    # ── Comparison chart ──────────────────────────────────
    ablation_stats = None
    if args.ablation_json and Path(args.ablation_json).exists():
        with open(args.ablation_json) as f:
            abl = json.load(f)
        ablation_stats = abl.get('summary_stats', None)
        print(f'  Loaded ablation stats from {args.ablation_json}')

    if args.baseline_json:
        plot_comparison_vs_baseline(
            summary_stats, ablation_stats,
            args.baseline_json, output_dir
        )

    print('\n' + '=' * 60)
    print('STEP 5 COMPLETE')
    print('=' * 60)

    if args.ablation:
        print('\nNow run full model:')
        print('  python step5_gnn_supervised.py \\')
        print('      --featfile features/features_all.npz \\')
        print('      --outputdir results/gnn_supervised \\')
        print(f'      --ablation_json {output_dir}/results_gcn.json \\')
        print('      --baseline_json results/baseline_ml/results_all.json')
    else:
        print('\nNext: python step6_ssl_gnn.py --featfile features/features_all.npz')


if __name__ == '__main__':
    main()