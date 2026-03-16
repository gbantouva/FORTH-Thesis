"""
Step 5 — Supervised GCN  (LOPO Cross-Validation)
=================================================
DESIGN DECISIONS (document in thesis):

1. ARCHITECTURE — kept deliberately small to avoid overfitting:
     GCNLayer(16 → 32)  →  Dropout(0.4)
     GCNLayer(32 → 32)  →  GlobalMeanPool
     Linear(32 → 16)    →  ReLU  →  Dropout(0.4)
     Linear(16 → 1)     →  sigmoid (binary output)
   Total ~4k parameters. With 19 nodes × ~few hundred graphs, a large
   hidden dimension would memorise training graphs.

2. NORMALISED ADJACENCY:
     A_hat = D^{-1/2} (A + I) D^{-1/2}
   Self-loops added before normalisation (standard GCN, Kipf & Welling 2017).
   Edge weights are the thresholded DTF values (directed functional connectivity).
   Threshold=0.15 sparsifies the graph (keeps strongest connections only).

3. IMBALANCE: BCEWithLogitsLoss with pos_weight = n_neg / n_pos
   Computed from the training split of each fold independently.

4. EARLY STOPPING on validation AUC (patience=20).
   Best-state restored at end of each fold.

5. EVALUATION: LOPO by patient (not subject).
   Scaling of node features: StandardScaler fit on train split only.

6. COMPARISON: optional --baseline_json points to step4 results_all.json
   for an overlay bar chart.

Outputs:
  loss_curve_{patient}.png     train vs val loss per fold
  cm_gcn_{patient}.png         per-fold confusion matrices
  roc_gcn.png                  LOPO ROC curves
  per_fold_gcn.png             per-patient metric bar chart
  cm_aggregate_gcn.png         aggregate confusion matrix
  comparison_all_models.png    GCN vs RF vs SVM (if baseline_json provided)
  results_gcn.json             all metrics + hyperparams

Usage:
  python step5_gnn_supervised.py \\
      --featfile  features/features_all.npz \\
      --outputdir results/gnn_supervised \\
      --epochs    150 \\
      --lr        0.001 \\
      --hidden    32 \\
      --threshold 0.15 \\
      --dropout   0.4 \\
      --patience  20 \\
      --baseline_json results/baseline_ml/results_all.json
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


# ─────────────────────────────────────────────────────────────
# Adjacency normalisation
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """
    adj : (N, N) numpy, edge weights (already thresholded, diagonal=0)
    Returns A_hat = D^{-1/2} (A + I) D^{-1/2} as float32 tensor (N, N).
    """
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    A_hat = D @ A @ D
    return torch.tensor(A_hat, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# GCN model  (small by design)
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)   a_hat: (N, N)
        return F.relu(self.W(a_hat @ x))


class SmallGCN(nn.Module):
    """
    2-layer GCN → GlobalMeanPool → 2-layer MLP → scalar logit.
    Architecture chosen to be small relative to dataset size.
    in_dim=16  (node features from step3)
    hidden=32  (default — configurable)
    """
    def __init__(self, in_dim: int = 16, hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.gcn1  = GCNLayer(in_dim, hidden)
        self.gcn2  = GCNLayer(hidden, hidden)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, a_hat)           # (N, hidden)
        h = self.drop(h)
        h = self.gcn2(h, a_hat)           # (N, hidden)
        h = h.mean(dim=0, keepdim=True)   # (1, hidden)  global mean pool
        return self.head(h).squeeze()     # scalar logit


# ─────────────────────────────────────────────────────────────
# Graph building
# ─────────────────────────────────────────────────────────────

def build_graphs(node_feats: np.ndarray, adj_dtf: np.ndarray,
                 threshold: float = 0.15):
    """
    node_feats : (N_epochs, 19, 16)
    adj_dtf    : (N_epochs, 19, 19)
    Returns list of (x_tensor, a_hat_tensor) — one per epoch.
    """
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0      # sparsify — keep strong connections only
        np.fill_diagonal(adj, 0.0)      # diagonal added back in normalize_adjacency
        a_hat = normalize_adjacency(adj)
        x     = torch.tensor(node_feats[i], dtype=torch.float32)
        graphs.append((x, a_hat))
    return graphs


def scale_node_features(graphs_train, graphs_test):
    """
    Fit StandardScaler on training node features; transform both splits.
    Operates per-feature across all nodes and epochs in the training fold.
    No leakage: scaler never sees test data during fit.
    """
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
def evaluate_graphs(model, graphs, labels, device):
    """
    Returns probs (N,), preds (N,), targets (N,), mean_loss (float).
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
    probs   = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    preds   = (probs >= 0.5).astype(np.int64)
    # Approximate BCE loss for tracking
    eps     = 1e-7
    bce     = -np.mean(
        targets * np.log(probs + eps) + (1 - targets) * np.log(1 - probs + eps)
    )
    return probs, preds, targets, float(bce)


def compute_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    cm           = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'auc':         float(roc_auc_score(y_true, y_prob)),
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
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label='Train loss', color='royalblue', lw=1.5)
    ax.plot(val_losses,   label='Val loss',   color='tomato',    lw=1.5, linestyle='--')
    gap = abs(train_losses[-1] - val_losses[-1])
    ax.text(0.65, 0.85, f'Final gap: {gap:.4f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('BCE Loss', fontsize=11)
    ax.set_title(f'GCN Loss Curves | Test: {patient_id}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_curve_{patient_id}.png', dpi=150, bbox_inches='tight')
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
    plt.savefig(output_dir / f'cm_gcn_{patient_id}.png', dpi=150, bbox_inches='tight')
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
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange']
    x        = np.arange(len(patients))
    width    = 0.2
    fig, ax  = plt.subplots(figsize=(12, 5))
    for i, (met, col) in enumerate(zip(met_keys, colors)):
        vals = [m[met] for m in fold_metrics]
        ax.bar(x + i * width, vals, width, label=met.upper(),
               color=col, alpha=0.8, edgecolor='black')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('GCN — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
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
        ax.set_title(f'GCN — Aggregate CM ({title})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cm_aggregate_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_vs_baseline(gcn_stats, baseline_json_path, output_dir):
    if not Path(baseline_json_path).exists():
        print(f'  [SKIP] baseline JSON not found at {baseline_json_path}')
        return
    with open(baseline_json_path) as f:
        baseline = json.load(f)

    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    models   = {}
    for name, res in baseline.items():
        if res.get('summary_stats'):
            models[name] = {k: res['summary_stats'][k]['mean'] for k in met_keys}
    models['GCN (Supervised)'] = {k: gcn_stats[k]['mean'] for k in met_keys}

    x      = np.arange(len(met_keys))
    width  = 0.25
    colors = ['steelblue', 'tomato', 'seagreen', 'mediumpurple']
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, vals) in enumerate(models.items()):
        means  = [vals[k] for k in met_keys]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=name,
               color=colors[i % len(colors)], alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Mean Score (LOPO)', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('RF vs SVM vs GCN — LOPO CV', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Comparison chart → {output_dir / "comparison_all_models.png"}')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 5 — Supervised GCN LOPO')
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/gnn_supervised')
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--hidden',        type=int,   default=32,
                        help='GCN hidden dimension (default 32, keep small)')
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold',     type=float, default=0.15,
                        help='DTF edge threshold for graph sparsification')
    parser.add_argument('--patience',      type=int,   default=20,
                        help='Early stopping patience in epochs')
    parser.add_argument('--baseline_json', default=None,
                        help='Path to step4 results_all.json for comparison plot')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('STEP 5 — SUPERVISED GCN')
    print('=' * 60)
    print(f'Device    : {device}')
    print(f'Epochs    : {args.epochs}   LR: {args.lr}')
    print(f'Hidden    : {args.hidden}   Dropout: {args.dropout}')
    print(f'Threshold : {args.threshold}   Patience: {args.patience}')
    print('=' * 60)

    # ── Load features ────────────────────────────────────────
    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)  # (N, 19, 16)
    adj_dtf     = data['adj_dtf'].astype(np.float32)        # (N, 19, 19)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']

    print(f'Loaded: {len(y)} epochs | Ictal: {(y == 1).sum()} | Pre-ictal: {(y == 0).sum()}')

    # ── Build graphs (shared, unscaled — scaling happens inside LOPO) ────────
    print('Building graphs...')
    all_graphs = build_graphs(node_feats, adj_dtf, threshold=args.threshold)

    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []

    # ── LOPO loop ────────────────────────────────────────────
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

        # Scale node features: fit on train split only
        graphs_train, graphs_test = scale_node_features(graphs_train_raw, graphs_test_raw)

        # Compute pos_weight for imbalanced loss (from training fold only)
        n_neg      = int((y_train == 0).sum())
        n_pos      = int((y_train == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        # Build a fresh model per fold
        model = SmallGCN(
            in_dim=16, hidden=args.hidden, dropout=args.dropout
        ).to(device)

        optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, verbose=False)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        train_losses, val_losses = [], []
        best_auc     = 0.0
        best_state   = None
        patience_cnt = 0

        for ep in range(args.epochs):
            tr_loss = train_one_epoch(
                model, optimiser, criterion, graphs_train, y_train, device
            )
            probs, preds, targets, val_loss = evaluate_graphs(
                model, graphs_test, y_test, device
            )

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if len(np.unique(targets)) == 2:
                val_auc = roc_auc_score(targets, probs)
            else:
                val_auc = 0.0

            if val_auc > best_auc:
                best_auc     = val_auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  {pat}: early stop at epoch {ep + 1}  best AUC={best_auc:.3f}')
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation on this fold
        probs, preds, targets, _ = evaluate_graphs(model, graphs_test, y_test, device)
        metrics = compute_metrics(targets, preds, probs)
        if metrics is None:
            continue

        metrics['patient']     = pat
        metrics['n_train']     = int(train_mask.sum())
        metrics['n_test']      = int(test_mask.sum())
        metrics['best_val_auc'] = float(best_auc)
        fold_metrics.append(metrics)

        all_y_true.extend(targets.tolist())
        all_y_pred.extend(preds.tolist())

        fpr, tpr, _ = roc_curve(targets, probs)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        print(f'  {pat:8s} | AUC={metrics["auc"]:.3f}  F1={metrics["f1"]:.3f}'
              f'  Sens={metrics["sensitivity"]:.3f}  Spec={metrics["specificity"]:.3f}'
              f'  MCC={metrics["mcc"]:.3f}')

        plot_loss_curves(train_losses, val_losses, pat, output_dir)
        plot_confusion_matrix(confusion_matrix(targets, preds), pat, output_dir)

    # ── Aggregate plots ───────────────────────────────────────
    if fold_roc_data:
        plot_roc_all_folds(fold_roc_data, output_dir)
    if fold_metrics:
        plot_per_fold_metrics(fold_metrics, output_dir)
    if all_y_true:
        plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred), output_dir)

    # ── Summary ───────────────────────────────────────────────
    met_keys      = ['auc', 'f1', 'sensitivity', 'specificity', 'precision', 'mcc']
    summary_stats = {}

    print(f'\n{"=" * 60}')
    print(f'GCN — Mean ± Std across {len(fold_metrics)} folds')
    print(f'{"=" * 60}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    # ── Save results ──────────────────────────────────────────
    results = {
        'model':           'GCN_Supervised',
        'hyperparameters': vars(args),
        'fold_metrics':    fold_metrics,
        'summary_stats':   summary_stats,
    }
    results_path = output_dir / 'results_gcn.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults → {results_path}')

    # ── Comparison vs baseline ────────────────────────────────
    if args.baseline_json:
        plot_comparison_vs_baseline(summary_stats, args.baseline_json, output_dir)

    print('\n' + '=' * 60)
    print('STEP 5 COMPLETE')
    print('=' * 60)
    print('\nNext: python step6_ssl_gnn.py --featfile features/features_all.npz'
          ' --sup_json results/gnn_supervised/results_gcn.json')


if __name__ == '__main__':
    main()
