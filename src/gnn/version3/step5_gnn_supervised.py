"""
Step 5 - Supervised GNN (2-layer GCN)
======================================
- Graph per epoch: 19 nodes (EEG channels), edges = thresholded DTF
- Node features: (19, 16) from step3
- Architecture: GCN(16→32→64) → GlobalMeanPool → Linear(2)
- Training: Leave-One-Patient-Out (LOPO) CV
- Loss: BCEWithLogitsLoss with pos_weight for imbalance
- Metrics: AUC, F1, Sensitivity, Specificity, MCC
- Outputs: per-fold metrics, ROC curves, confusion matrices,
           loss curves (overfit check), final comparison vs baseline

Usage:
  python step5_gnn_supervised.py \
      --featfile  features/features_all.npz \
      --outputdir results/gnn_supervised \
      --epochs    100 \
      --lr        0.001 \
      --hidden    32 \
      --threshold 0.15
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, precision_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Simple GCN implementation (no PyG dependency)
# Uses manual sparse-style message passing: A_hat @ X @ W
# A_hat = D^{-1/2} (A+I) D^{-1/2}  (symmetric normalized)
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(adj):
    """
    adj: (N, N) numpy array, edge weights
    Returns normalized adjacency A_hat as torch tensor (N, N)
    """
    A = adj + np.eye(adj.shape[0])           # add self-loops
    D = np.diag(A.sum(axis=1) ** -0.5)
    A_hat = D @ A @ D
    return torch.tensor(A_hat, dtype=torch.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, a_hat):
        # x: (N, in_dim)   a_hat: (N, N)
        return F.relu(self.W(a_hat @ x))


class GCN(nn.Module):
    """
    2-layer GCN → GlobalMeanPool → MLP classifier
    Architecture kept deliberately small to avoid overfitting.
    """
    def __init__(self, in_dim=16, hidden=32, out_dim=64, dropout=0.3):
        super().__init__()
        self.gcn1   = GCNLayer(in_dim, hidden)
        self.gcn2   = GCNLayer(hidden, out_dim)
        self.drop   = nn.Dropout(dropout)
        self.head   = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),          # binary output (logit)
        )

    #def forward(self, x, a_hat):
        # x: (N, in_dim)   a_hat: (N, N)
        #h = self.gcn1(x, a_hat)        # (N, hidden)
        #h = self.drop(h)
        #h = self.gcn2(h, a_hat)        # (N, out_dim)
        #h = h.mean(dim=0, keepdim=True)  # global mean pool → (1, out_dim)
        #return self.head(h).squeeze()   # scalar logit
        #return self.head(h).squeeze(-1)  # squeezes only last dim → shape [1]

    def forward(self, x, a_hat):
        h = self.gcn1(x, a_hat)
        h = self.drop(h)
        h = self.gcn2(h, a_hat)
        h = h.mean(dim=0, keepdim=True)   # (1, out_dim)
        return self.head(h).squeeze()      # squeeze ALL → scalar []



# ─────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────

def build_graphs(node_feats, adj_dtf, threshold=0.15):
    """
    node_feats : (N_epochs, 19, 16)
    adj_dtf    : (N_epochs, 19, 19)
    threshold  : keep edges above this DTF value (sparsify)
    Returns list of (x_tensor, a_hat_tensor) per epoch
    """
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0          # sparsify
        np.fill_diagonal(adj, 0.0)          # remove self-loops (added back in normalize)
        a_hat = normalize_adjacency(adj)    # (19,19)
        x     = torch.tensor(node_feats[i], dtype=torch.float32)  # (19,16)
        graphs.append((x, a_hat))
    return graphs


def scale_node_features(graphs_train, graphs_test):
    """
    Fit StandardScaler on train node features, apply to both.
    Operates per-feature across all nodes and epochs.
    """
    # Collect all train node features: (N_train * 19, 16)
    all_train = np.concatenate(
        [g[0].numpy() for g in graphs_train], axis=0
    )
    scaler = StandardScaler()
    scaler.fit(all_train)

    def apply(graphs):
        scaled = []
        for x, a in graphs:
            x_sc = torch.tensor(
                scaler.transform(x.numpy()), dtype=torch.float32
            )
            scaled.append((x_sc, a))
        return scaled

    return apply(graphs_train), apply(graphs_test), scaler


# ─────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────

def train_epoch(model, optimizer, graphs, labels, pos_weight, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    total_loss = 0.0
    indices = np.random.permutation(len(graphs))
    for i in indices:
        x, a = graphs[i]
        x, a = x.to(device), a.to(device)
        optimizer.zero_grad()
        logit = model(x, a)                                              # scalar []
        label = torch.tensor(float(labels[i]), device=device).view(1)   # [1]
        loss  = criterion(logit.view(1), label)
        #logit = model(x, a)
        #loss  = criterion(logit, torch.tensor(float(labels[i]), device=device))
        #loss = criterion(logit.unsqueeze(0), torch.tensor([float(labels[i])], device=device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(graphs)


@torch.no_grad()
def evaluate(model, graphs, labels, device):
    model.eval()
    logits, targets = [], []
    for i in range(len(graphs)):
        x, a = graphs[i]
        logit = model(x.to(device), a.to(device))
        logits.append(logit.cpu().item())
        targets.append(int(labels[i]))
    logits  = np.array(logits)
    targets = np.array(targets)
    probs   = 1 / (1 + np.exp(-logits))     # sigmoid
    preds   = (probs >= 0.5).astype(int)
    loss    = float(np.mean(
        [max(0, 1 - t * l + (1 - t) * l) for t, l in zip(targets, logits)]
    ))  # approximate, just for tracking
    return probs, preds, targets, loss


def compute_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    cm = confusion_matrix(y_true, y_pred)
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
    ax.plot(train_losses, label='Train loss', color='royalblue')
    ax.plot(val_losses,   label='Val loss (test patient)', color='tomato', linestyle='--')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'GCN Loss Curves | Test: {patient_id}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate gap at final epoch
    gap = abs(train_losses[-1] - val_losses[-1])
    ax.text(0.65, 0.85, f'Final gap: {gap:.4f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
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
    ax.set_title(f'GCN Confusion Matrix | Test: {patient_id}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_gcn_{patient_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_all_folds(fold_roc_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for fpr, tpr, auc, pat in fold_roc_data:
        ax.plot(fpr, tpr, alpha=0.6, lw=1.5, label=f'{pat} (AUC={auc:.2f})')
        aucs.append(auc)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'GCN — LOPO ROC Curves\nMean AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}',
        fontsize=12, fontweight='bold'
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
    x = np.arange(len(patients))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (met, col) in enumerate(zip(met_keys, colors)):
        vals = [m[met] for m in fold_metrics]
        ax.bar(x + i * width, vals, width, label=met.upper(),
               color=col, alpha=0.8, edgecolor='black')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
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
        (axes[0], cm,      'd',    'Counts'),
        (axes[1], cm_norm, '.2f',  'Normalized'),
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


def plot_comparison_vs_baseline(gcn_stats, baseline_path, output_dir):
    """
    Overlay GCN results against RF and SVM from step4 results_all.json
    """
    if not Path(baseline_path).exists():
        print('  [SKIP] baseline results_all.json not found for comparison plot')
        return

    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    models   = {}
    for model_name, res in baseline.items():
        models[model_name] = {k: res['summary_stats'][k]['mean'] for k in met_keys}
    models['GCN (Supervised)'] = {k: gcn_stats[k]['mean'] for k in met_keys}

    x      = np.arange(len(met_keys))
    width  = 0.25
    colors = ['steelblue', 'tomato', 'seagreen']
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, vals) in enumerate(models.items()):
        means = [vals[k] for k in met_keys]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=model_name,
               color=colors[i % len(colors)], alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Mean Score (LOPO)', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_title('Model Comparison: RF vs SVM vs GCN (LOPO CV)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Comparison chart vs baseline saved.')


# ─────────────────────────────────────────────────────────────
# Main LOPO loop
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 5 - Supervised GCN LOPO')
    parser.add_argument('--featfile',     required=True)
    parser.add_argument('--outputdir',    default='results/gnn_supervised')
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--lr',           type=float, default=0.001)
    parser.add_argument('--hidden',       type=int,   default=32)
    parser.add_argument('--outdim',       type=int,   default=64)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--threshold',    type=float, default=0.15,
                        help='DTF edge threshold for graph sparsification')
    parser.add_argument('--patience',     type=int,   default=20,
                        help='Early stopping patience (epochs)')
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
    print(f'Epochs    : {args.epochs}  |  LR: {args.lr}  |  Hidden: {args.hidden}')
    print(f'Dropout   : {args.dropout}  |  DTF threshold: {args.threshold}')
    print(f'Early stop: patience={args.patience}')
    print('=' * 60)

    # ── Load features ────────────────────────────────────────
    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)  # (N, 19, 16)
    adj_dtf     = data['adj_dtf'].astype(np.float32)        # (N, 19, 19)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']
    subject_ids = data['subject_ids']

    print(f'Loaded: {len(y)} epochs  |  Ictal: {(y==1).sum()}  Pre-ictal: {(y==0).sum()}')

    # ── Build graphs ─────────────────────────────────────────
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

        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]

        y_train = y[train_idx]
        y_test  = y[test_idx]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: test set has only one class')
            continue

        graphs_train = [all_graphs[i] for i in train_idx]
        graphs_test  = [all_graphs[i] for i in test_idx]

        # Scale node features (fit on train only)
        graphs_train, graphs_test, _ = scale_node_features(graphs_train, graphs_test)

        # Compute pos_weight for imbalanced loss
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        # Build model fresh for each fold
        model = GCN(
            in_dim=16,
            hidden=args.hidden,
            out_dim=args.outdim,
            dropout=args.dropout,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=False)

        # ── Training loop ─────────────────────────────────
        train_losses, val_losses = [], []
        best_auc     = 0.0
        best_state   = None
        patience_cnt = 0

        for ep in range(args.epochs):
            tr_loss = train_epoch(model, optimizer, graphs_train, y_train, pos_weight, device)
            probs, preds, targets, _ = evaluate(model, graphs_test, y_test, device)

            if len(np.unique(targets)) == 2:
                val_auc = roc_auc_score(targets, probs)
            else:
                val_auc = 0.0

            # Approximate val loss for plotting
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=device)
            )
            with torch.no_grad():
                val_logits = []
                for i in range(len(graphs_test)):
                    x, a = graphs_test[i]
                    val_logits.append(model(x.to(device), a.to(device)).cpu().item())
            #val_logits_t = torch.tensor(val_logits)
            #val_labels_t = torch.tensor(y_test, dtype=torch.float32)
            #val_loss = criterion(
            #    val_logits_t,
            #    val_labels_t
            #).item()
            #val_logits_t = torch.tensor(val_logits).unsqueeze(1)   # (N, 1)
            #val_labels_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # (N, 1)
            #val_loss = nn.BCEWithLogitsLoss(
            #    pos_weight=torch.tensor([pos_weight])
            #)(val_logits_t, val_labels_t).item()
            val_logits_t = torch.tensor(val_logits)               # (N,)
            val_labels_t = torch.tensor(y_test, dtype=torch.float32)  # (N,)
            val_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )(val_logits_t, val_labels_t).item()



            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            # Early stopping on AUC
            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  {pat}: early stop at epoch {ep+1}  best AUC={best_auc:.3f}')
                    break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Final evaluation ──────────────────────────────
        probs, preds, targets, _ = evaluate(model, graphs_test, y_test, device)
        metrics = compute_metrics(targets, preds, probs)
        if metrics is None:
            continue

        metrics['patient']  = pat
        metrics['n_train']  = int(train_mask.sum())
        metrics['n_test']   = int(test_mask.sum())
        fold_metrics.append(metrics)

        all_y_true.extend(targets.tolist())
        all_y_pred.extend(preds.tolist())

        fpr, tpr, _ = roc_curve(targets, probs)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        print(f'  {pat:8s} | AUC={metrics["auc"]:.3f}  F1={metrics["f1"]:.3f}'
              f'  Sens={metrics["sensitivity"]:.3f}  Spec={metrics["specificity"]:.3f}'
              f'  MCC={metrics["mcc"]:.3f}')

        # Per-fold plots
        plot_loss_curves(train_losses, val_losses, pat, output_dir)
        plot_confusion_matrix(confusion_matrix(targets, preds), pat, output_dir)

    # ── Aggregate plots ───────────────────────────────────────
    plot_roc_all_folds(fold_roc_data, output_dir)
    plot_per_fold_metrics(fold_metrics, output_dir)
    plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred), output_dir)

    # ── Summary ───────────────────────────────────────────────
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'precision', 'mcc']
    summary_stats = {}
    print(f'\n{"="*60}')
    print(f'GCN — Mean +/- Std across {len(fold_metrics)} folds')
    print(f'{"="*60}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = np.mean(vals), np.std(vals)
        summary_stats[k] = {'mean': float(mean_), 'std': float(std_)}
        print(f'  {k:15s}: {mean_:.3f} +/- {std_:.3f}')

    # ── Save results ──────────────────────────────────────────
    results = {
        'model': 'GCN_Supervised',
        'hyperparameters': vars(args),
        'fold_metrics':    fold_metrics,
        'summary_stats':   summary_stats,
    }
    results_path = output_dir / 'results_gcn.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {results_path}')

    # Per-fold bar chart
    plot_per_fold_metrics(fold_metrics, output_dir)

    # Comparison vs baseline (if provided)
    if args.baseline_json:
        plot_comparison_vs_baseline(summary_stats, args.baseline_json, output_dir)

    print('\n' + '=' * 60)
    print('STEP 5 COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
