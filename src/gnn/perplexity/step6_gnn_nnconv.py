"""
Step 6 v2 — GNN with NNConv (uses edge attributes DTF + PDC)
=============================================================
Key difference from step6 v1:
  SAGEConv ignored edge_attr entirely.
  NNConv uses edge_attr in message passing:
    m_ij = MLP(edge_attr_ij) @ x_j
  So both DTF and PDC weights now influence how node j's
  features are aggregated into node i's embedding.

Architecture:
  NNConv(24->64) -> NNConv(64->64) -> NNConv(64->64)
  -> global mean pool -> MLP(64->32->2)

Input : graphs/all_graphs_dtf_topk6.pt  (from step5 v2)
Output: results_nnconv/
          metrics_summary.csv  fold_metrics.csv
          confusion_matrix.png  metrics_per_fold.png
          roc_curve.png  training_loss.png

Usage:
  python step6_gnn_nnconv.py \
    --graphs  path/to/all_graphs_dtf_topk6.pt \
    --outdir  path/to/results_nnconv \
    --epochs  100 \
    --hidden  64
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn     import NNConv, global_mean_pool

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

EDGE_DIM = 2   # [dtf_integrated, pdc_integrated]


# ═══════════════════════════════════════════════════════════════
# 1. MODEL
# ═══════════════════════════════════════════════════════════════

class NNConvClassifier(nn.Module):
    """
    3-layer NNConv GNN that uses edge attributes in message passing.

    NNConv computes messages as:
        m_ij = MLP_k(edge_attr_ij) @ x_j
    where MLP_k maps edge_attr -> (in_channels * out_channels).
    This means DTF and PDC weights directly scale and rotate
    the source node's features before aggregation.

    After 3 layers: global mean pool -> MLP classifier.
    """
    def __init__(self, in_channels, hidden_channels,
                 edge_dim=2, dropout=0.3):
        super().__init__()

        # NNConv requires an MLP that maps
        # edge_attr (edge_dim,) -> (in_ch * out_ch,)
        def edge_nn(in_ch, out_ch):
            return nn.Sequential(
                nn.Linear(edge_dim, 32),
                nn.ReLU(),
                nn.Linear(32, in_ch * out_ch),
            )

        self.conv1 = NNConv(in_channels,      hidden_channels,
                            edge_nn(in_channels,      hidden_channels),
                            aggr='mean')
        self.conv2 = NNConv(hidden_channels,   hidden_channels,
                            edge_nn(hidden_channels,   hidden_channels),
                            aggr='mean')
        self.conv3 = NNConv(hidden_channels,   hidden_channels,
                            edge_nn(hidden_channels,   hidden_channels),
                            aggr='mean')

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)       # (B, hidden)
        return self.classifier(x)            # (B, 2)


# ═══════════════════════════════════════════════════════════════
# 2. METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_proba):
    cm           = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1           = (2 * precision * sensitivity / (precision + sensitivity)
                    if (precision + sensitivity) > 0 else 0.0)
    bal_acc      = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float('nan')
    return dict(sensitivity=sensitivity, specificity=specificity,
                precision=precision, f1_ictal=f1,
                bal_acc=bal_acc, auc=auc, cm=cm)


# ═══════════════════════════════════════════════════════════════
# 3. TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0.0
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index,
                       batch.edge_attr, batch.batch)
        loss   = criterion(logits, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_proba = [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index,
                       batch.edge_attr, batch.batch)
        proba  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_proba.extend(proba.tolist())
        all_labels.extend(batch.y.cpu().numpy().tolist())
    y_true = np.array(all_labels)
    y_prob = np.array(all_proba)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


# ═══════════════════════════════════════════════════════════════
# 4. LOPO
# ═══════════════════════════════════════════════════════════════

def run_lopo(graphs, args, out_dir, device):
    patients        = sorted(set(g.patient.item() for g in graphs))
    node_dim        = graphs[0].x.shape[1]
    fold_records    = []
    cm_total        = np.zeros((2, 2), dtype=int)
    all_y_true      = []
    all_y_proba     = []
    all_fold_losses = []

    print(f"\n{'='*65}")
    print(f"  NNConv GNN LOPO — {len(patients)} folds")
    print(f"  hidden={args.hidden}  epochs={args.epochs}  lr={args.lr}")
    print(f"  Edge attributes: [DTF, PDC] used in message passing")
    print(f"{'='*65}")

    for fold, test_pat in enumerate(patients):
        train_graphs = [g for g in graphs if g.patient.item() != test_pat]
        test_graphs  = [g for g in graphs if g.patient.item() == test_pat]

        y_tr   = torch.tensor([g.y.item() for g in train_graphs])
        n0, n1 = (y_tr == 0).sum().item(), (y_tr == 1).sum().item()
        total  = n0 + n1
        w0     = total / (2 * n0) if n0 > 0 else 1.0
        w1     = total / (2 * n1) if n1 > 0 else 1.0
        class_weights = torch.tensor([w0, w1], dtype=torch.float)

        train_loader = DataLoader(train_graphs,
                                  batch_size=args.batch_size,
                                  shuffle=True)
        test_loader  = DataLoader(test_graphs,
                                  batch_size=args.batch_size,
                                  shuffle=False)

        model     = NNConvClassifier(node_dim, args.hidden,
                                     EDGE_DIM, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=30, gamma=0.5)

        fold_losses = []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader,
                               optimizer, device, class_weights)
            fold_losses.append(loss)
            scheduler.step()

        all_fold_losses.append(fold_losses)

        y_true, y_pred, y_proba = evaluate(model, test_loader, device)
        m = compute_metrics(y_true, y_pred, y_proba)
        cm_total += m['cm']
        all_y_true.extend(y_true.tolist())
        all_y_proba.extend(y_proba.tolist())

        n_ict = int((y_true == 1).sum())
        n_pre = int((y_true == 0).sum())

        fold_records.append({
            'fold':        fold + 1,
            'patient':     int(test_pat),
            'n_test':      len(y_true),
            'n_ictal':     n_ict,
            'n_preictal':  n_pre,
            'sensitivity': m['sensitivity'],
            'specificity': m['specificity'],
            'precision':   m['precision'],
            'f1_ictal':    m['f1_ictal'],
            'bal_acc':     m['bal_acc'],
            'auc':         m['auc'],
            'final_loss':  fold_losses[-1],
        })

        print(f"  Fold {fold+1:2d} | patient={int(test_pat):2d} "
              f"| n={len(y_true):3d} "
              f"(ict={n_ict:2d} pre={n_pre:2d}) "
              f"| Sens={m['sensitivity']:.3f} "
              f"Spec={m['specificity']:.3f} "
              f"AUC={m['auc']:.3f} "
              f"loss={fold_losses[-1]:.4f}")

    # ── Summary ────────────────────────────────────────────────
    metric_keys = ['sensitivity', 'specificity', 'precision',
                   'f1_ictal', 'bal_acc', 'auc']
    summary_row = {'model': 'NNConv-GNN'}
    print(f"\n  {'─'*55}")
    print(f"  Summary ({len(fold_records)} folds):")
    for k in metric_keys:
        vals = [r[k] for r in fold_records if not np.isnan(r[k])]
        mu, sd = np.mean(vals), np.std(vals)
        summary_row[f'{k}_mean'] = mu
        summary_row[f'{k}_std']  = sd
        print(f"    {k:15s}: {mu:.3f} ± {sd:.3f}")

    print(f"\n  Pooled confusion matrix:")
    print(f"  {cm_total}")

    # ── Plots ──────────────────────────────────────────────────
    print(f"\n  Saving plots...")
    _plot_cm(cm_total, out_dir)
    _plot_per_fold(fold_records, out_dir)
    _plot_roc(np.array(all_y_true), np.array(all_y_proba),
              [r['auc'] for r in fold_records], out_dir)
    _plot_loss(all_fold_losses, out_dir)

    pd.DataFrame(fold_records).to_csv(
        out_dir / 'fold_metrics.csv', index=False, float_format='%.4f')
    pd.DataFrame([summary_row]).to_csv(
        out_dir / 'metrics_summary.csv', index=False, float_format='%.4f')
    print(f"  Saved: fold_metrics.csv  metrics_summary.csv")

    return summary_row


# ═══════════════════════════════════════════════════════════════
# 5. PLOTS
# ═══════════════════════════════════════════════════════════════

def _plot_cm(cm_total, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'],
                linewidths=0.5, linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title('Confusion Matrix — NNConv GNN\n(all LOPO folds pooled)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'confusion_matrix_NNConv.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_per_fold(fold_records, out_dir):
    df      = pd.DataFrame(fold_records)
    metrics = ['sensitivity', 'specificity', 'f1_ictal', 'bal_acc', 'auc']
    labels  = ['Sensitivity', 'Specificity', 'F1 (Ictal)',
               'Bal. Accuracy', 'AUC']
    colors  = ['#4C72B0', '#55A868', '#DD8452', '#C44E52', '#8172B2']

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    fig.suptitle('Per-Fold Metrics — NNConv GNN (LOPO)',
                 fontsize=13, fontweight='bold', y=1.02)
    for ax, m, label, color in zip(axes, metrics, labels, colors):
        vals  = df[m].values
        folds = np.arange(1, len(vals) + 1)
        ax.bar(folds, vals, color=color, alpha=0.75,
               edgecolor='black', linewidth=0.6)
        ax.axhline(np.nanmean(vals), color='red', linestyle='--',
                   linewidth=1.5, label=f'Mean={np.nanmean(vals):.2f}')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Patient', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(folds)
        ax.set_xticklabels([f"P{int(p)}" for p in df['patient'].values],
                           fontsize=8, rotation=45)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Score', fontsize=10)
    plt.tight_layout()
    path = out_dir / 'metrics_per_fold_NNConv.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_roc(all_y_true, all_y_proba, fold_aucs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    overall_auc = roc_auc_score(all_y_true, all_y_proba)
    ax.plot(fpr, tpr, color='#55A868', lw=2.5,
            label=f'Overall ROC (AUC = {overall_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
    ax.set_title(f'ROC Curve — NNConv GNN\n'
                 f'Mean fold AUC = {np.nanmean(fold_aucs):.3f} '
                 f'± {np.nanstd(fold_aucs):.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'roc_curve_NNConv.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_loss(all_fold_losses, out_dir):
    max_ep = max(len(l) for l in all_fold_losses)
    padded = np.full((len(all_fold_losses), max_ep), np.nan)
    for i, l in enumerate(all_fold_losses):
        padded[i, :len(l)] = l
    mean = np.nanmean(padded, axis=0)
    std  = np.nanstd(padded,  axis=0)
    ep   = np.arange(1, max_ep + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, mean, color='#55A868', lw=2, label='Mean train loss')
    ax.fill_between(ep, mean - std, mean + std,
                    alpha=0.2, color='#55A868', label='±1 std')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax.set_title('Training Loss — NNConv GNN (mean across LOPO folds)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'training_loss_NNConv.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════

def main(args):
    out_dir  = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs   = torch.load(args.graphs, weights_only=False)
    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]

    print(f"\n{'='*65}")
    print(f"  STEP 6 v2 — NNConv GNN  (edge attributes: DTF + PDC)")
    print(f"{'='*65}")
    print(f"  Device    : {device}")
    print(f"  Graphs    : {len(graphs)}")
    print(f"  Labels    : {Counter(g.y.item() for g in graphs)}")
    print(f"  Node dim  : {node_dim}")
    print(f"  Edge dim  : {edge_dim}  ← used in message passing")
    print(f"  Patients  : {sorted(set(g.patient.item() for g in graphs))}")

    summary = run_lopo(graphs, args, out_dir, device)

    print(f"\n{'='*65}")
    print(f"  FINAL RESULT — NNConv GNN")
    print(f"{'='*65}")
    for k in ['sensitivity', 'specificity', 'f1_ictal', 'bal_acc', 'auc']:
        print(f"    {k:15s}: {summary[f'{k}_mean']:.3f} "
              f"± {summary[f'{k}_std']:.3f}")
    print(f"\n  All outputs saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Step 6 v2 — NNConv GNN with edge attributes")
    parser.add_argument('--graphs',     required=True)
    parser.add_argument('--outdir',     required=True)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--hidden',     type=int,   default=64)
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--batch_size', type=int,   default=32)
    args = parser.parse_args()
    main(args)
