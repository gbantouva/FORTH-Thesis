"""
Step 6 — GNN (GraphSAGE) with Leave-One-Patient-Out Cross-Validation
=====================================================================
Input : graphs/all_graphs_dtf_topk6.pt  (from step5)
Output: results_gnn/
          baseline_results.json       ← full metrics + hyperparams
          metrics_summary.csv
          fold_metrics.csv
          confusion_matrix_GraphSAGE.png
          metrics_per_fold_GraphSAGE.png
          roc_curve_GraphSAGE.png
          training_loss_GraphSAGE.png  ← train vs test loss (overfitting check)

Architecture:
  GraphSAGE (3 layers) → global mean pool → MLP classifier
  Note: SAGEConv does not use edge_attr — topology + node features only.

Labels : 0 = pre-ictal,  1 = ictal

Usage:
  python step6_gnn_graphsage.py \
    --graphs  path/to/graphs/all_graphs_dtf_topk6.pt \
    --outdir  path/to/results_gnn \
    --epochs  100 \
    --lr      0.001 \
    --hidden  64
"""

import argparse
import warnings
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn     import SAGEConv, global_mean_pool

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

METRICS_KEYS = ['sensitivity', 'specificity', 'precision',
                'f1_ictal', 'bal_acc', 'auc']


# ── JSON serialiser ───────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════
# 1. MODEL
# ═══════════════════════════════════════════════════════════════

class GraphSAGEClassifier(nn.Module):
    """
    3-layer GraphSAGE encoder + global mean pooling + MLP head.
    SAGEConv aggregates neighbour node features (mean aggregation).
    Edge attributes are NOT used — topology + node features only.
    """
    def __init__(self, in_channels, hidden_channels, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels,     hidden_channels)
        self.conv2 = SAGEConv(hidden_channels,  hidden_channels)
        self.conv3 = SAGEConv(hidden_channels,  hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)       # (B, hidden)
        return self.classifier(x)             # (B, 2)


# ═══════════════════════════════════════════════════════════════
# 2. METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)
    bal_acc     = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float('nan')

    return dict(sensitivity=sensitivity, specificity=specificity,
                precision=precision, f1_ictal=f1,
                bal_acc=bal_acc, auc=auc, cm=cm)


# ═══════════════════════════════════════════════════════════════
# 3. TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0.0
    for batch in loader:
        batch  = batch.to(device)
        optimizer.zero_grad()
        loss   = criterion(model(batch.x, batch.edge_index, batch.batch),
                           batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_with_loss(model, loader, device, class_weights):
    """Returns y_true, y_pred, y_proba AND mean cross-entropy loss."""
    model.eval()
    criterion   = nn.CrossEntropyLoss(weight=class_weights.to(device))
    all_labels, all_proba = [], []
    total_loss  = 0.0

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss   = criterion(logits, batch.y)
        total_loss += loss.item()
        proba  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_proba.extend(proba.tolist())
        all_labels.extend(batch.y.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_proba)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob, total_loss / len(loader)


# ═══════════════════════════════════════════════════════════════
# 4. PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm_total, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'],
                linewidths=0.5, linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title('Confusion Matrix — GraphSAGE\n(all LOPO folds pooled)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'confusion_matrix_GraphSAGE.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_metrics_per_fold(fold_records, out_dir):
    df      = pd.DataFrame(fold_records)
    metrics = ['sensitivity', 'specificity', 'f1_ictal', 'bal_acc', 'auc']
    labels  = ['Sensitivity', 'Specificity', 'F1 (Ictal)', 'Bal. Accuracy', 'AUC']
    colors  = ['#4C72B0', '#55A868', '#DD8452', '#C44E52', '#8172B2']

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    fig.suptitle('Per-Fold Metrics — GraphSAGE (LOPO)',
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
    path = out_dir / 'metrics_per_fold_GraphSAGE.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_roc_curve(all_y_true, all_y_proba, fold_aucs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    overall_auc = roc_auc_score(all_y_true, all_y_proba)
    ax.plot(fpr, tpr, color='#C44E52', lw=2.5,
            label=f'Pooled ROC  (AUC = {overall_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
    mean_auc = np.nanmean(fold_aucs)
    std_auc  = np.nanstd(fold_aucs)
    ax.set_title(f'ROC Curve — GraphSAGE\n'
                 f'Mean fold AUC = {mean_auc:.3f} ± {std_auc:.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'roc_curve_GraphSAGE.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_loss_curves(all_train_losses, all_test_losses, out_dir):
    """
    Overfitting check: plot mean train loss vs mean test loss across folds.
    A large and growing gap between the two = overfitting.
    """
    max_ep = max(len(l) for l in all_train_losses)

    def pad(loss_list):
        arr = np.full((len(loss_list), max_ep), np.nan)
        for i, l in enumerate(loss_list):
            arr[i, :len(l)] = l
        return arr

    tr  = pad(all_train_losses)
    te  = pad(all_test_losses)
    ep  = np.arange(1, max_ep + 1)

    tr_mean, tr_std = np.nanmean(tr, axis=0), np.nanstd(tr, axis=0)
    te_mean, te_std = np.nanmean(te, axis=0), np.nanstd(te, axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ep, tr_mean, color='#4C72B0', lw=2, label='Train loss (mean)')
    ax.fill_between(ep, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color='#4C72B0')
    ax.plot(ep, te_mean, color='#DD8452', lw=2, label='Test loss  (mean)')
    ax.fill_between(ep, te_mean - te_std, te_mean + te_std,
                    alpha=0.15, color='#DD8452')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax.set_title('Train vs Test Loss — GraphSAGE  (overfitting check)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'training_loss_GraphSAGE.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════
# 5. LOPO TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def run_lopo(graphs, args, out_dir, device):
    patients    = torch.tensor([g.patient.item() for g in graphs])
    unique_pats = patients.unique().tolist()
    node_dim    = graphs[0].x.shape[1]

    fold_records     = []
    cm_total         = np.zeros((2, 2), dtype=int)
    all_y_true       = []
    all_y_proba      = []
    all_train_losses = []
    all_test_losses  = []

    print(f"\n{'='*65}")
    print(f"  GraphSAGE LOPO — {len(unique_pats)} folds")
    print(f"  hidden={args.hidden}  epochs={args.epochs}  "
          f"lr={args.lr}  dropout={args.dropout}  batch={args.batch_size}")
    print(f"{'='*65}")

    for fold, test_pat in enumerate(unique_pats):
        train_graphs = [g for g in graphs if g.patient.item() != test_pat]
        test_graphs  = [g for g in graphs if g.patient.item() == test_pat]

        y_tr   = torch.tensor([g.y.item() for g in train_graphs])
        n0, n1 = (y_tr == 0).sum().item(), (y_tr == 1).sum().item()
        total  = n0 + n1
        w0     = total / (2 * n0) if n0 > 0 else 1.0
        w1     = total / (2 * n1) if n1 > 0 else 1.0
        class_weights = torch.tensor([w0, w1], dtype=torch.float)

        train_loader = DataLoader(train_graphs,
                                  batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(test_graphs,
                                  batch_size=args.batch_size, shuffle=False)

        model     = GraphSAGEClassifier(node_dim, args.hidden,
                                        dropout=args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=30, gamma=0.5)

        fold_train_losses = []
        fold_test_losses  = []

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, train_loader,
                                  optimizer, device, class_weights)
            _, _, _, te_loss = evaluate_with_loss(
                model, test_loader, device, class_weights)
            fold_train_losses.append(tr_loss)
            fold_test_losses.append(te_loss)
            scheduler.step()

        all_train_losses.append(fold_train_losses)
        all_test_losses.append(fold_test_losses)

        y_true, y_pred, y_proba, _ = evaluate_with_loss(
            model, test_loader, device, class_weights)
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
            'final_train_loss': fold_train_losses[-1],
            'final_test_loss':  fold_test_losses[-1],
        })

        print(f"  Fold {fold+1:2d} | patient={int(test_pat):2d} "
              f"| n={len(y_true):3d} "
              f"(ict={n_ict:2d} pre={n_pre:2d}) "
              f"| Sens={m['sensitivity']:.3f} "
              f"Spec={m['specificity']:.3f} "
              f"AUC={m['auc']:.3f} "
              f"tr_loss={fold_train_losses[-1]:.4f} "
              f"te_loss={fold_test_losses[-1]:.4f}")

    # ── Summary ────────────────────────────────────────────────
    summary_row = {'model': 'GraphSAGE'}
    print(f"\n  {'─'*55}")
    print(f"  Summary ({len(fold_records)} folds):")
    for k in METRICS_KEYS:
        vals = [r[k] for r in fold_records if not np.isnan(r[k])]
        mu, sd = np.mean(vals), np.std(vals)
        summary_row[f'{k}_mean'] = mu
        summary_row[f'{k}_std']  = sd
        print(f"    {k:15s}: {mu:.3f} ± {sd:.3f}")

    print(f"\n  Pooled confusion matrix:\n  {cm_total}")
    print(f"  rows=true (pre-ictal, ictal)  cols=predicted")

    # ── Plots ──────────────────────────────────────────────────
    print(f"\n  Saving plots...")
    plot_confusion_matrix(cm_total, out_dir)
    plot_metrics_per_fold(fold_records, out_dir)
    plot_roc_curve(np.array(all_y_true), np.array(all_y_proba),
                   [r['auc'] for r in fold_records], out_dir)
    plot_loss_curves(all_train_losses, all_test_losses, out_dir)

    # ── CSVs ───────────────────────────────────────────────────
    pd.DataFrame(fold_records).to_csv(
        out_dir / 'fold_metrics.csv', index=False, float_format='%.4f')
    pd.DataFrame([summary_row]).to_csv(
        out_dir / 'metrics_summary.csv', index=False, float_format='%.4f')
    print(f"  Saved: fold_metrics.csv")
    print(f"  Saved: metrics_summary.csv")

    return summary_row, fold_records, cm_total


# ═══════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════

def main(args):
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graphs = torch.load(args.graphs, weights_only=False)
    labels = [g.y.item() for g in graphs]

    print(f"\n{'='*65}")
    print(f"  STEP 6 — GraphSAGE GNN  (LOPO cross-validation)")
    print(f"{'='*65}")
    print(f"  Device   : {device}")
    print(f"  Graphs   : {len(graphs)}")
    print(f"  Labels   : {Counter(labels)}  (0=pre-ictal, 1=ictal)")
    print(f"  Node dim : {graphs[0].x.shape[1]}")
    print(f"  Edge dim : {graphs[0].edge_attr.shape[1]}  (not used by SAGEConv)")
    print(f"  Patients : {sorted(set(g.patient.item() for g in graphs))}")

    summary, fold_records, cm_total = run_lopo(graphs, args, out_dir, device)

    # ── JSON ───────────────────────────────────────────────────
    results_json = {
        'experiment': {
            'script'      : 'step6_gnn_graphsage.py',
            'timestamp'   : datetime.now().isoformat(timespec='seconds'),
            'graphs_file' : str(args.graphs),
            'n_graphs'    : len(graphs),
            'n_ictal'     : int(sum(1 for l in labels if l == 1)),
            'n_preictal'  : int(sum(1 for l in labels if l == 0)),
            'patients'    : sorted(set(g.patient.item() for g in graphs)),
            'device'      : str(device),
            'cv_strategy' : 'LeaveOnePatientOut (LOPO)',
            'edge_attr_used': False,
            'note'        : 'SAGEConv uses topology + node features only. '
                            'edge_attr stored in graph but not passed to conv.',
        },
        'hyperparameters': {
            'hidden_channels' : args.hidden,
            'n_layers'        : 3,
            'dropout'         : args.dropout,
            'epochs'          : args.epochs,
            'lr'              : args.lr,
            'weight_decay'    : 1e-4,
            'batch_size'      : args.batch_size,
            'optimizer'       : 'Adam',
            'scheduler'       : 'StepLR(step_size=30, gamma=0.5)',
            'class_weighting' : 'balanced (computed per fold from training set)',
            'pooling'         : 'global_mean_pool',
        },
        'results': {
            k: {
                'mean': round(summary[f'{k}_mean'], 4),
                'std' : round(summary[f'{k}_std'],  4),
            }
            for k in METRICS_KEYS
        },
        'confusion_matrix': {
            'layout': '[[TN, FP], [FN, TP]]',
            'matrix': cm_total.tolist(),
            'TN': int(cm_total[0, 0]),
            'FP': int(cm_total[0, 1]),
            'FN': int(cm_total[1, 0]),
            'TP': int(cm_total[1, 1]),
        },
        'fold_details': fold_records,
    }

    json_path = out_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {json_path.name}")

    print(f"\n{'='*65}")
    print(f"  FINAL RESULT — GraphSAGE")
    print(f"{'='*65}")
    for k in METRICS_KEYS:
        print(f"    {k:15s}: {summary[f'{k}_mean']:.3f} ± {summary[f'{k}_std']:.3f}")
    print(f"\n  All outputs saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Step 6 — GraphSAGE GNN (LOPO)")
    parser.add_argument('--graphs',     required=True,
                        help='Path to .pt graph file (step5 output)')
    parser.add_argument('--outdir',     required=True,
                        help='Output folder for results')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--hidden',     type=int,   default=64)
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--batch_size', type=int,   default=32)
    args = parser.parse_args()
    main(args)
