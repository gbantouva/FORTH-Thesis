"""
Step 7 — Self-Supervised Pretraining (GraphCL) + Fine-tuning (LOPO)
====================================================================
Phase 1 — Pretrain GraphSAGE encoder on ALL graphs (no labels)
           using Graph Contrastive Learning (GraphCL):
           - Two augmented views per graph (feature mask + edge drop)
           - NT-Xent contrastive loss

Phase 2 — Two-stage fine-tuning with LOPO cross-validation:
           Stage A (epochs 1-30):   encoder FROZEN, train classifier only
                                    → classifier head warms up without
                                      disturbing pretrained encoder
           Stage B (epochs 31-100): encoder UNFROZEN, train everything
                                    with 10x smaller lr for encoder
                                    → gentle fine-tuning preserves
                                      pretrained representations

Input : graphs/all_graphs.pt  (from step5)
Output: results_ssl/
          metrics_summary.csv
          fold_metrics.csv
          confusion_matrix_SSL_GraphSAGE.png
          metrics_per_fold_SSL_GraphSAGE.png
          roc_curve_SSL_GraphSAGE.png
          training_loss_pretrain.png
          training_loss_finetune.png
          pretrained_encoder.pt

Usage:
  python step7_ssl_graphsage.py \
    --graphs            path/to/graphs/all_graphs.pt \
    --outdir            path/to/results_ssl \
    --pretrain_epochs   300 \
    --finetune_epochs   100 \
    --freeze_epochs     30  \
    --hidden            64  \
    --lr_pretrain       0.001 \
    --lr_finetune       0.001 \
    --temperature       0.1 \
    --mask_ratio        0.1 \
    --drop_ratio        0.1 \
    --batch_size        128
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data   import Data
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


# ═══════════════════════════════════════════════════════════════
# 1. MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════

class GraphSAGEEncoder(nn.Module):
    """
    3-layer GraphSAGE encoder.
    Shared between pretraining and fine-tuning phases.
    Outputs graph-level embedding via global mean pooling.
    """
    def __init__(self, in_channels, hidden_channels, dropout=0.3):
        super().__init__()
        self.conv1   = SAGEConv(in_channels,     hidden_channels)
        self.conv2   = SAGEConv(hidden_channels,  hidden_channels)
        self.conv3   = SAGEConv(hidden_channels,  hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)          # (B, hidden)


class ProjectionHead(nn.Module):
    """
    MLP projection head — used ONLY during pretraining.
    Maps encoder output to contrastive embedding space.
    Discarded after pretraining (standard GraphCL practice).
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
        )

    def forward(self, x):
        return self.net(x)


class ClassifierHead(nn.Module):
    """MLP classifier — used ONLY during fine-tuning."""
    def __init__(self, hidden_channels, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# 2. AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════

def augment_features(x, mask_ratio=0.1):
    """
    Feature masking: randomly zero out mask_ratio of node features.
    Forces encoder to learn representations not dependent on
    individual channels.
    """
    mask = torch.bernoulli(
        torch.full(x.shape, 1 - mask_ratio, device=x.device)
    )
    return x * mask


def augment_edges(edge_index, drop_ratio=0.1):
    """
    Edge dropping: randomly remove drop_ratio of edges.
    Forces encoder to learn structural patterns beyond
    individual strong connections.
    """
    n_edges = edge_index.shape[1]
    n_keep  = max(1, int(n_edges * (1 - drop_ratio)))
    perm    = torch.randperm(n_edges, device=edge_index.device)[:n_keep]
    return edge_index[:, perm]


def augment_graph(data, mask_ratio=0.1, drop_ratio=0.1):
    """Apply feature masking + edge dropping, return new Data object."""
    return Data(
        x          = augment_features(data.x, mask_ratio),
        edge_index = augment_edges(data.edge_index, drop_ratio),
        edge_attr  = data.edge_attr,
    )


# ═══════════════════════════════════════════════════════════════
# 3. NT-XENT CONTRASTIVE LOSS
# ═══════════════════════════════════════════════════════════════

def nt_xent_loss(z1, z2, temperature=0.1):
    """
    NT-Xent loss for a batch of N graphs.
    Positive pair : (z1[i], z2[i])  — two views of graph i
    Negative pairs: all (z1[i], z1[j]) and (z1[i], z2[j]) for j≠i

    Lower temperature = sharper distribution = harder negatives.
    """
    N  = z1.shape[0]
    z  = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2N, dim)
    sim = torch.mm(z, z.T) / temperature                  # (2N, 2N)

    # Mask self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # Positive indices: z1[i] ↔ z2[i] at position i+N, and vice versa
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ═══════════════════════════════════════════════════════════════
# 4. PRETRAINING
# ═══════════════════════════════════════════════════════════════

def pretrain(graphs, encoder, proj_head, args, device):
    """Phase 1: GraphCL pretraining — no labels used."""
    loader    = DataLoader(graphs,
                           batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr_pretrain, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs
    )

    encoder.train()
    proj_head.train()
    losses = []

    print(f"\n{'='*65}")
    print(f"  PHASE 1 — GraphCL Pretraining  ({args.pretrain_epochs} epochs)")
    print(f"  Graphs={len(graphs)}  batch={args.batch_size}  "
          f"temp={args.temperature}  "
          f"mask={args.mask_ratio}  drop={args.drop_ratio}")
    print(f"{'='*65}")

    for epoch in range(1, args.pretrain_epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            v1    = augment_graph(batch, args.mask_ratio, args.drop_ratio)
            v2    = augment_graph(batch, args.mask_ratio, args.drop_ratio)

            h1 = encoder(v1.x, v1.edge_index, batch.batch)
            h2 = encoder(v2.x, v2.edge_index, batch.batch)
            z1 = proj_head(h1)
            z2 = proj_head(h2)

            loss = nt_xent_loss(z1, z2, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.pretrain_epochs} "
                  f"| Loss = {avg:.4f}")

    print(f"  Pretraining done. Final loss = {losses[-1]:.4f}")
    return losses


# ═══════════════════════════════════════════════════════════════
# 5. FINE-TUNING HELPERS
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(encoder, clf_head, loader,
                    optimizer, device, class_weights):
    encoder.train()
    clf_head.train()
    criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0.0
    for batch in loader:
        batch  = batch.to(device)
        h      = encoder(batch.x, batch.edge_index, batch.batch)
        logits = clf_head(h)
        loss   = criterion(logits, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(encoder, clf_head, loader, device):
    encoder.eval()
    clf_head.eval()
    all_labels, all_proba = [], []
    for batch in loader:
        batch  = batch.to(device)
        h      = encoder(batch.x, batch.edge_index, batch.batch)
        proba  = F.softmax(clf_head(h), dim=1)[:, 1].cpu().numpy()
        all_proba.extend(proba.tolist())
        all_labels.extend(batch.y.cpu().numpy().tolist())
    y_true = np.array(all_labels)
    y_prob = np.array(all_proba)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


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
# 6. TWO-STAGE LOPO FINE-TUNING
# ═══════════════════════════════════════════════════════════════

def run_lopo_finetune(graphs, pretrained_encoder, args, device, out_dir):
    """
    Phase 2: Two-stage fine-tuning with LOPO CV.

    Stage A (epochs 1 → freeze_epochs):
      Encoder is FROZEN. Only classifier head trains.
      The randomly-initialized head stabilizes without pulling
      the pretrained encoder away from its learned representations.

    Stage B (epochs freeze_epochs+1 → finetune_epochs):
      Encoder is UNFROZEN. Both encoder + head train together,
      but encoder uses lr * 0.1 to preserve pretrained weights.
    """
    patients        = sorted(set(g.patient.item() for g in graphs))
    fold_records    = []
    cm_total        = np.zeros((2, 2), dtype=int)
    all_y_true      = []
    all_y_proba     = []
    all_fold_losses = []

    print(f"\n{'='*65}")
    print(f"  PHASE 2 — Two-Stage Fine-tuning  ({len(patients)} LOPO folds)")
    print(f"  Stage A: epochs 1–{args.freeze_epochs}  "
          f"encoder FROZEN  lr={args.lr_finetune}")
    print(f"  Stage B: epochs {args.freeze_epochs+1}–{args.finetune_epochs}  "
          f"encoder UNFROZEN  lr={args.lr_finetune * 0.1:.5f}")
    print(f"{'='*65}")

    for fold, test_pat in enumerate(patients):
        train_graphs = [g for g in graphs if g.patient.item() != test_pat]
        test_graphs  = [g for g in graphs if g.patient.item() == test_pat]

        # Class weights from this fold's training set
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

        # Fresh copy of pretrained encoder per fold
        encoder  = deepcopy(pretrained_encoder).to(device)
        clf_head = ClassifierHead(args.hidden, args.dropout).to(device)

        fold_losses = []

        # ── Stage A: frozen encoder ────────────────────────────
        for param in encoder.parameters():
            param.requires_grad = False

        opt_frozen = torch.optim.Adam(
            clf_head.parameters(),
            lr=args.lr_finetune, weight_decay=1e-4,
        )

        for epoch in range(1, args.freeze_epochs + 1):
            loss = train_one_epoch(encoder, clf_head, train_loader,
                                   opt_frozen, device, class_weights)
            fold_losses.append(loss)

        # ── Stage B: unfreeze encoder, small lr ───────────────
        for param in encoder.parameters():
            param.requires_grad = True

        opt_full = torch.optim.Adam([
            {'params': encoder.parameters(),
             'lr': args.lr_finetune * 0.1},   # 10x smaller for encoder
            {'params': clf_head.parameters(),
             'lr': args.lr_finetune},
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.StepLR(
            opt_full, step_size=20, gamma=0.5
        )

        for epoch in range(args.freeze_epochs + 1, args.finetune_epochs + 1):
            loss = train_one_epoch(encoder, clf_head, train_loader,
                                   opt_full, device, class_weights)
            fold_losses.append(loss)
            scheduler.step()

        all_fold_losses.append(fold_losses)

        # ── Evaluate ───────────────────────────────────────────
        y_true, y_pred, y_proba = evaluate(encoder, clf_head,
                                           test_loader, device)
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

    # ── Aggregate ──────────────────────────────────────────────
    metric_keys = ['sensitivity', 'specificity', 'precision',
                   'f1_ictal', 'bal_acc', 'auc']
    summary_row = {'model': 'SSL-GraphSAGE'}
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
    print(f"  rows=true (pre-ictal, ictal)  cols=predicted")

    # ── Save plots + CSVs ─────────────────────────────────────
    print(f"\n  Saving plots...")
    _plot_confusion_matrix(cm_total, out_dir)
    _plot_metrics_per_fold(fold_records, out_dir)
    _plot_roc(np.array(all_y_true), np.array(all_y_proba),
              [r['auc'] for r in fold_records], out_dir)
    _plot_finetune_loss(all_fold_losses, args.freeze_epochs, out_dir)

    pd.DataFrame(fold_records).to_csv(
        out_dir / 'fold_metrics.csv', index=False, float_format='%.4f')
    pd.DataFrame([summary_row]).to_csv(
        out_dir / 'metrics_summary.csv', index=False, float_format='%.4f')
    print(f"  Saved: fold_metrics.csv")
    print(f"  Saved: metrics_summary.csv")

    return summary_row


# ═══════════════════════════════════════════════════════════════
# 7. PLOTS
# ═══════════════════════════════════════════════════════════════

def _plot_pretrain_loss(losses, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, color='#4C72B0', lw=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('NT-Xent Loss', fontsize=11)
    ax.set_title('GraphCL Pretraining Loss',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'training_loss_pretrain.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_finetune_loss(all_fold_losses, freeze_epochs, out_dir):
    max_ep = max(len(l) for l in all_fold_losses)
    padded = np.full((len(all_fold_losses), max_ep), np.nan)
    for i, l in enumerate(all_fold_losses):
        padded[i, :len(l)] = l
    mean = np.nanmean(padded, axis=0)
    std  = np.nanstd(padded,  axis=0)
    ep   = np.arange(1, max_ep + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ep, mean, color='#DD8452', lw=2, label='Mean train loss')
    ax.fill_between(ep, mean - std, mean + std,
                    alpha=0.2, color='#DD8452', label='±1 std')
    # Mark stage boundary
    ax.axvline(freeze_epochs, color='gray', linestyle='--',
               linewidth=1.5, label=f'Unfreeze encoder (epoch {freeze_epochs})')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax.set_title('Fine-tuning Loss — SSL-GraphSAGE\n'
                 'Stage A: frozen encoder  |  Stage B: unfrozen encoder',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'training_loss_finetune.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_confusion_matrix(cm_total, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'],
                linewidths=0.5, linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title('Confusion Matrix — SSL-GraphSAGE\n(all LOPO folds pooled)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'confusion_matrix_SSL_GraphSAGE.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_metrics_per_fold(fold_records, out_dir):
    df      = pd.DataFrame(fold_records)
    metrics = ['sensitivity', 'specificity', 'f1_ictal', 'bal_acc', 'auc']
    labels  = ['Sensitivity', 'Specificity', 'F1 (Ictal)', 'Bal. Accuracy', 'AUC']
    colors  = ['#4C72B0', '#55A868', '#DD8452', '#C44E52', '#8172B2']

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    fig.suptitle('Per-Fold Metrics — SSL-GraphSAGE (LOPO)',
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
    path = out_dir / 'metrics_per_fold_SSL_GraphSAGE.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_roc(all_y_true, all_y_proba, fold_aucs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    overall_auc = roc_auc_score(all_y_true, all_y_proba)
    ax.plot(fpr, tpr, color='#DD8452', lw=2.5,
            label=f'Overall ROC (AUC = {overall_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
    ax.set_title(f'ROC Curve — SSL-GraphSAGE\n'
                 f'Mean fold AUC = {np.nanmean(fold_aucs):.3f} '
                 f'± {np.nanstd(fold_aucs):.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'roc_curve_SSL_GraphSAGE.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════

def main(args):
    out_dir  = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs   = torch.load(args.graphs, weights_only=False)
    node_dim = graphs[0].x.shape[1]

    print(f"\n{'='*65}")
    print(f"  STEP 7 — SSL GraphSAGE  (GraphCL + Two-Stage Fine-tune)")
    print(f"{'='*65}")
    print(f"  Device   : {device}")
    print(f"  Graphs   : {len(graphs)}")
    print(f"  Labels   : {Counter(g.y.item() for g in graphs)}")
    print(f"  Node dim : {node_dim}")
    print(f"  Patients : {sorted(set(g.patient.item() for g in graphs))}")

    # ── Phase 1: Pretrain ──────────────────────────────────────
    encoder   = GraphSAGEEncoder(node_dim, args.hidden,
                                 args.dropout).to(device)
    proj_head = ProjectionHead(args.hidden).to(device)

    pretrain_losses = pretrain(graphs, encoder, proj_head, args, device)
    _plot_pretrain_loss(pretrain_losses, out_dir)

    enc_path = out_dir / 'pretrained_encoder.pt'
    torch.save(encoder.state_dict(), enc_path)
    print(f"  Saved pretrained encoder → {enc_path.name}")

    # ── Phase 2: Two-stage fine-tune ───────────────────────────
    summary = run_lopo_finetune(graphs, encoder, args, device, out_dir)

    print(f"\n{'='*65}")
    print(f"  FINAL RESULT — SSL-GraphSAGE (two-stage fine-tuning)")
    print(f"{'='*65}")
    for k in ['sensitivity', 'specificity', 'f1_ictal', 'bal_acc', 'auc']:
        print(f"    {k:15s}: {summary[f'{k}_mean']:.3f} "
              f"± {summary[f'{k}_std']:.3f}")
    print(f"\n  All outputs saved to: {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Step 7 — SSL GraphSAGE with two-stage fine-tuning")
    parser.add_argument('--graphs',           required=True)
    parser.add_argument('--outdir',           required=True)
    parser.add_argument('--pretrain_epochs',  type=int,   default=300)
    parser.add_argument('--finetune_epochs',  type=int,   default=100)
    parser.add_argument('--freeze_epochs',    type=int,   default=30,
                        help='Epochs to keep encoder frozen (Stage A)')
    parser.add_argument('--hidden',           type=int,   default=64)
    parser.add_argument('--lr_pretrain',      type=float, default=0.001)
    parser.add_argument('--lr_finetune',      type=float, default=0.001)
    parser.add_argument('--dropout',          type=float, default=0.3)
    parser.add_argument('--batch_size',       type=int,   default=128)
    parser.add_argument('--temperature',      type=float, default=0.1)
    parser.add_argument('--mask_ratio',       type=float, default=0.1)
    parser.add_argument('--drop_ratio',       type=float, default=0.1)
    args = parser.parse_args()
    main(args)
