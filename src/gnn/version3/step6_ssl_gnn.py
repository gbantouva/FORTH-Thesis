"""
Step 6 - Self-Supervised GCN (GraphCL-style Contrastive Learning)
==================================================================
SSL Strategy: Graph-level contrastive learning (GraphCL)
  - Two augmented views of each graph are created
  - The encoder is trained to maximize agreement between views
    of the same graph (positive pair) vs different graphs (negative)
  - Loss: NT-Xent (normalized temperature-scaled cross-entropy)

Augmentations (EEG-appropriate):
  1. Edge dropout     : randomly zero out edges (disrupts weak connections)
  2. Node feature noise: add Gaussian noise to node features
  3. Edge reweighting : multiply edges by uniform random in [0.8, 1.2]

Pipeline:
  Phase 1 — SSL pre-training (all epochs, no labels used)
  Phase 2 — Fine-tuning classifier (LOPO CV, frozen encoder + linear head)
  Phase 3 — Full fine-tuning (LOPO CV, all layers trainable, small LR)

Outputs:
  - pretrained_encoder.pt
  - per-fold metrics, ROC curves, confusion matrices, loss curves
  - comparison_all_models.png  (RF vs SVM vs GCN vs SSL-GCN)
  - results_ssl.json

Usage:
  python step6_ssl_gnn.py \
      --featfile   features/features_all.npz \
      --outputdir  results/ssl_gnn \
      --ssl_epochs 200 \
      --ft_epochs  100 \
      --lr_ssl     0.001 \
      --lr_ft      0.0005 \
      --hidden     32 \
      --threshold  0.15 \
      --temperature 0.5
"""

import argparse
import copy
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
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, precision_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Adjacency normalization (same as step5)
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(adj):
    A = adj + np.eye(adj.shape[0])
    D = np.diag(A.sum(axis=1) ** -0.5)
    A_hat = D @ A @ D
    return torch.tensor(A_hat, dtype=torch.float32)


def build_graphs(node_feats, adj_dtf, threshold=0.15):
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        x     = torch.tensor(node_feats[i], dtype=torch.float32)
        graphs.append((x, a_hat))
    return graphs


def scale_node_features(graphs_train, graphs_test):
    all_train = np.concatenate([g[0].numpy() for g in graphs_train], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_train)
    def apply(graphs):
        return [(torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32), a)
                for x, a in graphs]
    return apply(graphs_train), apply(graphs_test), scaler


def scale_all_graphs(all_graphs):
    """Scale all graphs using global scaler (for SSL pre-training)."""
    all_x = np.concatenate([g[0].numpy() for g in all_graphs], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_x)
    return [(torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32), a)
            for x, a in all_graphs], scaler


# ─────────────────────────────────────────────────────────────
# Graph Augmentations (EEG-appropriate)
# ─────────────────────────────────────────────────────────────

def augment_edge_dropout(x, a_hat, p=0.2):
    """Randomly zero out edges with probability p."""
    mask = (torch.rand_like(a_hat) > p).float()
    np.fill_diagonal(mask.numpy(), 1.0)   # keep self-loops
    a_aug = a_hat * mask
    # Re-normalize
    A = a_aug.numpy()
    d = A.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    A_norm = A / d
    return x, torch.tensor(A_norm, dtype=torch.float32)


def augment_node_noise(x, a_hat, sigma=0.1):
    """Add Gaussian noise to node features."""
    noise = torch.randn_like(x) * sigma
    return x + noise, a_hat


def augment_edge_reweight(x, a_hat, low=0.8, high=1.2):
    """Multiply edge weights by random uniform in [low, high]."""
    scale = torch.empty_like(a_hat).uniform_(low, high)
    np.fill_diagonal(scale.numpy(), 1.0)
    a_aug = a_hat * scale
    A = a_aug.numpy()
    d = A.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    A_norm = A / d
    return x, torch.tensor(A_norm, dtype=torch.float32)


def random_augment(x, a_hat):
    """Apply two random augmentations to produce a view."""
    augs = [augment_edge_dropout, augment_node_noise, augment_edge_reweight]
    # Pick 2 distinct augmentations
    chosen = np.random.choice(len(augs), size=2, replace=False)
    x1, a1 = augs[chosen[0]](x.clone(), a_hat.clone())
    x2, a2 = augs[chosen[1]](x.clone(), a_hat.clone())
    return (x1, a1), (x2, a2)


# ─────────────────────────────────────────────────────────────
# GCN Encoder (shared between SSL and fine-tuning)
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, a_hat):
        return F.relu(self.W(a_hat @ x))


class GCNEncoder(nn.Module):
    """
    Shared encoder: 2-layer GCN → GlobalMeanPool → embedding
    Output: graph-level embedding vector (out_dim,)
    """
    def __init__(self, in_dim=16, hidden=32, out_dim=64, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, out_dim)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x, a_hat):
        h = self.gcn1(x, a_hat)       # (19, hidden)
        h = self.drop(h)
        h = self.gcn2(h, a_hat)       # (19, out_dim)
        return h.mean(dim=0)          # (out_dim,)  graph embedding


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive loss (discarded after pre-training)."""
    def __init__(self, in_dim=64, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, h):
        return self.net(h)


class ClassifierHead(nn.Module):
    """Linear + small MLP head for fine-tuning."""
    def __init__(self, in_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze()


# ─────────────────────────────────────────────────────────────
# NT-Xent Contrastive Loss
# ─────────────────────────────────────────────────────────────

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (N, proj_dim) — embeddings of two views of N graphs
    Returns scalar NT-Xent loss.
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)           # (2N, proj_dim)
    z = F.normalize(z, dim=1)

    # Cosine similarity matrix (2N, 2N)
    sim = torch.mm(z, z.T) / temperature

    # Mask out self-similarities
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])

    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────
# SSL Pre-training
# ─────────────────────────────────────────────────────────────

def ssl_pretrain(encoder, proj_head, all_graphs, ssl_epochs, lr, device,
                 batch_size=32, temperature=0.5):
    """
    Train encoder + projection head with NT-Xent on all graphs (no labels).
    Returns list of per-epoch losses.
    """
    encoder.train()
    proj_head.train()
    params    = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=ssl_epochs)

    losses = []
    N = len(all_graphs)

    print(f'\n  SSL Pre-training: {ssl_epochs} epochs, {N} graphs, batch={batch_size}')

    for ep in range(ssl_epochs):
        idx = np.random.permutation(N)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = idx[start: start + batch_size]
            if len(batch_idx) < 2:
                continue

            z1_list, z2_list = [], []
            for i in batch_idx:
                x, a = all_graphs[i]
                (x1, a1), (x2, a2) = random_augment(x, a)
                x1, a1 = x1.to(device), a1.to(device)
                x2, a2 = x2.to(device), a2.to(device)
                h1 = encoder(x1, a1)
                h2 = encoder(x2, a2)
                z1_list.append(proj_head(h1))
                z2_list.append(proj_head(h2))

            z1 = torch.stack(z1_list)   # (B, proj_dim)
            z2 = torch.stack(z2_list)

            loss = nt_xent_loss(z1, z2, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            ep_loss   += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = ep_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (ep + 1) % 20 == 0:
            print(f'    Epoch [{ep+1:3d}/{ssl_epochs}]  SSL loss: {avg_loss:.4f}')

    return losses


# ─────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────

def finetune(encoder, clf_head, graphs_train, y_train,
             graphs_test, y_test, ft_epochs, lr, pos_weight,
             device, patience=20, freeze_encoder=False):
    """
    Fine-tune encoder + classifier head with BCE loss.
    If freeze_encoder=True: only clf_head is trained (linear probe).
    Returns train_losses, val_losses, best_auc
    """
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = list(clf_head.parameters())
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        params = list(encoder.parameters()) + list(clf_head.parameters())

    optimizer = Adam(params, lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    train_losses, val_losses = [], []
    best_auc     = 0.0
    best_enc_state = None
    best_clf_state = None
    patience_cnt = 0

    for ep in range(ft_epochs):
        # Train
        encoder.train()
        clf_head.train()
        ep_loss = 0.0
        idx = np.random.permutation(len(graphs_train))
        for i in idx:
            x, a = graphs_train[i]
            x, a = x.to(device), a.to(device)
            optimizer.zero_grad()
            h     = encoder(x, a)
            logit = clf_head(h)
            label = torch.tensor(float(y_train[i]), device=device).view(1)
            loss  = criterion(logit.view(1), label)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
        train_losses.append(ep_loss / len(graphs_train))

        # Validate
        encoder.eval()
        clf_head.eval()
        with torch.no_grad():
            val_logits = []
            for i in range(len(graphs_test)):
                x, a = graphs_test[i]
                h = encoder(x.to(device), a.to(device))
                val_logits.append(clf_head(h).cpu().item())
        val_logits_t = torch.tensor(val_logits)
        val_labels_t = torch.tensor(y_test, dtype=torch.float32)
        val_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )(val_logits_t, val_labels_t).item()
        val_losses.append(val_loss)

        probs = torch.sigmoid(val_logits_t).numpy()
        if len(np.unique(y_test)) == 2:
            val_auc = roc_auc_score(y_test, probs)
        else:
            val_auc = 0.0

        if val_auc > best_auc:
            best_auc       = val_auc
            best_enc_state = copy.deepcopy(encoder.state_dict())
            best_clf_state = copy.deepcopy(clf_head.state_dict())
            patience_cnt   = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # Restore best
    if best_enc_state:
        encoder.load_state_dict(best_enc_state)
    if best_clf_state:
        clf_head.load_state_dict(best_clf_state)

    return train_losses, val_losses, best_auc


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(encoder, clf_head, graphs, labels, device):
    encoder.eval()
    clf_head.eval()
    logits = []
    for i in range(len(graphs)):
        x, a = graphs[i]
        h = encoder(x.to(device), a.to(device))
        logits.append(clf_head(h).cpu().item())
    logits  = np.array(logits)
    probs   = 1 / (1 + np.exp(-logits))
    preds   = (probs >= 0.5).astype(int)
    return probs, preds


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

def plot_ssl_loss(ssl_losses, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ssl_losses, color='mediumpurple', lw=2)
    ax.set_xlabel('SSL Epoch', fontsize=12)
    ax.set_ylabel('NT-Xent Loss', fontsize=12)
    ax.set_title('SSL Pre-training Loss (NT-Xent Contrastive)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ssl_pretrain_loss.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_ft_loss_curves(train_losses, val_losses, patient_id, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label='Train loss', color='royalblue')
    ax.plot(val_losses,   label='Val loss',   color='tomato', linestyle='--')
    gap = abs(train_losses[-1] - val_losses[-1])
    ax.text(0.65, 0.85, f'Final gap: {gap:.4f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'SSL-GCN Fine-tune Loss | Test: {patient_id}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'ft_loss_{patient_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, patient_id, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'SSL-GCN CM | Test: {patient_id}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_ssl_{patient_id}.png', dpi=150, bbox_inches='tight')
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
        f'SSL-GCN LOPO ROC\nMean AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_ssl_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['mediumpurple', 'tomato', 'seagreen', 'darkorange']
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
    ax.set_title('SSL-GCN Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_fold_ssl_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion(all_y_true, all_y_pred, output_dir):
    cm      = confusion_matrix(all_y_true, all_y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',    'Counts'),
        (axes[1], cm_norm, '.2f',  'Normalized'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Purples', ax=ax,
                    xticklabels=['Pre-ictal', 'Ictal'],
                    yticklabels=['Pre-ictal', 'Ictal'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'SSL-GCN Aggregate CM ({title})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cm_aggregate_ssl_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_final_comparison(ssl_stats, sup_json, baseline_json, output_dir):
    models = {}

    if baseline_json and Path(baseline_json).exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        for name, res in bl.items():
            models[name] = {k: res['summary_stats'][k]['mean']
                            for k in ['auc','f1','sensitivity','specificity']}

    if sup_json and Path(sup_json).exists():
        with open(sup_json) as f:
            sup = json.load(f)
        models['GCN (Supervised)'] = {k: sup['summary_stats'][k]['mean']
                                       for k in ['auc','f1','sensitivity','specificity']}

    models['SSL-GCN (Ours)'] = {k: ssl_stats[k]['mean']
                                  for k in ['auc','f1','sensitivity','specificity']}

    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    x        = np.arange(len(met_keys))
    width    = 0.18
    colors   = ['steelblue', 'tomato', 'seagreen', 'mediumpurple']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, vals) in enumerate(models.items()):
        means  = [vals[k] for k in met_keys]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=name,
               color=colors[i % len(colors)], alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Mean Score (LOPO)', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('Full Model Comparison: RF / SVM / GCN / SSL-GCN (LOPO CV)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Final comparison chart saved.')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/ssl_gnn')
    parser.add_argument('--ssl_epochs',    type=int,   default=200)
    parser.add_argument('--ft_epochs',     type=int,   default=100)
    parser.add_argument('--lr_ssl',        type=float, default=0.001)
    parser.add_argument('--lr_ft',         type=float, default=0.0005)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--outdim',        type=int,   default=64)
    parser.add_argument('--proj_dim',      type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--threshold',     type=float, default=0.15)
    parser.add_argument('--temperature',   type=float, default=0.5)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--baseline_json', default=None)
    parser.add_argument('--sup_json',      default=None,
                        help='Path to step5 results_gcn.json for comparison')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('STEP 6 — SSL GCN (GraphCL Contrastive)')
    print('=' * 60)
    print(f'Device      : {device}')
    print(f'SSL epochs  : {args.ssl_epochs}  LR: {args.lr_ssl}  Temp: {args.temperature}')
    print(f'FT epochs   : {args.ft_epochs}   LR: {args.lr_ft}')
    print(f'Hidden: {args.hidden}  OutDim: {args.outdim}  ProjDim: {args.proj_dim}')
    print(f'DTF threshold: {args.threshold}  Batch: {args.batch_size}')
    print('=' * 60)

    # ── Load data ────────────────────────────────────────────
    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)
    adj_dtf     = data['adj_dtf'].astype(np.float32)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']

    print(f'Loaded: {len(y)} epochs | Ictal: {(y==1).sum()} Pre-ictal: {(y==0).sum()}')

    # ── Build & scale all graphs ─────────────────────────────
    print('Building graphs...')
    all_graphs = build_graphs(node_feats, adj_dtf, threshold=args.threshold)
    all_graphs_scaled, global_scaler = scale_all_graphs(all_graphs)

    # ── Phase 1: SSL Pre-training (ALL data, no labels) ──────
    encoder   = GCNEncoder(in_dim=16, hidden=args.hidden,
                           out_dim=args.outdim, dropout=args.dropout).to(device)
    proj_head = ProjectionHead(in_dim=args.outdim, proj_dim=args.proj_dim).to(device)

    ssl_losses = ssl_pretrain(
        encoder, proj_head, all_graphs_scaled,
        ssl_epochs=args.ssl_epochs,
        lr=args.lr_ssl,
        device=device,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )

    plot_ssl_loss(ssl_losses, output_dir)

    # Save pre-trained encoder
    torch.save(encoder.state_dict(), output_dir / 'pretrained_encoder.pt')
    print(f'  Pre-trained encoder saved.')

    # ── Phase 2+3: LOPO Fine-tuning ──────────────────────────
    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []

    print(f'\n{"="*60}')
    print(f'  LOPO Fine-tuning ({len(patients)} folds)')
    print(f'{"="*60}')

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

        # Use globally scaled graphs
        graphs_train = [all_graphs_scaled[i] for i in train_idx]
        graphs_test  = [all_graphs_scaled[i] for i in test_idx]

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        # Clone pre-trained encoder for this fold
        enc  = copy.deepcopy(encoder).to(device)
        clf  = ClassifierHead(in_dim=args.outdim, dropout=args.dropout).to(device)

        # Phase 2: Linear probe (frozen encoder, 30 epochs)
        finetune(enc, clf, graphs_train, y_train,
                 graphs_test, y_test,
                 ft_epochs=30, lr=args.lr_ft * 5,
                 pos_weight=pos_weight, device=device,
                 patience=args.patience, freeze_encoder=True)

        # Phase 3: Full fine-tuning (all layers, small LR)
        tr_losses, val_losses, best_auc = finetune(
            enc, clf, graphs_train, y_train,
            graphs_test, y_test,
            ft_epochs=args.ft_epochs, lr=args.lr_ft,
            pos_weight=pos_weight, device=device,
            patience=args.patience, freeze_encoder=False,
        )

        probs, preds = evaluate(enc, clf, graphs_test, y_test, device)
        metrics = compute_metrics(y_test, preds, probs)
        if metrics is None:
            continue

        metrics['patient'] = pat
        metrics['n_train'] = int(train_mask.sum())
        metrics['n_test']  = int(test_mask.sum())
        fold_metrics.append(metrics)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(preds.tolist())

        fpr, tpr, _ = roc_curve(y_test, probs)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        print(f'  {pat:8s} | AUC={metrics["auc"]:.3f}  F1={metrics["f1"]:.3f}'
              f'  Sens={metrics["sensitivity"]:.3f}  Spec={metrics["specificity"]:.3f}'
              f'  MCC={metrics["mcc"]:.3f}')

        plot_ft_loss_curves(tr_losses, val_losses, pat, output_dir)
        plot_confusion_matrix(confusion_matrix(y_test, preds), pat, output_dir)

    # ── Aggregate plots ───────────────────────────────────────
    plot_roc_all_folds(fold_roc_data, output_dir)
    plot_per_fold_metrics(fold_metrics, output_dir)
    plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred), output_dir)

    # ── Summary ───────────────────────────────────────────────
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'precision', 'mcc']
    summary_stats = {}
    print(f'\n{"="*60}')
    print(f'SSL-GCN — Mean +/- Std across {len(fold_metrics)} folds')
    print(f'{"="*60}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = np.mean(vals), np.std(vals)
        summary_stats[k] = {'mean': float(mean_), 'std': float(std_)}
        print(f'  {k:15s}: {mean_:.3f} +/- {std_:.3f}')

    # ── Save ──────────────────────────────────────────────────
    results = {
        'model': 'SSL_GCN_GraphCL',
        'hyperparameters': vars(args),
        'fold_metrics':    fold_metrics,
        'summary_stats':   summary_stats,
        'ssl_final_loss':  float(ssl_losses[-1]),
    }
    with open(output_dir / 'results_ssl.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {output_dir}/results_ssl.json')

    # Final 4-model comparison
    plot_final_comparison(summary_stats, args.sup_json, args.baseline_json, output_dir)

    print('\n' + '=' * 60)
    print('STEP 6 COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
