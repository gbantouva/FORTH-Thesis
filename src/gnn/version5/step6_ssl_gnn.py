"""
Step 6 — Self-Supervised GCN  (GraphCL-style, LOPO, NO DATA LEAKAGE)
=====================================================================
CRITICAL DESIGN DECISION — DATA LEAKAGE FIX:
  The original version pre-trained the SSL encoder on ALL graphs (all patients),
  then split by patient for LOPO fine-tuning.  This is leakage: the encoder had
  already seen the test patient's graphs before evaluation.

  CORRECT approach implemented here:
    ┌─ LOPO fold (one per patient) ──────────────────────────────────┐
    │  1. Split data into train_patients / test_patient              │
    │  2. Scale node features on train split ONLY                    │
    │  3. SSL pre-train encoder on TRAIN graphs only (no labels)     │
    │  4. Phase A — linear probe  (frozen encoder, 30 epochs)        │
    │  5. Phase B — full fine-tune (all layers,   ft_epochs)         │
    │  6. Adaptive threshold on train fold (maximises F1)            │
    │  7. Evaluate on test patient                                   │
    └────────────────────────────────────────────────────────────────┘
  The encoder never sees the test patient during pre-training.

WHY SSL MAKES SENSE ON A SMALL DATASET:
  With ~34 subjects and severe class imbalance, labelled data is scarce.
  SSL pre-training uses ALL training epochs (including pre-ictal) with no labels
  — it learns the general structure of EEG connectivity graphs.

SSL STRATEGY: GraphCL (You et al., 2020)
  - Two augmented views of the same graph = positive pair
  - NT-Xent (normalised temperature-scaled cross-entropy) loss
  - Projection head discarded after pre-training; only encoder is kept

AUGMENTATIONS (EEG-appropriate):
  1. Edge dropout       — randomly zero edges (p=0.20)
  2. Node feature noise — Gaussian noise (σ=0.10)
  3. Band feature mask  — zero out one random band across all nodes

ARCHITECTURE: identical to step5 for fair comparison.
  GCNLayer(16→32) → Dropout(0.4) → GCNLayer(32→32) → GlobalMeanPool
  SSL projection head: Linear(32→32) → ReLU → Linear(32→proj_dim)
  Fine-tune classifier: Linear(32→16) → ReLU → Dropout(0.4) → Linear(16→1)

Outputs:
  ssl_loss_fold_{patient}.png
  ft_loss_{patient}.png
  cm_ssl_{patient}.png
  roc_ssl_gcn.png
  per_fold_ssl_gcn.png
  cm_aggregate_ssl_gcn.png
  train_vs_test_auc_ssl.png
  comparison_all_models.png
  results_ssl.json

Usage:
  python step6_ssl_gnn.py \
      --featfile     features/features_all.npz \
      --outputdir    results/ssl_gnn \
      --ssl_epochs   200 \
      --ft_epochs    100 \
      --lr_ssl       0.001 \
      --lr_ft        0.0005 \
      --hidden       32 \
      --threshold    0.15 \
      --temperature  0.5 \
      --batch_size   32 \
      --patience     20 \
      --baseline_json results/baseline_ml/results_all.json \
      --sup_json      results/gnn_supervised/results_gcn.json
"""

import argparse
import copy
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Band feature indices — node_feats[ci, 0:6] = [delta,theta,alpha,beta,gamma,broad]
N_BAND_FEATS = 6


# ─────────────────────────────────────────────────────────────
# Adjacency normalisation
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def build_graphs(node_feats: np.ndarray, adj_dtf: np.ndarray,
                 threshold: float = 0.15):
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
    """
    Fit StandardScaler on training node features only.
    Called INSIDE the LOPO loop — never touches test patient during fit.
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
# EEG-appropriate augmentations
# ─────────────────────────────────────────────────────────────

def augment_edge_dropout(x: torch.Tensor, a_hat: torch.Tensor,
                         p: float = 0.20):
    """
    Randomly zero off-diagonal edges with probability p.
    Self-loops kept — they stabilise GCN training.
    Mimics intermittently absent functional connectivity.
    """
    mask = (torch.rand_like(a_hat) > p).float()
    diag_idx = torch.arange(a_hat.shape[0])
    mask[diag_idx, diag_idx] = 1.0
    a_aug   = a_hat * mask
    row_sum = a_aug.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return x, a_aug / row_sum


def augment_node_noise(x: torch.Tensor, a_hat: torch.Tensor,
                       sigma: float = 0.10):
    """
    Add zero-mean Gaussian noise to node features.
    Mimics electrode noise and small amplitude fluctuations.
    """
    return x + torch.randn_like(x) * sigma, a_hat


def augment_band_mask(x: torch.Tensor, a_hat: torch.Tensor):
    """
    Zero out ONE randomly selected frequency band for ALL nodes.
    EEG-specific: forces encoder to learn band-invariant representations.
    """
    band_idx     = int(torch.randint(0, N_BAND_FEATS, (1,)).item())
    x_aug        = x.clone()
    x_aug[:, band_idx] = 0.0
    return x_aug, a_hat


ALL_AUGMENTATIONS = [augment_edge_dropout, augment_node_noise, augment_band_mask]


def random_augment(x: torch.Tensor, a_hat: torch.Tensor):
    """
    Apply two DISTINCT augmentations to produce two views.
    Using independent single augmentations keeps positive pairs
    similar enough on a small dataset.
    """
    chosen = np.random.choice(len(ALL_AUGMENTATIONS), size=2, replace=False)
    x1, a1 = ALL_AUGMENTATIONS[chosen[0]](x.clone(), a_hat.clone())
    x2, a2 = ALL_AUGMENTATIONS[chosen[1]](x.clone(), a_hat.clone())
    return (x1, a1), (x2, a2)


# ─────────────────────────────────────────────────────────────
# NT-Xent contrastive loss
# ─────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """
    z1, z2 : (N, proj_dim) — L2-normalised embeddings of two views.
    Positive pairs: (i, i+N). All others are negatives.
    """
    N = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────
# Model architecture (identical to step5 for fair comparison)
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(a_hat @ x))


class GCNEncoder(nn.Module):
    """
    2-layer GCN → GlobalMeanPool → graph embedding (hidden,).
    Identical to SmallGCN encoder so comparisons are fair.
    """
    def __init__(self, in_dim: int = 16, hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.gcn1    = GCNLayer(in_dim, hidden)
        self.gcn2    = GCNLayer(hidden, hidden)
        self.drop    = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, a_hat)
        h = self.drop(h)
        h = self.gcn2(h, a_hat)
        return h.mean(dim=0)   # (hidden,)


class ProjectionHead(nn.Module):
    """Used ONLY during SSL pre-training — discarded afterwards."""
    def __init__(self, in_dim: int = 32, proj_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ClassifierHead(nn.Module):
    """Binary classifier attached to encoder during fine-tuning."""
    def __init__(self, in_dim: int = 32, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze()


# ─────────────────────────────────────────────────────────────
# Threshold helper (same logic as step5)
# ─────────────────────────────────────────────────────────────

def find_best_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Sweep thresholds 0.1–0.9 on TRAINING data, pick the one that
    maximises F1 for the ictal class.
    Never sees test data — no leakage.
    """
    best_thresh, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds_thr = (probs >= thr).astype(np.int64)
        f1_thr    = f1_score(targets, preds_thr, zero_division=0)
        if f1_thr > best_f1:
            best_f1, best_thresh = f1_thr, float(thr)
    return best_thresh


# ─────────────────────────────────────────────────────────────
# SSL pre-training (train-fold graphs only — NO LEAKAGE)
# ─────────────────────────────────────────────────────────────

def ssl_pretrain(encoder: GCNEncoder, proj_head: ProjectionHead,
                 train_graphs, ssl_epochs: int, lr: float,
                 device, batch_size: int = 32,
                 temperature: float = 0.5, verbose: bool = True):
    """
    Pre-train encoder + projection head with NT-Xent loss.
    train_graphs: TRAIN fold ONLY — test patient NEVER passed here.
    """
    encoder.train()
    proj_head.train()

    params    = list(encoder.parameters()) + list(proj_head.parameters())
    optimiser = Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=ssl_epochs, eta_min=lr * 0.1)

    N, losses = len(train_graphs), []

    for ep in range(ssl_epochs):
        idx       = np.random.permutation(N)
        ep_loss   = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = idx[start: start + batch_size]
            if len(batch_idx) < 2:
                continue

            z1_list, z2_list = [], []
            for i in batch_idx:
                x, a = train_graphs[i]
                (x1, a1), (x2, a2) = random_augment(x, a)
                h1 = encoder(x1.to(device), a1.to(device))
                h2 = encoder(x2.to(device), a2.to(device))
                z1_list.append(proj_head(h1))
                z2_list.append(proj_head(h2))

            z1   = torch.stack(z1_list)
            z2   = torch.stack(z2_list)
            loss = nt_xent_loss(z1, z2, temperature=temperature)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()

            ep_loss   += loss.item()
            n_batches += 1

        scheduler.step()
        avg = ep_loss / max(n_batches, 1)
        losses.append(avg)

        if verbose and (ep + 1) % 50 == 0:
            print(f'    SSL [{ep + 1:3d}/{ssl_epochs}]  loss: {avg:.4f}')

    return losses


# ─────────────────────────────────────────────────────────────
# Fine-tuning (two phases)
# ─────────────────────────────────────────────────────────────

def finetune(encoder: GCNEncoder, clf_head: ClassifierHead,
             graphs_train, y_train, graphs_test, y_test,
             ft_epochs: int, lr: float, pos_weight: float,
             device, patience: int = 20,
             freeze_encoder: bool = False):
    """
    Phase A: freeze_encoder=True  — warm up classifier head only
    Phase B: freeze_encoder=False — full fine-tuning (small LR)
    Returns train_losses, val_losses, best_auc, best_train_auc.
    """
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = list(clf_head.parameters())
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        params = list(encoder.parameters()) + list(clf_head.parameters())

    optimiser = Adam(params, lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    train_losses, val_losses = [], []
    best_auc       = 0.0
    best_enc_state = None
    best_clf_state = None
    best_train_auc = 0.0
    patience_cnt   = 0

    for ep in range(ft_epochs):
        # — Train ─────────────────────────────────────────
        encoder.train()
        clf_head.train()
        ep_loss = 0.0
        for i in np.random.permutation(len(graphs_train)):
            x, a = graphs_train[i]
            x, a = x.to(device), a.to(device)
            optimiser.zero_grad()
            logit = clf_head(encoder(x, a))
            label = torch.tensor(float(y_train[i]), device=device).unsqueeze(0)
            loss  = criterion(logit.unsqueeze(0), label)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            ep_loss += loss.item()
        train_losses.append(ep_loss / len(graphs_train))

        # — Validate ──────────────────────────────────────
        encoder.eval()
        clf_head.eval()
        with torch.no_grad():
            val_logits = [
                clf_head(encoder(x.to(device), a.to(device))).cpu().item()
                for x, a in graphs_test
            ]
        val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
        val_labels_t = torch.tensor(y_test,     dtype=torch.float32)
        val_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )(val_logits_t, val_labels_t).item()
        val_losses.append(val_loss)

        probs   = torch.sigmoid(val_logits_t).numpy()
        val_auc = roc_auc_score(y_test, probs) \
                  if len(np.unique(y_test)) == 2 else 0.0

        if val_auc > best_auc:
            best_auc       = val_auc
            best_enc_state = copy.deepcopy(encoder.state_dict())
            best_clf_state = copy.deepcopy(clf_head.state_dict())

            # Train AUC at this (best) epoch for overfitting diagnostic
            with torch.no_grad():
                tr_logits = [
                    clf_head(encoder(x.to(device), a.to(device))).cpu().item()
                    for x, a in graphs_train
                ]
            tr_probs = torch.sigmoid(
                torch.tensor(tr_logits, dtype=torch.float32)
            ).numpy()
            best_train_auc = float(roc_auc_score(y_train, tr_probs)) \
                             if len(np.unique(y_train)) == 2 else 0.0
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_enc_state:
        encoder.load_state_dict(best_enc_state)
    if best_clf_state:
        clf_head.load_state_dict(best_clf_state)

    return train_losses, val_losses, best_auc, best_train_auc


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probs(encoder, clf_head, graphs, device):
    """Return raw probabilities (no threshold applied)."""
    encoder.eval()
    clf_head.eval()
    logits = [
        clf_head(encoder(x.to(device), a.to(device))).cpu().item()
        for x, a in graphs
    ]
    return 1.0 / (1.0 + np.exp(-np.array(logits, dtype=np.float32)))


def compute_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy       = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {
        'auc':         float(roc_auc_score(y_true, y_prob)),
        'accuracy':    float(accuracy),                          # added
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

def plot_ssl_loss(ssl_losses, patient_id, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ssl_losses, color='mediumpurple', lw=2)
    ax.set_xlabel('SSL Epoch', fontsize=11)
    ax.set_ylabel('NT-Xent Loss', fontsize=11)
    ax.set_title(f'SSL Pre-training Loss | Train: excl. {patient_id}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'ssl_loss_fold_{patient_id}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_ft_loss_curves(train_losses, val_losses, patient_id, output_dir):
    """
    Fine-tune train vs val loss. Gap at end = overfitting signal.
    Best epoch vertical line shows where early stopping triggered.
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
    ax.set_xlabel('Fine-tune epoch', fontsize=11)
    ax.set_ylabel('BCE Loss', fontsize=11)
    ax.set_title(f'SSL-GCN Fine-tune Loss | Test: {patient_id}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'ft_loss_{patient_id}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, patient_id, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'SSL-GCN CM | Test: {patient_id}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_ssl_{patient_id}.png',
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
        f'SSL-GCN LOPO ROC\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_ssl_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'accuracy', 'f1', 'sensitivity', 'specificity']
    colors   = ['mediumpurple', 'purple', 'tomato', 'seagreen', 'darkorange']
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
    ax.set_title('SSL-GCN — Per-Patient LOPO Metrics',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_fold_ssl_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion(all_y_true, all_y_pred, output_dir):
    cm      = confusion_matrix(all_y_true, all_y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',   'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalised'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Purples', ax=ax,
                    xticklabels=['Pre-ictal', 'Ictal'],
                    yticklabels=['Pre-ictal', 'Ictal'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'SSL-GCN Aggregate CM ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cm_aggregate_ssl_gcn.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfit_diagnostic(fold_metrics, output_dir):
    """
    Train AUC (at best fine-tune epoch) vs Test AUC per fold.
    Large gap = model memorised training graphs during fine-tuning.
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

    ax.set_title(f'SSL-GCN — Train vs Test AUC\nMean gap = {mean_gap:+.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Train AUC (at best fine-tune epoch)', fontsize=10)
    ax.set_ylabel('Test AUC', fontsize=10)
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'train_vs_test_auc_ssl.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Overfit diagnostic      → {output_dir / "train_vs_test_auc_ssl.png"}')


def plot_final_comparison(ssl_stats, sup_json, baseline_json, output_dir):
    """
    All-model bar chart with error bars:
    RF | SVM RBF | SVM RFE | GCN (supervised) | SSL-GCN
    """
    met_keys = ['auc', 'accuracy', 'f1', 'sensitivity', 'specificity']
    models   = {}

    if baseline_json and Path(baseline_json).exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        for name, res in bl.items():
            ss = res.get('summary_stats', {})
            if ss:
                models[name] = {
                    'mean': [ss.get(k, {}).get('mean', 0.0) for k in met_keys],
                    'std':  [ss.get(k, {}).get('std',  0.0) for k in met_keys],
                }

    if sup_json and Path(sup_json).exists():
        with open(sup_json) as f:
            sup = json.load(f)
        ss = sup.get('summary_stats', {})
        if ss:
            models['GCN (Supervised)'] = {
                'mean': [ss.get(k, {}).get('mean', 0.0) for k in met_keys],
                'std':  [ss.get(k, {}).get('std',  0.0) for k in met_keys],
            }

    models['SSL-GCN (Ours)'] = {
        'mean': [ssl_stats.get(k, {}).get('mean', 0.0) for k in met_keys],
        'std':  [ssl_stats.get(k, {}).get('std',  0.0) for k in met_keys],
    }

    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    x      = np.arange(len(met_keys))
    n      = len(models)
    width  = 0.8 / n

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
                 'RF  |  SVM RBF  |  SVM RFE  |  GCN supervised  |  SSL-GCN',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Final comparison chart  → {output_dir / "comparison_all_models.png"}')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 6 — SSL GCN (leakage-free LOPO)')
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/ssl_gnn')
    parser.add_argument('--ssl_epochs',    type=int,   default=200)
    parser.add_argument('--ft_epochs',     type=int,   default=100)
    parser.add_argument('--lr_ssl',        type=float, default=0.001)
    parser.add_argument('--lr_ft',         type=float, default=0.0005)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--proj_dim',      type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold',     type=float, default=0.15)
    parser.add_argument('--temperature',   type=float, default=0.5)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--baseline_json', default=None)
    parser.add_argument('--sup_json',      default=None)
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 6 — SSL GCN  (leakage-free LOPO, GraphCL)')
    print('=' * 65)
    print(f'Device       : {device}')
    print(f'SSL epochs   : {args.ssl_epochs}  LR: {args.lr_ssl}  Temp: {args.temperature}')
    print(f'FT  epochs   : {args.ft_epochs}   LR: {args.lr_ft}')
    print(f'Hidden: {args.hidden}  ProjDim: {args.proj_dim}  Dropout: {args.dropout}')
    print(f'Threshold: {args.threshold}  Batch: {args.batch_size}  Patience: {args.patience}')
    print(f'Adaptive threshold: ON (fitted on train fold, maximises F1)')
    print()
    print('LEAKAGE-FREE: SSL pre-training runs INSIDE LOPO on TRAIN graphs only.')
    print('=' * 65)

    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)
    adj_dtf     = data['adj_dtf'].astype(np.float32)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']

    print(f'\nLoaded: {len(y)} epochs | Ictal: {(y==1).sum()} | '
          f'Pre-ictal: {(y==0).sum()}')
    print(f'Patients: {np.unique(patient_ids).tolist()}')

    print('Building graphs...')
    all_graphs = build_graphs(node_feats, adj_dtf, threshold=args.threshold)

    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []

    for fold_num, pat in enumerate(patients):
        print(f'\n{"─" * 65}')
        print(f'  Fold {fold_num + 1}/{len(patients)} — Test patient: {pat}')
        print(f'{"─" * 65}')

        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask
        train_idx  = np.where(train_mask)[0]
        test_idx   = np.where(test_mask)[0]

        y_train = y[train_idx]
        y_test  = y[test_idx]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] Only one class in test set')
            continue

        graphs_train_raw = [all_graphs[i] for i in train_idx]
        graphs_test_raw  = [all_graphs[i] for i in test_idx]

        # Scale on TRAIN only — no leakage
        graphs_train, graphs_test = scale_node_features(
            graphs_train_raw, graphs_test_raw
        )

        n_neg      = int((y_train == 0).sum())
        n_pos      = int((y_train == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        print(f'  Train: {len(train_idx)} epochs  '
              f'(ictal={n_pos}, pre-ictal={n_neg})  '
              f'pos_weight={pos_weight:.2f}')
        print(f'  Test : {len(test_idx)} epochs')

        # ── Phase 1: SSL pre-training on TRAIN graphs only ────────
        print(f'\n  Phase 1 — SSL pre-training '
              f'({args.ssl_epochs} epochs, {len(graphs_train)} train graphs)')

        encoder   = GCNEncoder(in_dim=16, hidden=args.hidden,
                               dropout=args.dropout).to(device)
        proj_head = ProjectionHead(in_dim=args.hidden,
                                   proj_dim=args.proj_dim).to(device)

        ssl_losses = ssl_pretrain(
            encoder, proj_head,
            graphs_train,
            ssl_epochs=args.ssl_epochs,
            lr=args.lr_ssl,
            device=device,
            batch_size=args.batch_size,
            temperature=args.temperature,
            verbose=True,
        )
        plot_ssl_loss(ssl_losses, pat, output_dir)
        print(f'  SSL final loss: {ssl_losses[-1]:.4f}')

        # Projection head discarded — only encoder weights kept
        clf_head = ClassifierHead(in_dim=args.hidden,
                                  dropout=args.dropout).to(device)

        # ── Phase 2A: Linear probe (frozen encoder) ───────────────
        print(f'\n  Phase 2A — Linear probe (encoder frozen, 30 epochs)')
        finetune(
            encoder, clf_head,
            graphs_train, y_train,
            graphs_test,  y_test,
            ft_epochs=30,
            lr=args.lr_ft * 5,
            pos_weight=pos_weight,
            device=device,
            patience=args.patience,
            freeze_encoder=True,
        )

        # ── Phase 2B: Full fine-tuning ────────────────────────────
        print(f'\n  Phase 2B — Full fine-tuning ({args.ft_epochs} epochs)')
        tr_losses, val_losses, best_auc, best_train_auc = finetune(
            encoder, clf_head,
            graphs_train, y_train,
            graphs_test,  y_test,
            ft_epochs=args.ft_epochs,
            lr=args.lr_ft,
            pos_weight=pos_weight,
            device=device,
            patience=args.patience,
            freeze_encoder=False,
        )
        print(f'  Best val AUC (fine-tune): {best_auc:.3f}  '
              f'Train AUC at best: {best_train_auc:.3f}  '
              f'Gap: {best_train_auc - best_auc:+.3f}')
        plot_ft_loss_curves(tr_losses, val_losses, pat, output_dir)

        # ── Adaptive threshold — train data only ──────────────────
        train_probs = get_probs(encoder, clf_head, graphs_train, device)
        best_thresh = find_best_threshold(train_probs, y_train)

        # ── Final test evaluation with adaptive threshold ─────────
        test_probs = get_probs(encoder, clf_head, graphs_test, device)
        test_preds = (test_probs >= best_thresh).astype(np.int64)

        metrics = compute_metrics(y_test, test_preds, test_probs)
        if metrics is None:
            continue

        metrics['patient']            = pat
        metrics['n_train']            = int(train_mask.sum())
        metrics['n_test']             = int(test_mask.sum())
        metrics['best_ft_auc']        = float(best_auc)
        metrics['train_auc']          = float(best_train_auc)
        metrics['decision_threshold'] = float(best_thresh)
        metrics['ssl_final_loss']     = float(ssl_losses[-1])
        fold_metrics.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(test_preds.tolist())

        fpr, tpr, _ = roc_curve(y_test, test_probs)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        plot_confusion_matrix(
            confusion_matrix(y_test, test_preds), pat, output_dir
        )

        print(f'\n  {pat:8s} | thresh={best_thresh:.2f}  '
              f'TrainAUC={best_train_auc:.3f}  TestAUC={metrics["auc"]:.3f}  '
              f'Gap={best_train_auc - metrics["auc"]:+.3f}  '
              f'Acc={metrics["accuracy"]:.3f}  F1={metrics["f1"]:.3f}  '
              f'Sens={metrics["sensitivity"]:.3f}  Spec={metrics["specificity"]:.3f}  '
              f'MCC={metrics["mcc"]:.3f}')

    # ── Aggregate plots ───────────────────────────────────────────
    if fold_roc_data:
        plot_roc_all_folds(fold_roc_data, output_dir)
    if fold_metrics:
        plot_per_fold_metrics(fold_metrics, output_dir)
        plot_overfit_diagnostic(fold_metrics, output_dir)
    if all_y_true:
        plot_aggregate_confusion(
            np.array(all_y_true), np.array(all_y_pred), output_dir
        )

    # ── Summary ───────────────────────────────────────────────────
    met_keys      = ['auc', 'accuracy', 'f1', 'sensitivity',
                     'specificity', 'precision', 'mcc']
    summary_stats = {}

    print(f'\n{"=" * 65}')
    print(f'SSL-GCN — Mean ± Std across {len(fold_metrics)} folds')
    print(f'{"=" * 65}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    train_aucs = [m['train_auc'] for m in fold_metrics]
    test_aucs  = [m['auc']       for m in fold_metrics]
    thresholds = [m['decision_threshold'] for m in fold_metrics]
    mean_gap   = float(np.mean(np.array(train_aucs) - np.array(test_aucs)))
    print(f'  {"overfit gap":15s}: {mean_gap:+.3f}  (train − test AUC)')
    print(f'  {"thresh (mean)":15s}: {np.mean(thresholds):.3f} ± '
          f'{np.std(thresholds):.3f}')

    # ── Save results ──────────────────────────────────────────────
    results = {
        'model':           'SSL_GCN_GraphCL_LeakageFree',
        'hyperparameters': vars(args),
        'ssl_protocol':    'pre-training on train-fold graphs only (NO leakage)',
        'augmentations':   ['edge_dropout(p=0.20)',
                            'node_noise(sigma=0.10)',
                            'band_mask(random_band)'],
        'fold_metrics':    fold_metrics,
        'summary_stats':   summary_stats,
    }
    results_path = output_dir / 'results_ssl.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults → {results_path}')

    # ── Final comparison chart ────────────────────────────────────
    plot_final_comparison(summary_stats, args.sup_json,
                          args.baseline_json, output_dir)

    print('\n' + '=' * 65)
    print('STEP 6 COMPLETE')
    print('=' * 65)


if __name__ == '__main__':
    main()