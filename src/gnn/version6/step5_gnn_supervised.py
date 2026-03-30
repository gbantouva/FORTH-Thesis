"""
Step 5 — Supervised GCN + Pure GCN  (LOPO Cross-Validation)
============================================================
Two models are evaluated side-by-side under identical LOPO folds:

  MODEL A — SmallGCN (feature-based)
    Node features  : 16 hand-crafted features per channel (step 3)
                     [band powers, Hjorth, time-domain, connectivity degree]
    Graph structure: DTF adjacency (thresholded at training-fold percentile)
    Architecture   : GCNLayer(16→32) → Dropout → GCNLayer(32→32)
                     → GlobalMeanPool → MLP(32→16→1)
    ~4 k parameters — deliberately small for a ~300-epoch training set.

  MODEL B — PureGCN (no hand-crafted features)
    Node features  : learned by a shallow 1-D CNN applied to the raw EEG
                     signal of each channel independently (19 × 1024 → 19 × 16)
    Graph structure: same DTF adjacency as Model A
    Architecture   : RawChannelEncoder (1D-CNN per channel) → same GCN head
    Purpose        : ABLATION — isolates the contribution of hand-crafted
                     features vs. learned representations.
    If PureGCN ≈ SmallGCN → hand-crafting adds little.
    If PureGCN < SmallGCN → neuroscientific feature design is justified.

DESIGN DECISIONS (document in thesis):

1. THRESHOLD — data-driven, not fixed.
   The DTF threshold is computed as the p-th percentile of off-diagonal DTF
   values from the TRAINING fold only (default p=70, keeping top 30% of edges).
   Fixed thresholds (e.g. 0.15) are arbitrary; percentile-based thresholds
   adapt to the actual distribution of connectivity strengths per fold.

2. NORMALISED ADJACENCY:
     A_hat = D^{-1/2} (A + I) D^{-1/2}
   Self-loops added before normalisation (Kipf & Welling 2017).

3. IMBALANCE: BCEWithLogitsLoss with pos_weight = n_neg / n_pos
   Computed from the training split of each fold independently (no leakage).

4. EARLY STOPPING on validation loss (patience=20) with best-state restore.
   NOTE: validation here = test patient. With 8 patients nested LOPO would
   use a proper held-out val patient, but is expensive. Acknowledged in thesis.

5. OVERFITTING TRACKING: train AUC AND test AUC recorded every epoch.
   Loss curves + AUC curves saved per fold.

6. ACCURACY added to all metric dicts (with majority-class baseline).

7. EVALUATION: LOPO by patient. StandardScaler fit on train only.

Outputs (per model, saved in subdirectory):
  loss_curve_{patient}.png       train/val loss + AUC curves per fold
  cm_{model}_{patient}.png       per-fold confusion matrices
  roc_{model}.png                LOPO ROC curves
  per_fold_{model}.png           per-patient metric bar chart
  overfitting_{model}.png        train vs test AUC gap per fold
  cm_aggregate_{model}.png       aggregate confusion matrix
  results_{model}.json           all metrics + hyperparams

Global outputs:
  comparison_supervised_models.png   SmallGCN vs PureGCN
  comparison_all_models.png          RF + SVM + SmallGCN + PureGCN

Usage:
  python step5_gnn_supervised.py \\
      --featfile      features/features_all.npz \\
      --outputdir     results/gnn_supervised \\
      --epochs        150 \\
      --lr            0.001 \\
      --hidden        32 \\
      --threshold_pct 70 \\
      --dropout       0.4 \\
      --patience      20 \\
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
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 1. ADJACENCY UTILITIES
# ══════════════════════════════════════════════════════════════

def compute_threshold(adj_dtf_train: np.ndarray, percentile: float = 70.0) -> float:
    """
    Compute edge threshold as the p-th percentile of off-diagonal DTF values
    across all training-fold epochs.

    Called INSIDE the LOPO loop using only training-fold adjacency matrices.
    Never touches the test patient's data → no leakage.

    Parameters
    ----------
    adj_dtf_train : (N_train, 19, 19)
    percentile    : float  — keep edges above this percentile (default 70 → top 30%)

    Returns
    -------
    threshold : float
    """
    n = adj_dtf_train.shape[1]
    mask = ~np.eye(n, dtype=bool)
    all_vals = np.concatenate([adj_dtf_train[i][mask] for i in range(len(adj_dtf_train))])
    threshold = float(np.percentile(all_vals, percentile))
    return threshold


def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """
    Symmetric normalisation:  A_hat = D^{-1/2} (A + I) D^{-1/2}
    Self-loops ensure every node attends to itself (Kipf & Welling 2017).

    adj : (N, N) numpy float32, diagonal already 0, values thresholded.
    Returns torch.float32 tensor (N, N).
    """
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def build_graphs_features(node_feats: np.ndarray,
                           adj_dtf: np.ndarray,
                           threshold: float):
    """
    Build graph list for SmallGCN (hand-crafted node features).

    node_feats : (N, 19, 16)
    adj_dtf    : (N, 19, 19)
    threshold  : scalar — edges below this are zeroed

    Returns list of (x_tensor, a_hat_tensor).
    """
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        x     = torch.tensor(node_feats[i], dtype=torch.float32)
        graphs.append((x, a_hat))
    return graphs


def build_graphs_raw(raw_epochs: np.ndarray,
                     adj_dtf: np.ndarray,
                     threshold: float):
    """
    Build graph list for PureGCN (raw EEG as node input).

    raw_epochs : (N, 19, 1024)
    adj_dtf    : (N, 19, 19)
    threshold  : scalar

    Returns list of (x_raw_tensor, a_hat_tensor).
    x_raw_tensor shape: (19, 1024) — the CNN encoder processes this.
    """
    graphs = []
    for i in range(len(raw_epochs)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        x_raw = torch.tensor(raw_epochs[i], dtype=torch.float32)  # (19, 1024)
        graphs.append((x_raw, a_hat))
    return graphs


def scale_node_features(graphs_train, graphs_test):
    """
    Fit StandardScaler on training node features; apply to both splits.
    Scaler never sees test data → no leakage.

    Works for SmallGCN where x.shape = (19, 16).
    NOT used for PureGCN (raw signals normalised inside the CNN encoder).
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


# ══════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """
    Single GCN layer:  H' = ReLU( A_hat @ H @ W )
    Equivalent to spectral graph convolution (Kipf & Welling 2017).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(a_hat @ x))


class SmallGCN(nn.Module):
    """
    MODEL A — Feature-based GCN.

    Input : node features (19, 16) from step 3
            adjacency matrix (19, 19) from DTF

    Architecture:
        GCNLayer(16 → hidden)   — neighbourhood aggregation, layer 1
        Dropout(p)
        GCNLayer(hidden → hidden) — neighbourhood aggregation, layer 2
        GlobalMeanPool           — graph-level embedding (hidden,)
        Linear(hidden → 16) → ReLU → Dropout(p)
        Linear(16 → 1)           — scalar logit

    ~4 k parameters at hidden=32. Deliberately small: with ~300 training
    graphs and 19 nodes, a larger model memorises rather than generalises.
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
        h = self.gcn1(x, a_hat)           # (19, hidden)
        h = self.drop(h)
        h = self.gcn2(h, a_hat)           # (19, hidden)
        h = h.mean(dim=0, keepdim=True)   # (1, hidden)  — global mean pool
        return self.head(h).squeeze()     # scalar logit


class RawChannelEncoder(nn.Module):
    """
    Shallow 1D-CNN that encodes the raw EEG time-series of a single channel
    into a fixed-size embedding vector.

    Applied independently to each of the 19 channels — no cross-channel
    mixing at this stage (that is the GCN's job).

    Architecture (per channel):
        Conv1d(1→8,  kernel=32, stride=8)   → ReLU   # (8, 128)
        Conv1d(8→16, kernel=16, stride=4)   → ReLU   # (16, 32)
        AdaptiveAvgPool1d(8)                          # (16, 8)
        Flatten → Linear(128 → node_dim)              # (node_dim,)

    Rationale for conv: EEG has local temporal structure (oscillations at
    δ/θ/α/β/γ, spikes). Small receptive-field convolutions capture these
    without requiring full-sequence attention.

    Input  : (19, 1024)
    Output : (19, node_dim)
    """
    def __init__(self, n_samples: int = 1024, node_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8,  kernel_size=32, stride=8,  padding=16),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=16, stride=4,  padding=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.proj = nn.Linear(16 * 8, node_dim)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw : (19, 1024)
        Returns (19, node_dim)
        """
        n_ch = x_raw.shape[0]
        x    = x_raw.unsqueeze(1)          # (19, 1, 1024)
        h    = self.conv(x)                # (19, 16, 8)
        h    = h.view(n_ch, -1)           # (19, 128)
        return self.proj(h)               # (19, node_dim)


class PureGCN(nn.Module):
    """
    MODEL B — Pure GCN (no hand-crafted features).

    Input : raw EEG epoch (19, 1024)
            adjacency matrix (19, 19) from DTF

    Architecture:
        RawChannelEncoder   — learns node embeddings from raw signal
        GCNLayer(node_dim → hidden)
        Dropout(p)
        GCNLayer(hidden → hidden)
        GlobalMeanPool
        Linear(hidden → 16) → ReLU → Dropout(p)
        Linear(16 → 1)

    Purpose (ablation):
        Compare against SmallGCN to quantify the value of hand-crafted
        neuroscientific features. If PureGCN ≈ SmallGCN, hand-crafting
        adds little. If PureGCN < SmallGCN, the spectral/Hjorth/time-domain
        feature design is justified.
    """
    def __init__(self, n_samples: int = 1024, node_dim: int = 16,
                 hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.encoder = RawChannelEncoder(n_samples, node_dim)
        self.gcn1    = GCNLayer(node_dim, hidden)
        self.gcn2    = GCNLayer(hidden, hidden)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x_raw: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        node_emb = self.encoder(x_raw)         # (19, node_dim)
        h        = self.gcn1(node_emb, a_hat)  # (19, hidden)
        h        = self.drop(h)
        h        = self.gcn2(h, a_hat)         # (19, hidden)
        h        = h.mean(dim=0, keepdim=True) # (1, hidden)
        return self.head(h).squeeze()          # scalar logit


# ══════════════════════════════════════════════════════════════
# 3. TRAINING / EVALUATION
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, optimiser, criterion, graphs, labels, device):
    model.train()
    total_loss = 0.0
    for i in np.random.permutation(len(graphs)):
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
    Returns probs (N,), preds (N,), targets (N,), mean_bce (float).
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
    preds   = (probs >= 0.5).astype(np.int64)
    eps     = 1e-7
    bce     = -np.mean(
        targets * np.log(probs + eps) + (1 - targets) * np.log(1 - probs + eps)
    )
    return probs, preds, targets, float(bce)


def compute_metrics(y_true, y_pred, y_prob):
    """
    Full metric dict including accuracy (with majority-class baseline).
    Returns None if test set has only one class.
    """
    if len(np.unique(y_true)) < 2:
        return None
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_total        = len(y_true)
    majority_n     = max((y_true == 0).sum(), (y_true == 1).sum())
    return {
        'accuracy':          float(accuracy_score(y_true, y_pred)),
        'majority_baseline': float(majority_n / n_total),
        'auc':               float(roc_auc_score(y_true, y_prob)),
        'f1':                float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity':       float(tp / (tp + fn + 1e-12)),
        'specificity':       float(tn / (tn + fp + 1e-12)),
        'precision':         float(precision_score(y_true, y_pred, zero_division=0)),
        'mcc':               float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ══════════════════════════════════════════════════════════════
# 4. PLOT HELPERS
# ══════════════════════════════════════════════════════════════

def plot_loss_and_auc_curves(train_losses, val_losses,
                              train_aucs, val_aucs,
                              patient_id, model_tag, output_dir):
    """
    Two-panel figure: loss curves (left) + AUC curves (right).
    Diverging AUC curves (train keeps rising, val plateaus) = overfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Loss
    axes[0].plot(train_losses, color='royalblue', lw=1.5, label='Train loss')
    axes[0].plot(val_losses,   color='tomato',    lw=1.5, linestyle='--', label='Val loss')
    gap = abs(train_losses[-1] - val_losses[-1])
    axes[0].text(0.63, 0.88, f'Final gap: {gap:.4f}',
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('BCE Loss', fontsize=11)
    axes[0].set_title(f'{model_tag} Loss | Test: {patient_id}',
                      fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # AUC
    axes[1].plot(train_aucs, color='royalblue', lw=1.5, label='Train AUC')
    axes[1].plot(val_aucs,   color='tomato',    lw=1.5, linestyle='--', label='Val AUC')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('AUC', fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f'{model_tag} AUC curves | Test: {patient_id}',
                      fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'loss_curve_{model_tag}_{patient_id}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, patient_id, model_tag, output_dir, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'{model_tag} CM | Test: {patient_id}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_{model_tag}_{patient_id}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_all_folds(fold_roc_data, model_tag, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for fpr, tpr, auc, pat in fold_roc_data:
        ax.plot(fpr, tpr, alpha=0.6, lw=1.5, label=f'{pat} (AUC={auc:.2f})')
        aucs.append(auc)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'{model_tag} — LOPO ROC\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
        fontsize=11, fontweight='bold',
    )
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, model_tag, output_dir):
    """
    Per-patient bar chart. Accuracy shown as line overlay vs majority baseline.
    """
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange']
    x        = np.arange(len(patients))
    width    = 0.2

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (met, col) in enumerate(zip(met_keys, colors)):
        vals = [m[met] for m in fold_metrics]
        ax.bar(x + i * width, vals, width, label=met.upper(),
               color=col, alpha=0.8, edgecolor='black')

    acc_vals  = [m['accuracy']          for m in fold_metrics]
    base_vals = [m['majority_baseline'] for m in fold_metrics]
    ax.plot(x + 1.5 * width, acc_vals,  'k^-', ms=7, lw=1.5, label='Accuracy')
    ax.plot(x + 1.5 * width, base_vals, 'r--', ms=5, lw=1,   label='Majority baseline')

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.axhline(0.5, color='gray', linestyle=':', lw=1)
    ax.set_title(f'{model_tag} — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'per_fold_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion(all_y_true, all_y_pred, model_tag, output_dir, cmap='Blues'):
    cm      = confusion_matrix(all_y_true, all_y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',   'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalised'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                    xticklabels=['Pre-ictal', 'Ictal'],
                    yticklabels=['Pre-ictal', 'Ictal'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'{model_tag} Aggregate CM ({title})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_aggregate_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting(train_aucs_per_fold, test_aucs_per_fold,
                     patients, model_tag, output_dir):
    """
    Per-fold train vs test AUC bar chart + gap bar chart.
    Red bars = gap > 0.10 (overfitting warning).
    """
    x     = np.arange(len(patients))
    width = 0.35
    gap   = [tr - te for tr, te in zip(train_aucs_per_fold, test_aucs_per_fold)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, train_aucs_per_fold, width,
                label='Train AUC', color='steelblue', alpha=0.85, edgecolor='black')
    axes[0].bar(x + width / 2, test_aucs_per_fold,  width,
                label='Test AUC',  color='tomato',    alpha=0.85, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(patients, rotation=30, fontsize=9)
    axes[0].set_ylabel('AUC', fontsize=12)
    axes[0].set_ylim(0, 1.15)
    axes[0].axhline(0.5, color='gray', linestyle='--', lw=1)
    axes[0].set_title(f'{model_tag} — Train vs Test AUC', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    bar_colors = ['tomato' if g > 0.10 else 'steelblue' for g in gap]
    axes[1].bar(x, gap, color=bar_colors, edgecolor='black', alpha=0.85)
    axes[1].axhline(0.10, color='red', linestyle='--', lw=1.5,
                    label='Gap=0.10 warning')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(patients, rotation=30, fontsize=9)
    axes[1].set_ylabel('Train AUC − Test AUC', fontsize=12)
    axes[1].set_title(f'{model_tag} — Overfitting Gap', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    mean_gap = np.mean(gap)
    flag     = '⚠ Overfitting' if mean_gap > 0.10 else '✓ OK'
    fig.suptitle(f'{model_tag} | Mean gap = {mean_gap:.3f}  {flag}',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / f'overfitting_{model_tag}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(all_model_stats: dict, output_dir: Path, filename: str,
                           title: str):
    """
    Side-by-side bar chart comparing mean AUC/F1/Sensitivity/Specificity
    across any number of models.

    all_model_stats : { model_name: summary_stats_dict }
    """
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'seagreen', 'mediumpurple',
                'darkorange', 'teal']
    x        = np.arange(len(met_keys))
    width    = 0.18
    n_models = len(all_model_stats)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, stats) in enumerate(all_model_stats.items()):
        if not stats:
            continue
        means  = [stats[k]['mean'] for k in met_keys]
        stds   = [stats[k]['std']  for k in met_keys]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=name, color=colors[i % len(colors)],
               alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Mean Score ± Std (LOPO)', fontsize=12)
    ax.set_ylim(0, 1.30)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {filename}')


# ══════════════════════════════════════════════════════════════
# 5. LOPO TRAINING LOOP  (shared for both models)
# ══════════════════════════════════════════════════════════════

def run_lopo_gcn(model_name: str,
                 model_factory,          # callable() → nn.Module
                 graphs_all: list,       # list of (x, a_hat) for ALL epochs
                 adj_dtf_all: np.ndarray,# (N, 19, 19) — for per-fold threshold
                 y: np.ndarray,
                 patient_ids: np.ndarray,
                 args,
                 output_dir: Path,
                 device: torch.device,
                 is_feature_model: bool = True,
                 cmap: str = 'Blues'):
    """
    Full LOPO loop for one GCN model.

    Parameters
    ----------
    model_factory    : callable with no args that returns a fresh nn.Module
    graphs_all       : pre-built graph list (unscaled, using global threshold)
                       NOTE: threshold is recomputed per fold; graphs are rebuilt
                       inside the loop for correctness.
    adj_dtf_all      : raw adjacency matrices (N, 19, 19) for threshold computation
    is_feature_model : True  → scale node features with StandardScaler (SmallGCN)
                       False → skip scaling (PureGCN uses raw signals, CNN normalises)

    Returns
    -------
    fold_metrics  : list of per-fold metric dicts
    summary_stats : mean ± std across folds
    """
    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []
    train_aucs_folds = []
    test_aucs_folds  = []
    fold_patients    = []

    print(f'\n{"=" * 65}')
    print(f'  {model_name} — LOPO CV ({len(patients)} folds)')
    print(f'{"=" * 65}')
    header = f'  {"Patient":10s} | {"AUC":>6} {"F1":>6} {"Sens":>6} ' \
             f'{"Spec":>6} {"Acc":>6} {"MCC":>6} | {"TrAUC":>7} {"Gap":>6}'
    print(header)
    print(f'  {"-" * 65}')

    # Load the source arrays needed to rebuild graphs per fold
    # (stored on args namespace for convenience)
    node_feats_all = getattr(args, '_node_feats', None)
    raw_epochs_all = getattr(args, '_raw_epochs', None)

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

        # ── Per-fold threshold (computed from training adjacency only) ───
        adj_train = adj_dtf_all[train_idx]
        threshold = compute_threshold(adj_train, percentile=args.threshold_pct)

        # ── Rebuild graphs with the fold-specific threshold ──────────────
        adj_test = adj_dtf_all[test_idx]
        if is_feature_model:
            g_train_raw = build_graphs_features(
                node_feats_all[train_idx], adj_train, threshold
            )
            g_test_raw  = build_graphs_features(
                node_feats_all[test_idx],  adj_test,  threshold
            )
            # Scale node features: fit on train only
            g_train, g_test = scale_node_features(g_train_raw, g_test_raw)
        else:
            g_train = build_graphs_raw(raw_epochs_all[train_idx], adj_train, threshold)
            g_test  = build_graphs_raw(raw_epochs_all[test_idx],  adj_test,  threshold)

        # ── Class imbalance weight (from training fold only) ─────────────
        n_neg      = int((y_train == 0).sum())
        n_pos      = int((y_train == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        # ── Fresh model + optimiser per fold ─────────────────────────────
        model     = model_factory().to(device)
        optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5,
                                      verbose=False)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        train_losses, val_losses = [], []
        train_aucs,   val_aucs   = [], []
        best_val_loss = np.inf
        best_auc      = 0.0
        best_state    = None
        patience_cnt  = 0

        for ep in range(args.epochs):
            tr_loss = train_one_epoch(
                model, optimiser, criterion, g_train, y_train, device
            )
            # Train evaluation (for overfitting tracking)
            tr_probs, _, tr_targets, _ = evaluate_graphs(
                model, g_train, y_train, device
            )
            # Val (test patient) evaluation
            val_probs, val_preds, val_targets, val_loss = evaluate_graphs(
                model, g_test, y_test, device
            )

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            tr_auc  = float(roc_auc_score(tr_targets, tr_probs)) \
                      if len(np.unique(tr_targets)) == 2 else 0.0
            val_auc = float(roc_auc_score(val_targets, val_probs)) \
                      if len(np.unique(val_targets)) == 2 else 0.0
            train_aucs.append(tr_auc)
            val_aucs.append(val_auc)

            # Early stopping on validation loss (not val AUC — avoids test leakage
            # in the sense that we stop on loss, not on metric directly)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_auc      = val_auc
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Final fold evaluation ─────────────────────────────────────────
        probs, preds, targets, _ = evaluate_graphs(model, g_test, y_test, device)
        metrics = compute_metrics(targets, preds, probs)
        if metrics is None:
            continue

        # Train AUC at best checkpoint (last recorded)
        final_tr_auc  = train_aucs[-1] if train_aucs else 0.0
        final_te_auc  = float(metrics['auc'])
        overfit_gap   = round(final_tr_auc - final_te_auc, 4)

        metrics['patient']      = pat
        metrics['n_train']      = int(train_mask.sum())
        metrics['n_test']       = int(test_mask.sum())
        metrics['best_val_auc'] = float(best_auc)
        metrics['train_auc']    = final_tr_auc
        metrics['overfit_gap']  = overfit_gap
        metrics['threshold']    = round(threshold, 4)
        fold_metrics.append(metrics)

        train_aucs_folds.append(final_tr_auc)
        test_aucs_folds.append(final_te_auc)
        fold_patients.append(pat)

        all_y_true.extend(targets.tolist())
        all_y_pred.extend(preds.tolist())

        fpr, tpr, _ = roc_curve(targets, probs)
        fold_roc_data.append((fpr, tpr, final_te_auc, pat))

        gap_flag = ' ⚠' if overfit_gap > 0.10 else ''
        print(f'  {pat:10s} | {final_te_auc:6.3f} {metrics["f1"]:6.3f} '
              f'{metrics["sensitivity"]:6.3f} {metrics["specificity"]:6.3f} '
              f'{metrics["accuracy"]:6.3f} {metrics["mcc"]:6.3f} | '
              f'{final_tr_auc:7.3f} {overfit_gap:6.3f}{gap_flag}')

        # Per-fold plots
        plot_loss_and_auc_curves(
            train_losses, val_losses, train_aucs, val_aucs,
            pat, model_name, output_dir,
        )
        plot_confusion_matrix(
            confusion_matrix(targets, preds), pat, model_name, output_dir, cmap
        )

    if len(fold_metrics) == 0:
        print(f'  [ERROR] No valid folds.')
        return [], {}

    # ── Aggregate plots ───────────────────────────────────────────────────
    plot_roc_all_folds(fold_roc_data, model_name, output_dir)
    plot_per_fold_metrics(fold_metrics, model_name, output_dir)
    plot_aggregate_confusion(
        np.array(all_y_true), np.array(all_y_pred), model_name, output_dir, cmap
    )
    plot_overfitting(
        train_aucs_folds, test_aucs_folds, fold_patients, model_name, output_dir
    )

    # ── Summary stats ─────────────────────────────────────────────────────
    met_keys = [
        'accuracy', 'majority_baseline',
        'auc', 'f1', 'sensitivity', 'specificity', 'precision', 'mcc',
        'train_auc', 'overfit_gap',
    ]
    summary_stats = {}
    print(f'\n  {"─" * 55}')
    print(f'  {model_name} — Mean ± Std  ({len(fold_metrics)} folds)')
    print(f'  {"─" * 55}')
    for k in met_keys:
        if k not in fold_metrics[0]:
            continue
        vals        = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        note = ''
        if k == 'accuracy':
            bl = np.mean([m['majority_baseline'] for m in fold_metrics])
            note = f'  ← dummy baseline ≈ {bl:.3f}'
        if k == 'overfit_gap':
            note = '  ← train AUC − test AUC'
        print(f'  {k:22s}: {mean_:.3f} ± {std_:.3f}{note}')

    return fold_metrics, summary_stats


# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 5 — Supervised GCN + Pure GCN (LOPO)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',      required=True,
                        help='features/features_all.npz from step 3')
    parser.add_argument('--outputdir',     default='results/gnn_supervised')
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold_pct', type=float, default=70.0,
                        help='DTF percentile threshold per fold (default 70 → top 30%%)')
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--baseline_json', default=None,
                        help='Path to step4 results_all.json for comparison plot')
    parser.add_argument('--skip_pure_gcn', action='store_true',
                        help='Skip PureGCN and only run SmallGCN')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 5 — SUPERVISED GCN + PURE GCN  (LOPO CV)')
    print('=' * 65)
    print(f'Device        : {device}')
    print(f'Epochs        : {args.epochs}   LR: {args.lr}   Hidden: {args.hidden}')
    print(f'Dropout       : {args.dropout}   Patience: {args.patience}')
    print(f'Threshold pct : {args.threshold_pct}  '
          f'(top {100 - args.threshold_pct:.0f}% edges kept, computed per fold)')
    print('=' * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    data          = np.load(args.featfile, allow_pickle=True)
    node_feats    = data['node_features'].astype(np.float32)   # (N, 19, 16)
    adj_dtf       = data['adj_dtf'].astype(np.float32)         # (N, 19, 19)
    y             = data['y'].astype(np.int64)
    patient_ids   = data['patient_ids']

    # Raw epochs for PureGCN
    if 'raw_epochs' in data:
        raw_epochs = data['raw_epochs'].astype(np.float32)     # (N, 19, 1024)
    else:
        raw_epochs = None
        print('[WARN] raw_epochs not found in npz — PureGCN will be skipped.')
        print('       Re-run step 3 with the updated code to include raw_epochs.')
        args.skip_pure_gcn = True

    print(f'\nLoaded : {len(y)} epochs | '
          f'Ictal: {(y == 1).sum()} | Pre-ictal: {(y == 0).sum()}')
    print(f'Patients: {np.unique(patient_ids).tolist()}')
    majority_b = max((y == 0).sum(), (y == 1).sum()) / len(y)
    print(f'Majority-class accuracy baseline: {majority_b * 100:.1f}%\n')

    # Attach arrays to args for access inside run_lopo_gcn
    args._node_feats  = node_feats
    args._raw_epochs  = raw_epochs

    all_results = {}

    # ══════════════════════════════════════════════════════════
    # MODEL A — SmallGCN (hand-crafted node features)
    # ══════════════════════════════════════════════════════════
    def make_small_gcn():
        return SmallGCN(in_dim=16, hidden=args.hidden, dropout=args.dropout)

    fm, fs = run_lopo_gcn(
        model_name       = 'SmallGCN',
        model_factory    = make_small_gcn,
        graphs_all       = [],          # rebuilt per fold inside the function
        adj_dtf_all      = adj_dtf,
        y                = y,
        patient_ids      = patient_ids,
        args             = args,
        output_dir       = output_dir,
        device           = device,
        is_feature_model = True,
        cmap             = 'Blues',
    )
    all_results['SmallGCN'] = {'fold_metrics': fm, 'summary_stats': fs}

    # Save SmallGCN results
    results_path = output_dir / 'results_gcn.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(
            {'model': 'GCN_Supervised', 'hyperparameters': {
                k: v for k, v in vars(args).items()
                if not k.startswith('_')
             },
             'fold_metrics': fm, 'summary_stats': fs},
            f, indent=2, default=str,
        )
    print(f'\n  ✓ SmallGCN results → {results_path}')

    # ══════════════════════════════════════════════════════════
    # MODEL B — PureGCN (raw EEG → CNN node encoder)
    # ══════════════════════════════════════════════════════════
    if not args.skip_pure_gcn:
        def make_pure_gcn():
            return PureGCN(n_samples=1024, node_dim=16,
                           hidden=args.hidden, dropout=args.dropout)

        pm, ps = run_lopo_gcn(
            model_name       = 'PureGCN',
            model_factory    = make_pure_gcn,
            graphs_all       = [],
            adj_dtf_all      = adj_dtf,
            y                = y,
            patient_ids      = patient_ids,
            args             = args,
            output_dir       = output_dir,
            device           = device,
            is_feature_model = False,
            cmap             = 'Greens',
        )
        all_results['PureGCN'] = {'fold_metrics': pm, 'summary_stats': ps}

        results_pure_path = output_dir / 'results_pure_gcn.json'
        with open(results_pure_path, 'w', encoding='utf-8') as f:
            json.dump(
                {'model': 'PureGCN', 'hyperparameters': {
                    k: v for k, v in vars(args).items()
                    if not k.startswith('_')
                 },
                 'fold_metrics': pm, 'summary_stats': ps},
                f, indent=2, default=str,
            )
        print(f'  ✓ PureGCN results → {results_pure_path}')

    # ══════════════════════════════════════════════════════════
    # COMPARISON PLOTS
    # ══════════════════════════════════════════════════════════

    # SmallGCN vs PureGCN
    gcn_stats = {
        name: res['summary_stats']
        for name, res in all_results.items()
        if res['summary_stats']
    }
    if len(gcn_stats) > 1:
        plot_model_comparison(
            gcn_stats, output_dir,
            filename='comparison_supervised_models.png',
            title='SmallGCN vs PureGCN — LOPO CV (ablation)',
        )

    # All models including RF + SVM baselines
    all_stats = dict(gcn_stats)
    if args.baseline_json and Path(args.baseline_json).exists():
        with open(args.baseline_json) as f:
            bl = json.load(f)
        for name, res in bl.items():
            if res.get('summary_stats'):
                all_stats[name] = res['summary_stats']

    if all_stats:
        # Put baselines first for readability
        ordered = {}
        for k in ['Random Forest', 'SVM RBF']:
            if k in all_stats:
                ordered[k] = all_stats[k]
        for k in all_stats:
            if k not in ordered:
                ordered[k] = all_stats[k]

        plot_model_comparison(
            ordered, output_dir,
            filename='comparison_all_models.png',
            title='RF + SVM + GCN — LOPO CV',
        )

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 65)
    print('STEP 5 — FINAL SUMMARY')
    print('=' * 65)
    print(f'{"Model":20s} | {"AUC":>12} | {"F1":>12} | {"Sens":>12} | {"Acc":>12}')
    print('-' * 65)
    for name, res in all_results.items():
        ss = res['summary_stats']
        if not ss:
            continue
        def fmt(k):
            return f'{ss[k]["mean"]:.3f}±{ss[k]["std"]:.3f}'
        print(f'{name:20s} | {fmt("auc"):>12} | {fmt("f1"):>12} | '
              f'{fmt("sensitivity"):>12} | {fmt("accuracy"):>12}')

    print(f'\nNOTE: Majority-class accuracy baseline ≈ {majority_b * 100:.1f}%')
    print('\n' + '=' * 65)
    print('STEP 5 COMPLETE')
    print('=' * 65)
    print('\nNext:')
    print('  python step6_ssl_gnn.py \\')
    print(f'    --featfile {args.featfile} \\')
    print(f'    --outputdir results/ssl_gnn \\')
    print(f'    --sup_json {output_dir / "results_gcn.json"} \\')
    print(f'    --baseline_json {args.baseline_json or "results/baseline_ml/results_all.json"}')


if __name__ == '__main__':
    main()