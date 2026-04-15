"""
Step 5 — Supervised GCN + Adjacency-Only GCN  (Nested LOPO CV)
================================================================
DESIGN DECISIONS (document in thesis):

1. EVALUATION: Nested LOPO CV — outer loop leaves 1 patient out for test,
   inner loop holds out 1 of the remaining 7 as a validation patient.
   Early stopping is driven by the INNER val patient — the outer test
   patient's data is NEVER seen during training or model selection.

   INNER VAL ROTATION:
   The inner validation patient rotates with each outer fold so that no
   single patient is always used as validator. Specifically, for outer fold
   i (0-indexed), the inner val patient is:
       sorted_patients[(i + 1) % len(sorted_patients)]
   skipping the current test patient if they coincide.
   With 8 patients this means each patient serves as inner validator in
   approximately one fold, avoiding the bias of a fixed inner val patient.

2. TWO VARIANTS:
   (a) GCN with node features (in_dim=16): spectral + Hjorth + time-domain
       + connectivity degrees — full hand-crafted representation.
   (b) Adjacency-only GCN (in_dim=19): node features = the DTF adjacency
       row (each node's connectivity profile). Tests whether connectivity
       topology alone suffices, without any spectral or Hjorth features.

3. DTF THRESHOLD — DATA-DRIVEN, PER FOLD:
   The edge threshold is the p-th percentile of off-diagonal DTF values
   computed from the INNER TRAINING adjacency matrices only.
   Default: p=70 (top 30% of edges are kept).
   This replaces a fixed threshold=0.15, which is arbitrary and does not
   adapt to the actual connectivity distribution of each fold's training data.
   The percentile is computed before building graphs and is stored per fold.

4. METRICS: AUC, F1, Sensitivity, Specificity, Accuracy, MCC.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERFITTING PREVENTION — METHODS (document each in thesis):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A) Small model capacity:
   - 2-layer GCN with hidden=32, ~4k total parameters.
   - With ~few hundred training graphs, a deeper or wider model would
     memorise individual patient graphs. 4k parameters is deliberately
     small relative to dataset size.

B) Dropout (p=0.4):
   - Applied after both GCN layers and inside the MLP head.
   - At p=0.4, 40% of units are zeroed each forward pass during training,
     forcing the network to learn redundant representations rather than
     relying on specific neurons.

C) L2 weight decay (1e-4):
   - Added to the Adam optimiser.
   - Penalises large weights, preventing the model from assigning
     extreme importance to any single feature or edge.

D) Early stopping on inner-val AUC (patience=20):
   - Training halts when inner-val loss does not improve for 20 epochs.
   - Best model state (not final state) is restored.
   - Monitored on INNER val patient — never on the test patient.
   - Without early stopping, the model would continue fitting noise in
     training graphs beyond the point of generalisation.

E) Learning rate scheduling (ReduceLROnPlateau):
   - LR reduced by 0.5x when inner-val loss plateaus for 10 epochs.
   - Prevents overshooting minima and improves convergence stability.

F) Gradient clipping (max_norm=1.0):
   - Prevents exploding gradients, especially important for small batches
     (each graph is a single sample — the most extreme batch size).

G) Node feature scaling:
   - StandardScaler fit on INNER TRAIN graphs only.
   - Applied to inner-val and outer-test graphs using the train statistics.
   - Prevents test-data statistics from influencing normalisation.

H) Class imbalance handling:
   - BCEWithLogitsLoss with pos_weight = n_neg/n_pos.
   - Computed from the inner training fold only.

I) Data-driven threshold (per fold):
   - DTF edge threshold = p70 percentile of inner-train adjacency values.
   - Adapts to the actual connectivity distribution of each fold.
   - Computed from inner-train only — no leakage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERFITTING DETECTION — PLOTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  training_curves_{tag}_{patient}.png
      Two-panel: train vs inner-val LOSS (left) and AUC (right).
      Diverging curves (val >> train) = overfitting to training graphs.

  train_vs_test_gap_{tag}.png
      Bar chart: train AUC, inner-val AUC, outer-test AUC per fold.
      Same interpretation: gap near 0 = no overfitting.

  overfitting_summary_{tag}.png
      (train AUC − test AUC) per fold — red if > 0.10.

Outputs:
  results_gcn_with_features.json
  results_gcn_adj_only.json
  results_gcn.json              (alias for step6 compatibility)
  comparison_gcn_variants.png
  comparison_all_models.png     (if --baseline_json provided)
  + per-fold loss/AUC/CM/ROC plots for both variants

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
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Threshold utility
# ─────────────────────────────────────────────────────────────

def compute_threshold(adj_dtf_train: np.ndarray, percentile: float = 70.0) -> float:
    """
    Data-driven DTF edge threshold.

    Computes the p-th percentile of ALL off-diagonal DTF values across
    the inner training fold adjacency matrices. Edges below this value
    are zeroed (sparsification). This keeps the top (100-p)% of edges.

    Called INSIDE the LOPO loop using INNER TRAIN adjacency only.
    The inner-val and test patients are never touched.

    Parameters
    ----------
    adj_dtf_train : (N_train, 19, 19)  inner-train adjacency matrices
    percentile    : float, default 70  → keeps top 30% of edges

    Returns
    -------
    threshold : float
    """
    n    = adj_dtf_train.shape[1]
    mask = ~np.eye(n, dtype=bool)          # off-diagonal only
    vals = np.concatenate([adj_dtf_train[i][mask] for i in range(len(adj_dtf_train))])
    return float(np.percentile(vals, percentile))


# ─────────────────────────────────────────────────────────────
# Adjacency normalisation
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """A_hat = D^{-1/2} (A + I) D^{-1/2}  (Kipf & Welling 2017)."""
    A          = adj + np.eye(adj.shape[0], dtype=np.float32)
    d          = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D          = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, a_hat):
        return F.relu(self.W(a_hat @ x))


class SmallGCN(nn.Module):
    """
    Deliberately small: 2-layer GCN → GlobalMeanPool → 2-layer MLP.
    in_dim=16 for feature-based GCN, in_dim=19 for adjacency-only GCN.
    ~4k total parameters — chosen to be small relative to dataset size.
    Dropout(0.4) after both GCN layers and inside the MLP head.
    """
    def __init__(self, in_dim: int = 16, hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x, a_hat):
        h = self.gcn1(x, a_hat)
        h = self.drop(h)
        h = self.gcn2(h, a_hat)
        h = self.drop(h)
        h = h.mean(dim=0, keepdim=True)    # global mean pool
        return self.head(h).squeeze()


# ─────────────────────────────────────────────────────────────
# Graph building — threshold applied here
# ─────────────────────────────────────────────────────────────

def build_graphs_with_features(node_feats: np.ndarray,
                                adj_dtf: np.ndarray,
                                threshold: float) -> list:
    """
    Build graph list for SmallGCN (hand-crafted node features).
    threshold is computed from inner-train adjacency before this call.
    """
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        graphs.append((
            torch.tensor(node_feats[i], dtype=torch.float32),
            normalize_adjacency(adj),
        ))
    return graphs


def build_graphs_adj_only(adj_dtf: np.ndarray, threshold: float) -> list:
    """
    Build graph list for adjacency-only GCN.
    Node feature = the raw DTF adjacency row (connectivity profile, 19 values).
    Graph structure = same thresholded normalised adjacency.
    This tests whether connectivity topology alone suffices for classification.
    """
    graphs = []
    for i in range(len(adj_dtf)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        # Node feature = unthresholded adjacency row (full connectivity profile)
        x_raw = adj_dtf[i].copy()
        np.fill_diagonal(x_raw, 0.0)
        graphs.append((torch.tensor(x_raw.astype(np.float32)), a_hat))
    return graphs


def _apply_scaler(graphs: list, scaler) -> list:
    return [
        (torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32), a)
        for x, a in graphs
    ]


def scale_three_splits(g_tr: list, g_val: list, g_te: list):
    """
    Fit StandardScaler on inner-train node features only.
    Apply to inner-val and outer-test. No leakage.
    """
    sc = StandardScaler()
    sc.fit(np.concatenate([g[0].numpy() for g in g_tr], axis=0))
    return _apply_scaler(g_tr, sc), _apply_scaler(g_val, sc), _apply_scaler(g_te, sc)


# ─────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, optimiser, criterion, graphs, labels, device):
    model.train()
    total = 0.0
    for i in np.random.permutation(len(graphs)):
        x, a = graphs[i]
        optimiser.zero_grad()
        logit = model(x.to(device), a.to(device))
        loss  = criterion(logit.unsqueeze(0),
                          torch.tensor([float(labels[i])], device=device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimiser.step()
        total += loss.item()
    return total / len(graphs)


@torch.no_grad()
def evaluate_graphs(model, graphs, labels, device):
    model.eval()
    logits, targets = [], []
    for i in range(len(graphs)):
        x, a = graphs[i]
        logits.append(model(x.to(device), a.to(device)).cpu().item())
        targets.append(int(labels[i]))
    logits  = np.array(logits,  dtype=np.float32)
    targets = np.array(targets, dtype=np.int64)
    probs   = 1.0 / (1.0 + np.exp(-logits))
    preds   = (probs >= 0.5).astype(np.int64)
    eps     = 1e-7
    bce     = -np.mean(targets * np.log(probs + eps)
                       + (1 - targets) * np.log(1 - probs + eps))
    auc     = roc_auc_score(targets, probs) \
              if len(np.unique(targets)) == 2 else float('nan')
    return probs, preds, targets, float(bce), float(auc)


def compute_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'auc':         float(roc_auc_score(y_true, y_prob)),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(tp / (tp + fn + 1e-12)),
        'specificity': float(tn / (tn + fp + 1e-12)),
        'precision':   float(precision_score(y_true, y_pred, zero_division=0)),
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'mcc':         float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ─────────────────────────────────────────────────────────────
# Inner val patient rotation
# ─────────────────────────────────────────────────────────────

def pick_inner_val_patient(sorted_all_patients: list,
                           outer_fold_idx: int,
                           outer_test_pat: str) -> str:
    """
    Rotate the inner validation patient across outer folds so that no
    single patient is always the validator.

    Strategy: cycle through sorted_all_patients starting at index
    (outer_fold_idx + 1) % n, skipping the outer test patient.

    With 8 patients and 8 outer folds, each patient will serve as inner
    validator in exactly one fold (since the test patient is already
    excluded from the inner set, the rotation naturally avoids it).

    Parameters
    ----------
    sorted_all_patients : list of all patient IDs, sorted alphabetically
    outer_fold_idx      : 0-based index of the current outer fold
    outer_test_pat      : the current outer test patient (must be skipped)

    Returns
    -------
    inner_val_pat : str
    """
    n = len(sorted_all_patients)
    for offset in range(1, n + 1):
        candidate = sorted_all_patients[(outer_fold_idx + offset) % n]
        if candidate != outer_test_pat:
            return candidate
    # Fallback (should never happen with >= 3 patients)
    raise RuntimeError("Could not find a valid inner val patient.")


# ─────────────────────────────────────────────────────────────
# Overfitting plots
# ─────────────────────────────────────────────────────────────

def plot_loss_and_auc_curves(train_losses, val_losses, train_aucs, val_aucs,
                              patient_id, model_tag, output_dir):
    """
    Two-panel per fold: loss (left) and AUC (right).
    val here = INNER val patient (not the test patient).
    """
    best_ep = int(np.argmax(val_aucs)) if val_aucs else 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(train_losses, color='steelblue',  lw=1.5, label='Train loss')
    ax.plot(val_losses,   color='tomato',     lw=1.5, linestyle='--', label='Inner-val loss')
    ax.axvline(best_ep, color='green', lw=1, linestyle=':', label=f'Best epoch ({best_ep})')
    gap = abs(train_losses[-1] - val_losses[-1]) if train_losses else 0
    ax.text(0.97, 0.97, f'Final gap: {gap:.4f}', transform=ax.transAxes,
            fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('BCE Loss', fontsize=11)
    ax.set_title(f'Loss curves | Test: {patient_id}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(train_aucs, color='steelblue', lw=1.5, label='Train AUC')
    ax.plot(val_aucs,   color='tomato',    lw=1.5, linestyle='--', label='Inner-val AUC')
    ax.axvline(best_ep, color='green', lw=1, linestyle=':', label=f'Best epoch ({best_ep})')
    if train_aucs and val_aucs:
        auc_gap = train_aucs[-1] - val_aucs[-1]
        ax.text(0.97, 0.03, f'Final AUC gap: {auc_gap:+.3f}',
                transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f'AUC curves | Test: {patient_id}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{model_tag} — Overfitting Diagnostic | Test: {patient_id}',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_{model_tag}_{patient_id}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_train_vs_test_gap(fold_metrics, model_tag, output_dir):
    patients  = [m['patient'] for m in fold_metrics]
    tr_aucs   = [m.get('train_auc',     float('nan')) for m in fold_metrics]
    val_aucs  = [m.get('inner_val_auc', float('nan')) for m in fold_metrics]
    test_aucs = [m['auc'] for m in fold_metrics]

    x, w = np.arange(len(patients)), 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, tr_aucs,   w, label='Train AUC',     color='steelblue',  alpha=0.85, edgecolor='black')
    ax.bar(x,     val_aucs,  w, label='Inner val AUC',  color='darkorange', alpha=0.85, edgecolor='black')
    ax.bar(x + w, test_aucs, w, label='Outer test AUC', color='tomato',     alpha=0.85, edgecolor='black')
    for i, (tr, te) in enumerate(zip(tr_aucs, test_aucs)):
        if not (np.isnan(tr) or np.isnan(te)):
            ax.text(x[i], max(tr, te, 0) + 0.02, f'Δ{tr - te:+.2f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.22)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title(f'{model_tag} — Train / Inner-val / Test AUC per fold\n'
                 f'(Δ = train−test; near zero = no overfitting)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'train_vs_test_gap_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting_summary(fold_metrics, model_tag, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    gaps     = [m.get('train_auc', float('nan')) - m['auc'] for m in fold_metrics]
    colors   = ['tomato' if g > 0.1 else 'steelblue' for g in gaps]
    fig, ax  = plt.subplots(figsize=(10, 4))
    bars     = ax.bar(patients, gaps, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0,   color='black', lw=1)
    ax.axhline(0.1, color='red',   lw=1, linestyle='--', alpha=0.5, label='0.10 threshold')
    ax.set_ylabel('Train AUC − Test AUC', fontsize=12)
    ax.set_title(f'{model_tag} — Overfitting Gap per Patient Fold\n'
                 f'(Red bars > 0.10 indicate potential memorisation)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, g in zip(bars, gaps):
        if not np.isnan(g):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f'{g:.2f}',
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f'overfitting_summary_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────
# Standard plots
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, patient_id, model_tag, output_dir):
    cmap = 'Blues' if 'feat' in model_tag else 'Purples'
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'{model_tag} CM | Test: {patient_id}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_{model_tag}_{patient_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_all_folds(fold_roc_data, model_tag, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for fpr, tpr, auc, pat in fold_roc_data:
        ax.plot(fpr, tpr, alpha=0.55, lw=1.5, label=f'{pat} (AUC={auc:.2f})')
        aucs.append(auc)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_tag} — LOPO ROC\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, model_tag, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    x, w     = np.arange(len(patients)), 0.16
    fig, ax  = plt.subplots(figsize=(14, 5))
    for i, (met, col) in enumerate(zip(met_keys, colors)):
        ax.bar(x + i * w, [m[met] for m in fold_metrics], w,
               label=met.upper(), color=col, alpha=0.8, edgecolor='black')
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title(f'{model_tag} — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'per_fold_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion(all_y_true, all_y_pred, model_tag, output_dir):
    cmap    = 'Blues' if 'feat' in model_tag else 'Purples'
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
        ax.set_title(f'{model_tag} — Aggregate CM ({title})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_aggregate_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_gcn_variants_comparison(feat_stats, adj_stats, output_dir):
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy', 'mcc']
    x, w     = np.arange(len(met_keys)), 0.35
    fig, ax  = plt.subplots(figsize=(10, 5))
    for i, (label, stats, col) in enumerate([
        ('GCN (w/ features)', feat_stats, 'steelblue'),
        ('GCN (adj-only)',    adj_stats,  'mediumpurple'),
    ]):
        means = [stats.get(k, {}).get('mean', 0) for k in met_keys]
        stds  = [stats.get(k, {}).get('std',  0) for k in met_keys]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds,
               label=label, color=col, capsize=4, edgecolor='black', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('GCN with Features vs Adjacency-Only GCN (Ablation)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_gcn_variants.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_vs_baseline(feat_stats, adj_stats, baseline_json_path, output_dir):
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    models   = {}
    if baseline_json_path and Path(baseline_json_path).exists():
        with open(baseline_json_path) as f:
            bl = json.load(f)
        for name, res in bl.items():
            if res.get('summary_stats'):
                models[name] = {k: res['summary_stats'].get(k, {}).get('mean', 0)
                                for k in met_keys}
    models['GCN (w/ features)'] = {k: feat_stats.get(k, {}).get('mean', 0) for k in met_keys}
    models['GCN (adj-only)']    = {k: adj_stats.get(k,  {}).get('mean', 0) for k in met_keys}

    colors = ['steelblue', 'tomato', 'darkorange', 'seagreen', 'mediumpurple']
    x      = np.arange(len(met_keys))
    n, w   = len(models), 0.15
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (name, vals) in enumerate(models.items()):
        offset = (i - n / 2 + 0.5) * w
        ax.bar(x + offset, [vals.get(k, 0) for k in met_keys], w,
               label=name, color=colors[i % len(colors)], alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Mean Score (LOPO)', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('RF vs SVM vs SVM-RFE vs GCN variants — LOPO CV',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────
# Core LOPO loop
# ─────────────────────────────────────────────────────────────

def run_gcn_lopo(model_tag, in_dim, node_feats_or_none, adj_dtf_all,
                 y, patient_ids, args, device, output_dir):
    """
    Full nested LOPO loop with:
      - Data-driven per-fold DTF threshold (p70 of inner-train adjacency)
      - Rotating inner val patient
      - Early stopping on inner-val loss (test patient never seen)
      - StandardScaler fit on inner-train only
    """
    sorted_all_pats = sorted(np.unique(patient_ids).tolist())
    fold_metrics    = []
    fold_roc_data   = []
    all_y_true      = []
    all_y_pred      = []

    is_feat = (node_feats_or_none is not None)

    print(f'\n{"=" * 65}')
    print(f'  {model_tag} — Nested LOPO  (node feat dim: {in_dim})')
    print(f'  Threshold: p{args.threshold_pct:.0f} percentile of inner-train DTF (data-driven)')
    print(f'  Inner val: rotates across folds (no fixed validator)')
    print(f'  Overfitting controls: dropout={args.dropout}, weight_decay=1e-4,')
    print(f'  gradient_clipping=1.0, early_stopping(patience={args.patience}),')
    print(f'  ReduceLROnPlateau, hidden={args.hidden} (~4k params)')
    print(f'{"=" * 65}')

    for fold_idx, pat in enumerate(sorted_all_pats):
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask
        train_idx  = np.where(train_mask)[0]
        test_idx   = np.where(test_mask)[0]

        y_train    = y[train_idx]
        y_test     = y[test_idx]
        train_pats = patient_ids[train_idx]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: only one class in test')
            continue

        # ── Rotating inner val patient ──────────────────────────
        inner_val_pat = pick_inner_val_patient(sorted_all_pats, fold_idx, pat)
        iv_mask = (train_pats == inner_val_pat)
        it_mask = ~iv_mask
        it_idx  = train_idx[it_mask]
        iv_idx  = train_idx[iv_mask]

        y_it = y[it_idx]
        y_iv = y[iv_idx]

        # ── Data-driven threshold from inner-train adjacency ────
        threshold = compute_threshold(adj_dtf_all[it_idx], args.threshold_pct)

        print(f'\n  Outer fold {fold_idx + 1}/{len(sorted_all_pats)} '
              f'— Test: {pat}  |  Inner val: {inner_val_pat}  '
              f'|  Threshold: {threshold:.4f} (p{args.threshold_pct:.0f})')

        # ── Build graphs with this fold's threshold ─────────────
        if is_feat:
            g_all_it = build_graphs_with_features(node_feats_or_none[it_idx],
                                                   adj_dtf_all[it_idx], threshold)
            g_all_iv = build_graphs_with_features(node_feats_or_none[iv_idx],
                                                   adj_dtf_all[iv_idx], threshold)
            g_all_te = build_graphs_with_features(node_feats_or_none[test_idx],
                                                   adj_dtf_all[test_idx], threshold)
        else:
            g_all_it = build_graphs_adj_only(adj_dtf_all[it_idx], threshold)
            g_all_iv = build_graphs_adj_only(adj_dtf_all[iv_idx], threshold)
            g_all_te = build_graphs_adj_only(adj_dtf_all[test_idx], threshold)

        # ── Scale: fit on inner-train only ──────────────────────
        g_it, g_iv, g_te = scale_three_splits(g_all_it, g_all_iv, g_all_te)

        n_neg      = int((y_it == 0).sum())
        n_pos      = int((y_it == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        model     = SmallGCN(in_dim=in_dim, hidden=args.hidden,
                             dropout=args.dropout).to(device)
        optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, verbose=False)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        train_losses, val_losses   = [], []
        train_aucs,   val_aucs_ep  = [], []
        best_val_loss  = np.inf
        best_val_auc   = 0.0
        best_state     = None
        patience_cnt   = 0

        for ep in range(args.epochs):
            tr_loss = train_one_epoch(model, optimiser, criterion, g_it, y_it, device)

            # Inner-val evaluation (never the test patient)
            _, _, _, val_loss, iv_auc = evaluate_graphs(model, g_iv, y_iv, device)
            _, _, _, _, tr_auc_ep    = evaluate_graphs(model, g_it, y_it, device)

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            train_aucs.append(tr_auc_ep)
            val_aucs_ep.append(iv_auc if not np.isnan(iv_auc) else 0.0)

            scheduler.step(val_loss)

            # Early stopping on inner-val loss (not test AUC)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc  = iv_auc if not np.isnan(iv_auc) else 0.0
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  {pat}: early stop @ epoch {ep + 1}  '
                          f'inner val loss={best_val_loss:.4f}  AUC={best_val_auc:.3f}')
                    break

        if best_state:
            model.load_state_dict(best_state)

        # ── Final evaluation on outer TEST patient ───────────────
        probs, preds, targets, _, _ = evaluate_graphs(model, g_te, y_test, device)
        metrics = compute_metrics(targets, preds, probs)
        if metrics is None:
            continue

        _, _, _, _, final_tr_auc = evaluate_graphs(model, g_it, y_it, device)
        overfit_gap = final_tr_auc - metrics['auc']

        metrics['train_auc']      = float(final_tr_auc)
        metrics['inner_val_auc']  = float(best_val_auc)
        metrics['inner_val_pat']  = inner_val_pat
        metrics['patient']        = pat
        metrics['n_train']        = int(train_mask.sum())
        metrics['n_test']         = int(test_mask.sum())
        metrics['threshold']      = float(threshold)
        fold_metrics.append(metrics)

        all_y_true.extend(targets.tolist())
        all_y_pred.extend(preds.tolist())
        fpr, tpr, _ = roc_curve(targets, probs)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        print(f'  {pat:8s} | Test AUC={metrics["auc"]:.3f}  Acc={metrics["accuracy"]:.3f}'
              f'  F1={metrics["f1"]:.3f}  Sens={metrics["sensitivity"]:.3f}'
              f'  Spec={metrics["specificity"]:.3f}  MCC={metrics["mcc"]:.3f}')
        print(f'           | Train AUC={final_tr_auc:.3f}  '
              f'Overfit gap={overfit_gap:+.3f}  '
              f'Inner val AUC={best_val_auc:.3f}  Threshold={threshold:.4f}')

        plot_loss_and_auc_curves(train_losses, val_losses, train_aucs, val_aucs_ep,
                                 pat, model_tag, output_dir)
        plot_confusion_matrix(confusion_matrix(targets, preds), pat, model_tag, output_dir)

    if not fold_metrics:
        print(f'  [ERROR] No valid folds completed.')
        return [], {}

    if fold_roc_data:
        plot_roc_all_folds(fold_roc_data, model_tag, output_dir)
    if fold_metrics:
        plot_per_fold_metrics(fold_metrics, model_tag, output_dir)
        plot_train_vs_test_gap(fold_metrics, model_tag, output_dir)
        plot_overfitting_summary(fold_metrics, model_tag, output_dir)
    if all_y_true:
        plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred),
                                 model_tag, output_dir)

    met_keys      = ['auc', 'f1', 'sensitivity', 'specificity', 'precision', 'accuracy', 'mcc']
    summary_stats = {}
    print(f'\n  {"─" * 55}')
    print(f'  {model_tag} — Mean ± Std across {len(fold_metrics)} folds')
    print(f'  {"─" * 55}')
    for k in met_keys:
        vals        = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    tr_aucs = [m['train_auc'] for m in fold_metrics]
    mean_gap = float(np.mean(tr_aucs)) - summary_stats['auc']['mean']
    print(f'\n  Mean train AUC   : {np.mean(tr_aucs):.3f}')
    print(f'  Mean test  AUC   : {summary_stats["auc"]["mean"]:.3f}')
    print(f'  Mean overfit gap : {mean_gap:+.3f}')
    summary_stats['train_auc']   = {'mean': float(np.mean(tr_aucs)), 'std': float(np.std(tr_aucs))}
    summary_stats['overfit_gap'] = {'mean': mean_gap}

    iv_aucs = [m.get('inner_val_auc', float('nan')) for m in fold_metrics]
    iv_aucs = [v for v in iv_aucs if not np.isnan(v)]
    if iv_aucs:
        print(f'  Mean inner val AUC: {np.mean(iv_aucs):.3f} ± {np.std(iv_aucs):.3f}')
        summary_stats['inner_val_auc'] = {'mean': float(np.mean(iv_aucs)),
                                          'std':  float(np.std(iv_aucs))}

    thresholds = [m.get('threshold', float('nan')) for m in fold_metrics]
    thresholds = [v for v in thresholds if not np.isnan(v)]
    if thresholds:
        print(f'  DTF threshold (mean): {np.mean(thresholds):.4f} ± {np.std(thresholds):.4f}')

    return fold_metrics, summary_stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Step 5 — Supervised GCN (Nested LOPO, data-driven threshold, rotating inner val)'
    )
    parser.add_argument('--featfile',       required=True)
    parser.add_argument('--outputdir',      default='results/gnn_supervised')
    parser.add_argument('--epochs',         type=int,   default=150)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--hidden',         type=int,   default=32)
    parser.add_argument('--dropout',        type=float, default=0.4)
    parser.add_argument('--threshold_pct',  type=float, default=70.0,
                        help='DTF percentile threshold per fold (default 70 → keeps top 30%% of edges)')
    parser.add_argument('--patience',       type=int,   default=20)
    parser.add_argument('--baseline_json',  default=None)
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 5 — SUPERVISED GCN (Nested LOPO + Overfitting Analysis)')
    print('=' * 65)
    print(f'Device        : {device}')
    print(f'Epochs        : {args.epochs}   LR: {args.lr}')
    print(f'Hidden        : {args.hidden}   Dropout: {args.dropout}  (~4k params)')
    print(f'Threshold pct : {args.threshold_pct}  (data-driven, computed per fold from inner-train)')
    print(f'Patience      : {args.patience}')
    print()
    print('Overfitting prevention:')
    print('  Dropout(0.4), L2 weight decay(1e-4), gradient clipping(1.0)')
    print('  Early stopping on inner-val loss (patience=20)')
    print('  ReduceLROnPlateau (patience=10, factor=0.5)')
    print('  StandardScaler fit on inner-train only')
    print('  Small model: 2-layer GCN, hidden=32 (~4k params)')
    print('  Data-driven threshold per fold (no fixed arbitrary value)')
    print('  Rotating inner val patient (no fixed validator)')

    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)  # (N, 19, 16)
    #adj_dtf     = data['adj_dtf'].astype(np.float32)        # (N, 19, 19)
    adj_dtf = data['adj_pdc'].astype(np.float32)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']

    print(f'\nLoaded: {len(y)} epochs | Ictal: {(y == 1).sum()} | Pre-ictal: {(y == 0).sum()}')
    print(f'Patients: {sorted(np.unique(patient_ids).tolist())}')
    majority_b = max((y == 0).sum(), (y == 1).sum()) / len(y)
    print(f'Majority-class accuracy baseline: {majority_b * 100:.1f}%')

    # ── Model A: GCN with hand-crafted node features ────────────
    feat_metrics, feat_stats = run_gcn_lopo(
        model_tag        = 'gcn_feat',
        in_dim           = 16,
        node_feats_or_none = node_feats,
        adj_dtf_all      = adj_dtf,
        y                = y,
        patient_ids      = patient_ids,
        args             = args,
        device           = device,
        output_dir       = output_dir,
    )

    # ── Model B: Adjacency-only GCN (ablation) ──────────────────
    adj_metrics, adj_stats = run_gcn_lopo(
        model_tag        = 'gcn_adj_only',
        in_dim           = 19,
        node_feats_or_none = None,     # node features = DTF row, built inside loop
        adj_dtf_all      = adj_dtf,
        y                = y,
        patient_ids      = patient_ids,
        args             = args,
        device           = device,
        output_dir       = output_dir,
    )

    # ── Save results ─────────────────────────────────────────────
    for tag, metrics, stats, fname in [
        ('GCN_Supervised_WithFeatures', feat_metrics, feat_stats, 'results_gcn_with_features.json'),
        ('GCN_Supervised_AdjOnly',      adj_metrics,  adj_stats,  'results_gcn_adj_only.json'),
    ]:
        res = {
            'model':          tag,
            'hyperparameters': {k: v for k, v in vars(args).items()},
            'threshold_note': f'Data-driven p{args.threshold_pct:.0f} percentile per fold from inner-train adjacency',
            'inner_val_note': 'Rotating inner val patient — cycles with fold index to avoid fixed validator bias',
            'fold_metrics':   metrics,
            'summary_stats':  stats,
        }
        with open(output_dir / fname, 'w') as f:
            json.dump(res, f, indent=2, default=str)
        print(f'  Saved: {fname}')

    # Alias for step6 compatibility
    with open(output_dir / 'results_gcn.json', 'w') as f:
        json.dump({
            'model':          'GCN_Supervised_WithFeatures',
            'hyperparameters': {k: v for k, v in vars(args).items()},
            'fold_metrics':   feat_metrics,
            'summary_stats':  feat_stats,
        }, f, indent=2, default=str)

    # ── Comparison plots ─────────────────────────────────────────
    if feat_stats and adj_stats:
        plot_gcn_variants_comparison(feat_stats, adj_stats, output_dir)
    if args.baseline_json:
        plot_comparison_vs_baseline(feat_stats, adj_stats, args.baseline_json, output_dir)

    # ── Final summary table ──────────────────────────────────────
    print('\n' + '=' * 65)
    print('STEP 5 — FINAL SUMMARY')
    print('=' * 65)
    print(f'{"Model":25s} | {"AUC":>12} | {"F1":>12} | {"Sens":>12} | {"Acc":>12}')
    print('-' * 65)
    for name, stats in [('GCN (with features)', feat_stats), ('GCN (adj-only)', adj_stats)]:
        if not stats:
            continue
        def fmt(k):
            return f'{stats[k]["mean"]:.3f}±{stats[k]["std"]:.3f}'
        print(f'{name:25s} | {fmt("auc"):>12} | {fmt("f1"):>12} | '
              f'{fmt("sensitivity"):>12} | {fmt("accuracy"):>12}')
    print(f'\nNOTE: Majority-class accuracy baseline ≈ {majority_b * 100:.1f}%')
    print('\n' + '=' * 65)
    print('STEP 5 COMPLETE')
    print('=' * 65)
    print('\nNext: python step6_ssl_gnn.py --featfile features/features_all.npz'
          ' --sup_json results/gnn_supervised/results_gcn.json')


if __name__ == '__main__':
    main()
