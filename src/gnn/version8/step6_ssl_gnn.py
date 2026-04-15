"""
Step 6 — Self-Supervised GCN (GraphCL, Leakage-Free, Nested LOPO)
==================================================================
DATA LEAKAGE FIX:
  SSL pre-training runs on INNER TRAIN graphs only inside each fold.
  The test patient's graphs are NEVER seen before evaluation.

  Full nested protocol per fold:
    1. Outer test patient removed.
    2. Inner val patient selected by rotation (see note below).
    3. DTF threshold computed from inner-train adjacency only (data-driven).
    4. StandardScaler fit on inner-train only; applied to inner-val and test.
    5. SSL pre-trains on inner-train graphs (no labels).
    6. Linear probe on inner-val (measures SSL quality).
    7. Phase A: linear probe fine-tune (encoder frozen, 30 epochs).
    8. Phase B: full fine-tune, early stopping on inner-val AUC.
    9. Final evaluation on outer test patient.

INNER VAL ROTATION:
  The inner validation patient rotates with each outer fold using:
      sorted_all_patients[(outer_fold_idx + 1) % n]
  skipping the test patient if they coincide.
  This prevents any single patient from always being the validator,
  which would give that patient disproportionate influence on model
  selection across all folds.

DTF THRESHOLD — DATA-DRIVEN, PER FOLD:
  Edge threshold = p-th percentile of off-diagonal DTF values from
  inner-train adjacency matrices. Default p=70 (top 30% edges kept).
  Replaces fixed threshold=0.15, which is arbitrary and does not adapt
  to the actual connectivity distribution of each fold's training data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERFITTING PREVENTION — METHODS (document each in thesis):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SSL pre-training phase:
  A) Contrastive learning as implicit regulariser:
     NT-Xent loss forces encoder to learn augmentation-invariant graph
     representations rather than patient-specific idiosyncrasies.

  B) Three EEG-appropriate augmentations:
     - Edge dropout (p=0.20): zeros edges, simulating missing connectivity.
     - Node feature noise (sigma=0.10): Gaussian noise on node features,
       simulating electrode noise.
     - Band feature mask: zeros one random frequency band for all nodes,
       preventing reliance on a single EEG band.

  C) CosineAnnealingLR during SSL: smoothly decays LR to lr*0.1.

  D) Gradient clipping (max_norm=1.0): applied during SSL pre-training.

Fine-tuning phase:
  E) Two-phase fine-tuning (linear probe then full fine-tune):
     Phase A (frozen encoder, 30 epochs): warms up classifier head before
     allowing the encoder to be updated.
     Phase B (full fine-tune, lr_ft=0.0005): small LR preserves SSL
     representations.

  F) Dropout (p=0.4) and L2 weight decay (1e-4).

  G) Early stopping on INNER val AUC (patience=20):
     Fine-tuning halts when inner-val AUC stops improving.
     The test patient is never consulted.

  H) BCEWithLogitsLoss with pos_weight = n_neg/n_pos (inner-train only).

  I) StandardScaler fit on inner-train graphs only.

  J) Small model (~4k params), identical to step 5.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERFITTING DETECTION — PLOTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ssl_overview_{patient}.png       — NT-Xent loss + linear probe AUC
  ft_training_curves_{patient}.png — FT loss + AUC (train vs inner-val)
  train_vs_test_gap_ssl.png        — train/val/test AUC bar chart
  overfitting_summary_ssl.png      — gap per fold

Outputs:
  results_ssl.json
  ssl_overview_{patient}.png
  ft_training_curves_{patient}.png
  cm_ssl_{patient}.png
  roc_ssl_gcn.png
  per_fold_ssl_gcn.png
  cm_aggregate_ssl_gcn.png
  train_vs_test_gap_ssl.png
  overfitting_summary_ssl.png
  comparison_all_models.png

Usage:
  python step6_ssl_gnn.py \\
      --featfile      features/features_all.npz \\
      --outputdir     results/ssl_gnn \\
      --ssl_epochs    200 \\
      --ft_epochs     100 \\
      --lr_ssl        0.001 \\
      --lr_ft         0.0005 \\
      --hidden        32 \\
      --threshold_pct 70 \\
      --temperature   0.5 \\
      --batch_size    32 \\
      --patience      20 \\
      --baseline_json results/baseline_ml/results_all.json \\
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
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N_BAND_FEATS = 6


# ─────────────────────────────────────────────────────────────
# Threshold utility  (identical to step 5)
# ─────────────────────────────────────────────────────────────

def compute_threshold(adj_dtf_train: np.ndarray, percentile: float = 70.0) -> float:
    """
    Data-driven DTF edge threshold.
    p-th percentile of off-diagonal values from inner-train adjacency only.
    Called INSIDE the LOPO loop — test patient never touched.
    """
    n    = adj_dtf_train.shape[1]
    mask = ~np.eye(n, dtype=bool)
    vals = np.concatenate([adj_dtf_train[i][mask] for i in range(len(adj_dtf_train))])
    return float(np.percentile(vals, percentile))


# ─────────────────────────────────────────────────────────────
# Inner val patient rotation  (identical to step 5)
# ─────────────────────────────────────────────────────────────

def pick_inner_val_patient(sorted_all_patients: list,
                           outer_fold_idx: int,
                           outer_test_pat: str) -> str:
    """
    Rotate the inner validation patient across outer folds.
    For outer fold i, candidate = sorted_all_patients[(i+1) % n],
    skipping the outer test patient if they coincide.
    """
    n = len(sorted_all_patients)
    for offset in range(1, n + 1):
        candidate = sorted_all_patients[(outer_fold_idx + offset) % n]
        if candidate != outer_test_pat:
            return candidate
    raise RuntimeError("Could not find a valid inner val patient.")


# ─────────────────────────────────────────────────────────────
# Adjacency normalisation
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(adj):
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    D = np.diag(np.where(d > 0, 1.0 / np.sqrt(d), 0.0))
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def build_graphs(node_feats: np.ndarray, adj_dtf: np.ndarray,
                 threshold: float) -> list:
    """Build graph list. Threshold computed from inner-train before this call."""
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        graphs.append((torch.tensor(node_feats[i], dtype=torch.float32),
                        normalize_adjacency(adj)))
    return graphs


def _apply_scaler(graphs, scaler):
    return [(torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32), a)
            for x, a in graphs]


def scale_three_splits(g_tr, g_val, g_te):
    """Fit on inner-train; apply to inner-val and test. No leakage."""
    sc = StandardScaler()
    sc.fit(np.concatenate([g[0].numpy() for g in g_tr], axis=0))
    return _apply_scaler(g_tr, sc), _apply_scaler(g_val, sc), _apply_scaler(g_te, sc)


# ─────────────────────────────────────────────────────────────
# EEG augmentations
# ─────────────────────────────────────────────────────────────

def augment_edge_dropout(x, a, p=0.20):
    mask = (torch.rand_like(a) > p).float()
    idx  = torch.arange(a.shape[0])
    mask[idx, idx] = 1.0
    a_aug   = a * mask
    row_sum = a_aug.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return x, a_aug / row_sum


def augment_node_noise(x, a, sigma=0.10):
    return x + torch.randn_like(x) * sigma, a


def augment_band_mask(x, a):
    b     = int(torch.randint(0, N_BAND_FEATS, (1,)).item())
    x_aug = x.clone()
    x_aug[:, b] = 0.0
    return x_aug, a


ALL_AUGMENTATIONS = [augment_edge_dropout, augment_node_noise, augment_band_mask]


def random_augment(x, a):
    chosen = np.random.choice(len(ALL_AUGMENTATIONS), size=2, replace=False)
    x1, a1 = ALL_AUGMENTATIONS[chosen[0]](x.clone(), a.clone())
    x2, a2 = ALL_AUGMENTATIONS[chosen[1]](x.clone(), a.clone())
    return (x1, a1), (x2, a2)


# ─────────────────────────────────────────────────────────────
# NT-Xent loss
# ─────────────────────────────────────────────────────────────

def nt_xent_loss(z1, z2, temperature=0.5):
    N  = z1.shape[0]
    z  = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = torch.mm(z, z.T) / temperature
    sim.masked_fill_(torch.eye(2 * N, dtype=torch.bool, device=z.device), -9e15)
    labels = torch.cat([torch.arange(N, 2 * N, device=z.device),
                         torch.arange(0, N,      device=z.device)])
    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, a):
        return F.relu(self.W(a @ x))


class GCNEncoder(nn.Module):
    def __init__(self, in_dim=16, hidden=32, dropout=0.4):
        super().__init__()
        self.gcn1    = GCNLayer(in_dim, hidden)
        self.gcn2    = GCNLayer(hidden, hidden)
        self.drop    = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x, a):
        h = self.gcn1(x, a)
        h = self.drop(h)
        h = self.gcn2(h, a)
        h = self.drop(h)
        return h.mean(dim=0)   # global mean pool → (hidden,)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=32, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, h):
        return self.net(h)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=32, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze()


# ─────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval(encoder, clf_head, graphs, labels, device):
    encoder.eval(); clf_head.eval()
    logits = [clf_head(encoder(x.to(device), a.to(device))).cpu().item()
              for x, a in graphs]
    logits  = np.array(logits, dtype=np.float32)
    probs   = 1.0 / (1.0 + np.exp(-logits))
    preds   = (probs >= 0.5).astype(np.int64)
    labels  = np.asarray(labels)
    eps     = 1e-7
    bce     = float(-np.mean(labels * np.log(probs + eps)
                             + (1 - labels) * np.log(1 - probs + eps)))
    auc     = float(roc_auc_score(labels, probs)) \
              if len(np.unique(labels)) == 2 else float('nan')
    return probs, preds, bce, auc


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
# SSL pre-training
# ─────────────────────────────────────────────────────────────

def ssl_pretrain(encoder, proj_head, train_graphs, ssl_epochs, lr,
                 device, batch_size=32, temperature=0.5, verbose=True):
    """Pre-train on INNER TRAIN graphs only. Returns per-epoch NT-Xent loss."""
    encoder.train(); proj_head.train()
    params    = list(encoder.parameters()) + list(proj_head.parameters())
    optimiser = Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=ssl_epochs, eta_min=lr * 0.1)
    losses    = []

    for ep in range(ssl_epochs):
        idx     = np.random.permutation(len(train_graphs))
        ep_loss = 0.0
        n_b     = 0
        for start in range(0, len(train_graphs), batch_size):
            batch = idx[start: start + batch_size]
            if len(batch) < 2:
                continue
            z1_list, z2_list = [], []
            for i in batch:
                x, a = train_graphs[i]
                (x1, a1), (x2, a2) = random_augment(x, a)
                z1_list.append(proj_head(encoder(x1.to(device), a1.to(device))))
                z2_list.append(proj_head(encoder(x2.to(device), a2.to(device))))
            loss = nt_xent_loss(torch.stack(z1_list), torch.stack(z2_list), temperature)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            ep_loss += loss.item()
            n_b     += 1
        scheduler.step()
        avg = ep_loss / max(n_b, 1)
        losses.append(avg)
        if verbose and (ep + 1) % 50 == 0:
            print(f'    SSL [{ep + 1:3d}/{ssl_epochs}]  NT-Xent: {avg:.4f}')
    return losses


# ─────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def linear_probe_auc(encoder, graphs, labels, device):
    """
    AUC of SSL encoder embeddings without any fine-tuning.
    Scores by projecting onto (mean_ictal - mean_preictal) direction.
    High AUC before fine-tuning = encoder learned discriminative features
    from unlabelled data (supports the SSL motivation).
    """
    encoder.eval()
    embs   = np.stack([encoder(x.to(device), a.to(device)).cpu().numpy()
                       for x, a in graphs])
    labels = np.asarray(labels)
    pos_m  = embs[labels == 1].mean(axis=0) if (labels == 1).any() else embs.mean(axis=0)
    neg_m  = embs[labels == 0].mean(axis=0) if (labels == 0).any() else embs.mean(axis=0)
    scores = embs @ (pos_m - neg_m)
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float('nan')


# ─────────────────────────────────────────────────────────────
# Fine-tuning (two-phase)
# ─────────────────────────────────────────────────────────────

def finetune(encoder, clf_head,
             g_train, y_train, g_val, y_val,
             ft_epochs, lr, pos_weight,
             device, patience=20, freeze_encoder=False):
    """
    Early stopping monitored on INNER VAL AUC — test patient is never seen.
    Phase A (freeze_encoder=True): linear probe warmup.
    Phase B (freeze_encoder=False): full fine-tune.
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
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, verbose=False)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    train_losses, val_losses   = [], []
    train_aucs_ep, val_aucs_ep = [], []
    best_auc       = 0.0
    best_val_loss  = np.inf
    best_enc_state = None
    best_clf_state = None
    patience_cnt   = 0

    for ep in range(ft_epochs):
        encoder.train(); clf_head.train()
        ep_loss = 0.0
        for i in np.random.permutation(len(g_train)):
            x, a = g_train[i]
            x, a = x.to(device), a.to(device)
            optimiser.zero_grad()
            logit = clf_head(encoder(x, a))
            label = torch.tensor([float(y_train[i])], device=device)
            loss  = criterion(logit.unsqueeze(0), label)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            ep_loss += loss.item()
        train_losses.append(ep_loss / len(g_train))

        _, _, val_bce, val_auc = _eval(encoder, clf_head, g_val, y_val, device)
        val_losses.append(val_bce)
        val_aucs_ep.append(val_auc if not np.isnan(val_auc) else 0.0)
        scheduler.step(val_bce)

        _, _, _, tr_auc = _eval(encoder, clf_head, g_train, y_train, device)
        train_aucs_ep.append(tr_auc if not np.isnan(tr_auc) else 0.0)

        # Early stopping on inner-val AUC
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc       = val_auc
            best_enc_state = copy.deepcopy(encoder.state_dict())
            best_clf_state = copy.deepcopy(clf_head.state_dict())
            patience_cnt   = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_enc_state:
        encoder.load_state_dict(best_enc_state)
    if best_clf_state:
        clf_head.load_state_dict(best_clf_state)

    return train_losses, val_losses, train_aucs_ep, val_aucs_ep, best_auc


# ─────────────────────────────────────────────────────────────
# Overfitting plots
# ─────────────────────────────────────────────────────────────

def plot_ssl_loss_and_probe(ssl_losses, probe_auc, patient_id, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [3, 1]})

    ax = axes[0]
    ax.plot(ssl_losses, color='mediumpurple', lw=2)
    ax.set_xlabel('SSL Epoch', fontsize=11)
    ax.set_ylabel('NT-Xent Loss', fontsize=11)
    ax.set_title(f'SSL Pre-training Loss | excl. {patient_id}',
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if ssl_losses:
        drop = ssl_losses[0] - ssl_losses[-1]
        ax.text(0.97, 0.97, f'Total drop: {drop:.3f}', transform=ax.transAxes,
                fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    ax = axes[1]
    color = 'seagreen' if probe_auc > 0.6 else ('darkorange' if probe_auc > 0.5 else 'tomato')
    ax.bar(['SSL probe\nAUC'], [probe_auc], color=color, edgecolor='black', alpha=0.85)
    ax.axhline(0.5, color='red', linestyle='--', lw=1, label='Chance')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('Linear probe\n(frozen encoder)', fontsize=10, fontweight='bold')
    ax.text(0, probe_auc + 0.02, f'{probe_auc:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    plt.suptitle(f'SSL quality | Test: {patient_id}', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / f'ssl_overview_{patient_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_ft_training_curves(train_losses, val_losses, train_aucs, val_aucs,
                             patient_id, output_dir):
    best_ep = int(np.argmax(val_aucs)) if val_aucs else 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(train_losses, color='steelblue',  lw=1.5, label='Train loss')
    ax.plot(val_losses,   color='tomato',     lw=1.5, linestyle='--', label='Inner-val loss')
    ax.axvline(best_ep, color='green', lw=1, linestyle=':', label=f'Best ({best_ep})')
    gap = abs(train_losses[-1] - val_losses[-1]) if train_losses else 0
    ax.text(0.97, 0.97, f'Final gap: {gap:.4f}', transform=ax.transAxes,
            fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('BCE Loss', fontsize=11)
    ax.set_title(f'FT loss | Test: {patient_id}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(train_aucs, color='steelblue', lw=1.5, label='Train AUC')
    ax.plot(val_aucs,   color='tomato',    lw=1.5, linestyle='--', label='Inner-val AUC')
    ax.axvline(best_ep, color='green', lw=1, linestyle=':', label=f'Best ({best_ep})')
    if train_aucs and val_aucs:
        auc_gap = train_aucs[-1] - val_aucs[-1]
        ax.text(0.97, 0.03, f'Final AUC gap: {auc_gap:+.3f}',
                transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('AUC', fontsize=11)
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f'FT AUC | Test: {patient_id}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f'SSL-GCN Fine-tuning | Test: {patient_id}', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / f'ft_training_curves_{patient_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_train_vs_test_gap(fold_metrics, output_dir):
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
    ax.set_xticks(x); ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('AUC', fontsize=12); ax.set_ylim(0, 1.22)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('SSL-GCN — Train / Inner-val / Test AUC per fold\n'
                 '(Δ = train−test; near zero = no overfitting)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'train_vs_test_gap_ssl.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting_summary(fold_metrics, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    gaps     = [m.get('train_auc', float('nan')) - m['auc'] for m in fold_metrics]
    colors   = ['tomato' if g > 0.1 else 'steelblue' for g in gaps]
    fig, ax  = plt.subplots(figsize=(10, 4))
    bars     = ax.bar(patients, gaps, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0,   color='black', lw=1)
    ax.axhline(0.1, color='red',   lw=1, linestyle='--', alpha=0.5, label='0.10 threshold')
    ax.set_ylabel('Train AUC − Test AUC', fontsize=12)
    ax.set_title('SSL-GCN — Overfitting Gap per Patient Fold', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    for bar, g in zip(bars, gaps):
        if not np.isnan(g):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f'{g:.2f}',
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_summary_ssl.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────
# Standard plots
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, patient_id, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=['Pre-ictal', 'Ictal'],
                yticklabels=['Pre-ictal', 'Ictal'])
    ax.set_xlabel('Predicted', fontsize=11); ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'SSL-GCN CM | Test: {patient_id}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_ssl_{patient_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_all_folds(fold_roc_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for fpr, tpr, auc, pat in fold_roc_data:
        ax.plot(fpr, tpr, alpha=0.55, lw=1.5, label=f'{pat} (AUC={auc:.2f})')
        aucs.append(auc)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'SSL-GCN LOPO ROC\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_ssl_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    colors   = ['mediumpurple', 'tomato', 'seagreen', 'darkorange', 'steelblue']
    x, w     = np.arange(len(patients)), 0.16
    fig, ax  = plt.subplots(figsize=(14, 5))
    for i, (met, col) in enumerate(zip(met_keys, colors)):
        ax.bar(x + i * w, [m[met] for m in fold_metrics], w,
               label=met.upper(), color=col, alpha=0.8, edgecolor='black')
    ax.set_xticks(x + 2 * w); ax.set_xticklabels(patients, rotation=30, fontsize=10)
    ax.set_ylabel('Score', fontsize=12); ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('SSL-GCN — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
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
        ax.set_xlabel('Predicted', fontsize=11); ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'SSL-GCN Aggregate CM ({title})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cm_aggregate_ssl_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_final_comparison(ssl_stats, sup_json, baseline_json, output_dir):
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    models   = {}
    if baseline_json and Path(baseline_json).exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        for name, res in bl.items():
            if res.get('summary_stats'):
                models[name] = {k: res['summary_stats'].get(k, {}).get('mean', 0) for k in met_keys}
    if sup_json and Path(sup_json).exists():
        with open(sup_json) as f:
            sup = json.load(f)
        if sup.get('summary_stats'):
            models['GCN (Supervised)'] = {k: sup['summary_stats'].get(k, {}).get('mean', 0)
                                          for k in met_keys}
    models['SSL-GCN (Ours)'] = {k: ssl_stats.get(k, {}).get('mean', 0) for k in met_keys}

    colors = ['steelblue', 'tomato', 'darkorange', 'seagreen', 'mediumpurple']
    x, n, w = np.arange(len(met_keys)), len(models), 0.15
    fig, ax  = plt.subplots(figsize=(13, 5))
    for i, (name, vals) in enumerate(models.items()):
        ax.bar(x + (i - n / 2 + 0.5) * w,
               [vals.get(k, 0) for k in met_keys], w,
               label=name, color=colors[i % len(colors)], alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Mean Score (LOPO)', fontsize=12); ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('Full Model Comparison — Nested LOPO CV', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Final comparison → {output_dir / "comparison_all_models.png"}')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Step 6 — SSL GCN (leakage-free, nested LOPO, data-driven threshold, rotating inner val)'
    )
    parser.add_argument('--featfile',       required=True)
    parser.add_argument('--outputdir',      default='results/ssl_gnn')
    parser.add_argument('--ssl_epochs',     type=int,   default=200)
    parser.add_argument('--ft_epochs',      type=int,   default=100)
    parser.add_argument('--lr_ssl',         type=float, default=0.001)
    parser.add_argument('--lr_ft',          type=float, default=0.0005)
    parser.add_argument('--hidden',         type=int,   default=32)
    parser.add_argument('--proj_dim',       type=int,   default=32)
    parser.add_argument('--dropout',        type=float, default=0.4)
    parser.add_argument('--threshold_pct',  type=float, default=70.0,
                        help='DTF percentile threshold per fold (default 70 → top 30%% edges)')
    parser.add_argument('--temperature',    type=float, default=0.5)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--patience',       type=int,   default=20)
    parser.add_argument('--baseline_json',  default=None)
    parser.add_argument('--sup_json',       default=None)
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 6 — SSL GCN (leakage-free, nested LOPO)')
    print('=' * 65)
    print(f'Device       : {device}')
    print(f'SSL epochs   : {args.ssl_epochs}  LR: {args.lr_ssl}  Temp: {args.temperature}')
    print(f'FT  epochs   : {args.ft_epochs}   LR: {args.lr_ft}')
    print(f'Hidden: {args.hidden}  ProjDim: {args.proj_dim}  Dropout: {args.dropout}')
    print(f'Threshold pct: {args.threshold_pct}  (data-driven, per fold, from inner-train)')
    print()
    print('Overfitting prevention:')
    print('  SSL: contrastive augmentation (edge_dropout, node_noise, band_mask)')
    print('  SSL: CosineAnnealingLR, gradient clipping(1.0), weight_decay=1e-4')
    print('  FT : two-phase (linear probe then full), dropout(0.4)')
    print('  FT : early stopping on INNER val AUC (test patient never seen)')
    print('  FT : rotating inner val patient (avoids fixed validator bias)')
    print('  All: StandardScaler fit on inner-train only')
    print('  All: data-driven threshold per fold (no fixed arbitrary value)')

    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)
    #adj_dtf     = data['adj_dtf'].astype(np.float32)
    adj_dtf     = data['adj_pdc'].astype(np.float32)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']

    print(f'\nLoaded: {len(y)} epochs | Ictal: {(y == 1).sum()} | Pre-ictal: {(y == 0).sum()}')
    sorted_all_pats = sorted(np.unique(patient_ids).tolist())
    print(f'Patients: {sorted_all_pats}')

    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []

    for fold_idx, pat in enumerate(sorted_all_pats):
        print(f'\n{"─" * 65}')
        print(f'  Fold {fold_idx + 1}/{len(sorted_all_pats)} — Test patient: {pat}')
        print(f'{"─" * 65}')

        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask
        train_idx  = np.where(train_mask)[0]
        test_idx   = np.where(test_mask)[0]

        y_train    = y[train_idx]
        y_test     = y[test_idx]
        train_pats = patient_ids[train_idx]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] Only one class in test')
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
        threshold = compute_threshold(adj_dtf[it_idx], args.threshold_pct)

        print(f'  Inner val: {inner_val_pat}  |  Threshold: {threshold:.4f} (p{args.threshold_pct:.0f})')
        print(f'  Inner train: {len(it_idx)} epochs  |  Inner val: {len(iv_idx)} epochs  |  Test: {len(test_idx)} epochs')

        # ── Build graphs with this fold's threshold ─────────────
        g_all_it = build_graphs(node_feats[it_idx],    adj_dtf[it_idx],    threshold)
        g_all_iv = build_graphs(node_feats[iv_idx],    adj_dtf[iv_idx],    threshold)
        g_all_te = build_graphs(node_feats[test_idx],  adj_dtf[test_idx],  threshold)

        # ── Scale: fit on inner-train only ──────────────────────
        g_it, g_iv, g_te = scale_three_splits(g_all_it, g_all_iv, g_all_te)

        n_neg      = int((y_it == 0).sum())
        n_pos      = int((y_it == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        # ── Phase 1: SSL pre-training (inner-train only) ────────
        print(f'\n  Phase 1 — SSL ({args.ssl_epochs} epochs, {len(g_it)} graphs, no labels)')
        encoder   = GCNEncoder(in_dim=16, hidden=args.hidden, dropout=args.dropout).to(device)
        proj_head = ProjectionHead(in_dim=args.hidden, proj_dim=args.proj_dim).to(device)

        ssl_losses = ssl_pretrain(
            encoder, proj_head, g_it,
            ssl_epochs=args.ssl_epochs, lr=args.lr_ssl,
            device=device, batch_size=args.batch_size,
            temperature=args.temperature, verbose=True,
        )

        # Linear probe on inner-val
        probe_auc = linear_probe_auc(encoder, g_iv, y_iv, device)
        print(f'  SSL final NT-Xent: {ssl_losses[-1]:.4f}  |  Probe AUC (inner val): {probe_auc:.3f}')
        plot_ssl_loss_and_probe(ssl_losses, probe_auc, pat, output_dir)

        clf_head = ClassifierHead(in_dim=args.hidden, dropout=args.dropout).to(device)

        # ── Phase 2A: Linear probe warmup (frozen encoder) ─────
        print(f'\n  Phase 2A — Linear probe FT (encoder frozen, 30 epochs)')
        finetune(encoder, clf_head, g_it, y_it, g_iv, y_iv,
                 ft_epochs=30, lr=args.lr_ft * 5, pos_weight=pos_weight,
                 device=device, patience=args.patience, freeze_encoder=True)

        # ── Phase 2B: Full fine-tune ────────────────────────────
        print(f'\n  Phase 2B — Full FT ({args.ft_epochs} epochs, early stop on inner val AUC)')
        tr_losses, val_losses, tr_aucs, val_aucs, best_auc = finetune(
            encoder, clf_head, g_it, y_it, g_iv, y_iv,
            ft_epochs=args.ft_epochs, lr=args.lr_ft, pos_weight=pos_weight,
            device=device, patience=args.patience, freeze_encoder=False,
        )
        print(f'  Best inner val AUC: {best_auc:.3f}')
        plot_ft_training_curves(tr_losses, val_losses, tr_aucs, val_aucs, pat, output_dir)

        # ── Final evaluation on outer TEST patient ───────────────
        probs, preds, _, _ = _eval(encoder, clf_head, g_te, y_test, device)
        metrics = compute_metrics(y_test, preds, probs)
        if metrics is None:
            continue

        _, _, _, final_tr_auc = _eval(encoder, clf_head, g_it, y_it, device)
        overfit_gap = final_tr_auc - metrics['auc']

        metrics['patient']        = pat
        metrics['n_train']        = int(train_mask.sum())
        metrics['n_test']         = int(test_mask.sum())
        metrics['inner_val_pat']  = inner_val_pat
        metrics['inner_val_auc']  = float(best_auc)
        metrics['train_auc']      = float(final_tr_auc)
        metrics['overfit_gap']    = float(overfit_gap)
        metrics['probe_auc']      = float(probe_auc)
        metrics['ssl_final_loss'] = float(ssl_losses[-1])
        metrics['threshold']      = float(threshold)
        fold_metrics.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(preds.tolist())
        fpr, tpr, _ = roc_curve(y_test, probs)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        print(f'\n  {pat:8s} | Test AUC={metrics["auc"]:.3f}  Acc={metrics["accuracy"]:.3f}'
              f'  F1={metrics["f1"]:.3f}  Sens={metrics["sensitivity"]:.3f}'
              f'  Spec={metrics["specificity"]:.3f}  MCC={metrics["mcc"]:.3f}')
        print(f'           | Train AUC={final_tr_auc:.3f}  '
              f'Overfit gap={overfit_gap:+.3f}  '
              f'Inner val AUC={best_auc:.3f}  Probe AUC={probe_auc:.3f}  '
              f'Threshold={threshold:.4f}')

        plot_confusion_matrix(confusion_matrix(y_test, preds), pat, output_dir)

    # ── Aggregate plots ─────────────────────────────────────────
    if fold_roc_data:
        plot_roc_all_folds(fold_roc_data, output_dir)
    if fold_metrics:
        plot_per_fold_metrics(fold_metrics, output_dir)
        plot_train_vs_test_gap(fold_metrics, output_dir)
        plot_overfitting_summary(fold_metrics, output_dir)
    if all_y_true:
        plot_aggregate_confusion(np.array(all_y_true), np.array(all_y_pred), output_dir)

    # ── Summary ─────────────────────────────────────────────────
    met_keys      = ['auc', 'f1', 'sensitivity', 'specificity', 'precision', 'accuracy', 'mcc']
    summary_stats = {}
    print(f'\n{"=" * 65}')
    print(f'SSL-GCN — Mean ± Std across {len(fold_metrics)} folds')
    print(f'{"=" * 65}')
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

    probe_aucs = [m.get('probe_auc', float('nan')) for m in fold_metrics]
    probe_aucs = [v for v in probe_aucs if not np.isnan(v)]
    if probe_aucs:
        print(f'  Mean SSL probe AUC: {np.mean(probe_aucs):.3f} ± {np.std(probe_aucs):.3f}')
        summary_stats['ssl_probe_auc'] = {'mean': float(np.mean(probe_aucs)),
                                          'std':  float(np.std(probe_aucs))}

    thresholds = [m.get('threshold', float('nan')) for m in fold_metrics]
    thresholds = [v for v in thresholds if not np.isnan(v)]
    if thresholds:
        print(f'  DTF threshold (mean): {np.mean(thresholds):.4f} ± {np.std(thresholds):.4f}')

    results = {
        'model':           'SSL_GCN_GraphCL_NestedLOPO_LeakageFree',
        'hyperparameters': vars(args),
        'threshold_note':  f'Data-driven p{args.threshold_pct:.0f} percentile per fold from inner-train adjacency',
        'inner_val_note':  'Rotating inner val patient — cycles with fold index to avoid fixed validator bias',
        'overfitting_prevention': [
            'contrastive_augmentation (edge_dropout, node_noise, band_mask)',
            'cosine_annealing_lr_ssl',
            'gradient_clipping (max_norm=1.0)',
            'l2_weight_decay=1e-4',
            'dropout=0.4 (encoder + classifier head)',
            'two_phase_finetuning (linear_probe_then_full)',
            f'early_stopping (patience={args.patience}) on inner_val_patient AUC',
            'standard_scaler_fit_on_inner_train_only',
            f'data_driven_threshold (p{args.threshold_pct:.0f} percentile, per fold)',
            'rotating_inner_val_patient (no fixed validator)',
        ],
        'fold_metrics':    fold_metrics,
        'summary_stats':   summary_stats,
    }
    results_path = output_dir / 'results_ssl.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults → {results_path}')

    plot_final_comparison(summary_stats, args.sup_json, args.baseline_json, output_dir)

    print('\n' + '=' * 65)
    print('STEP 6 COMPLETE')
    print('=' * 65)


if __name__ == '__main__':
    main()
