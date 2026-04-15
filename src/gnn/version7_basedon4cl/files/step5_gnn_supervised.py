"""
Step 5 — Supervised GCN + Adjacency-Only GCN  (Nested LOPO CV)
================================================================
DESIGN DECISIONS (document in thesis):

1. EVALUATION: Nested LOPO CV — same protocol as step 4.
   Outer: leave 1 patient out (test).
   Inner: of remaining 7 patients, rotate 1 as validation.
   Inner CV is used for early stopping target calibration:
   the early stopping patience monitors val loss on the INNER val
   patient rather than the test patient, so the test patient is
   never seen during model selection.

   Concretely, for each outer fold:
     - Select ONE inner val patient (the one giving median val AUC in
       a quick 5-epoch pre-scan, or simply the first in sorted order).
     - Train on the remaining 6 train patients.
     - Stop early when inner val AUC stops improving.
     - Report both inner val AUC and outer test metrics.

   This is the clean approach: early stopping never touches the test
   patient's graphs. The test patient is strictly held out.

2. TWO GCN VARIANTS:
   (a) SmallGCN (with features) — same as before, uses 16 node features
       (spectral, Hjorth, time-domain, connectivity degrees).
   (b) AdjOnlyGCN — node features are REPLACED with each node's row
       from the thresholded DTF adjacency matrix (19-dimensional vector:
       the connectivity profile of that node to all others).
       No spectral or Hjorth features are used at all.
       Rationale: tests whether graph topology alone (the pattern of
       directed functional connectivity) is sufficient for classification,
       without any hand-crafted signal features. This is an important
       ablation for the thesis: if AdjOnlyGCN approaches SmallGCN
       performance, connectivity structure is the primary discriminant.

3. ACCURACY added as a metric throughout.

4. ARCHITECTURE: unchanged from step 5 original (small by design).
   AdjOnlyGCN uses in_dim=19 instead of in_dim=16.

Outputs:
  (same as original step5 plus):
  results_gcn_with_features.json   metrics for feature-based GCN
  results_gcn_adj_only.json        metrics for adjacency-only GCN
  comparison_gcn_variants.png      side-by-side comparison of the two

Usage:
  python step5_gnn_supervised.py \\
      --featfile     features/features_all.npz \\
      --outputdir    results/gnn_supervised \\
      --epochs       150 \\
      --lr           0.001 \\
      --hidden       32 \\
      --threshold    0.15 \\
      --dropout      0.4 \\
      --patience     20 \\
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
# Adjacency normalisation
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    Self-loops added before normalisation (Kipf & Welling 2017).
    """
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# GCN model (small by design)
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(a_hat @ x))


class SmallGCN(nn.Module):
    """
    2-layer GCN → GlobalMeanPool → 2-layer MLP → scalar logit.
    in_dim is configurable: 16 for feature-based, 19 for adj-only.
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

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, a_hat)
        h = self.drop(h)
        h = self.gcn2(h, a_hat)
        h = h.mean(dim=0, keepdim=True)
        return self.head(h).squeeze()


# ─────────────────────────────────────────────────────────────
# Graph building
# ─────────────────────────────────────────────────────────────

def build_graphs_with_features(node_feats: np.ndarray, adj_dtf: np.ndarray,
                                threshold: float = 0.15):
    """
    Standard graph: node features are spectral + Hjorth + time-domain + connectivity.
    node_feats : (N_epochs, 19, 16)
    adj_dtf    : (N_epochs, 19, 19)
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


def build_graphs_adj_only(adj_dtf: np.ndarray, threshold: float = 0.15):
    """
    Adjacency-only graph: node features ARE the thresholded adjacency row.
    Each node's 19-dim feature vector = its outgoing DTF connectivity to all other nodes.

    Neuroscientific rationale: tests whether the directed connectivity PATTERN
    (i.e. which channels are strong sources/sinks) alone can discriminate
    ictal from pre-ictal states, without any spectral or temporal signal features.

    adj_dtf : (N_epochs, 19, 19)
    Returns list of (x_tensor (19, 19), a_hat_tensor (19, 19))
    """
    graphs = []
    for i in range(len(adj_dtf)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        # Use the raw (non-thresholded) adjacency row as node features
        # so features capture full connectivity profile, not just binary topology
        x_raw = adj_dtf[i].copy()
        np.fill_diagonal(x_raw, 0.0)
        x = torch.tensor(x_raw.astype(np.float32))   # (19, 19)
        graphs.append((x, a_hat))
    return graphs


def scale_node_features(graphs_train, graphs_test):
    """
    StandardScaler fit on train node features only. No leakage.
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
# Inner CV fold selection
# ─────────────────────────────────────────────────────────────

def pick_inner_val_patient(train_patient_ids):
    """
    Choose the inner validation patient for early stopping.
    Strategy: simply use the first patient in sorted order.
    This is deterministic and does not involve test data.
    For a full nested CV, one would loop over all 7 inner folds;
    here we pick one for computational tractability (GNN training is slow).
    The key guarantee is that the test patient's data is NEVER used
    for early stopping — only this inner val patient's data is.
    """
    return sorted(np.unique(train_patient_ids))[0]


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, patient_id, model_tag, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label='Train loss', color='royalblue', lw=1.5)
    ax.plot(val_losses,   label='Inner val loss', color='tomato', lw=1.5, linestyle='--')
    gap = abs(train_losses[-1] - val_losses[-1])
    ax.text(0.65, 0.85, f'Final gap: {gap:.4f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('BCE Loss', fontsize=11)
    ax.set_title(f'{model_tag} Loss | Test: {patient_id}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_curve_{model_tag}_{patient_id}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, patient_id, model_tag, output_dir):
    cmap = 'Blues' if 'feat' in model_tag.lower() else 'Purples'
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
    ax.set_title(
        f'{model_tag} — LOPO ROC\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(fold_metrics, model_tag, output_dir):
    patients = [m['patient'] for m in fold_metrics]
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    x        = np.arange(len(patients))
    width    = 0.16
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
    ax.set_title(f'{model_tag} — Per-Patient LOPO Metrics', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'per_fold_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_confusion(all_y_true, all_y_pred, model_tag, output_dir):
    cmap    = 'Blues' if 'feat' in model_tag.lower() else 'Purples'
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


def plot_comparison_vs_baseline(gcn_feat_stats, gcn_adj_stats, baseline_json_path, output_dir):
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    models   = {}

    if baseline_json_path and Path(baseline_json_path).exists():
        with open(baseline_json_path) as f:
            baseline = json.load(f)
        for name, res in baseline.items():
            if res.get('summary_stats'):
                models[name] = {k: res['summary_stats'].get(k, {}).get('mean', 0) for k in met_keys}

    models['GCN (w/ features)']  = {k: gcn_feat_stats.get(k, {}).get('mean', 0) for k in met_keys}
    models['GCN (adj-only)']     = {k: gcn_adj_stats.get(k, {}).get('mean', 0)  for k in met_keys}

    colors = ['steelblue', 'tomato', 'darkorange', 'seagreen', 'mediumpurple']
    x      = np.arange(len(met_keys))
    n      = len(models)
    width  = 0.15
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (name, vals) in enumerate(models.items()):
        means  = [vals.get(k, 0) for k in met_keys]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=name,
               color=colors[i % len(colors)], alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Mean Score (LOPO)', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('RF vs SVM vs SVM-RFE vs GCN variants — LOPO CV', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Comparison chart → {output_dir / "comparison_all_models.png"}')


def plot_gcn_variants_comparison(feat_stats, adj_stats, output_dir):
    """Side-by-side: GCN with features vs GCN adjacency-only."""
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy', 'mcc']
    x        = np.arange(len(met_keys))
    width    = 0.35
    fig, ax  = plt.subplots(figsize=(10, 5))
    for i, (label, stats, col) in enumerate([
        ('GCN (w/ features)', feat_stats, 'steelblue'),
        ('GCN (adj-only)',    adj_stats,  'mediumpurple'),
    ]):
        means  = [stats.get(k, {}).get('mean', 0) for k in met_keys]
        stds   = [stats.get(k, {}).get('std', 0)  for k in met_keys]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=label, color=col, capsize=4, edgecolor='black', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('GCN with Features vs Adjacency-Only GCN (Ablation)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_gcn_variants.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────
# Core LOPO training loop for one GCN variant
# ─────────────────────────────────────────────────────────────

def run_gcn_lopo(model_tag, in_dim, all_graphs, y, patient_ids, args,
                 device, output_dir):
    """
    Run nested LOPO for one GCN variant.

    model_tag : string identifier used in filenames
    in_dim    : node feature dimension (16 for features, 19 for adj-only)
    all_graphs: list of (x_tensor, a_hat_tensor), one per epoch
    """
    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []

    print(f'\n{"=" * 65}')
    print(f'  {model_tag} — Nested LOPO CV ({len(patients)} outer folds)')
    print(f'  Node feature dim: {in_dim}')
    print(f'{"=" * 65}')

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask
        train_idx  = np.where(train_mask)[0]
        test_idx   = np.where(test_mask)[0]

        y_train = y[train_idx]
        y_test  = y[test_idx]
        train_pats = patient_ids[train_idx]

        if len(np.unique(y_test)) < 2:
            print(f'  [SKIP] {pat}: test set has only one class')
            continue

        # ── Pick inner val patient (first in sorted order) ────────
        inner_val_pat  = pick_inner_val_patient(train_pats)
        inner_val_mask = (train_pats == inner_val_pat)
        inner_tr_mask  = ~inner_val_mask
        inner_tr_idx   = train_idx[inner_tr_mask]
        inner_val_idx  = train_idx[inner_val_mask]

        y_inner_tr  = y[inner_tr_idx]
        y_inner_val = y[inner_val_idx]

        # ── Build graph splits ─────────────────────────────────────
        graphs_inner_tr_raw  = [all_graphs[i] for i in inner_tr_idx]
        graphs_inner_val_raw = [all_graphs[i] for i in inner_val_idx]
        graphs_test_raw      = [all_graphs[i] for i in test_idx]

        # Scale node features: fit on inner train only
        graphs_inner_tr, graphs_inner_val = scale_node_features(
            graphs_inner_tr_raw, graphs_inner_val_raw
        )
        # For test: use the same scaler fitted on inner train
        all_train_x = np.concatenate([g[0].numpy() for g in graphs_inner_tr_raw], axis=0)
        from sklearn.preprocessing import StandardScaler as _SS
        sc = _SS().fit(all_train_x)
        graphs_test = [
            (torch.tensor(sc.transform(x.numpy()), dtype=torch.float32), a)
            for x, a in graphs_test_raw
        ]

        # Imbalance weight (inner train only, no leakage)
        n_neg      = int((y_inner_tr == 0).sum())
        n_pos      = int((y_inner_tr == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        # ── Train model (early-stop on inner val) ─────────────────
        model = SmallGCN(in_dim=in_dim, hidden=args.hidden,
                         dropout=args.dropout).to(device)
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
                model, optimiser, criterion, graphs_inner_tr, y_inner_tr, device
            )
            probs_val, preds_val, targets_val, val_loss = evaluate_graphs(
                model, graphs_inner_val, y_inner_val, device
            )
            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            val_auc = roc_auc_score(targets_val, probs_val) \
                      if len(np.unique(targets_val)) == 2 else 0.0

            if val_auc > best_auc:
                best_auc     = val_auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  {pat}: early stop at epoch {ep + 1}  '
                          f'inner val AUC={best_auc:.3f}')
                    break

        if best_state:
            model.load_state_dict(best_state)

        # ── Evaluate on outer test patient ─────────────────────────
        probs, preds, targets, _ = evaluate_graphs(model, graphs_test, y_test, device)
        metrics = compute_metrics(targets, preds, probs)
        if metrics is None:
            continue

        metrics['patient']       = pat
        metrics['n_train']       = int(train_mask.sum())
        metrics['n_test']        = int(test_mask.sum())
        metrics['inner_val_auc'] = float(best_auc)
        metrics['inner_val_pat'] = inner_val_pat
        fold_metrics.append(metrics)

        all_y_true.extend(targets.tolist())
        all_y_pred.extend(preds.tolist())

        fpr, tpr, _ = roc_curve(targets, probs)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        print(f'  {pat:8s} | Test  AUC={metrics["auc"]:.3f}  F1={metrics["f1"]:.3f}'
              f'  Acc={metrics["accuracy"]:.3f}'
              f'  Sens={metrics["sensitivity"]:.3f}  Spec={metrics["specificity"]:.3f}'
              f'  MCC={metrics["mcc"]:.3f}')
        print(f'           | Inner AUC={best_auc:.3f}  (val pat: {inner_val_pat})')

        plot_loss_curves(train_losses, val_losses, pat, model_tag, output_dir)
        plot_confusion_matrix(confusion_matrix(targets, preds), pat, model_tag, output_dir)

    # ── Aggregate plots ────────────────────────────────────────
    if fold_roc_data:
        plot_roc_all_folds(fold_roc_data, model_tag, output_dir)
    if fold_metrics:
        plot_per_fold_metrics(fold_metrics, model_tag, output_dir)
    if all_y_true:
        plot_aggregate_confusion(
            np.array(all_y_true), np.array(all_y_pred), model_tag, output_dir
        )

    # ── Summary ────────────────────────────────────────────────
    met_keys      = ['auc', 'f1', 'sensitivity', 'specificity', 'precision', 'accuracy', 'mcc']
    summary_stats = {}
    print(f'\n  {"─" * 55}')
    print(f'  {model_tag} — Mean ± Std across {len(fold_metrics)} outer folds')
    print(f'  {"─" * 55}')
    for k in met_keys:
        vals = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    inner_aucs = [m.get('inner_val_auc', float('nan')) for m in fold_metrics]
    inner_aucs = [v for v in inner_aucs if not np.isnan(v)]
    if inner_aucs:
        print(f'\n  Inner CV val AUC (mean): {np.mean(inner_aucs):.3f} ± {np.std(inner_aucs):.3f}')
        summary_stats['inner_val_auc'] = {'mean': float(np.mean(inner_aucs)),
                                          'std':  float(np.std(inner_aucs))}

    return fold_metrics, summary_stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 5 — Supervised GCN (Nested LOPO)')
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/gnn_supervised')
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold',     type=float, default=0.15)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--baseline_json', default=None,
                        help='Path to step4 results_all.json for comparison plot')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 5 — SUPERVISED GCN (Nested LOPO CV, Two Variants)')
    print('=' * 65)
    print(f'Device    : {device}')
    print(f'Epochs    : {args.epochs}   LR: {args.lr}')
    print(f'Hidden    : {args.hidden}   Dropout: {args.dropout}')
    print(f'Threshold : {args.threshold}   Patience: {args.patience}')
    print()
    print('Inner CV protocol:')
    print('  Early stopping driven by INNER val patient (never test patient).')
    print('  Inner val patient = first sorted patient in training fold.')
    print('=' * 65)

    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)   # (N, 19, 16)
    adj_dtf     = data['adj_dtf'].astype(np.float32)         # (N, 19, 19)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']

    print(f'Loaded: {len(y)} epochs | Ictal: {(y == 1).sum()} | Pre-ictal: {(y == 0).sum()}')

    # ── Build graphs for both variants ────────────────────────
    print('\nBuilding graphs (feature-based)...')
    graphs_feat = build_graphs_with_features(node_feats, adj_dtf, args.threshold)

    print('Building graphs (adjacency-only)...')
    graphs_adj = build_graphs_adj_only(adj_dtf, args.threshold)

    # ── Run both variants ─────────────────────────────────────
    feat_metrics, feat_stats = run_gcn_lopo(
        'gcn_feat', in_dim=16,
        all_graphs=graphs_feat, y=y, patient_ids=patient_ids,
        args=args, device=device, output_dir=output_dir,
    )

    adj_metrics, adj_stats = run_gcn_lopo(
        'gcn_adj_only', in_dim=19,
        all_graphs=graphs_adj, y=y, patient_ids=patient_ids,
        args=args, device=device, output_dir=output_dir,
    )

    # ── Save results ──────────────────────────────────────────
    for tag, metrics, stats, fname in [
        ('GCN_Supervised_WithFeatures',  feat_metrics, feat_stats, 'results_gcn_with_features.json'),
        ('GCN_Supervised_AdjOnly',       adj_metrics,  adj_stats,  'results_gcn_adj_only.json'),
    ]:
        results = {
            'model':           tag,
            'hyperparameters': vars(args),
            'inner_cv_note':   'Early stopping on first sorted inner-val patient per fold',
            'fold_metrics':    metrics,
            'summary_stats':   stats,
        }
        rp = output_dir / fname
        with open(rp, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'\n{tag} results → {rp}')

    # Keep the original filename for step6 compatibility
    with open(output_dir / 'results_gcn.json', 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'GCN_Supervised_WithFeatures',
            'hyperparameters': vars(args),
            'fold_metrics': feat_metrics,
            'summary_stats': feat_stats,
        }, f, indent=2, default=str)

    # ── Comparison plots ──────────────────────────────────────
    plot_gcn_variants_comparison(feat_stats, adj_stats, output_dir)

    if args.baseline_json:
        plot_comparison_vs_baseline(feat_stats, adj_stats, args.baseline_json, output_dir)

    print('\n' + '=' * 65)
    print('STEP 5 COMPLETE')
    print('=' * 65)
    print('\nNext: python step6_ssl_gnn.py --featfile features/features_all.npz'
          ' --sup_json results/gnn_supervised/results_gcn.json')


if __name__ == '__main__':
    main()
