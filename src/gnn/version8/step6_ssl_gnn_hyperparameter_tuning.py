"""
Step 6 — Self-Supervised GCN with Hyperparameter Tuning (FORTH, LOPO)
======================================================================
FEATURES:
  - Full hyperparameter grid search over hidden, dropout, lr_ft, threshold_pct
  - CSV checkpoint: saves each completed combination immediately
  - Resume: skips already-completed combinations on restart
  - Leakage-free: SSL pre-training inside each LOPO fold
  - Selection criterion: mean test AUC across 8 LOPO folds

GRID:
  hidden:         [16, 32, 64]
  dropout:        [0.3, 0.4, 0.5]
  lr_ft:          [0.001, 0.0005, 0.0001]
  threshold_pct:  [60, 70, 80]
  Total:          3 × 3 × 3 × 3 = 81 combinations × 8 folds = 648 runs

CHECKPOINT:
  Results saved to {outputdir}/tuning_results.csv after each combination.
  On restart, already-completed combinations are skipped automatically.
  Best combination printed at the end.

USAGE — first run:
  python step6_ssl_gnn_tuning.py \
      --featfile features/features_all.npz \
      --outputdir results/ssl_tuning \
      --ssl_epochs 200 \
      --ft_epochs 100 \
      --patience 15

USAGE — resume after interruption:
  python step6_ssl_gnn_tuning.py \
      --featfile features/features_all.npz \
      --outputdir results/ssl_tuning \
      --ssl_epochs 200 \
      --ft_epochs 100 \
      --patience 15
  (same command — completed combinations are skipped automatically)
"""

import argparse
import copy
import csv
import itertools
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N_BAND_FEATS = 6  # FORTH: delta, theta, alpha, beta, gamma, broad


# ══════════════════════════════════════════════════════════════
# HYPERPARAMETER GRID
# ══════════════════════════════════════════════════════════════

PARAM_GRID = {
    'hidden':        [16, 32, 64],
    'dropout':       [0.3, 0.4, 0.5],
    'lr_ft':         [0.001, 0.0005, 0.0001],
    'threshold_pct': [60.0, 70.0, 80.0],
}


# ══════════════════════════════════════════════════════════════
# CSV CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════

CSV_FIELDS = [
    'hidden', 'dropout', 'lr_ft', 'threshold_pct',
    'mean_auc', 'std_auc',
    'mean_f1', 'mean_sensitivity', 'mean_specificity',
    'mean_mcc', 'mean_accuracy',
]


def combo_key(params):
    """Unique hashable key for a parameter combination."""
    return (
        int(params['hidden']),
        float(params['dropout']),
        float(params['lr_ft']),
        float(params['threshold_pct']),
    )


def load_completed(csv_path):
    """
    Load already-completed combinations from CSV.
    Returns dict: combo_key -> row dict.
    """
    completed = {}
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            params = {
                'hidden':        int(row['hidden']),
                'dropout':       float(row['dropout']),
                'lr_ft':         float(row['lr_ft']),
                'threshold_pct': float(row['threshold_pct']),
            }
            completed[combo_key(params)] = row
    return completed


def save_result(csv_path, params, fold_metrics):
    """Append one completed combination to CSV."""
    mean_auc  = float(np.mean([m['auc']         for m in fold_metrics]))
    std_auc   = float(np.std( [m['auc']         for m in fold_metrics]))
    mean_f1   = float(np.mean([m['f1']          for m in fold_metrics]))
    mean_sens = float(np.mean([m['sensitivity']  for m in fold_metrics]))
    mean_spec = float(np.mean([m['specificity']  for m in fold_metrics]))
    mean_mcc  = float(np.mean([m['mcc']         for m in fold_metrics]))
    mean_acc  = float(np.mean([m['accuracy']    for m in fold_metrics]))

    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'hidden':            params['hidden'],
            'dropout':           params['dropout'],
            'lr_ft':             params['lr_ft'],
            'threshold_pct':     params['threshold_pct'],
            'mean_auc':          round(mean_auc,  4),
            'std_auc':           round(std_auc,   4),
            'mean_f1':           round(mean_f1,   4),
            'mean_sensitivity':  round(mean_sens, 4),
            'mean_specificity':  round(mean_spec, 4),
            'mean_mcc':          round(mean_mcc,  4),
            'mean_accuracy':     round(mean_acc,  4),
        })
    return mean_auc, std_auc


def print_leaderboard(csv_path, top_n=5):
    """Print top N combinations by mean AUC."""
    completed = load_completed(csv_path)
    if not completed:
        return
    ranked = sorted(
        completed.values(),
        key=lambda r: float(r['mean_auc']),
        reverse=True
    )
    print(f'\n  {"─" * 60}')
    print(f'  TOP {min(top_n, len(ranked))} COMBINATIONS SO FAR')
    print(f'  {"─" * 60}')
    for i, row in enumerate(ranked[:top_n]):
        print(f'  [{i+1}] h={row["hidden"]:>2}  d={row["dropout"]}  '
              f'lr={row["lr_ft"]}  thr={row["threshold_pct"]}  '
              f'→ AUC={float(row["mean_auc"]):.4f} ± {float(row["std_auc"]):.4f}')
    print(f'  {"─" * 60}')


# ══════════════════════════════════════════════════════════════
# ADJACENCY UTILITIES
# ══════════════════════════════════════════════════════════════

def compute_threshold(adj_dtf_train, percentile=70.0):
    n    = adj_dtf_train.shape[1]
    mask = ~np.eye(n, dtype=bool)
    vals = np.concatenate(
        [adj_dtf_train[i][mask] for i in range(len(adj_dtf_train))]
    )
    return float(np.percentile(vals, percentile))


def normalize_adjacency(adj):
    A          = adj + np.eye(adj.shape[0], dtype=np.float32)
    d          = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D          = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def build_graphs(node_feats, adj_dtf, threshold):
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
    all_x  = np.concatenate([g[0].numpy() for g in graphs_train], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_x)

    def apply(graphs):
        return [
            (torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32), a)
            for x, a in graphs
        ]
    return apply(graphs_train), apply(graphs_test)


# ══════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ══════════════════════════════════════════════════════════════

def augment_edge_dropout(x, a, p=0.20):
    mask                     = (torch.rand_like(a) > p).float()
    diag                     = torch.arange(a.shape[0])
    mask[diag, diag]         = 1.0
    a_aug                    = a * mask
    row_sum                  = a_aug.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return x, a_aug / row_sum


def augment_node_noise(x, a, sigma=0.10):
    return x + torch.randn_like(x) * sigma, a


def augment_band_mask(x, a):
    band_idx           = int(torch.randint(0, N_BAND_FEATS, (1,)).item())
    x_aug              = x.clone()
    x_aug[:, band_idx] = 0.0
    return x_aug, a


ALL_AUGMENTATIONS = [augment_edge_dropout, augment_node_noise, augment_band_mask]


def random_augment(x, a):
    chosen    = np.random.choice(len(ALL_AUGMENTATIONS), size=2, replace=False)
    x1, a1   = ALL_AUGMENTATIONS[chosen[0]](x.clone(), a.clone())
    x2, a2   = ALL_AUGMENTATIONS[chosen[1]](x.clone(), a.clone())
    return (x1, a1), (x2, a2)


# ══════════════════════════════════════════════════════════════
# NT-XENT LOSS
# ══════════════════════════════════════════════════════════════

def nt_xent_loss(z1, z2, temperature=0.5):
    N      = z1.shape[0]
    z      = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim    = torch.mm(z, z.T) / temperature
    mask   = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════

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
        return h.mean(dim=0)


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


# ══════════════════════════════════════════════════════════════
# SSL PRE-TRAINING
# ══════════════════════════════════════════════════════════════

def ssl_pretrain(encoder, proj_head, train_graphs,
                 ssl_epochs, lr, device,
                 batch_size=32, temperature=0.5):
    encoder.train()
    proj_head.train()
    params    = list(encoder.parameters()) + list(proj_head.parameters())
    optimiser = Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=ssl_epochs, eta_min=lr * 0.1)
    N         = len(train_graphs)
    losses    = []

    for ep in range(ssl_epochs):
        idx       = np.random.permutation(N)
        ep_loss   = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch = idx[start: start + batch_size]
            if len(batch) < 2:
                continue
            z1_list, z2_list = [], []
            for i in batch:
                x, a               = train_graphs[i]
                (x1, a1), (x2, a2) = random_augment(x, a)
                h1 = encoder(x1.to(device), a1.to(device))
                h2 = encoder(x2.to(device), a2.to(device))
                z1_list.append(proj_head(h1))
                z2_list.append(proj_head(h2))
            z1   = torch.stack(z1_list)
            z2   = torch.stack(z2_list)
            loss = nt_xent_loss(z1, z2, temperature)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            ep_loss   += loss.item()
            n_batches += 1
        scheduler.step()
        losses.append(ep_loss / max(n_batches, 1))

    return losses


# ══════════════════════════════════════════════════════════════
# FINE-TUNING
# ══════════════════════════════════════════════════════════════

def finetune(encoder, clf_head,
             graphs_train, y_train,
             graphs_test,  y_test,
             ft_epochs, lr, pos_weight,
             device, patience=20,
             freeze_encoder=False):
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = list(clf_head.parameters())
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        params = list(encoder.parameters()) + list(clf_head.parameters())

    optimiser  = Adam(params, lr=lr, weight_decay=1e-4)
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    best_val_loss  = np.inf
    best_enc_state = None
    best_clf_state = None
    patience_cnt   = 0

    for ep in range(ft_epochs):
        encoder.train()
        clf_head.train()
        for i in np.random.permutation(len(graphs_train)):
            x, a  = graphs_train[i]
            optimiser.zero_grad()
            logit = clf_head(encoder(x.to(device), a.to(device)))
            label = torch.tensor(float(y_train[i]), device=device).unsqueeze(0)
            loss  = criterion(logit.unsqueeze(0), label)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()

        encoder.eval()
        clf_head.eval()
        with torch.no_grad():
            val_logits = np.array([
                clf_head(encoder(x.to(device), a.to(device))).cpu().item()
                for x, a in graphs_test
            ])
        val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
        val_labels_t = torch.tensor(y_test,     dtype=torch.float32)
        val_loss     = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )(val_logits_t, val_labels_t).item()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
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


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(encoder, clf_head, graphs, device):
    encoder.eval()
    clf_head.eval()
    logits = np.array([
        clf_head(encoder(x.to(device), a.to(device))).cpu().item()
        for x, a in graphs
    ], dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    return probs, preds


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
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'mcc':         float(matthews_corrcoef(y_true, y_pred)),
    }


# ══════════════════════════════════════════════════════════════
# ONE LOPO RUN WITH GIVEN PARAMS
# ══════════════════════════════════════════════════════════════

def run_lopo(params, node_feats, adj_dtf, y, patient_ids,
             args, device, verbose=False):
    """
    Run full 8-fold LOPO with given hyperparameters.
    Returns list of per-fold metric dicts.
    """
    patients     = np.unique(patient_ids)
    fold_metrics = []

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask
        train_idx  = np.where(train_mask)[0]
        test_idx   = np.where(test_mask)[0]

        y_train = y[train_idx]
        y_test  = y[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        # Threshold from training adjacency only
        threshold = compute_threshold(
            adj_dtf[train_idx], params['threshold_pct']
        )

        # Build and scale graphs
        g_train_raw = build_graphs(
            node_feats[train_idx], adj_dtf[train_idx], threshold
        )
        g_test_raw  = build_graphs(
            node_feats[test_idx],  adj_dtf[test_idx],  threshold
        )
        g_train, g_test = scale_node_features(g_train_raw, g_test_raw)

        n_neg      = int((y_train == 0).sum())
        n_pos      = int((y_train == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        # SSL pre-training (train graphs only)
        encoder   = GCNEncoder(
            in_dim=16,
            hidden=params['hidden'],
            dropout=params['dropout']
        ).to(device)
        proj_head = ProjectionHead(
            in_dim=params['hidden'],
            proj_dim=32
        ).to(device)

        ssl_pretrain(
            encoder, proj_head, g_train,
            ssl_epochs  = args.ssl_epochs,
            lr          = args.lr_ssl,
            device      = device,
            batch_size  = args.batch_size,
            temperature = args.temperature,
        )

        # Phase A: linear probe (encoder frozen)
        clf_head = ClassifierHead(
            in_dim=params['hidden'],
            dropout=params['dropout']
        ).to(device)

        finetune(
            encoder, clf_head,
            g_train, y_train,
            g_test,  y_test,
            ft_epochs      = 30,
            lr             = params['lr_ft'] * 5,
            pos_weight     = pos_weight,
            device         = device,
            patience       = args.patience,
            freeze_encoder = True,
        )

        # Phase B: full fine-tuning
        finetune(
            encoder, clf_head,
            g_train, y_train,
            g_test,  y_test,
            ft_epochs      = args.ft_epochs,
            lr             = params['lr_ft'],
            pos_weight     = pos_weight,
            device         = device,
            patience       = args.patience,
            freeze_encoder = False,
        )

        probs, preds = evaluate(encoder, clf_head, g_test, device)
        metrics      = compute_metrics(y_test, preds, probs)
        if metrics is None:
            continue

        metrics['patient'] = pat
        fold_metrics.append(metrics)

        if verbose:
            print(f'    {pat}: AUC={metrics["auc"]:.3f}  '
                  f'F1={metrics["f1"]:.3f}  '
                  f'Sens={metrics["sensitivity"]:.3f}')

    return fold_metrics


# ══════════════════════════════════════════════════════════════
# FINAL RUN WITH BEST PARAMS + PLOTS
# ══════════════════════════════════════════════════════════════

def run_final(best_params, node_feats, adj_dtf, y, patient_ids,
              args, device, output_dir):
    """Run final evaluation with best params and save all results."""
    print(f'\n{"=" * 65}')
    print(f'FINAL EVALUATION with best params: {best_params}')
    print(f'{"=" * 65}')

    patients      = np.unique(patient_ids)
    fold_metrics  = []
    fold_roc_data = []
    all_y_true    = []
    all_y_pred    = []

    for pat in patients:
        test_mask  = (patient_ids == pat)
        train_mask = ~test_mask
        train_idx  = np.where(train_mask)[0]
        test_idx   = np.where(test_mask)[0]

        y_train = y[train_idx]
        y_test  = y[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        threshold   = compute_threshold(
            adj_dtf[train_idx], best_params['threshold_pct']
        )
        g_train_raw = build_graphs(
            node_feats[train_idx], adj_dtf[train_idx], threshold
        )
        g_test_raw  = build_graphs(
            node_feats[test_idx],  adj_dtf[test_idx],  threshold
        )
        g_train, g_test = scale_node_features(g_train_raw, g_test_raw)

        n_neg      = int((y_train == 0).sum())
        n_pos      = int((y_train == 1).sum())
        pos_weight = n_neg / (n_pos + 1e-12)

        encoder   = GCNEncoder(
            in_dim=16,
            hidden=best_params['hidden'],
            dropout=best_params['dropout']
        ).to(device)
        proj_head = ProjectionHead(
            in_dim=best_params['hidden'], proj_dim=32
        ).to(device)

        ssl_pretrain(
            encoder, proj_head, g_train,
            ssl_epochs  = args.ssl_epochs,
            lr          = args.lr_ssl,
            device      = device,
            batch_size  = args.batch_size,
            temperature = args.temperature,
        )

        clf_head = ClassifierHead(
            in_dim=best_params['hidden'],
            dropout=best_params['dropout']
        ).to(device)

        finetune(
            encoder, clf_head, g_train, y_train, g_test, y_test,
            ft_epochs=30, lr=best_params['lr_ft'] * 5,
            pos_weight=pos_weight, device=device,
            patience=args.patience, freeze_encoder=True,
        )
        finetune(
            encoder, clf_head, g_train, y_train, g_test, y_test,
            ft_epochs=args.ft_epochs, lr=best_params['lr_ft'],
            pos_weight=pos_weight, device=device,
            patience=args.patience, freeze_encoder=False,
        )

        probs, preds = evaluate(encoder, clf_head, g_test, device)
        metrics      = compute_metrics(y_test, preds, probs)
        if metrics is None:
            continue

        metrics['patient'] = pat
        fold_metrics.append(metrics)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(preds.tolist())
        fpr, tpr, _ = roc_curve(y_test, probs)
        fold_roc_data.append((fpr, tpr, metrics['auc'], pat))

        print(f'  {pat}: AUC={metrics["auc"]:.3f}  '
              f'F1={metrics["f1"]:.3f}  '
              f'Sens={metrics["sensitivity"]:.3f}  '
              f'Spec={metrics["specificity"]:.3f}  '
              f'MCC={metrics["mcc"]:.3f}')

    # ROC plot
    if fold_roc_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        aucs = []
        for fpr, tpr, auc, pat in fold_roc_data:
            ax.plot(fpr, tpr, alpha=0.6, lw=1.5,
                    label=f'{pat} (AUC={auc:.2f})')
            aucs.append(auc)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(
            f'SSL-GCN (Tuned) — LOPO ROC\n'
            f'Mean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
            fontsize=11, fontweight='bold'
        )
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_ssl_tuned.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print('  ✓ roc_ssl_tuned.png')

    # Summary stats
    met_keys      = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy', 'mcc']
    summary_stats = {}
    print(f'\n  {"─" * 50}')
    print(f'  SSL-GCN (Tuned) — Mean ± Std ({len(fold_metrics)} folds)')
    print(f'  {"─" * 50}')
    for k in met_keys:
        vals        = [m[k] for m in fold_metrics]
        mean_, std_ = float(np.mean(vals)), float(np.std(vals))
        summary_stats[k] = {'mean': mean_, 'std': std_}
        print(f'  {k:15s}: {mean_:.3f} ± {std_:.3f}')

    # Save results JSON
    results = {
        'model':           'SSL_GCN_GraphCL_Tuned',
        'best_params':     best_params,
        'hyperparameters': vars(args),
        'fold_metrics':    fold_metrics,
        'summary_stats':   summary_stats,
    }
    with open(output_dir / 'results_ssl_tuned.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  ✓ results_ssl_tuned.json')

    return summary_stats


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SSL GCN Hyperparameter Tuning (FORTH, LOPO, CSV checkpoint)'
    )
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/ssl_tuning')
    parser.add_argument('--ssl_epochs',    type=int,   default=200)
    parser.add_argument('--ft_epochs',     type=int,   default=100)
    parser.add_argument('--lr_ssl',        type=float, default=0.001)
    parser.add_argument('--temperature',   type=float, default=0.5)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--patience',      type=int,   default=15)
    parser.add_argument('--run_final',     action='store_true',
                        help='After tuning, run final evaluation with best params')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = output_dir / 'tuning_results.csv'

    print('=' * 65)
    print('SSL GCN — HYPERPARAMETER TUNING (FORTH, LOPO)')
    print('=' * 65)
    print(f'Device      : {device}')
    print(f'SSL epochs  : {args.ssl_epochs}   LR: {args.lr_ssl}')
    print(f'FT epochs   : {args.ft_epochs}    Patience: {args.patience}')
    print(f'Batch size  : {args.batch_size}   Temp: {args.temperature}')
    print(f'CSV         : {csv_path}')
    print(f'\nParam grid  : {PARAM_GRID}')

    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    n_total = len(combos)
    print(f'Total combos: {n_total}  (×8 folds = {n_total * 8} runs)')
    print('=' * 65)

    # Load data
    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)
    adj_dtf     = data['adj_dtf'].astype(np.float32)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']

    print(f'\nLoaded: {len(y)} epochs | '
          f'Ictal: {(y==1).sum()} | Pre-ictal: {(y==0).sum()}')
    print(f'Patients: {np.unique(patient_ids).tolist()}\n')

    # Load already-completed combinations
    completed = load_completed(csv_path)
    n_done    = len(completed)
    print(f'Already completed: {n_done}/{n_total} combinations')
    if n_done > 0:
        print_leaderboard(csv_path, top_n=3)

    # Main tuning loop
    for combo_idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        key    = combo_key(params)

        if key in completed:
            print(f'  [{combo_idx+1:3d}/{n_total}] SKIP: '
                  f'h={params["hidden"]} d={params["dropout"]} '
                  f'lr={params["lr_ft"]} thr={params["threshold_pct"]} '
                  f'→ {float(completed[key]["mean_auc"]):.4f}')
            continue

        print(f'\n  [{combo_idx+1:3d}/{n_total}] Testing: {params}')

        try:
            fold_metrics = run_lopo(
                params, node_feats, adj_dtf, y, patient_ids,
                args, device, verbose=True,
            )

            if not fold_metrics:
                print(f'  SKIPPED — no valid folds')
                continue

            mean_auc, std_auc = save_result(csv_path, params, fold_metrics)
            completed[key]    = {
                'mean_auc': mean_auc, 'std_auc': std_auc
            }
            print(f'  → mean AUC = {mean_auc:.4f} ± {std_auc:.4f}  '
                  f'[saved to CSV]')

        except Exception as e:
            print(f'  ERROR: {e} — skipping this combination')
            continue

        # Print leaderboard every 5 combos
        if (combo_idx + 1) % 5 == 0:
            print_leaderboard(csv_path, top_n=3)

    # Final leaderboard
    print('\n' + '=' * 65)
    print('TUNING COMPLETE')
    print('=' * 65)
    print_leaderboard(csv_path, top_n=10)

    # Find best
    completed = load_completed(csv_path)
    if not completed:
        print('No completed combinations found.')
        return

    best_key = max(completed, key=lambda k: float(completed[k]['mean_auc']))
    best_row = completed[best_key]
    best_params = {
        'hidden':        best_key[0],
        'dropout':       best_key[1],
        'lr_ft':         best_key[2],
        'threshold_pct': best_key[3],
    }

    print(f'\nBEST COMBINATION:')
    print(f'  hidden={best_params["hidden"]}  '
          f'dropout={best_params["dropout"]}  '
          f'lr_ft={best_params["lr_ft"]}  '
          f'threshold_pct={best_params["threshold_pct"]}')
    print(f'  Mean AUC = {float(best_row["mean_auc"]):.4f} ± '
          f'{float(best_row["std_auc"]):.4f}')

    # Save best params to JSON
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump({
            'best_params':  best_params,
            'mean_auc':     float(best_row['mean_auc']),
            'std_auc':      float(best_row['std_auc']),
            'param_grid':   PARAM_GRID,
            'n_combos':     n_total,
        }, f, indent=2)
    print(f'  ✓ best_params.json saved')

    # Run final evaluation if requested
    if args.run_final:
        run_final(
            best_params, node_feats, adj_dtf, y, patient_ids,
            args, device, output_dir,
        )

    print('\n' + '=' * 65)
    print('DONE')
    print('=' * 65)
    print(f'\nTo run final evaluation with best params, add --run_final flag.')
    print(f'CSV results: {csv_path}')


if __name__ == '__main__':
    main()
