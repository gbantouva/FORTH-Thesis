"""
Step 8 — GraphCL: Self-Supervised Pretraining + Fine-tuning
=============================================================

MOTIVATION:
    All supervised models peaked at epoch 1-5 before overfitting to
    patient-specific patterns. The core problem: only 196 labeled ictal
    epochs across 8 patients is not enough for patient-independent
    supervised learning.

    Solution: pretrain the encoder WITHOUT labels on pre-ictal graphs only,
    then fine-tune with the small labeled set.

HOW IT WORKS:
    Phase 1 — Contrastive Pretraining (no labels, pre-ictal only):
        For each pre-ictal graph g, create two augmented views g1, g2.
        Train encoder so that representations of g1 and g2 are similar,
        while representations of different graphs are dissimilar.
        Loss: NT-Xent (normalized temperature-scaled cross entropy)

    Phase 2 — Fine-tuning (frozen encoder + classifier head):
        Freeze the pretrained encoder, add a small classification head,
        train only the head on the labeled data.

AUGMENTATIONS:
    View 1: edge dropping   — randomly zero out p_edge fraction of edges
    View 2: feature masking — randomly zero out p_feat fraction of node features

Outputs:
    pretrain_encoder.pt
    pretrain_loss_curve.png
    finetuned_model.pt
    finetune_training_curves.png
    finetune_overfitting.png
    finetune_roc_curve.png
    finetune_confusion_matrix.png
    graphcl_results.json

Usage:
    python step8_graphcl.py \
        --dataset_dir F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/dataset_dtf \
        --output_dir  F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/results/graphcl
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── encoder ───────────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """
    2-layer GCN backbone — identical architecture to supervised GCN baseline
    so that any performance difference is due to training strategy, not
    model capacity.
    """
    def __init__(self, n_features=9, hidden=64, dropout=0.4):
        super().__init__()
        self.conv1   = GCNConv(n_features, hidden)
        self.bn1     = nn.BatchNorm1d(hidden)
        self.conv2   = GCNConv(hidden, hidden)
        self.bn2     = nn.BatchNorm1d(hidden)
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))),
                      p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))),
                      p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)   # (batch_size, hidden)


class ProjectionHead(nn.Module):
    """
    MLP projector for contrastive pretraining (SimCLR-style).
    Discarded after pretraining — not used during fine-tuning.
    """
    def __init__(self, hidden=64, proj_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, proj_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ClassifierHead(nn.Module):
    """Small MLP classification head added on top of the frozen encoder."""
    def __init__(self, hidden=64, dropout=0.4):
        super().__init__()
        self.fc1     = nn.Linear(hidden, 32)
        self.fc2     = nn.Linear(32, 2)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout, training=self.training)
        return self.fc2(x)


# ── augmentations ─────────────────────────────────────────────────────────────

def augment_edge_drop(data: Data, p: float = 0.2) -> Data:
    edge_index = data.edge_index
    edge_attr  = data.edge_attr
    n_edges    = edge_index.size(1)
    keep_mask  = torch.rand(n_edges) > p
    return Data(
        x          = data.x,
        edge_index = edge_index[:, keep_mask],
        edge_attr  = edge_attr[keep_mask] if edge_attr is not None else None,
        y          = data.y,
    )


def augment_feature_mask(data: Data, p: float = 0.2) -> Data:
    x_aug = data.x.clone()
    x_aug[torch.rand_like(x_aug) < p] = 0.0
    return Data(
        x          = x_aug,
        edge_index = data.edge_index,
        edge_attr  = data.edge_attr,
        y          = data.y,
    )


def make_two_views(graphs: list, p_edge: float, p_feat: float):
    view1 = Batch.from_data_list([augment_edge_drop(g,    p=p_edge) for g in graphs])
    view2 = Batch.from_data_list([augment_feature_mask(g, p=p_feat) for g in graphs])
    return view1, view2


# ── NT-Xent loss ──────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    N  = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z  = torch.cat([z1, z2], dim=0)                          # (2N, proj_dim)
    sim = torch.mm(z, z.t()) / temperature                    # (2N, 2N)
    sim.masked_fill_(torch.eye(2*N, dtype=torch.bool,
                               device=z.device), float('-inf'))
    labels = torch.cat([torch.arange(N, 2*N, device=z.device),
                        torch.arange(0, N,   device=z.device)])
    return F.cross_entropy(sim, labels)


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(encoder, head, loader, device):
    encoder.eval(); head.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        all_logits.append(head(encoder(batch)).cpu())
        all_labels.append(batch.y.cpu())

    logits    = torch.cat(all_logits)
    labels    = torch.cat(all_labels)
    labels_np = labels.numpy()
    probs     = F.softmax(logits, dim=1)[:, 1].numpy()
    preds     = logits.argmax(dim=1).numpy()

    try:    auroc = roc_auc_score(labels_np, probs)
    except: auroc = float('nan')

    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'loss'       : round(F.cross_entropy(logits, labels).item(), 4),
        'auroc'      : round(float(auroc), 4),
        'sensitivity': round(tp / max(tp+fn, 1), 4),
        'specificity': round(tn / max(tn+fp, 1), 4),
        'f1_ictal'   : round(float(f1_score(labels_np, preds,
                                            pos_label=1, zero_division=0)), 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'probs'    : probs,
        'labels_np': labels_np,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def save_pretrain_curve(history, out_path):
    epochs = [h['epoch'] for h in history]
    losses = [h['loss']  for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color='steelblue', linewidth=1.5)
    ax.set_title('GraphCL Pretraining — NT-Xent Contrastive Loss',
                 fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('NT-Xent Loss')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_finetune_curves(history, best_epoch, out_path):
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('GraphCL Fine-tuning Curves', fontsize=14, fontweight='bold')
    for ax, key, title, ylabel, color in [
        (axes[0,0], 'train_loss',      'Training Loss',          'Loss',        'steelblue'),
        (axes[0,1], 'val_auroc',       'Validation AUROC',       'AUROC',       'darkorange'),
        (axes[1,0], 'val_sensitivity', 'Validation Sensitivity', 'Sensitivity', 'seagreen'),
        (axes[1,1], 'val_specificity', 'Validation Specificity', 'Specificity', 'mediumpurple'),
    ]:
        ax.plot(epochs, [h[key] for h in history], color=color, linewidth=1.5)
        ax.axvline(best_epoch, color='red', linestyle='--',
                   linewidth=1.2, label=f'Best epoch {best_epoch}')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_overfitting_curves(history, best_epoch, out_path):
    epochs      = [h['epoch']       for h in history]
    train_loss  = [h['train_loss']  for h in history]
    val_loss    = [h['val_loss']    for h in history]
    train_auroc = [h['train_auroc'] for h in history]
    val_auroc   = [h['val_auroc']   for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('GraphCL Fine-tuning — Overfitting Diagnostic',
                 fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(epochs, train_loss, color='steelblue',  lw=2, label='Train loss')
    ax.plot(epochs, val_loss,   color='darkorange', lw=2, label='Val loss')
    ax.axvline(best_epoch, color='red', linestyle='--', lw=1.2,
               label=f'Best epoch {best_epoch}')
    ax.axvspan(best_epoch, max(epochs), alpha=0.06, color='red')
    ax.set_title('Loss: Train vs Validation', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, train_auroc, color='steelblue',  lw=2, label='Train AUROC')
    ax.plot(epochs, val_auroc,   color='darkorange', lw=2, label='Val AUROC')
    ax.axvline(best_epoch, color='red', linestyle='--', lw=1.2,
               label=f'Best epoch {best_epoch}')
    ax.axvspan(best_epoch, max(epochs), alpha=0.06, color='red')
    ax.set_title('AUROC: Train vs Validation', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('AUROC')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_roc_curve(labels_np, probs, metrics, out_path):
    fpr, tpr, _ = roc_curve(labels_np, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='seagreen', lw=2,
            label=f'AUROC = {metrics["auroc"]:.4f}')
    ax.plot([0,1],[0,1], color='grey', linestyle='--', lw=1,
            label='Random classifier')
    ax.scatter(1 - metrics['specificity'], metrics['sensitivity'],
               color='red', zorder=5, s=80,
               label=f'Sens={metrics["sensitivity"]:.3f}  '
                     f'Spec={metrics["specificity"]:.3f}')
    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)',       fontsize=12)
    ax.set_title('GraphCL — ROC Curve (Test Set)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_confusion_matrix(metrics, out_path):
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Greens', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    classes = ['Pre-ictal (0)', 'Ictal (1)']
    ax.set_xticks([0,1]); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([0,1]); ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label',      fontsize=12)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i,j] / max(cm[i].sum(), 1)
            ax.text(j, i, f'{cm[i,j]}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if cm[i,j] > thresh else 'black')
    ax.set_title(
        f'GraphCL — Test Set Confusion Matrix\n'
        f'AUROC={metrics["auroc"]:.4f}  Sens={metrics["sensitivity"]:.4f}  '
        f'Spec={metrics["specificity"]:.4f}  F1={metrics["f1_ictal"]:.4f}',
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ── phase 1: pretraining ──────────────────────────────────────────────────────

def pretrain(args, train_graphs, device, output_dir):
    print('\n' + '=' * 70)
    print('PHASE 1 — CONTRASTIVE PRETRAINING (pre-ictal only, no labels)')
    print('=' * 70)

    preictal_graphs = [g for g in train_graphs if g.y.item() == 0]
    print(f'  Pre-ictal graphs: {len(preictal_graphs)}  '
          f'(ictal graphs excluded from pretraining)')
    print(f'  Augmentations: edge_drop(p={args.p_edge}) + feat_mask(p={args.p_feat})')
    print(f'  Temperature: {args.temperature}  |  Projection dim: {args.proj_dim}')

    pretrain_loader = DataLoader(preictal_graphs, batch_size=args.pretrain_batch,
                                 shuffle=True)

    encoder   = GCNEncoder(n_features=9, hidden=args.hidden,
                           dropout=args.dropout).to(device)
    proj_head = ProjectionHead(hidden=args.hidden,
                               proj_dim=args.proj_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.pretrain_lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs, eta_min=1e-5)

    n_params = (sum(p.numel() for p in encoder.parameters()) +
                sum(p.numel() for p in proj_head.parameters()))
    print(f'  Encoder + projection head params: {n_params:,}')
    print(f'\n{"Epoch":>6}  {"NT-Xent Loss":>14}')
    print('-' * 25)

    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        encoder.train(); proj_head.train()
        epoch_loss = 0.0

        for batch in pretrain_loader:
            graphs_list = batch.to_data_list()
            v1, v2 = make_two_views(graphs_list, args.p_edge, args.p_feat)
            v1, v2 = v1.to(device), v2.to(device)

            z1 = proj_head(encoder(v1))
            z2 = proj_head(encoder(v2))
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(proj_head.parameters()), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(pretrain_loader)
        history.append({'epoch': epoch, 'loss': round(avg_loss, 4)})

        if epoch % 20 == 0 or epoch == 1:
            print(f'{epoch:>6}  {avg_loss:>14.4f}')

    ckpt_path = output_dir / 'pretrain_encoder.pt'
    torch.save(encoder.state_dict(), ckpt_path)
    print(f'\n  Pretrained encoder saved: {ckpt_path.name}')
    print('  (Projection head discarded)')

    save_pretrain_curve(history, output_dir / 'pretrain_loss_curve.png')
    return encoder


# ── phase 2: fine-tuning ──────────────────────────────────────────────────────

def finetune(args, encoder, train_graphs, val_graphs, test_graphs,
             device, output_dir):
    print('\n' + '=' * 70)
    print('PHASE 2 — FINE-TUNING (frozen encoder + trainable classifier head)')
    print('=' * 70)

    # freeze encoder — representations are fixed, only head is trained
    for param in encoder.parameters():
        param.requires_grad = False
    print('  Encoder: FROZEN')

    head = ClassifierHead(hidden=args.hidden, dropout=args.dropout).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(f'  Classifier head params: {n_head:,}')

    # imbalance correction — sampler only, no loss weights
    labels      = [g.y.item() for g in train_graphs]
    class_count = [labels.count(0), labels.count(1)]
    weights     = [1.0 / class_count[l] for l in labels]
    sampler     = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
    print(f'  Sampler weights — pre-ictal: {weights[0]:.5f}  '
          f'ictal: {weights[labels.index(1)]:.5f}')

    train_loader = DataLoader(train_graphs, batch_size=args.finetune_batch,
                              sampler=sampler)
    val_loader   = DataLoader(val_graphs,   batch_size=args.finetune_batch,
                              shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=args.finetune_batch,
                              shuffle=False)

    optimizer = torch.optim.Adam(head.parameters(),
                                 lr=args.finetune_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=1e-5)

    print(f'\n{"Epoch":>6}  {"TrainLoss":>10}  {"ValLoss":>8}  '
          f'{"ValAUROC":>9}  {"ValSens":>8}  {"ValSpec":>8}  {"ValF1":>7}')
    print('-' * 70)

    best_val_auroc   = -1.0
    best_epoch       = 0
    patience_counter = 0
    history          = []
    ckpt_path        = output_dir / 'finetuned_model.pt'

    for epoch in range(1, args.finetune_epochs + 1):
        head.train(); encoder.eval()   # encoder stays in eval mode (frozen)
        epoch_loss = 0.0
        all_logits_tr, all_labels_tr = [], []

        for batch in train_loader:
            batch = batch.to(device)
            with torch.no_grad():
                emb = encoder(batch)   # frozen — no gradients needed
            logits = head(emb)
            loss   = F.cross_entropy(logits, batch.y.squeeze())  # no loss weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_logits_tr.append(logits.detach().cpu())
            all_labels_tr.append(batch.y.cpu())

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        tr_logits = torch.cat(all_logits_tr)
        tr_labels = torch.cat(all_labels_tr).numpy()
        tr_probs  = F.softmax(tr_logits, dim=1)[:, 1].numpy()
        try:    train_auroc = roc_auc_score(tr_labels, tr_probs)
        except: train_auroc = float('nan')

        val_m = evaluate(encoder, head, val_loader, device)

        history.append({
            'epoch'          : epoch,
            'train_loss'     : round(avg_loss, 4),
            'val_loss'       : val_m['loss'],
            'train_auroc'    : round(float(train_auroc), 4),
            'val_auroc'      : val_m['auroc'],
            'val_sensitivity': val_m['sensitivity'],
            'val_specificity': val_m['specificity'],
            'val_f1_ictal'   : val_m['f1_ictal'],
        })

        if val_m['auroc'] > best_val_auroc:
            best_val_auroc   = val_m['auroc']
            best_epoch       = epoch
            patience_counter = 0
            torch.save({'encoder': encoder.state_dict(),
                        'head':    head.state_dict()}, ckpt_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'{epoch:>6}  {avg_loss:>10.4f}  '
                  f'{val_m["loss"]:>8.4f}  '
                  f'{val_m["auroc"]:>9.4f}  '
                  f'{val_m["sensitivity"]:>8.4f}  '
                  f'{val_m["specificity"]:>8.4f}  '
                  f'{val_m["f1_ictal"]:>7.4f}')

        if patience_counter >= args.finetune_patience:
            print(f'\nEarly stop at epoch {epoch}  '
                  f'(best val AUROC={best_val_auroc:.4f} at epoch {best_epoch})')
            break

    # ── test ──────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f'TEST EVALUATION  (checkpoint from fine-tune epoch {best_epoch})')
    print('=' * 70)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    head.load_state_dict(ckpt['head'])
    test_m = evaluate(encoder, head, test_loader, device)

    print(f'  AUROC       : {test_m["auroc"]:.4f}')
    print(f'  Sensitivity : {test_m["sensitivity"]:.4f}  '
          f'— caught {test_m["tp"]} / {test_m["tp"]+test_m["fn"]} ictal epochs')
    print(f'  Specificity : {test_m["specificity"]:.4f}  '
          f'— correct on {test_m["tn"]} / {test_m["tn"]+test_m["fp"]} pre-ictal')
    print(f'  F1-ictal    : {test_m["f1_ictal"]:.4f}')
    print(f'\n  Confusion matrix (test set):')
    print(f'                  Predicted')
    print(f'                  Pre-ictal   Ictal')
    print(f'  Actual Pre-ictal  [ {test_m["tn"]:4d}      {test_m["fp"]:4d} ]')
    print(f'  Actual Ictal      [ {test_m["fn"]:4d}      {test_m["tp"]:4d} ]')

    return test_m, history, best_epoch


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GraphCL SSL + fine-tuning')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir',  required=True)

    # shared
    parser.add_argument('--hidden',  type=int,   default=64)
    parser.add_argument('--dropout', type=float, default=0.4)

    # pretraining
    parser.add_argument('--pretrain_epochs', type=int,   default=100)
    parser.add_argument('--pretrain_lr',     type=float, default=1e-3)
    parser.add_argument('--pretrain_batch',  type=int,   default=32)
    parser.add_argument('--temperature',     type=float, default=0.5)
    parser.add_argument('--proj_dim',        type=int,   default=32)
    parser.add_argument('--p_edge',          type=float, default=0.2)
    parser.add_argument('--p_feat',          type=float, default=0.2)

    # fine-tuning
    parser.add_argument('--finetune_epochs',   type=int,   default=200)
    parser.add_argument('--finetune_lr',       type=float, default=2e-4)
    parser.add_argument('--finetune_batch',    type=int,   default=32)
    parser.add_argument('--finetune_patience', type=int,   default=50)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('STEP 8 — GraphCL: Self-Supervised Pretraining + Fine-tuning')
    print('=' * 70)
    print(f'Device : {device}')
    print(f'Encoder: GCN  hidden={args.hidden}  dropout={args.dropout}')

    print('\nLoading dataset...')
    train_graphs = torch.load(dataset_dir / 'train_graphs.pt', weights_only=False)
    val_graphs   = torch.load(dataset_dir / 'val_graphs.pt',   weights_only=False)
    test_graphs  = torch.load(dataset_dir / 'test_graphs.pt',  weights_only=False)
    print(f'  Train: {len(train_graphs):5d}  '
          f'(ictal={sum(1 for g in train_graphs if g.y.item()==1)})')
    print(f'  Val  : {len(val_graphs):5d}  '
          f'(ictal={sum(1 for g in val_graphs if g.y.item()==1)})')
    print(f'  Test : {len(test_graphs):5d}  '
          f'(ictal={sum(1 for g in test_graphs if g.y.item()==1)})')

    encoder = pretrain(args, train_graphs, device, output_dir)

    test_m, ft_history, best_ft_epoch = finetune(
        args, encoder, train_graphs, val_graphs, test_graphs,
        device, output_dir
    )

    # ── save ──────────────────────────────────────────────────────────────────
    print('\nSaving outputs...')

    test_m_json = {k: v for k, v in test_m.items()
                   if k not in ('probs', 'labels_np')}

    results = {
        'experiment': {
            'script'     : 'step8_graphcl.py',
            'timestamp'  : datetime.now().isoformat(timespec='seconds'),
            'model'      : 'GraphCL',
            'description': (
                'Self-supervised contrastive pretraining on pre-ictal graphs '
                'only (NT-Xent loss), followed by fine-tuning with frozen '
                'encoder and small classifier head.'
            ),
        },
        'hyperparameters': {
            'hidden'           : args.hidden,
            'dropout'          : args.dropout,
            'pretrain_epochs'  : args.pretrain_epochs,
            'pretrain_lr'      : args.pretrain_lr,
            'pretrain_batch'   : args.pretrain_batch,
            'temperature'      : args.temperature,
            'proj_dim'         : args.proj_dim,
            'p_edge'           : args.p_edge,
            'p_feat'           : args.p_feat,
            'finetune_epochs'  : args.finetune_epochs,
            'finetune_lr'      : args.finetune_lr,
            'finetune_batch'   : args.finetune_batch,
            'finetune_patience': args.finetune_patience,
            'class_weight'     : 'WeightedRandomSampler only',
            'seed'             : SEED,
        },
        'data': {
            'dataset_dir'      : str(args.dataset_dir),
            'pretrain_graphs'  : sum(1 for g in train_graphs if g.y.item()==0),
            'split_sizes': {
                'train': {
                    'total'   : len(train_graphs),
                    'ictal'   : sum(1 for g in train_graphs if g.y.item()==1),
                    'preictal': sum(1 for g in train_graphs if g.y.item()==0),
                },
                'val': {
                    'total'   : len(val_graphs),
                    'ictal'   : sum(1 for g in val_graphs if g.y.item()==1),
                    'preictal': sum(1 for g in val_graphs if g.y.item()==0),
                },
                'test': {
                    'total'   : len(test_graphs),
                    'ictal'   : sum(1 for g in test_graphs if g.y.item()==1),
                    'preictal': sum(1 for g in test_graphs if g.y.item()==0),
                },
            },
        },
        'training': {
            'best_finetune_epoch': best_ft_epoch,
            'best_val_auroc'     : max(h['val_auroc'] for h in ft_history),
            'total_finetune_epochs': len(ft_history),
            'stopped_early'      : len(ft_history) < args.finetune_epochs,
        },
        'results': {
            'primary_metric': 'auroc',
            'test'          : test_m_json,
            'confusion_matrix_test': {
                'layout': '[[TN, FP], [FN, TP]]',
                'matrix': [[test_m['tn'], test_m['fp']],
                           [test_m['fn'], test_m['tp']]],
            },
        },
    }

    with open(output_dir / 'graphcl_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('  Saved: graphcl_results.json')

    np.save(output_dir / 'finetune_history.npy', np.array(ft_history))
    print('  Saved: finetune_history.npy')

    save_finetune_curves(ft_history, best_ft_epoch,
                         output_dir / 'finetune_training_curves.png')
    save_overfitting_curves(ft_history, best_ft_epoch,
                            output_dir / 'finetune_overfitting.png')
    save_roc_curve(test_m['labels_np'], test_m['probs'], test_m_json,
                   output_dir / 'finetune_roc_curve.png')
    save_confusion_matrix(test_m_json,
                          output_dir / 'finetune_confusion_matrix.png')

    print(f'\nAll outputs saved to: {output_dir}')


if __name__ == '__main__':
    main()
