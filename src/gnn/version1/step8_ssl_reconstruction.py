"""
Step 8b — SSL via Masked Node Feature Reconstruction (v2)
==========================================================

Fix over v1: reconstruction is performed from NODE-LEVEL embeddings,
not from the pooled graph embedding. GlobalMeanPool destroys node
identity — broadcasting a single graph vector to all 19 nodes gives
the decoder identical input for every node, causing MSE to collapse
to the feature variance regardless of training.

Node-level reconstruction forces the GCN to produce discriminative
per-node embeddings that preserve individual channel information,
which is exactly what is needed for seizure detection.

Outputs:
    pretrain_encoder.pt
    pretrain_loss_curve.png
    finetuned_model.pt
    finetune_training_curves.png
    finetune_overfitting.png
    finetune_roc_curve.png
    finetune_confusion_matrix.png
    ssl_reconstruction_results.json

Usage:
    python step8_ssl_reconstruction.py \
        --dataset_dir F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/dataset_dtf \
        --output_dir  F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/results/ssl_recon_v2
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── model components ──────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """
    2-layer GCN. forward() returns graph-level pooled embedding by default.
    With return_node_emb=True also returns node-level embeddings for
    reconstruction pretraining.
    """
    def __init__(self, n_features=9, hidden=64, dropout=0.4):
        super().__init__()
        self.conv1   = GCNConv(n_features, hidden)
        self.bn1     = nn.BatchNorm1d(hidden)
        self.conv2   = GCNConv(hidden, hidden)
        self.bn2     = nn.BatchNorm1d(hidden)
        self.dropout = dropout

    def forward(self, data: Data, return_node_emb=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))),
                      p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))),
                      p=self.dropout, training=self.training)
        # x: (N_nodes, hidden) — unique embedding per node
        graph_emb = global_mean_pool(x, batch)   # (B, hidden)
        if return_node_emb:
            return graph_emb, x                  # both levels
        return graph_emb


class ReconstructionDecoder(nn.Module):
    """
    Decodes node-level embeddings back to node features.
    Input : (N_nodes, hidden) — one distinct vector per node
    Output: (N_nodes, n_features)
    Discarded after pretraining.
    """
    def __init__(self, hidden=64, n_features=9):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, n_features)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ClassifierHead(nn.Module):
    def __init__(self, hidden=64, dropout=0.4):
        super().__init__()
        self.fc1     = nn.Linear(hidden, 32)
        self.fc2     = nn.Linear(32, 2)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout,
                      training=self.training)
        return self.fc2(x)


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
    ax.set_title(
        'SSL Pretraining — Node-Level Masked Feature Reconstruction (MSE)',
        fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (masked positions only)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_finetune_curves(history, best_epoch, out_path):
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('SSL Reconstruction (v2) — Fine-tuning Curves',
                 fontsize=14, fontweight='bold')
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
    fig.suptitle('SSL Reconstruction — Fine-tuning Overfitting Diagnostic',
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
    ax.set_title('SSL Reconstruction — ROC Curve (Test Set)', fontweight='bold')
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
        f'SSL Reconstruction — Test Set Confusion Matrix\n'
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
    print('PHASE 1 — NODE-LEVEL MASKED FEATURE RECONSTRUCTION')
    print('=' * 70)

    preictal_graphs = [g for g in train_graphs if g.y.item() == 0]
    print(f'  Pre-ictal graphs : {len(preictal_graphs)}')
    print(f'  Mask probability : p_feat={args.p_feat}')
    print(f'  Reconstruction   : from NODE embeddings (not pooled graph embedding)')

    pretrain_loader = DataLoader(preictal_graphs,
                                 batch_size=args.pretrain_batch, shuffle=True)

    encoder = GCNEncoder(n_features=9, hidden=args.hidden,
                         dropout=args.dropout).to(device)
    decoder = ReconstructionDecoder(hidden=args.hidden, n_features=9).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.pretrain_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs, eta_min=1e-5)

    n_params = (sum(p.numel() for p in encoder.parameters()) +
                sum(p.numel() for p in decoder.parameters()))
    print(f'  Encoder + decoder params: {n_params:,}')
    print(f'\n{"Epoch":>6}  {"MSE Loss":>12}  {"Masked %":>10}')
    print('-' * 35)

    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        encoder.train(); decoder.train()
        epoch_loss   = 0.0
        total_masked = 0
        total_elems  = 0

        for batch in pretrain_loader:
            batch  = batch.to(device)
            x_orig = batch.x.clone()

            # mask p_feat fraction of feature values
            mask         = torch.rand_like(x_orig) < args.p_feat
            batch.x      = x_orig.clone()
            batch.x[mask] = 0.0

            # encode → node-level embeddings (N_nodes, hidden)
            _, node_emb = encoder(batch, return_node_emb=True)

            # decode from NODE embeddings — each node has a distinct vector
            x_recon = decoder(node_emb)   # (N_nodes, 9)

            if mask.sum() == 0:
                continue
            loss = F.mse_loss(x_recon[mask], x_orig[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()

            epoch_loss   += loss.item()
            total_masked += mask.sum().item()
            total_elems  += mask.numel()

        scheduler.step()
        avg_loss   = epoch_loss / len(pretrain_loader)
        masked_pct = 100 * total_masked / max(total_elems, 1)
        history.append({'epoch': epoch, 'loss': round(avg_loss, 6)})

        if epoch % 20 == 0 or epoch == 1:
            print(f'{epoch:>6}  {avg_loss:>12.6f}  {masked_pct:>9.1f}%')

    ckpt_path = output_dir / 'pretrain_encoder.pt'
    torch.save(encoder.state_dict(), ckpt_path)
    print(f'\n  Encoder saved: {ckpt_path.name}')
    print('  (Decoder discarded)')

    save_pretrain_curve(history, output_dir / 'pretrain_loss_curve.png')
    return encoder


# ── phase 2: fine-tuning ──────────────────────────────────────────────────────

def finetune(args, encoder, train_graphs, val_graphs, test_graphs,
             device, output_dir):
    print('\n' + '=' * 70)
    print('PHASE 2 — FINE-TUNING (frozen encoder + trainable classifier head)')
    print('=' * 70)

    for param in encoder.parameters():
        param.requires_grad = False
    print('  Encoder: FROZEN')

    head   = ClassifierHead(hidden=args.hidden, dropout=args.dropout).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(f'  Classifier head params: {n_head:,}')

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
        head.train(); encoder.eval()
        epoch_loss = 0.0
        all_logits_tr, all_labels_tr = [], []

        for batch in train_loader:
            batch = batch.to(device)
            with torch.no_grad():
                emb = encoder(batch)
            logits = head(emb)
            loss   = F.cross_entropy(logits, batch.y.squeeze())
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
    parser = argparse.ArgumentParser(
        description='SSL Node-Level Masked Reconstruction + Fine-tuning')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir',  required=True)

    parser.add_argument('--hidden',  type=int,   default=64)
    parser.add_argument('--dropout', type=float, default=0.4)

    parser.add_argument('--pretrain_epochs', type=int,   default=200)
    parser.add_argument('--pretrain_lr',     type=float, default=1e-3)
    parser.add_argument('--pretrain_batch',  type=int,   default=64)
    parser.add_argument('--p_feat',          type=float, default=0.3)

    parser.add_argument('--finetune_epochs',   type=int,   default=200)
    parser.add_argument('--finetune_lr',       type=float, default=1e-3)
    parser.add_argument('--finetune_batch',    type=int,   default=32)
    parser.add_argument('--finetune_patience', type=int,   default=50)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('STEP 8b v2 — SSL: Node-Level Masked Reconstruction + Fine-tuning')
    print('=' * 70)
    print(f'Device : {device}')
    print(f'Encoder: GCN  hidden={args.hidden}  dropout={args.dropout}')
    print(f'\nPhase 1: epochs={args.pretrain_epochs}  lr={args.pretrain_lr}  '
          f'batch={args.pretrain_batch}  p_feat={args.p_feat}')
    print(f'Phase 2: epochs={args.finetune_epochs}  lr={args.finetune_lr}  '
          f'patience={args.finetune_patience}')

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

    print('\nSaving outputs...')

    test_m_json = {k: v for k, v in test_m.items()
                   if k not in ('probs', 'labels_np')}

    results = {
        'experiment': {
            'script'     : 'step8_ssl_reconstruction.py',
            'timestamp'  : datetime.now().isoformat(timespec='seconds'),
            'model'      : 'SSL_NodeMaskedReconstruction_v2',
            'description': (
                'GCN encoder pretrained via node-level masked feature '
                'reconstruction on pre-ictal graphs only. MSE loss computed '
                'on masked node feature positions using per-node GCN embeddings '
                '(not pooled graph embedding). Encoder frozen during fine-tuning.'
            ),
        },
        'hyperparameters': {
            'hidden'           : args.hidden,
            'dropout'          : args.dropout,
            'pretrain_epochs'  : args.pretrain_epochs,
            'pretrain_lr'      : args.pretrain_lr,
            'pretrain_batch'   : args.pretrain_batch,
            'p_feat'           : args.p_feat,
            'finetune_epochs'  : args.finetune_epochs,
            'finetune_lr'      : args.finetune_lr,
            'finetune_batch'   : args.finetune_batch,
            'finetune_patience': args.finetune_patience,
            'class_weight'     : 'WeightedRandomSampler only',
            'seed'             : SEED,
        },
        'data': {
            'dataset_dir'    : str(args.dataset_dir),
            'pretrain_graphs': sum(1 for g in train_graphs if g.y.item()==0),
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
            'best_finetune_epoch'  : best_ft_epoch,
            'best_val_auroc'       : max(h['val_auroc'] for h in ft_history),
            'total_finetune_epochs': len(ft_history),
            'stopped_early'        : len(ft_history) < args.finetune_epochs,
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

    with open(output_dir / 'ssl_reconstruction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('  Saved: ssl_reconstruction_results.json')

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
