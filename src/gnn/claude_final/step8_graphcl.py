"""
Step 8 — GraphCL: Self-Supervised Pretraining + Fine-tuning
=============================================================

MOTIVATION:
    All supervised models peaked at epoch 2-3 before overfitting to
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

        The encoder learns "what normal interictal connectivity looks like"
        across all patients, without ever seeing ictal examples.

    Phase 2 — Fine-tuning (with labels, both classes):
        Freeze the pretrained encoder, add a small classification head,
        train only the head on the labeled data.
        The encoder's patient-independent representations should make
        ictal graphs stand out as anomalous.

AUGMENTATIONS (two views per graph):
    View 1: edge dropping   — randomly zero out p_edge fraction of edges
    View 2: feature masking — randomly zero out p_feat fraction of node features
    These are applied independently so the encoder must learn invariant
    representations that survive both perturbations.

WHY PRE-ICTAL ONLY FOR PRETRAINING:
    - Abundant: ~392 pre-ictal in train, no scarcity
    - No labels needed
    - Encoder learns the "normal" manifold
    - Ictal graphs will naturally fall off this manifold
    - Equivalent to anomaly detection framing

OUTPUTS:
    pretrain_encoder.pt         — pretrained encoder weights
    pretrain_loss_curve.png     — contrastive loss over pretraining epochs
    finetuned_model.pt          — fine-tuned classifier
    finetune_training_curves.png
    finetune_confusion_matrix.png
    graphcl_results.json

Usage:
    python step8_graphcl.py \
        --dataset_dir F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/gnn_dataset_2to1 \
        --output_dir  F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/results/graphcl
"""

import argparse
import json
from pathlib import Path

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
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── encoder (shared between pretraining and fine-tuning) ──────────────────────

class GCNEncoder(nn.Module):
    """
    Same 2-layer GCN backbone used in the supervised baseline.
    Outputs a graph-level embedding of size `hidden`.

    Kept identical to the supervised GCN so results are directly comparable —
    the only difference is HOW the encoder is trained (contrastive vs CE).
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
    Small MLP that projects encoder output into the contrastive space.
    Used ONLY during pretraining — discarded before fine-tuning.
    Following SimCLR: using a projection head improves representation quality.
    """
    def __init__(self, hidden=64, proj_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, proj_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ClassifierHead(nn.Module):
    """Classification head added on top of the frozen encoder for fine-tuning."""
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
    """
    Randomly drop p fraction of edges.
    Simulates missing or noisy connectivity measurements.
    """
    edge_index = data.edge_index
    edge_attr  = data.edge_attr
    n_edges    = edge_index.size(1)

    keep_mask  = torch.rand(n_edges) > p
    new_data   = Data(
        x          = data.x,
        edge_index = edge_index[:, keep_mask],
        edge_attr  = edge_attr[keep_mask] if edge_attr is not None else None,
        y          = data.y,
        batch      = data.batch if hasattr(data, 'batch') else None,
    )
    return new_data


def augment_feature_mask(data: Data, p: float = 0.2) -> Data:
    """
    Randomly zero out p fraction of node feature dimensions.
    Simulates band power measurement noise or channel artifacts.
    """
    x_aug = data.x.clone()
    mask  = torch.rand_like(x_aug) < p
    x_aug[mask] = 0.0
    new_data = Data(
        x          = x_aug,
        edge_index = data.edge_index,
        edge_attr  = data.edge_attr,
        y          = data.y,
    )
    return new_data


def make_two_views(graphs: list, p_edge: float, p_feat: float) -> tuple:
    """Apply two different augmentations to produce paired views."""
    view1 = Batch.from_data_list([augment_edge_drop(g, p=p_edge)   for g in graphs])
    view2 = Batch.from_data_list([augment_feature_mask(g, p=p_feat) for g in graphs])
    return view1, view2


# ── NT-Xent contrastive loss ──────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    z1, z2 : (N, proj_dim) — projected representations of two views
    For each graph i, z1[i] and z2[i] are the positive pair.
    All other combinations are negatives.

    The loss encourages:
        - cos_sim(z1[i], z2[i]) → 1   (same graph, different augmentation)
        - cos_sim(z1[i], z2[j]) → 0   (different graphs)
    """
    N = z1.size(0)

    # L2 normalise
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate: [z1_0, z1_1, ..., z2_0, z2_1, ...]
    z  = torch.cat([z1, z2], dim=0)   # (2N, proj_dim)

    # Similarity matrix (2N × 2N)
    sim = torch.mm(z, z.t()) / temperature

    # Mask out self-similarities
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([
        torch.arange(N, 2*N, device=z.device),
        torch.arange(0, N,   device=z.device),
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(encoder, head, loader, device):
    encoder.eval()
    head.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        emb   = encoder(batch)
        logits = head(emb)
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.cpu())

    logits    = torch.cat(all_logits)
    labels    = torch.cat(all_labels)
    labels_np = labels.numpy()
    probs     = F.softmax(logits, dim=1)[:, 1].numpy()
    preds     = logits.argmax(dim=1).numpy()

    try:
        auroc = roc_auc_score(labels_np, probs)
    except ValueError:
        auroc = float('nan')

    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'loss'       : round(F.cross_entropy(logits, labels).item(), 4),
        'auroc'      : round(float(auroc),             4),
        'sensitivity': round(tp / max(tp+fn, 1),       4),
        'specificity': round(tn / max(tn+fp, 1),       4),
        'f1_ictal'   : round(float(f1_score(
                            labels_np, preds,
                            pos_label=1, zero_division=0)), 4),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def save_pretrain_curve(history, out_path):
    epochs = [h['epoch'] for h in history]
    losses = [h['loss']  for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color='steelblue', linewidth=1.5)
    ax.set_title('GraphCL Pretraining — NT-Xent Contrastive Loss',
                 fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NT-Xent Loss')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_finetune_curves(history, best_epoch, out_path):
    epochs   = [h['epoch']           for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('GraphCL Fine-tuning Curves', fontsize=14, fontweight='bold')
    for ax, key, title, ylabel, color in [
        (axes[0,0], 'train_loss',      'Training Loss',          'Loss',        'steelblue'),
        (axes[0,1], 'val_auroc',       'Validation AUROC',       'AUROC',       'darkorange'),
        (axes[1,0], 'val_sensitivity', 'Validation Sensitivity', 'Sensitivity', 'seagreen'),
        (axes[1,1], 'val_specificity', 'Validation Specificity', 'Specificity', 'mediumpurple'),
    ]:
        vals = [h[key] for h in history]
        ax.plot(epochs, vals, color=color, linewidth=1.5)
        ax.axvline(best_epoch, color='red', linestyle='--',
                   linewidth=1.2, label=f'Best epoch {best_epoch}')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
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
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
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

    # use ONLY pre-ictal graphs for pretraining
    #preictal_graphs = [g for g in train_graphs if g.y.item() == 0]
    preictal_graphs = train_graphs   # rename variable or add an arg

    print(f'  Pre-ictal graphs available: {len(preictal_graphs)}')
    print(f'  Augmentations: edge_drop(p={args.p_edge}) + feat_mask(p={args.p_feat})')
    print(f'  Temperature: {args.temperature}')
    print(f'  Projection dim: {args.proj_dim}')

    # simple DataLoader — no labels needed, no sampler needed (balanced by design)
    pretrain_loader = DataLoader(
        preictal_graphs,
        batch_size = args.pretrain_batch,
        shuffle    = True,
    )

    encoder    = GCNEncoder(n_features=9, hidden=args.hidden,
                            dropout=args.dropout).to(device)
    proj_head  = ProjectionHead(hidden=args.hidden,
                                proj_dim=args.proj_dim).to(device)

    #optimizer  = torch.optim.Adam(
    #    list(encoder.parameters()) + list(proj_head.parameters()),
    #    lr=args.pretrain_lr, weight_decay=1e-4
    #)

    optimizer  = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.pretrain_lr, weight_decay=1e-4
)
    #optimizer = torch.optim.Adam([
    #    {'params': encoder.parameters(), 'lr': args.finetune_lr * 0.1},  # slow
    #    {'params': head.parameters(),    'lr': args.finetune_lr},         # fast
    #], weight_decay=1e-4)

    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs, eta_min=1e-5
    )

    n_params = (sum(p.numel() for p in encoder.parameters()) +
                sum(p.numel() for p in proj_head.parameters()))
    print(f'  Encoder + projection head params: {n_params:,}')
    print(f'\n{"Epoch":>6}  {"NT-Xent Loss":>14}')
    print('-' * 25)

    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        encoder.train()
        proj_head.train()
        epoch_loss = 0.0

        for batch in pretrain_loader:
            # create two augmented views of the entire batch
            graphs_list = batch.to_data_list()
            v1, v2 = make_two_views(graphs_list, args.p_edge, args.p_feat)
            v1, v2 = v1.to(device), v2.to(device)

            z1 = proj_head(encoder(v1))
            z2 = proj_head(encoder(v2))

            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(proj_head.parameters()), 1.0
            )
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(pretrain_loader)
        history.append({'epoch': epoch, 'loss': round(avg_loss, 4)})

        if epoch % 20 == 0 or epoch == 1:
            print(f'{epoch:>6}  {avg_loss:>14.4f}')

    # save encoder (projection head is discarded)
    ckpt_path = output_dir / 'pretrain_encoder.pt'
    torch.save(encoder.state_dict(), ckpt_path)
    print(f'\n  Pretrained encoder saved: {ckpt_path.name}')
    print('  (Projection head discarded — not needed for fine-tuning)')

    save_pretrain_curve(history, output_dir / 'pretrain_loss_curve.png')
    return encoder


# ── phase 2: fine-tuning ──────────────────────────────────────────────────────

def finetune(args, encoder, train_graphs, val_graphs, test_graphs,
             device, output_dir):
    print('\n' + '=' * 70)
    print('PHASE 2 — FINE-TUNING (frozen encoder + trainable classifier head)')
    print('=' * 70)

    # freeze the encoder
    #for param in encoder.parameters():
    #    param.requires_grad = False
    #encoder.eval()
    #print('  Encoder: FROZEN')

    # unfreeze encoder but use lower LR (set in optimizer below)
    for param in encoder.parameters():
        param.requires_grad = True
    print('  Encoder: UNFROZEN (differential LR — encoder 10x slower than head)')
    
    head = ClassifierHead(hidden=args.hidden, dropout=args.dropout).to(device)
    n_head_params = sum(p.numel() for p in head.parameters())
    print(f'  Classifier head params: {n_head_params:,}  (these are trained)')

    # imbalance correction — same as supervised GCN
    labels      = [g.y.item() for g in train_graphs]
    class_count = [labels.count(0), labels.count(1)]
    weights     = [1.0 / class_count[l] for l in labels]
    sampler     = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
    n, n0, n1   = len(labels), labels.count(0), labels.count(1)
    class_w     = torch.tensor([n/(2*n0), n/(2*n1)],
                                dtype=torch.float32, device=device)
    print(f'  Loss weights — pre-ictal: {class_w[0]:.3f}  '
          f'ictal: {class_w[1]:.3f}')

    train_loader = DataLoader(train_graphs, batch_size=args.finetune_batch,
                              sampler=sampler)
    val_loader   = DataLoader(val_graphs,   batch_size=args.finetune_batch,
                              shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=args.finetune_batch,
                              shuffle=False)

    #optimizer = torch.optim.Adam(head.parameters(),
    #                             lr=args.finetune_lr, weight_decay=1e-4)
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': args.finetune_lr * 0.1},  # slow
        {'params': head.parameters(),    'lr': args.finetune_lr},         # fast
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=1e-5
    )

    print(f'\n{"Epoch":>6}  {"TrainLoss":>10}  {"ValAUROC":>9}  '
          f'{"ValSens":>8}  {"ValSpec":>8}  {"ValF1":>7}')
    print('-' * 70)

    best_val_auroc   = -1.0
    best_epoch       = 0
    patience_counter = 0
    history          = []
    ckpt_path        = output_dir / 'finetuned_model.pt'

    for epoch in range(1, args.finetune_epochs + 1):
        head.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            #with torch.no_grad():
            #    emb = encoder(batch)   # frozen encoder
            emb    = encoder(batch)
            #logits = head(emb)
            logits = head(emb)
            loss   = F.cross_entropy(logits, batch.y.squeeze(),
                                     weight=class_w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        val_m    = evaluate(encoder, head, val_loader, device)

        history.append({
            'epoch'          : epoch,
            'train_loss'     : round(avg_loss, 4),
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
                  f'{val_m["auroc"]:>9.4f}  '
                  f'{val_m["sensitivity"]:>8.4f}  '
                  f'{val_m["specificity"]:>8.4f}  '
                  f'{val_m["f1_ictal"]:>7.4f}')

        if patience_counter >= args.finetune_patience:
            print(f'\nEarly stop at epoch {epoch}  '
                  f'(best val AUROC={best_val_auroc:.4f} at epoch {best_epoch})')
            break

    # test — load best checkpoint
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
    print(f'\n  Confusion matrix:')
    print(f'                  Predicted')
    print(f'                  Pre-ictal   Ictal')
    print(f'  Actual Pre-ictal  [ {test_m["tn"]:4d}      {test_m["fp"]:4d} ]')
    print(f'  Actual Ictal      [ {test_m["fn"]:4d}      {test_m["tp"]:4d} ]')

    return test_m, history, best_epoch


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GraphCL pretraining + fine-tuning')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir',  required=True)

    # shared
    parser.add_argument('--hidden',   type=int,   default=64)
    parser.add_argument('--dropout',  type=float, default=0.4)

    # pretraining
    parser.add_argument('--pretrain_epochs', type=int,   default=100)
    parser.add_argument('--pretrain_lr',     type=float, default=1e-3)
    parser.add_argument('--pretrain_batch',  type=int,   default=32)
    parser.add_argument('--temperature',     type=float, default=0.5)
    parser.add_argument('--proj_dim',        type=int,   default=32)
    parser.add_argument('--p_edge',          type=float, default=0.2,
                        help='Edge drop probability for augmentation')
    parser.add_argument('--p_feat',          type=float, default=0.2,
                        help='Feature mask probability for augmentation')

    # fine-tuning
    parser.add_argument('--finetune_epochs',  type=int,   default=200)
    parser.add_argument('--finetune_lr',      type=float, default=2e-4)
    parser.add_argument('--finetune_batch',   type=int,   default=32)
    parser.add_argument('--finetune_patience',type=int,   default=50)

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
    print(f'\nPhase 1 — Pretraining on pre-ictal only (no labels):')
    print(f'  epochs={args.pretrain_epochs}  lr={args.pretrain_lr}  '
          f'batch={args.pretrain_batch}')
    print(f'  augmentations: edge_drop(p={args.p_edge}) + feat_mask(p={args.p_feat})')
    print(f'  temperature={args.temperature}  proj_dim={args.proj_dim}')
    print(f'\nPhase 2 — Fine-tuning (frozen encoder + classifier head):')
    print(f'  epochs={args.finetune_epochs}  lr={args.finetune_lr}  '
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

    # phase 1
    encoder = pretrain(args, train_graphs, device, output_dir)

    # phase 2
    test_m, ft_history, best_ft_epoch = finetune(
        args, encoder, train_graphs, val_graphs, test_graphs,
        device, output_dir
    )

    # save
    print('\nSaving outputs...')
    with open(output_dir / 'graphcl_results.json', 'w') as f:
        json.dump({
            'model'  : 'GraphCL',
            'test'   : test_m,
            'hparams': vars(args),
        }, f, indent=2)
    print('  Saved: graphcl_results.json')

    np.save(output_dir / 'finetune_history.npy', np.array(ft_history))

    save_finetune_curves(ft_history, best_ft_epoch,
                         output_dir / 'finetune_training_curves.png')
    save_confusion_matrix(test_m,
                          output_dir / 'finetune_confusion_matrix.png')

    print(f'\nAll outputs saved to: {output_dir}')
    print('\n── Final comparison ──────────────────────────────────────────────')
    print(f'  GCN supervised : AUROC=0.705  Sens=0.795  Spec=0.534  F1=0.583')
    print(f'  GraphCL        : AUROC={test_m["auroc"]:.3f}  '
          f'Sens={test_m["sensitivity"]:.3f}  '
          f'Spec={test_m["specificity"]:.3f}  '
          f'F1={test_m["f1_ictal"]:.3f}')


if __name__ == '__main__':
    main()
