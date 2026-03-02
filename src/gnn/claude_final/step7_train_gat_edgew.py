"""
Step 7 — GAT with Edge-Aware Attention (v2)
============================================
Key fix over v1: passes DTF edge weights into the attention computation
via edge_dim=1. This prevents attention collapse on the fully-connected
19-node graph by giving the attention mechanism explicit information about
directed connectivity strength.

Attention score for edge j→i:
    e_ij = LeakyReLU( a^T [ W·h_i || W·h_j || W_e·dtf_ij ] )
    α_ij = softmax over neighbours j of e_ij

This means high-DTF edges get systematically higher attention, while the
model can still learn to override this when node features suggest otherwise.

Architecture:
    GATConv(9 → 64, heads=4, edge_dim=1, concat=True)  → 256
    BatchNorm → ELU → Dropout
    GATConv(256 → 64, heads=1, edge_dim=1, concat=False) → 64
    BatchNorm → ELU → Dropout
    GlobalMeanPool
    Linear(64 → 32) → ELU → Dropout
    Linear(32 → 2)

Outputs:
    best_gat.pt
    gat_results.json
    gat_history.npy
    gat_training_curves.png
    gat_confusion_matrix.png
    gat_attention_weights.png   ← pre-ictal vs ictal attention heatmaps

Usage:
    python step7_train_gat.py \
        --dataset_dir F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/gnn_dataset_2to1 \
        --output_dir  F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/results/gat_v2
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
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CHANNEL_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T3','C3','Cz','C4','T4',
    'T5','P3','Pz','P4','T6',
    'O1','O2',
]


# ── model ─────────────────────────────────────────────────────────────────────

class GATClassifier(nn.Module):
    """
    2-layer GAT with edge-aware attention.

    edge_dim=1 tells GATConv to include the DTF edge weight in the
    attention score computation. Without this, all 18 neighbours of a
    fully-connected node receive approximately equal attention (collapse).
    """
    def __init__(self, n_features=9, hidden=64, heads=4, dropout=0.4):
        super().__init__()
        self.dropout = dropout

        # Layer 1: 4 heads, concat → hidden*heads output
        self.conv1 = GATConv(
            in_channels  = n_features,
            out_channels = hidden,
            heads        = heads,
            concat       = True,
            dropout      = dropout,
            edge_dim     = 1,        # ← use DTF weight in attention
        )
        self.bn1 = nn.BatchNorm1d(hidden * heads)

        # Layer 2: 1 head, no concat → hidden output
        self.conv2 = GATConv(
            in_channels  = hidden * heads,
            out_channels = hidden,
            heads        = 1,
            concat       = False,
            dropout      = dropout,
            edge_dim     = 1,        # ← use DTF weight in attention
        )
        self.bn2 = nn.BatchNorm1d(hidden)

        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, data: Data, return_attention=False):
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr   # (n_edges, 1)  DTF weights
        batch      = data.batch

        # Layer 1
        if return_attention:
            x, (ei1, a1) = self.conv1(x, edge_index, edge_attr=edge_attr,
                                      return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(self.bn1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        if return_attention:
            x, (ei2, a2) = self.conv2(x, edge_index, edge_attr=edge_attr,
                                      return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(self.bn2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x      = global_mean_pool(x, batch)
        x      = F.elu(self.fc1(x))
        x      = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.fc2(x)

        if return_attention:
            return logits, (ei1, a1), (ei2, a2)
        return logits


# ── imbalance ─────────────────────────────────────────────────────────────────

def make_weighted_sampler(graphs):
    labels      = [g.y.item() for g in graphs]
    class_count = [labels.count(0), labels.count(1)]
    weights     = [1.0 / class_count[l] for l in labels]
    print(f'  Sampler — pre-ictal: {weights[0]:.5f}  '
          f'ictal: {weights[labels.index(1)]:.5f}')
    return WeightedRandomSampler(weights, num_samples=len(weights),
                                 replacement=True)


def compute_class_weights(graphs, device):
    labels = [g.y.item() for g in graphs]
    n      = len(labels)
    n0, n1 = labels.count(0), labels.count(1)
    w0, w1 = n / (2 * n0), n / (2 * n1)
    print(f'  Loss   — pre-ictal: {w0:.3f}  ictal: {w1:.3f}')
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        all_logits.append(model(batch).cpu())
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
        'sensitivity': round(tp / max(tp + fn, 1),     4),
        'specificity': round(tn / max(tn + fp, 1),     4),
        'f1_ictal'   : round(float(f1_score(
                            labels_np, preds,
                            pos_label=1, zero_division=0)), 4),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
    }


# ── attention visualisation ───────────────────────────────────────────────────

@torch.no_grad()
def extract_attention_matrix(model, graphs, device, n_graphs=50):
    """
    Average layer-2 attention weights over up to n_graphs of each class.
    Returns two (19,19) matrices: pre-ictal and ictal mean attention.
    """
    model.eval()
    ictal_graphs    = [g for g in graphs if g.y.item() == 1][:n_graphs]
    preictal_graphs = [g for g in graphs if g.y.item() == 0][:n_graphs]

    def avg_attn(graph_list):
        attn_sum   = np.zeros((19, 19), dtype=np.float64)
        attn_count = np.zeros((19, 19), dtype=np.float64)
        for g in graph_list:
            b = Batch.from_data_list([g]).to(device)
            _, _, (ei2, a2) = model(b, return_attention=True)
            src = ei2[0].cpu().numpy()
            dst = ei2[1].cpu().numpy()
            w   = a2[:, 0].cpu().numpy()
            for s, d, a in zip(src, dst, w):
                if s < 19 and d < 19:
                    attn_sum[d, s]   += a
                    attn_count[d, s] += 1
        return attn_sum / np.where(attn_count > 0, attn_count, 1)

    return avg_attn(preictal_graphs), avg_attn(ictal_graphs)


def save_attention_plot(pre_attn, ict_attn, out_path):
    diff = ict_attn - pre_attn
    vmax = max(pre_attn.max(), ict_attn.max())
    dlim = np.abs(diff).max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'GAT Layer-2 Attention Weights (edge-aware)\n'
        'Rows = sink (receives), Cols = source (sends)',
        fontsize=13, fontweight='bold'
    )

    for ax, mat, title, cmap, vmin, vmx in [
        (axes[0], pre_attn, 'Pre-ictal mean attention', 'Blues',   0,    vmax),
        (axes[1], ict_attn, 'Ictal mean attention',     'Reds',    0,    vmax),
        (axes[2], diff,     'Difference (ictal − pre)', 'RdBu_r', -dlim, dlim),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmx, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(19))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=7)
        ax.set_yticks(range(19))
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=7)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Source channel (sends)')
        ax.set_ylabel('Sink channel (receives)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')

    # also print the top 5 channel pairs with increased attention during ictal
    pairs = []
    for i in range(19):
        for j in range(19):
            if i != j:
                pairs.append((diff[i, j], CHANNEL_NAMES[j], CHANNEL_NAMES[i]))
    pairs.sort(reverse=True)
    print('\n  Top 5 channel pairs with INCREASED attention during ictal:')
    for val, src, dst in pairs[:5]:
        print(f'    {src} → {dst}  Δattention={val:+.5f}')
    print('  Top 5 channel pairs with DECREASED attention during ictal:')
    for val, src, dst in pairs[-5:]:
        print(f'    {src} → {dst}  Δattention={val:+.5f}')


def save_training_curves(history, best_epoch, out_path):
    epochs     = [h['epoch']           for h in history]
    train_loss = [h['train_loss']      for h in history]
    val_auroc  = [h['val_auroc']       for h in history]
    val_sens   = [h['val_sensitivity'] for h in history]
    val_spec   = [h['val_specificity'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('GAT Training Curves', fontsize=14, fontweight='bold')

    for ax, vals, title, ylabel, color in [
        (axes[0,0], train_loss, 'Training Loss',          'Loss',        'steelblue'),
        (axes[0,1], val_auroc,  'Validation AUROC',       'AUROC',       'darkorange'),
        (axes[1,0], val_sens,   'Validation Sensitivity', 'Sensitivity', 'seagreen'),
        (axes[1,1], val_spec,   'Validation Specificity', 'Specificity', 'mediumpurple'),
    ]:
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


def save_confusion_matrix(metrics, out_path, model_name='GAT'):
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Oranges')
    plt.colorbar(im, ax=ax)
    classes = ['Pre-ictal (0)', 'Ictal (1)']
    ax.set_xticks([0,1]); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([0,1]); ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label',      fontsize=12)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i,j] / max(cm[i].sum(), 1)
            ax.text(j, i, f'{cm[i,j]}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if cm[i,j] > thresh else 'black')
    ax.set_title(
        f'{model_name} — Test Set Confusion Matrix\n'
        f'AUROC={metrics["auroc"]:.4f}  Sens={metrics["sensitivity"]:.4f}  '
        f'Spec={metrics["specificity"]:.4f}  F1={metrics["f1_ictal"]:.4f}',
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ── training ──────────────────────────────────────────────────────────────────

def train(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    print('Loading dataset...')
    train_graphs = torch.load(dataset_dir / 'train_graphs.pt', weights_only=False)
    val_graphs   = torch.load(dataset_dir / 'val_graphs.pt',   weights_only=False)
    test_graphs  = torch.load(dataset_dir / 'test_graphs.pt',  weights_only=False)

    print(f'  Train: {len(train_graphs):5d}  '
          f'(ictal={sum(1 for g in train_graphs if g.y.item()==1)})')
    print(f'  Val  : {len(val_graphs):5d}  '
          f'(ictal={sum(1 for g in val_graphs if g.y.item()==1)})')
    print(f'  Test : {len(test_graphs):5d}  '
          f'(ictal={sum(1 for g in test_graphs if g.y.item()==1)})')

    print('\nImbalance correction:')
    sampler = make_weighted_sampler(train_graphs)
    class_w = compute_class_weights(train_graphs, device)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size,
                              sampler=sampler)
    val_loader   = DataLoader(val_graphs,   batch_size=args.batch_size,
                              shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=args.batch_size,
                              shuffle=False)

    model = GATClassifier(
        n_features = 9,
        hidden     = args.hidden,
        heads      = args.heads,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel: GAT (edge-aware)  hidden={args.hidden}  heads={args.heads}  '
          f'dropout={args.dropout}  params={n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    print('\n' + '=' * 70)
    print(f'TRAINING  epochs={args.epochs}  patience={args.patience}  '
          f'lr={args.lr}  batch={args.batch_size}')
    print('=' * 70)
    print(f'{"Epoch":>6}  {"TrainLoss":>10}  {"ValAUROC":>9}  '
          f'{"ValSens":>8}  {"ValSpec":>8}  {"ValF1":>7}')
    print('-' * 70)

    best_val_auroc   = -1.0
    best_epoch       = 0
    patience_counter = 0
    history          = []
    ckpt_path        = output_dir / 'best_gat.pt'

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y.squeeze(),
                                   weight=class_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        val_m    = evaluate(model, val_loader, device)

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
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'{epoch:>6}  {avg_loss:>10.4f}  '
                  f'{val_m["auroc"]:>9.4f}  '
                  f'{val_m["sensitivity"]:>8.4f}  '
                  f'{val_m["specificity"]:>8.4f}  '
                  f'{val_m["f1_ictal"]:>7.4f}')

        if patience_counter >= args.patience:
            print(f'\nEarly stop at epoch {epoch}  '
                  f'(best val AUROC={best_val_auroc:.4f} at epoch {best_epoch})')
            break

    # ── test ──────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f'TEST EVALUATION  (checkpoint from epoch {best_epoch})')
    print('=' * 70)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False,
                                     map_location=device))
    test_m = evaluate(model, test_loader, device)

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

    # ── save ──────────────────────────────────────────────────────────────────
    print('\nSaving outputs...')
    with open(output_dir / 'gat_results.json', 'w') as f:
        json.dump({
            'model'         : 'GAT_edge_aware',
            'best_epoch'    : best_epoch,
            'best_val_auroc': best_val_auroc,
            'test'          : test_m,
            'hparams'       : vars(args),
        }, f, indent=2)
    print('  Saved: gat_results.json')

    np.save(output_dir / 'gat_history.npy', np.array(history))
    print('  Saved: gat_history.npy')

    save_training_curves(history, best_epoch,
                         output_dir / 'gat_training_curves.png')
    save_confusion_matrix(test_m,
                          output_dir / 'gat_confusion_matrix.png')

    print('\nExtracting attention weights...')
    pre_attn, ict_attn = extract_attention_matrix(
        model, test_graphs, device, n_graphs=50
    )
    np.save(output_dir / 'gat_attn_preictal.npy', pre_attn)
    np.save(output_dir / 'gat_attn_ictal.npy',    ict_attn)
    save_attention_plot(pre_attn, ict_attn,
                        output_dir / 'gat_attention_weights.png')

    print(f'\nAll outputs saved to: {output_dir}')


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train edge-aware GAT')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--hidden',      type=int,   default=64)
    parser.add_argument('--heads',       type=int,   default=4)
    parser.add_argument('--dropout',     type=float, default=0.4)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--epochs',      type=int,   default=300)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--patience',    type=int,   default=50)
    args = parser.parse_args()

    print('=' * 70)
    print('STEP 7 — GAT (edge-aware, v2)')
    print('=' * 70)
    train(args)


if __name__ == '__main__':
    main()