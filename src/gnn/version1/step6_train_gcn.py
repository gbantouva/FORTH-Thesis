"""
Step 6 — GCN Baseline (Supervised)
=====================================
2-layer Graph Convolutional Network for ictal vs pre-ictal classification.

Saves to output_dir:
    best_gcn.pt                  — best model checkpoint
    gcn_results.json             — all metrics
    gcn_history.npy              — per-epoch training history
    gcn_training_curves.png      — loss + AUROC + sensitivity + specificity
    gcn_overfitting.png          — train vs val loss/AUROC to detect overfitting
    gcn_confusion_matrix.png     — confusion matrix heatmap on test set
    gcn_roc_curve.png            — ROC curve on test set

Usage:
    python step6_train_gcn.py \
        --dataset_dir F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/dataset_dtf \
        --output_dir  F:/FORTH_Final_Thesis/FORTH-Thesis/gnn/results/gcn \
        --lr 1e-3 --dropout 0.4 --patience 30 --epochs 200
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
from sklearn.preprocessing import StandardScaler

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

class GCNClassifier(nn.Module):
    def __init__(self, n_features=9, hidden=64, dropout=0.4):
        super().__init__()
        self.conv1   = GCNConv(n_features, hidden)
        self.bn1     = nn.BatchNorm1d(hidden)
        self.conv2   = GCNConv(hidden, hidden)
        self.bn2     = nn.BatchNorm1d(hidden)
        self.fc1     = nn.Linear(hidden, 32)
        self.fc2     = nn.Linear(32, 2)
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))),
                      p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))),
                      p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout, training=self.training)
        return self.fc2(x)


# ── imbalance ─────────────────────────────────────────────────────────────────

def make_weighted_sampler(graphs):
    labels      = [g.y.item() for g in graphs]
    class_count = [labels.count(0), labels.count(1)]
    weights     = [1.0 / class_count[l] for l in labels]
    print(f'  Sampler weights — pre-ictal: {weights[0]:.5f}  '
          f'ictal: {weights[labels.index(1)]:.5f}')
    return WeightedRandomSampler(weights, num_samples=len(weights),
                                 replacement=True)

def compute_class_weights(graphs, device):
    labels = [g.y.item() for g in graphs]
    n      = len(labels)
    n0, n1 = labels.count(0), labels.count(1)
    w0, w1 = n / (2 * n0), n / (2 * n1)
    print(f'  Loss weights   — pre-ictal: {w0:.3f}  ictal: {w1:.3f}')
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)

def normalize_graphs_inplace(graphs, scaler=None, fit=False):
    # stack all node features: (N_epochs * 19, 9)
    all_feats = np.vstack([g.x.numpy() for g in graphs])
    if fit:
        scaler = StandardScaler()
        scaler.fit(all_feats)
    for g in graphs:
        g.x = torch.tensor(
            scaler.transform(g.x.numpy()),
            dtype=torch.float32
        )
    return scaler

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
        'auroc'      : round(float(auroc), 4),
        'sensitivity': round(tp / max(tp + fn, 1), 4),
        'specificity': round(tn / max(tn + fp, 1), 4),
        'f1_ictal'   : round(float(f1_score(labels_np, preds,
                                            pos_label=1,
                                            zero_division=0)), 4),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
        'probs'     : probs,        # kept for plotting, excluded from JSON
        'labels_np' : labels_np,    # kept for plotting, excluded from JSON
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def save_training_curves(history: list, best_epoch: int, out_path: Path):
    """4-panel: train loss, val AUROC, val sensitivity, val specificity."""
    epochs     = [h['epoch']            for h in history]
    train_loss = [h['train_loss']       for h in history]
    val_auroc  = [h['val_auroc']        for h in history]
    val_sens   = [h['val_sensitivity']  for h in history]
    val_spec   = [h['val_specificity']  for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('GCN Training Curves', fontsize=14, fontweight='bold')

    panels = [
        (axes[0, 0], train_loss, 'Training Loss',         'Loss',        'steelblue'),
        (axes[0, 1], val_auroc,  'Validation AUROC',      'AUROC',       'darkorange'),
        (axes[1, 0], val_sens,   'Validation Sensitivity','Sensitivity', 'seagreen'),
        (axes[1, 1], val_spec,   'Validation Specificity','Specificity', 'mediumpurple'),
    ]

    for ax, values, title, ylabel, color in panels:
        ax.plot(epochs, values, color=color, linewidth=1.5)
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


def save_overfitting_curves(history: list, best_epoch: int, out_path: Path):
    """
    2-panel overfitting diagnostic:
      Left  — Train loss vs Val loss
      Right — Train AUROC (approx) vs Val AUROC

    How to read:
      - If train loss keeps falling but val loss rises   → overfitting
      - If both curves fall together and plateau         → healthy training
      - If both curves are high and flat                 → underfitting
    """
    epochs     = [h['epoch']           for h in history]
    train_loss = [h['train_loss']      for h in history]
    val_loss   = [h['val_loss']        for h in history]
    val_auroc  = [h['val_auroc']       for h in history]
    train_auroc= [h['train_auroc']     for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('GCN — Overfitting Diagnostic', fontsize=14, fontweight='bold')

    # ── Loss panel ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_loss, color='steelblue',  lw=2, label='Train loss')
    ax.plot(epochs, val_loss,   color='darkorange', lw=2, label='Val loss')
    ax.axvline(best_epoch, color='red', linestyle='--',
               lw=1.2, label=f'Best epoch {best_epoch}')
    ax.set_title('Loss: Train vs Validation', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # shade the gap after best epoch to highlight overfitting region
    ax.axvspan(best_epoch, max(epochs), alpha=0.06, color='red',
               label='Overfitting region')

    # ── AUROC panel ───────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, train_auroc, color='steelblue',  lw=2, label='Train AUROC')
    ax.plot(epochs, val_auroc,   color='darkorange', lw=2, label='Val AUROC')
    ax.axvline(best_epoch, color='red', linestyle='--',
               lw=1.2, label=f'Best epoch {best_epoch}')
    ax.set_title('AUROC: Train vs Validation', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax.axvspan(best_epoch, max(epochs), alpha=0.06, color='red')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_roc_curve(labels_np, probs, metrics: dict,
                   out_path: Path, model_name: str = 'GCN'):
    fpr, tpr, _ = roc_curve(labels_np, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'AUROC = {metrics["auroc"]:.4f}')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--',
            lw=1, label='Random classifier')

    # mark the operating point (sensitivity, 1-specificity) from best threshold
    ax.scatter(1 - metrics['specificity'], metrics['sensitivity'],
               color='red', zorder=5, s=80,
               label=f'Operating point\n'
                     f'Sens={metrics["sensitivity"]:.3f}  '
                     f'Spec={metrics["specificity"]:.3f}')

    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)',       fontsize=12)
    ax.set_title(f'{model_name} — ROC Curve (Test Set)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


def save_confusion_matrix(metrics: dict, out_path: Path,
                           model_name: str = 'GCN'):
    tn, fp = metrics['tn'], metrics['fp']
    fn, tp = metrics['fn'], metrics['tp']
    cm     = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes = ['Pre-ictal (0)', 'Ictal (1)']
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label',      fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / max(cm[i].sum(), 1)
            ax.text(j, i, f'{cm[i, j]}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=13,
                    color='white' if cm[i, j] > thresh else 'black',
                    fontweight='bold')

    ax.set_title(
        f'{model_name} — Test Set Confusion Matrix\n'
        f'AUROC={metrics["auroc"]:.4f}  '
        f'Sens={metrics["sensitivity"]:.4f}  '
        f'Spec={metrics["specificity"]:.4f}  '
        f'F1={metrics["f1_ictal"]:.4f}',
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

    print('Normalizing features across splits...')
    scaler = normalize_graphs_inplace(train_graphs, fit=True)
    normalize_graphs_inplace(val_graphs,  scaler=scaler)
    normalize_graphs_inplace(test_graphs, scaler=scaler)

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

    model = GCNClassifier(n_features=9, hidden=args.hidden,
                          dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel: GCN  hidden={args.hidden}  dropout={args.dropout}  '
          f'params={n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    print('\n' + '=' * 70)
    print(f'TRAINING  epochs={args.epochs}  patience={args.patience}  '
          f'lr={args.lr}  batch={args.batch_size}')
    print('=' * 70)
    print(f'{"Epoch":>6}  {"TrainLoss":>10}  {"ValLoss":>8}  '
          f'{"ValAUROC":>9}  {"ValSens":>8}  {"ValSpec":>8}  {"ValF1":>7}')
    print('-' * 70)

    best_val_auroc   = -1.0
    best_epoch       = 0
    patience_counter = 0
    history          = []
    ckpt_path        = output_dir / 'best_gcn.pt'

    for epoch in range(1, args.epochs + 1):
        # ── train step ────────────────────────────────────────────────────────
        model.train()
        epoch_loss    = 0.0
        all_logits_tr, all_labels_tr = [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            #loss   = F.cross_entropy(logits, batch.y.squeeze(), weight=class_w)
            loss   = F.cross_entropy(logits, batch.y.squeeze())  # ← same logits
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            all_logits_tr.append(logits.detach().cpu())
            all_labels_tr.append(batch.y.cpu())

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # ── train AUROC (for overfitting plot) ────────────────────────────────
        tr_logits  = torch.cat(all_logits_tr)
        tr_labels  = torch.cat(all_labels_tr).numpy()
        tr_probs   = F.softmax(tr_logits, dim=1)[:, 1].numpy()
        try:
            train_auroc = roc_auc_score(tr_labels, tr_probs)
        except ValueError:
            train_auroc = float('nan')

        # ── val step ──────────────────────────────────────────────────────────
        val_m = evaluate(model, val_loader, device)

        history.append({
            'epoch'          : epoch,
            'train_loss'     : round(avg_train_loss, 4),
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
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'{epoch:>6}  {avg_train_loss:>10.4f}  '
                  f'{val_m["loss"]:>8.4f}  '
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
          f'— correct on {test_m["tn"]} / {test_m["tn"]+test_m["fp"]} pre-ictal epochs')
    print(f'  F1-ictal    : {test_m["f1_ictal"]:.4f}')
    print(f'\n  Confusion matrix (test set):')
    print(f'                  Predicted')
    print(f'                  Pre-ictal   Ictal')
    print(f'  Actual Pre-ictal  [ {test_m["tn"]:4d}      {test_m["fp"]:4d} ]')
    print(f'  Actual Ictal      [ {test_m["fn"]:4d}      {test_m["tp"]:4d} ]')

    # ── save ──────────────────────────────────────────────────────────────────
    print('\nSaving outputs...')

    test_m_json = {k: v for k, v in test_m.items()
                   if k not in ('probs', 'labels_np')}

    results = {
        'experiment': {
            'script'     : 'step6_train_gcn.py',
            'timestamp'  : datetime.now().isoformat(timespec='seconds'),
            'model'      : 'GCN',
            'description': (
                '2-layer GCN supervised baseline. Edge weights (DTF) define '
                'graph topology only — not used in message-passing.'
            ),
        },
        'hyperparameters': {
            'hidden'      : args.hidden,
            'dropout'     : args.dropout,
            'lr'          : args.lr,
            'epochs_max'  : args.epochs,
            'batch_size'  : args.batch_size,
            'patience'    : args.patience,
            'optimizer'   : 'Adam',
            'weight_decay': 1e-4,
            'scheduler'   : 'CosineAnnealingLR',
            'grad_clip'   : 1.0,
            #'class_weight': 'balanced (loss + WeightedRandomSampler)',
            'class_weight': 'WeightedRandomSampler only (no loss reweighting)',
            'seed'        : SEED,
        },
        'data': {
            'dataset_dir': str(args.dataset_dir),
            'split_sizes': {
                'train': {
                    'total'   : len(train_graphs),
                    'ictal'   : sum(1 for g in train_graphs if g.y.item() == 1),
                    'preictal': sum(1 for g in train_graphs if g.y.item() == 0),
                },
                'val': {
                    'total'   : len(val_graphs),
                    'ictal'   : sum(1 for g in val_graphs if g.y.item() == 1),
                    'preictal': sum(1 for g in val_graphs if g.y.item() == 0),
                },
                'test': {
                    'total'   : len(test_graphs),
                    'ictal'   : sum(1 for g in test_graphs if g.y.item() == 1),
                    'preictal': sum(1 for g in test_graphs if g.y.item() == 0),
                },
            },
        },
        'training': {
            'best_epoch'      : best_epoch,
            'best_val_auroc'  : round(best_val_auroc, 4),
            'total_epochs_run': len(history),
            'stopped_early'   : len(history) < args.epochs,
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
        'outputs': {
            'output_dir'  : str(output_dir),
            'checkpoint'  : 'best_gcn.pt',
            'history_file': 'gcn_history.npy',
            'plots'       : [
                'gcn_training_curves.png',
                'gcn_overfitting.png',
                'gcn_roc_curve.png',
                'gcn_confusion_matrix.png',
            ],
        },
    }

    with open(output_dir / 'gcn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('  Saved: gcn_results.json')

    np.save(output_dir / 'gcn_history.npy', np.array(history))
    print('  Saved: gcn_history.npy')

    save_training_curves(history, best_epoch,
                         output_dir / 'gcn_training_curves.png')
    save_overfitting_curves(history, best_epoch,
                            output_dir / 'gcn_overfitting.png')
    save_roc_curve(test_m['labels_np'], test_m['probs'], test_m_json,
                   output_dir / 'gcn_roc_curve.png', model_name='GCN')
    save_confusion_matrix(test_m_json,
                          output_dir / 'gcn_confusion_matrix.png',
                          model_name='GCN')

    print(f'\nAll outputs saved to: {output_dir}')
    print('Next: python step7_train_gat.py --dataset_dir <dataset_dir>')


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train GCN baseline')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--hidden',      type=int,   default=64)
    parser.add_argument('--dropout',     type=float, default=0.4)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--patience',    type=int,   default=30)
    args = parser.parse_args()

    print('=' * 70)
    print('STEP 6 — GCN BASELINE')
    print('=' * 70)
    train(args)


if __name__ == '__main__':
    main()
