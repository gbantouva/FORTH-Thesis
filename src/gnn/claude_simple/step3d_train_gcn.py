"""
Step 3d — Supervised GCN Training (LOSO-CV)
=============================================
Trains a simple 2-layer Graph Convolutional Network (GCN) for ictal vs
pre-ictal classification using Leave-One-Subject-Out cross validation.

Architecture (deliberately simple for a thesis baseline)
──────────────────────────────────────────────────────────
  Input: node features (19, 12)
  ↓
  GCNConv(12 → 64)  + ReLU + Dropout
  ↓
  GCNConv(64 → 64)  + ReLU + Dropout
  ↓
  Global Mean Pooling  → (64,)   [graph-level representation]
  ↓
  Linear(64 → 32) + ReLU + Dropout
  ↓
  Linear(32 → 2)   [logits for 2 classes]
  ↓
  Softmax → predicted label

Why this architecture?
  - 2 GCN layers: standard starting point, enough for 2-hop neighbourhood
  - Global mean pooling: aggregates all node embeddings → graph-level vector
  - Simple MLP head: maps graph embedding to class logits
  - class_weight: handles ictal/pre-ictal imbalance

Usage:
    python step3d_train_gcn.py \\
        --datadir  path/to/graphs \\
        --outdir   path/to/gcn_results

    # Quick test with fewer epochs:
    python step3d_train_gcn.py --datadir path/to/graphs --epochs 20

    # Specific LOSO fold (useful for debugging):
    python step3d_train_gcn.py --datadir path/to/graphs --fold subject_01
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 0. IMPORTS (with helpful errors)
# ══════════════════════════════════════════════════════════════════════════════

def import_deps():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("Install PyTorch: pip install torch")

    try:
        from torch_geometric.data    import DataLoader
        from torch_geometric.nn      import GCNConv, global_mean_pool
    except ImportError:
        raise ImportError("Install PyG: pip install torch_geometric")

    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        confusion_matrix,
    )

    return torch, nn, F, DataLoader, GCNConv, global_mean_pool


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(torch, nn, F, GCNConv, global_mean_pool,
                in_channels=12, hidden=64, n_classes=2, dropout=0.3):
    """
    Returns a simple 2-layer GCN model class.
    """

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()

            # ── Graph convolutional layers ────────────────────────────────
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, hidden)

            # ── MLP classifier head ───────────────────────────────────────
            self.lin1  = nn.Linear(hidden, hidden // 2)
            self.lin2  = nn.Linear(hidden // 2, n_classes)

            self.dropout = nn.Dropout(p=dropout)
            self.bn1     = nn.BatchNorm1d(hidden)
            self.bn2     = nn.BatchNorm1d(hidden)

        def forward(self, data):
            x, edge_index, edge_weight, batch = (
                data.x,
                data.edge_index,
                data.edge_attr.squeeze(-1) if data.edge_attr is not None else None,
                data.batch,
            )

            # ── Layer 1 ───────────────────────────────────────────────────
            x = self.conv1(x, edge_index, edge_weight)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout(x)

            # ── Layer 2 ───────────────────────────────────────────────────
            x = self.conv2(x, edge_index, edge_weight)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout(x)

            # ── Global pooling → graph embedding ──────────────────────────
            x = global_mean_pool(x, batch)   # (batch_size, hidden)

            # ── MLP head ──────────────────────────────────────────────────
            x = F.relu(self.lin1(x))
            x = self.dropout(x)
            x = self.lin2(x)                 # (batch_size, 2)

            return x

    return GCN()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASS WEIGHTS  (handle ictal/pre-ictal imbalance)
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(graph_list, torch):
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    """
    labels = np.array([int(g.y.item()) for g in graph_list])
    n_total   = len(labels)
    n_ictal   = (labels == 1).sum()
    n_pre     = (labels == 0).sum()

    # weight[c] = total / (n_classes * n_c)
    w0 = n_total / (2 * n_pre)   if n_pre   > 0 else 1.0
    w1 = n_total / (2 * n_ictal) if n_ictal > 0 else 1.0

    return torch.tensor([w0, w1], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, torch):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        #loss   = criterion(logits, batch.y.squeeze())
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device, torch):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix
    import torch.nn.functional as F

    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch)
            labels = batch.y.view(-1)          # ← view(-1) not squeeze()
            loss   = criterion(logits, labels)
            probs  = F.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item()
            n_batches  += 1

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    avg_loss = total_loss / max(n_batches, 1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = specificity = 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    return {
        'loss':        avg_loss,
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'auc':         float(auc),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN ONE LOSO FOLD
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(fold_name, train_graphs, test_graphs,
               args, torch, nn, F, DataLoader, GCNConv, global_mean_pool,
               device, output_dir):
    """
    Full training loop for one LOSO fold.

    Returns
    -------
    best_metrics : dict  — test metrics at best validation AUC epoch
    history      : dict  — per-epoch train/test metrics
    """
    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_graphs,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(torch, nn, F, GCNConv, global_mean_pool,
                        in_channels=args.in_channels,
                        hidden=args.hidden,
                        n_classes=2,
                        dropout=args.dropout).to(device)

    # ── Loss with class weights ────────────────────────────────────────────
    class_weights = compute_class_weights(train_graphs, torch).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────
    history     = {'train_loss': [], 'test_loss': [],
                   'train_auc': [],  'test_auc': [],
                   'test_acc':  [],  'test_f1':  []}
    best_auc    = 0.0
    best_metrics = {}
    best_epoch  = 0
    patience_counter = 0

    pbar = tqdm(range(1, args.epochs + 1),
                desc=f"  {fold_name}", leave=False)

    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer,
                                  criterion, device, torch)
        test_m     = evaluate(model, test_loader, criterion, device, torch)
        train_m    = evaluate(model, train_loader, criterion, device, torch)

        scheduler.step(test_m['auc'])

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_m['loss'])
        history['train_auc'].append(train_m['auc'])
        history['test_auc'].append(test_m['auc'])
        history['test_acc'].append(test_m['accuracy'])
        history['test_f1'].append(test_m['f1'])

        pbar.set_postfix({
            'loss': f"{train_loss:.3f}",
            'auc':  f"{test_m['auc']:.3f}",
            'best': f"{best_auc:.3f}",
        })

        # ── Save best model ───────────────────────────────────────────────
        if test_m['auc'] > best_auc:
            best_auc     = test_m['auc']
            best_metrics = dict(test_m)
            best_epoch   = epoch
            patience_counter = 0
            torch.save(model.state_dict(),
                       output_dir / f"{fold_name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                tqdm.write(f"  Early stopping at epoch {epoch} "
                           f"(best AUC {best_auc:.3f} at epoch {best_epoch})")
                break

    best_metrics['best_epoch'] = best_epoch
    return best_metrics, history


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_fold_history(history, fold_name, best_epoch, out_path):
    """Learning curves for one fold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train loss', color='blue')
    axes[0].plot(epochs, history['test_loss'],  label='Test loss',  color='red')
    axes[0].axvline(best_epoch, color='green', linestyle='--', label=f'Best epoch {best_epoch}')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{fold_name} — Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_auc'], label='Train AUC', color='blue')
    axes[1].plot(epochs, history['test_auc'],  label='Test AUC',  color='red')
    axes[1].axvline(best_epoch, color='green', linestyle='--', label=f'Best epoch {best_epoch}')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title(f'{fold_name} — AUC'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_final_results(all_results, output_dir):
    """Summary plot across all LOSO folds."""
    subjects = [r['subject'] for r in all_results]
    metrics  = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    colors   = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # ── AUC per subject ───────────────────────────────────────────────────
    aucs = [r['auc'] for r in all_results]
    bar_colors = ['#e74c3c' if a < 0.6 else '#f39c12' if a < 0.8 else '#27ae60'
                  for a in aucs]
    axes[0].bar(range(len(subjects)), aucs, color=bar_colors,
                alpha=0.85, edgecolor='black')
    axes[0].axhline(np.mean(aucs), color='navy', linestyle='--', lw=2,
                    label=f'Mean AUC = {np.mean(aucs):.3f}')
    axes[0].axhline(0.5, color='gray', linestyle=':', lw=1.5, label='Chance')
    axes[0].set_xticks(range(len(subjects)))
    axes[0].set_xticklabels(subjects, rotation=90, fontsize=8)
    axes[0].set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    axes[0].set_title('Per-Subject AUC (LOSO-CV)', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 1.05); axes[0].legend(); axes[0].grid(alpha=0.3, axis='y')

    # ── All metrics boxplot ───────────────────────────────────────────────
    data_for_box = [[r[m] for r in all_results] for m in metrics]
    bp = axes[1].boxplot(data_for_box, labels=metrics, patch_artist=True,
                          showfliers=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title('All Metrics Distribution (LOSO-CV)',
                       fontsize=13, fontweight='bold')
    axes[1].axhline(0.5, color='gray', linestyle='--', lw=1.5, label='Chance')
    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3, axis='y'); axes[1].legend()

    plt.suptitle('Step 3d — GCN Results (LOSO-CV)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'gcn_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_dir / 'gcn_results.png'}")


def print_summary(all_results):
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    print("\n" + "=" * 70)
    print("GCN RESULTS SUMMARY (LOSO-CV)")
    print("=" * 70)
    header = f"{'Subject':<20} " + "  ".join([f"{m[:4].upper():>8}" for m in metrics])
    print(header)
    print("-" * 70)
    for r in all_results:
        line = f"{r['subject']:<20} "
        line += "  ".join([f"{r[m]:>8.3f}" for m in metrics])
        print(line)
    print("-" * 70)
    means = {m: np.mean([r[m] for r in all_results]) for m in metrics}
    stds  = {m: np.std( [r[m] for r in all_results]) for m in metrics}
    line  = f"{'MEAN ± STD':<20} "
    line += "  ".join([f"{means[m]:>5.3f}±{stds[m]:.3f}" for m in metrics])
    print(line)
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train supervised GCN with LOSO-CV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--datadir',      required=True,
                        help='Directory with dataset.pt and loso_splits.json (Step 3c)')
    parser.add_argument('--outdir',       required=True,
                        help='Output directory for results and model checkpoints')

    # Model hyperparameters
    parser.add_argument('--hidden',       type=int,   default=64,
                        help='GCN hidden dimension (default: 64)')
    parser.add_argument('--dropout',      type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--in_channels',  type=int,   default=12,
                        help='Node feature dimension (default: 12)')

    # Training hyperparameters
    parser.add_argument('--epochs',       type=int,   default=100,
                        help='Max training epochs per fold (default: 100)')
    parser.add_argument('--batch_size',   type=int,   default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr',           type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 regularization (default: 1e-4)')
    parser.add_argument('--patience',     type=int,   default=20,
                        help='Early stopping patience in epochs (default: 20)')

    # Fold selection (for debugging)
    parser.add_argument('--fold',         type=str,   default=None,
                        help='Run only this fold, e.g. subject_01')
    parser.add_argument('--device',       type=str,   default='auto',
                        help='Device: auto | cpu | cuda | mps (default: auto)')
    args = parser.parse_args()

    # ── Load dependencies ─────────────────────────────────────────────────
    torch, nn, F, DataLoader, GCNConv, global_mean_pool = import_deps()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else
            'cpu'
        )
    else:
        device = torch.device(args.device)

    data_dir   = Path(args.datadir)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / 'learning_curves'
    curves_dir.mkdir(exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    print("=" * 72)
    print("STEP 3d — SUPERVISED GCN TRAINING")
    print("=" * 72)
    print(f"  Device:       {device}")
    print(f"  Architecture: GCNConv({args.in_channels}→{args.hidden}) × 2 + MLP")
    print(f"  Epochs:       {args.epochs}  (patience={args.patience})")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}  |  WD: {args.weight_decay}")
    print(f"  Dropout:      {args.dropout}")
    print("=" * 72)

    print("\nLoading dataset...")
    graphs = torch.load(data_dir / 'dataset_filtered.pt', map_location='cpu')
    print(f"  ✅ Loaded {len(graphs):,} graphs")

    #with open(data_dir / 'loso_splits.json') as f:
    #    loso_splits = json.load(f)

    splits_file = Path(args.datadir) / 'loso_splits_patient.json'
    with open(splits_file) as f:
        loso_splits = json.load(f)
    
    folds = sorted(loso_splits.keys())
    if args.fold:
        if args.fold not in loso_splits:
            raise ValueError(f"Fold '{args.fold}' not found. "
                             f"Available: {list(loso_splits.keys())}")
        folds = [args.fold]

    print(f"  LOSO folds:   {len(folds)}")
    print()

    # ── LOSO-CV ───────────────────────────────────────────────────────────
    all_results = []

    for fold_name in folds:
        split        = loso_splits[fold_name]
        train_graphs = [graphs[i] for i in split['train']]
        test_graphs  = [graphs[i] for i in split['test']]

        n_train_ict = sum(1 for g in train_graphs if g.y.item() == 1)
        n_test_ict  = sum(1 for g in test_graphs  if g.y.item() == 1)

        print(f"  Fold: {fold_name}")
        print(f"    Train: {len(train_graphs):,} graphs  (ictal={n_train_ict})")
        print(f"    Test:  {len(test_graphs):,}  graphs  (ictal={n_test_ict})")

        best_metrics, history = train_fold(
            fold_name, train_graphs, test_graphs,
            args, torch, nn, F, DataLoader, GCNConv, global_mean_pool,
            device, output_dir,
        )

        best_metrics['subject'] = fold_name
        all_results.append(best_metrics)

        print(f"    → AUC={best_metrics['auc']:.3f}  "
              f"Acc={best_metrics['accuracy']:.3f}  "
              f"Sens={best_metrics['sensitivity']:.3f}  "
              f"Spec={best_metrics['specificity']:.3f}  "
              f"F1={best_metrics['f1']:.3f}  "
              f"(epoch {best_metrics['best_epoch']})\n")

        # Save learning curves
        plot_fold_history(
            history, fold_name,
            best_metrics['best_epoch'],
            curves_dir / f"{fold_name}_curves.png",
        )

    # ── Save results ──────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'gcn_results.csv', index=False)

    with open(output_dir / 'gcn_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Print + plot summary ──────────────────────────────────────────────
    print_summary(all_results)
    plot_final_results(all_results, output_dir)

    # ── Final comparison reminder ──────────────────────────────────────────
    metrics  = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    means    = {m: float(np.mean([r[m] for r in all_results])) for m in metrics}

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"\n  GCN mean AUC:  {means['auc']:.3f}")
    print(f"  GCN mean F1:   {means['f1']:.3f}")
    print(f"  GCN mean Acc:  {means['accuracy']:.3f}")
    print()
    print("  Compare these against your Step 3b baseline numbers.")
    print("  If GCN > baseline → graph structure helps → motivation for")
    print("  moving to self-supervised GNN (your professor's final step).")
    print()
    print("  Saved files:")
    print(f"    {output_dir}/gcn_results.csv      — per-fold metrics")
    print(f"    {output_dir}/gcn_results.json     — same, JSON format")
    print(f"    {output_dir}/gcn_results.png      — summary plot")
    print(f"    {output_dir}/learning_curves/     — per-fold loss/AUC curves")
    print(f"    {output_dir}/subject_XX_best.pt   — best model per fold")
    print("=" * 72)


if __name__ == '__main__':
    main()
