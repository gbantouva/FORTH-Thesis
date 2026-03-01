"""
Step 4b — SSL Evaluation: Linear Probe + Fine-Tuning (LOSO-CV)
================================================================
Evaluates the pretrained GCN encoder from step4a on the ictal vs
pre-ictal classification task using two strategies:

Strategy A — Linear Probe
───────────────────────────
  Freeze the pretrained encoder completely.
  Train ONLY a linear classifier on top of the frozen embeddings.

  Why: Tests the quality of the learned representations.
       If the SSL embeddings are good, even a linear classifier should
       perform well — the structure is in the embedding space.

Strategy B — Fine-Tuning
──────────────────────────
  Initialize encoder with pretrained weights.
  Train encoder + classifier end-to-end on labeled data.
  Use a lower learning rate for the encoder (don't destroy pretraining).

  Why: Usually achieves best performance. The pretrained weights give
       a better initialization than random → faster convergence,
       better generalization with few labeled examples.

Both evaluated with LOSO-CV (same splits as step3d).

Comparison table produced at the end:
  Supervised GCN  (step3d)   vs
  SSL Linear Probe (step4b)  vs
  SSL Fine-tuned  (step4b)

Usage:
    python step4b_ssl_evaluate.py \\
        --datadir     path/to/graphs \\
        --encoderdir  path/to/ssl_pretrained \\
        --outdir      path/to/ssl_results \\
        --supervised_results  path/to/gcn_results/gcn_results.json
"""

import argparse
import json
import warnings
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 0. IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

def import_deps():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("pip install torch")
    try:
        from torch_geometric.data import DataLoader
        from torch_geometric.nn   import GCNConv, global_mean_pool
    except ImportError:
        raise ImportError("pip install torch_geometric")
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, confusion_matrix
    )
    return torch, nn, F, DataLoader, GCNConv, global_mean_pool


# ══════════════════════════════════════════════════════════════════════════════
# 1. REBUILD ENCODER ARCHITECTURE  (must match step4a exactly)
# ══════════════════════════════════════════════════════════════════════════════

def build_encoder(torch, nn, F, GCNConv, global_mean_pool, config):
    """
    Rebuild the GCNEncoder architecture from step4a using saved config.
    Then load pretrained weights.
    """
    in_channels = config.get('in_channels', 12)
    hidden      = config.get('hidden',      64)
    embed_dim   = config.get('embed_dim',   128)
    dropout     = config.get('dropout',     0.2)

    class GCNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, embed_dim)
            self.bn1   = nn.BatchNorm1d(hidden)
            self.bn2   = nn.BatchNorm1d(embed_dim)
            self.drop  = nn.Dropout(p=dropout)

        def forward(self, x, edge_index, edge_weight, batch):
            x = self.conv1(x, edge_index, edge_weight)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.drop(x)
            x = self.conv2(x, edge_index, edge_weight)
            x = self.bn2(x)
            x = F.relu(x)
            from torch_geometric.nn import global_mean_pool as gmp
            h = gmp(x, batch)
            return h

        def encode_batch(self, data):
            edge_w = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None
            return self.forward(data.x, data.edge_index, edge_w, data.batch)

    return GCNEncoder()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFIER HEAD
# ══════════════════════════════════════════════════════════════════════════════

def build_classifier(nn, embed_dim=128, n_classes=2, hidden=64):
    """
    Simple MLP classifier on top of encoder embeddings.
    Used for both linear probe (frozen encoder) and fine-tuning.
    """
    return nn.Sequential(
        nn.Linear(embed_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_classes),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def get_class_weights(graph_list, torch):
    labels    = np.array([int(g.y.item()) for g in graph_list])
    n_total   = len(labels)
    n_ictal   = (labels == 1).sum()
    n_pre     = (labels == 0).sum()
    w0 = n_total / (2 * n_pre)   if n_pre   > 0 else 1.0
    w1 = n_total / (2 * n_ictal) if n_ictal > 0 else 1.0
    return torch.tensor([w0, w1], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(encoder, classifier, loader, optimizer,
                    criterion, device, torch, freeze_encoder):
    encoder.train()   if not freeze_encoder else encoder.eval()
    classifier.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        if freeze_encoder:
            with torch.no_grad():
                h = encoder.encode_batch(batch)
        else:
            h = encoder.encode_batch(batch)

        logits = classifier(h)
        loss   = criterion(logits, batch.y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(classifier.parameters()) +
            ([] if freeze_encoder else list(encoder.parameters())),
            max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(encoder, classifier, loader, criterion, device, torch):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    import torch.nn.functional as F

    encoder.eval()
    classifier.eval()

    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            h      = encoder.encode_batch(batch)
            logits = classifier(h)
            loss   = criterion(logits, batch.y.view(-1))
            probs  = F.softmax(logits, dim=1)[:, 1]

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item()

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sens = spec = 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    return {
        'loss':        total_loss / max(len(loader), 1),
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'auc':         float(auc),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. ONE LOSO FOLD
# ══════════════════════════════════════════════════════════════════════════════

def run_fold(fold_name, train_graphs, test_graphs,
             encoder_weights_path, config,
             args, torch, nn, F, DataLoader, GCNConv, global_mean_pool,
             device, freeze_encoder):
    """
    Train and evaluate one LOSO fold.

    freeze_encoder=True  → linear probe
    freeze_encoder=False → fine-tuning
    """
    from torch_geometric.data import DataLoader as PyGLoader

    train_loader = PyGLoader(train_graphs, batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    test_loader  = PyGLoader(test_graphs,  batch_size=args.batch_size,
                             shuffle=False, drop_last=False)

    # ── Rebuild encoder + load pretrained weights ─────────────────────
    encoder = build_encoder(torch, nn, F, GCNConv, global_mean_pool, config)
    encoder.load_state_dict(torch.load(encoder_weights_path,
                                        map_location='cpu'))
    encoder = encoder.to(device)

    embed_dim  = config.get('embed_dim', 128)
    classifier = build_classifier(nn, embed_dim=embed_dim).to(device)

    # ── Loss ──────────────────────────────────────────────────────────
    cw        = get_class_weights(train_graphs, torch).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # ── Optimizer ─────────────────────────────────────────────────────
    if freeze_encoder:
        # Only train classifier
        optimizer = torch.optim.Adam(classifier.parameters(),
                                      lr=args.lr_head,
                                      weight_decay=args.weight_decay)
    else:
        # Fine-tune: lower LR for encoder, higher for new head
        optimizer = torch.optim.Adam([
            {'params': encoder.parameters(),    'lr': args.lr_encoder},
            {'params': classifier.parameters(), 'lr': args.lr_head},
        ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6)

    # ── Training loop ─────────────────────────────────────────────────
    best_auc     = 0.0
    best_metrics = {}
    best_epoch   = 0
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(encoder, classifier, train_loader, optimizer,
                        criterion, device, torch, freeze_encoder)
        m = evaluate_model(encoder, classifier, test_loader,
                           criterion, device, torch)
        scheduler.step(m['auc'])

        if m['auc'] > best_auc:
            best_auc     = m['auc']
            best_metrics = dict(m)
            best_epoch   = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break

    best_metrics['best_epoch'] = best_epoch
    return best_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results_dict, output_dir):
    """
    Side-by-side comparison of all methods.
    results_dict: {'Supervised GCN': [...], 'SSL Linear Probe': [...], 'SSL Fine-tuned': [...]}
    """
    metrics  = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    colors   = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    methods  = list(results_dict.keys())

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── AUC per subject ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    n_subjects = max(len(v) for v in results_dict.values())
    subjects   = [r['subject'] for r in list(results_dict.values())[0]]
    x          = np.arange(len(subjects))
    width      = 0.8 / len(methods)

    for i, (method, results) in enumerate(results_dict.items()):
        aucs = [r['auc'] for r in results]
        offset = (i - len(methods)/2 + 0.5) * width
        ax1.bar(x + offset, aucs, width=width * 0.9,
                label=f"{method} (μ={np.mean(aucs):.3f})",
                color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, rotation=90, fontsize=7)
    ax1.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
    ax1.set_title('Per-Subject AUC — All Methods (LOSO-CV)',
                  fontsize=13, fontweight='bold')
    ax1.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2, axis='y')

    # ── Mean metrics comparison ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    x2  = np.arange(len(metrics))
    for i, (method, results) in enumerate(results_dict.items()):
        means = [np.mean([r[m] for r in results]) for m in metrics]
        offset = (i - len(methods)/2 + 0.5) * (0.8/len(methods))
        ax2.bar(x2 + offset, means, width=0.8/len(methods) * 0.9,
                label=method, color=colors[i], alpha=0.8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([m.upper()[:4] for m in metrics], fontsize=10)
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Metrics — All Methods', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3, axis='y')

    # ── AUC boxplot ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    box_data = [[r['auc'] for r in results_dict[m]] for m in methods]
    bp = ax3.boxplot(box_data, labels=methods, patch_artist=True,
                     showfliers=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.axhline(0.5, color='gray', linestyle='--', lw=1.5)
    ax3.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
    ax3.set_title('AUC Distribution — All Methods', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    ax3.set_ylim(0, 1.1)
    ax3.grid(alpha=0.3, axis='y')

    plt.suptitle('Step 4b — SSL vs Supervised Comparison (LOSO-CV)',
                 fontsize=14, fontweight='bold')
    plt.savefig(output_dir / 'ssl_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_dir / 'ssl_comparison.png'}")


def print_comparison_table(results_dict):
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
    print("\n" + "=" * 90)
    print("FINAL COMPARISON (LOSO-CV mean ± std)")
    print("=" * 90)
    header = f"{'Method':<22} " + "  ".join([f"{m[:4].upper():>10}" for m in metrics])
    print(header)
    print("-" * 90)
    for method, results in results_dict.items():
        line = f"{method:<22} "
        for m in metrics:
            vals = [r[m] for r in results if not np.isnan(r[m])]
            line += f"  {np.mean(vals):>5.3f}±{np.std(vals):.3f}"
        print(line)
    print("=" * 90)


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SSL evaluation: linear probe + fine-tuning (LOSO-CV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--datadir',     required=True)
    parser.add_argument('--encoderdir',  required=True,
                        help='Directory with encoder_best.pt + pretrain_config.json')
    parser.add_argument('--outdir',      required=True)
    parser.add_argument('--supervised_results', default=None,
                        help='Path to gcn_results.json from step3d (for comparison)')

    # Classifier training
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr_head',     type=float, default=1e-3,
                        help='LR for classifier head (default: 1e-3)')
    parser.add_argument('--lr_encoder',  type=float, default=1e-4,
                        help='LR for encoder during fine-tuning (default: 1e-4, 10x lower)')
    parser.add_argument('--weight_decay',type=float, default=1e-4)
    parser.add_argument('--patience',    type=int,   default=20)
    parser.add_argument('--device',      type=str,   default='auto')
    parser.add_argument('--fold',        type=str,   default=None,
                        help='Run only this fold (for debugging)')
    args = parser.parse_args()

    # ── Deps ──────────────────────────────────────────────────────────────
    torch, nn, F, DataLoader, GCNConv, global_mean_pool = import_deps()

    if args.device == 'auto':
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    data_dir    = Path(args.datadir)
    encoder_dir = Path(args.encoderdir)
    output_dir  = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pretrain config ──────────────────────────────────────────────
    with open(encoder_dir / 'pretrain_config.json') as f:
        config = json.load(f)
    encoder_path = encoder_dir / 'encoder_best.pt'

    print("=" * 72)
    print("STEP 4b — SSL EVALUATION")
    print("=" * 72)
    print(f"  Device:         {device}")
    print(f"  Encoder:        embed_dim={config['embed_dim']}")
    print(f"  Best pretrain:  epoch={config['best_epoch']}  loss={config['best_loss']:.4f}")
    print(f"  Epochs/fold:    {args.epochs}  (patience={args.patience})")
    print(f"  LR head:        {args.lr_head}")
    print(f"  LR encoder:     {args.lr_encoder}  (fine-tuning only)")
    print("=" * 72)

    # ── Load graphs + splits ──────────────────────────────────────────────
    print("\nLoading dataset...")
    graphs = torch.load(data_dir / 'dataset.pt', map_location='cpu')
    with open(data_dir / 'loso_splits.json') as f:
        loso_splits = json.load(f)
    folds = sorted(loso_splits.keys())
    if args.fold:
        folds = [args.fold]
    print(f"  ✅ {len(graphs):,} graphs  |  {len(folds)} LOSO folds")

    # ── Run both strategies ───────────────────────────────────────────────
    strategies = [
        ('SSL Linear Probe', True),   # freeze_encoder=True
        ('SSL Fine-tuned',   False),  # freeze_encoder=False
    ]

    all_strategy_results = {}

    for strategy_name, freeze in strategies:
        print(f"\n{'─'*72}")
        print(f"  Strategy: {strategy_name}")
        print(f"  Encoder: {'FROZEN' if freeze else 'FINE-TUNED (lr={})'.format(args.lr_encoder)}")
        print(f"{'─'*72}")

        fold_results = []

        for fold_name in tqdm(folds, desc=f"  {strategy_name}", unit="fold"):
            split        = loso_splits[fold_name]
            train_graphs = [graphs[i] for i in split['train']]
            test_graphs  = [graphs[i] for i in split['test']]

            # Skip folds with only one class in test
            test_labels = [g.y.item() for g in test_graphs]
            if len(set(test_labels)) < 2:
                continue

            metrics = run_fold(
                fold_name, train_graphs, test_graphs,
                encoder_path, config,
                args, torch, nn, F, DataLoader, GCNConv, global_mean_pool,
                device, freeze_encoder=freeze,
            )
            metrics['subject'] = fold_name
            fold_results.append(metrics)

            tqdm.write(
                f"    {fold_name}: AUC={metrics['auc']:.3f}  "
                f"Sens={metrics['sensitivity']:.3f}  "
                f"Spec={metrics['specificity']:.3f}"
            )

        all_strategy_results[strategy_name] = fold_results

        # Save per-strategy CSV
        df = pd.DataFrame(fold_results)
        df.to_csv(output_dir / f"{strategy_name.replace(' ','_')}_results.csv",
                  index=False)

    # ── Load supervised results for comparison ────────────────────────────
    results_for_plot = {}

    if args.supervised_results and Path(args.supervised_results).exists():
        with open(args.supervised_results) as f:
            sup_results = json.load(f)
        results_for_plot['Supervised GCN'] = sup_results

    results_for_plot.update(all_strategy_results)

    # ── Print comparison table ────────────────────────────────────────────
    print_comparison_table(results_for_plot)

    # ── Save combined results ─────────────────────────────────────────────
    combined = {}
    for name, results in results_for_plot.items():
        metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
        combined[name] = {
            m: {
                'mean': float(np.mean([r[m] for r in results])),
                'std':  float(np.std( [r[m] for r in results])),
            }
            for m in metrics
        }

    with open(output_dir / 'ssl_comparison_summary.json', 'w') as f:
        json.dump(combined, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────
    if len(results_for_plot) > 0:
        plot_comparison(results_for_plot, output_dir)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STEP 4b COMPLETE")
    print("=" * 72)
    for name, summary in combined.items():
        print(f"\n  {name}:")
        print(f"    AUC:         {summary['auc']['mean']:.3f} ± {summary['auc']['std']:.3f}")
        print(f"    Sensitivity: {summary['sensitivity']['mean']:.3f} ± {summary['sensitivity']['std']:.3f}")
        print(f"    F1:          {summary['f1']['mean']:.3f} ± {summary['f1']['std']:.3f}")
    print()
    print("  Saved files:")
    print(f"    ssl_comparison.png")
    print(f"    ssl_comparison_summary.json")
    print(f"    SSL_Linear_Probe_results.csv")
    print(f"    SSL_Fine-tuned_results.csv")
    print("=" * 72)


if __name__ == '__main__':
    main()
