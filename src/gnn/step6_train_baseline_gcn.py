"""
Step 6 - Baseline GCN with Leave-One-Subject-Out (LOSO) Cross-Validation
==========================================================================
Architecture (baseline — intentionally simple):
  Input: Node features (19 nodes, 8 features each)
  GCN layer 1: 8 → 64, ReLU
  GCN layer 2: 64 → 64, ReLU
  Global Mean Pooling → graph-level vector (64,)
  MLP: 64 → 32 → 2 (binary classification)

LOSO CV:
  - 34 subjects → 34 folds
  - Each fold: train on 33 subjects, test on 1 subject
  - Within training: hold out 1 random subject as validation (for early stopping)
  - No data leakage between subjects at any point

Early Stopping:
  - Monitors validation AUC (higher = better)
  - Stops when val AUC hasn't improved for `patience` epochs
  - Restores the best model weights before final test evaluation
  - Prevents overfitting to training subjects
  - Academically justified: Prechelt (1998), standard in deep learning

Why LOSO for small datasets?
  - Only 34 subjects, random split would give unreliable estimates
  - LOSO gives an honest generalisation estimate
  - Standard in EEG / clinical ML literature

Usage:
  python step6_train_baseline_gcn.py \
      --graphs_dir data/graphs \
      --output_dir results/baseline_gcn \
      --max_epochs 200 \
      --patience 20 \
      --lr 0.001 \
      --hidden 64
"""

import argparse
import json
import copy
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class BaselineGCN(nn.Module):
    """
    Simple 2-layer GCN for graph-level binary classification.

    Graph → GCN → GCN → GlobalMeanPool → MLP → logits
    """
    def __init__(self, in_channels=8, hidden_channels=64, n_classes=2,
                 dropout=0.3):
        super().__init__()

        # GCN layers
        self.conv1 = GCNConv(in_channels,     hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # MLP classifier head
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, edge_weight, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Layer 1
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Graph-level pooling
        x = global_mean_pool(x, batch)   # (batch_size, hidden_channels)

        # MLP
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x   # raw logits


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def compute_class_weights(graphs):
    """
    Compute inverse-frequency class weights to handle any remaining imbalance.
    """
    labels  = [g.y.item() for g in graphs]
    n_total = len(labels)
    n_pos   = sum(labels)
    n_neg   = n_total - n_pos

    w0 = n_total / (2.0 * n_neg + 1e-12)
    w1 = n_total / (2.0 * n_pos + 1e-12)

    return torch.tensor([w0, w1], dtype=torch.float)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss   = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_probs  = []
    all_labels = []

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # P(ictal)
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # AUC for early stopping monitor
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5   # fallback if only one class in small val set

    return y_true, y_pred, y_prob, auc


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    """
    Compute average loss on a data loader.
    Used for early stopping — loss is smoother than AUC on small val sets.
    """
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        loss   = criterion(logits, batch.y)
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
    else:
        sensitivity = specificity = float('nan')

    return {
        'accuracy':    float(acc),
        'f1':          float(f1),
        'auc':         float(auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
    }


def find_optimal_threshold(y_true, y_prob):
    """
    Find the classification threshold that maximises Youden's J statistic
    on a given set of labels and probabilities.

    Youden's J = Sensitivity + Specificity - 1
               = TPR - FPR

    This is equivalent to finding the point on the ROC curve that is
    furthest from the diagonal (random chance line). It balances
    sensitivity and specificity optimally without needing to specify
    which is more important.

    Why not use 0.5?
      The default threshold of 0.5 assumes the model's output is perfectly
      calibrated — that P(ictal) = 0.5 is truly the decision boundary.
      In practice, especially with class-weighted loss and small test sets,
      the model may output consistently high or low probabilities even when
      it correctly ranks ictal above control (high AUC). Using 0.5 in these
      cases gives Sens=0 despite AUC=1.0.

    Where does the threshold come from?
      The threshold is found on the VALIDATION set (3 subjects held out
      during training) and then applied to the TEST subject. This means
      the test subject never influences threshold selection — no leakage.

    Parameters
    ----------
    y_true : np.ndarray  ground truth labels (0/1)
    y_prob : np.ndarray  predicted probabilities for class 1

    Returns
    -------
    float  optimal threshold in [0, 1]
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr          # Youden's J at each threshold
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    except Exception:
        return 0.5   # safe fallback if val set has only one class




class EarlyStopping:
    """
    Monitors validation LOSS and stops training when it stops improving.

    Why loss instead of AUC?
      - AUC on a single small validation subject (6-10 epochs) is extremely
        noisy — it jumps to 1.0 immediately and gives a false stop signal.
      - Loss is a continuous signal that changes smoothly every epoch,
        making it a much more reliable stopping criterion on small datasets.
      - We still REPORT AUC as the evaluation metric — loss is only
        used internally to decide when to stop.

    After `patience` epochs without improvement, training stops and
    the best model weights are restored automatically.
    """
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = np.inf    # lower is better
        self.best_epoch  = 0
        self.counter     = 0
        self.best_state  = None

    def step(self, val_loss, model, epoch):
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        """Load the best weights back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================================
# LOSO CROSS-VALIDATION
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars to Python native types for JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_loso_cv(graphs_dir, output_dir, n_subjects=34,
                max_epochs=200, patience=20,
                lr=1e-3, hidden=64,
                batch_size=32, dropout=0.3):
    """
    Leave-One-Subject-Out cross-validation with early stopping.

    Validation strategy (nested LOSO):
      For each fold (test = subject T):
        - Training pool = all subjects except T
        - Validation    = 1 randomly chosen subject from training pool
        - Train set     = remaining 32 subjects
        - Early stopping monitors val AUC on the validation subject
        - Final evaluation on subject T (never seen during training or val)
    """
    graphs_dir = Path(graphs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Load per-subject graph lists ───────────────────────────────────────
    subject_graphs = {}
    for subj_id in range(1, n_subjects + 1):
        path = graphs_dir / f"subject_{subj_id:02d}_graphs.pt"
        if path.exists():
            g = torch.load(path, weights_only=False)
            if len(g) > 0:
                subject_graphs[subj_id] = g

    available = sorted(subject_graphs.keys())
    print(f"Loaded {len(available)} subjects")
    print(f"Total graphs: {sum(len(v) for v in subject_graphs.values())}")

    if len(available) < 3:
        print("Need at least 3 subjects for LOSO CV")
        return

    # ── LOSO folds ────────────────────────────────────────────────────────
    fold_results      = []
    all_true, all_pred, all_prob = [], [], []
    train_loss_curves = []
    train_auc_curves  = []   # NEW: for overfitting diagnosis
    val_auc_curves    = []
    stopped_epochs    = []

    print("\n" + "=" * 70)
    print("LOSO CROSS-VALIDATION  (with early stopping)")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)

    for fold_idx, test_subj in enumerate(available):

        # ── Split subjects ─────────────────────────────────────────────────
        train_pool = [s for s in available if s != test_subj]

        # Use 3 validation subjects instead of 1.
        # Why 3? A single subject (6-10 epochs) gives noisy val loss.
        # 3 subjects gives ~25-30 validation epochs which is enough
        # for a stable loss signal without sacrificing too much training data.
        n_val_subjects = min(3, len(train_pool) - 1)
        val_subjects   = list(rng.choice(train_pool, size=n_val_subjects,
                                         replace=False))
        train_subjects = [s for s in train_pool if s not in val_subjects]

        # Build graph lists
        train_graphs = []
        for s in train_subjects:
            train_graphs.extend(subject_graphs[s])
        val_graphs = []
        for s in val_subjects:
            val_graphs.extend(subject_graphs[s])
        test_graphs = subject_graphs[test_subj]

        # ── Class weights from training data only ──────────────────────────
        class_weights = compute_class_weights(train_graphs).to(device)

        # ── DataLoaders ────────────────────────────────────────────────────
        train_loader = DataLoader(train_graphs, batch_size=batch_size,
                                  shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_graphs,   batch_size=batch_size,
                                  shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_graphs,  batch_size=batch_size,
                                  shuffle=False, drop_last=False)

        # ── Model, optimiser, scheduler ────────────────────────────────────
        model     = BaselineGCN(in_channels=8, hidden_channels=hidden,
                                n_classes=2, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ReduceLROnPlateau monitors val loss (mode='min')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=10, min_lr=1e-5
        )

        # Early stopping monitors val loss (lower = better)
        early_stop = EarlyStopping(patience=patience, min_delta=1e-4)

        # ── Training loop ──────────────────────────────────────────────────
        train_losses  = []
        train_aucs    = []
        val_aucs      = []
        stopped_at    = max_epochs

        for ep in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader,
                                         optimizer, criterion, device)
            val_loss                   = evaluate_loss(model, val_loader,
                                                       criterion, device)
            _, _, _, val_auc           = evaluate(model, val_loader,   device)
            _, _, _, train_auc         = evaluate(model, train_loader, device)

            # Scheduler and early stopping both use val LOSS
            scheduler.step(val_loss)
            train_losses.append(train_loss)
            train_aucs.append(float(train_auc))
            val_aucs.append(float(val_auc))

            if early_stop.step(val_loss, model, ep):
                stopped_at = ep + 1
                break

        # Restore best weights (lowest val loss) before final evaluation
        early_stop.restore_best(model)

        train_loss_curves.append(train_losses)
        train_auc_curves.append(train_aucs)
        val_auc_curves.append(val_aucs)
        stopped_epochs.append(stopped_at)

        # ── Final evaluation on held-out test subject ──────────────────────
        # Step 1: get training AUC for overfitting gap
        y_true_tr, _, y_prob_tr, train_auc_final = evaluate(model, train_loader, device)

        # Step 2: find optimal threshold from VALIDATION set
        # (val set was used for early stopping only — threshold selection
        #  is a separate, legitimate use of the same held-out subjects)
        y_true_val, _, y_prob_val, _ = evaluate(model, val_loader, device)
        optimal_threshold = find_optimal_threshold(y_true_val, y_prob_val)

        # Step 3: evaluate on TEST subject with BOTH thresholds
        y_true, _, y_prob, _ = evaluate(model, test_loader, device)

        # Fixed threshold (0.5) — standard approach
        y_pred_fixed = (y_prob >= 0.5).astype(int)
        metrics_fixed = compute_metrics(y_true, y_pred_fixed, y_prob)

        # Optimal threshold from val set — improves calibration
        y_pred_opt = (y_prob >= optimal_threshold).astype(int)
        metrics_opt = compute_metrics(y_true, y_pred_opt, y_prob)

        # Primary metrics = optimal threshold (better F1/sens/spec)
        # AUC is threshold-independent so it's the same either way
        metrics = metrics_opt.copy()
        metrics['test_subject']      = int(test_subj)
        metrics['val_subjects']      = [int(s) for s in val_subjects]
        metrics['n_test']            = int(len(test_graphs))
        metrics['stopped_epoch']     = int(stopped_at)
        metrics['best_val_loss']     = float(early_stop.best_loss)
        metrics['final_train_auc']   = float(train_auc_final)
        metrics['overfit_gap']       = float(train_auc_final - metrics['auc'])
        metrics['optimal_threshold'] = float(optimal_threshold)
        # Also save fixed-threshold metrics for comparison
        metrics['f1_fixed']          = float(metrics_fixed['f1'])
        metrics['sensitivity_fixed'] = float(metrics_fixed['sensitivity'])
        metrics['specificity_fixed'] = float(metrics_fixed['specificity'])
        metrics['accuracy_fixed']    = float(metrics_fixed['accuracy'])

        fold_results.append(metrics)
        all_true.extend(y_true)
        all_pred.extend(y_pred_opt)   # pooled uses optimal threshold
        all_prob.extend(y_prob)

        n_ctrl  = sum(1 for g in test_graphs if g.y.item() == 0)
        n_ict   = sum(1 for g in test_graphs if g.y.item() == 1)
        gap     = metrics['overfit_gap']
        gap_str = f"+{gap:.3f}" if gap >= 0 else f"{gap:.3f}"
        thr_str = f"{optimal_threshold:.2f}"
        print(f"  Fold {fold_idx+1:02d} | Test S{test_subj:02d} "
              f"(ctrl={n_ctrl}, ictal={n_ict}) | "
              f"Stopped@ep{stopped_at:03d} | Thr={thr_str} | "
              f"AUC={metrics['auc']:.3f}  "
              f"F1={metrics['f1']:.3f}(was {metrics_fixed['f1']:.3f})  "
              f"Sens={metrics['sensitivity']:.3f}  "
              f"Spec={metrics['specificity']:.3f}  "
              f"Gap={gap_str}")

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    micro = compute_metrics(
        np.array(all_true), np.array(all_pred), np.array(all_prob)
    )

    macro = {}
    for key in ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']:
        vals = [r[key] for r in fold_results if not np.isnan(r[key])]
        macro[f'{key}_mean'] = float(np.mean(vals))
        macro[f'{key}_std']  = float(np.std(vals))

    # Fixed threshold comparison
    macro_fixed = {}
    for key in ['accuracy', 'f1', 'sensitivity', 'specificity']:
        fixed_key = f'{key}_fixed'
        vals = [r[fixed_key] for r in fold_results if not np.isnan(r[fixed_key])]
        macro_fixed[f'{key}_mean'] = float(np.mean(vals))
        macro_fixed[f'{key}_std']  = float(np.std(vals))

    avg_threshold = float(np.mean([r['optimal_threshold'] for r in fold_results]))

    print(f"\nEarly stopping: avg stopped at epoch "
          f"{np.mean(stopped_epochs):.1f} ± {np.std(stopped_epochs):.1f}")
    print(f"Avg optimal threshold: {avg_threshold:.3f}  (vs fixed 0.500)")

    print(f"\n{'─'*70}")
    print(f"{'Metric':<18} {'Optimal Thr (val set)':>22} {'Fixed Thr (0.5)':>18}  {'Improvement':>12}")
    print(f"{'─'*70}")
    for key in ['accuracy', 'f1', 'sensitivity', 'specificity']:
        opt_m = macro[f'{key}_mean']
        opt_s = macro[f'{key}_std']
        fix_m = macro_fixed[f'{key}_mean']
        fix_s = macro_fixed[f'{key}_std']
        diff  = opt_m - fix_m
        sign  = '+' if diff >= 0 else ''
        print(f"  {key:<16} {opt_m:.3f} ± {opt_s:.3f}          "
              f"{fix_m:.3f} ± {fix_s:.3f}    {sign}{diff:.3f}")
    print(f"  {'auc':<16} {macro['auc_mean']:.3f} ± {macro['auc_std']:.3f}          "
          f"(threshold-independent)")
    print(f"{'─'*70}")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        'model':       'BaselineGCN',
        'n_folds':     len(available),
        'hyperparams': {
            'max_epochs': max_epochs,
            'patience':   patience,
            'lr':         lr,
            'hidden':     hidden,
            'batch_size': batch_size,
            'dropout':    dropout,
        },
        'threshold_method': 'youden_j_on_val_set',
        'avg_optimal_threshold': avg_threshold,
        'early_stopping': {
            'monitor':           'val_loss',
            'n_val_subjects':    3,
            'avg_stopped_epoch': float(np.mean(stopped_epochs)),
            'std_stopped_epoch': float(np.std(stopped_epochs)),
            'stopped_epochs':    stopped_epochs,
        },
        'macro_metrics':       macro,
        'macro_metrics_fixed': macro_fixed,
        'micro_metrics':       micro,
        'fold_results':        fold_results,
    }
    with open(output_dir / 'loso_results.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_confusion_matrix(np.array(all_true), np.array(all_pred), output_dir)
    _plot_metric_boxplots(fold_results, output_dir)
    _plot_training_curves(train_loss_curves, train_auc_curves,
                          val_auc_curves, stopped_epochs, output_dir)
    _plot_overfit_gap(fold_results, output_dir)

    print(f"\n✅ Results saved to {output_dir}")
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', 'Ictal'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Control', 'Ictal'])
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True',      fontweight='bold')
    ax.set_title('Confusion Matrix (all folds pooled)', fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=200)
    plt.close()


def _plot_metric_boxplots(fold_results, output_dir):
    metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
    data    = [[r[m] for r in fold_results
                if not np.isnan(r[m])] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(metrics) + 1))
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel('Score')
    ax.set_title('LOSO Cross-Validation — Per-Subject Metrics', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_boxplots.png', dpi=200)
    plt.close()


def _plot_training_curves(loss_curves, train_auc_curves, val_auc_curves,
                           stopped_epochs, output_dir):
    """
    Three-panel plot:
      Left:   Training loss
      Middle: Train AUC vs Val AUC (the overfitting gap)
      Right:  Val AUC with early stop markers

    The middle panel is the key overfitting diagnostic.
    A large and growing gap between train AUC and val AUC = overfitting.
    Curves that stay close together = good generalisation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # ── Panel 1: Training loss ─────────────────────────────────────────────
    ax = axes[0]
    for curve in loss_curves:
        ax.plot(curve, alpha=0.2, linewidth=1, color='steelblue')
    min_len   = min(len(c) for c in loss_curves)
    mean_loss = np.mean([c[:min_len] for c in loss_curves], axis=0)
    ax.plot(mean_loss, color='navy', linewidth=2.5, label='Mean train loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss (all folds)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel 2: Train AUC vs Val AUC ─────────────────────────────────────
    # This is the OVERFITTING DIAGNOSTIC
    ax = axes[1]
    for t_curve, v_curve in zip(train_auc_curves, val_auc_curves):
        n = min(len(t_curve), len(v_curve))
        ax.plot(t_curve[:n], alpha=0.2, linewidth=1, color='green')
        ax.plot(v_curve[:n], alpha=0.2, linewidth=1, color='orange')

    min_t = min(len(c) for c in train_auc_curves)
    min_v = min(len(c) for c in val_auc_curves)
    min_len = min(min_t, min_v)

    mean_train_auc = np.mean([c[:min_len] for c in train_auc_curves], axis=0)
    mean_val_auc   = np.mean([c[:min_len] for c in val_auc_curves],   axis=0)

    ax.plot(mean_train_auc, color='green',  linewidth=2.5,
            label='Mean train AUC')
    ax.plot(mean_val_auc,   color='orange', linewidth=2.5,
            label='Mean val AUC')

    # Shade the gap between train and val AUC
    x = np.arange(min_len)
    ax.fill_between(x, mean_val_auc, mean_train_auc,
                    alpha=0.25, color='red',
                    label='Overfitting gap')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('Train vs Val AUC — Overfitting Diagnostic\n'
                 '(large red gap = overfitting)', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel 3: Val AUC with early stop markers ───────────────────────────
    ax = axes[2]
    for curve, ep in zip(val_auc_curves, stopped_epochs):
        ax.plot(curve, alpha=0.2, linewidth=1, color='darkorange')
        ax.axvline(ep - 1, alpha=0.15, linewidth=1,
                   color='red', linestyle='--')
    ax.plot(mean_val_auc, color='darkorange', linewidth=2.5,
            label='Mean val AUC')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='Chance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation AUC')
    ax.set_title(f'Val AUC — Early Stop Points\n'
                 f'(avg stop @ epoch {np.mean(stopped_epochs):.0f})',
                 fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=200)
    plt.close()


def _plot_overfit_gap(fold_results, output_dir):
    """
    Bar chart showing the train AUC vs test AUC gap per subject.

    How to read this:
      - Blue bar  = train AUC (what the model achieved on its own training data)
      - Orange bar = test AUC  (what it achieved on the unseen subject)
      - Gap = blue - orange: small gap = good generalisation
        - Gap < 0.1 → no significant overfitting
        - Gap 0.1-0.3 → mild overfitting
        - Gap > 0.3 → severe overfitting for that fold
    """
    subjects    = [r['test_subject']    for r in fold_results]
    train_aucs  = [r['final_train_auc'] for r in fold_results]
    test_aucs   = [r['auc']             for r in fold_results]
    gaps        = [r['overfit_gap']     for r in fold_results]

    x     = np.arange(len(subjects))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # ── Panel 1: Train vs Test AUC bars ───────────────────────────────────
    ax = axes[0]
    ax.bar(x - width/2, train_aucs, width, label='Train AUC',
           color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, test_aucs,  width, label='Test AUC',
           color='darkorange', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects], rotation=45, ha='right')
    ax.set_ylabel('AUC')
    ax.set_ylim(0, 1.1)
    ax.set_title('Train AUC vs Test AUC per Subject\n'
                 '(large gap between blue and orange = overfitting)',
                 fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='Chance')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Annotate problem folds
    for i, (ta, tea, gap) in enumerate(zip(train_aucs, test_aucs, gaps)):
        if gap > 0.3:
            ax.text(i, max(ta, tea) + 0.03, '⚠', ha='center',
                    fontsize=14, color='red')

    # ── Panel 2: Gap per subject ───────────────────────────────────────────
    ax = axes[1]
    colors = ['red' if g > 0.3 else 'orange' if g > 0.1 else 'green'
              for g in gaps]
    ax.bar(x, gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(0.0,  color='black', linewidth=1)
    ax.axhline(0.1,  color='orange', linestyle='--', linewidth=1.5,
               label='Mild overfitting threshold (0.1)')
    ax.axhline(0.3,  color='red',    linestyle='--', linewidth=1.5,
               label='Severe overfitting threshold (0.3)')

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects], rotation=45, ha='right')
    ax.set_ylabel('Train AUC − Test AUC  (Gap)')
    ax.set_title('Overfitting Gap per Subject\n'
                 'Green < 0.1 (good) | Orange 0.1-0.3 (mild) | Red > 0.3 (severe)',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Summary stats in title
    mean_gap = np.mean(gaps)
    severe   = sum(1 for g in gaps if g > 0.3)
    mild     = sum(1 for g in gaps if 0.1 < g <= 0.3)
    good     = sum(1 for g in gaps if g <= 0.1)
    fig.suptitle(
        f'Overfitting Analysis — Mean gap: {mean_gap:.3f} | '
        f'Good: {good} | Mild: {mild} | Severe: {severe}',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_analysis.png', dpi=200,
                bbox_inches='tight')
    plt.close()
    print("✅ Overfitting analysis plot saved")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline GCN with LOSO CV")
    parser.add_argument("--graphs_dir",  required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--max_epochs",  type=int,   default=200,
                        help="Maximum training epochs per fold (default 200)")
    parser.add_argument("--patience",    type=int,   default=20,
                        help="Early stopping patience in epochs (default 20)")
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--hidden",      type=int,   default=64)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--n_subjects",  type=int,   default=34)
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 6 — BASELINE GCN (Leave-One-Subject-Out CV)")
    print("=" * 70)
    print(f"Architecture  : GCN(8→{args.hidden}→{args.hidden}) + MLP")
    print(f"Max epochs    : {args.max_epochs}")
    print(f"Early stopping: patience={args.patience} (monitor: val AUC)")
    print(f"LR            : {args.lr}  (ReduceLROnPlateau, halves on plateau)")
    print(f"Dropout       : {args.dropout}")
    print(f"Batch size    : {args.batch_size}")
    print("=" * 70)

    run_loso_cv(
        graphs_dir  = args.graphs_dir,
        output_dir  = args.output_dir,
        n_subjects  = args.n_subjects,
        max_epochs  = args.max_epochs,
        patience    = args.patience,
        lr          = args.lr,
        hidden      = args.hidden,
        batch_size  = args.batch_size,
        dropout     = args.dropout,
    )


if __name__ == "__main__":
    main()