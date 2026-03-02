"""
Step 7 — GCN (Graph Convolutional Network) with LOPO Cross-Validation
======================================================================
Architecture
------------
  - Nodes  : 19 EEG channels
  - Node features : (n_features,) per channel  from step5 node_features.npy
  - Edges  : directed DTF integrated-band connectivity (19x19 matrix)
             thresholded at DTF > threshold (default 0.1) to keep top connections
  - Edge weights : DTF value for each surviving edge
  - GCN layers : 2x GCNConv → mean pooling → 2x Linear → binary classification

Graph construction per epoch
-----------------------------
  1. Load node features  : (19, n_node_feat)
  2. Load DTF matrix     : (19, 19)  from _graphs.npz  key='dtf_integrated'
  3. Threshold + build edge_index  (2, n_edges) and edge_attr (n_edges,)
  4. Label               : 0=pre-ictal, 1=ictal
  → PyG Data object

Training details
----------------
  - Loss     : BCEWithLogitsLoss  with pos_weight to handle class imbalance
  - Optimizer: Adam  lr=1e-3, weight_decay=1e-4
  - Scheduler: ReduceLROnPlateau  (patience=10)
  - Early stopping: patience=20  on val AUROC
  - Epochs   : max 150
  - Batch    : full-batch DataLoader (all graphs in one batch per step)

Output files
------------
  results_per_fold.json    — per-fold metrics
  results_summary.json     — mean +- std across folds
  training_curves.png      — loss + AUROC per epoch for each fold
  roc_curves.png           — ROC curves per fold
  gcn_results.txt          — thesis-ready table

Installation (run once on your machine)
-----------------------------------------
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install torch_geometric
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
      -f https://data.pyg.org/whl/torch-2.x.x+cu118.html

  For CPU only:
  pip install torch torchvision torchaudio
  pip install torch_geometric

Usage
-----
  python step7_gcn.py \
      --features_dir   F:\\...\\node_features \
      --connectivity_dir F:\\...\\connectivity \
      --splits         F:\\...\\splits\\splits.json \
      --output_dir     F:\\...\\gcn_results

  Optional flags:
      --dtf_threshold  0.1    (edge pruning threshold, default 0.1)
      --hidden_dim     64     (GCN hidden dimension, default 64)
      --epochs         150    (max training epochs, default 150)
      --lr             1e-3   (learning rate, default 1e-3)
      --device         cuda   (or cpu, default: auto-detect)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data   import Data, DataLoader
from torch_geometric.nn     import GCNConv, global_mean_pool

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, \
                            precision_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
N_CHANNELS = 19

CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6","O1","O2",
]


# ─────────────────────────────────────────────────────────────────────────────
# GCN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SeizureGCN(nn.Module):
    """
    2-layer GCN for seizure detection.

    Architecture:
      GCNConv(in, hidden) → ReLU → Dropout
      GCNConv(hidden, hidden) → ReLU → Dropout
      global_mean_pool  →  (batch_size, hidden)
      Linear(hidden, hidden//2) → ReLU → Dropout
      Linear(hidden//2, 1)  →  logit
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        self.conv1   = GCNConv(in_channels, hidden_dim)
        self.conv2   = GCNConv(hidden_dim,  hidden_dim)

        self.fc1     = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2     = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.bn1     = nn.BatchNorm1d(hidden_dim)
        self.bn2     = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Graph-level readout
        x = global_mean_pool(x, batch)   # (batch_size, hidden_dim)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                  # (batch_size, 1)

        return x.squeeze(1)              # (batch_size,)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(node_feat: np.ndarray,
                dtf_matrix: np.ndarray,
                label: int,
                threshold: float = 0.1) -> Data:
    """
    Build a PyG Data object for one epoch.

    Parameters
    ----------
    node_feat  : (19, n_feat) — node feature matrix
    dtf_matrix : (19, 19)    — DTF connectivity (diagonal already 0)
    label      : int          — 0 or 1
    threshold  : float        — minimum DTF value to keep an edge

    Returns
    -------
    PyG Data object
    """
    # Node features
    x = torch.tensor(node_feat, dtype=torch.float)   # (19, n_feat)

    # Build edge list from DTF matrix
    # DTF[i,j] = j -> i  (source j, sink i)
    src, dst, weights = [], [], []
    for i in range(N_CHANNELS):
        for j in range(N_CHANNELS):
            if i != j and dtf_matrix[i, j] > threshold:
                src.append(j)       # source
                dst.append(i)       # sink
                weights.append(dtf_matrix[i, j])

    if len(src) == 0:
        # Fallback: fully connected with uniform weights if all below threshold
        for i in range(N_CHANNELS):
            for j in range(N_CHANNELS):
                if i != j:
                    src.append(j)
                    dst.append(i)
                    weights.append(1.0 / (N_CHANNELS - 1))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def load_graphs_for_subjects(subject_ids: list,
                              features_dir: Path,
                              connectivity_dir: Path,
                              selected_epochs: dict,
                              threshold: float,
                              scaler: StandardScaler = None,
                              fit_scaler: bool = False
                              ) -> tuple:
    """
    Load and build graph list for a set of subjects.

    Returns
    -------
    graphs   : list of PyG Data objects
    scaler   : fitted StandardScaler (if fit_scaler=True)
    flat_dim : int  (n_channels * n_node_features)
    """
    all_feats   = []   # collect raw features first for scaling
    all_dtf     = []
    all_labels  = []

    for sid in subject_ids:
        sid_str = str(sid)
        info    = selected_epochs.get(sid_str)
        if info is None or info.get("excluded", False):
            continue

        feat_file = features_dir / f"subject_{sid:02d}_node_features.npy"
        npz_file  = connectivity_dir / f"subject_{sid:02d}_graphs.npz"

        if not feat_file.exists() or not npz_file.exists():
            continue

        node_feats_all = np.load(feat_file)    # (n_sel_epochs, 19, n_feat)
        npz            = np.load(npz_file)
        dtf_all        = npz["dtf_integrated"] # (n_conn_epochs, 19, 19)
        conn_labels    = npz["labels"]         # (n_conn_epochs,)
        conn_indices   = npz["indices"]        # original epoch indices

        # Match selected epochs to connectivity epochs via original indices
        ictal_idx = info["ictal_indices"]
        pre_idx   = info["pre_ictal_indices"]
        selected_orig_idx = sorted(
            [(i, 1) for i in ictal_idx] + [(i, 0) for i in pre_idx],
            key=lambda x: x[0]
        )

        # Build lookup: original_epoch_idx -> position in npz
        conn_idx_map = {int(orig): pos
                        for pos, orig in enumerate(conn_indices)}

        # Also build lookup for node_features (already in selected order)
        # node_feats_all is indexed 0..n_selected-1 in chronological order
        # We need to map original epoch index -> position in node_feats_all
        feat_orig_indices = []
        for (orig_idx, lbl) in selected_orig_idx:
            feat_orig_indices.append(orig_idx)

        feat_pos_map = {orig: pos for pos, orig in enumerate(feat_orig_indices)}

        for orig_idx, label in selected_orig_idx:
            if orig_idx not in conn_idx_map:
                continue   # epoch was dropped by connectivity step (bad VAR)
            if orig_idx not in feat_pos_map:
                continue

            conn_pos = conn_idx_map[orig_idx]
            feat_pos = feat_pos_map[orig_idx]

            all_feats.append(node_feats_all[feat_pos])   # (19, n_feat)
            all_dtf.append(dtf_all[conn_pos])             # (19, 19)
            all_labels.append(label)

    if len(all_feats) == 0:
        return [], scaler, None

    all_feats_arr = np.array(all_feats)   # (N, 19, n_feat)
    N, C, F = all_feats_arr.shape
    flat_dim = C * F

    # Fit or apply scaler on flattened features
    flat = all_feats_arr.reshape(N, flat_dim)
    if fit_scaler:
        scaler = StandardScaler()
        flat   = scaler.fit_transform(flat)
    elif scaler is not None:
        flat = scaler.transform(flat)

    all_feats_arr = flat.reshape(N, C, F)

    # Build graph list
    graphs = []
    for i in range(N):
        g = build_graph(
            all_feats_arr[i],
            all_dtf[i],
            all_labels[i],
            threshold=threshold
        )
        graphs.append(g)

    return graphs, scaler, flat_dim


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        loss   = criterion(logits, batch.y)
        total_loss += loss.item() * batch.num_graphs

        probs  = torch.sigmoid(logits).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels)

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= 0.5).astype(int)

    avg_loss = total_loss / len(loader.dataset)

    metrics = {
        "loss"      : float(avg_loss),
        "auroc"     : float(roc_auc_score(all_labels, all_probs))
                      if len(np.unique(all_labels)) > 1 else 0.5,
        "f1_macro"  : float(f1_score(all_labels, all_preds,
                                     average="macro", zero_division=0)),
        "f1_ictal"  : float(f1_score(all_labels, all_preds,
                                     pos_label=1, zero_division=0)),
        "accuracy"  : float(accuracy_score(all_labels, all_preds)),
        "precision" : float(precision_score(all_labels, all_preds,
                                            pos_label=1, zero_division=0)),
        "recall"    : float(recall_score(all_labels, all_preds,
                                         pos_label=1, zero_division=0)),
        "n_total"   : int(len(all_labels)),
        "n_ictal"   : int(np.sum(all_labels == 1)),
    }

    return metrics, all_probs, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(fold_histories, output_dir):
    """Plot loss + AUROC training curves for all folds."""
    n_folds = len(fold_histories)
    fig, axes = plt.subplots(n_folds, 2,
                             figsize=(12, 3.5 * n_folds))
    if n_folds == 1:
        axes = axes[np.newaxis, :]

    for fi, hist in enumerate(fold_histories):
        epochs_range = range(1, len(hist["train_loss"]) + 1)

        ax_loss = axes[fi, 0]
        ax_loss.plot(epochs_range, hist["train_loss"],
                     label="Train", color="steelblue")
        ax_loss.plot(epochs_range, hist["val_loss"],
                     label="Val",   color="orange")
        ax_loss.set_title(f"Fold {hist['fold']} — Loss", fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("BCE Loss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)

        ax_auc = axes[fi, 1]
        ax_auc.plot(epochs_range, hist["val_auroc"],
                    label="Val AUROC", color="green")
        best_ep = np.argmax(hist["val_auroc"]) + 1
        ax_auc.axvline(best_ep, color="red", linestyle="--",
                       alpha=0.7, label=f"Best ep={best_ep}")
        ax_auc.set_title(f"Fold {hist['fold']} — Val AUROC", fontweight="bold")
        ax_auc.set_xlabel("Epoch")
        ax_auc.set_ylabel("AUROC")
        ax_auc.set_ylim([0, 1])
        ax_auc.legend()
        ax_auc.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_roc_curves(all_roc_data, output_dir):
    """Plot ROC curves for all folds."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(all_roc_data)))
    aucs    = []

    for fi, (fpr, tpr, auc_val, fold_label) in enumerate(all_roc_data):
        ax.plot(fpr, tpr, color=colors[fi], linewidth=1.8, alpha=0.8,
                label=f"{fold_label}  AUC={auc_val:.3f}")
        aucs.append(auc_val)

    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(
        f"GCN — ROC Curves (LOPO)\n"
        f"Mean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    path = output_dir / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# THESIS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def save_thesis_table(results_summary, args, output_dir):
    metric_keys = ["auroc", "f1_macro", "f1_ictal",
                   "accuracy", "precision", "recall"]

    lines = [
        "GCN RESULTS — LOPO CROSS-VALIDATION",
        "=" * 72,
        "Model: 2-layer GCN (SeizureGCN)",
        f"Hidden dim: {args.hidden_dim}  |  Dropout: 0.3  |  "
        f"Max epochs: {args.epochs}",
        f"Edge threshold (DTF): {args.dtf_threshold}  |  "
        f"LR: {args.lr}  |  Early stopping: patience=20",
        "=" * 72,
        "",
        f"{'Metric':<14} {'Mean':>8}  {'Std':>8}  {'Per-fold values'}",
        "-" * 72,
    ]

    for key in metric_keys:
        m   = results_summary[key]["mean"]
        s   = results_summary[key]["std"]
        vals = [f"{v:.3f}" for v in results_summary[key]["values"]]
        lines.append(
            f"{key:<14} {m:>8.3f}  {s:>8.3f}  {', '.join(vals)}"
        )

    lines += [
        "-" * 72,
        "",
        "Notes:",
        "  - Graph nodes: 19 EEG channels (10-20 system)",
        "  - Node features: band power + Hjorth per channel",
        "  - Edges: DTF integrated band, thresholded",
        "  - Split: Leave-One-Patient-Out (7 folds)",
        "  - Normalisation: StandardScaler on train subjects only",
    ]

    text = "\n".join(lines)
    path = output_dir / "gcn_results.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  Saved: {path.name}")
    print()
    print(text)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GCN for seizure detection — LOPO cross-validation"
    )
    parser.add_argument("--features_dir",     required=True)
    parser.add_argument("--connectivity_dir", required=True)
    parser.add_argument("--splits",           required=True)
    parser.add_argument("--selected_epochs",  required=True,
                        help="Path to selected_epochs.json (from step4)")
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--dtf_threshold", type=float, default=0.1)
    parser.add_argument("--hidden_dim",    type=int,   default=64)
    parser.add_argument("--epochs",        type=int,   default=150)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--device",        type=str,   default="auto")
    args = parser.parse_args()

    # Setup
    features_dir     = Path(args.features_dir)
    connectivity_dir = Path(args.connectivity_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load splits and selected epochs
    with open(args.splits) as f:
        splits = json.load(f)
    with open(args.selected_epochs) as f:
        selected_epochs = json.load(f)

    folds = splits["folds"]

    print("=" * 70)
    print("STEP 7 — GCN: GRAPH CONVOLUTIONAL NETWORK")
    print("=" * 70)
    print(f"  Device          : {device}")
    print(f"  Features dir    : {features_dir}")
    print(f"  Connectivity dir: {connectivity_dir}")
    print(f"  Hidden dim      : {args.hidden_dim}")
    print(f"  DTF threshold   : {args.dtf_threshold}")
    print(f"  Max epochs      : {args.epochs}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  LOPO folds      : {len(folds)}")
    print()

    all_fold_results = []
    fold_histories   = []
    all_roc_data     = []

    for fold in folds:
        fold_num   = fold["fold"]
        test_pat   = fold["test_patient"]
        val_pat    = fold["val_patient"]
        train_subs = fold["train_subjects"]
        val_subs   = fold["val_subjects"]
        test_subs  = fold["test_subjects"]

        print(f"{'='*70}")
        print(f"FOLD {fold_num}/{len(folds)}  |  "
              f"Test: PAT_{test_pat}  Val: PAT_{val_pat}  "
              f"Train: {len(train_subs)} subjects")
        print(f"{'='*70}")

        # ── Build graph datasets ─────────────────────────────────────────────
        print("  Building graphs...")

        train_graphs, scaler, flat_dim = load_graphs_for_subjects(
            train_subs, features_dir, connectivity_dir,
            selected_epochs, args.dtf_threshold,
            fit_scaler=True
        )
        val_graphs, _, _ = load_graphs_for_subjects(
            val_subs, features_dir, connectivity_dir,
            selected_epochs, args.dtf_threshold,
            scaler=scaler
        )
        test_graphs, _, _ = load_graphs_for_subjects(
            test_subs, features_dir, connectivity_dir,
            selected_epochs, args.dtf_threshold,
            scaler=scaler
        )

        if not train_graphs or not test_graphs:
            print(f"  [SKIP] Not enough data for fold {fold_num}")
            continue

        # Detect node feature dimension from first graph
        n_node_feat = train_graphs[0].x.shape[1]

        n_train_ict = sum(int(g.y.item()) for g in train_graphs)
        n_val_ict   = sum(int(g.y.item()) for g in val_graphs)
        n_test_ict  = sum(int(g.y.item()) for g in test_graphs)

        print(f"  Train: {len(train_graphs)} graphs "
              f"({n_train_ict} ictal / {len(train_graphs)-n_train_ict} pre)")
        print(f"  Val:   {len(val_graphs)} graphs "
              f"({n_val_ict} ictal / {len(val_graphs)-n_val_ict} pre)")
        print(f"  Test:  {len(test_graphs)} graphs "
              f"({n_test_ict} ictal / {len(test_graphs)-n_test_ict} pre)")
        print(f"  Node features: {n_node_feat}  |  "
              f"Edges per graph: {train_graphs[0].edge_index.shape[1]}")

        # ── DataLoaders ──────────────────────────────────────────────────────
        train_loader = DataLoader(train_graphs, batch_size=args.batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(val_graphs,   batch_size=args.batch_size,
                                  shuffle=False)
        test_loader  = DataLoader(test_graphs,  batch_size=args.batch_size,
                                  shuffle=False)

        # ── Model ────────────────────────────────────────────────────────────
        model = SeizureGCN(
            in_channels=n_node_feat,
            hidden_dim=args.hidden_dim,
            dropout=0.3
        ).to(device)

        # Class-weighted loss (handles imbalance)
        pos_weight = torch.tensor(
            [(len(train_graphs) - n_train_ict) / max(n_train_ict, 1)],
            dtype=torch.float
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=1e-4
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=10, verbose=False
        )

        # ── Training loop ────────────────────────────────────────────────────
        best_val_auroc  = -1.0
        best_state_dict = None
        patience_counter = 0
        early_stop_patience = 20

        history = {
            "fold"      : fold_num,
            "train_loss": [],
            "val_loss"  : [],
            "val_auroc" : [],
        }

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_metrics, _, _ = evaluate(
                model, val_loader, criterion, device
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_auroc"].append(val_metrics["auroc"])

            scheduler.step(val_metrics["auroc"])

            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc   = val_metrics["auroc"]
                best_state_dict  = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:3d}/{args.epochs}  "
                      f"train_loss={train_loss:.4f}  "
                      f"val_loss={val_metrics['loss']:.4f}  "
                      f"val_AUROC={val_metrics['auroc']:.3f}"
                      f"{'  *' if patience_counter == 0 else ''}")

            if patience_counter >= early_stop_patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(best val AUROC={best_val_auroc:.3f})")
                break

        # ── Test evaluation ──────────────────────────────────────────────────
        model.load_state_dict(best_state_dict)
        model.to(device)
        test_metrics, test_probs, test_labels = evaluate(
            model, test_loader, criterion, device
        )

        print(f"\n  TEST RESULTS:")
        print(f"    AUROC     = {test_metrics['auroc']:.3f}")
        print(f"    F1-ictal  = {test_metrics['f1_ictal']:.3f}")
        print(f"    F1-macro  = {test_metrics['f1_macro']:.3f}")
        print(f"    Accuracy  = {test_metrics['accuracy']:.3f}")
        print(f"    Precision = {test_metrics['precision']:.3f}")
        print(f"    Recall    = {test_metrics['recall']:.3f}")
        print()

        fold_result = {
            "fold"          : fold_num,
            "test_patient"  : test_pat,
            "best_val_auroc": float(best_val_auroc),
            "n_train"       : len(train_graphs),
            "n_val"         : len(val_graphs),
            "n_test"        : len(test_graphs),
            "test_metrics"  : test_metrics,
        }
        all_fold_results.append(fold_result)
        fold_histories.append(history)

        # ROC data for plotting
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        all_roc_data.append((fpr, tpr, test_metrics["auroc"],
                             f"Fold{fold_num} PAT_{test_pat}"))

        # Save best model for this fold
        torch.save(best_state_dict,
                   output_dir / f"gcn_fold{fold_num}_best.pt")

    # ── Aggregate results ────────────────────────────────────────────────────
    print("=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    metric_keys = ["auroc", "f1_macro", "f1_ictal",
                   "accuracy", "precision", "recall"]
    results_summary = {}

    for key in metric_keys:
        values = [f["test_metrics"][key] for f in all_fold_results]
        results_summary[key] = {
            "mean"  : float(np.mean(values)),
            "std"   : float(np.std(values)),
            "values": [float(v) for v in values],
        }

    # Save JSON
    with open(output_dir / "results_per_fold.json", "w",
              encoding="utf-8") as f:
        json.dump(all_fold_results, f, indent=2, default=str)
    with open(output_dir / "results_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"  Saved: results_per_fold.json")
    print(f"  Saved: results_summary.json")

    # Plots
    print("\nGenerating plots...")
    plot_training_curves(fold_histories, output_dir)
    plot_roc_curves(all_roc_data, output_dir)
    save_thesis_table(results_summary, args, output_dir)

    print()
    print("=" * 70)
    print("GCN COMPLETE")
    print("=" * 70)
    print()
    print("  Next step: step8_gcn_ssl.py")
    print("    Add self-supervised pre-training (contrastive)")
    print("    before fine-tuning on seizure labels.")
    print("=" * 70)


if __name__ == "__main__":
    main()