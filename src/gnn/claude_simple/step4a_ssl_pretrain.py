"""
Step 4a — Self-Supervised GNN Pretraining (GraphCL)
=====================================================
Contrastive pretraining of a GCN encoder on EEG graphs WITHOUT labels.

Method: GraphCL (You et al. 2020) with NT-Xent loss (SimCLR)
─────────────────────────────────────────────────────────────
For each graph G in a batch:
  1. Apply augmentation A1 (edge dropping)   → G1
  2. Apply augmentation A2 (node masking)    → G2
  3. Encode both:  GCN(G1) → h1,  GCN(G2) → h2
  4. Project:      MLP(h1) → z1,  MLP(h2) → z2
  5. NT-Xent loss: maximize agreement between z1,z2 (positive pair)
                   while pushing apart all other pairs in the batch

NT-Xent Loss (normalized temperature-scaled cross entropy)
───────────────────────────────────────────────────────────
For a batch of N graphs → 2N augmented views.
For each view i, its positive is the other view of the same graph.
All 2(N-1) other views are negatives.

  L(i) = -log [ exp(sim(zi,zj)/τ) / Σ_{k≠i} exp(sim(zi,zk)/τ) ]

where sim() = cosine similarity, τ = temperature (default 0.5)

Augmentations
─────────────
  Edge Drop   : randomly remove p_edge fraction of edges
  Node Masking: randomly zero out p_node fraction of node features

Architecture
────────────
  GCNConv(12→64) + BN + ReLU
  GCNConv(64→128) + BN + ReLU      ← wider than supervised GCN
  Global Mean Pool → h (128,)       ← this is the graph embedding
  MLP projection head:
    Linear(128→128) + ReLU
    Linear(128→64)  → z (64,)       ← used only for contrastive loss

The projection head is DISCARDED after pretraining.
Only the GCN encoder is saved and reused in step4b.

Usage:
    python step4a_ssl_pretrain.py \\
        --datadir   path/to/graphs \\
        --outdir    path/to/ssl_pretrained \\
        --epochs    200 \\
        --batch_size 64

Output:
    ssl_pretrained/
        encoder_final.pt       ← full model weights (end of training)
        encoder_best.pt        ← weights at lowest contrastive loss
        pretrain_config.json   ← hyperparameters used
        pretrain_history.png   ← loss curve
"""

import argparse
import json
import warnings
import random
import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add this line:
try:
    import torch
except ImportError:
    pass  # will be caught later by import_deps()

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
        raise ImportError("Install PyTorch: pip install torch")
    try:
        from torch_geometric.data    import Data, DataLoader
        from torch_geometric.nn      import GCNConv, global_mean_pool
    except ImportError:
        raise ImportError("Install PyG: pip install torch_geometric")
    return torch, nn, F, Data, DataLoader, GCNConv, global_mean_pool


# ══════════════════════════════════════════════════════════════════════════════
# 1. AUGMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def augment_edge_drop(graph, p_drop=0.20):
    """
    Randomly drop p_drop fraction of edges.

    Rationale: EEG connectivity estimates have uncertainty.
    The encoder should learn representations robust to missing edges.

    Parameters
    ----------
    graph  : PyG Data object
    p_drop : float  fraction of edges to remove (default 0.20 = drop 20%)

    Returns
    -------
    New PyG Data object with subset of edges
    """
    g = copy.copy(graph)
    n_edges = g.edge_index.shape[1]

    if n_edges == 0:
        return g

    # Keep each edge with probability (1 - p_drop)
    keep_mask = torch.rand(n_edges) > p_drop
    
    # Always keep at least one edge to avoid isolated graphs
    if keep_mask.sum() == 0:
        keep_mask[torch.randint(0, n_edges, (1,))] = True

    g.edge_index = graph.edge_index[:, keep_mask]
    if graph.edge_attr is not None:
        g.edge_attr = graph.edge_attr[keep_mask]

    return g


def augment_node_masking(graph, p_mask=0.10):
    """
    Randomly zero out p_mask fraction of node feature vectors.

    Rationale: EEG channels can have transient artifacts or poor contact.
    The encoder should learn to infer missing channel information
    from the graph structure and remaining channels.

    Parameters
    ----------
    graph  : PyG Data object
    p_mask : float  fraction of nodes to mask (default 0.10 = mask 10%)

    Returns
    -------
    New PyG Data object with some node features zeroed
    """
    g = copy.copy(graph)
    n_nodes = g.x.shape[0]

    # Select nodes to mask
    n_mask   = max(1, int(n_nodes * p_mask))
    mask_idx = torch.randperm(n_nodes)[:n_mask]

    x_new               = graph.x.clone()
    x_new[mask_idx, :] = 0.0
    g.x                 = x_new

    return g


# ══════════════════════════════════════════════════════════════════════════════
# 2. NT-Xent LOSS
# ══════════════════════════════════════════════════════════════════════════════

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Given batch of N graphs with embeddings z1 (N, D) and z2 (N, D):
    - Positive pair for sample i: (z1_i, z2_i)
    - Negatives: all other 2(N-1) embeddings in the batch

    Parameters
    ----------
    z1, z2      : torch.Tensor  (N, D)  L2-normalized embeddings
    temperature : float  τ — lower = sharper distribution (default 0.5)

    Returns
    -------
    loss : scalar tensor
    """
    N    = z1.shape[0]
    F_   = torch.nn  # alias
    
    # L2 normalize
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    # Concatenate: (2N, D)
    z = torch.cat([z1, z2], dim=0)

    # Similarity matrix: (2N, 2N)
    sim = torch.mm(z, z.T) / temperature

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs:
    # For i in [0, N): positive is i+N
    # For i in [N, 2N): positive is i-N
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])  # (2N,)

    loss = torch.nn.functional.cross_entropy(sim, labels)
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_ssl_model(torch, nn, F, GCNConv, global_mean_pool,
                    in_channels=12, hidden=64, embed_dim=128, proj_dim=64,
                    dropout=0.2):
    """
    GCN encoder + MLP projection head for contrastive pretraining.

    After pretraining:
      - Encoder (GCN layers + pooling) is saved → used in step4b
      - Projection head is discarded

    Parameters
    ----------
    in_channels : node feature dim (12)
    hidden      : first GCN layer width (64)
    embed_dim   : second GCN layer / graph embedding dim (128)
    proj_dim    : projection head output dim (64)
    dropout     : dropout rate (0.2)
    """

    class GCNEncoder(nn.Module):
        """
        The part that gets SAVED and reused in step4b.
        Output: graph-level embedding h of shape (batch, embed_dim)
        """
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, embed_dim)
            self.bn1   = nn.BatchNorm1d(hidden)
            self.bn2   = nn.BatchNorm1d(embed_dim)
            self.drop  = nn.Dropout(p=dropout)

        def forward(self, x, edge_index, edge_weight, batch):
            # Layer 1
            x = self.conv1(x, edge_index, edge_weight)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.drop(x)
            # Layer 2
            x = self.conv2(x, edge_index, edge_weight)
            x = self.bn2(x)
            x = F.relu(x)
            # Global pooling → graph embedding
            h = global_mean_pool(x, batch)   # (batch_size, embed_dim)
            return h

    class ProjectionHead(nn.Module):
        """
        Small MLP used ONLY during pretraining.
        Maps h → z for NT-Xent loss.
        Discarded after pretraining (SimCLR finding: projecting into
        a lower-dim space before the loss improves representation quality).
        """
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, proj_dim),
            )

        def forward(self, h):
            return self.net(h)

    class SSLModel(nn.Module):
        """Full model = Encoder + Projection Head."""
        def __init__(self):
            super().__init__()
            self.encoder    = GCNEncoder()
            self.projector  = ProjectionHead()

        def forward(self, data):
            edge_w = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None
            h = self.encoder(data.x, data.edge_index, edge_w, data.batch)
            z = self.projector(h)
            return z

        def encode(self, data):
            """Get graph embeddings (no projection). Used in step4b."""
            edge_w = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None
            return self.encoder(data.x, data.edge_index, edge_w, data.batch)

    return SSLModel(), GCNEncoder, ProjectionHead


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def pretrain_epoch(model, loader, optimizer, temperature, device,
                   torch, p_edge_drop, p_node_mask):
    """
    One epoch of contrastive pretraining.

    For each batch:
      1. Apply edge_drop  → view1
      2. Apply node_mask  → view2
      3. Encode both
      4. Compute NT-Xent loss
      5. Backprop
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)

        # ── Create two augmented views ─────────────────────────────────
        # We need to augment each graph in the batch individually
        # then re-batch them for the GCN

        from torch_geometric.data import Batch

        graphs_v1 = []
        graphs_v2 = []

        # Unbatch: get individual graphs
        batch_size = batch.num_graphs
        for i in range(batch_size):
            # Extract single graph from batch
            g = batch.get_example(i)

            v1 = augment_edge_drop(g,    p_drop=p_edge_drop)
            v2 = augment_node_masking(g, p_mask=p_node_mask)

            graphs_v1.append(v1)
            graphs_v2.append(v2)

        # Re-batch
        batch1 = Batch.from_data_list(graphs_v1).to(device)
        batch2 = Batch.from_data_list(graphs_v2).to(device)

        # ── Forward pass ───────────────────────────────────────────────
        z1 = model(batch1)   # (N, proj_dim)
        z2 = model(batch2)   # (N, proj_dim)

        # ── NT-Xent loss ───────────────────────────────────────────────
        loss = nt_xent_loss(z1, z2, temperature=temperature)

        # ── Backprop ───────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_pretrain_history(history, output_dir):
    """Loss curve + learning rate curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['loss']) + 1)

    # Loss curve
    axes[0].plot(epochs, history['loss'], color='steelblue', lw=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('NT-Xent Loss', fontsize=12)
    axes[0].set_title('Contrastive Pretraining Loss', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)

    best_ep = int(np.argmin(history['loss'])) + 1
    best_l  = min(history['loss'])
    axes[0].axvline(best_ep, color='red', linestyle='--', lw=1.5,
                    label=f'Best epoch {best_ep} (loss={best_l:.4f})')
    axes[0].legend()

    # Smoothed loss
    if len(history['loss']) >= 10:
        window = max(5, len(history['loss']) // 20)
        smoothed = np.convolve(history['loss'],
                               np.ones(window)/window, mode='valid')
        axes[1].plot(range(window, len(history['loss'])+1), smoothed,
                     color='darkorange', lw=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('NT-Xent Loss (smoothed)', fontsize=12)
        axes[1].set_title(f'Smoothed Loss (window={window})',
                          fontsize=13, fontweight='bold')
        axes[1].grid(alpha=0.3)

    plt.suptitle('Step 4a — SSL Contrastive Pretraining',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pretrain_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_dir / 'pretrain_history.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Self-supervised GCN pretraining with contrastive loss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # I/O
    parser.add_argument('--datadir',     required=True,
                        help='Directory with dataset.pt (Step 3c)')
    parser.add_argument('--outdir',      required=True,
                        help='Output directory for pretrained encoder')

    # Architecture
    parser.add_argument('--hidden',      type=int,   default=64)
    parser.add_argument('--embed_dim',   type=int,   default=128,
                        help='Graph embedding dimension (default: 128)')
    parser.add_argument('--proj_dim',    type=int,   default=64,
                        help='Projection head output dim (default: 64)')
    parser.add_argument('--dropout',     type=float, default=0.2)

    # Training
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--weight_decay',type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='NT-Xent temperature τ (default: 0.5)')
    parser.add_argument('--patience',    type=int,   default=30,
                        help='Early stopping patience (default: 30)')

    # Augmentation
    parser.add_argument('--p_edge_drop', type=float, default=0.20,
                        help='Edge drop probability (default: 0.20)')
    parser.add_argument('--p_node_mask', type=float, default=0.10,
                        help='Node masking probability (default: 0.10)')

    parser.add_argument('--device',      type=str,   default='auto')
    args = parser.parse_args()

    # ── Dependencies ──────────────────────────────────────────────────────
    torch, nn, F, Data, DataLoader, GCNConv, global_mean_pool = import_deps()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    data_dir   = Path(args.datadir)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ALL graphs (no labels used) ──────────────────────────────────
    print("=" * 72)
    print("STEP 4a — SELF-SUPERVISED CONTRASTIVE PRETRAINING")
    print("=" * 72)
    print(f"  Device:       {device}")
    print(f"  Architecture: GCNConv({12}→{args.hidden}→{args.embed_dim}) + proj({args.proj_dim})")
    print(f"  Epochs:       {args.epochs}  (patience={args.patience})")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Temperature:  τ = {args.temperature}")
    print(f"  Aug1 (view1): Edge drop  p={args.p_edge_drop}")
    print(f"  Aug2 (view2): Node mask  p={args.p_node_mask}")
    print(f"  LR:           {args.lr}  |  WD: {args.weight_decay}")
    print("=" * 72)

    print("\nLoading graphs (labels ignored during pretraining)...")
    all_graphs = torch.load(data_dir / 'dataset_filtered.pt', map_location='cpu')
    print(f"  ✅ Loaded {len(all_graphs):,} graphs")
    print(f"  ℹ️  Labels exist but are NOT used during pretraining")

    # DataLoader — shuffle all graphs, no label needed
    loader = DataLoader(all_graphs, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)
    print(f"  Batches per epoch: {len(loader)}")

    # ── Build model ───────────────────────────────────────────────────────
    model, GCNEncoder, _ = build_ssl_model(
        torch, nn, F, GCNConv, global_mean_pool,
        in_channels = 12,
        hidden      = args.hidden,
        embed_dim   = args.embed_dim,
        proj_dim    = args.proj_dim,
        dropout     = args.dropout,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {n_params:,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training ──────────────────────────────────────────────────────────
    print("\nStarting pretraining...")
    print("  (Loss should decrease and stabilize — no accuracy metric during SSL)")
    print()

    history       = {'loss': []}
    best_loss     = float('inf')
    patience_ctr  = 0
    best_epoch    = 0

    pbar = tqdm(range(1, args.epochs + 1), desc="Pretraining", unit="epoch")

    for epoch in pbar:
        loss = pretrain_epoch(
            model, loader, optimizer,
            temperature  = args.temperature,
            device       = device,
            torch        = torch,
            p_edge_drop  = args.p_edge_drop,
            p_node_mask  = args.p_node_mask,
        )
        scheduler.step()
        history['loss'].append(loss)

        pbar.set_postfix({
            'loss': f"{loss:.4f}",
            'best': f"{best_loss:.4f}",
            'ep':   best_epoch,
        })

        # ── Save best encoder ──────────────────────────────────────────
        if loss < best_loss:
            best_loss    = loss
            best_epoch   = epoch
            patience_ctr = 0
            # Save only the encoder (not projection head)
            torch.save(model.encoder.state_dict(),
                       output_dir / 'encoder_best.pt')
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best loss {best_loss:.4f} at epoch {best_epoch})")
                break

        # Periodic log every 20 epochs
        if epoch % 20 == 0:
            tqdm.write(f"  Epoch {epoch:4d} | loss={loss:.4f} | "
                       f"best={best_loss:.4f} @ ep{best_epoch}")

    # ── Save final encoder ────────────────────────────────────────────────
    torch.save(model.encoder.state_dict(),
               output_dir / 'encoder_final.pt')

    # ── Save config ───────────────────────────────────────────────────────
    config = {
        'in_channels':   12,
        'hidden':        args.hidden,
        'embed_dim':     args.embed_dim,
        'proj_dim':      args.proj_dim,
        'dropout':       args.dropout,
        'temperature':   args.temperature,
        'p_edge_drop':   args.p_edge_drop,
        'p_node_mask':   args.p_node_mask,
        'epochs_run':    len(history['loss']),
        'best_epoch':    best_epoch,
        'best_loss':     float(best_loss),
        'final_loss':    float(history['loss'][-1]),
        'n_graphs':      len(all_graphs),
        'augmentations': ['edge_drop', 'node_masking'],
        'loss_fn':       'NT-Xent',
    }

    with open(output_dir / 'pretrain_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    with open(output_dir / 'pretrain_loss_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_pretrain_history(history, output_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PRETRAINING COMPLETE")
    print("=" * 72)
    print(f"  Epochs run:      {len(history['loss'])}")
    print(f"  Best epoch:      {best_epoch}")
    print(f"  Best loss:       {best_loss:.4f}")
    print(f"  Final loss:      {history['loss'][-1]:.4f}")
    print()
    print("  Saved files:")
    print(f"    encoder_best.pt          ← encoder at lowest loss")
    print(f"    encoder_final.pt         ← encoder at last epoch")
    print(f"    pretrain_config.json     ← hyperparameters")
    print(f"    pretrain_history.png     ← loss curve")
    print()
    print("  Architecture saved:")
    print(f"    GCNConv(12→{args.hidden}) + GCNConv({args.hidden}→{args.embed_dim})")
    print(f"    Global mean pool → embedding dim = {args.embed_dim}")
    print()
    print("  ⚠️  The projection head was NOT saved (discarded after SSL).")
    print("  ✅ Use encoder_best.pt in step4b for linear probe + fine-tuning.")
    print()
    print("  Next:")
    print(f"    python step4b_ssl_evaluate.py \\")
    print(f"        --datadir   {data_dir} \\")
    print(f"        --encoderdir {output_dir} \\")
    print(f"        --outdir    path/to/ssl_results")
    print("=" * 72)


if __name__ == '__main__':
    main()
