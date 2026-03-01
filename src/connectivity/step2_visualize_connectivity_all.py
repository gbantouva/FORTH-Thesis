"""
Step 2 — Comprehensive Connectivity Visualization
==================================================
All standard plot types from the literature, with per-channel granularity
(each channel treated independently, as required).

Plot types produced
───────────────────
A) Heatmap pair           — DTF + PDC, integrated band, fixed scale [0, 1]
B) Multi-band grid        — 2 rows (DTF/PDC) × 6 frequency bands
C) Per-channel timeline   — outflow AND inflow per channel over time,
                            one subplot per channel, coloured by label
D) Circular diagram       — arrow plot of dominant connections per epoch,
                            node size = total outflow
E) Band bar charts        — per-channel mean outflow per band,
                            pre-ictal vs ictal with ±1 std error bars
F) Baccalá frequency grid — spectrum per channel pair (selected epochs only)

Key design choice (professor's requirement)
────────────────────────────────────────────
All connectivity measures are computed and displayed PER CHANNEL:
  • Outflow of channel j  = mean DTF over sinks i  (column mean, excluding diagonal)
                          = how strongly channel j drives all other channels
  • Inflow  to channel i  = mean DTF over sources j (row mean, excluding diagonal)
                          = how strongly all other channels drive channel i

Output structure
────────────────
output_dir/
  subject_01/
    heatmap/         ep000_preictal.png  ep005_ictal.png  ...
    multiband/       ep000_preictal.png  ...
    channel_timeline/
      dtf_timeline.png       ← all 19 channels, outflow + inflow vs time
      pdc_timeline.png
    circular/        ep000_preictal.png  ep005_ictal.png  ...
    band_bars/
      dtf_band_bars.png      ← per-channel, per-band, pre vs ictal
      pdc_band_bars.png
    baccala/         ep000_preictal_dtf.png  ...  (selected epochs only)
    timeline.png             ← subject-level mean connectivity over time

Usage
─────
  # Full run (all plot types):
  python step2_visualize_all.py \\
      --npzdir   path/to/connectivity \\
      --epochdir path/to/preprocessed_epochs \\
      --outdir   path/to/figures \\
      --fixedorder 12

  # Skip Baccalá (saves time for large datasets):
  python step2_visualize_all.py ... --no_baccala

  # Only specific subjects:
  python step2_visualize_all.py ... --subjects subject_01 subject_02

  # Only specific plot types:
  python step2_visualize_all.py ... --plot_types heatmap channel_timeline circular

  # Limit epochs per subject (useful for quick inspection):
  python step2_visualize_all.py ... --max_epochs 50
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2',
]

BANDS = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1']
BAND_LABELS = {
    'integrated': 'Integrated\n0.5–45 Hz',
    'delta':      'Delta\n0.5–4 Hz',
    'theta':      'Theta\n4–8 Hz',
    'alpha':      'Alpha\n8–15 Hz',
    'beta':       'Beta\n15–30 Hz',
    'gamma1':     'Gamma1\n30–45 Hz',
}
BAND_COLORS = {
    'integrated': '#2c3e50',
    'delta':      '#8e44ad',
    'theta':      '#2980b9',
    'alpha':      '#27ae60',
    'beta':       '#f39c12',
    'gamma1':     '#e74c3c',
}
BANDS_NO_INT = ['delta', 'theta', 'alpha', 'beta', 'gamma1']

LABEL_NAMES  = {0: 'preictal', 1: 'ictal'}
LABEL_COLORS = {0: '#2980b9',  1: '#c0392b'}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _npz_get(data, key, default=None):
    """Safe key access for NpzFile objects."""
    return data[key] if key in data else default


def _chan(K):
    return CHANNEL_NAMES[:K]


def _label_info(label):
    return LABEL_NAMES.get(label, 'unknown'), LABEL_COLORS.get(label, 'gray')


def _tstr(t):
    return f'{t:+.1f}s from onset' if t is not None else ''


# ══════════════════════════════════════════════════════════════════════════════
# A) HEATMAP PAIR  (integrated band, fixed [0, 1])
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap_pair(dtf, pdc, subject_name, epoch_idx, label,
                      time_from_onset, order, out_path):
    """
    DTF and PDC side by side.
    Colour scale fixed [0, 1] — same across all epochs and subjects.
    matrix[i, j] = source j → sink i.
    DTF: bright columns = strong sources.
    PDC: bright rows    = strong sinks.
    """
    K = dtf.shape[0]
    chan = _chan(K)
    lbl_name, lcolor = _label_info(label)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, mat, mname, note in [
        (axes[0], dtf, 'DTF',
         'Bright COLUMNS = strong source channels\n'
         'row-normalised  |  direct + indirect'),
        (axes[1], pdc, 'PDC',
         'Bright ROWS = strong sink channels\n'
         'col-normalised  |  direct connections only'),
    ]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Connectivity  (diagonal = 0)')
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(chan, fontsize=7, rotation=90)
        ax.set_yticklabels(chan, fontsize=7)
        ax.set_xlabel('Source  (From j)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Sink  (To i)',     fontsize=10, fontweight='bold')
        ax.set_title(f'{mname}\n{note}', fontsize=10, fontweight='bold')

        if K <= 10:
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                            fontsize=6,
                            color='white' if mat[i, j] > 0.5 else 'black')

    fig.suptitle(
        f'{subject_name}  |  Epoch {epoch_idx:03d}  |  {lbl_name.upper()}  |  '
        f'{_tstr(time_from_onset)}  |  order p={order}\n'
        'Colour scale fixed [0, 1] — same across all epochs and subjects',
        fontsize=12, fontweight='bold', color=lcolor,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# B) MULTI-BAND HEATMAP GRID  (2 rows × 6 bands)
# ══════════════════════════════════════════════════════════════════════════════

def plot_multiband_grid(dtf_bands, pdc_bands, subject_name, epoch_idx,
                        label, time_from_onset, order, out_path):
    """
    2 rows (DTF / PDC) × 6 columns (frequency bands).
    Fixed colour scale [0, 1].
    """
    K = dtf_bands['integrated'].shape[0]
    lbl_name, lcolor = _label_info(label)
    n_bands = len(BANDS)

    fig = plt.figure(figsize=(3.8 * n_bands + 1, 8.5))
    gs  = gridspec.GridSpec(2, n_bands + 1,
                             width_ratios=[1] * n_bands + [0.06],
                             hspace=0.35, wspace=0.12)

    for row, (bands_dict, row_label) in enumerate([
        (dtf_bands, 'DTF  (direct + indirect)'),
        (pdc_bands, 'PDC  (direct only)'),
    ]):
        for col, band in enumerate(BANDS):
            ax  = fig.add_subplot(gs[row, col])
            mat = bands_dict[band]
            im  = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight='bold')
            if row == 0:
                ax.set_title(BAND_LABELS[band], fontsize=8, fontweight='bold')

        cax = fig.add_subplot(gs[row, -1])
        plt.colorbar(im, cax=cax, label='[0, 1]')
        cax.tick_params(labelsize=7)

    fig.suptitle(
        f'{subject_name}  |  Epoch {epoch_idx:03d}  |  {lbl_name.upper()}  |  '
        f'{_tstr(time_from_onset)}  |  p={order}\n'
        '2 rows (DTF / PDC)  ×  6 frequency bands  |  fixed scale [0, 1]',
        fontsize=11, fontweight='bold', color=lcolor, y=1.02,
    )
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# C) PER-CHANNEL TIMELINE
#
# For each channel, a separate subplot shows:
#   • Outflow (circles) — how strongly THIS channel drives all others
#                         = column mean of DTF matrix, excluding diagonal
#   • Inflow  (triangles) — how strongly all others drive THIS channel
#                           = row mean of DTF matrix, excluding diagonal
# Epochs are coloured blue (pre-ictal) or red (ictal).
# A vertical dashed red line marks seizure onset (t = 0).
#
# This satisfies the requirement of treating each channel independently:
# you can see per-channel whether it becomes a stronger driver or receiver
# before/during the seizure.
# ══════════════════════════════════════════════════════════════════════════════

def plot_channel_timelines(dtf_all, labels, time_from_onset,
                           subject_name, out_path, metric_name='DTF'):
    """
    Per-channel timeline of outflow and inflow.

    Parameters
    ----------
    dtf_all : ndarray (n_epochs, K, K)
        Band-averaged connectivity matrix, diagonal already zeroed.
    labels  : ndarray (n_epochs,)
    time_from_onset : ndarray (n_epochs,) or None
    """
    n_epochs, K, _ = dtf_all.shape
    chan = _chan(K)

    # outflow[e, j] = mean over sinks i of DTF[i, j] at epoch e
    #               = how strongly channel j drives others
    # Diagonal is already 0, so mean(axis=0) gives correct result
    # dtf_all has shape (n_epochs, K, K): dtf_all[e, i, j]
    outflow = dtf_all.mean(axis=1)   # (n_epochs, K) — mean over sinks (rows)
    inflow  = dtf_all.mean(axis=2)   # (n_epochs, K) — mean over sources (cols)

    t = time_from_onset if time_from_onset is not None else np.arange(n_epochs)

    n_cols = 5
    n_rows = int(np.ceil(K / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3 * n_rows),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ch in range(K):
        ax = axes_flat[ch]

        for lbl in [0, 1]:
            idx = np.where(labels == lbl)[0]
            if len(idx) == 0:
                continue
            lc = LABEL_COLORS[lbl]

            # Scatter — outflow (circles) and inflow (triangles)
            ax.scatter(t[idx], outflow[idx, ch],
                       c=lc, s=10, alpha=0.80, marker='o', zorder=3)
            ax.scatter(t[idx], inflow[idx, ch],
                       c=lc, s=10, alpha=0.40, marker='^', zorder=3)

        # Connecting lines (gray, faint) for visual continuity
        ax.plot(t, outflow[:, ch], color='gray', lw=0.5, alpha=0.4, zorder=2)
        ax.plot(t, inflow[:, ch],  color='gray', lw=0.5, alpha=0.4,
                linestyle='--', zorder=2)

        ax.axvline(0, color='red', lw=1.5, linestyle='--', alpha=0.9, zorder=4)
        ax.set_title(chan[ch], fontsize=9, fontweight='bold')
        ax.set_ylim(0, None)
        ax.grid(alpha=0.2)

    # Hide unused subplots
    for ch in range(K, len(axes_flat)):
        axes_flat[ch].set_visible(False)

    # Shared axis labels
    fig.text(0.5,  0.01, 'Time from seizure onset (s)',
             ha='center', fontsize=11, fontweight='bold')
    fig.text(0.01, 0.5,  f'Mean {metric_name}',
             va='center', rotation='vertical', fontsize=11, fontweight='bold')

    # Legend (built manually so it doesn't repeat per subplot)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=LABEL_COLORS[0], markersize=8,
                   label='Pre-ictal  outflow (drives others)'),
        plt.Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=LABEL_COLORS[0], markersize=8, alpha=0.5,
                   label='Pre-ictal  inflow (driven by others)'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=LABEL_COLORS[1], markersize=8,
                   label='Ictal  outflow'),
        plt.Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=LABEL_COLORS[1], markersize=8, alpha=0.5,
                   label='Ictal  inflow'),
        plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5,
                   label='Seizure onset  (t = 0)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               fontsize=8, ncol=2, bbox_to_anchor=(0.99, 0.99))

    fig.suptitle(
        f'{subject_name} — {metric_name} per channel over time\n'
        'Circles = outflow (channel drives others)  |  '
        'Triangles = inflow (others drive channel)',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# D) CIRCULAR DIAGRAM
#
# Channels arranged on a circle.
# An arrow from node j to node i means channel j → channel i (DTF[i,j]).
# Only connections above a threshold are drawn.
# Arrow thickness and opacity encode connection strength.
# Node size encodes total outflow (how strong a driver each channel is).
# ══════════════════════════════════════════════════════════════════════════════

def plot_circular_diagram(dtf_mean, subject_name, epoch_idx, label,
                          time_from_onset, order, out_path, threshold=0.15):
    """
    Circular connectivity diagram for one epoch.

    Parameters
    ----------
    dtf_mean  : ndarray (K, K)   band-averaged DTF, diagonal zeroed
    threshold : float            only draw connections above this value
    """
    K = dtf_mean.shape[0]
    chan = _chan(K)
    lbl_name, _ = _label_info(label)

    # Node positions — equally spaced on unit circle, starting from top
    angles = np.pi / 2 - np.linspace(0, 2 * np.pi, K, endpoint=False)
    node_x = np.cos(angles)
    node_y = np.sin(angles)

    # Total outflow per channel (column sum, since DTF[i,j] = j→i)
    # Diagonal is 0 so sum gives outflow excluding self
    outflow     = dtf_mean.sum(axis=0)          # (K,)
    max_outflow = max(outflow.max(), 1e-6)
    max_val     = max(dtf_mean.max(),  1e-6)

    cmap = cm.get_cmap('Reds')

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.axis('off')

    # Draw arrows for connections above threshold
    for src_j in range(K):
        for snk_i in range(K):
            if src_j == snk_i:
                continue
            val = dtf_mean[snk_i, src_j]
            if val < threshold:
                continue

            norm_val = val / max_val
            color    = cmap(norm_val)
            lw       = 0.5 + 4.0 * norm_val
            alpha    = 0.25 + 0.65 * norm_val

            arrow = FancyArrowPatch(
                posA=(node_x[src_j], node_y[src_j]),
                posB=(node_x[snk_i], node_y[snk_i]),
                connectionstyle='arc3,rad=0.20',
                arrowstyle='-|>',
                color=color,
                linewidth=lw,
                alpha=alpha,
                mutation_scale=12,
                zorder=2,
            )
            ax.add_patch(arrow)

    # Draw nodes
    for ch in range(K):
        node_size = 180 + 700 * (outflow[ch] / max_outflow)
        ax.scatter(node_x[ch], node_y[ch],
                   s=node_size, c='white',
                   edgecolors='navy', linewidths=2.0, zorder=4)

        # Channel label just outside the circle
        lx = 1.20 * node_x[ch]
        ly = 1.20 * node_y[ch]
        ha = ('left' if node_x[ch] > 0.1
              else 'right' if node_x[ch] < -0.1
              else 'center')
        ax.text(lx, ly, chan[ch], ha=ha, va='center',
                fontsize=9, fontweight='bold', zorder=5)

    # Colour bar
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max_val))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.45)
    cbar.set_label('DTF strength', fontsize=9)

    ax.set_title(
        f'{subject_name}  |  Epoch {epoch_idx:03d}  |  '
        f'{lbl_name.upper()}  |  {_tstr(time_from_onset)}  |  p={order}\n'
        f'DTF circular diagram  |  threshold = {threshold}  |  '
        'node size = total outflow',
        fontsize=11, fontweight='bold', pad=20,
    )
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# E) BAND BAR CHARTS PER CHANNEL
#
# For each channel: grouped bar chart with one bar group per frequency band.
# Within each group: blue bar = pre-ictal mean, red bar = ictal mean.
# Error bars show ±1 std across epochs.
#
# Outflow is shown (column mean of the connectivity matrix):
# how strongly each channel drives the rest of the network,
# broken down by frequency band.
# ══════════════════════════════════════════════════════════════════════════════

def plot_band_per_channel(dtf_bands_all, pdc_bands_all, labels,
                          subject_name, out_dir, metric_name='DTF'):
    """
    Per-channel band bar chart.

    Parameters
    ----------
    dtf_bands_all : dict {band: ndarray (n_epochs, K, K)}
    labels        : ndarray (n_epochs,)
    out_dir       : Path
    """
    bands_dict = dtf_bands_all if metric_name == 'DTF' else pdc_bands_all
    K          = bands_dict['integrated'].shape[1]
    chan       = _chan(K)

    n_cols  = 5
    n_rows  = int(np.ceil(K / n_cols))
    x       = np.arange(len(BANDS_NO_INT))
    width   = 0.35

    pre_idx = np.where(labels == 0)[0]
    ict_idx = np.where(labels == 1)[0]

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows),
                              sharey=True)
    axes_flat = axes.flatten()

    for ch in range(K):
        ax = axes_flat[ch]

        pre_means, ict_means = [], []
        pre_stds,  ict_stds  = [], []

        for band in BANDS_NO_INT:
            mat_all = bands_dict[band]          # (n_epochs, K, K)
            # outflow of channel ch: column mean (DTF[i,ch] averaged over sinks i)
            # Diagonal is 0, so this correctly excludes self
            outflow_ch = mat_all[:, :, ch].mean(axis=1)   # (n_epochs,)

            pre_means.append(outflow_ch[pre_idx].mean() if len(pre_idx) else 0.0)
            ict_means.append(outflow_ch[ict_idx].mean() if len(ict_idx) else 0.0)
            pre_stds.append(outflow_ch[pre_idx].std()  if len(pre_idx) else 0.0)
            ict_stds.append(outflow_ch[ict_idx].std()  if len(ict_idx) else 0.0)

        ax.bar(x - width / 2, pre_means, width,
               yerr=pre_stds, capsize=3,
               color=LABEL_COLORS[0], alpha=0.85, label='Pre-ictal',
               error_kw={'lw': 1, 'capthick': 1})
        ax.bar(x + width / 2, ict_means, width,
               yerr=ict_stds, capsize=3,
               color=LABEL_COLORS[1], alpha=0.85, label='Ictal',
               error_kw={'lw': 1, 'capthick': 1})

        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in BANDS_NO_INT],
                           fontsize=7, rotation=40, ha='right')
        ax.set_title(chan[ch], fontsize=9, fontweight='bold')
        ax.grid(alpha=0.2, axis='y')
        ax.set_ylim(0, None)

    for ch in range(K, len(axes_flat)):
        axes_flat[ch].set_visible(False)

    axes_flat[0].legend(fontsize=7, loc='upper right')

    fig.text(0.5,  0.005, 'Frequency band',
             ha='center', fontsize=11, fontweight='bold')
    fig.text(0.005, 0.5,  f'Mean {metric_name} outflow ± std',
             va='center', rotation='vertical', fontsize=11, fontweight='bold')

    fig.suptitle(
        f'{subject_name} — {metric_name} outflow per channel per band\n'
        'Pre-ictal vs Ictal  |  error bars = ±1 std  |  '
        'outflow = mean over all target channels',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.97])
    fig.savefig(out_dir / f'{metric_name.lower()}_band_bars.png',
                dpi=130, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# F) BACCALÁ FREQUENCY GRID
#    Each cell [sink_i, source_j] = connectivity vs frequency curve.
#    Requires the full spectrum (K, K, n_freqs), recomputed from raw epochs.
# ══════════════════════════════════════════════════════════════════════════════

def plot_baccala_grid(dtf_s, pdc_s, freqs, subject_name, epoch_idx,
                      label, time_from_onset, order, out_dir):
    """
    Baccalá-style grid.  dtf_s / pdc_s : (K, K, n_freqs), spectrum level.
    """
    K         = dtf_s.shape[0]
    chan      = _chan(K)
    lbl_name, _ = _label_info(label)

    for metric_s, mname, mcolor in [
        (dtf_s, 'DTF', 'steelblue'),
        (pdc_s, 'PDC', 'tomato'),
    ]:
        fig, axes = plt.subplots(K, K, figsize=(2.8 * K, 2.4 * K))
        fig.suptitle(
            f'{subject_name}  |  Epoch {epoch_idx:03d}  |  '
            f'{lbl_name.upper()}  |  {_tstr(time_from_onset)}  |  p={order}\n'
            f'{mname}  —  Baccalá style  |  '
            'Row = Sink (To i),  Col = Source (From j)',
            fontsize=10, fontweight='bold', y=0.999,
        )

        for snk_i in range(K):
            for src_j in range(K):
                ax   = axes[snk_i, src_j]
                vals = metric_s[snk_i, src_j, :]

                if snk_i == src_j:
                    ax.fill_between(freqs, 0, vals, alpha=0.2, color='gray')
                    ax.plot(freqs, vals, color='gray', lw=0.8)
                else:
                    ax.fill_between(freqs, 0, vals, alpha=0.3, color=mcolor)
                    ax.plot(freqs, vals, color=mcolor, lw=1.0)

                ax.set_ylim(0, 1.05)
                ax.set_xlim(freqs[0], freqs[-1])
                ax.set_xticks([]); ax.set_yticks([0, 0.5, 1])
                ax.tick_params(labelsize=4)
                ax.grid(alpha=0.2)

                if src_j == 0:
                    ax.set_ylabel(chan[snk_i], fontsize=7, fontweight='bold',
                                  rotation=0, labelpad=28, va='center')
                if snk_i == K - 1:
                    ax.set_xlabel(chan[src_j], fontsize=7, fontweight='bold')

        fig.text(0.02, 0.5, 'Sink  →', va='center', rotation='vertical',
                 fontsize=10, fontweight='bold', color='navy')
        fig.text(0.5, 0.005, 'Source  →', ha='center',
                 fontsize=10, fontweight='bold', color='navy')

        plt.tight_layout(rect=[0.04, 0.03, 1, 0.97])
        fname = out_dir / f'ep{epoch_idx:03d}_{lbl_name}_{mname.lower()}.png'
        fig.savefig(fname, dpi=110, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Subject-level mean timeline (classic, no per-channel breakdown)
# ══════════════════════════════════════════════════════════════════════════════

def plot_subject_timeline(dtf_integrated, labels, time_from_onset,
                          subject_name, out_path):
    """Mean off-diagonal DTF over time, coloured by label."""
    n_epochs, K, _ = dtf_integrated.shape
    mask      = ~np.eye(K, dtype=bool)
    mean_conn = np.array([dtf_integrated[e][mask].mean()
                          for e in range(n_epochs)])

    t = time_from_onset if time_from_onset is not None else np.arange(n_epochs)

    fig, ax = plt.subplots(figsize=(16, 4))

    for lbl, lname, lc in [(0, 'Pre-ictal', LABEL_COLORS[0]),
                             (1, 'Ictal',     LABEL_COLORS[1])]:
        idx = np.where(labels == lbl)[0]
        if len(idx) > 0:
            ax.scatter(t[idx], mean_conn[idx],
                       c=lc, s=15, label=lname, alpha=0.7, zorder=3)

    ax.plot(t, mean_conn, color='gray', lw=0.6, alpha=0.5, zorder=2)
    ax.axvline(0, color='red', lw=2, linestyle='--',
               label='Seizure onset', zorder=4)
    ax.set_xlabel('Time from seizure onset (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean DTF\n(off-diagonal)',     fontsize=11, fontweight='bold')
    ax.set_title(
        f'{subject_name} — Mean connectivity over time  (DTF integrated band)',
        fontsize=12, fontweight='bold',
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive DTF/PDC visualization — per-channel granularity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--npzdir',      required=True,
                        help='Directory with subject_XX_graphs.npz files')
    parser.add_argument('--outdir',      required=True,
                        help='Output directory for all figures')
    parser.add_argument('--epochdir',    default=None,
                        help='Directory with subject_XX_epochs.npy '
                             '(needed for Baccalá recomputation)')
    parser.add_argument('--fixedorder',  type=int, default=12,
                        help='MVAR order used in step2 (default: 12)')
    parser.add_argument('--subjects',    nargs='+', default=None,
                        help='Specific subjects to process (default: all)')
    parser.add_argument('--plot_types',  nargs='+',
                        default=['heatmap', 'multiband', 'channel_timeline',
                                 'circular', 'band_bars', 'timeline'],
                        choices=['heatmap', 'multiband', 'channel_timeline',
                                 'circular', 'band_bars', 'baccala', 'timeline'],
                        help='Which plot types to produce')
    parser.add_argument('--no_baccala',  action='store_true',
                        help='Skip Baccalá grids (they are slow for 19 channels)')
    parser.add_argument('--baccala_epochs', nargs='+', default=['auto'],
                        help='Epoch indices for Baccalá, or "auto" '
                             '(first pre-ictal + first ictal per subject)')
    parser.add_argument('--max_epochs',  type=int, default=None,
                        help='Max epochs to visualize per subject (default: all)')
    parser.add_argument('--circular_every', type=int, default=10,
                        help='Make circular diagram every N epochs (default: 10)')
    parser.add_argument('--circ_threshold', type=float, default=0.15,
                        help='Connection threshold for circular diagram (default: 0.15)')
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    npz_dir = Path(args.npzdir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_heatmap   = 'heatmap'          in args.plot_types
    do_multiband = 'multiband'        in args.plot_types
    do_ch_time   = 'channel_timeline' in args.plot_types
    do_circular  = 'circular'         in args.plot_types
    do_band_bars = 'band_bars'        in args.plot_types
    do_baccala   = ('baccala'         in args.plot_types) and not args.no_baccala
    do_timeline  = 'timeline'         in args.plot_types

    if do_baccala:
        if args.epochdir is None:
            print('⚠️  Baccalá requires --epochdir.  Disabling.')
            do_baccala = False
        else:
            from statsmodels.tsa.vector_ar.var_model import VAR
            from step2_compute_connectivity_updated import compute_dtf_pdc_from_var
            epoch_dir = Path(args.epochdir)

    npz_files = sorted(npz_dir.glob('subject_*_graphs.npz'))
    if args.subjects:
        npz_files = [f for f in npz_files
                     if f.stem.replace('_graphs', '') in args.subjects]

    print('=' * 72)
    print('STEP 2 — COMPREHENSIVE VISUALIZATION')
    print('=' * 72)
    print(f'  NPZ dir:      {npz_dir}')
    print(f'  Output:       {out_dir}')
    print(f'  Subjects:     {len(npz_files)}')
    active = [t for t, flag in [
        ('heatmap',          do_heatmap),
        ('multiband',        do_multiband),
        ('channel_timeline', do_ch_time),
        ('circular',         do_circular),
        ('band_bars',        do_band_bars),
        ('baccala',          do_baccala),
        ('timeline',         do_timeline),
    ] if flag]
    print(f'  Plot types:   {", ".join(active)}')
    print('=' * 72)

    # ── Per-subject loop ──────────────────────────────────────────────────────
    for npz_file in tqdm(npz_files, desc='Subjects', unit='subject'):
        subject_name = npz_file.stem.replace('_graphs', '')
        data = np.load(npz_file)

        n_epochs    = len(data['labels'])
        labels      = data['labels']                            # (n_epochs,)
        indices     = _npz_get(data, 'indices',
                               np.arange(n_epochs))             # (n_epochs,)
        tfo         = _npz_get(data, 'time_from_onset', None)   # (n_epochs,) or None
        orders      = _npz_get(data, 'orders',
                               np.full(n_epochs, args.fixedorder))
        fixed_order = int(_npz_get(data, 'fixed_order', args.fixedorder))

        ep_range = range(min(n_epochs, args.max_epochs)
                         if args.max_epochs else n_epochs)

        # Load all band matrices (needed for band_bars and channel_timeline)
        dtf_bands_all = {b: data[f'dtf_{b}'] for b in BANDS}  # {band: (E,K,K)}
        pdc_bands_all = {b: data[f'pdc_{b}'] for b in BANDS}

        # ── Create output directories ─────────────────────────────────────
        sd = out_dir / subject_name
        dirs = {
            'heat':   sd / 'heatmap',
            'mband':  sd / 'multiband',
            'circ':   sd / 'circular',
            'bar':    sd / 'band_bars',
            'bacc':   sd / 'baccala',
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # ── A: Subject-level mean timeline ────────────────────────────────
        if do_timeline:
            plot_subject_timeline(
                dtf_bands_all['integrated'],
                labels, tfo,
                subject_name,
                sd / 'timeline.png',
            )

        # ── B: Per-channel timelines (one figure, all channels) ───────────
        if do_ch_time:
            ch_time_dir = sd / 'channel_timeline'
            ch_time_dir.mkdir(exist_ok=True)

            dtf_int_epochs = dtf_bands_all['integrated'][:len(ep_range)]
            pdc_int_epochs = pdc_bands_all['integrated'][:len(ep_range)]
            tfo_sub = tfo[:len(ep_range)] if tfo is not None else None
            lbl_sub = labels[:len(ep_range)]

            plot_channel_timelines(
                dtf_int_epochs, lbl_sub, tfo_sub,
                subject_name,
                ch_time_dir / 'dtf_timeline.png',
                metric_name='DTF',
            )
            plot_channel_timelines(
                pdc_int_epochs, lbl_sub, tfo_sub,
                subject_name,
                ch_time_dir / 'pdc_timeline.png',
                metric_name='PDC',
            )

        # ── C: Band bar charts (one figure per metric) ────────────────────
        if do_band_bars:
            for metric in ['DTF', 'PDC']:
                plot_band_per_channel(
                    dtf_bands_all, pdc_bands_all, labels,
                    subject_name, dirs['bar'],
                    metric_name=metric,
                )

        # ── D: Baccalá setup ─────────────────────────────────────────────
        if do_baccala:
            if args.baccala_epochs == ['auto']:
                bacc_ep_list = []
                pre_idx = np.where(labels == 0)[0]
                ict_idx = np.where(labels == 1)[0]
                if len(pre_idx): bacc_ep_list.append(int(pre_idx[0]))
                if len(ict_idx): bacc_ep_list.append(int(ict_idx[0]))
            else:
                bacc_ep_list = [int(x) for x in args.baccala_epochs
                                if int(x) < n_epochs]

            ep_file = epoch_dir / f'{subject_name}_epochs.npy'
            if ep_file.exists():
                orig_epochs = np.load(ep_file)
            else:
                print(f'  ⚠️  epoch file missing for {subject_name}, skip Baccalá')
                bacc_ep_list = []
                orig_epochs  = None

        # ── Per-epoch loop ────────────────────────────────────────────────
        for ep in tqdm(ep_range, desc=f'  {subject_name}',
                       leave=False, unit='epoch'):
            lbl         = int(labels[ep])
            lbl_name    = LABEL_NAMES.get(lbl, 'unknown')
            t_val       = float(tfo[ep]) if tfo is not None else None
            order       = int(orders[ep])
            fname_stem  = f'ep{ep:03d}_{lbl_name}'

            dtf_bands = {b: dtf_bands_all[b][ep] for b in BANDS}
            pdc_bands = {b: pdc_bands_all[b][ep] for b in BANDS}

            # A) Heatmap
            if do_heatmap:
                plot_heatmap_pair(
                    dtf_bands['integrated'],
                    pdc_bands['integrated'],
                    subject_name, ep, lbl, t_val, order,
                    dirs['heat'] / f'{fname_stem}.png',
                )

            # B) Multi-band
            if do_multiband:
                plot_multiband_grid(
                    dtf_bands, pdc_bands,
                    subject_name, ep, lbl, t_val, order,
                    dirs['mband'] / f'{fname_stem}.png',
                )

            # D) Circular (every N epochs to avoid thousands of files)
            if do_circular and ep % args.circular_every == 0:
                plot_circular_diagram(
                    dtf_bands['integrated'],
                    subject_name, ep, lbl, t_val, order,
                    dirs['circ'] / f'{fname_stem}.png',
                    threshold=args.circ_threshold,
                )

            # F) Baccalá (selected epochs only)
            if do_baccala and ep in bacc_ep_list and orig_epochs is not None:
                orig_idx   = int(indices[ep])
                epoch_data = orig_epochs[orig_idx]
                data_std   = np.std(epoch_data)
                if data_std > 1e-10:
                    try:
                        res = VAR((epoch_data / data_std).T).fit(
                            maxlags=fixed_order, trend='c', verbose=False)
                        if res.k_ar > 0:
                            dtf_s, pdc_s, freqs = compute_dtf_pdc_from_var(
                                res.coefs, fs=256.0, nfft=512)
                            plot_baccala_grid(
                                dtf_s, pdc_s, freqs,
                                subject_name, ep, lbl, t_val, order,
                                dirs['bacc'],
                            )
                    except Exception as exc:
                        print(f'    Baccalá ep{ep} failed: {exc}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"=" * 72}')
    print(f'  Done.  All figures saved to: {out_dir}')
    print(f'{"=" * 72}')
    print()
    print('  Output structure per subject:')
    print('    heatmap/          ep000_preictal.png  ...')
    print('    multiband/        ep000_preictal.png  ...')
    print('    channel_timeline/ dtf_timeline.png  pdc_timeline.png')
    print('    circular/         ep000_preictal.png  (every N epochs)')
    print('    band_bars/        dtf_band_bars.png  pdc_band_bars.png')
    print('    baccala/          ep000_preictal_dtf.png  ...')
    print('    timeline.png      (mean connectivity over time)')
    print()
    print('  Per-channel convention:')
    print('    outflow[ch] = mean DTF over sinks   = how strongly ch drives others')
    print('    inflow[ch]  = mean DTF over sources = how strongly others drive ch')


if __name__ == '__main__':
    main()