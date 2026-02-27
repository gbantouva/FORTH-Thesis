"""
Step 2b - Pairwise Temporal Connectivity Evolution
====================================================
Visualises how connectivity between PAIRS of channels evolves over time,
centred on seizure onset (time 0 = first ictal epoch).

MOTIVATION (professor's feedback):
-----------------------------------
Total strength (out + in collapsed to one number) can mask the true picture:
- A channel with high out AND high in looks the same as one with both moderate
- Out and in can counteract each other, hiding directional information
- We lose the PAIR relationship: who is sending to whom specifically?

THIS SCRIPT INSTEAD:
--------------------
For each channel X, it produces TWO plots:

  1. OUTFLOW plot — "What does channel X send to each other channel over time?"
     → connectivity[:, :, X]  (column X of matrix = what X sends OUT)
     → 18 lines, one per target channel
     → Bright line at seizure onset = X starts DRIVING that target

  2. INFLOW plot  — "What does channel X receive from each other channel over time?"
     → connectivity[:, X, :]  (row X of matrix = what X receives IN)
     → 18 lines, one per source channel
     → Bright line at seizure onset = that source starts DRIVING X

Matrix convention reminder:
  matrix[i, j] = connectivity FROM channel j TO channel i
  → row i = all sources feeding INTO channel i  (inflow  of i)
  → col j = all targets that channel j feeds   (outflow of j)

Usage:
------
python step2b_pairwise_temporal.py \
    --connectivity_dir F:/FORTH_Final_Thesis/FORTH-Thesis/connectivity \
    --output_dir F:/FORTH_Final_Thesis/FORTH-Thesis/figures/pairwise_temporal \
    --band integrated \
    --metric pdc \
    --subject_ids 22
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for saving files
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# CHANNEL DEFINITIONS
# ==============================================================================

CHANNELS = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2'
]
N_CH = len(CHANNELS)   # 19

# Anatomical groups — used to colour-code lines in pairwise plots
# so you can immediately see if e.g. all temporal channels jump together
CHANNEL_GROUPS = {
    'Frontal':   ['Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8'],
    'Temporal':  ['T3',  'T4',  'T5',  'T6'],
    'Central':   ['C3',  'Cz',  'C4'],
    'Parietal':  ['P3',  'Pz',  'P4'],
    'Occipital': ['O1',  'O2'],
}

# Assign a colour family to each group so lines from the same brain region
# share a similar hue — makes the plots much easier to read
GROUP_COLOURS = {
    'Frontal':   'Blues',
    'Temporal':  'Reds',
    'Central':   'Greens',
    'Parietal':  'Purples',
    'Occipital': 'Oranges',
}

def channel_to_group(ch_name):
    """Return the anatomical group name for a given channel."""
    for group, members in CHANNEL_GROUPS.items():
        if ch_name in members:
            return group
    return 'Other'


def build_colour_map():
    """
    Build a dict: channel_name -> RGB colour.
    Channels in the same anatomical group get similar hues so they are
    visually grouped in the pairwise plots.
    """
    colour_map = {}
    for group, members in CHANNEL_GROUPS.items():
        cmap = cm.get_cmap(GROUP_COLOURS[group])
        n = len(members)
        for k, ch in enumerate(members):
            # Use the middle range of the colormap (0.4–0.9) to avoid
            # very light or very dark shades that are hard to see
            colour_map[ch] = cmap(0.4 + 0.5 * k / max(n - 1, 1))
    return colour_map

COLOUR_MAP = build_colour_map()


# ==============================================================================
# TIME AXIS HELPER
# ==============================================================================

def build_time_axis(labels):
    """
    Build a time array (seconds) where time=0 is the FIRST ictal epoch.

    Parameters
    ----------
    labels : np.ndarray, shape (n_epochs,)
        0 = control/pre-ictal, 1 = ictal

    Returns
    -------
    time : np.ndarray, shape (n_epochs,)
        Time in seconds relative to seizure onset
    first_ictal_idx : int
        Index of the first ictal epoch in the filtered data
    last_ictal_idx : int
        Index of the last ictal epoch in the filtered data

    Notes
    -----
    Each epoch is 4 seconds (1024 samples at 256 Hz).
    We use the ACTUAL filtered epochs (after step2 discarded bad epochs),
    NOT the original raw epoch numbering, to avoid time alignment errors
    caused by discarded epochs.
    """
    ictal_indices = np.where(labels == 1)[0]

    if len(ictal_indices) == 0:
        raise ValueError("No ictal epochs found in this subject file.")

    first_ictal_idx = ictal_indices[0]
    last_ictal_idx  = ictal_indices[-1]

    # Each epoch = 4 seconds
    time = np.arange(len(labels)) * 4.0

    # Shift so that time[first_ictal_idx] = 0
    time = time - time[first_ictal_idx]

    return time, first_ictal_idx, last_ictal_idx


# ==============================================================================
# PAIRWISE OUTFLOW PLOT
# ==============================================================================

def plot_outflow(channel_idx, connectivity, time, first_ictal_idx,
                 last_ictal_idx, subject_name, band, metric, output_dir):
    """
    For a given SOURCE channel, plot how much it sends to EACH OTHER channel
    over time.

    What we extract from the matrix:
        connectivity[:, :, channel_idx]
        → shape (n_epochs, 19)
        → for each epoch, column `channel_idx` of the 19×19 matrix
        → column j tells us "how much channel j sends to every other channel"
        → so this gives us the outflow FROM `channel_idx` TO every other channel

    Parameters
    ----------
    channel_idx : int
        Index of the source channel (0–18)
    connectivity : np.ndarray, shape (n_epochs, 19, 19)
        Full connectivity matrices over time
    time : np.ndarray, shape (n_epochs,)
        Time axis in seconds (0 = seizure onset)
    first_ictal_idx, last_ictal_idx : int
        Epoch indices of seizure start and end
    subject_name, band, metric : str
        For titles and filenames
    output_dir : Path
        Where to save the figure
    """
    src_name = CHANNELS[channel_idx]

    # -------------------------------------------------------------------------
    # Extract outflow: for every epoch, take column `channel_idx`
    # Result shape: (n_epochs, 19)
    # outflow[t, i] = connectivity FROM src TO channel i at epoch t
    # -------------------------------------------------------------------------
    outflow = connectivity[:, :, channel_idx]   # shape (n_epochs, 19)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot one line per TARGET channel (skip self — diagonal was zeroed in step2)
    for tgt_idx in range(N_CH):
        if tgt_idx == channel_idx:
            continue   # skip self-connection (always 0)

        tgt_name  = CHANNELS[tgt_idx]
        colour    = COLOUR_MAP[tgt_name]
        group     = channel_to_group(tgt_name)

        ax.plot(time, outflow[:, tgt_idx],
                color=colour,
                linewidth=1.2,
                alpha=0.8,
                label=f'{tgt_name} ({group})')

    # Shade the ictal period in light red
    ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
               alpha=0.15, color='red', zorder=0, label='Ictal period')

    # Vertical dashed line at seizure onset (time = 0)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5,
               alpha=0.9, label='Seizure onset (t=0)')

    # Labels and title
    ax.set_xlabel('Time relative to seizure onset (seconds)\n'
                  '(negative = pre-ictal, positive = ictal)',
                  fontsize=11)
    ax.set_ylabel(f'{metric.upper()} value\n(0 = no directed influence, 1 = maximum)',
                  fontsize=11)
    ax.set_title(
        f'{subject_name}  |  OUTFLOW from  {src_name}  →  all other channels\n'
        f'Metric: {metric.upper()}  |  Band: {band}\n'
        f'Each line = how much {src_name} sends to one specific target channel.\n'
        f'A line that RISES at t=0 means {src_name} starts DRIVING that channel at seizure onset.',
        fontsize=11, fontweight='bold'
    )

    # Colour-coded legend grouped by brain region
    # Put it outside the plot so it doesn't cover the lines
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=8, ncol=1, title='Target channel\n(colour = brain region)',
              title_fontsize=9, framealpha=0.9)

    ax.grid(alpha=0.3)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()

    fname = output_dir / f'{subject_name}_OUTFLOW_{src_name}_{metric}_{band}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# PAIRWISE INFLOW PLOT
# ==============================================================================

def plot_inflow(channel_idx, connectivity, time, first_ictal_idx,
                last_ictal_idx, subject_name, band, metric, output_dir):
    """
    For a given SINK channel, plot how much it receives FROM EACH OTHER channel
    over time.

    What we extract from the matrix:
        connectivity[:, channel_idx, :]
        → shape (n_epochs, 19)
        → for each epoch, row `channel_idx` of the 19×19 matrix
        → row i tells us "how much every other channel sends TO channel i"
        → so this gives us the inflow INTO `channel_idx` FROM every other channel

    Parameters
    ----------
    channel_idx : int
        Index of the sink channel (0–18)
    (rest same as plot_outflow)
    """
    snk_name = CHANNELS[channel_idx]

    # -------------------------------------------------------------------------
    # Extract inflow: for every epoch, take row `channel_idx`
    # Result shape: (n_epochs, 19)
    # inflow[t, j] = connectivity FROM channel j TO snk at epoch t
    # -------------------------------------------------------------------------
    inflow = connectivity[:, channel_idx, :]   # shape (n_epochs, 19)

    fig, ax = plt.subplots(figsize=(14, 6))

    for src_idx in range(N_CH):
        if src_idx == channel_idx:
            continue

        src_name = CHANNELS[src_idx]
        colour   = COLOUR_MAP[src_name]
        group    = channel_to_group(src_name)

        ax.plot(time, inflow[:, src_idx],
                color=colour,
                linewidth=1.2,
                alpha=0.8,
                label=f'{src_name} ({group})')

    ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
               alpha=0.15, color='red', zorder=0, label='Ictal period')

    ax.axvline(0, color='red', linestyle='--', linewidth=1.5,
               alpha=0.9, label='Seizure onset (t=0)')

    ax.set_xlabel('Time relative to seizure onset (seconds)\n'
                  '(negative = pre-ictal, positive = ictal)',
                  fontsize=11)
    ax.set_ylabel(f'{metric.upper()} value\n(0 = no directed influence, 1 = maximum)',
                  fontsize=11)
    ax.set_title(
        f'{subject_name}  |  INFLOW into  {snk_name}  ←  all other channels\n'
        f'Metric: {metric.upper()}  |  Band: {band}\n'
        f'Each line = how much one specific source channel sends TO {snk_name}.\n'
        f'A line that RISES at t=0 means that source starts DRIVING {snk_name} at seizure onset.',
        fontsize=11, fontweight='bold'
    )

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=8, ncol=1, title='Source channel\n(colour = brain region)',
              title_fontsize=9, framealpha=0.9)

    ax.grid(alpha=0.3)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()

    fname = output_dir / f'{subject_name}_INFLOW_{snk_name}_{metric}_{band}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# PAIRWISE TOTAL PLOT
# ==============================================================================

def plot_total(channel_idx, connectivity, time, first_ictal_idx,
               last_ictal_idx, subject_name, band, metric, output_dir):
    """
    For a given channel, plot the TOTAL pairwise coupling with each other
    channel over time.

    Total for pair (channel_idx ↔ Y) at time t:
        total[t, Y] = connectivity[t, Y, channel_idx]   (channel → Y, outflow)
                    + connectivity[t, channel_idx, Y]   (Y → channel, inflow)

    This is the sum of both directed connections between the pair.
    It tells you HOW STRONGLY the two channels are coupled overall,
    without collapsing the directional information (you still get one line
    per partner, unlike the old total_strength which summed everything).

    Use this alongside outflow and inflow to interpret:
      - High total, high outflow, low inflow  → channel is DRIVING the partner
      - High total, low outflow, high inflow  → partner is DRIVING the channel
      - High total, both outflow and inflow   → bidirectional coupling
    """
    ch_name = CHANNELS[channel_idx]

    # outflow: connectivity[:, :, channel_idx]  shape (n_epochs, 19)
    # inflow:  connectivity[:, channel_idx, :]  shape (n_epochs, 19)
    outflow = connectivity[:, :, channel_idx]
    inflow  = connectivity[:, channel_idx, :]
    total   = outflow + inflow   # shape (n_epochs, 19)

    fig, ax = plt.subplots(figsize=(14, 6))

    for pair_idx in range(N_CH):
        if pair_idx == channel_idx:
            continue
        pair_name = CHANNELS[pair_idx]
        colour    = COLOUR_MAP[pair_name]
        group     = channel_to_group(pair_name)

        ax.plot(time, total[:, pair_idx],
                color=colour,
                linewidth=1.2,
                alpha=0.8,
                label=f'{pair_name} ({group})')

    ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
               alpha=0.15, color='red', zorder=0, label='Ictal period')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5,
               alpha=0.9, label='Seizure onset (t=0)')

    ax.set_xlabel('Time relative to seizure onset (seconds)\n'
                  '(negative = pre-ictal, positive = ictal)',
                  fontsize=11)
    ax.set_ylabel(f'{metric.upper()} total (outflow + inflow, max=2)\n'
                  '0 = no coupling, 2 = maximum bidirectional coupling',
                  fontsize=11)
    ax.set_title(
        f'{subject_name}  |  TOTAL coupling  {ch_name} ↔  each other channel\n'
        f'Metric: {metric.upper()}  |  Band: {band}\n'
        f'Each line = outflow({ch_name}→partner) + inflow(partner→{ch_name})\n'
        f'Rising line at t=0 = strong coupling between {ch_name} and that partner during seizure.\n'
        f'Compare with OUTFLOW and INFLOW plots to determine who is driving whom.',
        fontsize=11, fontweight='bold'
    )

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=8, ncol=1,
              title='Partner channel\n(colour = brain region)',
              title_fontsize=9, framealpha=0.9)

    ax.grid(alpha=0.3)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([-0.02, None])   # total can exceed 1.0 so let it float upward

    plt.tight_layout()

    fname = output_dir / f'{subject_name}_TOTAL_{ch_name}_{metric}_{band}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

def plot_summary_grid(connectivity, labels, time, first_ictal_idx,
                      last_ictal_idx, subject_name, band, metric, output_dir):
    """
    One overview figure with 19 rows × 3 columns.
    Col 0 = outflow  (mean of what each channel SENDS across all partners)
    Col 1 = inflow   (mean of what each channel RECEIVES across all partners)
    Col 2 = total    (mean outflow + mean inflow per channel)

    Each subplot is one averaged time series — not pairwise detail.
    Use the focused pair plot for pairwise detail.

    All 19 channels share the same y-axis so amplitudes are directly comparable.
    Channels that become drivers at t=0 will show a rising outflow line.
    Channels that become sinks will show a rising inflow line.
    Channels with both rising = bidirectionally coupled during seizure.
    """
    fig, axes = plt.subplots(N_CH, 3, figsize=(20, N_CH * 1.8), sharex=True)

    # ------------------------------------------------------------------
    # Compute mean outflow and inflow per channel per epoch
    #
    # connectivity shape: (n_epochs, 19, 19)  [epoch, sink, source]
    #
    # Mean outflow of channel j at epoch t:
    #   average over all sinks i of connectivity[t, i, j]
    #   = np.mean(connectivity[:, :, j], axis=1)  for each j
    #   = np.mean(connectivity, axis=1)  → shape (n_epochs, 19)
    #     where result[:, j] = mean outflow of channel j
    #
    # Mean inflow of channel i at epoch t:
    #   average over all sources j of connectivity[t, i, j]
    #   = np.mean(connectivity[:, i, :], axis=1)  for each i
    #   = np.mean(connectivity, axis=2)  → shape (n_epochs, 19)
    #     where result[:, i] = mean inflow of channel i
    # ------------------------------------------------------------------
    mean_outflow = np.mean(connectivity, axis=1)   # (n_epochs, 19) — col mean
    mean_inflow  = np.mean(connectivity, axis=2)   # (n_epochs, 19) — row mean
    mean_total   = mean_outflow + mean_inflow       # (n_epochs, 19)

    # Shared y limits across ALL channels AND all three columns
    global_ymax = np.max(mean_total) * 1.1
    global_ymin = -0.01

    for ch_idx in range(N_CH):
        ch_name = CHANNELS[ch_idx]
        colour  = COLOUR_MAP[ch_name]

        ax_out   = axes[ch_idx, 0]
        ax_in    = axes[ch_idx, 1]
        ax_total = axes[ch_idx, 2]

        for ax, values, ls in [
            (ax_out,   mean_outflow[:, ch_idx], '-'),
            (ax_in,    mean_inflow[:, ch_idx],  '--'),
            (ax_total, mean_total[:, ch_idx],   '-'),
        ]:
            ax.plot(time, values, color=colour, linewidth=1.5, linestyle=ls)
            ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
                       alpha=0.15, color='red', zorder=0)
            ax.axvline(0, color='red', linestyle='--',
                       linewidth=1.0, alpha=0.7)
            ax.set_ylim([global_ymin, global_ymax])
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)

        # Channel name label on the left
        ax_out.set_ylabel(ch_name, fontsize=8, rotation=0,
                          labelpad=30, fontstyle='italic')

    # Column headers on top row only
    axes[0, 0].set_title(
        'OUTFLOW (mean sent)\nRising = channel becomes DRIVER',
        fontsize=10, fontweight='bold', pad=6
    )
    axes[0, 1].set_title(
        'INFLOW (mean received)\nRising = channel becomes SINK/FOLLOWER',
        fontsize=10, fontweight='bold', pad=6
    )
    axes[0, 2].set_title(
        'TOTAL = outflow + inflow\nRising = overall coupling increases',
        fontsize=10, fontweight='bold', pad=6
    )

    # X axis labels on bottom row only
    for col in range(3):
        axes[-1, col].set_xlabel('Time from seizure onset (s)', fontsize=9)

    fig.suptitle(
        f'{subject_name}  |  All-channel connectivity overview '
        f'Metric: {metric.upper()}  |  Band: {band}  |  '
        f'Shared y-axis [{global_ymin:.2f}, {global_ymax:.2f}] '
        f'Solid = outflow / dashed = inflow  |  '
        f'Red shading = ictal period  |  Dashed red line = t=0 onset',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()

    fname = output_dir / f'{subject_name}_SUMMARY_{metric}_{band}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Summary grid saved: {fname.name}')


# ==============================================================================
# FOCUSED PAIR PLOT — one specific source to all targets side by side
# ==============================================================================

def plot_focused_pairs(focus_channels, connectivity, time, first_ictal_idx,
                       last_ictal_idx, subject_name, band, metric, output_dir):
    """
    For a SHORT LIST of focus channels (e.g. the suspected ictal focus),
    plot outflow, inflow, and total side by side in one figure.

    Layout: one row per focus channel, 3 columns:
      Col 0 — OUTFLOW : what the channel SENDS to each partner
      Col 1 — INFLOW  : what the channel RECEIVES from each partner
      Col 2 — TOTAL   : outflow + inflow per pair — overall coupling strength

    Why three columns?
    ------------------
    Outflow and inflow can move in opposite directions or at different times.
    Looking at them separately avoids the masking problem (they no longer
    cancel each other). The TOTAL column adds context: if both outflow and
    inflow rise together for a specific pair, total will be highest there —
    confirming that pair is strongly coupled bidirectionally during the seizure.

    How TOTAL is computed per pair (channel X ↔ channel Y):
      total[t, Y] = outflow[t, Y]  +  inflow[t, Y]
                  = connectivity[t, Y, X]  +  connectivity[t, X, Y]
      → the sum of both directed connections between the pair
      → symmetric by definition (same value whether you look from X or Y)

    All three columns share the same y-axis and x-axis for direct comparison.
    """
    n_focus = len(focus_channels)

    # 3 columns: outflow | inflow | total
    fig, axes = plt.subplots(n_focus, 3,
                             figsize=(22, 4 * n_focus),
                             sharex=True, sharey=True)

    if n_focus == 1:
        axes = axes[np.newaxis, :]   # always 2D array

    # Global y limits — total can reach up to 2.0 (outflow + inflow each max 1)
    # Use 99th percentile to avoid one outlier stretching the axis
    outflow_all = connectivity.sum(axis=1)   # shape (n_epochs, 19) — col sums
    inflow_all  = connectivity.sum(axis=2)   # shape (n_epochs, 19) — row sums
    global_ymax = np.percentile(outflow_all + inflow_all, 99) * 1.15
    global_ymin = -0.01

    for row, ch_name in enumerate(focus_channels):
        ch_idx = CHANNELS.index(ch_name)

        ax_out   = axes[row, 0]
        ax_in    = axes[row, 1]
        ax_total = axes[row, 2]

        # ------------------------------------------------------------------
        # OUTFLOW — column ch_idx of the connectivity matrix
        #
        # connectivity[:, :, ch_idx]  shape (n_epochs, 19)
        # outflow[t, i] = PDC/DTF value FROM ch_idx TO channel i at epoch t
        #
        # Reading: a line for channel T5 that rises at t=0 means
        # ch_idx starts sending strongly TO T5 when the seizure begins.
        # ------------------------------------------------------------------
        outflow = connectivity[:, :, ch_idx]   # (n_epochs, 19)

        for tgt_idx in range(N_CH):
            if tgt_idx == ch_idx:
                continue   # skip self (always 0 after diagonal zeroing)
            tgt_name = CHANNELS[tgt_idx]
            ax_out.plot(time, outflow[:, tgt_idx],
                        color=COLOUR_MAP[tgt_name],
                        linewidth=1.0, alpha=0.75,
                        label=tgt_name)

        ax_out.axvspan(time[first_ictal_idx], time[last_ictal_idx],
                       alpha=0.12, color='red', zorder=0)
        ax_out.axvline(0, color='red', linestyle='--', linewidth=1.5,
                       label='Seizure onset')
        ax_out.set_title(
            f'OUTFLOW  {ch_name} → others\n'
            f'What {ch_name} SENDS to each partner\n'
            f'Rising line = {ch_name} starts DRIVING that channel',
            fontsize=9, fontweight='bold'
        )
        ax_out.set_ylabel(f'{metric.upper()} value', fontsize=9)
        ax_out.set_ylim([global_ymin, global_ymax])
        ax_out.grid(alpha=0.3)
        ax_out.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      fontsize=7, ncol=1,
                      title='Target channel', title_fontsize=8)

        # ------------------------------------------------------------------
        # INFLOW — row ch_idx of the connectivity matrix
        #
        # connectivity[:, ch_idx, :]  shape (n_epochs, 19)
        # inflow[t, j] = PDC/DTF value FROM channel j TO ch_idx at epoch t
        #
        # Reading: a line for channel C3 that rises at t=0 means
        # C3 starts driving ch_idx when the seizure begins.
        # ------------------------------------------------------------------
        inflow = connectivity[:, ch_idx, :]    # (n_epochs, 19)

        for src_idx in range(N_CH):
            if src_idx == ch_idx:
                continue
            src_name = CHANNELS[src_idx]
            ax_in.plot(time, inflow[:, src_idx],
                       color=COLOUR_MAP[src_name],
                       linewidth=1.0, alpha=0.75,
                       label=src_name)

        ax_in.axvspan(time[first_ictal_idx], time[last_ictal_idx],
                      alpha=0.12, color='red', zorder=0)
        ax_in.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax_in.set_title(
            f'INFLOW  others → {ch_name}\n'
            f'What {ch_name} RECEIVES from each partner\n'
            f'Rising line = that channel starts DRIVING {ch_name}',
            fontsize=9, fontweight='bold'
        )
        ax_in.set_ylabel(f'{metric.upper()} value', fontsize=9)
        ax_in.set_ylim([global_ymin, global_ymax])
        ax_in.grid(alpha=0.3)
        ax_in.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                     fontsize=7, ncol=1,
                     title='Source channel', title_fontsize=8)

        # ------------------------------------------------------------------
        # TOTAL — outflow + inflow per pair
        #
        # For each partner channel Y:
        #   total[t, Y] = connectivity[t, Y, ch_idx]   (ch sends TO Y)
        #               + connectivity[t, ch_idx, Y]   (Y sends TO ch)
        #
        # This is the SUM of both directed connections between the pair.
        # It answers: "how strongly are ch_idx and Y coupled overall?"
        # regardless of direction.
        #
        # If TOTAL is high but OUTFLOW is low  → ch is mostly a RECEIVER
        # If TOTAL is high but INFLOW  is low  → ch is mostly a SENDER
        # If both OUTFLOW and INFLOW are high  → bidirectional coupling
        # ------------------------------------------------------------------
        total = outflow + inflow   # element-wise sum, shape (n_epochs, 19)
        # outflow[t, Y] = ch → Y
        # inflow[t, Y]  = Y → ch
        # total[t, Y]   = ch ↔ Y  (both directions combined)

        for pair_idx in range(N_CH):
            if pair_idx == ch_idx:
                continue
            pair_name = CHANNELS[pair_idx]
            ax_total.plot(time, total[:, pair_idx],
                          color=COLOUR_MAP[pair_name],
                          linewidth=1.0, alpha=0.75,
                          label=pair_name)

        ax_total.axvspan(time[first_ictal_idx], time[last_ictal_idx],
                         alpha=0.12, color='red', zorder=0)
        ax_total.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax_total.set_title(
            f'TOTAL  {ch_name} ↔ others  (outflow + inflow)\n'
            f'Overall coupling strength between {ch_name} and each partner\n'
            f'High TOTAL = strong pair coupling regardless of direction',
            fontsize=9, fontweight='bold'
        )
        ax_total.set_ylabel(f'{metric.upper()} value (max=2)', fontsize=9)
        ax_total.set_ylim([global_ymin, global_ymax])
        ax_total.grid(alpha=0.3)
        ax_total.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                        fontsize=7, ncol=1,
                        title='Partner channel', title_fontsize=8)

    # X axis labels on bottom row only
    axes[-1, 0].set_xlabel('Time from seizure onset (seconds)\n'
                           '(negative = pre-ictal, 0 = onset, positive = ictal)',
                           fontsize=9)
    axes[-1, 1].set_xlabel('Time from seizure onset (seconds)', fontsize=9)
    axes[-1, 2].set_xlabel('Time from seizure onset (seconds)', fontsize=9)

    focus_str = ', '.join(focus_channels)
    fig.suptitle(
        f'{subject_name}  |  Pairwise analysis: [{focus_str}]\n'
        f'Metric: {metric.upper()}  |  Band: {band}  |  '
        f'Shared axes — red shading = ictal — dashed line = t=0 onset\n'
        f'LEFT: outflow (what channel sends)  |  '
        f'MIDDLE: inflow (what channel receives)  |  '
        f'RIGHT: total = outflow + inflow (overall pair coupling)',
        fontsize=11, fontweight='bold'
    )

    plt.tight_layout()

    focus_tag = '_'.join(focus_channels)
    fname = output_dir / f'{subject_name}_FOCUSED_{focus_tag}_{metric}_{band}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Focused pair plot saved: {fname.name}')


# ==============================================================================
# PROCESS ONE SUBJECT
# ==============================================================================

def process_subject(graphs_file, output_dir, band, metric,
                    focus_channels, individual_plots):
    """
    Full pipeline for one subject:
      1. Load connectivity data
      2. Build time axis
      3. Generate summary grid
      4. Generate focused pair plots
      5. Optionally generate individual outflow/inflow per channel

    Parameters
    ----------
    graphs_file : Path
        Path to the subject's .npz connectivity file from step2
    output_dir : Path
        Where to save all figures
    band : str
        Frequency band key ('integrated', 'delta', etc.)
    metric : str
        'pdc' or 'dtf'
    focus_channels : list of str
        Channels to highlight in the focused pair plot
    individual_plots : bool
        If True, also save individual outflow/inflow plots for every channel
    """
    subject_name = Path(graphs_file).stem.replace('_graphs', '')
    subj_dir = output_dir / subject_name
    subj_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'Processing: {subject_name}')
    print(f'{"="*70}')

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = np.load(graphs_file)

    connectivity_key = f'{metric}_{band}'
    if connectivity_key not in data:
        print(f'  ❌ Key "{connectivity_key}" not found. '
              f'Available: {list(data.keys())}')
        return

    connectivity = data[connectivity_key]   # shape (n_epochs, 19, 19)
    labels       = data['labels']            # shape (n_epochs,)
    n_epochs     = len(labels)

    print(f'  Epochs (valid): {n_epochs}')
    print(f'  Ictal epochs:   {np.sum(labels == 1)}')
    print(f'  Control epochs: {np.sum(labels == 0)}')

    # ------------------------------------------------------------------
    # Build time axis
    # ------------------------------------------------------------------
    time, first_ictal_idx, last_ictal_idx = build_time_axis(labels)

    print(f'  Seizure onset:  epoch {first_ictal_idx} → t = 0s')
    print(f'  Seizure end:    epoch {last_ictal_idx}  '
          f'→ t = {time[last_ictal_idx]:.0f}s')
    print(f'  Pre-ictal range: {time[0]:.0f}s to 0s')
    print(f'  Ictal range:     0s to {time[last_ictal_idx]:.0f}s')
    if last_ictal_idx < n_epochs - 1:
        print(f'  Post-ictal range: {time[last_ictal_idx]:.0f}s '
              f'to {time[-1]:.0f}s')

    # ------------------------------------------------------------------
    # Plot 1: Summary grid (all 19 channels, outflow vs inflow)
    # ------------------------------------------------------------------
    print(f'\n  Generating summary grid...')
    plot_summary_grid(connectivity, labels, time,
                      first_ictal_idx, last_ictal_idx,
                      subject_name, band, metric, subj_dir)

    # ------------------------------------------------------------------
    # Plot 2: Focused pair plots for channels of interest
    # ------------------------------------------------------------------
    # Filter focus_channels to only those that actually exist
    valid_focus = [ch for ch in focus_channels if ch in CHANNELS]
    if valid_focus:
        print(f'  Generating focused pair plot for: {valid_focus}')
        plot_focused_pairs(valid_focus, connectivity, time,
                           first_ictal_idx, last_ictal_idx,
                           subject_name, band, metric, subj_dir)

    # ------------------------------------------------------------------
    # Plot 3 (optional): Individual outflow + inflow per channel
    # This produces 19×2 = 38 figures — useful for detailed inspection
    # ------------------------------------------------------------------
    if individual_plots:
        print(f'  Generating individual pairwise plots '
              f'(19 outflow + 19 inflow + 19 total)...')
        ind_dir = subj_dir / 'individual'
        ind_dir.mkdir(exist_ok=True)

        for ch_idx in range(N_CH):
            plot_outflow(ch_idx, connectivity, time,
                         first_ictal_idx, last_ictal_idx,
                         subject_name, band, metric, ind_dir)
            plot_inflow(ch_idx, connectivity, time,
                        first_ictal_idx, last_ictal_idx,
                        subject_name, band, metric, ind_dir)
            plot_total(ch_idx, connectivity, time,
                       first_ictal_idx, last_ictal_idx,
                       subject_name, band, metric, ind_dir)

        print(f'  ✅ Individual plots saved in: {ind_dir}')

    print(f'  ✅ Done — all plots in: {subj_dir}')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pairwise temporal connectivity evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--connectivity_dir', required=True,
                        help='Directory with subject_XX_graphs.npz files')
    parser.add_argument('--output_dir', required=True,
                        help='Where to save all figures')
    parser.add_argument('--band', default='integrated',
                        choices=['integrated','delta','theta',
                                 'alpha','beta','gamma1'],
                        help='Frequency band (default: integrated)')
    parser.add_argument('--metric', default='pdc',
                        choices=['pdc', 'dtf'],
                        help='Connectivity metric (default: pdc)')
    parser.add_argument('--subject_ids', nargs='+', type=int,
                        default=list(range(1, 35)),
                        help='Subject IDs to process (default: 1–34)')
    parser.add_argument('--focus_channels', nargs='+',
                        default=['C3', 'T3', 'T5'],
                        help='Channels for focused pair plots '
                             '(default: C3 T3 T5)')
    parser.add_argument('--individual_plots', action='store_true',
                        help='Also save individual outflow/inflow plots '
                             'per channel (38 extra files per subject)')

    args = parser.parse_args()

    connectivity_dir = Path(args.connectivity_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('STEP 2b — PAIRWISE TEMPORAL CONNECTIVITY EVOLUTION')
    print('=' * 70)
    print(f'Metric:          {args.metric.upper()}')
    print(f'Band:            {args.band}')
    print(f'Focus channels:  {args.focus_channels}')
    print(f'Subjects:        {len(args.subject_ids)}')
    print(f'Individual plots:{args.individual_plots}')
    print(f'Output:          {output_dir}')
    print('=' * 70)
    print()
    print('KEY:')
    print('  OUTFLOW  = connectivity[:, :, ch_idx]  '
          '(column = what ch sends OUT)')
    print('  INFLOW   = connectivity[:, ch_idx, :]  '
          '(row    = what ch receives IN)')
    print('  time 0   = first ictal epoch in filtered data')
    print('=' * 70)

    success = 0
    errors  = 0

    for subj_id in args.subject_ids:
        subject_name = f'subject_{subj_id:02d}'
        graphs_file  = connectivity_dir / f'{subject_name}_graphs.npz'

        if not graphs_file.exists():
            print(f'\n⚠  Skip {subject_name}: file not found')
            errors += 1
            continue

        try:
            process_subject(
                graphs_file     = graphs_file,
                output_dir      = output_dir,
                band            = args.band,
                metric          = args.metric,
                focus_channels  = args.focus_channels,
                individual_plots= args.individual_plots
            )
            success += 1
        except Exception as e:
            import traceback
            print(f'\n❌ Error on {subject_name}: {e}')
            traceback.print_exc()
            errors += 1

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'  ✅ Success: {success}')
    print(f'  ❌ Errors:  {errors}')
    print()
    print('FILES CREATED PER SUBJECT (in output_dir/subject_XX/):')
    print('  *_SUMMARY_*.png        — all 19 channels: outflow | inflow | total grid')
    print('  *_FOCUSED_*.png        — focused 3-column plots for channels of interest')
    print('  individual/*_OUTFLOW_* — (if --individual_plots) outflow per channel')
    print('  individual/*_INFLOW_*  — (if --individual_plots) inflow per channel')
    print('  individual/*_TOTAL_*   — (if --individual_plots) total per channel')
    print('=' * 70)


if __name__ == '__main__':
    main()