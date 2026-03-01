"""
Step 3ab - Pairwise Temporal Connectivity Evolution
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

Clinical mapping (from patients_and_subjects diagram):
  Subject 1         → PAT 11 → not in paper         (excluded)
  Subject 2         → PAT 13 → focal right frontal   F4, F8, Fp2
  Subjects 3–10     → PAT 14 → focal right frontal   F4, F8, Fp2
  Subject 11        → PAT 15 → focal fronto-polar     Fp1, Fp2
  Subjects 12–25    → PAT 24 → focal left frontal     F3, F7, Fp1
  Subjects 26–32    → PAT 27 → focal right frontal    F4, F8, Fp2
  Subject 33        → PAT 29 → focal bifrontal        F3, F4, Fz, Fp1, Fp2
  Subject 34        → PAT 35 → focal (unspecified)    Fp1, Fp2, F3, F4, Fz

Usage:
------
python step3ab_pairwise_temporal.py \
    --connectivity_dir F:/FORTH_Final_Thesis/FORTH-Thesis/connectivity \
    --output_dir F:/FORTH_Final_Thesis/FORTH-Thesis/figures/pairwise_temporal \
    --band integrated \
    --metric pdc \
    --subject_ids 2 3 4
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# CLINICAL FOCUS CHANNEL MAPPING
# Subject ID → clinically expected focus channels
# Based on patients_and_subjects diagram
# ==============================================================================

CLINICAL_FOCUS = {
    1:  None,                              # PAT 11 — not in paper, skip
    2:  ['F4', 'F8', 'Fp2'],              # PAT 13 — right frontal
    3:  ['F4', 'F8', 'Fp2'],              # PAT 14 — right frontal
    4:  ['F4', 'F8', 'Fp2'],
    5:  ['F4', 'F8', 'Fp2'],
    6:  ['F4', 'F8', 'Fp2'],
    7:  ['F4', 'F8', 'Fp2'],
    8:  ['F4', 'F8', 'Fp2'],
    9:  ['F4', 'F8', 'Fp2'],
    10: ['F4', 'F8', 'Fp2'],
    11: ['Fp1', 'Fp2'],                   # PAT 15 — fronto-polar
    12: ['F3', 'F7', 'Fp1'],             # PAT 24 — left frontal
    13: ['F3', 'F7', 'Fp1'],
    14: ['F3', 'F7', 'Fp1'],
    15: ['F3', 'F7', 'Fp1'],
    16: ['F3', 'F7', 'Fp1'],
    17: ['F3', 'F7', 'Fp1'],
    18: ['F3', 'F7', 'Fp1'],
    19: ['F3', 'F7', 'Fp1'],
    20: ['F3', 'F7', 'Fp1'],
    21: ['F3', 'F7', 'Fp1'],
    22: ['F3', 'F7', 'Fp1'],
    23: ['F3', 'F7', 'Fp1'],
    24: ['F3', 'F7', 'Fp1'],
    25: ['F3', 'F7', 'Fp1'],
    26: ['F4', 'F8', 'Fp2'],             # PAT 27 — right frontal
    27: ['F4', 'F8', 'Fp2'],
    28: ['F4', 'F8', 'Fp2'],
    29: ['F4', 'F8', 'Fp2'],
    30: ['F4', 'F8', 'Fp2'],
    31: ['F4', 'F8', 'Fp2'],
    32: ['F4', 'F8', 'Fp2'],
    33: ['F3', 'F4', 'Fz', 'Fp1', 'Fp2'],  # PAT 29 — bifrontal
    34: ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],  # PAT 35 — focal unspecified
}

# Human-readable diagnosis per subject for plot titles
CLINICAL_DIAGNOSIS = {
    1:  'Not in paper',
    2:  'Focal right frontal (PAT 13)',
    3:  'Focal right frontal (PAT 14)',
    4:  'Focal right frontal (PAT 14)',
    5:  'Focal right frontal (PAT 14)',
    6:  'Focal right frontal (PAT 14)',
    7:  'Focal right frontal (PAT 14)',
    8:  'Focal right frontal (PAT 14)',
    9:  'Focal right frontal (PAT 14)',
    10: 'Focal right frontal (PAT 14)',
    11: 'Focal fronto-polar (PAT 15)',
    12: 'Focal left frontal (PAT 24)',
    13: 'Focal left frontal (PAT 24)',
    14: 'Focal left frontal (PAT 24)',
    15: 'Focal left frontal (PAT 24)',
    16: 'Focal left frontal (PAT 24)',
    17: 'Focal left frontal (PAT 24)',
    18: 'Focal left frontal (PAT 24)',
    19: 'Focal left frontal (PAT 24)',
    20: 'Focal left frontal (PAT 24)',
    21: 'Focal left frontal (PAT 24)',
    22: 'Focal left frontal (PAT 24)',
    23: 'Focal left frontal (PAT 24)',
    24: 'Focal left frontal (PAT 24)',
    25: 'Focal left frontal (PAT 24)',
    26: 'Focal right frontal (PAT 27)',
    27: 'Focal right frontal (PAT 27)',
    28: 'Focal right frontal (PAT 27)',
    29: 'Focal right frontal (PAT 27)',
    30: 'Focal right frontal (PAT 27)',
    31: 'Focal right frontal (PAT 27)',
    32: 'Focal right frontal (PAT 27)',
    33: 'Focal bifrontal (PAT 29)',
    34: 'Focal unspecified (PAT 35)',
}


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

CHANNEL_GROUPS = {
    'Frontal':   ['Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8'],
    'Temporal':  ['T3',  'T4',  'T5',  'T6'],
    'Central':   ['C3',  'Cz',  'C4'],
    'Parietal':  ['P3',  'Pz',  'P4'],
    'Occipital': ['O1',  'O2'],
}

GROUP_COLOURS = {
    'Frontal':   'Blues',
    'Temporal':  'Reds',
    'Central':   'Greens',
    'Parietal':  'Purples',
    'Occipital': 'Oranges',
}

def channel_to_group(ch_name):
    for group, members in CHANNEL_GROUPS.items():
        if ch_name in members:
            return group
    return 'Other'


def build_colour_map():
    colour_map = {}
    for group, members in CHANNEL_GROUPS.items():
        cmap = cm.get_cmap(GROUP_COLOURS[group])
        n = len(members)
        for k, ch in enumerate(members):
            colour_map[ch] = cmap(0.4 + 0.5 * k / max(n - 1, 1))
    return colour_map

COLOUR_MAP = build_colour_map()


# ==============================================================================
# TIME AXIS HELPER
# ==============================================================================

def build_time_axis(labels, data):
    """
    Build time array (seconds) where time=0 is the first ictal epoch.
    Uses time_from_onset stored in the .npz if available (correct alignment
    even when bad epochs were discarded). Falls back to reconstruction.
    """
    ictal_indices = np.where(labels == 1)[0]
    if len(ictal_indices) == 0:
        raise ValueError("No ictal epochs found in this subject file.")

    first_ictal_idx = ictal_indices[0]
    last_ictal_idx  = ictal_indices[-1]

    if 'time_from_onset' in data:
        time = data['time_from_onset']
    else:
        # Fallback: assumes contiguous epochs (use only if npz has no time array)
        time = np.arange(len(labels)) * 4.0
        time = time - time[first_ictal_idx]

    return time, first_ictal_idx, last_ictal_idx


# ==============================================================================
# PAIRWISE OUTFLOW PLOT
# ==============================================================================

def plot_outflow(channel_idx, connectivity, time, first_ictal_idx,
                 last_ictal_idx, subject_name, band, metric, output_dir,
                 diagnosis='', focus_channels=None):
    """
    For a given SOURCE channel, plot how much it sends to each other channel.

    connectivity[:, :, channel_idx]  →  outflow FROM channel_idx TO each sink
    """
    src_name   = CHANNELS[channel_idx]
    is_focus   = focus_channels and src_name in focus_channels
    title_tag  = '  ★ CLINICAL FOCUS CHANNEL ★' if is_focus else ''

    outflow = connectivity[:, :, channel_idx]   # (n_epochs, 19)

    fig, ax = plt.subplots(figsize=(14, 6))

    for tgt_idx in range(N_CH):
        if tgt_idx == channel_idx:
            continue
        tgt_name = CHANNELS[tgt_idx]
        lw       = 2.0 if (focus_channels and tgt_name in focus_channels) else 1.2
        alpha    = 0.95 if (focus_channels and tgt_name in focus_channels) else 0.6

        ax.plot(time, outflow[:, tgt_idx],
                color=COLOUR_MAP[tgt_name],
                linewidth=lw, alpha=alpha,
                label=f'{tgt_name} ({channel_to_group(tgt_name)})')

    ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
               alpha=0.15, color='red', zorder=0, label='Ictal period')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5,
               alpha=0.9, label='Seizure onset (t=0)')

    ax.set_xlabel('Time relative to seizure onset (s)', fontsize=11)
    ax.set_ylabel(f'{metric.upper()} value  [0–1]', fontsize=11)
    ax.set_title(
        f'{subject_name}  |  {diagnosis}\n'
        f'OUTFLOW from {src_name}{title_tag}  →  all other channels\n'
        f'Metric: {metric.upper()}  |  Band: {band}\n'
        f'Rising line = {src_name} starts DRIVING that channel at seizure onset',
        fontsize=10, fontweight='bold'
    )
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=8, ncol=1,
              title='Target  (colour = region)', title_fontsize=9)
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
                last_ictal_idx, subject_name, band, metric, output_dir,
                diagnosis='', focus_channels=None):
    """
    For a given SINK channel, plot how much it receives from each other channel.

    connectivity[:, channel_idx, :]  →  inflow INTO channel_idx FROM each source
    """
    snk_name  = CHANNELS[channel_idx]
    is_focus  = focus_channels and snk_name in focus_channels
    title_tag = '  ★ CLINICAL FOCUS CHANNEL ★' if is_focus else ''

    inflow = connectivity[:, channel_idx, :]   # (n_epochs, 19)

    fig, ax = plt.subplots(figsize=(14, 6))

    for src_idx in range(N_CH):
        if src_idx == channel_idx:
            continue
        src_name = CHANNELS[src_idx]
        lw    = 2.0 if (focus_channels and src_name in focus_channels) else 1.2
        alpha = 0.95 if (focus_channels and src_name in focus_channels) else 0.6

        ax.plot(time, inflow[:, src_idx],
                color=COLOUR_MAP[src_name],
                linewidth=lw, alpha=alpha,
                label=f'{src_name} ({channel_to_group(src_name)})')

    ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
               alpha=0.15, color='red', zorder=0, label='Ictal period')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5,
               alpha=0.9, label='Seizure onset (t=0)')

    ax.set_xlabel('Time relative to seizure onset (s)', fontsize=11)
    ax.set_ylabel(f'{metric.upper()} value  [0–1]', fontsize=11)
    ax.set_title(
        f'{subject_name}  |  {diagnosis}\n'
        f'INFLOW into {snk_name}{title_tag}  ←  all other channels\n'
        f'Metric: {metric.upper()}  |  Band: {band}\n'
        f'Rising line = that source starts DRIVING {snk_name} at seizure onset',
        fontsize=10, fontweight='bold'
    )
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=8, ncol=1,
              title='Source  (colour = region)', title_fontsize=9)
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
               last_ictal_idx, subject_name, band, metric, output_dir,
               diagnosis='', focus_channels=None):
    """
    Total pairwise coupling = outflow + inflow per partner channel.

    total[t, Y] = connectivity[t, Y, ch_idx]   (ch → Y)
                + connectivity[t, ch_idx, Y]   (Y → ch)
    """
    ch_name   = CHANNELS[channel_idx]
    is_focus  = focus_channels and ch_name in focus_channels
    title_tag = '  ★ CLINICAL FOCUS CHANNEL ★' if is_focus else ''

    outflow = connectivity[:, :, channel_idx]
    inflow  = connectivity[:, channel_idx, :]
    total   = outflow + inflow   # (n_epochs, 19)

    fig, ax = plt.subplots(figsize=(14, 6))

    for pair_idx in range(N_CH):
        if pair_idx == channel_idx:
            continue
        pair_name = CHANNELS[pair_idx]
        lw    = 2.0 if (focus_channels and pair_name in focus_channels) else 1.2
        alpha = 0.95 if (focus_channels and pair_name in focus_channels) else 0.6

        ax.plot(time, total[:, pair_idx],
                color=COLOUR_MAP[pair_name],
                linewidth=lw, alpha=alpha,
                label=f'{pair_name} ({channel_to_group(pair_name)})')

    ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
               alpha=0.15, color='red', zorder=0, label='Ictal period')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5,
               alpha=0.9, label='Seizure onset (t=0)')

    ax.set_xlabel('Time relative to seizure onset (s)', fontsize=11)
    ax.set_ylabel(f'{metric.upper()} total (outflow + inflow, max=2)', fontsize=11)
    ax.set_title(
        f'{subject_name}  |  {diagnosis}\n'
        f'TOTAL coupling  {ch_name}{title_tag} ↔ each other channel\n'
        f'Metric: {metric.upper()}  |  Band: {band}\n'
        f'Rising line = strong coupling between {ch_name} and that partner during seizure',
        fontsize=10, fontweight='bold'
    )
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=8, ncol=1,
              title='Partner  (colour = region)', title_fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([-0.02, None])
    plt.tight_layout()

    fname = output_dir / f'{subject_name}_TOTAL_{ch_name}_{metric}_{band}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# SUMMARY GRID  (19 rows × 3 columns — mean outflow | inflow | total)
# ==============================================================================

def plot_summary_grid(connectivity, time, first_ictal_idx, last_ictal_idx,
                      subject_name, band, metric, output_dir,
                      diagnosis='', focus_channels=None):
    """
    Overview: 19 rows (channels) × 3 columns (mean outflow | inflow | total).
    Clinical focus channels are highlighted with a bold coloured background.
    Shared y-axis across all channels for direct comparison.
    """
    # mean outflow of channel j = mean over sinks  = mean(connectivity, axis=1)[:, j]
    # mean inflow  of channel i = mean over sources = mean(connectivity, axis=2)[:, i]
    mean_outflow = np.mean(connectivity, axis=1)   # (n_epochs, 19)
    mean_inflow  = np.mean(connectivity, axis=2)   # (n_epochs, 19)
    mean_total   = mean_outflow + mean_inflow       # (n_epochs, 19)

    global_ymax = np.max(mean_total) * 1.1
    global_ymin = -0.01

    fig, axes = plt.subplots(N_CH, 3, figsize=(20, N_CH * 1.8), sharex=True)

    for ch_idx in range(N_CH):
        ch_name   = CHANNELS[ch_idx]
        colour    = COLOUR_MAP[ch_name]
        is_focus  = focus_channels and ch_name in focus_channels

        ax_out   = axes[ch_idx, 0]
        ax_in    = axes[ch_idx, 1]
        ax_total = axes[ch_idx, 2]

        for ax, values, ls in [
            (ax_out,   mean_outflow[:, ch_idx], '-'),
            (ax_in,    mean_inflow[:, ch_idx],  '--'),
            (ax_total, mean_total[:, ch_idx],   '-'),
        ]:
            # Highlight clinical focus channels with a light yellow background
            if is_focus:
                ax.set_facecolor('#fffde7')

            ax.plot(time, values, color=colour,
                    linewidth=2.0 if is_focus else 1.2,
                    linestyle=ls)
            ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
                       alpha=0.15, color='red', zorder=0)
            ax.axvline(0, color='red', linestyle='--',
                       linewidth=1.0, alpha=0.7)
            ax.set_ylim([global_ymin, global_ymax])
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)

        # Channel label — add star for focus channels
        label_text = f'★ {ch_name}' if is_focus else ch_name
        ax_out.set_ylabel(label_text, fontsize=8, rotation=0,
                          labelpad=32, fontstyle='italic',
                          fontweight='bold' if is_focus else 'normal',
                          color='darkorange' if is_focus else 'black')

    axes[0, 0].set_title('OUTFLOW (mean sent)\nRising = channel becomes DRIVER',
                          fontsize=10, fontweight='bold', pad=6)
    axes[0, 1].set_title('INFLOW (mean received)\nRising = channel becomes SINK',
                          fontsize=10, fontweight='bold', pad=6)
    axes[0, 2].set_title('TOTAL = outflow + inflow\nRising = overall coupling increases',
                          fontsize=10, fontweight='bold', pad=6)

    for col in range(3):
        axes[-1, col].set_xlabel('Time from seizure onset (s)', fontsize=9)

    focus_str = ', '.join(focus_channels) if focus_channels else 'none'
    fig.suptitle(
        f'{subject_name}  |  {diagnosis}\n'
        f'Metric: {metric.upper()}  |  Band: {band}  |  '
        f'Clinical focus: {focus_str}  (★ highlighted)\n'
        f'Shared y-axis [{global_ymin:.2f}, {global_ymax:.2f}]  |  '
        f'Red shading = ictal  |  Dashed red = t=0 onset',
        fontsize=11, fontweight='bold'
    )

    plt.tight_layout()
    fname = output_dir / f'{subject_name}_SUMMARY_{metric}_{band}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname.name}')


# ==============================================================================
# FOCUSED PAIR PLOT
# ==============================================================================

def plot_focused_pairs(focus_channels, connectivity, time, first_ictal_idx,
                       last_ictal_idx, subject_name, band, metric, output_dir,
                       diagnosis=''):
    """
    For the clinically expected focus channels, plot outflow | inflow | total
    side by side — one row per focus channel.

    This directly tests whether the clinical focus shows connectivity changes
    at seizure onset as expected.
    """
    n_focus = len(focus_channels)
    fig, axes = plt.subplots(n_focus, 3,
                             figsize=(22, 4 * n_focus),
                             sharex=True, sharey=True)
    if n_focus == 1:
        axes = axes[np.newaxis, :]

    # y limits from 99th percentile of total to avoid outlier stretching
    outflow_all = connectivity.sum(axis=1)
    inflow_all  = connectivity.sum(axis=2)
    global_ymax = np.percentile(outflow_all + inflow_all, 99) * 1.15
    global_ymin = -0.01

    for row, ch_name in enumerate(focus_channels):
        ch_idx   = CHANNELS.index(ch_name)
        ax_out   = axes[row, 0]
        ax_in    = axes[row, 1]
        ax_total = axes[row, 2]

        outflow = connectivity[:, :, ch_idx]   # (n_epochs, 19) — what ch sends
        inflow  = connectivity[:, ch_idx, :]   # (n_epochs, 19) — what ch receives
        total   = outflow + inflow

        for tgt_idx in range(N_CH):
            if tgt_idx == ch_idx:
                continue
            tgt_name = CHANNELS[tgt_idx]
            # Thicker line if target is also a focus channel
            lw    = 2.0 if tgt_name in focus_channels else 1.0
            alpha = 0.9 if tgt_name in focus_channels else 0.6

            ax_out.plot(time, outflow[:, tgt_idx],
                        color=COLOUR_MAP[tgt_name],
                        linewidth=lw, alpha=alpha, label=tgt_name)
            ax_in.plot(time, inflow[:, tgt_idx],
                       color=COLOUR_MAP[tgt_name],
                       linewidth=lw, alpha=alpha, label=tgt_name)
            ax_total.plot(time, total[:, tgt_idx],
                          color=COLOUR_MAP[tgt_name],
                          linewidth=lw, alpha=alpha, label=tgt_name)

        for ax in [ax_out, ax_in, ax_total]:
            ax.axvspan(time[first_ictal_idx], time[last_ictal_idx],
                       alpha=0.12, color='red', zorder=0)
            ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
            ax.set_ylim([global_ymin, global_ymax])
            ax.grid(alpha=0.3)

        ax_out.set_title(
            f'OUTFLOW  {ch_name} → others\n'
            f'What {ch_name} SENDS  |  Rising = {ch_name} drives that channel',
            fontsize=9, fontweight='bold')
        ax_in.set_title(
            f'INFLOW  others → {ch_name}\n'
            f'What {ch_name} RECEIVES  |  Rising = that channel drives {ch_name}',
            fontsize=9, fontweight='bold')
        ax_total.set_title(
            f'TOTAL  {ch_name} ↔ others\n'
            f'Overall coupling  |  High = strong pair coupling',
            fontsize=9, fontweight='bold')

        for ax, title in [(ax_out, 'Target'), (ax_in, 'Source'), (ax_total, 'Partner')]:
            ax.set_ylabel(f'{metric.upper()} value', fontsize=9)
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      fontsize=7, ncol=1, title=title, title_fontsize=8)

    for col in range(3):
        axes[-1, col].set_xlabel(
            'Time from seizure onset (s)\n(negative = pre-ictal, 0 = onset)',
            fontsize=9)

    focus_str = ', '.join(focus_channels)
    fig.suptitle(
        f'{subject_name}  |  {diagnosis}\n'
        f'Pairwise focused analysis: clinical focus channels [{focus_str}]\n'
        f'Metric: {metric.upper()}  |  Band: {band}  |  '
        f'Shared axes  |  Red = ictal  |  Dashed = t=0\n'
        f'LEFT: outflow (sends)  |  MIDDLE: inflow (receives)  |  '
        f'RIGHT: total (overall coupling)',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()

    focus_tag = '_'.join(focus_channels)
    fname = output_dir / f'{subject_name}_FOCUSED_{focus_tag}_{metric}_{band}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname.name}')


# ==============================================================================
# PROCESS ONE SUBJECT
# ==============================================================================

def process_subject(graphs_file, output_dir, band, metric,
                    override_focus, individual_plots):
    """
    Full pipeline for one subject.
    Focus channels are taken from the clinical mapping unless overridden.
    """
    subject_name = Path(graphs_file).stem.replace('_graphs', '')
    subj_id      = int(subject_name.split('_')[1])

    # ------------------------------------------------------------------
    # Skip subject 1 (not in paper)
    # ------------------------------------------------------------------
    if subj_id == 1 and override_focus is None:
        print(f'\n  Skipping {subject_name} — PAT 11 not in paper')
        return

    # ------------------------------------------------------------------
    # Determine focus channels
    # ------------------------------------------------------------------
    if override_focus:
        focus_channels = override_focus
    else:
        focus_channels = CLINICAL_FOCUS.get(subj_id, None)

    diagnosis = CLINICAL_DIAGNOSIS.get(subj_id, 'Unknown')

    subj_dir = output_dir / subject_name
    subj_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'Processing: {subject_name}')
    print(f'Diagnosis:  {diagnosis}')
    print(f'Focus channels: {focus_channels}')
    print(f'{"="*70}')

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = np.load(graphs_file)

    connectivity_key = f'{metric}_{band}'
    if connectivity_key not in data:
        print(f'  Key "{connectivity_key}" not found. '
              f'Available: {list(data.keys())}')
        return

    connectivity = data[connectivity_key]   # (n_epochs, 19, 19)
    labels       = data['labels']            # (n_epochs,)
    n_epochs     = len(labels)

    print(f'  Epochs:  {n_epochs}  '
          f'(pre={int((labels==0).sum())}, ictal={int((labels==1).sum())})')

    # ------------------------------------------------------------------
    # Time axis
    # ------------------------------------------------------------------
    time, first_ictal_idx, last_ictal_idx = build_time_axis(labels, data)

    print(f'  Seizure: t=0 at epoch {first_ictal_idx}, '
          f'ends t={time[last_ictal_idx]:.0f}s')

    # ------------------------------------------------------------------
    # Plot 1: Summary grid — all 19 channels, focus channels highlighted
    # ------------------------------------------------------------------
    print('  Generating summary grid...')
    plot_summary_grid(connectivity, time, first_ictal_idx, last_ictal_idx,
                      subject_name, band, metric, subj_dir,
                      diagnosis=diagnosis, focus_channels=focus_channels)

    # ------------------------------------------------------------------
    # Plot 2: Focused pair plots for clinical focus channels
    # ------------------------------------------------------------------
    if focus_channels:
        valid_focus = [ch for ch in focus_channels if ch in CHANNELS]
        if valid_focus:
            print(f'  Generating focused pair plot for: {valid_focus}')
            plot_focused_pairs(valid_focus, connectivity, time,
                               first_ictal_idx, last_ictal_idx,
                               subject_name, band, metric, subj_dir,
                               diagnosis=diagnosis)

    # ------------------------------------------------------------------
    # Plot 3 (optional): Individual outflow + inflow + total per channel
    # ------------------------------------------------------------------
    if individual_plots:
        ind_dir = subj_dir / 'individual'
        ind_dir.mkdir(exist_ok=True)
        print(f'  Generating individual pairwise plots (19×3)...')

        for ch_idx in range(N_CH):
            plot_outflow(ch_idx, connectivity, time,
                         first_ictal_idx, last_ictal_idx,
                         subject_name, band, metric, ind_dir,
                         diagnosis=diagnosis, focus_channels=focus_channels)
            plot_inflow(ch_idx, connectivity, time,
                        first_ictal_idx, last_ictal_idx,
                        subject_name, band, metric, ind_dir,
                        diagnosis=diagnosis, focus_channels=focus_channels)
            plot_total(ch_idx, connectivity, time,
                       first_ictal_idx, last_ictal_idx,
                       subject_name, band, metric, ind_dir,
                       diagnosis=diagnosis, focus_channels=focus_channels)

        print(f'  Individual plots saved in: {ind_dir}')

    print(f'  Done — plots in: {subj_dir}')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pairwise temporal connectivity — clinical focus auto-selected',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--connectivity_dir', required=True,
                        help='Directory with subject_XX_graphs.npz files')
    parser.add_argument('--output_dir', required=True,
                        help='Where to save all figures')
    parser.add_argument('--band', default='integrated',
                        choices=['integrated','delta','theta',
                                 'alpha','beta','gamma1'])
    parser.add_argument('--metric', default='pdc',
                        choices=['pdc', 'dtf'])
    parser.add_argument('--subject_ids', nargs='+', type=int,
                        default=list(range(1, 35)),
                        help='Subject IDs to process (default: 1–34)')
    parser.add_argument('--focus_channels', nargs='+', default=None,
                        help='Override clinical focus channels for ALL subjects '
                             '(default: auto from clinical mapping per subject)')
    parser.add_argument('--individual_plots', action='store_true',
                        help='Also save individual outflow/inflow/total plots '
                             'per channel (57 extra files per subject)')

    args = parser.parse_args()

    connectivity_dir = Path(args.connectivity_dir)
    output_dir       = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('STEP 3ab — PAIRWISE TEMPORAL CONNECTIVITY EVOLUTION')
    print('=' * 70)
    print(f'Metric:          {args.metric.upper()}')
    print(f'Band:            {args.band}')
    print(f'Focus channels:  {"auto (clinical mapping)" if not args.focus_channels else args.focus_channels}')
    print(f'Subjects:        {len(args.subject_ids)}')
    print(f'Individual plots:{args.individual_plots}')
    print(f'Output:          {output_dir}')
    print('=' * 70)
    print()
    print('MATRIX CONVENTION:')
    print('  connectivity[epoch, sink_i, source_j] = source_j → sink_i')
    print('  OUTFLOW of ch j = connectivity[:, :, j]  (column j)')
    print('  INFLOW  of ch i = connectivity[:, i, :]  (row i)')
    print('  time 0 = first ictal epoch (from time_from_onset in .npz)')
    print('=' * 70)

    success, errors = 0, 0

    for subj_id in args.subject_ids:
        subject_name = f'subject_{subj_id:02d}'
        graphs_file  = connectivity_dir / f'{subject_name}_graphs.npz'

        if not graphs_file.exists():
            print(f'\n  Skip {subject_name}: file not found')
            errors += 1
            continue

        try:
            process_subject(
                graphs_file      = graphs_file,
                output_dir       = output_dir,
                band             = args.band,
                metric           = args.metric,
                override_focus   = args.focus_channels,
                individual_plots = args.individual_plots,
            )
            success += 1
        except Exception as e:
            import traceback
            print(f'\n  Error on {subject_name}: {e}')
            traceback.print_exc()
            errors += 1

    print('\n' + '=' * 70)
    print(f'Success: {success}  |  Errors: {errors}')
    print('=' * 70)
    print()
    print('FILES CREATED PER SUBJECT (output_dir/subject_XX/):')
    print('  *_SUMMARY_*.png   — 19-channel overview, focus channels highlighted')
    print('  *_FOCUSED_*.png   — outflow | inflow | total for focus channels')
    print('  individual/       — (if --individual_plots) per-channel detail plots')
    print('=' * 70)


if __name__ == '__main__':
    main()