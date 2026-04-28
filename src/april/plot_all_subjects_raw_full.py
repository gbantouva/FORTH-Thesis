"""
Plot raw EEG signals for all 34 subjects with seizure period highlighted in red.
Two figures per subject:
  1) Full signal  → subj_XX_full.png
  2) Zoomed ±30s around seizure → subj_XX_zoom.png
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

# ── CONFIGURE THESE ────────────────────────────────────────────────────
mat_file = r"F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat"
out_dir  = r"F:\FORTH_Final_Thesis\FORTH-Thesis\figures\april\raw_signal"
# ───────────────────────────────────────────────────────────────────────

fs     = 256   # sampling frequency (Hz)
MARGIN = 30    # seconds before/after seizure for zoomed view

data = sio.loadmat(mat_file, squeeze_me=False, struct_as_record=False)
s = data['seizure'][0, 0]

SUBJECTS = s.x.shape[0]
os.makedirs(out_dir, exist_ok=True)


def plot_eeg(ax, t, x, y_pos, ymin, ymax, seg_all, t_on, t_off):
    """Draw seizure shading + EEG traces on a given axes."""

    # shade seizure periods
    if seg_all.ndim == 2 and seg_all.shape[1] == 2:
        for row in range(seg_all.shape[0]):
            s_on  = (seg_all[row, 0] - 1) / fs
            s_off = (seg_all[row, 1] - 1) / fs
            label = 'Seizure period' if row == 0 else None
            ax.axvspan(s_on, s_off, facecolor=(1, 0.75, 0.75), alpha=0.55,
                       edgecolor='none', zorder=0, label=label)
            ax.plot([s_on, s_on],   [ymin, ymax], 'r--', lw=1, alpha=0.7)
            ax.plot([s_off, s_off], [ymin, ymax], 'r--', lw=1, alpha=0.7)
    else:
        ax.axvspan(t_on, t_off, facecolor=(1, 0.75, 0.75), alpha=0.55,
                   edgecolor='none', zorder=0, label='Seizure period')
        ax.plot([t_on, t_on],   [ymin, ymax], 'r--', lw=1)
        ax.plot([t_off, t_off], [ymin, ymax], 'r--', lw=1)

    # plot signals
    C = x.shape[1]
    for ch in range(C):
        ax.plot(t, x[:, ch] + y_pos[ch], color='#2a2a2a', linewidth=0.35, alpha=0.85)


# ── Loop over all subjects ─────────────────────────────────────────────
for subj in range(SUBJECTS):
    x = s.x[subj, 0]          # [samples x channels]
    N, C = x.shape

    # channel names
    chan_struct = s.chans[subj, 0][0, 0]
    sel = chan_struct.selected
    chnames = [str(sel[i, 0][0]) for i in range(sel.shape[0])]

    # seizure annotation (1-based MATLAB indexing)
    ann_start = int(s.annotation[subj, 0][0, 0])
    ann_stop  = int(s.annotation[subj, 1][0, 0])

    # all seizures in the segment
    seg_all = s.seg_all_seizures[subj, 0]

    info = str(s.info[subj, 0][0])

    t = np.arange(N) / fs
    t_on  = (ann_start - 1) / fs
    t_off = (ann_stop  - 1) / fs
    seizure_dur = t_off - t_on

    # vertical spacing
    offset = 12 * np.median(np.std(x, axis=0))
    y_pos  = np.arange(C - 1, -1, -1) * offset
    ymin, ymax = -offset, C * offset
    n_labels = min(C, len(chnames))

    # ── 1) FULL signal ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_eeg(ax, t, x, y_pos, ymin, ymax, seg_all, t_on, t_off)

    ax.set_yticks(y_pos[:n_labels][::-1])
    ax.set_yticklabels(chnames[:n_labels][::-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(
        f'Subject {subj+1} — {info}   |   '
        f'seizure: {t_on:.1f}–{t_off:.1f} s  ({seizure_dur:.1f} s)',
        fontsize=13
    )
    ax.legend(loc='upper right')
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=10)
    plt.tight_layout()

    fname_full = os.path.join(out_dir, f'subj_{subj+1:02d}_full.png')
    fig.savefig(fname_full, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 2) ZOOMED around seizure ───────────────────────────────────
    t_view_start = max(0, t_on - MARGIN)
    t_view_end   = min(t[-1], t_off + MARGIN)

    fig, ax = plt.subplots(figsize=(14, 10))
    plot_eeg(ax, t, x, y_pos, ymin, ymax, seg_all, t_on, t_off)

    ax.set_yticks(y_pos[:n_labels][::-1])
    ax.set_yticklabels(chnames[:n_labels][::-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(t_view_start, t_view_end)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(
        f'Subject {subj+1} — {info}   |   '
        f'seizure: {t_on:.1f}–{t_off:.1f} s  ({seizure_dur:.1f} s)  [zoomed]',
        fontsize=13
    )
    ax.legend(loc='upper right')
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=10)
    plt.tight_layout()

    fname_zoom = os.path.join(out_dir, f'subj_{subj+1:02d}_zoom.png')
    fig.savefig(fname_zoom, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'[{subj+1:2d}/{SUBJECTS}]  {info:30s}  ch={C}  '
          f'seizure={t_on:.1f}–{t_off:.1f}s ({seizure_dur:.1f}s)')

print(f'\nDone — {SUBJECTS * 2} plots saved to {out_dir}/')
