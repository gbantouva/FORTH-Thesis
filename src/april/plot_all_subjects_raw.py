"""
Plot raw EEG signals for all 34 subjects with seizure period highlighted in red.
Each subject is saved as a separate PNG file.
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Load data ──────────────────────────────────────────────────────────
mat_file = r"F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat"
data = sio.loadmat(mat_file, squeeze_me=False, struct_as_record=False)
s = data['seizure'][0, 0]

fs = 256  # sampling frequency (Hz)
SUBJECTS = s.x.shape[0]

out_dir = r"F:\FORTH_Final_Thesis\FORTH-Thesis\figures\april\raw_signal"
os.makedirs(out_dir, exist_ok=True)

# ── Loop over all subjects ─────────────────────────────────────────────
for subj in range(SUBJECTS):
    x = s.x[subj, 0]  # [samples x channels]
    N, C = x.shape

    # channel names from the 'selected' field
    chan_struct = s.chans[subj, 0][0, 0]
    sel = chan_struct.selected
    chnames = [str(sel[i, 0][0]) for i in range(sel.shape[0])]

    # seizure annotation (start, stop samples — 1-based MATLAB indexing)
    ann_start = int(s.annotation[subj, 0][0, 0])
    ann_stop  = int(s.annotation[subj, 1][0, 0])

    # check for multiple seizures in the segment
    seg_all = s.seg_all_seizures[subj, 0]  # Nx2 array of onset/offset pairs

    info = str(s.info[subj, 0][0])

    t = np.arange(N) / fs
    t_on  = (ann_start - 1) / fs
    t_off = (ann_stop  - 1) / fs

    # vertical spacing
    offset = 6 * np.median(np.std(x, axis=0))
    y_pos = np.arange(C - 1, -1, -1) * offset

    fig, ax = plt.subplots(figsize=(14, 9))
    ymin, ymax = -offset, C * offset

    # shade ALL seizure periods in the segment
    if seg_all.ndim == 2 and seg_all.shape[1] == 2:
        for row in range(seg_all.shape[0]):
            s_on  = (seg_all[row, 0] - 1) / fs
            s_off = (seg_all[row, 1] - 1) / fs
            label = 'Seizure period' if row == 0 else None
            ax.axvspan(s_on, s_off, facecolor=(1, 0.75, 0.75), alpha=0.55,
                       edgecolor='none', zorder=0, label=label)
            ax.plot([s_on, s_on],   [ymin, ymax], 'r--', linewidth=1, alpha=0.7)
            ax.plot([s_off, s_off], [ymin, ymax], 'r--', linewidth=1, alpha=0.7)
    else:
        # fallback: single seizure from annotation
        ax.axvspan(t_on, t_off, facecolor=(1, 0.75, 0.75), alpha=0.55,
                   edgecolor='none', zorder=0, label='Seizure period')
        ax.plot([t_on, t_on],   [ymin, ymax], 'r--', linewidth=1)
        ax.plot([t_off, t_off], [ymin, ymax], 'r--', linewidth=1)

    # plot signals
    for ch in range(C):
        ax.plot(t, x[:, ch] + y_pos[ch], color='black', linewidth=0.4)

    n_labels = min(C, len(chnames))
    ax.set_yticks(y_pos[:n_labels][::-1])
    ax.set_yticklabels(chnames[:n_labels][::-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(
        f'Subject {subj + 1} — {info}   |   seizure: {t_on:.1f}–{t_off:.1f} s',
        fontsize=13
    )
    ax.legend(loc='upper right')
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    fname = os.path.join(out_dir, f'subj_{subj + 1:02d}.png')
    fig.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close(fig)

    print(f'[{subj + 1:2d}/{SUBJECTS}]  {info:30s}  channels={C}  '
          f'duration={N / fs:.1f}s  seizure={t_on:.1f}–{t_off:.1f}s  → {fname}')

print(f'\nDone — {SUBJECTS} plots saved to {out_dir}/')
