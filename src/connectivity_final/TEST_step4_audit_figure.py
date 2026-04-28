"""
Per-subject connectivity audit figure
======================================
For ONE subject, produce a single multi-panel figure that lets a reader
check — by eye — whether the DTF/PDC output is consistent with the raw EEG.

Panels (top to bottom), all sharing the same time axis where possible:

  (A) Header text:  subject id, patient id, clinical focal mapping,
                    seizure timing, number of epochs

  (B) Stacked 19-channel EEG traces, ±window_sec around seizure onset
      Seizure-onset line at t=0; ictal period shaded.

  (C) Spectrogram (0-45 Hz) for the clinically-expected focal channel.

  (D) Per-channel out-strength vs time (DTF, broadband).  19 lines;
      clinically-expected focal channels highlighted in colour, rest in grey.

  (E) Per-channel in-strength vs time (PDC, broadband).  Same layout.

  (F) Three connectivity snapshots — pre-ictal, onset, mid-ictal — for DTF
      and PDC, broadband.  Small 19x19 heatmaps, shared colour scale.

The figure is saved as  <outdir>/subject_XX_audit.png

Usage
-----
  python step4_audit_figure.py \
      --subject        22 \
      --epochs_dir     F:/FORTH_Final_Thesis/FORTH-Thesis/preprocessed_epochs \
      --connect_dir    F:/FORTH_Final_Thesis/FORTH-Thesis/final_connectivity \
      --outdir         F:/FORTH_Final_Thesis/FORTH-Thesis/figures/audit \
      --focal          F3 F7 Fp1        # defaults for subject 22
      --window_sec     60

  # Offline smoke test (no real data required):
  python step4_audit_figure.py --demo --outdir demo_audit
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHANNELS = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',
    'T3',  'C3',  'Cz',  'C4',  'T4',
    'T5',  'P3',  'Pz',  'P4',  'T6',
    'O1',  'O2',
]
CH_IDX = {c: i for i, c in enumerate(CHANNELS)}

# Patient mapping (from step3ab header — same source of truth)
PATIENT_MAP = {
    **{1:  ('PAT 11', 'not in paper',        [])},
    **{2:  ('PAT 13', 'focal right frontal', ['F4', 'F8', 'Fp2'])},
    **{s:  ('PAT 14', 'focal right frontal', ['F4', 'F8', 'Fp2'])
       for s in range(3, 11)},
    **{11: ('PAT 15', 'focal fronto-polar',  ['Fp1', 'Fp2'])},
    **{s:  ('PAT 24', 'focal left frontal',  ['F3', 'F7', 'Fp1'])
       for s in range(12, 26)},
    **{s:  ('PAT 27', 'focal right frontal', ['F4', 'F8', 'Fp2'])
       for s in range(26, 33)},
    **{33: ('PAT 29', 'focal bifrontal',     ['F3', 'F4', 'Fz', 'Fp1', 'Fp2'])},
    **{34: ('PAT 35', 'focal (unspecified)', ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'])},
}

FS = 256
EPOCH_LEN = 4.0  # seconds
SAMPLES_PER_EPOCH = int(FS * EPOCH_LEN)

BAND_COLOR_FOCAL = '#d62728'  # red-ish for focal channels
BAND_COLOR_OTHER = '#b0b0b0'  # grey for the rest


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_subject(subject_id: int, epochs_dir: Path, connect_dir: Path):
    name = f'subject_{subject_id:02d}'
    eps_path  = epochs_dir / f'{name}_epochs.npy'
    lab_path  = epochs_dir / f'{name}_labels.npy'
    tfo_path  = epochs_dir / f'{name}_time_from_onset.npy'
    meta_path = epochs_dir / f'{name}_metadata.json'
    npz_path  = connect_dir / f'{name}_graphs.npz'

    for p in (eps_path, lab_path, tfo_path, npz_path):
        if not p.exists():
            raise FileNotFoundError(f'Missing: {p}')

    epochs = np.load(eps_path)          # (N, 19, 1024)
    labels = np.load(lab_path)          # (N,)
    tfo    = np.load(tfo_path)          # (N,)  seconds from seizure onset

    meta = {}
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)

    npz = dict(np.load(npz_path, allow_pickle=True))

    # The connectivity file carries indices (valid epochs) and possibly
    # time_from_onset. Validate alignment.
    valid_idx = npz.get('indices', np.arange(len(labels)))
    valid_idx = np.asarray(valid_idx, dtype=int)

    return {
        'name':       name,
        'epochs':     epochs,
        'labels':     labels,
        'tfo':        tfo,
        'metadata':   meta,
        'npz':        npz,
        'valid_idx':  valid_idx,
    }


# ---------------------------------------------------------------------------
# Synthetic data for --demo
# ---------------------------------------------------------------------------

def _synthesize_subject(focal_chs, n_pre=40, n_ict=15, seed=0):
    rng = np.random.default_rng(seed)
    n = n_pre + n_ict
    labels = np.concatenate([np.zeros(n_pre, int), np.ones(n_ict, int)])
    tfo = np.arange(n) * EPOCH_LEN - n_pre * EPOCH_LEN  # 0 at first ictal

    epochs = np.zeros((n, 19, SAMPLES_PER_EPOCH), np.float32)
    t = np.arange(SAMPLES_PER_EPOCH) / FS
    focal_idx = [CH_IDX[c] for c in focal_chs]
    for e in range(n):
        base = rng.standard_normal((19, SAMPLES_PER_EPOCH)) * 5
        # Always-present 10 Hz alpha, posterior-dominant
        for ch in ['O1', 'O2', 'Pz']:
            base[CH_IDX[ch]] += 10 * np.sin(2*np.pi*10*t)
        if labels[e] == 1:
            # Ictal: rhythmic 6 Hz theta + 22 Hz beta on focal channels
            for fi in focal_idx:
                base[fi] += 45 * np.sin(2*np.pi*6*t + rng.uniform(0, 2*np.pi))
                base[fi] += 25 * np.sin(2*np.pi*22*t + rng.uniform(0, 2*np.pi))
        epochs[e] = base

    # Synthetic DTF/PDC: ictal bright columns at focal channels
    K = 19
    n_valid = n
    def make_conn(src_bias=0.25):
        dtf, pdc = [], []
        for e in range(n_valid):
            m = 0.12 + 0.03 * rng.standard_normal((K, K))
            np.fill_diagonal(m, 0)
            if labels[e] == 1:
                for fi in focal_idx:
                    m[:, fi] += src_bias + 0.03 * rng.standard_normal(K)
            m = np.clip(m, 0, 1)
            np.fill_diagonal(m, 0)
            dtf.append(m)
            p = 0.6 * m + 0.03 * rng.standard_normal((K, K))
            np.fill_diagonal(p, 0)
            pdc.append(np.clip(p, 0, 1))
        return np.stack(dtf).astype(np.float32), np.stack(pdc).astype(np.float32)

    dtf_int, pdc_int = make_conn()
    npz = {
        'dtf_integrated': dtf_int,
        'pdc_integrated': pdc_int,
        'labels':         labels,
        'indices':        np.arange(n),
        'time_from_onset': tfo,
        'fixed_order':    12,
    }
    return {
        'name':       'subject_DEMO',
        'epochs':     epochs,
        'labels':     labels,
        'tfo':        tfo,
        'metadata':   {'seizure_duration_sec': n_ict * EPOCH_LEN},
        'npz':        npz,
        'valid_idx':  np.arange(n),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def out_strength_over_time(dtf_epochs):
    """Per-channel DTF out-strength (column sum, off-diag) per epoch.
    dtf_epochs shape (N, K, K). Returns (N, K)."""
    K = dtf_epochs.shape[-1]
    # Column sum = axis=1 (sum over sinks i). Diagonals are already 0 in step2.
    return dtf_epochs.sum(axis=1) * (K / (K - 1))  # rescale crudely for diag=0


def in_strength_over_time(pdc_epochs):
    """Per-channel PDC in-strength (row sum, off-diag) per epoch.
    Returns (N, K)."""
    K = pdc_epochs.shape[-1]
    return pdc_epochs.sum(axis=2) * (K / (K - 1))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _window_mask(tfo, window_sec):
    return (tfo >= -window_sec) & (tfo <= window_sec)


def _reconstruct_continuous_signal(epochs_slice):
    """(n, 19, T) -> (19, n*T) by concatenation."""
    n, K, T = epochs_slice.shape
    return epochs_slice.transpose(1, 0, 2).reshape(K, n * T)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_audit_figure(data, focal_chs, window_sec, outpath: Path):
    epochs  = data['epochs']
    labels  = data['labels']
    tfo     = data['tfo']
    meta    = data['metadata']
    npz     = data['npz']

    # Subject and patient info
    sid = int(data['name'].split('_')[-1]) if data['name'].split('_')[-1].isdigit() else 0
    pat = PATIENT_MAP.get(sid, ('UNKNOWN', 'unknown', []))
    patient_code, clinical_note, expected_chs = pat
    if not focal_chs:
        focal_chs = expected_chs or ['Fp1']  # fall back

    focal_idx = [CH_IDX[c] for c in focal_chs if c in CH_IDX]

    # Restrict to window around onset
    mask = _window_mask(tfo, window_sec)
    if mask.sum() < 3:
        # Fall back to everything we've got
        mask = np.ones_like(mask, bool)

    # Indices into epochs that fall into the window (time-axis for panels B-E)
    ep_idx_window = np.where(mask)[0]
    tfo_w = tfo[ep_idx_window]
    ep_w  = epochs[ep_idx_window]  # (n_w, 19, T)
    lab_w = labels[ep_idx_window]

    # Continuous signal for panel B and spectrogram
    sig = _reconstruct_continuous_signal(ep_w)  # (19, n_w*T)
    # Time base, seconds from onset: each sample's timestamp
    n_w = ep_w.shape[0]
    t_continuous = (
        np.repeat(tfo_w, SAMPLES_PER_EPOCH)
        + np.tile(np.arange(SAMPLES_PER_EPOCH) / FS, n_w)
    )

    # Ictal time range within the window
    ictal_mask_w = lab_w == 1
    if ictal_mask_w.any():
        ictal_t0 = tfo_w[ictal_mask_w].min()
        ictal_t1 = tfo_w[ictal_mask_w].max() + EPOCH_LEN
    else:
        ictal_t0, ictal_t1 = 0.0, 0.0

    # Connectivity: align valid_idx with epoch indices
    valid_idx = data['valid_idx']
    # Build map from original epoch index -> position in npz arrays
    pos_in_npz = {int(orig): k for k, orig in enumerate(valid_idx)}

    # Get broadband DTF/PDC for the window's valid epochs
    have_dtf = 'dtf_integrated' in npz
    have_pdc = 'pdc_integrated' in npz
    if not (have_dtf and have_pdc):
        raise KeyError("Need 'dtf_integrated' and 'pdc_integrated' in the .npz")

    dtf_all = np.asarray(npz['dtf_integrated'])
    pdc_all = np.asarray(npz['pdc_integrated'])

    # Time series of strengths restricted to window, with NaN gaps for
    # epochs that weren't in the connectivity output.
    dtf_out_ts = np.full((len(ep_idx_window), 19), np.nan)
    pdc_in_ts  = np.full((len(ep_idx_window), 19), np.nan)
    for k, orig in enumerate(ep_idx_window):
        if int(orig) in pos_in_npz:
            j = pos_in_npz[int(orig)]
            dtf_out_ts[k] = dtf_all[j].sum(axis=0)  # col sums (sum over sinks)
            pdc_in_ts[k]  = pdc_all[j].sum(axis=1)  # row sums (sum over sources)

    # Three snapshot epochs: pre, onset, mid-ictal
    snapshot_times = _pick_snapshot_epoch_indices(tfo_w, ictal_t0, ictal_t1)
    snapshot_labels = ['pre-ictal (t \u2248 {:+.0f} s)',
                       'onset     (t \u2248 {:+.0f} s)',
                       'mid-ictal (t \u2248 {:+.0f} s)']
    snaps_dtf = []
    snaps_pdc = []
    snaps_when = []
    for lab_txt, st_idx in zip(snapshot_labels, snapshot_times):
        orig_ep = ep_idx_window[st_idx]
        when = tfo_w[st_idx]
        snaps_when.append(lab_txt.format(when))
        if int(orig_ep) in pos_in_npz:
            j = pos_in_npz[int(orig_ep)]
            snaps_dtf.append(dtf_all[j])
            snaps_pdc.append(pdc_all[j])
        else:
            snaps_dtf.append(np.zeros((19, 19)))
            snaps_pdc.append(np.zeros((19, 19)))

    # ======================================================================
    # FIGURE
    # ======================================================================
    fig = plt.figure(figsize=(18, 32))
    gs = GridSpec(
        7, 3,
        height_ratios=[0.45, 4.5, 1.4, 1.8, 1.8, 3.6, 3.2],
        hspace=0.75, wspace=0.25,
        figure=fig,
    )

    # --- (A) header -------------------------------------------------------
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.axis('off')
    dur = meta.get('seizure_duration_sec', (ictal_t1 - ictal_t0))
    hdr_txt = (
        f"{data['name'].replace('_', ' ').upper()}   |   {patient_code}   |   "
        f"{clinical_note}\n"
        f"Clinically-expected focal channels: {', '.join(expected_chs) or '—'}   "
        f"(highlighted in red in panels D and E)\n"
        f"Total epochs in window: {mask.sum()}   "
        f"(pre-ictal: {(lab_w == 0).sum()}, ictal: {(lab_w == 1).sum()})   "
        f"Seizure duration: {dur:.1f} s   "
        f"Window: \u00b1{window_sec} s around onset"
    )
    ax_hdr.text(0.0, 0.5, hdr_txt, va='center', ha='left',
                fontsize=12, family='monospace')

    # --- (B) stacked EEG traces ------------------------------------------
    ax_eeg = fig.add_subplot(gs[1, :])
    # Normalize each channel to unit variance, offset by channel index
    sig_z = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-9)
    # Gain: spread channels vertically
    gain = 3.5
    for ch_i in range(19):
        color = BAND_COLOR_FOCAL if ch_i in focal_idx else '#333333'
        lw = 0.8 if ch_i in focal_idx else 0.45
        ax_eeg.plot(t_continuous,
                    sig_z[ch_i] + (19 - 1 - ch_i) * gain,
                    color=color, linewidth=lw, alpha=0.95)
    ax_eeg.set_yticks([(19 - 1 - i) * gain for i in range(19)])
    ax_eeg.set_yticklabels(CHANNELS, fontsize=8)
    ax_eeg.set_xlim(-window_sec, window_sec)
    ax_eeg.set_xlabel('Time from seizure onset (s)', fontsize=10)
    ax_eeg.set_title('(B)  19-channel EEG traces  (focal channels in red, z-scored)',
                     fontsize=11, fontweight='bold', loc='left')
    ax_eeg.axvline(0, color='red', linestyle='--', linewidth=1.4, alpha=0.8)
    if ictal_t1 > ictal_t0:
        ax_eeg.axvspan(ictal_t0, ictal_t1, color='red', alpha=0.08)
    ax_eeg.grid(True, alpha=0.2, axis='x')

    # --- (C) spectrogram for first focal channel -------------------------
    ax_spec = fig.add_subplot(gs[2, :])
    spec_ch = focal_chs[0]
    spec_idx = CH_IDX[spec_ch]
    f_sp, t_sp, Sxx = spectrogram(
        sig[spec_idx], fs=FS, nperseg=256, noverlap=192, nfft=512,
    )
    t_sp_abs = t_sp + t_continuous[0]  # map to absolute time-from-onset
    band_mask = f_sp <= 45
    Sxx_db = 10 * np.log10(Sxx[band_mask] + 1e-10)
    im = ax_spec.pcolormesh(t_sp_abs, f_sp[band_mask], Sxx_db,
                            shading='auto', cmap='magma')
    ax_spec.set_ylabel('Freq (Hz)', fontsize=10)
    ax_spec.set_xlim(-window_sec, window_sec)
    ax_spec.set_xlabel('Time from seizure onset (s)', fontsize=10)
    ax_spec.set_title(f'(C)  Spectrogram of {spec_ch}  (0–45 Hz, dB)',
                      fontsize=11, fontweight='bold', loc='left')
    ax_spec.axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.9)
    if ictal_t1 > ictal_t0:
        ax_spec.axvspan(ictal_t0, ictal_t1, color='red', alpha=0.12, zorder=3)
    fig.colorbar(im, ax=ax_spec, fraction=0.018, pad=0.01,
                 label='Power (dB)')

    # --- (D) DTF out-strength per channel -------------------------------
    ax_dtf = fig.add_subplot(gs[3, :])
    for ch_i in range(19):
        color = BAND_COLOR_FOCAL if ch_i in focal_idx else BAND_COLOR_OTHER
        lw = 1.8 if ch_i in focal_idx else 0.8
        alpha = 1.0 if ch_i in focal_idx else 0.55
        ax_dtf.plot(tfo_w, dtf_out_ts[:, ch_i],
                    color=color, linewidth=lw, alpha=alpha,
                    label=CHANNELS[ch_i] if ch_i in focal_idx else None)
    ax_dtf.axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
    if ictal_t1 > ictal_t0:
        ax_dtf.axvspan(ictal_t0, ictal_t1, color='red', alpha=0.08)
    ax_dtf.set_xlim(-window_sec, window_sec)
    ax_dtf.set_xlabel('Time from seizure onset (s)', fontsize=10)
    ax_dtf.set_ylabel('DTF out-strength', fontsize=10)
    ax_dtf.set_title('(D)  Per-channel DTF out-strength  (broadband; focal channels bold red)',
                     fontsize=11, fontweight='bold', loc='left')
    ax_dtf.legend(fontsize=9, loc='upper left', ncol=len(focal_chs))
    ax_dtf.grid(True, alpha=0.3)

    # --- (E) PDC in-strength per channel --------------------------------
    ax_pdc = fig.add_subplot(gs[4, :])
    for ch_i in range(19):
        color = BAND_COLOR_FOCAL if ch_i in focal_idx else BAND_COLOR_OTHER
        lw = 1.8 if ch_i in focal_idx else 0.8
        alpha = 1.0 if ch_i in focal_idx else 0.55
        ax_pdc.plot(tfo_w, pdc_in_ts[:, ch_i],
                    color=color, linewidth=lw, alpha=alpha,
                    label=CHANNELS[ch_i] if ch_i in focal_idx else None)
    ax_pdc.axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
    if ictal_t1 > ictal_t0:
        ax_pdc.axvspan(ictal_t0, ictal_t1, color='red', alpha=0.08)
    ax_pdc.set_xlim(-window_sec, window_sec)
    ax_pdc.set_xlabel('Time from seizure onset (s)', fontsize=10)
    ax_pdc.set_ylabel('PDC in-strength', fontsize=10)
    ax_pdc.set_title('(E)  Per-channel PDC in-strength  (broadband; focal channels bold red)',
                     fontsize=11, fontweight='bold', loc='left')
    ax_pdc.legend(fontsize=9, loc='upper left', ncol=len(focal_chs))
    ax_pdc.grid(True, alpha=0.3)

    # --- (F) snapshot matrices ------------------------------------------
    # 2 rows x 3 cols of small heatmaps inside the last gs cell.
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_snap = GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=gs[5, :],
        wspace=0.25, hspace=0.35,
    )
    vmax_dtf = max(m.max() for m in snaps_dtf) if snaps_dtf else 1.0
    vmax_pdc = max(m.max() for m in snaps_pdc) if snaps_pdc else 1.0

    for col, (dtf_m, pdc_m, when_txt) in enumerate(
            zip(snaps_dtf, snaps_pdc, snaps_when)):
        ax_d = fig.add_subplot(gs_snap[0, col])
        sns.heatmap(dtf_m, ax=ax_d, cmap='viridis', square=True,
                    vmin=0, vmax=vmax_dtf,
                    xticklabels=CHANNELS, yticklabels=CHANNELS,
                    cbar=(col == 2),
                    cbar_kws={'label': 'DTF'} if col == 2 else None)
        ax_d.set_title(f'DTF  {when_txt}', fontsize=10, fontweight='bold')
        ax_d.set_xlabel('')                           # top row: no x label
        ax_d.set_xticklabels([])                      # top row: no tick labels
        ax_d.set_ylabel('sink i' if col == 0 else '', fontsize=8)
        ax_d.tick_params(axis='y', rotation=0,  labelsize=6)

        ax_p = fig.add_subplot(gs_snap[1, col])
        sns.heatmap(pdc_m, ax=ax_p, cmap='viridis', square=True,
                    vmin=0, vmax=vmax_pdc,
                    xticklabels=CHANNELS, yticklabels=CHANNELS,
                    cbar=(col == 2),
                    cbar_kws={'label': 'PDC'} if col == 2 else None)
        ax_p.set_title(f'PDC  {when_txt}', fontsize=10, fontweight='bold')
        ax_p.set_xlabel('source j', fontsize=8)
        ax_p.set_ylabel('sink i' if col == 0 else '', fontsize=8)
        ax_p.tick_params(axis='x', rotation=90, labelsize=6)
        ax_p.tick_params(axis='y', rotation=0,  labelsize=6)

    # --- (G) Difference matrices: mean ictal − mean pre-ictal -----------
    # Average over all epochs of each class in the window, strip the common-
    # mode structure by subtraction, so each subject's focal signature shows.
    gs_diff = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[6, :],
        wspace=0.30,
        width_ratios=[1.0, 1.0, 0.05],
    )

    # Indices into npz arrays for pre and ictal epochs in the window
    pre_pos, ict_pos = [], []
    for k, orig in enumerate(ep_idx_window):
        if int(orig) not in pos_in_npz:
            continue
        if lab_w[k] == 0:
            pre_pos.append(pos_in_npz[int(orig)])
        elif lab_w[k] == 1:
            ict_pos.append(pos_in_npz[int(orig)])

    def _safe_mean(arr, idxs):
        if not idxs:
            return np.zeros((19, 19))
        return arr[idxs].mean(axis=0)

    dtf_pre_mean = _safe_mean(dtf_all, pre_pos)
    dtf_ict_mean = _safe_mean(dtf_all, ict_pos)
    pdc_pre_mean = _safe_mean(pdc_all, pre_pos)
    pdc_ict_mean = _safe_mean(pdc_all, ict_pos)

    dtf_diff = dtf_ict_mean - dtf_pre_mean
    pdc_diff = pdc_ict_mean - pdc_pre_mean

    # Symmetric color scale across both measures so they are comparable
    vmax_diff = max(
        float(np.abs(dtf_diff).max()),
        float(np.abs(pdc_diff).max()),
        1e-6,
    )
    vmin_diff = -vmax_diff

    for col, (mat, name) in enumerate(
            [(dtf_diff, 'DTF'), (pdc_diff, 'PDC')]):
        ax = fig.add_subplot(gs_diff[0, col])
        sns.heatmap(
            mat, ax=ax, cmap='RdBu_r', square=True,
            vmin=vmin_diff, vmax=vmax_diff,
            xticklabels=CHANNELS, yticklabels=CHANNELS,
            cbar=False,
        )
        ax.set_title(
            f'{name}:  mean ictal − mean pre-ictal  '
            f'(n_ict={len(ict_pos)}, n_pre={len(pre_pos)})',
            fontsize=10, fontweight='bold',
        )
        ax.set_xlabel('source j', fontsize=8)
        ax.set_ylabel('sink i' if col == 0 else '', fontsize=8)
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', rotation=0,  labelsize=6)

        # Highlight the clinically-expected focal channels with green
        # boxes, once on the row axis (sinks) and once on the column axis
        # (sources). If the seizure really localises to them you expect
        # red (increase) inside or next to the boxes for the relevant
        # direction.
        for f_i in focal_idx:
            # Row highlight (sink = focal): spans full width
            ax.add_patch(plt.Rectangle(
                (0, f_i), 19, 1,
                fill=False, edgecolor='#1b9e77', linewidth=1.4, zorder=5,
            ))
            # Column highlight (source = focal): spans full height
            ax.add_patch(plt.Rectangle(
                (f_i, 0), 1, 19,
                fill=False, edgecolor='#1b9e77', linewidth=1.4, zorder=5,
            ))

    # Single shared colorbar on the right
    cax = fig.add_subplot(gs_diff[0, 2])
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=vmin_diff, vmax=vmax_diff)
    sm = mcm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label='ictal − pre')

    fig.suptitle(
        f'Connectivity audit — {data["name"]}   '
        '(G: red = edge stronger in ictal; green boxes = clinical focal channels)',
        fontsize=13, fontweight='bold', y=0.995,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'\u2713 Saved {outpath}')


def _pick_snapshot_epoch_indices(tfo_w, ictal_t0, ictal_t1):
    """Return three epoch indices into tfo_w: pre, onset, mid-ictal.
    If ictal absent, fall back to three evenly-spaced points."""
    n = len(tfo_w)
    if ictal_t1 <= ictal_t0:
        return [n // 4, n // 2, 3 * n // 4]
    # pre-ictal: 10 s before onset (nearest)
    pre_target   = ictal_t0 - 10.0
    onset_target = ictal_t0 + 2.0
    mid_target   = 0.5 * (ictal_t0 + ictal_t1)
    def nearest(t):
        return int(np.argmin(np.abs(tfo_w - t)))
    return [nearest(pre_target), nearest(onset_target), nearest(mid_target)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--subject',      type=int,
                    help='Subject number (1..34)')
    ap.add_argument('--epochs_dir',   type=str,
                    help='Folder with subject_XX_epochs.npy etc.')
    ap.add_argument('--connect_dir',  type=str,
                    help='Folder with subject_XX_graphs.npz')
    ap.add_argument('--outdir',       type=str, required=True)
    ap.add_argument('--focal',        nargs='*', default=None,
                    help='Focal channels (defaults to patient-map for subject)')
    ap.add_argument('--window_sec',   type=float, default=60.0,
                    help='Half-window around seizure onset (default 60 s)')
    ap.add_argument('--demo',         action='store_true',
                    help='Run on synthetic data, no real files required')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print('[DEMO] Using synthetic data')
        focal = args.focal or ['F3', 'F7', 'Fp1']
        data = _synthesize_subject(focal)
        outpath = outdir / 'subject_DEMO_audit.png'
        make_audit_figure(data, focal, args.window_sec, outpath)
        return

    if args.subject is None or args.epochs_dir is None or args.connect_dir is None:
        ap.error('--subject, --epochs_dir, --connect_dir are required unless --demo')

    data = load_subject(args.subject,
                        Path(args.epochs_dir), Path(args.connect_dir))

    # Default focal channels from patient mapping if not supplied
    focal = args.focal
    if not focal:
        focal = PATIENT_MAP.get(args.subject, ('', '', ['Fp1']))[2] or ['Fp1']

    outpath = outdir / f'subject_{args.subject:02d}_audit.png'
    make_audit_figure(data, focal, args.window_sec, outpath)


if __name__ == '__main__':
    main()