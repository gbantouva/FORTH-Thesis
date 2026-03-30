"""
Feature Analysis & Sanity Check
================================
Loads features_all.npz and produces diagnostic plots to verify
that extracted features make neurophysiological sense.

Key checks:
  1. Asymmetry index (DTF + PDC) per patient          <- THE main sanity check
       PAT13, PAT14, PAT27 -> right frontal -> asymmetry should be NEGATIVE during ictal
       PAT24               -> left frontal  -> asymmetry should be POSITIVE during ictal

  2. Spectral features: ictal vs pre-ictal distributions
       delta should INCREASE, alpha/beta should DECREASE during seizure

  3. Connectivity global mean: ictal vs pre-ictal
       should INCREASE during seizure (hypersynchrony)

  4. Feature values over time per patient
       lets you visually confirm features change at the right moment (t=0)

Usage:
  python feature_analysis.py \
      --featfile  path/to/features/features_all.npz \
      --outputdir path/to/results/feature_analysis
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Patient metadata ──────────────────────────────────────────────────────────
# Laterality from your dataset diagram.
# RIGHT frontal seizure -> right hemisphere is source -> right out-degree higher
#   -> asymmetry = (left - right)/(left + right) should be NEGATIVE
# LEFT frontal seizure  -> left hemisphere is source  -> left out-degree higher
#   -> asymmetry should be POSITIVE
PATIENT_INFO = {
    'PAT11': 'unknown (not in paper)',
    'PAT13': 'focal RIGHT frontal',
    'PAT14': 'focal RIGHT frontal',
    'PAT15': 'focal fronto-polar',
    'PAT24': 'focal LEFT frontal',
    'PAT27': 'focal RIGHT frontal',
    'PAT29': 'focal bifrontal',
    'PAT35': 'focal (unspecified)',
}


# ─────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────

def load_data(featfile):
    data        = np.load(featfile, allow_pickle=True)
    X           = data['X'].astype(np.float32)
    y           = data['y'].astype(np.int64)
    patient_ids = data['patient_ids']
    tfo         = data['time_from_onset'].astype(np.float32)
    feat_names  = data['feature_names'].tolist()
    return X, y, patient_ids, tfo, feat_names


# ─────────────────────────────────────────────────────────────
# Check 1 — Asymmetry index: the main sanity check
# ─────────────────────────────────────────────────────────────

def plot_asymmetry_check(X, y, patient_ids, feat_names, output_dir):
    """
    For each patient: box plot of DTF and PDC asymmetry index,
    split by ictal (orange) vs pre-ictal (blue).

    Red shading  = right-frontal patients  -> ictal boxes should be BELOW zero
    Green shading = left-frontal patient   -> ictal boxes should be ABOVE zero
    """
    dtf_idx = feat_names.index('graph_dtf_asymmetry')
    pdc_idx = feat_names.index('graph_pdc_asymmetry')

    patients    = sorted(np.unique(patient_ids))
    PRE_COLOR   = '#5B9BD5'
    ICTAL_COLOR = '#ED7D31'

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        'Asymmetry Index per Patient  (blue = pre-ictal, orange = ictal)\n'
        'RIGHT frontal (PAT13, PAT14, PAT27) -> ictal should be NEGATIVE  [red zone]\n'
        'LEFT  frontal (PAT24)               -> ictal should be POSITIVE  [green zone]',
        fontsize=11, fontweight='bold'
    )

    for ax, feat_idx, title in [
        (axes[0], dtf_idx, 'DTF Asymmetry  = (left_outdeg - right_outdeg) / (left + right)'),
        (axes[1], pdc_idx, 'PDC Asymmetry  = (left_outdeg - right_outdeg) / (left + right)'),
    ]:
        x_positions = []
        x_labels    = []

        for i, pat in enumerate(patients):
            mask  = (patient_ids == pat)
            pre   = X[mask & (y == 0), feat_idx]
            ictal = X[mask & (y == 1), feat_idx]
            base  = i * 3.0

            # Background shading by expected sign
            info = PATIENT_INFO.get(pat, '')
            if 'RIGHT' in info:
                ax.axvspan(base - 1.4, base + 1.4, alpha=0.07, color='red')
            elif 'LEFT' in info:
                ax.axvspan(base - 1.4, base + 1.4, alpha=0.07, color='green')

            # Box plots
            ax.boxplot(pre,   positions=[base - 0.6], widths=0.8,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=PRE_COLOR,   alpha=0.6),
                       medianprops=dict(color='navy',       linewidth=2),
                       whiskerprops=dict(color=PRE_COLOR),
                       capprops=dict(color=PRE_COLOR))

            ax.boxplot(ictal, positions=[base + 0.6], widths=0.8,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=ICTAL_COLOR, alpha=0.8),
                       medianprops=dict(color='darkred',    linewidth=2),
                       whiskerprops=dict(color=ICTAL_COLOR),
                       capprops=dict(color=ICTAL_COLOR))

            # Scatter (jittered) so individual points are visible
            np.random.seed(42)
            ax.scatter(base - 0.6 + np.random.uniform(-0.3, 0.3, len(pre)),
                       pre,   s=8,  color=PRE_COLOR,   alpha=0.35, zorder=3)
            ax.scatter(base + 0.6 + np.random.uniform(-0.3, 0.3, len(ictal)),
                       ictal, s=10, color=ICTAL_COLOR, alpha=0.55, zorder=3)

            x_positions.append(base)
            x_labels.append(pat)

        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel('Asymmetry index  (-1 to +1)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25, axis='y')

    # Legend
    pre_h   = plt.Line2D([0],[0], marker='s', color='w',
                          markerfacecolor=PRE_COLOR,   markersize=10, label='Pre-ictal')
    ict_h   = plt.Line2D([0],[0], marker='s', color='w',
                          markerfacecolor=ICTAL_COLOR, markersize=10, label='Ictal')
    red_h   = plt.Rectangle((0,0),1,1, fc='red',   alpha=0.2, label='Expected < 0 (right focal)')
    grn_h   = plt.Rectangle((0,0),1,1, fc='green', alpha=0.2, label='Expected > 0 (left focal)')
    axes[1].legend(handles=[pre_h, ict_h, red_h, grn_h], loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'sanity_asymmetry.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: sanity_asymmetry.png')


# ─────────────────────────────────────────────────────────────
# Check 2 — Spectral features: ictal vs pre-ictal
# ─────────────────────────────────────────────────────────────

def plot_spectral_check(X, y, patient_ids, feat_names, output_dir):
    """
    Mean relative band power per region, ictal vs pre-ictal.
    Expected: delta UP, alpha/beta DOWN during ictal.
    """
    bands   = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broad']
    regions = ['frontal', 'temporal', 'central', 'parietal', 'occipital']

    fig, axes = plt.subplots(1, len(regions), figsize=(18, 5), sharey=False)
    fig.suptitle(
        'Relative Band Power: ictal vs pre-ictal per brain region\n'
        'Expected: delta UP, alpha and beta DOWN during seizure',
        fontsize=11, fontweight='bold'
    )

    for ax, region in zip(axes, regions):
        indices    = [feat_names.index(f'spec_{b}_{region}') for b in bands]
        pre_means  = X[y == 0][:, indices].mean(axis=0)
        ict_means  = X[y == 1][:, indices].mean(axis=0)
        pre_std    = X[y == 0][:, indices].std(axis=0)
        ict_std    = X[y == 1][:, indices].std(axis=0)

        x = np.arange(len(bands))
        w = 0.35
        ax.bar(x - w/2, pre_means, w, yerr=pre_std, label='Pre-ictal',
               color='#5B9BD5', alpha=0.75, capsize=3)
        ax.bar(x + w/2, ict_means, w, yerr=ict_std, label='Ictal',
               color='#ED7D31', alpha=0.85, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45, fontsize=8)
        ax.set_title(region.capitalize(), fontsize=10, fontweight='bold')
        if region == 'frontal':
            ax.set_ylabel('Relative power', fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')

    axes[-1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'sanity_spectral.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: sanity_spectral.png')


# ─────────────────────────────────────────────────────────────
# Check 3 — Connectivity global mean: ictal vs pre-ictal
# ─────────────────────────────────────────────────────────────

def plot_connectivity_check(X, y, patient_ids, feat_names, output_dir):
    """
    Per-patient bar chart of DTF and PDC global mean connectivity.
    Ictal should be higher than pre-ictal (hypersynchrony during seizure).
    """
    dtf_idx = feat_names.index('graph_dtf_global_mean')
    pdc_idx = feat_names.index('graph_pdc_global_mean')
    patients = sorted(np.unique(patient_ids))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Global Connectivity Mean per Patient\n'
        'Expected: ictal >= pre-ictal (hypersynchrony)',
        fontsize=11, fontweight='bold'
    )

    for ax, feat_idx, title in [
        (axes[0], dtf_idx, 'DTF Global Mean'),
        (axes[1], pdc_idx, 'PDC Global Mean'),
    ]:
        pre_m, ict_m = [], []
        pre_s, ict_s = [], []
        for pat in patients:
            mask = (patient_ids == pat)
            pre  = X[mask & (y == 0), feat_idx]
            ict  = X[mask & (y == 1), feat_idx]
            pre_m.append(pre.mean())
            ict_m.append(ict.mean())
            pre_s.append(pre.std() / max(np.sqrt(len(pre)), 1))
            ict_s.append(ict.std() / max(np.sqrt(len(ict)), 1))

        x = np.arange(len(patients))
        w = 0.35
        ax.bar(x - w/2, pre_m, w, yerr=pre_s, color='#5B9BD5', alpha=0.75,
               label='Pre-ictal', capsize=4)
        ax.bar(x + w/2, ict_m, w, yerr=ict_s, color='#ED7D31', alpha=0.85,
               label='Ictal',     capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(patients, rotation=30, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean connectivity', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'sanity_connectivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: sanity_connectivity.png')


# ─────────────────────────────────────────────────────────────
# Check 4 — Features over time (one plot per patient)
# ─────────────────────────────────────────────────────────────

def plot_features_over_time(X, y, patient_ids, tfo, feat_names, output_dir):
    """
    For each patient: 3 key features plotted against time_from_onset.
    Vertical red dashed line = seizure onset (t = 0).
    You should visually see the feature change at or just before t=0.
    """
    patients = sorted(np.unique(patient_ids))

    feats_to_plot = [
        ('graph_dtf_asymmetry',  'DTF Asymmetry',        '#7F77DD'),
        ('spec_delta_frontal',   'Delta power (frontal)', '#ED7D31'),
        ('spec_alpha_frontal',   'Alpha power (frontal)', '#5B9BD5'),
    ]

    fig, axes = plt.subplots(
        len(patients), 3,
        figsize=(16, len(patients) * 2.2),
        sharex=False
    )
    fig.suptitle(
        'Features over time — red line = seizure onset (t=0)\n'
        'Asymmetry should shift; delta should spike UP; alpha should dip DOWN',
        fontsize=11, fontweight='bold'
    )

    for row, pat in enumerate(patients):
        mask     = (patient_ids == pat)
        t        = tfo[mask]
        sort_idx = np.argsort(t)
        t_sorted = t[sort_idx]
        y_sorted = y[mask][sort_idx]

        for col, (fname, flabel, color) in enumerate(feats_to_plot):
            ax    = axes[row, col]
            fidx  = feat_names.index(fname)
            vals  = X[mask, fidx][sort_idx]

            pre_m = (y_sorted == 0)
            ict_m = (y_sorted == 1)

            ax.scatter(t_sorted[pre_m], vals[pre_m],
                       s=10, color='#5B9BD5', alpha=0.5, label='Pre-ictal')
            ax.scatter(t_sorted[ict_m], vals[ict_m],
                       s=14, color='#ED7D31', alpha=0.8, zorder=3, label='Ictal')
            ax.axvline(0, color='red', linewidth=1.5, linestyle='--', alpha=0.9)
            ax.axhline(0, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)

            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(pat, fontsize=8, fontweight='bold')
            if row == 0:
                ax.set_title(flabel, fontsize=9, fontweight='bold')
            if row == len(patients) - 1:
                ax.set_xlabel('Time from onset (s)', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'sanity_time_series.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('  Saved: sanity_time_series.png')


# ─────────────────────────────────────────────────────────────
# Terminal table: numerical PASS/FAIL
# ─────────────────────────────────────────────────────────────

def print_sanity_tables(X, y, patient_ids, feat_names):

    dtf_idx = feat_names.index('graph_dtf_asymmetry')
    pdc_idx = feat_names.index('graph_pdc_asymmetry')
    patients = sorted(np.unique(patient_ids))

    print('\n' + '=' * 82)
    print('ASYMMETRY SANITY CHECK')
    print('Asymmetry = (left_outdeg - right_outdeg) / (left_outdeg + right_outdeg)')
    print('=' * 82)
    print(f"{'Patient':<10} {'Expected':<8} {'DTF pre':>10} {'DTF ictal':>10} "
          f"{'PDC pre':>10} {'PDC ictal':>10}  Result")
    print('-' * 82)

    for pat in patients:
        mask  = (patient_ids == pat)
        pre   = X[mask & (y == 0)]
        ictal = X[mask & (y == 1)]

        dtf_pre   = pre[:,  dtf_idx].mean()
        dtf_ictal = ictal[:, dtf_idx].mean()
        pdc_pre   = pre[:,  pdc_idx].mean()
        pdc_ictal = ictal[:, pdc_idx].mean()

        info = PATIENT_INFO.get(pat, '')
        if 'RIGHT' in info:
            expected = '< 0'
            ok = 'PASS' if dtf_ictal < 0 else 'FAIL'
        elif 'LEFT' in info:
            expected = '> 0'
            ok = 'PASS' if dtf_ictal > 0 else 'FAIL'
        else:
            expected = '~ 0'
            ok = 'N/A'

        mark = 'OK' if ok == 'PASS' else ('!!' if ok == 'FAIL' else '--')
        print(f"{pat:<10} {expected:<8} {dtf_pre:>10.4f} {dtf_ictal:>10.4f} "
              f"{pdc_pre:>10.4f} {pdc_ictal:>10.4f}  [{mark}] {ok}")

    print('=' * 82)

    # Spectral check
    print('\n' + '=' * 62)
    print('SPECTRAL CHECK (frontal region, all patients pooled)')
    print('Expected: delta UP, alpha DOWN, beta DOWN during ictal')
    print('=' * 62)
    print(f"{'Band':<10} {'Pre-ictal':>12} {'Ictal':>12}  {'Change':>8}  Result")
    print('-' * 62)
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broad']
    for band in bands:
        fname = f'spec_{band}_frontal'
        if fname not in feat_names:
            continue
        idx       = feat_names.index(fname)
        pre_mean  = X[y == 0, idx].mean()
        ict_mean  = X[y == 1, idx].mean()
        direction = 'UP' if ict_mean > pre_mean else 'DOWN'
        if band == 'delta':
            ok = 'PASS' if direction == 'UP'   else 'FAIL'
        elif band in ('alpha', 'beta'):
            ok = 'PASS' if direction == 'DOWN' else 'FAIL'
        else:
            ok = 'N/A'
        mark = 'OK' if ok == 'PASS' else ('!!' if ok == 'FAIL' else '--')
        print(f"{band:<10} {pre_mean:>12.5f} {ict_mean:>12.5f}  {direction:>8}  [{mark}] {ok}")
    print('=' * 62)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Feature sanity check')
    parser.add_argument('--featfile',  required=True, help='path to features_all.npz')
    parser.add_argument('--outputdir', default='results/feature_analysis')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading features...')
    X, y, patient_ids, tfo, feat_names = load_data(args.featfile)
    print(f'  Epochs    : {len(y)}  (ictal={int((y==1).sum())}, pre-ictal={int((y==0).sum())})')
    print(f'  Features  : {len(feat_names)}')
    print(f'  Patients  : {sorted(np.unique(patient_ids).tolist())}')

    # Print PASS/FAIL tables to terminal
    print_sanity_tables(X, y, patient_ids, feat_names)

    # Generate all plots
    print('\nGenerating plots...')
    plot_asymmetry_check(X, y, patient_ids, feat_names, output_dir)
    plot_spectral_check(X, y, patient_ids, feat_names, output_dir)
    plot_connectivity_check(X, y, patient_ids, feat_names, output_dir)
    plot_features_over_time(X, y, patient_ids, tfo, feat_names, output_dir)

    print(f'\nAll plots saved to: {output_dir}')
    print()
    print('What to look for:')
    print('  sanity_asymmetry.png    -- ictal boxes for PAT13/14/27 below zero line?')
    print('  sanity_spectral.png     -- delta bar taller for ictal? alpha shorter?')
    print('  sanity_connectivity.png -- ictal bars higher than pre-ictal?')
    print('  sanity_time_series.png  -- features shift at the red dashed line (t=0)?')


if __name__ == '__main__':
    main()