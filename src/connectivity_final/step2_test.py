"""
Step 2 — Extended Test Suite  (with full matrix printout + heatmap saving)
===========================================================================
All ground-truth tests print the full K×K matrix AND save a heatmap PNG.

Heatmaps saved to:  test_outputs/heatmaps/
  T1_random.png
  T5_baccala.png
  T6_hub_chain.png
  T9_epoch.png
  T11_baccala_spectrum.png
  T12_chain.png
  T13_tree.png
  T14_eeg19ch.png

Tests:
  T1–T3  : Algebraic properties (value range, row/col normalisation)
  T4     : Diagonal behaviour — ≤1 always; ≈1 only for isolated channels
  T5–T8  : Baccalá (1991) 5-ch ground-truth network recovery
  T9     : process_single_epoch() end-to-end with real epoch shape (19,1024)
  T11    : Spectrum shape, frequency axis, isolated-node diagonal property
  T12    : Linear chain — indirect path discrimination
  T13    : Tree topology — no false cross-branch PDC edges
  T14    : Full 19-ch epilepsy-inspired network

Usage:
    python step2_tests.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR

sys.path.insert(0, '.')
from connectivity_final.step2_compute_connectivity import (
    compute_dtf_pdc_from_var,
    verify_spectrum,
    process_single_epoch,
)

FS   = 256
NFFT = 512
SEED = 42
N    = 40_000
TOL  = 1e-6
THR  = 0.04
ISO  = 0.04

PASS = '\u2705 PASS'
FAIL = '\u274c FAIL'
results = {}

HEATMAP_DIR = Path('test_outputs/heatmaps')
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_ar_signal(A1, K, n=N, noise=0.25, seed=SEED):
    rng = np.random.default_rng(seed)
    x   = np.zeros((n, K))
    eps = rng.normal(0, noise, (n, K))
    for t in range(1, n):
        x[t] = A1 @ x[t-1] + eps[t]
    return x


def fit_var(signal, order=4):
    std = np.std(signal)
    res = VAR(signal / std).fit(maxlags=order, trend='c', verbose=False)
    return res.coefs


def band_avg(spectrum, freqs, flo=0.5, fhi=45.0):
    m = spectrum[:, :, (freqs >= flo) & (freqs <= fhi)].mean(axis=2)
    np.fill_diagonal(m, 0.0)
    return m


def section(title):
    sep = '\u2550' * 72
    print(f'\n{sep}')
    print(f'  {title}')
    print(sep)


def check(name, cond, detail=''):
    tag = PASS if cond else FAIL
    results[name] = cond
    print(f'  {tag}  {name}')
    if detail:
        print(f'       {detail}')
    return cond


def print_full_matrix(mat, chan_names, title, thr=THR):
    K = mat.shape[0]
    col_w = max(len(c) for c in chan_names) + 2
    sep = '\u2500' * (col_w * (K + 1) + 4)
    print(f'\n  \u250c\u2500 {title} \u2510')
    print(f'  {sep}')
    header = ' ' * (col_w + 2) + ' '.join(f'{c:>{col_w}}' for c in chan_names)
    print(f'  {header}')
    print(f'  {" " * (col_w + 2)}' + ' '.join(['\u2500' * col_w] * K))
    for i, row_name in enumerate(chan_names):
        row_vals = []
        for j in range(K):
            val = mat[i, j]
            if i == j:
                cell = f'[{val:.3f}]' if val > 0 else f' {val:.3f} '
            elif val > thr:
                cell = f'[{val:.3f}]'
            else:
                cell = f' {val:.3f} '
            row_vals.append(f'{cell:>{col_w}}')
        print(f'  {row_name:>{col_w}} \u2502 ' + ' '.join(row_vals))
    print(f'  {sep}')
    print(f'  Rows=Sinks(To), Cols=Sources(From)  \u2502  [value]=detected(>{thr})  \u2502  diag=self-conn')


def save_heatmap_pair(dtf, pdc, chan_names, test_id, title,
                      true_links=None, thr=THR):
    K = dtf.shape[0]
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(10, K * 0.7 + 2) * 2, max(7, K * 0.6 + 2))
    )
    for ax, mat, mname, note in [
        (axes[0], dtf, 'DTF',
         'Bright COLUMNS = strong sources\nrow-normalised | direct + indirect'),
        (axes[1], pdc, 'PDC',
         'Bright ROWS = strong sinks\ncol-normalised | direct only'),
    ]:
        sns.heatmap(
            mat, ax=ax,
            vmin=0, vmax=1 if mat.max() <= 1 else None,
            cmap='viridis', square=True,
            xticklabels=chan_names, yticklabels=chan_names,
            linewidths=0.3, linecolor='#333333',
            cbar_kws={'label': 'Connectivity (diag=0)'},
            annot=(K <= 12),
            fmt='.2f' if K <= 12 else '',
            annot_kws={'size': 7} if K <= 12 else {},
        )
        if true_links:
            for src, snk in true_links:
                ax.add_patch(mpatches.Rectangle(
                    (src, snk), 1, 1,
                    fill=False, edgecolor='red', lw=2.5, zorder=5
                ))
        for i in range(K):
            for j in range(K):
                if i != j and mat[i, j] > thr:
                    ax.plot(j + 0.12, i + 0.12, 'w.', markersize=4, zorder=6)
        ax.set_title(f'{mname}\n{note}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Source (From j)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Sink   (To i)',   fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=7 if K > 8 else 9)
        ax.tick_params(axis='y', rotation=0,  labelsize=7 if K > 8 else 9)

    legend_items = [
        mpatches.Patch(facecolor='none', edgecolor='red',
                       label='True link (ground truth)'),
        plt.Line2D([0], [0], marker='.', color='white',
                   markerfacecolor='white', markersize=8,
                   label=f'Detected (>{thr})'),
    ] if true_links else []
    if legend_items:
        fig.legend(handles=legend_items, loc='lower center',
                   ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'{test_id} — {title}\n'
        'Rows=Sinks(To i) \u2502 Cols=Sources(From j) \u2502 scale [0,1]',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    out = HEATMAP_DIR / f'{test_id.replace(" ", "_")}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  \U0001f4be  Saved: {out}')


def print_spectrum_stats(dtf_s, pdc_s, freqs, chan_names, title=''):
    mask  = (freqs >= 0.5) & (freqs <= 45.0)
    dtf_m = dtf_s[:, :, mask].mean(axis=2); np.fill_diagonal(dtf_m, 0)
    pdc_m = pdc_s[:, :, mask].mean(axis=2); np.fill_diagonal(pdc_m, 0)
    print_full_matrix(dtf_m, chan_names, f'DTF spectrum-avg {title}')
    print_full_matrix(pdc_m, chan_names, f'PDC spectrum-avg {title}')
    return dtf_m, pdc_m


# ─────────────────────────────────────────────────────────────────────────────
# Shared random signal  (used by T1–T4)
# ─────────────────────────────────────────────────────────────────────────────

K     = 5
rng   = np.random.default_rng(SEED)
A1_r  = np.diag([0.3] * K) + rng.normal(0, 0.05, (K, K))
sig_r = make_ar_signal(A1_r, K)
coefs = fit_var(sig_r, order=4)
dtf_s, pdc_s, freqs = compute_dtf_pdc_from_var(coefs, FS, NFFT)
chan_rand = [f'ch{i}' for i in range(K)]


# ══════════════════════════════════════════════════════════════════════════════
# T1 — Value range
# ══════════════════════════════════════════════════════════════════════════════

section('T1 — Value range  (random 5-ch AR signal)')
dtf_m1, pdc_m1 = print_spectrum_stats(dtf_s, pdc_s, freqs, chan_rand, '(random AR)')
save_heatmap_pair(dtf_m1, pdc_m1, chan_rand, 'T1_random', 'Random 5-ch AR signal')
check('T1a  DTF >= 0',     dtf_s.min() >= 0,       f'min={dtf_s.min():.6f}')
check('T1b  DTF <= 1',     dtf_s.max() <= 1 + TOL, f'max={dtf_s.max():.6f}')
check('T1c  PDC >= 0',     pdc_s.min() >= 0,       f'min={pdc_s.min():.6f}')
check('T1d  PDC <= 1',     pdc_s.max() <= 1 + TOL, f'max={pdc_s.max():.6f}')


# ══════════════════════════════════════════════════════════════════════════════
# T2 — DTF² row sums = 1
# ══════════════════════════════════════════════════════════════════════════════

section('T2 — DTF\u00b2 row sums = 1 at every frequency')
dtf2_row = (dtf_s ** 2).sum(axis=1)
dtf_dev  = float(np.abs(dtf2_row - 1.0).max())
print(f'  DTF^2 row-sum range: [{dtf2_row.min():.10f}, {dtf2_row.max():.10f}]')
check('T2   DTF^2 row-sums = 1', dtf_dev < TOL, f'max dev={dtf_dev:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T3 — PDC² col sums = 1
# ══════════════════════════════════════════════════════════════════════════════

section('T3 — PDC\u00b2 col sums = 1 at every frequency')
pdc2_col = (pdc_s ** 2).sum(axis=0)
pdc_dev  = float(np.abs(pdc2_col - 1.0).max())
print(f'  PDC^2 col-sum range: [{pdc2_col.min():.10f}, {pdc2_col.max():.10f}]')
check('T3   PDC^2 col-sums = 1', pdc_dev < TOL, f'max dev={pdc_dev:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T4 — Diagonal behaviour (corrected)
#
# The diagonal DTF[i,i,f] is NOT always 1.  From the row-sum rule:
#   sum_j DTF²[i,j,f] = 1  =>  DTF²[i,i,f] = 1 - sum_{j≠i} DTF²[i,j,f]
#
# Therefore:
#   • DTF[i,i,f] <= 1  always  (it is the remainder after off-diagonal terms)
#   • DTF[i,i,f] ≈ 1  only when channel i receives negligible input from others
#     (isolated channel)
#   • DTF[i,i,f] << 1 when channel i is strongly driven by other channels
#
# This test uses the shared random signal (very weak cross-channel coupling
# ~0.05) so ALL channels are approximately isolated → diagonal ≈ 1.
# T11 separately verifies that for a TRULY isolated node the diagonal is
# exactly ≈ 1, and for a hub node it is suppressed well below 1.
# ══════════════════════════════════════════════════════════════════════════════

section('T4 — Diagonal <= 1 always; ≈ 1 for weakly-coupled channels')
print('  Using random signal with weak cross-coupling (~0.05):')
print('  All channels are approximately isolated → diagonal expected ≈ 1')
print()

diag_vals_dtf = []
diag_vals_pdc = []
for i in range(K):
    dv = float(dtf_s[i, i, :].mean())
    pv = float(pdc_s[i, i, :].mean())
    diag_vals_dtf.append(dv)
    diag_vals_pdc.append(pv)
    print(f'    ch{i}:  DTF diag (mean)={dv:.4f}   PDC diag (mean)={pv:.4f}')

# Check 1: diagonal is ALWAYS <= 1 (strict mathematical property)
dtf_diag_max = max(float(dtf_s[i, i, :].max()) for i in range(K))
pdc_diag_max = max(float(pdc_s[i, i, :].max()) for i in range(K))
check('T4a  DTF diagonal <= 1 at all frequencies (always true)',
      dtf_diag_max <= 1 + TOL,
      f'max diagonal value = {dtf_diag_max:.6f}')
check('T4b  PDC diagonal <= 1 at all frequencies (always true)',
      pdc_diag_max <= 1 + TOL,
      f'max diagonal value = {pdc_diag_max:.6f}')

# Check 2: for this weakly-coupled signal, diagonal ≈ 1 (signal-specific)
dtf_dg_mean = float(np.mean(diag_vals_dtf))
pdc_dg_mean = float(np.mean(diag_vals_pdc))
check('T4c  DTF diagonal ≈ 1 for weakly-coupled signal (signal-specific)',
      abs(dtf_dg_mean - 1.0) < 0.05,
      f'mean across channels = {dtf_dg_mean:.4f}  (expected ≈ 1 because coupling ~0.05)')
check('T4d  PDC diagonal ≈ 1 for weakly-coupled signal (signal-specific)',
      abs(pdc_dg_mean - 1.0) < 0.05,
      f'mean across channels = {pdc_dg_mean:.4f}  (expected ≈ 1 because coupling ~0.05)')

print()
print('  NOTE: T11 further verifies that a TRULY isolated node has diagonal')
print('        exactly ≈ 1, while a hub (strong source) has diagonal << 1.')


# ══════════════════════════════════════════════════════════════════════════════
# T5 — Baccalá ground-truth network
# ══════════════════════════════════════════════════════════════════════════════

section('T5 — Ground truth: 5-ch Baccal\u00e1 network  (n1\u2192n2,n3,n4  n4\u2194n5)')
print('  Network from Baccala & Sameshima (2001), Fig. 1:')
print('    n1 -> n2  (direct, coefficient 0.85)')
print('    n1 -> n3  (direct, coefficient 0.80)')
print('    n1 -> n4  (direct, coefficient 0.75)')
print('    n4 -> n5  (direct, coefficient 0.72)')
print('    n5 -> n4  (feedback, coefficient 0.55)')
print('  AR coefficients are PLANTED in A — signal is generated to obey them.')
print('  Test checks whether DTF/PDC RECOVER what was planted.')

K5 = 5
A5 = np.zeros((K5, K5))
A5[0, 0] = 0.35
A5[1, 1] = A5[2, 2] = A5[3, 3] = A5[4, 4] = 0.10
A5[1, 0] = 0.85   # n1 -> n2
A5[2, 0] = 0.80   # n1 -> n3
A5[3, 0] = 0.75   # n1 -> n4
A5[4, 3] = 0.72   # n4 -> n5
A5[3, 4] = 0.55   # n5 -> n4  (feedback)

sig5 = make_ar_signal(A5, K5)
c5   = fit_var(sig5, 4)
dtf5_s, pdc5_s, f5 = compute_dtf_pdc_from_var(c5, FS, NFFT)
dtf5 = band_avg(dtf5_s, f5)
pdc5 = band_avg(pdc5_s, f5)
chan5   = ['n1', 'n2', 'n3', 'n4', 'n5']
direct5 = [(0, 1), (0, 2), (0, 3), (3, 4), (4, 3)]

print_full_matrix(dtf5, chan5, 'DTF band-averaged (diag=0)')
print_full_matrix(pdc5, chan5, 'PDC band-averaged (diag=0)')
save_heatmap_pair(dtf5, pdc5, chan5, 'T5_baccala',
                  '5-ch Baccal\u00e1: n1\u2192n2,n3,n4  n4\u2194n5',
                  true_links=direct5)

all5 = all(dtf5[t, s] > THR and pdc5[t, s] > THR for s, t in direct5)
check('T5   All 5 direct connections recovered (DTF+PDC > 0.04)', all5)
vp5, vd5, vc5 = verify_spectrum(dtf5_s, pdc5_s)
check('T5n  Normalisation at spectrum level', vp5,
      f'DTF dev={vd5:.2e}  PDC dev={vc5:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T6 — 10-ch hub+chain
# ══════════════════════════════════════════════════════════════════════════════

section('T6 — Ground truth: 10-ch hub+chain+bridge')
print('  True links: 0->1->2,  0->3,  0->4->5->6,  4->7->8->9')

K10 = 10
A10 = np.diag([0.25] * K10)
for s, t, v in [(0,1,.85),(1,2,.82),(0,3,.75),(0,4,.70),
                (4,5,.80),(5,6,.78),(4,7,.65),(7,8,.80),(8,9,.78)]:
    A10[t, s] = v

sig10 = make_ar_signal(A10, K10)
c10   = fit_var(sig10, 4)
dtf10_s, pdc10_s, f10 = compute_dtf_pdc_from_var(c10, FS, NFFT)
dtf10 = band_avg(dtf10_s, f10)
pdc10 = band_avg(pdc10_s, f10)
chan10   = [f'c{i}' for i in range(10)]
direct10 = [(0,1),(1,2),(0,3),(0,4),(4,5),(5,6),(4,7),(7,8),(8,9)]

print_full_matrix(dtf10, chan10, 'DTF band-averaged (diag=0)')
print_full_matrix(pdc10, chan10, 'PDC band-averaged (diag=0)')
save_heatmap_pair(dtf10, pdc10, chan10, 'T6_hub_chain',
                  '10-ch hub+chain: 0->1->2, 0->3, 0->4->5->6, 4->7->8->9',
                  true_links=direct10)

all10 = all(dtf10[t, s] > THR and pdc10[t, s] > THR for s, t in direct10)
check('T6   All 9 direct connections recovered (DTF+PDC > 0.04)', all10)
vp10, vd10, vc10 = verify_spectrum(dtf10_s, pdc10_s)
check('T6n  Normalisation at spectrum level', vp10,
      f'DTF dev={vd10:.2e}  PDC dev={vc10:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T7 — Indirect connection: DTF detects, PDC does NOT
# ══════════════════════════════════════════════════════════════════════════════

section('T7 — Indirect connection  (DTF detects, PDC does NOT)')
print('  Path: n1 -> n4 -> n5  (indirect, no direct n1->n5 edge)')
print('  DTF captures total causal influence (direct + indirect paths).')
print('  PDC captures only direct MVAR coefficients -> should be ~0 for n1->n5.')

dind = float(dtf5[4, 0])
pind = float(pdc5[4, 0])
print(f'  n1->n5  DTF[4,0]={dind:.4f}   PDC[4,0]={pind:.4f}')
check('T7a  Indirect n1->n5: DTF > 0.04 (detected)',  dind > THR, f'DTF={dind:.4f}')
check('T7b  Indirect n1->n5: PDC <= 0.04 (not direct)', pind <= ISO, f'PDC={pind:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# T8 — Bidirectional feedback n4 <-> n5
# ══════════════════════════════════════════════════════════════════════════════

section('T8 — Bidirectional feedback  n4 \u2194 n5')
print('  Planted: A[n5,n4]=0.72 (forward),  A[n4,n5]=0.55 (feedback)')
print('  PDC should detect both directions, with different strengths.')

fwd  = float(pdc5[4, 3])   # n4 -> n5
back = float(pdc5[3, 4])   # n5 -> n4
print(f'  PDC[n5,n4] forward  = {fwd:.4f}')
print(f'  PDC[n4,n5] feedback = {back:.4f}')
check('T8a  Forward  n4->n5: PDC > 0.04', fwd  > THR, f'PDC={fwd:.4f}')
check('T8b  Feedback n5->n4: PDC > 0.04', back > THR, f'PDC={back:.4f}')
check('T8c  Asymmetric (|fwd-back| > 0.01)',
      abs(fwd - back) > 0.01,
      f'|{fwd:.4f} - {back:.4f}| = {abs(fwd-back):.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# T9 — process_single_epoch end-to-end
# ══════════════════════════════════════════════════════════════════════════════

section('T9 — process_single_epoch() end-to-end  (19-ch, 1024-sample epoch)')
print('  Creates a synthetic 19-channel signal with three planted connections:')
print('    Fp1 -> Fp2  (coefficient 0.8)')
print('    Fp2 -> F7   (coefficient 0.8)')
print('    Fp1 -> F3   (coefficient 0.7)')
print('  Takes only the first 1024 samples -> shape (19, 1024), same as real EEG epochs.')
print('  Calls process_single_epoch() — the ACTUAL function used on real data.')
print('  Verifies output format, shape, band structure, diagonal zeroing.')

K19 = 19
A19 = np.diag([0.3] * K19)
A19[1, 0] = 0.8   # Fp1 -> Fp2
A19[2, 1] = 0.8   # Fp2 -> F7
A19[3, 0] = 0.7   # Fp1 -> F3

sig19 = make_ar_signal(A19, K19, n=5000)
epoch = sig19[:1024, :].T   # shape (19, 1024)

res9 = process_single_epoch(
    epoch, fs=256., fixed_order=4, nfft=512, verify=True, epoch_idx=0
)

CHANNEL_NAMES_19 = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T3','C3','Cz','C4','T4',
    'T5','P3','Pz','P4','T6',
    'O1','O2',
]

check('T9a  Returns dict (not rejected)', res9 is not None)
if res9:
    di = res9['dtf_bands']['integrated']
    pi = res9['pdc_bands']['integrated']
    print_full_matrix(di, CHANNEL_NAMES_19, 'T9 DTF integrated band (diag=0)')
    print_full_matrix(pi, CHANNEL_NAMES_19, 'T9 PDC integrated band (diag=0)')
    save_heatmap_pair(di, pi, CHANNEL_NAMES_19, 'T9_epoch',
                      'process_single_epoch — integrated band',
                      true_links=[(0,1),(1,2),(0,3)])
    check('T9b  All 6 bands present',
          all(b in res9['dtf_bands']
              for b in ['integrated','delta','theta','alpha','beta','gamma1']))
    check('T9c  Output shape (19,19)', di.shape == (19, 19), f'{di.shape}')
    check('T9d  Diagonal = 0 after band averaging',
          np.diag(di).max() == 0 and np.diag(pi).max() == 0)
    check('T9e  All values in [0,1]',
          di.max() <= 1 and di.min() >= 0 and pi.max() <= 1 and pi.min() >= 0)
    check('T9f  Planted conn Fp1->Fp2 detected (DTF > 0.04)',
          float(di[1, 0]) > THR, f'DTF[Fp2,Fp1]={di[1,0]:.4f}')
    check('T9g  Full spectrum saved for visualisation',
          'dtf_spectrum' in res9)


# ══════════════════════════════════════════════════════════════════════════════
# T11 — Spectrum shape + isolated-node diagonal property
# ══════════════════════════════════════════════════════════════════════════════

section('T11 — Spectrum shape, frequency axis, isolated-node diagonal')
print('  Modified Baccala network with n2 TRULY ISOLATED (zero AR coefficients):')
print('    n0 -> n1  (hub source)')
print('    n0 -> n3  ')
print('    n3 -> n4  ')
print('    n2: no connections whatsoever')
print()
print('  Checks:')
print('    (a) Output spectrum has correct shape: (K, K, NFFT/2+1) = (5,5,257)')
print('    (b) Frequency axis: 0 Hz to fs/2 = 128 Hz')
print('    (c) ISOLATED node n2: DTF[n2,n2,f] ≈ 1 at all frequencies')
print("        because no other channel drives n2, so all inflow is self-inflow")
print('    (d) CONNECTED hub n0: DTF[n0,n0,f] mean < 0.95')
print('        because n0 drives others, so its row energy spreads off-diagonal')

K11 = 5
A11 = np.zeros((K11, K11))
A11[0, 0] = 0.35
A11[1, 1] = A11[2, 2] = A11[3, 3] = A11[4, 4] = 0.10
A11[1, 0] = 0.85   # n0 -> n1
A11[3, 0] = 0.75   # n0 -> n3
A11[4, 3] = 0.72   # n3 -> n4
# n2 has NO connections at all

sig11 = make_ar_signal(A11, K11)
c11   = fit_var(sig11, 4)
dtf11_s, pdc11_s, f11 = compute_dtf_pdc_from_var(c11, FS, NFFT)
chan11 = [f'n{i}' for i in range(K11)]
n_freqs_exp = NFFT // 2 + 1

print('\n  Per-channel mean DTF diagonal (before zeroing):')
for i in range(K11):
    dv   = float(dtf11_s[i, i, :].mean())
    role = (' <- hub (strong source, diagonal suppressed)' if i == 0 else
            ' <- ISOLATED (no connections, diagonal = 1)' if i == 2 else
            ' <- sink')
    print(f'    n{i}: mean DTF[{i},{i}] = {dv:.4f}{role}')

dtf11_m, pdc11_m = print_spectrum_stats(dtf11_s, pdc11_s, f11, chan11, 'T11')
save_heatmap_pair(dtf11_m, pdc11_m, chan11, 'T11_baccala_spectrum',
                  'T11 Spectrum check (n0=hub, n2=isolated, n4=sink)',
                  true_links=[(0,1),(0,3),(3,4)])

check('T11a  DTF spectrum shape = (K, K, NFFT/2+1)',
      dtf11_s.shape == (K11, K11, n_freqs_exp), f'{dtf11_s.shape}')
check('T11b  PDC spectrum shape = (K, K, NFFT/2+1)',
      pdc11_s.shape == (K11, K11, n_freqs_exp), f'{pdc11_s.shape}')
check('T11c  Frequency axis max = fs/2 = 128 Hz',
      abs(f11[-1] - FS / 2) < 0.1, f'f[-1]={f11[-1]:.2f} Hz')
check('T11d  Frequency axis min = 0 Hz',
      abs(f11[0]) < 0.1, f'f[0]={f11[0]:.4f} Hz')

diag_isolated = dtf11_s[2, 2, :]
check('T11e  ISOLATED node n2: DTF diagonal ≈ 1 at all frequencies',
      float(np.abs(diag_isolated - 1.0).max()) < 0.05,
      f'max|DTF[n2,n2,f] - 1| = {np.abs(diag_isolated - 1.0).max():.4f}')

diag_hub = dtf11_s[0, 0, :]
check('T11f  HUB node n0: DTF diagonal mean < 0.95 (energy spread to outputs)',
      float(diag_hub.mean()) < 0.95,
      f'mean DTF[n0,n0,:] = {diag_hub.mean():.4f}')

check('T11g  DTF spectrum values in [0,1]',
      dtf11_s.min() >= 0 and dtf11_s.max() <= 1 + TOL)
check('T11h  PDC spectrum values in [0,1]',
      pdc11_s.min() >= 0 and pdc11_s.max() <= 1 + TOL)
conn_spec = dtf11_s[1, 0, :]
check('T11i  Known conn n0->n1: mean spectrum DTF > 0.04',
      float(conn_spec.mean()) > THR,
      f'mean DTF[n1,n0,:] = {conn_spec.mean():.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# T12 — Linear chain + isolated node
# ══════════════════════════════════════════════════════════════════════════════

section('T12 — Ground truth: chain  Ch1->Ch2->Ch3->Ch4,  Ch5=pure noise')
print('  Direct:   Ch1->Ch2, Ch2->Ch3, Ch3->Ch4')
print('  Indirect: Ch1->Ch3, Ch1->Ch4, Ch2->Ch4  (DTF yes / PDC no)')
print('  Isolated: Ch5 (pure noise, no AR coupling)')

K12 = 5
A12 = np.zeros((K12, K12))
A12[0,0]=0.30; A12[1,1]=0.30; A12[2,2]=0.30; A12[3,3]=0.30; A12[4,4]=0.10
A12[1,0]=0.88; A12[2,1]=0.85; A12[3,2]=0.82

sig12 = make_ar_signal(A12, K12)
c12   = fit_var(sig12, 4)
dtf12_s, pdc12_s, f12 = compute_dtf_pdc_from_var(c12, FS, NFFT)
dtf12 = band_avg(dtf12_s, f12)
pdc12 = band_avg(pdc12_s, f12)
chan12   = ['Ch1','Ch2','Ch3','Ch4','Ch5']
direct12   = [(0,1),(1,2),(2,3)]
indirect12 = [(0,2),(0,3),(1,3)]

print_full_matrix(dtf12, chan12, 'DTF band-averaged (diag=0)')
print_full_matrix(pdc12, chan12, 'PDC band-averaged (diag=0)')
save_heatmap_pair(dtf12, pdc12, chan12, 'T12_chain',
                  'Chain: Ch1->Ch2->Ch3->Ch4,  Ch5=noise',
                  true_links=direct12)

check('T12a  Direct links (DTF+PDC > 0.04)',
      all(dtf12[t,s] > THR and pdc12[t,s] > THR for s,t in direct12))
check('T12b  Indirect links detected by DTF',
      all(dtf12[t,s] > THR for s,t in indirect12),
      ' '.join(f'DTF[{t},{s}]={dtf12[t,s]:.3f}' for s,t in indirect12))
check('T12c  Indirect links NOT in PDC (direct edges only)',
      all(pdc12[t,s] <= ISO for s,t in indirect12),
      ' '.join(f'PDC[{t},{s}]={pdc12[t,s]:.3f}' for s,t in indirect12))
check('T12d  Isolated Ch5: DTF col ≈ 0', float(dtf12[:,4].max()) <= ISO)
check('T12e  Isolated Ch5: DTF row ≈ 0', float(dtf12[4,:].max()) <= ISO)
check('T12f  Isolated Ch5: PDC col ≈ 0', float(pdc12[:,4].max()) <= ISO)
check('T12g  Isolated Ch5: PDC row ≈ 0', float(pdc12[4,:].max()) <= ISO)
vp12, vd12, vc12 = verify_spectrum(dtf12_s, pdc12_s)
check('T12n  Normalisation at spectrum level', vp12,
      f'DTF dev={vd12:.2e}  PDC dev={vc12:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T13 — Tree topology
# ══════════════════════════════════════════════════════════════════════════════

section('T13 — Ground truth: tree  Ch1->Ch2->Ch4,  Ch1->Ch3,  Ch5=noise')
print('  Direct:   Ch1->Ch2, Ch1->Ch3, Ch2->Ch4')
print('  Indirect: Ch1->Ch4 via Ch2  (DTF yes / PDC no)')
print('  Cross-branch: Ch3<->Ch4 should NOT appear in PDC (no direct edge)')

K13 = 5
A13 = np.zeros((K13, K13))
A13[0,0]=0.30; A13[1,1]=0.25; A13[2,2]=0.25; A13[3,3]=0.25; A13[4,4]=0.10
A13[1,0]=0.88; A13[2,0]=0.82; A13[3,1]=0.85

sig13 = make_ar_signal(A13, K13)
c13   = fit_var(sig13, 4)
dtf13_s, pdc13_s, f13 = compute_dtf_pdc_from_var(c13, FS, NFFT)
dtf13 = band_avg(dtf13_s, f13)
pdc13 = band_avg(pdc13_s, f13)
chan13   = ['Ch1','Ch2','Ch3','Ch4','Ch5']
direct13 = [(0,1),(0,2),(1,3)]

print_full_matrix(dtf13, chan13, 'DTF band-averaged (diag=0)')
print_full_matrix(pdc13, chan13, 'PDC band-averaged (diag=0)')
save_heatmap_pair(dtf13, pdc13, chan13, 'T13_tree',
                  'Tree: Ch1->Ch2->Ch4,  Ch1->Ch3,  Ch5=noise',
                  true_links=direct13)

check('T13a  Direct tree links (DTF+PDC > 0.04)',
      all(dtf13[t,s] > THR and pdc13[t,s] > THR for s,t in direct13))
dtf_ind13 = float(dtf13[3,0]); pdc_ind13 = float(pdc13[3,0])
check('T13b  Indirect Ch1->Ch4: DTF detects',  dtf_ind13 > THR,
      f'DTF={dtf_ind13:.4f}')
check('T13c  Indirect Ch1->Ch4: PDC does NOT', pdc_ind13 <= ISO,
      f'PDC={pdc_ind13:.4f}')
check('T13d  No false cross-branch PDC: Ch3<->Ch4 ≈ 0',
      pdc13[3,2] <= ISO and pdc13[2,3] <= ISO,
      f'PDC[Ch4,Ch3]={pdc13[3,2]:.4f}  PDC[Ch3,Ch4]={pdc13[2,3]:.4f}')
check('T13e  Isolated Ch5 ≈ 0',
      float(dtf13[:,4].max()) <= ISO and float(dtf13[4,:].max()) <= ISO)
vp13, vd13, vc13 = verify_spectrum(dtf13_s, pdc13_s)
check('T13n  Normalisation at spectrum level', vp13,
      f'DTF dev={vd13:.2e}  PDC dev={vc13:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T14 — Full 19-ch epilepsy-inspired network
# ══════════════════════════════════════════════════════════════════════════════

section('T14 — Full 19-ch EEG network  (epilepsy-inspired)')
print('  Fp1 is the seizure-onset hub with direct connections:')
print('    Fp1->F3, Fp1->C3, Fp1->P3')
print('  Propagation chains:')
print('    F3->T3, T3->T5, C3->C4, T3->P3')
print('  12 channels are truly isolated.')
print('  Key checks:')
print('    - All 7 direct links detected by both DTF and PDC')
print('    - Fp1 has the highest mean DTF column (= strongest source)')
print('    - Indirect Fp1->T5 detected by DTF but NOT PDC')
print('    - All 12 isolated channels have DTF/PDC < 0.04')

Fp1,Fp2,F7,F3,Fz,F4,F8 = 0,1,2,3,4,5,6
T3, C3, Cz,C4,T4        = 7,8,9,10,11
T5, P3, Pz,P4,T6        = 12,13,14,15,16
O1, O2                   = 17,18
ISOLATED14 = [Fp2,F7,Fz,F4,F8,Cz,T4,Pz,P4,T6,O1,O2]

K14 = 19
A14 = np.diag([0.25] * K14)
DIRECT14 = [
    (Fp1, F3, .85), (Fp1, C3, .80), (Fp1, P3, .75),
    (F3,  T3, .82), (T3,  T5, .78), (C3,  C4, .70), (T3, P3, .65),
]
for src, snk, v in DIRECT14:
    A14[snk, src] = v

sig14 = make_ar_signal(A14, K14, n=N)
c14   = fit_var(sig14, order=6)
dtf14_s, pdc14_s, f14 = compute_dtf_pdc_from_var(c14, FS, NFFT)
dtf14 = band_avg(dtf14_s, f14)
pdc14 = band_avg(pdc14_s, f14)
direct_pairs = [(src, snk) for src, snk, _ in DIRECT14]

print_full_matrix(dtf14, CHANNEL_NAMES_19, 'T14 DTF band-averaged (diag=0)')
print_full_matrix(pdc14, CHANNEL_NAMES_19, 'T14 PDC band-averaged (diag=0)')
save_heatmap_pair(dtf14, pdc14, CHANNEL_NAMES_19, 'T14_eeg19ch',
                  '19-ch EEG: Fp1=hub, chains via F3/C3/T3, 12 isolated',
                  true_links=direct_pairs)

all_direct14 = all(
    dtf14[snk, src] > THR and pdc14[snk, src] > THR
    for src, snk in direct_pairs
)
check('T14a  All 7 direct links (DTF+PDC > 0.04)', all_direct14)

dtf_col_means = np.array([dtf14[:, j].mean() for j in range(K14)])
check('T14b  Fp1 is strongest SOURCE in DTF (highest column mean)',
      int(np.argmax(dtf_col_means)) == Fp1,
      f'Strongest: {CHANNEL_NAMES_19[int(np.argmax(dtf_col_means))]}')

dtf_ind14 = float(dtf14[T5, Fp1])
pdc_ind14 = float(pdc14[T5, Fp1])
check('T14c  Indirect Fp1->T5: DTF detects',  dtf_ind14 > THR,
      f'DTF={dtf_ind14:.4f}')
check('T14d  Indirect Fp1->T5: PDC does NOT', pdc_ind14 <= ISO,
      f'PDC={pdc_ind14:.4f}')

iso_ok_dtf = all(
    float(dtf14[:, j].max()) <= ISO and float(dtf14[j, :].max()) <= ISO
    for j in ISOLATED14
)
iso_ok_pdc = all(
    float(pdc14[:, j].max()) <= ISO and float(pdc14[j, :].max()) <= ISO
    for j in ISOLATED14
)
check('T14e  Isolated channels: DTF ≈ 0', iso_ok_dtf,
      f'max={max(max(float(dtf14[:,j].max()),float(dtf14[j,:].max())) for j in ISOLATED14):.4f}')
check('T14f  Isolated channels: PDC ≈ 0', iso_ok_pdc,
      f'max={max(max(float(pdc14[:,j].max()),float(pdc14[j,:].max())) for j in ISOLATED14):.4f}')
vp14, vd14, vc14 = verify_spectrum(dtf14_s, pdc14_s)
check('T14n  Normalisation at spectrum level', vp14,
      f'DTF dev={vd14:.2e}  PDC dev={vc14:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '\u2550'*72)
print('  FINAL REPORT')
print('\u2550'*72)
n_pass = sum(results.values())
n_fail = sum(not v for v in results.values())
for name, ok in results.items():
    icon = '\u2705' if ok else '\u274c'
    print(f'  {icon}  {name}')
summary = '\u2705  ALL PASS' if n_fail == 0 else f'\u274c  {n_fail} FAILED'
print(f'\n  {n_pass}/{len(results)} passed   {summary}')
print(f'\n  Heatmaps saved to: {HEATMAP_DIR.resolve()}')
print('\u2550'*72)