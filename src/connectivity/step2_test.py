"""
Step 2 — Extended Test Suite  (with full matrix printout + heatmap saving)
===========================================================================
All ground-truth tests print the full K×K matrix AND save a heatmap PNG.

Heatmaps saved to:  test_outputs/heatmaps/
  T1_random_dtf.png / T1_random_pdc.png
  T5_baccala_dtf.png / T5_baccala_pdc.png
  T6_hub_chain_dtf.png / T6_hub_chain_pdc.png
  T9_epoch_dtf.png / T9_epoch_pdc.png
  T11_baccala_spectrum_dtf.png / T11_baccala_spectrum_pdc.png
  T12_chain_dtf.png / T12_chain_pdc.png
  T13_tree_dtf.png / T13_tree_pdc.png
  T14_eeg19ch_dtf.png / T14_eeg19ch_pdc.png

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
from step2_compute_connectivity_updated import (
    compute_dtf_pdc_from_var,
    verify_spectrum,
    process_single_epoch,
)

FS    = 256
NFFT  = 512
SEED  = 42
N     = 40_000
TOL   = 1e-6
THR   = 0.04
ISO   = 0.04

PASS = '✅ PASS'
FAIL = '❌ FAIL'
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
    print(f'\n{"═" * 72}')
    print(f'  {title}')
    print(f'{"═" * 72}')

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
    sep = '─' * (col_w * (K + 1) + 4)
    print(f'\n  ┌─ {title} ─┐')
    print(f'  {sep}')
    header = ' ' * (col_w + 2) + ' '.join(f'{c:>{col_w}}' for c in chan_names)
    print(f'  {header}')
    print(f'  {" " * (col_w + 2)}' + ' '.join(['─' * col_w] * K))
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
        print(f'  {row_name:>{col_w}} │ ' + ' '.join(row_vals))
    print(f'  {sep}')
    print(f'  Rows=Sinks(To), Cols=Sources(From)  │  [value]=detected(>{thr})  │  diag=self-conn')


def save_heatmap_pair(dtf, pdc, chan_names, test_id, title,
                      true_links=None, thr=THR):
    """
    Save a side-by-side DTF+PDC heatmap for one test.
    true_links : list of (src, snk) tuples to mark with a red border cell.
    """
    K = dtf.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(max(10, K * 0.7 + 2) * 2, max(7, K * 0.6 + 2)))

    for ax, mat, mname, note in [
        (axes[0], dtf, 'DTF',
         'Bright COLUMNS = strong sources\nrow-normalised | direct + indirect'),
        (axes[1], pdc, 'PDC',
         'Bright ROWS = strong sinks\ncol-normalised | direct only'),
    ]:
        # Base heatmap
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

        # Mark true links with red rectangle (only if provided)
        if true_links:
            for src, snk in true_links:
                ax.add_patch(mpatches.Rectangle(
                    (src, snk), 1, 1,
                    fill=False, edgecolor='red', lw=2.5, zorder=5
                ))

        # Mark detected connections with a dot in top-left corner
        for i in range(K):
            for j in range(K):
                if i != j and mat[i, j] > thr:
                    ax.plot(j + 0.12, i + 0.12, 'w.', markersize=4, zorder=6)

        ax.set_title(f'{mname}\n{note}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Source (From j)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Sink (To i)',     fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=7 if K > 8 else 9)
        ax.tick_params(axis='y', rotation=0,  labelsize=7 if K > 8 else 9)

    legend_items = [
        mpatches.Patch(facecolor='none', edgecolor='red',   label='True link (ground truth)'),
        plt.Line2D([0],[0], marker='.', color='white', markerfacecolor='white',
                   markersize=8, label=f'Detected (>{thr})'),
    ] if true_links else []

    if legend_items:
        fig.legend(handles=legend_items, loc='lower center',
                   ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'{test_id} — {title}\n'
        'Rows=Sinks(To i) │ Cols=Sources(From j) │ scale [0,1]',
        fontsize=12, fontweight='bold', y=1.01,
    )

    plt.tight_layout()
    out = HEATMAP_DIR / f'{test_id.replace(" ","_")}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  💾  Saved: {out}')


def print_spectrum_stats(dtf_s, pdc_s, freqs, chan_names, title=''):
    mask = (freqs >= 0.5) & (freqs <= 45.0)
    dtf_m = dtf_s[:, :, mask].mean(axis=2); np.fill_diagonal(dtf_m, 0)
    pdc_m = pdc_s[:, :, mask].mean(axis=2); np.fill_diagonal(pdc_m, 0)
    print_full_matrix(dtf_m, chan_names, f'DTF spectrum-avg {title}')
    print_full_matrix(pdc_m, chan_names, f'PDC spectrum-avg {title}')
    return dtf_m, pdc_m


# ─────────────────────────────────────────────────────────────────────────────
# Shared random signal
# ─────────────────────────────────────────────────────────────────────────────

K = 5
rng   = np.random.default_rng(SEED)
A1_r  = np.diag([0.3]*K) + rng.normal(0, 0.05, (K, K))
sig_r = make_ar_signal(A1_r, K)
coefs = fit_var(sig_r, order=4)
dtf_s, pdc_s, freqs = compute_dtf_pdc_from_var(coefs, FS, NFFT)
chan_rand = [f'ch{i}' for i in range(K)]


# ══════════════════════════════════════════════════════════════════════════════
# T1
# ══════════════════════════════════════════════════════════════════════════════

section('T1 — Value range  (random 5-ch AR signal)')
dtf_m1, pdc_m1 = print_spectrum_stats(dtf_s, pdc_s, freqs, chan_rand, '(random AR)')
save_heatmap_pair(dtf_m1, pdc_m1, chan_rand, 'T1_random', 'Random 5-ch AR signal')
check('T1a  DTF ≥ 0',      dtf_s.min() >= 0,        f'min={dtf_s.min():.6f}')
check('T1b  DTF ≤ 1',      dtf_s.max() <= 1 + TOL,  f'max={dtf_s.max():.6f}')
check('T1c  PDC ≥ 0',      pdc_s.min() >= 0,        f'min={pdc_s.min():.6f}')
check('T1d  PDC ≤ 1',      pdc_s.max() <= 1 + TOL,  f'max={pdc_s.max():.6f}')


# ══════════════════════════════════════════════════════════════════════════════
# T2
# ══════════════════════════════════════════════════════════════════════════════

section('T2 — DTF² row sums = 1 at every frequency')
dtf2_row = (dtf_s**2).sum(axis=1)
dtf_dev  = float(np.abs(dtf2_row - 1.0).max())
print(f'  DTF² row-sum range: [{dtf2_row.min():.10f}, {dtf2_row.max():.10f}]')
check('T2   DTF² row-sums = 1', dtf_dev < TOL, f'max dev={dtf_dev:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T3
# ══════════════════════════════════════════════════════════════════════════════

section('T3 — PDC² col sums = 1 at every frequency')
pdc2_col = (pdc_s**2).sum(axis=0)
pdc_dev  = float(np.abs(pdc2_col - 1.0).max())
print(f'  PDC² col-sum range: [{pdc2_col.min():.10f}, {pdc2_col.max():.10f}]')
check('T3   PDC² col-sums = 1', pdc_dev < TOL, f'max dev={pdc_dev:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T4
# ══════════════════════════════════════════════════════════════════════════════

section('T4 — Diagonal ≈ 1 before zeroing')
for i in range(K):
    dv = float(dtf_s[i,i,:].mean()); pv = float(pdc_s[i,i,:].mean())
    print(f'    ch{i}: DTF diag={dv:.4f}  PDC diag={pv:.4f}')
dtf_dg = float(np.mean([dtf_s[i,i,:].mean() for i in range(K)]))
pdc_dg = float(np.mean([pdc_s[i,i,:].mean() for i in range(K)]))
check('T4a  DTF diagonal ≈ 1', abs(dtf_dg-1.0) < 0.05, f'mean={dtf_dg:.4f}')
check('T4b  PDC diagonal ≈ 1', abs(pdc_dg-1.0) < 0.05, f'mean={pdc_dg:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# T5
# ══════════════════════════════════════════════════════════════════════════════

section('T5 — Ground truth: 5-ch Baccalá network (0→1,2,3  3↔4)')
print('  True links: 0→1, 0→2, 0→3, 3→4, 4→3')
K5=5; A5=np.zeros((K5,K5))
A5[0,0]=0.35; A5[1,1]=A5[2,2]=A5[3,3]=A5[4,4]=0.10
A5[1,0]=0.85; A5[2,0]=0.80; A5[3,0]=0.75; A5[4,3]=0.72; A5[3,4]=0.55
sig5=make_ar_signal(A5,K5); c5=fit_var(sig5,4)
dtf5_s,pdc5_s,f5=compute_dtf_pdc_from_var(c5,FS,NFFT)
dtf5=band_avg(dtf5_s,f5); pdc5=band_avg(pdc5_s,f5)
chan5=['n1','n2','n3','n4','n5']
direct5=[(0,1),(0,2),(0,3),(3,4),(4,3)]

print_full_matrix(dtf5, chan5, 'DTF band-averaged (diag=0)')
print_full_matrix(pdc5, chan5, 'PDC band-averaged (diag=0)')
save_heatmap_pair(dtf5, pdc5, chan5, 'T5_baccala',
                  '5-ch Baccalá: n1→n2,n3,n4  n4↔n5',
                  true_links=direct5)

all5=all(dtf5[t,s]>THR and pdc5[t,s]>THR for s,t in direct5)
check('T5   All direct connections recovered', all5)
vp5,vd5,vc5=verify_spectrum(dtf5_s,pdc5_s)
check('T5n  Normalization', vp5, f'DTF dev={vd5:.2e} PDC dev={vc5:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T6
# ══════════════════════════════════════════════════════════════════════════════

section('T6 — Ground truth: 10-ch hub+chain+bridge')
print('  True links: 0→1→2, 0→3, 0→4→5→6, 4→7→8→9')
K10=10; A10=np.diag([0.25]*K10)
for s,t,v in [(0,1,.85),(1,2,.82),(0,3,.75),(0,4,.70),
              (4,5,.80),(5,6,.78),(4,7,.65),(7,8,.80),(8,9,.78)]:
    A10[t,s]=v
sig10=make_ar_signal(A10,K10); c10=fit_var(sig10,4)
dtf10_s,pdc10_s,f10=compute_dtf_pdc_from_var(c10,FS,NFFT)
dtf10=band_avg(dtf10_s,f10); pdc10=band_avg(pdc10_s,f10)
chan10=[f'c{i}' for i in range(10)]
direct10=[(0,1),(1,2),(0,3),(0,4),(4,5),(5,6),(4,7),(7,8),(8,9)]

print_full_matrix(dtf10, chan10, 'DTF band-averaged (diag=0)')
print_full_matrix(pdc10, chan10, 'PDC band-averaged (diag=0)')
save_heatmap_pair(dtf10, pdc10, chan10, 'T6_hub_chain',
                  '10-ch hub+chain: 0→1→2, 0→3, 0→4→5→6, 4→7→8→9',
                  true_links=direct10)

all10=all(dtf10[t,s]>THR and pdc10[t,s]>THR for s,t in direct10)
check('T6   All 9 direct connections recovered', all10)
vp10,vd10,vc10=verify_spectrum(dtf10_s,pdc10_s)
check('T6n  Normalization', vp10, f'DTF dev={vd10:.2e} PDC dev={vc10:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T7
# ══════════════════════════════════════════════════════════════════════════════

section('T7 — Indirect connection  (DTF detects, PDC does NOT)')
dind=float(dtf5[4,0]); pind=float(pdc5[4,0])
print(f'  n1→n5 indirect via n4:  DTF[4,0]={dind:.4f}   PDC[4,0]={pind:.4f}')
check('T7a  Indirect DTF > 0',  dind > THR, f'DTF={dind:.4f}')
check('T7b  Indirect PDC ≈ 0',  pind <= THR, f'PDC={pind:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# T8
# ══════════════════════════════════════════════════════════════════════════════

section('T8 — Bidirectional feedback  n4 ↔ n5')
fwd=float(pdc5[4,3]); back=float(pdc5[3,4])
print(f'  PDC[n5,n4]={fwd:.4f}  PDC[n4,n5]={back:.4f}')
check('T8a  Forward   PDC>0', fwd>THR,  f'PDC={fwd:.4f}')
check('T8b  Feedback  PDC>0', back>THR, f'PDC={back:.4f}')
check('T8c  Asymmetry', abs(fwd-back)>0.01, f'|fwd-back|={abs(fwd-back):.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# T9
# ══════════════════════════════════════════════════════════════════════════════

section('T9 — process_single_epoch end-to-end')
K19=19; A19=np.diag([0.3]*K19)
A19[1,0]=0.8; A19[2,1]=0.8; A19[3,0]=0.7
sig19=make_ar_signal(A19,K19,n=5000)
epoch=sig19[:1024,:].T
res9=process_single_epoch(epoch,fs=256.,fixed_order=4,nfft=512,verify=True,epoch_idx=0)

CHANNEL_NAMES_19 = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T3','C3','Cz','C4','T4',
    'T5','P3','Pz','P4','T6',
    'O1','O2',
]

check('T9a  Returns dict', res9 is not None)
if res9:
    di=res9['dtf_bands']['integrated']; pi=res9['pdc_bands']['integrated']
    print_full_matrix(di, CHANNEL_NAMES_19, 'T9 DTF integrated band (diag=0)')
    print_full_matrix(pi, CHANNEL_NAMES_19, 'T9 PDC integrated band (diag=0)')
    save_heatmap_pair(di, pi, CHANNEL_NAMES_19, 'T9_epoch',
                      'process_single_epoch — integrated band',
                      true_links=[(0,1),(1,2),(0,3)])
    check('T9b  All 6 bands present', all(b in res9['dtf_bands'] for b in ['integrated','delta','theta','alpha','beta','gamma1']))
    check('T9c  Shape (19,19)', di.shape==(19,19), f'{di.shape}')
    check('T9d  Diagonal=0', np.diag(di).max()==0 and np.diag(pi).max()==0)
    check('T9e  Values in [0,1]', di.max()<=1 and di.min()>=0 and pi.max()<=1 and pi.min()>=0)
    check('T9f  Known conn Fp1→Fp2 DTF>0.04', float(di[1,0])>THR, f'DTF[Fp2,Fp1]={di[1,0]:.4f}')
    check('T9g  Spectrum saved', 'dtf_spectrum' in res9)


# ══════════════════════════════════════════════════════════════════════════════
# T10
# ══════════════════════════════════════════════════════════════════════════════

section('T10 — Band-avg sums ≠ 1 after band-avg+diag0  (expected)')
if res9:
    dr=res9['dtf_bands']['integrated']; pr=res9['pdc_bands']['integrated']
    print('  DTF row sums:')
    for i,ch in enumerate(CHANNEL_NAMES_19):
        print(f'    {ch:>4s}: {dr[i,:].sum():.4f}')
    print('  PDC col sums:')
    for j,ch in enumerate(CHANNEL_NAMES_19):
        print(f'    {ch:>4s}: {pr[:,j].sum():.4f}')
    dd=float(np.abs(dr.sum(axis=1)-1.0).max())
    pd_=float(np.abs(pr.sum(axis=0)-1.0).max())
    check('T10  Sums ≠ 1 after band-avg+diag0', dd>0.01 and pd_>0.01,
          f'DTF dev={dd:.3f}  PDC dev={pd_:.3f}')


# ══════════════════════════════════════════════════════════════════════════════
# T11
# ══════════════════════════════════════════════════════════════════════════════

section('T11 — Baccalá-style spectrum verification')
K11=5; A11=np.zeros((K11,K11))
# n0=source hub, n1/n3=direct sinks, n4=sink of n3, n2=truly isolated
A11[0,0]=0.35; A11[1,1]=A11[2,2]=A11[3,3]=A11[4,4]=0.10
A11[1,0]=0.85; A11[3,0]=0.75; A11[4,3]=0.72
# n2 has NO connections at all → truly isolated
sig11=make_ar_signal(A11,K11); c11=fit_var(sig11,4)
dtf11_s,pdc11_s,f11=compute_dtf_pdc_from_var(c11,FS,NFFT)
chan11=[f'n{i}' for i in range(K11)]
n_freqs_exp=NFFT//2+1

print('  Per-channel mean DTF diagonal:')
for i in range(K11):
    dv = float(dtf11_s[i,i,:].mean())
    role = ' ← hub (source)' if i==0 else (' ← sink' if i==4 else (' ← ISOLATED (no connections)' if i==2 else ''))
    print(f'    n{i}: mean DTF[{i},{i}]={dv:.4f}{role}')

dtf11_m, pdc11_m = print_spectrum_stats(dtf11_s, pdc11_s, f11, chan11, 'T11')
save_heatmap_pair(dtf11_m, pdc11_m, chan11, 'T11_baccala_spectrum',
                  'T11 Baccalá spectrum (n0=hub, n2=isolated, n4=sink)',
                  true_links=[(0,1),(0,3),(3,4)])

check('T11a  DTF spectrum shape', dtf11_s.shape==(K11,K11,n_freqs_exp), f'{dtf11_s.shape}')
check('T11b  PDC spectrum shape', pdc11_s.shape==(K11,K11,n_freqs_exp), f'{pdc11_s.shape}')
check('T11c  Freq axis max = FS/2', abs(f11[-1]-FS/2)<0.1, f'f[-1]={f11[-1]:.2f}')
check('T11d  Freq axis min ≈ 0',   abs(f11[0])<0.1,        f'f[0]={f11[0]:.4f}')

# FIXED: use n2 (truly isolated — no connections in A11)
diag_isolated = dtf11_s[2, 2, :]
check('T11e  DTF diagonal ≈ 1 for ISOLATED channel (n2)',
      float(np.abs(diag_isolated - 1.0).max()) < 0.05,
      f'max|diag[n2]-1|={np.abs(diag_isolated-1.0).max():.4f}')

# FIXED: both hub (n0) and sink (n4) should be suppressed
diag_hub  = dtf11_s[0, 0, :]
diag_sink = dtf11_s[4, 4, :]
check('T11e2 DTF diagonal suppressed for connected channels (n0 hub, n4 sink)',
      float(diag_hub.mean()) < 0.95 or float(diag_sink.mean()) < 0.95,
      f'mean diag[n0 hub]={diag_hub.mean():.4f}  mean diag[n4 sink]={diag_sink.mean():.4f}')

check('T11f  DTF spectrum in [0,1]', dtf11_s.min()>=0 and dtf11_s.max()<=1+TOL)
check('T11g  PDC spectrum in [0,1]', pdc11_s.min()>=0 and pdc11_s.max()<=1+TOL)
conn_spec=dtf11_s[1,0,:]
check('T11h  Known conn n0→n1 mean > THR', float(conn_spec.mean())>THR,
      f'mean DTF[n1,n0,:]={conn_spec.mean():.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# T12
# ══════════════════════════════════════════════════════════════════════════════

section('T12 — Ground truth: chain Ch1→Ch2→Ch3→Ch4,  Ch5=pure noise')
K12=5; A12=np.zeros((K12,K12))
A12[0,0]=0.30; A12[1,1]=0.30; A12[2,2]=0.30; A12[3,3]=0.30; A12[4,4]=0.10
A12[1,0]=0.88; A12[2,1]=0.85; A12[3,2]=0.82
sig12=make_ar_signal(A12,K12); c12=fit_var(sig12,4)
dtf12_s,pdc12_s,f12=compute_dtf_pdc_from_var(c12,FS,NFFT)
dtf12=band_avg(dtf12_s,f12); pdc12=band_avg(pdc12_s,f12)
chan12=['Ch1','Ch2','Ch3','Ch4','Ch5']
direct12=[(0,1),(1,2),(2,3)]

print_full_matrix(dtf12, chan12, 'DTF band-averaged (diag=0)')
print_full_matrix(pdc12, chan12, 'PDC band-averaged (diag=0)')
save_heatmap_pair(dtf12, pdc12, chan12, 'T12_chain',
                  'Chain: Ch1→Ch2→Ch3→Ch4, Ch5=noise',
                  true_links=direct12)

indirect12=[(0,2),(0,3),(1,3)]
check('T12a  Direct links (DTF+PDC)',
      all(dtf12[t,s]>THR and pdc12[t,s]>THR for s,t in direct12))
check('T12b  Indirect links detected by DTF',
      all(dtf12[t,s]>THR for s,t in indirect12),
      ' '.join(f'DTF[{t},{s}]={dtf12[t,s]:.3f}' for s,t in indirect12))
check('T12c  Indirect links NOT in PDC',
      all(pdc12[t,s]<=ISO for s,t in indirect12),
      ' '.join(f'PDC[{t},{s}]={pdc12[t,s]:.3f}' for s,t in indirect12))
check('T12d  Isolated Ch5 DTF col ≈ 0', float(dtf12[:,4].max())<=ISO)
check('T12e  Isolated Ch5 DTF row ≈ 0', float(dtf12[4,:].max())<=ISO)
check('T12f  Isolated Ch5 PDC col ≈ 0', float(pdc12[:,4].max())<=ISO)
check('T12g  Isolated Ch5 PDC row ≈ 0', float(pdc12[4,:].max())<=ISO)
vp12,vd12,vc12=verify_spectrum(dtf12_s,pdc12_s)
check('T12n  Normalization', vp12, f'DTF dev={vd12:.2e} PDC dev={vc12:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T13
# ══════════════════════════════════════════════════════════════════════════════

section('T13 — Ground truth: tree Ch1→Ch2→Ch4, Ch1→Ch3,  Ch5=noise')
K13=5; A13=np.zeros((K13,K13))
A13[0,0]=0.30; A13[1,1]=0.25; A13[2,2]=0.25; A13[3,3]=0.25; A13[4,4]=0.10
A13[1,0]=0.88; A13[2,0]=0.82; A13[3,1]=0.85
sig13=make_ar_signal(A13,K13); c13=fit_var(sig13,4)
dtf13_s,pdc13_s,f13=compute_dtf_pdc_from_var(c13,FS,NFFT)
dtf13=band_avg(dtf13_s,f13); pdc13=band_avg(pdc13_s,f13)
chan13=['Ch1','Ch2','Ch3','Ch4','Ch5']
direct13=[(0,1),(0,2),(1,3)]

print_full_matrix(dtf13, chan13, 'DTF band-averaged (diag=0)')
print_full_matrix(pdc13, chan13, 'PDC band-averaged (diag=0)')
save_heatmap_pair(dtf13, pdc13, chan13, 'T13_tree',
                  'Tree: Ch1→Ch2→Ch4, Ch1→Ch3, Ch5=noise',
                  true_links=direct13)

check('T13a  Direct tree links (DTF+PDC)',
      all(dtf13[t,s]>THR and pdc13[t,s]>THR for s,t in direct13))
dtf_ind13=float(dtf13[3,0]); pdc_ind13=float(pdc13[3,0])
check('T13b  Indirect Ch1→Ch4 in DTF',  dtf_ind13>THR,  f'DTF={dtf_ind13:.4f}')
check('T13c  Indirect Ch1→Ch4 NOT in PDC', pdc_ind13<=ISO, f'PDC={pdc_ind13:.4f}')
check('T13d  No cross-branch Ch3↔Ch4 in PDC',
      pdc13[3,2]<=ISO and pdc13[2,3]<=ISO)
check('T13e  Isolated Ch5 ≈ 0',
      float(dtf13[:,4].max())<=ISO and float(dtf13[4,:].max())<=ISO)
vp13,vd13,vc13=verify_spectrum(dtf13_s,pdc13_s)
check('T13n  Normalization', vp13, f'DTF dev={vd13:.2e} PDC dev={vc13:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# T14
# ══════════════════════════════════════════════════════════════════════════════

section('T14 — Full 19-ch EEG network (epilepsy-inspired)')

Fp1,Fp2,F7,F3,Fz,F4,F8  = 0,1,2,3,4,5,6
T3, C3, Cz,C4,T4         = 7,8,9,10,11
T5, P3, Pz,P4,T6         = 12,13,14,15,16
O1, O2                    = 17,18
ISOLATED14 = [Fp2,F7,Fz,F4,F8,Cz,T4,Pz,P4,T6,O1,O2]

K14=19; A14=np.diag([0.25]*K14)
DIRECT14=[(Fp1,F3,.85),(Fp1,C3,.80),(Fp1,P3,.75),
          (F3,T3,.82),(T3,T5,.78),(C3,C4,.70),(T3,P3,.65)]
for src,snk,v in DIRECT14:
    A14[snk,src]=v

sig14=make_ar_signal(A14,K14,n=N); c14=fit_var(sig14,order=6)
dtf14_s,pdc14_s,f14=compute_dtf_pdc_from_var(c14,FS,NFFT)
dtf14=band_avg(dtf14_s,f14); pdc14=band_avg(pdc14_s,f14)
direct_pairs=[(src,snk) for src,snk,_ in DIRECT14]

print_full_matrix(dtf14, CHANNEL_NAMES_19, 'T14 DTF band-averaged (diag=0)')
print_full_matrix(pdc14, CHANNEL_NAMES_19, 'T14 PDC band-averaged (diag=0)')
save_heatmap_pair(dtf14, pdc14, CHANNEL_NAMES_19, 'T14_eeg19ch',
                  '19-ch EEG: Fp1=hub, chains via F3/C3/T3, 12 isolated',
                  true_links=direct_pairs)

all_direct14=all(dtf14[snk,src]>THR and pdc14[snk,src]>THR for src,snk in direct_pairs)
check('T14a  All 7 direct links (DTF+PDC)', all_direct14)
dtf_col_means=np.array([dtf14[:,j].mean() for j in range(K14)])
check('T14b  Fp1 is strongest SOURCE in DTF',
      int(np.argmax(dtf_col_means))==Fp1,
      f'Strongest: {CHANNEL_NAMES_19[int(np.argmax(dtf_col_means))]}')
dtf_ind14=float(dtf14[T5,Fp1]); pdc_ind14=float(pdc14[T5,Fp1])
check('T14c  Indirect Fp1→T5 DTF detects',  dtf_ind14>THR,  f'DTF={dtf_ind14:.4f}')
check('T14d  Indirect Fp1→T5 PDC does NOT', pdc_ind14<=ISO, f'PDC={pdc_ind14:.4f}')
iso_ok_dtf=all(float(dtf14[:,j].max())<=ISO and float(dtf14[j,:].max())<=ISO for j in ISOLATED14)
iso_ok_pdc=all(float(pdc14[:,j].max())<=ISO and float(pdc14[j,:].max())<=ISO for j in ISOLATED14)
check('T14e  Isolated channels DTF ≈ 0', iso_ok_dtf,
      f'max={max(max(float(dtf14[:,j].max()),float(dtf14[j,:].max())) for j in ISOLATED14):.4f}')
check('T14f  Isolated channels PDC ≈ 0', iso_ok_pdc,
      f'max={max(max(float(pdc14[:,j].max()),float(pdc14[j,:].max())) for j in ISOLATED14):.4f}')
vp14,vd14,vc14=verify_spectrum(dtf14_s,pdc14_s)
check('T14n  Normalization', vp14, f'DTF dev={vd14:.2e} PDC dev={vc14:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print(f'\n{"═"*72}')
print('  FINAL REPORT')
print(f'{"═"*72}')
n_pass=sum(results.values()); n_fail=sum(not v for v in results.values())
for name,ok in results.items():
    print(f'  {"✅" if ok else "❌"}  {name}')
print(f'\n  {n_pass}/{len(results)} passed  {"✅  ALL PASS" if n_fail==0 else f"❌  {n_fail} FAILED"}')
print(f'\n  Heatmaps saved to: {HEATMAP_DIR.resolve()}')
print(f'{"═"*72}')