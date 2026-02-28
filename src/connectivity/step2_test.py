"""
Step 2 — Test Suite
====================
Comprehensive correctness tests for compute_dtf_pdc_from_var.
All tests use YOUR exact functions imported from step2_compute_connectivity_updated.py.

Tests
-----
T1  Value range          DTF and PDC values ∈ [0, 1]
T2  DTF normalization    DTF²[i,:,f].sum(axis=1) = 1  at every freq  (before band avg)
T3  PDC normalization    PDC²[:,j,f].sum(axis=0) = 1  at every freq  (before band avg)
T4  Diagonal self-conn   DTF[i,i,f] ≈ 1 and PDC[i,i,f] ≈ 1 before zeroing
T5  Ground truth A       5-ch Baccalá diagram (1→2,3,4  and  4↔5 feedback)
T6  Ground truth B       10-ch hub+chain+bridge
T7  Indirect detection   DTF detects indirect; PDC does NOT
T8  Feedback             bidirectional links recovered in both directions
T9  process_single_epoch  end-to-end shape and label checks
T10 Band avg sums        sums are NOT 1 after averaging (documenting expected behaviour)

Usage:
    python step2_tests.py
"""

import sys
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR

sys.path.insert(0, '.')
from step2_compute_connectivity_updated import (
    compute_dtf_pdc_from_var,
    verify_spectrum,
    process_single_epoch,
)

FS   = 256
NFFT = 512
SEED = 42
N    = 40_000   # samples for synthetic signal
TOL  = 1e-6
THR  = 0.04     # minimum value to call a connection "detected"

PASS = '✅ PASS'
FAIL = '❌ FAIL'

results = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_ar_signal(A1, K, n=N, noise=0.25, seed=SEED):
    """AR(1) signal.  A1[i,j] = lag-1 effect of source j on sink i."""
    rng = np.random.default_rng(seed)
    x   = np.zeros((n, K))
    eps = rng.normal(0, noise, (n, K))
    for t in range(1, n):
        x[t] = A1 @ x[t - 1] + eps[t]
    return x


def fit_var(signal, order=4):
    std     = np.std(signal)
    results = VAR(signal / std).fit(maxlags=order, trend='c', verbose=False)
    return results.coefs


def band_avg(spectrum, freqs, flo=0.5, fhi=45.0):
    """Average spectrum over [flo, fhi] Hz, zero diagonal."""
    m = spectrum[:, :, (freqs >= flo) & (freqs <= fhi)].mean(axis=2)
    np.fill_diagonal(m, 0.0)
    return m


def section(title):
    print(f'\n{"═" * 65}')
    print(f'  {title}')
    print(f'{"═" * 65}')


def check(name, cond, detail=''):
    tag = PASS if cond else FAIL
    results[name] = cond
    print(f'  {tag}  {name}')
    if detail:
        print(f'       {detail}')
    return cond


# ─────────────────────────────────────────────────────────────────────────────
# Build one shared set of spectra (random AR signal)
# ─────────────────────────────────────────────────────────────────────────────

K   = 5
rng = np.random.default_rng(SEED)
A1_rand = np.diag([0.3] * K) + rng.normal(0, 0.05, (K, K))
sig_rand = make_ar_signal(A1_rand, K)
coefs    = fit_var(sig_rand, order=4)
dtf_s, pdc_s, freqs = compute_dtf_pdc_from_var(coefs, FS, NFFT)
N_FREQS = dtf_s.shape[2]


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Value range
# ─────────────────────────────────────────────────────────────────────────────
section('T1 — Value range  (both metrics must be in [0, 1])')

dtf_min, dtf_max = float(dtf_s.min()), float(dtf_s.max())
pdc_min, pdc_max = float(pdc_s.min()), float(pdc_s.max())
print(f'  DTF range: [{dtf_min:.6f}, {dtf_max:.6f}]')
print(f'  PDC range: [{pdc_min:.6f}, {pdc_max:.6f}]')

check('T1a  DTF ≥ 0',      dtf_min >= 0,      f'min={dtf_min:.6f}')
check('T1b  DTF ≤ 1',      dtf_max <= 1 + TOL, f'max={dtf_max:.6f}')
check('T1c  PDC ≥ 0',      pdc_min >= 0,      f'min={pdc_min:.6f}')
check('T1d  PDC ≤ 1',      pdc_max <= 1 + TOL, f'max={pdc_max:.6f}')


# ─────────────────────────────────────────────────────────────────────────────
# T2 — DTF normalization  (BEFORE band averaging, BEFORE diag zero)
# ─────────────────────────────────────────────────────────────────────────────
section('T2 — DTF² row sums = 1  at every frequency  (before band avg)')
print('  dtf²[i, :, f].sum(axis=1) should equal 1.0 for all i and f')

dtf2_row = (dtf_s ** 2).sum(axis=1)   # (K, n_freqs)
dtf_dev  = float(np.abs(dtf2_row - 1.0).max())

print(f'  DTF² row-sum max deviation from 1.0: {dtf_dev:.2e}')
for i in range(K):
    row_min = float(dtf2_row[i].min())
    row_max = float(dtf2_row[i].max())
    ok = abs(row_min - 1.0) < TOL and abs(row_max - 1.0) < TOL
    print(f'    sink {i}: row_sum ∈ [{row_min:.8f}, {row_max:.8f}]  '
          f'{"✅" if ok else "❌"}')

check('T2   DTF² row-sums = 1 (all sinks, all freqs)',
      dtf_dev < TOL, f'max deviation = {dtf_dev:.2e}  (tolerance = {TOL})')


# ─────────────────────────────────────────────────────────────────────────────
# T3 — PDC normalization
# ─────────────────────────────────────────────────────────────────────────────
section('T3 — PDC² col sums = 1  at every frequency  (before band avg)')
print('  pdc²[:, j, f].sum(axis=0) should equal 1.0 for all j and f')

pdc2_col = (pdc_s ** 2).sum(axis=0)   # (K, n_freqs)
pdc_dev  = float(np.abs(pdc2_col - 1.0).max())

print(f'  PDC² col-sum max deviation from 1.0: {pdc_dev:.2e}')
for j in range(K):
    col_min = float(pdc2_col[j].min())
    col_max = float(pdc2_col[j].max())
    ok = abs(col_min - 1.0) < TOL and abs(col_max - 1.0) < TOL
    print(f'    source {j}: col_sum ∈ [{col_min:.8f}, {col_max:.8f}]  '
          f'{"✅" if ok else "❌"}')

check('T3   PDC² col-sums = 1 (all sources, all freqs)',
      pdc_dev < TOL, f'max deviation = {pdc_dev:.2e}  (tolerance = {TOL})')


# ─────────────────────────────────────────────────────────────────────────────
# T4 — Self-connectivity (diagonal ≈ 1) before zeroing
# ─────────────────────────────────────────────────────────────────────────────
section('T4 — Diagonal ≈ 1 at spectrum level  (before diag zeroing)')
print('  DTF[i,i,f] = PDC[i,i,f] = 1 by construction')
print('  This is why we zero AFTER averaging, not before')

dtf_diag_mean = float(np.mean([dtf_s[i, i, :].mean() for i in range(K)]))
pdc_diag_mean = float(np.mean([pdc_s[i, i, :].mean() for i in range(K)]))
print(f'  Mean DTF diagonal value (avg over freq): {dtf_diag_mean:.4f}  (expect ≈1)')
print(f'  Mean PDC diagonal value (avg over freq): {pdc_diag_mean:.4f}  (expect ≈1)')

check('T4a  DTF diagonal ≈ 1', abs(dtf_diag_mean - 1.0) < 0.05,
      f'mean = {dtf_diag_mean:.4f}')
check('T4b  PDC diagonal ≈ 1', abs(pdc_diag_mean - 1.0) < 0.05,
      f'mean = {pdc_diag_mean:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# T5 — Ground truth: 5-ch Baccalá diagram
# ─────────────────────────────────────────────────────────────────────────────
section('T5 — Ground truth: 5-ch Baccalá diagram')
print('  Network:  node1→node2, node1→node3, node1→node4, node4↔node5')
print('  Indices:  0→1, 0→2, 0→3, 3→4, 4→3  (bidirectional feedback)')

K5 = 5
A5 = np.zeros((K5, K5))
A5[0,0]=0.35; A5[1,1]=A5[2,2]=A5[3,3]=A5[4,4]=0.10
A5[1,0]=0.85; A5[2,0]=0.80; A5[3,0]=0.75
A5[4,3]=0.72; A5[3,4]=0.55

eig5 = float(np.abs(np.linalg.eigvals(A5)).max())
print(f'  Spectral radius = {eig5:.4f}  (< 1 → stable ✅)')

sig5   = make_ar_signal(A5, K5)
coefs5 = fit_var(sig5, order=4)
dtf5_s, pdc5_s, f5 = compute_dtf_pdc_from_var(coefs5, FS, NFFT)
dtf5   = band_avg(dtf5_s, f5)
pdc5   = band_avg(pdc5_s, f5)

CHAN5 = ['node1','node2','node3','node4','node5']
direct5 = [(0,1),(0,2),(0,3),(3,4),(4,3)]
print(f'\n  Connection detection (threshold={THR}):')
print(f'  {"Connection":20s} {"DTF":>8s} {"PDC":>8s}  DTF? PDC?  Expected')
print('  ' + '─' * 60)
all5 = True
for s,t in direct5:
    dv = float(dtf5[t,s]); pv = float(pdc5[t,s])
    dok = dv > THR; pok = pv > THR
    ok = dok and pok
    if not ok: all5 = False
    print(f'  {CHAN5[s]}→{CHAN5[t]:15s} {dv:8.4f} {pv:8.4f}  '
          f'{"✅" if dok else "❌"}    {"✅" if pok else "❌"}    direct')

check('T5   All direct connections recovered', all5)

# verify_spectrum pass
vp5, vd5, vp5c = verify_spectrum(dtf5_s, pdc5_s)
check('T5n  Normalization (spectrum level)', vp5,
      f'DTF dev={vd5:.2e}  PDC dev={vp5c:.2e}')


# ─────────────────────────────────────────────────────────────────────────────
# T6 — Ground truth: 10-ch hub+chain
# ─────────────────────────────────────────────────────────────────────────────
section('T6 — Ground truth: 10-ch hub + chain + bridges')
print('  ch0→ch1→ch2, ch0→ch3, ch0→ch4→ch5→ch6, ch4→ch7→ch8→ch9')

K10 = 10
A10 = np.diag([0.25]*K10)
for s,t,v in [(0,1,0.85),(1,2,0.82),(0,3,0.75),(0,4,0.70),
              (4,5,0.80),(5,6,0.78),(4,7,0.65),(7,8,0.80),(8,9,0.78)]:
    A10[t,s] = v

sig10   = make_ar_signal(A10, K10)
coefs10 = fit_var(sig10, order=4)
dtf10_s, pdc10_s, f10 = compute_dtf_pdc_from_var(coefs10, FS, NFFT)
dtf10   = band_avg(dtf10_s, f10)
pdc10   = band_avg(pdc10_s, f10)

direct10 = [(0,1),(1,2),(0,3),(0,4),(4,5),(5,6),(4,7),(7,8),(8,9)]
all10 = True
print(f'  Connection detection (threshold={THR}):')
print(f'  {"Connection":12s} {"DTF":>8s} {"PDC":>8s}  DTF? PDC?')
print('  ' + '─' * 48)
for s,t in direct10:
    dv = float(dtf10[t,s]); pv = float(pdc10[t,s])
    dok = dv > THR; pok = pv > THR
    ok  = dok and pok
    if not ok: all10 = False
    print(f'  ch{s}→ch{t}{"":6s}   {dv:8.4f} {pv:8.4f}  '
          f'{"✅" if dok else "❌"}    {"✅" if pok else "❌"}')

check('T6   All 9 direct connections recovered', all10)

vp10, vd10, vp10c = verify_spectrum(dtf10_s, pdc10_s)
check('T6n  Normalization (spectrum level)', vp10,
      f'DTF dev={vd10:.2e}  PDC dev={vp10c:.2e}')


# ─────────────────────────────────────────────────────────────────────────────
# T7 — Indirect detection  (DTF yes, PDC no)
# ─────────────────────────────────────────────────────────────────────────────
section('T7 — Indirect connection  (DTF detects it, PDC does not)')
print('  node1→node5 is indirect (via node4).  DTF>0, PDC≈0 is correct.')

dtf_indir = float(dtf5[4, 0])
pdc_indir = float(pdc5[4, 0])
print(f'  node1→node5  DTF={dtf_indir:.4f}  PDC={pdc_indir:.4f}')

check('T7a  Indirect: DTF detects propagation', dtf_indir > THR,
      f'DTF={dtf_indir:.4f} (should be > {THR})')
check('T7b  Indirect: PDC ≈ 0 (no direct link)', pdc_indir <= THR,
      f'PDC={pdc_indir:.4f} (should be ≤ {THR})')


# ─────────────────────────────────────────────────────────────────────────────
# T8 — Bidirectional feedback
# ─────────────────────────────────────────────────────────────────────────────
section('T8 — Bidirectional feedback  node4 ↔ node5')
print('  Both PDC[4,3] (forward) and PDC[3,4] (feedback) must be > 0')

fwd  = float(pdc5[4, 3])
back = float(pdc5[3, 4])
print(f'  Forward   node4→node5  PDC={fwd:.4f}')
print(f'  Feedback  node5→node4  PDC={back:.4f}')

check('T8a  Forward   PDC[node5, node4] > 0', fwd  > THR, f'PDC={fwd:.4f}')
check('T8b  Feedback  PDC[node4, node5] > 0', back > THR, f'PDC={back:.4f}')
check('T8c  Asymmetry (forward ≠ feedback)',  abs(fwd - back) > 0.01,
      f'|fwd-back|={abs(fwd-back):.4f} (different coefficients → different values)')


# ─────────────────────────────────────────────────────────────────────────────
# T9 — process_single_epoch end-to-end
# ─────────────────────────────────────────────────────────────────────────────
section('T9 — process_single_epoch  end-to-end shape and content checks')

# Build a clean 19-channel epoch (shape expected by pipeline)
K19 = 19
A19 = np.diag([0.3]*K19)
A19[1,0]=0.8; A19[2,1]=0.8; A19[3,0]=0.7   # known connections
sig19  = make_ar_signal(A19, K19, n=5000)
epoch  = sig19[:1024, :].T                   # (19, 1024) — as step0 saves it

result = process_single_epoch(epoch, fs=256.0, fixed_order=4, nfft=512,
                               verify=True, epoch_idx=0)

check('T9a  Returns dict (not None)', result is not None)
if result is not None:
    dtf_int = result['dtf_bands']['integrated']
    pdc_int = result['pdc_bands']['integrated']
    all_bands = ['integrated','delta','theta','alpha','beta','gamma1']
    check('T9b  dtf_bands has all 6 bands',
          all(b in result['dtf_bands'] for b in all_bands))
    check('T9c  pdc_bands has all 6 bands',
          all(b in result['pdc_bands'] for b in all_bands))
    check('T9d  Shape = (19, 19)',
          dtf_int.shape == (19, 19) and pdc_int.shape == (19, 19),
          f'dtf shape={dtf_int.shape}  pdc shape={pdc_int.shape}')
    check('T9e  Diagonal = 0 (after band avg)',
          float(np.diag(dtf_int).max()) == 0.0 and float(np.diag(pdc_int).max()) == 0.0,
          f'DTF diag max={np.diag(dtf_int).max():.4f}  PDC diag max={np.diag(pdc_int).max():.4f}')
    check('T9f  Values in [0, 1]',
          float(dtf_int.max()) <= 1.0 and float(pdc_int.max()) <= 1.0 and
          float(dtf_int.min()) >= 0.0 and float(pdc_int.min()) >= 0.0,
          f'DTF [{dtf_int.min():.4f},{dtf_int.max():.4f}]  '
          f'PDC [{pdc_int.min():.4f},{pdc_int.max():.4f}]')
    # Known connection ch0→ch1 should be detectable
    check('T9g  Known conn ch0→ch1 detected (DTF>0.04)',
          float(dtf_int[1,0]) > THR, f'DTF[1,0]={dtf_int[1,0]:.4f}')
    check('T9h  process_single_epoch returns spectrum too',
          'dtf_spectrum' in result and 'pdc_spectrum' in result)


# ─────────────────────────────────────────────────────────────────────────────
# T10 — Band-averaged sums NOT equal to 1 (documenting expected behaviour)
# ─────────────────────────────────────────────────────────────────────────────
section('T10 — After band averaging + diag zero: sums are NOT 1  (expected)')
print('  This is correct behaviour — normalization property only holds at')
print('  individual frequency bins, not after averaging across a band.')

if result is not None:
    dtf_int = result['dtf_bands']['integrated']
    pdc_int = result['pdc_bands']['integrated']
    dtf_row = dtf_int.sum(axis=1)   # (19,)  — after diag=0
    pdc_col = pdc_int.sum(axis=0)   # (19,)
    print(f'  DTF row-sums  (post band-avg+diag0): '
          f'min={dtf_row.min():.3f}  max={dtf_row.max():.3f}  '
          f'(NOT 1.0 — correct ✓)')
    print(f'  PDC col-sums  (post band-avg+diag0): '
          f'min={pdc_col.min():.3f}  max={pdc_col.max():.3f}  '
          f'(NOT 1.0 — correct ✓)')

    # After diag=0 and band-averaging, row/col sums can be > 1 OR < 1.
    # We just confirm they are NOT pinned exactly to 1 (which would mean
    # we are incorrectly re-normalizing after averaging).
    dtf_deviation = float(np.abs(dtf_row - 1.0).max())
    pdc_deviation = float(np.abs(pdc_col - 1.0).max())
    check('T10  Sums ≠ 1 after band-avg+diag0  (documenting expected behaviour)',
          dtf_deviation > 0.01 and pdc_deviation > 0.01,
          f'DTF row-sum max dev from 1.0 = {dtf_deviation:.3f}  '
          f'PDC col-sum max dev = {pdc_deviation:.3f}  '
          '(> 0 confirms no accidental re-normalisation)')


# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"═" * 65}')
print('  FINAL REPORT')
print(f'{"═" * 65}')

n_pass  = sum(v for v in results.values())
n_fail  = sum(not v for v in results.values())
n_total = len(results)

for name, ok in results.items():
    print(f'  {"✅" if ok else "❌"}  {name}')

print(f'\n  {n_pass}/{n_total} passed  {"✅  ALL PASS" if n_fail == 0 else f"❌  {n_fail} FAILED"}')

print(f'\n{"═" * 65}')
print('  INTERPRETATION GUIDE')
print(f'{"═" * 65}')
print('  matrix[i, j]  =  source j  →  sink i')
print('  DTF: row-normalised via H(f)  →  bright COLUMNS = strong sources')
print('  PDC: col-normalised via A(f)  →  bright ROWS    = strong sinks')
print('  Both are CORRECT behaviour, not a bug.')
print()
print('  Normalization:')
print('    DTF² row sums = 1  at each freq bin  (before band avg)')
print('    PDC² col sums = 1  at each freq bin  (before band avg)')
print('    After band averaging → sums ≠ 1  (expected)')
print()
print('  Diagonal:')
print('    DTF[i,i,f] ≈ 1  at spectrum level (math artefact)')
print('    Zeroed AFTER band averaging (not before)')
print(f'{"═" * 65}')