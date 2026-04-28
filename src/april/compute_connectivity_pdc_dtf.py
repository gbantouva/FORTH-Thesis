"""
Effective Connectivity: PDC and DTF from MVAR models
=====================================================

Computes Partial Directed Coherence (PDC) and Directed Transfer Function (DTF)
for all subjects, in pre-ictal and ictal windows.

Mathematical background:
    Given a K-channel EEG signal x(t), the MVAR(p) model is:
        x(t) = sum_{k=1}^{p} A(k) x(t-k) + e(t)
    where A(k) are [K x K] coefficient matrices and e(t) is white noise
    with covariance C = cov(e).

    Frequency-domain representation:
        A(f) = I - sum_{k=1}^{p} A(k) exp(-j2πfk/fs)
        H(f) = A(f)^{-1}   (transfer function)

    PDC  (Baccalá & Sameshima, Biol. Cybern. 2001):
        PDC_{ij}(f) = A_{ij}(f) / sqrt( A_{:,j}^H(f) A_{:,j}(f) )
        Measures DIRECT causal influence from j → i at frequency f.
        Column-normalized: for each source j, sum_i |PDC_{ij}(f)|^2 = 1.

    DTF  (Kamiński & Blinowska, Biol. Cybern. 1991):
        DTF_{ij}(f) = H_{ij}(f) / sqrt( H_{i,:}(f) H_{i,:}^H(f) )
        Measures TOTAL (direct + indirect) causal influence from j → i.
        Row-normalized: for each sink i, sum_j |DTF_{ij}(f)|^2 = 1.

    Convention: matrix[i, j] = influence FROM channel j TO channel i.

MVAR model order selection:
    We use BIC (Bayesian Information Criterion) to select the optimal order p.
    Typical values for EEG at 256 Hz are p = 5–15.

Dependencies:
    pip install numpy scipy matplotlib statsmodels

References:
    [1] Baccalá & Sameshima (2001). Partial directed coherence: a new concept
        in neural structure determination. Biol. Cybernetics, 84(6), 463–474.
    [2] Kamiński & Blinowska (1991). A new method of the description of the
        information flow in the brain structures. Biol. Cybernetics, 65, 203–210.
    [3] Billinger et al. (2014). SCoT: a Python toolbox for EEG source
        connectivity. Frontiers in Neuroinformatics, 8, 22.
    [4] Astolfi et al. (2007). Comparison of different cortical connectivity
        estimators for high-resolution EEG recordings. Human Brain Mapping, 28, 143–157.

Author: [your name]
"""

import numpy as np
from scipy import linalg
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# ═══════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def fit_mvar(x, order):
    """
    Fit a multivariate autoregressive (MVAR) model of given order.

    Parameters
    ----------
    x : ndarray, shape (T, K)
        Multichannel time series. T = samples, K = channels.
    order : int
        Model order p.

    Returns
    -------
    A : ndarray, shape (K, K*p)
        Coefficient matrices [A(1) A(2) ... A(p)], each [K x K].
    C : ndarray, shape (K, K)
        Noise covariance matrix.
    """
    T, K = x.shape
    p = order

    # Build regression matrices: Y = X @ B + E
    # Y[t, :] = x[t, :]    for t = p, ..., T-1
    # X[t, :] = [x[t-1, :], x[t-2, :], ..., x[t-p, :]]
    Y = x[p:]                                         # (T-p, K)
    X = np.column_stack([x[p - k - 1:T - k - 1] for k in range(p)])  # (T-p, K*p)

    # OLS: B = (X'X)^{-1} X'Y, each column of B gives coefficients for one channel
    B, res, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # B: (K*p, K)

    # Rearrange: A(k) matrices
    # B rows are grouped as [A(1)^T ; A(2)^T ; ... ; A(p)^T]
    # We want A as (K, K*p) = [A(1) A(2) ... A(p)]
    A = B.T  # (K, K*p)

    # Residuals and noise covariance
    E = Y - X @ B                                     # (T-p, K)
    C = np.cov(E, rowvar=False, bias=False)            # (K, K)

    return A, C


def select_order_bic(x, max_order=20):
    """
    Select MVAR model order using BIC (Schwarz criterion).

    Parameters
    ----------
    x : ndarray, shape (T, K)
        Multichannel time series.
    max_order : int
        Maximum order to test.

    Returns
    -------
    best_order : int
        Order with lowest BIC.
    bic_values : ndarray
        BIC for each order from 1 to max_order.
    """
    T, K = x.shape
    bic_values = np.full(max_order, np.inf)

    for p in range(1, max_order + 1):
        n = T - p  # effective number of samples
        if n < K * p + 1:
            break

        A, C = fit_mvar(x, p)
        # BIC = n * log(det(C)) + K^2 * p * log(n)
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0:
            continue
        num_params = K * K * p
        bic_values[p - 1] = n * logdet + num_params * np.log(n)

    best_order = np.argmin(bic_values) + 1
    return best_order, bic_values


def compute_pdc_dtf(A, C, nfft=256, fs=256):
    """
    Compute PDC and DTF from MVAR coefficients.

    Parameters
    ----------
    A : ndarray, shape (K, K*p)
        MVAR coefficient matrices [A(1) A(2) ... A(p)].
    C : ndarray, shape (K, K)
        Noise covariance.
    nfft : int
        Number of frequency bins (0 to fs/2).
    fs : float
        Sampling frequency.

    Returns
    -------
    pdc : ndarray, shape (K, K, nfft)
        Partial directed coherence. pdc[i, j, f] = influence j → i.
    dtf : ndarray, shape (K, K, nfft)
        Directed transfer function. dtf[i, j, f] = influence j → i.
    freqs : ndarray, shape (nfft,)
        Frequency vector in Hz.
    """
    K = C.shape[0]
    p = A.shape[1] // K

    freqs = np.linspace(0, fs / 2, nfft)

    pdc = np.zeros((K, K, nfft))
    dtf = np.zeros((K, K, nfft))

    for fi, f in enumerate(freqs):
        # A(f) = I - sum_{k=1}^{p} A(k) * exp(-j*2*pi*f*k/fs)
        Af = np.eye(K, dtype=complex)
        for k in range(p):
            Ak = A[:, k * K:(k + 1) * K]
            Af -= Ak * np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)

        # H(f) = A(f)^{-1}
        Hf = linalg.inv(Af)

        # --- PDC: column-normalized ---
        # PDC_{ij}(f) = A_{ij}(f) / sqrt( sum_i |A_{ij}(f)|^2 )
        for j in range(K):
            col_norm = np.sqrt(np.real(np.vdot(Af[:, j], Af[:, j])))
            if col_norm > 0:
                pdc[:, j, fi] = np.abs(Af[:, j]) / col_norm

        # --- DTF: row-normalized ---
        # DTF_{ij}(f) = H_{ij}(f) / sqrt( sum_j |H_{ij}(f)|^2 )
        for i in range(K):
            row_norm = np.sqrt(np.real(np.vdot(Hf[i, :], Hf[i, :])))
            if row_norm > 0:
                dtf[i, :, fi] = np.abs(Hf[i, :]) / row_norm

    return pdc, dtf, freqs


def band_average_matrix(conn, freqs, bands):
    """
    Average a connectivity spectrum over frequency bands.

    Parameters
    ----------
    conn : ndarray, shape (K, K, nfft)
        Connectivity measure (PDC or DTF).
    freqs : ndarray, shape (nfft,)
        Frequency vector.
    bands : dict
        Band name → (fmin, fmax) in Hz.

    Returns
    -------
    matrices : dict
        Band name → ndarray (K, K), band-averaged connectivity.
    """
    matrices = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if mask.any():
            matrices[name] = np.mean(conn[:, :, mask], axis=2)
        else:
            matrices[name] = np.zeros(conn.shape[:2])
    return matrices


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_connectivity_matrix(matrix, chnames, title, ax=None, vmin=0, vmax=None):
    """Plot a single connectivity matrix as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    K = matrix.shape[0]

    if vmax is None:
        vmax = np.percentile(matrix, 95)

    # Zero the diagonal for clearer visualization
    mat_plot = matrix.copy()
    np.fill_diagonal(mat_plot, 0)

    im = ax.imshow(mat_plot, cmap='hot_r', vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest')
    n = min(K, len(chnames))
    ax.set_xticks(range(n))
    ax.set_xticklabels(chnames[:n], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(chnames[:n], fontsize=8)
    ax.set_ylabel('Sink (i)', fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_all_bands(matrices, chnames, suptitle, out_path=None):
    """Plot connectivity matrices for all frequency bands in a single figure."""
    bands = list(matrices.keys())
    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
    if n == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        plot_connectivity_matrix(matrices[band], chnames, band, ax=ax)

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  MAIN: run on all subjects
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── CONFIGURE ──────────────────────────────────────────────────
    mat_file = r"F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat"
    out_dir  = r"F:\FORTH_Final_Thesis\FORTH-Thesis\figures\april\connectivity"
    # ───────────────────────────────────────────────────────────────

    fs = 256
    NFFT = 256  # frequency resolution: fs/NFFT = 1 Hz
    MAX_ORDER = 20  # max MVAR order for BIC search
    WINDOW_SEC = 5  # window length in seconds (matches X_tw in your data)

    # Standard EEG frequency bands
    BANDS = {
        'Delta (0.5-4)': (0.5, 4),
        'Theta (4-8)':   (4, 8),
        'Alpha (8-13)':  (8, 13),
        'Beta (13-30)':  (13, 30),
        'Gamma (30-45)': (30, 45),
    }

    # Load data
    data = sio.loadmat(mat_file, squeeze_me=False, struct_as_record=False)
    s = data['seizure'][0, 0]
    SUBJECTS = s.x.shape[0]

    os.makedirs(out_dir, exist_ok=True)

    for subj in range(SUBJECTS):
        x = s.x[subj, 0]  # (samples, channels)
        N, C_ch = x.shape

        # Channel names
        chan_struct = s.chans[subj, 0][0, 0]
        sel = chan_struct.selected
        chnames = [str(sel[i, 0][0]) for i in range(sel.shape[0])]

        # Seizure annotation
        ann_start = int(s.annotation[subj, 0][0, 0])
        ann_stop  = int(s.annotation[subj, 1][0, 0])
        info = str(s.info[subj, 0][0])

        # ── Extract pre-ictal and ictal segments ───────────────────
        # Pre-ictal: WINDOW_SEC before seizure onset
        pre_start = max(0, ann_start - WINDOW_SEC * fs)
        pre_stop  = ann_start
        x_pre = x[pre_start:pre_stop, :]

        # Ictal: first WINDOW_SEC of seizure (or full seizure if shorter)
        ict_start = ann_start
        ict_stop  = min(ann_stop, ann_start + WINDOW_SEC * fs)
        x_ict = x[ict_start:ict_stop, :]

        # ── Select MVAR order (on pre-ictal, should be representative) ─
        best_p, bic_vals = select_order_bic(x_pre, MAX_ORDER)
        print(f'[{subj+1:2d}/{SUBJECTS}] {info:30s}  '
              f'MVAR order={best_p}  '
              f'pre-ictal={x_pre.shape[0]/fs:.1f}s  '
              f'ictal={x_ict.shape[0]/fs:.1f}s')

        # ── Fit MVAR and compute PDC/DTF ───────────────────────────
        # PRE-ICTAL
        A_pre, C_pre = fit_mvar(x_pre, best_p)
        pdc_pre, dtf_pre, freqs = compute_pdc_dtf(A_pre, C_pre, NFFT, fs)
        pdc_pre_bands = band_average_matrix(pdc_pre, freqs, BANDS)
        dtf_pre_bands = band_average_matrix(dtf_pre, freqs, BANDS)

        # ICTAL
        A_ict, C_ict = fit_mvar(x_ict, best_p)
        pdc_ict, dtf_ict, _ = compute_pdc_dtf(A_ict, C_ict, NFFT, fs)
        pdc_ict_bands = band_average_matrix(pdc_ict, freqs, BANDS)
        dtf_ict_bands = band_average_matrix(dtf_ict, freqs, BANDS)

        # ── Plot ───────────────────────────────────────────────────
        plot_all_bands(
            pdc_pre_bands, chnames,
            f'PDC – Pre-ictal – Subj {subj+1} ({info})',
            os.path.join(out_dir, f'subj_{subj+1:02d}_PDC_pre.png')
        )
        plot_all_bands(
            pdc_ict_bands, chnames,
            f'PDC – Ictal – Subj {subj+1} ({info})',
            os.path.join(out_dir, f'subj_{subj+1:02d}_PDC_ict.png')
        )
        plot_all_bands(
            dtf_pre_bands, chnames,
            f'DTF – Pre-ictal – Subj {subj+1} ({info})',
            os.path.join(out_dir, f'subj_{subj+1:02d}_DTF_pre.png')
        )
        plot_all_bands(
            dtf_ict_bands, chnames,
            f'DTF – Ictal – Subj {subj+1} ({info})',
            os.path.join(out_dir, f'subj_{subj+1:02d}_DTF_ict.png')
        )

    print(f'\nDone — connectivity plots saved to {out_dir}/')
