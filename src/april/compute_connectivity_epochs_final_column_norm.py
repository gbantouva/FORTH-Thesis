"""
Effective Connectivity Pipeline — PDC, gPDC, & DTF (Multiprocessing)
===================================================================

Mathematical Updates:
1) PDC & DTF: Switched to Column-Normalization (Equations 2.1, 2.7).
2) gPDC: Implemented Generalized PDC weighted by noise variance (Equation 2.8).
3) Visualization: Vertical columns consistently represent the SOURCE (driver).

Author: Gemini AI
"""

import numpy as np
from scipy import linalg
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from functools import partial
import time

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURE
# ═══════════════════════════════════════════════════════════════════════
mat_file = r"F:\FORTH_Final_Thesis\FORTH-Thesis\data\EEG data\focal_seizures_34_pre5min_sc_1.8_bw_0.5_45.mat"
out_dir  = r"F:\FORTH_Final_Thesis\FORTH-Thesis\figures\april\connectivity_epochs"

fs = 256                
EPOCH_SEC = 4           
EPOCH_LEN = EPOCH_SEC * fs  
NFFT = 256              
MAX_ORDER = 20          
N_WORKERS = None        

BANDS = {
    'Delta (0.5-4)':  (0.5, 4),
    'Theta (4-8)':    (4, 8),
    'Alpha (8-13)':   (8, 13),
    'Beta (13-30)':   (13, 30),
    'Gamma (30-45)':  (30, 45),
}

# ═══════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def fit_mvar(x, order):
    T, K = x.shape
    p = order
    Y = x[p:]
    X = np.column_stack([x[p - k - 1:T - k - 1] for k in range(p)])
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    A = B.T
    E = Y - X @ B
    C = np.cov(E, rowvar=False, bias=False)
    return A, C

def compute_bic(x, order):
    T, K = x.shape
    n = T - order
    if n < K * order + 1: return np.inf
    try:
        A, C = fit_mvar(x, order)
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0: return np.inf
        return n * logdet + (K * K * order) * np.log(n)
    except: return np.inf

def _bic_for_one_epoch(epoch, max_order):
    return [compute_bic(epoch, p) for p in range(1, max_order + 1)]

def compute_connectivity_trio(A, C, nfft=256, fs=256):
    """
    Computes PDC, gPDC, and DTF using Source (Column) Normalization.
    Matches Thesis Equations 2.1, 2.7, and 2.8.
    """
    K = C.shape[0]
    p = A.shape[1] // K
    freqs = np.linspace(0, fs / 2, nfft)
    sigmas = np.sqrt(np.diag(C)) # Noise variance for gPDC

    pdc  = np.zeros((K, K, nfft))
    gpdc = np.zeros((K, K, nfft))
    dtf  = np.zeros((K, K, nfft))

    for fi, f in enumerate(freqs):
        Af = np.eye(K, dtype=complex)
        for k in range(p):
            Ak = A[:, k * K:(k + 1) * K]
            Af -= Ak * np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
        
        Hf = linalg.inv(Af)

        for j in range(K):
            # 1. PDC (Standard - Column Normalized)
            denom_pdc = np.sqrt(np.sum(np.abs(Af[:, j])**2))
            if denom_pdc > 0:
                pdc[:, j, fi] = np.abs(Af[:, j]) / denom_pdc

            # 2. gPDC (Generalized - Weighted by Sigma)
            denom_gpdc = np.sqrt(np.sum(np.abs(Af[:, j] / sigmas)**2))
            if denom_gpdc > 0:
                gpdc[:, j, fi] = (np.abs(Af[:, j]) / sigmas[j]) / denom_gpdc

            # 3. DTF (Standard - Column Normalized)
            denom_dtf = np.sqrt(np.sum(np.abs(Hf[:, j])**2))
            if denom_dtf > 0:
                dtf[:, j, fi] = np.abs(Hf[:, j]) / denom_dtf

    return pdc, gpdc, dtf, freqs

def band_average(conn, freqs, bands):
    matrices = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        matrices[name] = np.mean(conn[:, :, mask], axis=2) if mask.any() else np.zeros(conn.shape[:2])
    return matrices

def classify_epoch(start, stop, ann_start, ann_stop):
    if stop <= ann_start: return 'Pre-ictal'
    elif start >= ann_stop: return 'Post-ictal'
    else: return 'Ictal'

# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def plot_matrix_set(matrices, chnames, suptitle, out_path, is_broadband=False):
    """Universal plotter for bands or broadband connectivity."""
    items = [('Broadband', matrices)] if is_broadband else list(matrices.items())
    n = len(items)
    
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n + 1, 4.5), squeeze=False)
    for idx, (label, mat) in enumerate(items):
        ax = axes[0, idx]
        m = mat.copy()
        np.fill_diagonal(m, 0)
        
        im = ax.imshow(m, cmap='hot_r', vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(len(chnames)))
        ax.set_xticklabels(chnames, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(chnames)))
        ax.set_yticklabels(chnames, fontsize=7)
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════

def _process_one_subject(args):
    (subj_idx, epochs_list, epoch_info_list, chnames, info, best_p, out_root) = args
    
    subj_dir = os.path.join(out_root, f'subj_{subj_idx+1:02d}')
    measures = ['pdc', 'gpdc', 'dtf']
    subfolders = ['_bands', '_broadband']
    
    paths = {}
    for m in measures:
        for sf in subfolders:
            p = os.path.join(subj_dir, m + sf)
            os.makedirs(p, exist_ok=True)
            paths[m+sf] = p

    for ei, epoch_data in enumerate(epochs_list):
        start, stop, label = epoch_info_list[ei]
        try:
            A, C = fit_mvar(epoch_data, best_p)
            pdc, gpdc, dtf, freqs = compute_connectivity_trio(A, C, NFFT, fs)
        except: continue

        tag = f"epoch_{ei+1:03d}_{label.replace('-', '')}"
        title = f"{label} | ep {ei+1} | Subj {subj_idx+1}"

        for m_name, m_data in zip(['pdc', 'gpdc', 'dtf'], [pdc, gpdc, dtf]):
            # Band plots
            plot_matrix_set(band_average(m_data, freqs, BANDS), chnames, 
                            f"{m_name.upper()} {title}", os.path.join(paths[m_name+'_bands'], f"{tag}.png"))
            # Broadband plots
            plot_matrix_set(np.mean(m_data, axis=2), chnames, 
                            f"{m_name.upper()} Broadband {title}", os.path.join(paths[m_name+'_broadband'], f"{tag}.png"), is_broadband=True)

    return subj_idx, info

# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    mp.freeze_support()
    t_start_global = time.time()

    data = sio.loadmat(mat_file, squeeze_me=False, struct_as_record=False)
    s = data['seizure'][0, 0]
    n_subj = s.x.shape[0]

    all_epochs_global = []
    subject_tasks = []

    print("PHASE 1: Segmenting...")
    for subj in range(n_subj):
        x = s.x[subj, 0]
        sel = s.chans[subj, 0][0, 0].selected
        chnames = [str(sel[i, 0][0]) for i in range(sel.shape[0])]
        a_start, a_stop = int(s.annotation[subj, 0][0, 0])-1, int(s.annotation[subj, 1][0, 0])-1
        
        n_ep = x.shape[0] // EPOCH_LEN
        epochs, info_list = [], []
        for ei in range(n_ep):
            ep = x[ei*EPOCH_LEN : (ei+1)*EPOCH_LEN, :]
            epochs.append(ep); all_epochs_global.append(ep)
            info_list.append((ei*EPOCH_LEN, (ei+1)*EPOCH_LEN, classify_epoch(ei*EPOCH_LEN, (ei+1)*EPOCH_LEN, a_start, a_stop)))
        
        subject_tasks.append((subj, epochs, info_list, chnames, str(s.info[subj, 0][0])))

    print("PHASE 2: Global Order Selection...")
    with mp.Pool(N_WORKERS) as pool:
        bic_results = pool.map(partial(_bic_for_one_epoch, max_order=MAX_ORDER), all_epochs_global)
    best_p = np.argmin(np.median(np.array(bic_results), axis=0)) + 1
    print(f"Optimal Order: p={best_p}")

    print(f"PHASE 3: Computing Trio Connectivity (PDC, gPDC, DTF)...")
    final_tasks = [t + (best_p, out_dir) for t in subject_tasks]
    with mp.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(_process_one_subject, final_tasks):
            print(f"  Subj {res[0]+1} ({res[1]}) Finished.")

    print(f"Pipeline Complete in {time.time()-t_start_global:.1f}s")